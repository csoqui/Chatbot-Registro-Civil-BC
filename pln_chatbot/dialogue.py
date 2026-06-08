"""
El «cerebro» del chatbot: entender qué pidió el usuario y armar la respuesta.

procesar_entrada_usuario → intención + tema + datos de sintaxis
generar_respuesta_chatbot → texto final (JSON, WordNet, RAG, saludos…)

interaction.py solo enruta; aquí está la lógica de conversación.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any, Literal

import nltk

from pln_chatbot.debug_trace import debug_trace
from pln_chatbot.extraction import extraer_tema_cc
from pln_chatbot.intents import parse_intent_with_meta
from pln_chatbot.knowledge import KnowledgeBase
from pln_chatbot.nlp_utils import expandir_acronimo_tema, normalizar_clave_busqueda
from pln_chatbot.rag import (
    construir_consulta_rag,
    format_rag_answer,
    format_rag_answer_chat,
    is_registro_civil_query,
    is_tech_query,
)
from pln_chatbot.resources import AppResources
from pln_chatbot.wordnet_tools import obtener_sinonimos_definiciones_wn, obtener_sinonimos_wn

logger = logging.getLogger(__name__)

UiChannel = Literal["console", "telegram"]


def _es_consulta_temas_dominio(texto_lower: str) -> bool:
    """¿El usuario pide orientación sobre qué puede preguntar o qué cubre el bot?"""
    if len(texto_lower) > 220:
        return False
    patrones = (
        "qué puedo preguntar",
        "que puedo preguntar",
        "sobre qué puedo",
        "sobre que puedo",
        "qué temas",
        "que temas",
        "en qué temas",
        "en que temas",
        "cuáles son los temas",
        "cuales son los temas",
        "lista de temas",
        "qué conceptos",
        "que conceptos",
        "qué cubre el",
        "que cubre el",
        "de qué trata este",
        "de que trata este",
        "ámbito del dominio",
        "ambito del dominio",
        "temas del dominio",
        "qué dominio",
        "que dominio",
        "sobre qué trata",
        "sobre que trata",
        "qué sabes hacer",
        "que sabes hacer",
    )
    return any(p in texto_lower for p in patrones)


def procesar_entrada_usuario(texto_usuario: str, kb: KnowledgeBase, res: AppResources) -> dict[str, Any]:
    """Un mensaje entra; sale un dict con intención, entidades y de dónde salió."""
    if not texto_usuario:
        return {
            "intencion": "vacio",
            "entidades": {},
            "texto_original": "",
            "procesador": "ninguno",
            "doc_spacy": None,
            "syntax": None,
        }

    texto_lower = texto_usuario.lower().strip()
    intencion = "desconocida"
    entidades: dict[str, Any] = {}
    doc_spacy_obj = None
    procesador = "ninguno"
    syntax_info = None

    debug_trace(f"DEBUG ENTRADA: texto_recibido='{texto_lower}'")

    # ¿Preguntó «qué temas cubres» sin pedir un concepto concreto?
    if _es_consulta_temas_dominio(texto_lower):
        debug_trace("DEBUG META_DOMINIO: tipo='lista_temas_y_ambito' origen='patrones_heuristicos'")
        return {
            "intencion": "meta_temas_dominio",
            "entidades": {},
            "texto_original": texto_usuario,
            "procesador": "meta_dominio",
            "doc_spacy": None,
            "syntax": None,
        }

    # CFG opcional: solo informa; no bloquea el resto del flujo
    if res.nlp is not None:
        doc_spacy_obj = res.nlp(texto_lower)
        syn = res.syntax_validator.validate_doc(doc_spacy_obj)
        syntax_info = {
            "accepted": syn.accepted,
            "sequence": list(syn.sequence),
            "ambiguity": syn.ambiguity,
        }
        logger.debug("Validación CFG: %s", syntax_info)
        seq_s = " ".join(syntax_info["sequence"]) if syntax_info["sequence"] else "(vacía)"
        debug_trace(
            f"DEBUG SINTAXIS_CFG: secuencia_preterminales='{seq_s}' aceptada='{ 'si' if syntax_info['accepted'] else 'no' }'"
        )

    logger.debug("Entrada: %r", texto_lower)

    # Comando explícito de WordNet (también llega como /wordnet desde interaction)
    match_wn = re.match(
        r"^(?:wordnet define|defineme con wordnet|significado wordnet de|wn define)\s+(.+)",
        texto_lower,
    )
    if match_wn:
        termino = match_wn.group(1).strip()
        debug_trace(f"DEBUG WORDNET_COMANDO: termino='{termino}'")
        if res.nlp is not None:
            doc_spacy_obj = res.nlp(termino)
        return {
            "intencion": "definir_termino_wordnet_usuario",
            "entidades": {"termino_wordnet": termino},
            "texto_original": texto_usuario,
            "procesador": "regex_wordnet",
            "doc_spacy": doc_spacy_obj,
            "syntax": syntax_info,
        }

    # Intento 1: gramática NLTK (rápida para saludos y «qué es…»)
    if res.intent_parser is not None:
        tokens = nltk.word_tokenize(texto_lower)
        feat, nltk_err = parse_intent_with_meta(tokens, res.intent_parser)
        if nltk_err:
            debug_trace("DEBUG GRAMATICA_NLTK: resultado='error_parseo' decision='usar_spacy'")
        elif feat is None and tokens:
            debug_trace("DEBUG GRAMATICA_NLTK: resultado='sin_coincidencia' decision='usar_spacy'")

        if feat:
            intencion_nltk = feat["intencion"]
            entidades.update(feat.get("entidades") or {})
            procesador = "nltk_feat"
            logger.debug("Intención NLTK: %s | %s", intencion_nltk, entidades)
            debug_trace(f"DEBUG GRAMATICA_NLTK: intencion_detectada='{intencion_nltk}'")

            if intencion_nltk == "pregunta_cc_inicio":
                if res.nlp is not None:
                    doc_spacy_obj = res.nlp(texto_lower)
                    tema = extraer_tema_cc(texto_usuario, doc_spacy_obj, res.nlp)
                    if tema:
                        entidades["tema_cc"] = tema
                        intencion = "preguntar_sobre_cc"
                    else:
                        intencion = "pregunta_cc_generica_sin_tema"
                else:
                    intencion = "pregunta_cc_generica_sin_tema"
            else:
                intencion = intencion_nltk

            debug_trace(f"DEBUG INTENCION_FINAL: intencion='{intencion}' origen='gramatica_nltk'")
            return {
                "intencion": intencion,
                "entidades": entidades,
                "texto_original": texto_usuario,
                "procesador": procesador,
                "doc_spacy": doc_spacy_obj,
                "syntax": syntax_info,
            }

    # Intento 2: spaCy + listas de palabras si NLTK no casó
    if res.nlp is not None:
        if doc_spacy_obj is None:
            doc_spacy_obj = res.nlp(texto_lower)
        procesador = "spacy"

        # Saludos y meta antes de sacar tema (si no, «hola» podría malinterpretarse)
        # clasificarlas como "preguntar_sobre_cc" por extracción de tema.
        if intencion == "desconocida":
            if any(
                kw == texto_lower or texto_lower.startswith(kw + " ")
                for kw in ["hola", "buenos días", "buenas tardes", "buenas noches", "qué tal", "que tal"]
            ):
                intencion = "saludo"
            elif any(kw == texto_lower for kw in ["adiós", "hasta luego", "chao", "salir"]):
                intencion = "despedida"
            elif any(kw in texto_lower for kw in ["cómo estás", "como estas"]):
                intencion = "como_estas"
            elif any(
                kw in texto_lower
                for kw in [
                    "quién eres",
                    "quien eres",
                    "cómo te llamas",
                    "como te llamas",
                    "qué eres",
                    "que eres",
                    "cuál es tu nombre",
                    "cual es tu nombre",
                ]
            ):
                intencion = "quien_eres"
            elif any(kw in texto_lower for kw in ["de dónde eres", "de donde eres", "cuál es tu origen"]):
                intencion = "de_donde_eres"
            elif any(
                kw in texto_lower
                for kw in [
                    "para qué sirves",
                    "para que sirves",
                    "qué haces",
                    "que haces",
                    "cuál es tu función",
                    "propósito",
                    "para qué estás programado",
                    "para que fuiste diseñado",
                ]
            ):
                intencion = "para_que_sirves"
            elif any(kw in texto_lower for kw in ["cómo funcionas", "como funcionas", "explícame tu funcionamiento"]):
                intencion = "como_funciona"

        if intencion in ("desconocida", "pregunta_cc_inicio"):
            tema = extraer_tema_cc(texto_usuario, doc_spacy_obj, res.nlp)
            if tema:
                entidades["tema_cc"] = tema
                intencion = "preguntar_sobre_cc"
                debug_trace(f"DEBUG SPACY: intencion='preguntar_sobre_cc' tema_principal='{tema}'")
            else:
                intencion = "pregunta_cc_generica_sin_tema"
                debug_trace("DEBUG SPACY: intencion='pregunta_cc_generica_sin_tema' tema='no_detectado'")

        if intencion == "desconocida":
            if intencion == "desconocida" and len(doc_spacy_obj) <= 3 and not any(
                tok.is_stop or tok.is_punct for tok in doc_spacy_obj
            ):
                posible = normalizar_clave_busqueda(texto_usuario)
                if posible and (
                    kb.lookup(posible)
                    or any(len(posible) > 1 and posible in key for key in kb.entries_normalized.keys())
                ):
                    entidades["tema_cc"] = posible
                    intencion = "preguntar_sobre_cc"
                    debug_trace(f"DEBUG SPACY: tema_corto_detectado='{posible}'")

        if intencion == "pregunta_cc_generica_sin_tema" and is_tech_query(texto_lower):
            # Pregunta tecnológica abierta: la mantenemos en pipeline de dominio para intentar RAG.
            tema = normalizar_clave_busqueda(texto_usuario) or texto_lower
            entidades["tema_cc"] = tema
            intencion = "preguntar_sobre_cc"
            debug_trace(f"DEBUG SPACY: pregunta_tecnologica_abierta tema='{tema}'")

        logger.debug("Salida spacy: intención=%s entidades=%s", intencion, entidades)
        debug_trace(f"DEBUG INTENCION_FINAL: intencion='{intencion}' entidades={entidades}")
        return {
            "intencion": intencion,
            "entidades": entidades,
            "texto_original": texto_usuario,
            "procesador": procesador,
            "doc_spacy": doc_spacy_obj,
            "syntax": syntax_info,
        }

    debug_trace(f"DEBUG INTENCION_FINAL: intencion='{intencion}' origen='sin_pln_avanzado'")
    return {
        "intencion": intencion,
        "entidades": entidades,
        "texto_original": texto_usuario,
        "procesador": "ninguno",
        "doc_spacy": None,
        "syntax": syntax_info,
    }


def _texto_temas_y_ambito(kb: KnowledgeBase, *, chat: bool = False, texto_pregunta: str = "") -> str:
    """Respuesta fija orientativa: qué dominio cubre y ejemplos de conceptos (alineado al JSON plantilla)."""
    if chat:
        claves = sorted(kb.entries_normalized.keys())
        max_m = 14
        if len(claves) > max_m:
            muestra_lines = "\n".join(f"• {c}" for c in claves[:max_m])
            sufijo = f"\n… y más ({len(claves)} conceptos en total en la base local)."
        elif claves:
            muestra_lines = "\n".join(f"• {c}" for c in claves)
            sufijo = ""
        else:
            muestra_lines = "(No hay entradas en el JSON cargado.)"
            sufijo = ""
        low = (texto_pregunta or "").lower()
        if "lista" in low:
            intro = f"Lista orientativa del dominio «{kb.domain_label}» (ejemplos que tiene la base local):\n"
        else:
            intro = (
                f"Atiendo consultas del dominio «{kb.domain_label}» "
                "(computación, IA, informática y afines).\n\n"
                "Puede preguntar con «¿qué es …?», «define …» o el comando /wordnet <término>. "
                "Ejemplos de conceptos en la base:\n"
            )
        return intro + muestra_lines + sufijo + "\n\nEn la consola del proyecto verá además la lista completa y las trazas DEBUG."
    claves = sorted(kb.entries_normalized.keys())
    if len(claves) > 20:
        muestra = ", ".join(claves[:20]) + ", …"
    else:
        muestra = ", ".join(claves) if claves else "(ninguna clave en el JSON cargado)"

    return (
        f"El dominio configurado en esta sesión es: **«{kb.domain_label}»**.\n\n"
        "Puede formular preguntas **definicionales** (por ejemplo «¿qué es …?», «define …», "
        "«explícame …») sobre conceptos de **computación, informática, inteligencia artificial** "
        "y temas de **tecnología** relacionados. Ejemplos típicos que encajan con la plantilla:\n\n"
        "• **Disciplina e ingeniería:** computación, informática, programación, desarrollo de software, algoritmos.\n"
        "• **Inteligencia artificial:** IA, machine learning, aprendizaje automático y profundo, deep learning, "
        "redes neuronales, NLP / procesamiento del lenguaje natural.\n"
        "• **Datos:** ciencia de datos, análisis de datos, estadística.\n"
        "• **Redes e infraestructura:** redes de computadoras, redes, telecomunicaciones.\n"
        "• **Seguridad:** ciberseguridad, seguridad.\n"
        "• **Lenguajes (ejemplo en el JSON):** Python, Java.\n"
        "• **Interacción / afecto computacional (ejemplo en el JSON):** ideas vinculadas a computación afectiva.\n\n"
        f"**Claves que existen hoy en su base local** (nombre interno tras normalizar): {muestra}\n\n"
        "Si no hay entrada exacta, el agente puede intentar **sinónimos con WordNet** y, si está activo el índice, "
        "**fragmentos RAG** (Wikipedia española + AbScientia, filtrados a temática tecnológica).\n\n"
        "Para ampliar el tema: edite `data/knowledge.*.json`. Para una definición léxica puntual: "
        "«/wordnet término» o «wordnet define término»."
    )


def generar_respuesta_chatbot(
    datos: dict[str, Any],
    kb: KnowledgeBase,
    res: AppResources,
    historial: dict[str, int],
    *,
    ui_channel: UiChannel = "console",
) -> str:
    """Con la intención ya detectada, arma el texto que verá el usuario."""
    intencion = datos.get("intencion", "desconocida")
    entidades = datos.get("entidades", {}) or {}
    texto_orig = datos.get("texto_original", "") or ""
    syntax = datos.get("syntax")
    chat = ui_channel == "telegram"

    def _syntax_note() -> str:
        if chat or not syntax or syntax.get("accepted"):
            return ""
        seq = syntax.get("sequence") or []
        return (
            "\n\n[Nota técnica] La oración no encajó en la CFG de preguntas definicionales "
            f"(secuencia observada: {' '.join(seq) or '∅'}). Aun así se intentó interpretación heurística."
        )

    if intencion == "definir_termino_wordnet_usuario":
        termino = entidades.get("termino_wordnet")
        if not termino:
            return "Indique el término tras el comando, por ejemplo: «wordnet define algoritmo»."
        debug_trace(f"DEBUG WORDNET: consulta='{termino}'")
        info = obtener_sinonimos_definiciones_wn(
            termino,
            nlp=res.nlp,
            lang=kb.wordnet_language,
            max_defs=2,
            max_syns_total=5,
        )
        defs_ = info.get("definiciones", [])
        syns = info.get("sinonimos", [])
        out = f"WordNet ({kb.wordnet_language}) — «{termino}»:\n"
        if defs_ and not str(defs_[0]).startswith("No encontré"):
            out += "Definiciones:\n" + "\n".join(defs_)
        else:
            out += defs_[0] if defs_ else "Sin definiciones."
        if syns:
            out += "\nSinónimos (muestra): " + ", ".join(syns)
        else:
            out += "\nSin sinónimos detectados en esta consulta."
        return out

    dominio = kb.domain_label

    if intencion == "saludo":
        historial["turnos_sin_respuesta_especifica"] = 0
        return (
            "Buen día. Soy un asistente capaz de responder preguntas sobre las Ciencias de la computación, "
            "la Inteligencia Artificial y otros temas informáticos. Puede preguntar sobre un concepto o pedir que lo defina."
        )
    if intencion == "como_estas":
        return random.choice(
            [
                "Operando con normalidad. ¿Qué concepto desea revisar?",
                "Listo para atender consultas del dominio configurado.",
            ]
        )
    if intencion == "quien_eres":
        return (
            "Me llaman el asistente Pragmático, capaz de responder preguntas sobre las Ciencias de la computación, "
            "la Inteligencia Artificial y otros temas informáticos. Así que dime cuál es tu pregunta."
        )
    if intencion == "de_donde_eres":
        return "Soy un programa: no tengo ubicación física; mi comportamiento depende del dominio cargado en la base de conocimiento."
    if intencion == "para_que_sirves":
        if chat:
            return f"Estoy pensado para responder con definiciones y consultas del dominio «{dominio}»."
        return f"Estoy diseñado para explicar conceptos del dominio «{dominio}» y para ilustrar técnicas de PLN en consola."
    if intencion == "como_funciona":
        if chat:
            return (
                "Analizo su mensaje con reglas de intención, identifico el tema y busco la respuesta en la base "
                "de conocimiento. El detalle paso a paso (trazas) está pensado para la versión en consola del proyecto."
            )
        return (
            "Pipeline técnico resumido: (1) tokenización NLTK para reglas léxicas; "
            "(2) gramática de rasgos para intenciones; (3) análisis morfosintáctico spaCy para extraer el tema; "
            "(4) validación sintáctica opcional con CFG sobre preterminales; "
            "(5) recuperación en base de conocimiento y expansión léxica con WordNet."
        )
    if intencion == "meta_temas_dominio":
        historial["turnos_sin_respuesta_especifica"] = 0
        return _texto_temas_y_ambito(kb, chat=chat, texto_pregunta=texto_orig)

    if intencion == "despedida":
        if chat:
            return random.choice(["Hasta luego.", "Que tenga buen día."])
        return random.choice(
            [
                "Hasta luego. Quedo a disposición cuando reinicie el agente.",
                "Que tenga buen día.",
            ]
        )

    # Pregunta por un concepto: JSON → WordNet → palabras sueltas → RAG
    if intencion == "preguntar_sobre_cc":
        tema = entidades.get("tema_cc")
        if tema:
            tema = expandir_acronimo_tema(tema)
            entidades["tema_cc"] = tema
        if not tema:
            historial["turnos_sin_respuesta_especifica"] = historial.get("turnos_sin_respuesta_especifica", 0) + 1
            if chat:
                return "No identifiqué bien el concepto. ¿Podría reformular la pregunta?"
            return "Detecté una consulta definicional, pero no identifiqué el tema con suficiente claridad. ¿Podría reformular?"

        debug_trace(f"DEBUG BASE_LOCAL: tema_consultado='{tema}'")
        info = kb.lookup(tema)

        if info:
            historial["turnos_sin_respuesta_especifica"] = 0
            debug_trace("DEBUG BASE_LOCAL: resultado='coincidencia_exacta'")
            msg = info
            if syntax and not syntax.get("accepted"):
                msg = msg + _syntax_note()
            return msg

        debug_trace("DEBUG BASE_LOCAL: resultado='sin_coincidencia' siguiente='wordnet'")
        sinonimos = (
            obtener_sinonimos_wn(
                tema,
                nlp=res.nlp,
                lemmatizer=res.wordnet_lemmatizer,
                lang=kb.wordnet_language,
                limit=5,
            )
            if res.nlp is not None
            else []
        )
        debug_trace(f"DEBUG WORDNET: sinonimos_encontrados={sinonimos[:5]}")
        for syn in sinonimos:
            hit = kb.lookup(syn)
            if hit:
                historial["turnos_sin_respuesta_especifica"] = 0
                debug_trace(f"DEBUG WORDNET: coincidencia_por_sinonimo='{syn}'")
                if chat:
                    msg = f"Relacionado con «{syn}»:\n{hit}"
                else:
                    msg = (
                        f"No hay entrada directa para «{texto_orig}». "
                        f"Se encontró una entrada relacionada («{syn}»):\n{hit}"
                    )
                if syntax and not syntax.get("accepted"):
                    msg += _syntax_note()
                return msg

        debug_trace("DEBUG BUSQUEDA_PARCIAL: estado='buscando_coincidencia_aproximada'")
        tema_tokens = set((tema or "").split())
        best_key = None
        best_score = 0.0
        for key_bc in kb.entries_normalized.keys():
            key_tokens = set(key_bc.split())
            overlap = [
                w
                for w in tema_tokens
                if w in key_tokens and len(w) > 2 and w not in {"de", "la", "el", "los", "las", "un", "una", "unos", "unas"}
            ]
            score = len(overlap) * (len(key_tokens) / (len(tema_tokens) + 0.1))
            if score > best_score:
                best_score = score
                best_key = key_bc

        if best_key and best_score >= 0.5:
            historial["turnos_sin_respuesta_especifica"] = 0
            debug_trace(
                f"DEBUG BUSQUEDA_PARCIAL: clave_seleccionada='{best_key}' puntaje='{best_score:.2f}'"
            )
            hit = kb.entries_normalized[best_key]
            if chat:
                msg = f"Sobre «{best_key}»:\n{hit}"
            else:
                msg = (
                    f"No hay entrada directa para «{texto_orig}». "
                    f"Información aproximada sobre «{best_key}»:\n{hit}"
                )
            if syntax and not syntax.get("accepted"):
                msg += _syntax_note()
            return msg

        historial["turnos_sin_respuesta_especifica"] = historial.get("turnos_sin_respuesta_especifica", 0) + 1
        debug_trace("DEBUG BASE_LOCAL: estado='sin_resultado_en_local_wordnet_parcial'")

        rag_aplica = is_tech_query(texto_orig) or (
            "registro civil" in dominio.lower() and is_registro_civil_query(f"{texto_orig} {tema or ''}")
        )
        if res.rag_index is not None and rag_aplica:
            rag_query = construir_consulta_rag(texto_orig, tema)
            rag_hits = res.rag_index.search(rag_query, top_k=8, min_score=0.10)
            if chat:
                rag_msg = format_rag_answer_chat(rag_hits, tema=tema, pregunta=texto_orig)
            else:
                rag_msg = format_rag_answer(texto_orig, rag_hits)
            if rag_msg:
                historial["turnos_sin_respuesta_especifica"] = 0
                fuentes = [f"{h.source}:{(h.title or '')[:24]}" for h in rag_hits[:3]]
                debug_trace(
                    f"DEBUG RAG_LOCAL: fragmentos_recuperados={len(rag_hits)} fuentes_principales={fuentes}"
                )
                if syntax and not syntax.get("accepted"):
                    rag_msg += _syntax_note()
                return rag_msg

        if chat:
            msg = (
                "No encontré una respuesta fiable en la base local ni en el índice de referencia para esa consulta. "
                "Pruebe reformulando (por ejemplo «¿qué es …?» o «define …») "
                f"o pregunte por otro concepto del dominio «{dominio}»."
            )
        else:
            msg = (
                f"No encontré información específica sobre «{texto_orig}» en la base del dominio «{dominio}». "
                "Puede ampliar `data/knowledge.*.json`, construir índice RAG local o reformular la consulta."
            )
        if syntax and not syntax.get("accepted"):
            msg += _syntax_note()
        return msg

    if intencion == "vacio":
        if chat:
            return "No recibí texto. Escriba su consulta cuando guste."
        return random.choice(["No capté texto. Escriba una consulta o «salir».", "Entrada vacía."])

    if intencion == "pregunta_cc_generica_sin_tema":
        historial["turnos_sin_respuesta_especifica"] = historial.get("turnos_sin_respuesta_especifica", 0) + 1
        if chat:
            return (
                "No identifiqué bien el tema. Pruebe con «¿Qué es …?» o «define …» "
                "incluyendo el concepto (por ejemplo inteligencia artificial o algoritmo)."
            )
        return (
            "La pregunta parece definicional, pero falta el tema explícito. "
            "Ejemplos: «¿Qué es inteligencia artificial?», «define algoritmo»."
        )

    historial["turnos_sin_respuesta_especifica"] = historial.get("turnos_sin_respuesta_especifica", 0) + 1
    c = historial["turnos_sin_respuesta_especifica"]
    if c == 1:
        if chat:
            return (
                f"No tengo una respuesta adecuada para eso. Estoy orientado al dominio «{dominio}»; "
                "formule una pregunta definicional o use /ayuda."
            )
        return f"No reconozco la intención. Este bot está acotado al dominio «{dominio}»."
    if c == 2:
        if chat:
            return "Puede probar «¿qué es …?», «define …» o el comando /wordnet seguido del término."
        return "Intente una pregunta con patrón definicional o use «wordnet define <término>»."
    historial["turnos_sin_respuesta_especifica"] = 0
    if chat:
        return "Escriba /ayuda para ver comandos. Si desea cerrar la conversación en Telegram, basta con dejar de escribir."
    return "Si desea terminar, escriba «salir». Para ayuda técnica, consulte el README del proyecto."


def initial_historial() -> dict[str, int]:
    """Estado mínimo por sesión: cuántas veces seguimos sin entender al usuario."""
    return {"turnos_sin_respuesta_especifica": 0}
