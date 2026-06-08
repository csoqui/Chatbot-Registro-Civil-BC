"""
Puerta de entrada de cada mensaje (consola y Telegram).

La idea: no duplicar lógica. Tanto `cli.py` como `telegram_bot.py` llaman
a handle_turn() con la misma línea de texto. Aquí se reparten comandos
(/morfologia, /syntax…) y lo que no es comando va a dialogue.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pln_chatbot.config import KNOWLEDGE_PATH
from pln_chatbot.debug_trace import pipeline_traces_to_stdout, set_trace_override
from pln_chatbot.dialogue import generar_respuesta_chatbot, procesar_entrada_usuario
from pln_chatbot.knowledge import KnowledgeBase
from pln_chatbot.morphology import morfologia_a_texto, morfologia_a_texto_bloques
from pln_chatbot.resources import AppResources


@dataclass(frozen=True)
class TurnResult:
    """Respuesta(s) a enviar al usuario y si la sesión de consola debe terminar."""

    messages: tuple[str, ...]
    quit_console: bool


def _clean_line(raw: str) -> str:
    return raw.strip().lstrip("\ufeff")


UiChannel = Literal["console", "telegram"]


def _texto_ayuda(kb: KnowledgeBase, *, ui_channel: UiChannel) -> str:
    """Ayuda: Telegram solo comandos didácticos; consola incluye trace/debug."""
    path_kb = Path(str(KNOWLEDGE_PATH)).name
    if ui_channel == "telegram":
        return (
            "Comandos en Telegram (también en el menú al pulsar / ):\n"
            "- /start — Bienvenida.\n"
            "- /ayuda o /help — Esta lista.\n"
            "- /temas — Ejemplos de conceptos y cómo formular preguntas.\n"
            "- /morfologia <frase> — Análisis morfológico (spaCy).\n"
            "- /syntax <frase> — Validación CFG (preterminales + árbol).\n"
            "- /wordnet <término> — WordNet (definiciones y sinónimos).\n"
            "- /salir — Despedida.\n\n"
            "Consultas al dominio sin comando: «¿qué es …?», «define …», etc.\n\n"
            f"Dominio: {kb.domain_label}\n"
            f"Base de conocimiento: {path_kb}"
        )
    return (
        "Comandos útiles:\n"
        "- /morfologia <frase>: Análisis morfológico (forma, lema, POS, rasgos).\n"
        "- /syntax <frase>: Validación CFG (preterminales + ChartParser).\n"
        "- /wordnet <término>: WordNet (definiciones/sinónimos).\n"
        "- /salir: Despedida (en consola termina sesión).\n"
        "- /debug on|off: nivel del logger de Python (mensajes técnicos en esta consola).\n"
        "- /trace on|off|default: líneas «DEBUG …» del pipeline en esta consola.\n"
        "- (compatibilidad) wordnet define <término> y salir.\n"
        "- Preguntas tipo «¿qué temas cubre?», «lista de temas», «¿sobre qué puedo preguntar?».\n\n"
        f"Dominio cargado: {kb.domain_label}\n"
        f"Base de conocimiento: {path_kb} (ruta local del equipo)"
    )


def handle_turn(
    line: str,
    *,
    kb: KnowledgeBase,
    res: AppResources,
    historial: dict,
    log: logging.Logger | None = None,
    ui_channel: UiChannel = "console",
) -> TurnResult:
    """Un turno = una línea del usuario → mensajes de respuesta (y si consola debe cerrarse)."""
    log = log or logging.getLogger(__name__)
    line = _clean_line(line)
    if not line:
        return TurnResult((), False)

    low = line.lower()

    # En Telegram no tiene sentido /trace ni /debug (ruido en el móvil)
    if ui_channel == "telegram" and (
        low == "/trace"
        or low.startswith("/trace ")
        or low == "/debug"
        or low.startswith("/debug ")
    ):
        return TurnResult(
            (
                "En Telegram no se usan /trace ni /debug: solo tienen sentido en la consola del proyecto "
                "(PowerShell) donde corre `python -m pln_chatbot`. Escriba /ayuda para ver los comandos activos aquí.",
            ),
            False,
        )

    # --- Comandos de ayuda y salida ---
    if low in ("/ayuda", "/help", "?", "ayuda"):
        return TurnResult((_texto_ayuda(kb, ui_channel=ui_channel),), False)

    if low == "/salir":
        line = "salir"
        low = "salir"

    if low == "/wordnet":
        return TurnResult(("Uso: /wordnet <término>",), False)

    if low.startswith("/wordnet "):
        parts = line.split(maxsplit=1)
        termino = parts[1].strip() if len(parts) > 1 else ""
        if not termino:
            return TurnResult(("Uso: /wordnet <término>",), False)
        # Reutiliza el flujo existente de WordNet en dialogue.py
        line = f"wordnet define {termino}"
        low = line.lower()

    if low == "/debug":
        return TurnResult(("Uso: /debug on|off",), False)

    if low.startswith("/debug "):
        parts = line.split(maxsplit=1)
        rest = parts[1].strip().lower() if len(parts) > 1 else ""
        if rest == "on":
            logging.getLogger().setLevel(logging.DEBUG)
            log.info("Depuración: ACTIVADA")
            return TurnResult(("Logging DEBUG activado.",), False)
        if rest == "off":
            logging.getLogger().setLevel(logging.INFO)
            log.info("Depuración: DESACTIVADA")
            return TurnResult(("Logging INFO (normal).",), False)
        return TurnResult(("Uso: /debug on|off",), False)

    if low == "/trace":
        return TurnResult(("Uso: /trace on|off|default",), False)

    if low.startswith("/trace "):
        parts = line.split(maxsplit=1)
        rest = parts[1].strip().lower() if len(parts) > 1 else ""
        if rest == "on":
            set_trace_override(True)
            return TurnResult(("Trazas DEBUG del pipeline: activadas.",), False)
        if rest == "off":
            set_trace_override(False)
            return TurnResult(("Trazas DEBUG: desactivadas.",), False)
        if rest in ("default", "env", "reset"):
            set_trace_override(None)
            return TurnResult(("Trazas DEBUG: se usa PLN_CHATBOT_TRACE del entorno.",), False)
        return TurnResult(("Uso: /trace on|off|default",), False)

    # --- Comandos didácticos: morfología y sintaxis (spaCy + CFG) ---
    if low == "/morfologia":
        return TurnResult(("Uso: /morfologia <frase>",), False)

    if low.startswith("/morfologia "):
        parts = line.split(maxsplit=1)
        frase = parts[1].strip() if len(parts) > 1 else ""
        if not frase:
            return TurnResult(("Uso: /morfologia <frase>",), False)
        if res.nlp is None:
            return TurnResult(("spaCy no disponible.",), False)
        doc = res.nlp(frase)
        if ui_channel == "telegram":
            return TurnResult((morfologia_a_texto_bloques(doc),), False)
        return TurnResult((morfologia_a_texto(doc),), False)

    if low == "/syntax":
        return TurnResult(("Uso: /syntax <frase>",), False)

    if low.startswith("/syntax "):
        parts = line.split(maxsplit=1)
        frase = parts[1].strip() if len(parts) > 1 else ""
        if not frase:
            return TurnResult(("Uso: /syntax <frase>",), False)
        if res.nlp is None:
            return TurnResult(("spaCy no disponible.",), False)
        doc = res.nlp(frase)
        lim = 3800 if ui_channel == "telegram" else None
        return TurnResult((res.syntax_validator.describe(doc, max_chars=lim),), False)

    # --- Diálogo normal: intención + respuesta (aquí va el «cerebro») ---
    trace_stdout = ui_channel == "console"  # en Telegram no spameamos DEBUG
    with pipeline_traces_to_stdout(trace_stdout):
        datos = procesar_entrada_usuario(line, kb, res)
        if datos.get("intencion") == "despedida":
            out = generar_respuesta_chatbot(datos, kb, res, historial, ui_channel=ui_channel)
            return TurnResult((out,), True)

        out = generar_respuesta_chatbot(datos, kb, res, historial, ui_channel=ui_channel)
        return TurnResult((out,), False)


def chunk_for_telegram(text: str, limit: int = 4000) -> list[str]:
    """Telegram limita ~4096 caracteres por mensaje; trocea sin cortar palabras si puede."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut <= start:
                cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks
