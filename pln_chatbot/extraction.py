"""
Sacar el «tema» de frases como «qué es machine learning».

Primero quitamos el inicio fijo («qué es», «define»…), luego spaCy
busca el trozo importante (noun chunk o sustantivos seguidos).
"""

from __future__ import annotations

import logging

from pln_chatbot.debug_trace import debug_trace
from pln_chatbot.nlp_utils import normalizar_clave_busqueda

logger = logging.getLogger(__name__)


def extraer_tema_cc(texto_original_usuario: str, doc_completo, nlp) -> str | None:
    """Prioriza noun chunks y secuencias nominales; funciona sin spaCy con normalización simple."""
    texto_sin_intro = texto_original_usuario
    # Ordenamos de más larga a más corta para no cortar mal («qué es» antes que «qué»)
    frases_intro = sorted(
        [
            "qué es lo que se conoce como",
            "qué es lo que se entiende por",
            "explícame que es",
            "explícame qué es",
            "información sobre",
            "háblame de",
            "cuéntame sobre",
            "cuéntame de",
            "qué sabes de",
            "que sabes de",
            "qué sabes acerca de",
            "que sabes acerca de",
            "definición de",
            "dime sobre",
            "qué es",
            "que es",
            "cómo funciona",
            "como funciona",
            "cuáles son",
            "cuales son",
            "qué diferencia hay entre",
            "que diferencia hay entre",
            "ventajas de",
            "desventajas de",
            "define",
            "explícame",
        ],
        key=len,
        reverse=True,
    )
    low = texto_sin_intro.lower()
    intro_quitada: str | None = None
    for kw in frases_intro:
        if low.startswith(kw):
            texto_sin_intro = texto_sin_intro[len(kw) :].strip()
            intro_quitada = kw
            logger.debug("Intro retirada (%s): resto=%r", kw, texto_sin_intro)
            break

    if intro_quitada is not None:
        debug_trace(f"DEBUG EXTRACCION_TEMA: prefijo_detectado='{intro_quitada}'")

    if not texto_sin_intro.strip():
        debug_trace("DEBUG EXTRACCION_TEMA: resultado='sin_tema_detectado'")
        return None

    if nlp is None:
        debug_trace("DEBUG EXTRACCION_TEMA: metodo='normalizacion_simple'")
        return normalizar_clave_busqueda(texto_sin_intro)

    doc_tema = nlp(texto_sin_intro)
    chunks = [c.text for c in doc_tema.noun_chunks if c.text.strip()]
    if chunks:
        tema_bruto = max(chunks, key=len) if len(chunks) > 1 else chunks[0]
        debug_trace(
            f"DEBUG EXTRACCION_TEMA: metodo='noun_chunk' tema_principal='{normalizar_clave_busqueda(tema_bruto)}'"
        )
        return normalizar_clave_busqueda(tema_bruto)

    tokens_relevantes: list[str] = []
    for token in doc_tema:
        if token.pos_ in ("PROPN", "NOUN", "ADJ", "X") and not token.is_stop and not token.is_punct:
            tokens_relevantes.append(token.text)
        elif tokens_relevantes:
            break
    if tokens_relevantes:
        unido = " ".join(tokens_relevantes)
        debug_trace(
            f"DEBUG EXTRACCION_TEMA: metodo='tokens_relevantes' tema_principal='{normalizar_clave_busqueda(unido)}'"
        )
        return normalizar_clave_busqueda(unido)

    debug_trace(
        f"DEBUG EXTRACCION_TEMA: metodo='fallback_texto' tema_principal='{normalizar_clave_busqueda(texto_sin_intro)}'"
    )
    return normalizar_clave_busqueda(texto_sin_intro)
