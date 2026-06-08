"""
Intenciones con gramática de rasgos (NLTK).

Aquí no entendemos toda la frase del mundo: solo patrones que ya
escribimos (hola, qué es…, adiós). Si la gramática no casa, dialogue.py
sigue con spaCy y reglas sueltas.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import nltk

logger = logging.getLogger(__name__)

# Reglas en texto: cada línea es «si la frase es así, la intención es X»
FEATURE_GRAMMAR = r"""
%start S
S[SEM=[INTENCION='saludo']] -> 'hola' | 'buenos' 'días' | 'buenas' 'tardes' | 'buenas' 'noches' | 'qué' 'tal' | 'que' 'tal'
S[SEM=[INTENCION='como_estas']] -> 'cómo' 'estás' | 'como' 'estas' | 'qué' 'tal' 'estás'
S[SEM=[INTENCION='quien_eres']] -> 'quién' 'eres' | 'quien' 'eres' | 'cómo' 'te' 'llamas' | 'como' 'te' 'llamas' | 'qué' 'eres' | 'que' 'eres' | 'cuál' 'es' 'tu' 'nombre' | 'cual' 'es' 'tu' 'nombre'
S[SEM=[INTENCION='de_donde_eres']] -> 'de' 'dónde' 'eres' | 'de' 'donde' 'eres' | 'cuál' 'es' 'tu' 'origen'
S[SEM=[INTENCION='para_que_sirves']] -> 'para' 'qué' 'sirves' | 'para' 'que' 'sirves' | \
        'qué' 'haces' | 'que' 'haces' | \
        'cuál' 'es' 'tu' 'función' | \
        'para' 'qué' 'estás' 'programado' | 'para' 'que' 'estás' 'programado' | \
        'para' 'qué' 'fuiste' 'diseñado' | 'para' 'que' 'fuiste' 'diseñado'
S[SEM=[INTENCION='como_funciona']] -> 'cómo' 'funcionas' | 'como' 'funcionas' | \
        'explícame' 'tu' 'funcionamiento'
S[SEM=[INTENCION='despedida']] -> 'adiós' | 'hasta' 'luego' | 'chao' | 'salir'

S[SEM=[INTENCION='pregunta_cc_inicio']] -> PREGUNTA_START_WORDS
PREGUNTA_START_WORDS -> 'qué' 'es' | 'que' 'es'
PREGUNTA_START_WORDS -> 'define'
PREGUNTA_START_WORDS -> 'explícame'
PREGUNTA_START_WORDS -> 'dime' 'sobre'
PREGUNTA_START_WORDS -> 'háblame' 'de'
PREGUNTA_START_WORDS -> 'información' 'sobre'
PREGUNTA_START_WORDS -> 'qué' 'sabes' 'de'
PREGUNTA_START_WORDS -> 'que' 'sabes' 'de'
PREGUNTA_START_WORDS -> 'cuéntame' 'sobre'
PREGUNTA_START_WORDS -> 'cuéntame' 'de'
PREGUNTA_START_WORDS -> 'qué' 'sabes' 'acerca' 'de'
PREGUNTA_START_WORDS -> 'que' 'sabes' 'acerca' 'de'
PREGUNTA_START_WORDS -> 'cómo' 'funciona'
PREGUNTA_START_WORDS -> 'como' 'funciona'
PREGUNTA_START_WORDS -> 'cuáles' 'son'
PREGUNTA_START_WORDS -> 'cuales' 'son'
PREGUNTA_START_WORDS -> 'qué' 'diferencia' 'hay' 'entre'
PREGUNTA_START_WORDS -> 'que' 'diferencia' 'hay' 'entre'
PREGUNTA_START_WORDS -> 'ventajas' 'de'
PREGUNTA_START_WORDS -> 'desventajas' 'de'
"""


def build_intent_parser():
    try:
        g = nltk.grammar.FeatureGrammar.fromstring(FEATURE_GRAMMAR)
        return nltk.parse.FeatureChartParser(g)
    except ValueError as e:
        logger.error("No se pudo compilar la gramática de intenciones: %s", e)
        return None


def parse_intent_with_meta(tokens: list[str], parser) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Devuelve (resultado, error).
    - error str si el parser lanzó excepción (p. ej. gramática no cubre tokens);
    - (None, None) si no hubo excepción pero no hay árbol;
    - (dict, None) si hubo intención reconocida.
    """
    if parser is None:
        return None, None
    try:
        trees = list(parser.parse(tokens))
    except (ValueError, Exception) as e:
        logger.debug("Intent parse fallido: %s", e)
        return None, str(e)
    if not trees:
        return None, None
    sem = trees[0].label().get("SEM", {}) or {}
    intent = sem.get("INTENCION")
    if not intent:
        return None, None
    entidades = {str(k): v for k, v in sem.items() if str(k).upper() != "INTENCION"}
    return {"intencion": str(intent), "entidades": entidades}, None


def parse_intent_feature(tokens: list[str], parser) -> Optional[dict[str, Any]]:
    feat, _err = parse_intent_with_meta(tokens, parser)
    return feat
