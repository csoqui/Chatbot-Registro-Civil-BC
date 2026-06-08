"""
Arranque pesado: descarga NLTK si falta, carga spaCy, parser de intenciones y RAG.

Se llama una vez al iniciar consola o Telegram. Todo lo que tarda
(modelos, índice .joblib) queda en AppResources para reutilizar en cada turno.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Optional

import nltk
import spacy
from nltk.stem import WordNetLemmatizer

from pln_chatbot.config import RAG_INDEX_PATH, SPACY_MODEL, USE_RAG_FALLBACK
from pln_chatbot.intents import build_intent_parser
from pln_chatbot.rag import RAGIndex, load_index
from pln_chatbot.syntax_cfg import SyntaxCFGValidator

logger = logging.getLogger(__name__)


@dataclass
class AppResources:
    nlp: Optional[Any]
    intent_parser: Optional[Any]
    wordnet_lemmatizer: WordNetLemmatizer
    syntax_validator: SyntaxCFGValidator
    rag_index: Optional[RAGIndex]


def ensure_nltk_resources() -> None:
    needed = {
        "punkt": "tokenizers/punkt",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "punkt_tab": "tokenizers/punkt_tab",
    }
    logger.info("Verificando recursos NLTK…")
    for name, rel in needed.items():
        try:
            nltk.data.find(rel)
            logger.info("NLTK OK: %s", name)
        except LookupError:
            logger.info("Descargando NLTK: %s", name)
            nltk.download(name, quiet=True)


def load_spacy(model_name: str = SPACY_MODEL):
    logger.info("Cargando spaCy: %s", model_name)
    try:
        return spacy.load(model_name)
    except OSError:
        logger.warning(
            "Modelo spaCy no encontrado (%s). Instale con: python -m spacy download %s",
            model_name,
            model_name,
        )
        return None


def build_app_resources() -> AppResources:
    # Orden: recursos ligeros primero, luego spaCy (el más pesado), luego RAG opcional
    ensure_nltk_resources()
    nlp = load_spacy()
    intent_parser = build_intent_parser()
    if intent_parser is None:
        logger.error("El parser de intenciones no está disponible.")
    rag_index = None
    if USE_RAG_FALLBACK:
        rag_index = load_index(RAG_INDEX_PATH)
        if rag_index is None:
            logger.warning(
                "RAG activado pero sin índice local. Ejecute: python -m pln_chatbot.rag_ingest"
            )
        else:
            logger.info("Índice RAG cargado: %s", RAG_INDEX_PATH)
    return AppResources(
        nlp=nlp,
        intent_parser=intent_parser,
        wordnet_lemmatizer=WordNetLemmatizer(),
        syntax_validator=SyntaxCFGValidator(),
        rag_index=rag_index,
    )
