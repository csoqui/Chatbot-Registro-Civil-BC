"""
Rutas y opciones en un solo sitio.

Así no andas buscando paths repartidos: si cambias el JSON de dominio
o el .joblib del RAG, lo haces por variable de entorno o editando los
valores por defecto de aquí.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Carpeta del paquete y raíz del repo (para armar rutas a data/)
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

_DEFAULT_KNOWLEDGE = PROJECT_ROOT / "data" / "knowledge.default.json"
_DEFAULT_RAG_INDEX = PROJECT_ROOT / "data" / "rag" / "rag_tech_index.joblib"

# Si en PowerShell pones PLN_CHATBOT_KNOWLEDGE, manda sobre el default
KNOWLEDGE_PATH = Path(os.environ.get("PLN_CHATBOT_KNOWLEDGE", str(_DEFAULT_KNOWLEDGE))).resolve()
RAG_INDEX_PATH = Path(os.environ.get("PLN_CHATBOT_RAG_INDEX", str(_DEFAULT_RAG_INDEX))).resolve()

SPACY_MODEL = os.environ.get("PLN_CHATBOT_SPACY_MODEL", "es_core_news_lg")
# RAG se puede apagar sin borrar el .joblib
USE_RAG_FALLBACK = os.environ.get("PLN_CHATBOT_USE_RAG", "1").strip().lower() not in ("0", "false", "off", "no")

LOG_LEVEL = os.environ.get("PLN_CHATBOT_LOG_LEVEL", "INFO").upper()

def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
