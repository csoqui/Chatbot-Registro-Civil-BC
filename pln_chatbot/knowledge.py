"""
Lee el JSON de dominio y busca respuestas por clave.

No hay base de datos: es un diccionario en archivo. Tú editas entries
y el bot responde con ese texto. Las claves se normalizan (sin acentos,
minúsculas) para que «IA» y «ia» den en el mismo sitio.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pln_chatbot.nlp_utils import normalizar_clave_busqueda

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnowledgeBase:
    domain_label: str
    assistant_name: str
    assistant_short_name: str
    welcome_hint: str
    wordnet_language: str
    entries_normalized: dict[str, str]

    def lookup(self, key: str) -> str | None:
        # Primero coincidencia exacta; si no, busca si la clave está contenida en otra
        k = normalizar_clave_busqueda(key)
        if not k:
            return None
        if k in self.entries_normalized:
            return self.entries_normalized[k]
        for bc_key, val in self.entries_normalized.items():
            if len(k) > 1 and k in bc_key:
                return val
        return None


def load_knowledge(path: Path) -> KnowledgeBase:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries_in = raw.get("entries") or {}
    normalized: dict[str, str] = {}
    for surface, text in entries_in.items():
        nk = normalizar_clave_busqueda(str(surface))
        if not nk:
            continue
        if nk in normalized and normalized[nk] != text:
            logger.warning("Clave duplicada tras normalizar (%r -> %r); se sobrescribe.", surface, nk)
        normalized[nk] = str(text).strip()

    return KnowledgeBase(
        domain_label=str(raw.get("domain_label", "Dominio")),
        assistant_name=str(raw.get("assistant_name", "Asistente")),
        assistant_short_name=str(raw.get("assistant_short_name", "Bot")),
        welcome_hint=str(raw.get("welcome_hint", "")),
        wordnet_language=str(raw.get("wordnet_language", "spa")),
        entries_normalized=normalized,
    )
