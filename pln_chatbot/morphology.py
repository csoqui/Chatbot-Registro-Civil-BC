"""
Salida del comando /morfologia: tabla de tokens con lema y POS.

spaCy ya trae casi todo; este módulo solo lo formatea bonito
(consola en tabla, Telegram en bloques).
"""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


def analizar_morfologia(doc) -> list[dict[str, Any]]:
    """Devuelve una fila por token visible (sin espacio)."""
    if doc is None:
        return []
    filas: list[dict[str, Any]] = []
    for token in doc:
        if token.is_space:
            continue
        try:
            feats = token.morph.to_dict() if len(token.morph) > 0 else {}
        except (AttributeError, TypeError):
            feats = {}
        filas.append(
            {
                "texto": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "rasgos": feats,
            }
        )
    return filas


def morfologia_a_texto(doc) -> str:
    """Salida legible para el usuario (tabla en texto plano; mejor en consola con fuente monoespaciada)."""
    filas = analizar_morfologia(doc)
    if not filas:
        return "No hay análisis morfológico disponible (modelo spaCy no cargado o frase vacía)."
    lineas = [f"{'Token':<16} {'Lema':<16} {'POS':<8} {'Rasgos':<1}"]
    lineas.append("-" * 72)
    for r in filas:
        rasgos = ", ".join(f"{k}={v}" for k, v in sorted(r["rasgos"].items()) if v)
        if not rasgos:
            rasgos = "—"
        lineas.append(f"{r['texto']:<16} {r['lemma']:<16} {r['pos']:<8} {rasgos}")
    return "\n".join(lineas)


def morfologia_a_texto_bloques(doc) -> str:
    """Un bloque por token: se lee bien en Telegram sin fuente monoespaciada."""
    filas = analizar_morfologia(doc)
    if not filas:
        return "No hay análisis morfológico disponible (modelo spaCy no cargado o frase vacía)."
    bloques: list[str] = []
    for i, r in enumerate(filas, 1):
        rasgos = ", ".join(f"{k}={v}" for k, v in sorted(r["rasgos"].items()) if v)
        if not rasgos:
            rasgos = "—"
        bloques.append(
            f"{i}) Token: {r['texto']}\n"
            f"   Lema: {r['lemma']}\n"
            f"   POS: {r['pos']} (tag: {r['tag']})\n"
            f"   Rasgos: {rasgos}"
        )
    return "\n\n".join(bloques)
