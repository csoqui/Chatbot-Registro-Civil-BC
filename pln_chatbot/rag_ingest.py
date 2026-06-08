"""
Arma el .joblib del RAG: baja corpus, filtra, trocea y guarda.

Correr una vez (o cuando cambies filtros): python -m pln_chatbot.rag_ingest
Fuentes: eswiki en español + AbScientia (Hugging Face). Ver README.
"""

from __future__ import annotations

import argparse
import logging
import re
from typing import Iterable

from datasets import load_dataset

from pln_chatbot.config import RAG_INDEX_PATH, setup_logging
from pln_chatbot.rag import build_index, has_tech_keyword, save_index

logger = logging.getLogger(__name__)


def _load_streaming_dataset(repo_id: str):
    """
    Carga streaming dataset intentando splits comunes.
    Algunos datasets (p.ej. eswiki_20240401_corpus) usan 'corpus' en vez de 'train'.
    """
    preferred_splits = ("train", "corpus", "validation", "test")
    last_error: Exception | None = None
    for split in preferred_splits:
        try:
            ds = load_dataset(repo_id, split=split, streaming=True)
            logger.info("Dataset %s cargado con split='%s'", repo_id, split)
            return ds
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No se pudo cargar dataset {repo_id} en modo streaming.")


def _clean_text(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t


def _chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    text = _clean_text(text)
    if len(text) <= chunk_size:
        return [text] if text else []
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        cut = text.rfind(". ", start, end)
        if cut > start + 250:
            end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return out


def _tech_filter(text: str) -> bool:
    return has_tech_keyword(text)


# --- Wikipedia ES: streaming, filtro tech, trozos de ~700 caracteres ---
def _stream_eswiki(limit_rows: int) -> Iterable[dict]:
    ds = _load_streaming_dataset("spanish-ir/eswiki_20240401_corpus")
    count = 0
    for row in ds:
        text = _clean_text(str(row.get("text", "")))
        if not text or not _tech_filter(text):
            continue
        title = str(row.get("title", "") or "")
        for chunk in _chunk_text(text):
            yield {"text": chunk, "source": "eswiki_20240401", "title": title, "meta": {"dataset": "eswiki"}}
            count += 1
            if count >= limit_rows:
                return


# --- AbScientia: abstracts científicos (columnas variables según el dataset) ---
def _stream_abscientia(limit_rows: int) -> Iterable[dict]:
    ds = _load_streaming_dataset("BSC-LT/AbScientia")
    count = 0
    for row in ds:
        # Dataset puede variar por columna; intentamos campos comunes de abstracts.
        fields = ["abstract", "text", "sentence", "content", "article", "body"]
        text = ""
        for f in fields:
            if row.get(f):
                text = str(row.get(f))
                break
        text = _clean_text(text)
        if not text:
            continue
        # AbScientia ya es STEM, pero mantenemos filtro tech para ajustar al dominio.
        if not _tech_filter(text):
            continue
        title = str(row.get("title", "") or row.get("label", "") or "")
        for chunk in _chunk_text(text):
            yield {"text": chunk, "source": "abscientia", "title": title, "meta": {"dataset": "AbScientia"}}
            count += 1
            if count >= limit_rows:
                return


def main() -> int:
    parser = argparse.ArgumentParser(description="Construye índice RAG tecnológico local.")
    parser.add_argument("--max-eswiki", type=int, default=25000, help="Máximo chunks desde eswiki (filtrados).")
    parser.add_argument("--max-abscientia", type=int, default=12000, help="Máximo chunks desde AbScientia (filtrados).")
    args = parser.parse_args()

    setup_logging()
    logger.info("Iniciando ingesta RAG (eswiki + AbScientia)")
    docs: list[dict] = []

    logger.info("Cargando eswiki...")
    docs.extend(list(_stream_eswiki(args.max_eswiki)))
    logger.info("Chunks eswiki: %d", len(docs))

    logger.info("Cargando AbScientia...")
    before = len(docs)
    docs.extend(list(_stream_abscientia(args.max_abscientia)))
    logger.info("Chunks AbScientia: %d", len(docs) - before)

    if not docs:
        logger.error("No se extrajeron documentos. Revise conectividad o filtros.")
        return 1

    logger.info("Construyendo índice TF-IDF con %d documentos...", len(docs))
    index = build_index(docs)
    save_index(index, RAG_INDEX_PATH)
    logger.info("Completado. Índice listo: %s", RAG_INDEX_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
