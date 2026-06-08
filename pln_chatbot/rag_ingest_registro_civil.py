"""
Construye un índice RAG local para Registro Civil BC desde documentos propios.

Uso recomendado:
python -m pln_chatbot.rag_ingest_registro_civil --source data/rag/registro_civil_bc --output data/rag/rag_registro_civil_bc_index.joblib
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from pln_chatbot.config import PROJECT_ROOT, setup_logging
from pln_chatbot.rag import build_index, save_index

logger = logging.getLogger(__name__)


DEFAULT_SOURCE = PROJECT_ROOT / "data" / "rag" / "registro_civil_bc"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "rag" / "rag_registro_civil_bc_index.joblib"


def _clean_text(text: str) -> str:
    text = (text or "").replace("\ufeff", " ").replace("\x0c", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 160) -> list[str]:
    text = re.sub(r"\s+", " ", _clean_text(text))
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        cut = text.rfind(". ", start, end)
        if cut > start + 300:
            end = cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def _iter_source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    allowed = {".txt", ".md"}
    return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in allowed)


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_docs(source: Path) -> list[dict]:
    docs: list[dict] = []
    files = _iter_source_files(source)
    for file_path in files:
        text = _clean_text(_read_text_file(file_path))
        if not text:
            logger.warning("Archivo sin texto útil: %s", file_path)
            continue
        title = file_path.stem.replace("_", " ").replace("-", " ")
        for chunk in _chunk_text(text):
            docs.append(
                {
                    "text": chunk,
                    "source": "manual_registro_civil_bc",
                    "title": title,
                    "meta": {"path": _relative_path(file_path), "domain": "Registro Civil de Baja California"},
                }
            )
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description="Construye índice RAG local para Registro Civil BC.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Archivo .txt/.md o carpeta con documentos.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Ruta del .joblib de salida.")
    args = parser.parse_args()

    setup_logging()
    source = args.source.resolve()
    output = args.output.resolve()

    if not source.exists():
        logger.error("No existe la fuente de documentos: %s", source)
        return 1

    docs = build_docs(source)
    if not docs:
        logger.error("No se extrajeron fragmentos. Convierte el manual .doc a .txt y vuelve a ejecutar.")
        return 1

    logger.info("Construyendo índice RAG Registro Civil BC con %d fragmentos...", len(docs))
    index = build_index(docs, max_features=50000)
    save_index(index, output)
    logger.info("Índice listo: %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
