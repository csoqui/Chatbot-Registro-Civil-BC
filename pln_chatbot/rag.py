"""
RAG casero: buscar párrafos parecidos en un índice local (TF-IDF).

No genera texto con GPT: recupera trozos de Wikipedia/AbScientia
que ya metiste en el .joblib. Solo se usa si JSON y WordNet no alcanzan.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pln_chatbot.nlp_utils import expandir_acronimo_tema, normalizar_clave_busqueda

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Lista para filtrar corpus y consultas: solo temática tech/IA (ingest + búsqueda)
TECH_KEYWORDS = {
    # IA y aprendizaje
    "inteligencia artificial",
    "ia",
    "aprendizaje automático",
    "aprendizaje automatico",
    "machine learning",
    "deep learning",
    "aprendizaje supervisado",
    "aprendizaje no supervisado",
    "aprendizaje por refuerzo",
    "reinforcement learning",
    "modelo fundacional",
    "modelos fundacionales",
    "modelo generativo",
    "modelos generativos",
    "red neuronal",
    "redes neuronales",
    "red neuronal convolucional",
    "cnn",
    "transformer",
    "embedding",
    "vector embedding",
    "modelo de lenguaje",
    "modelo de lenguaje grande",
    "llm",
    "fine tuning",
    "ajuste fino",
    "inferencia",
    "prompt",
    "ingeniería de prompts",
    "ingenieria de prompts",
    "rag",
    "retrieval augmented generation",
    "agente inteligente",
    "agentes inteligentes",
    "sistema experto",

    # NLP / lenguaje
    "nlp",
    "procesamiento de lenguaje natural",
    "procesamiento del lenguaje natural",
    "analisis sintactico",
    "análisis sintáctico",
    "analisis morfologico",
    "análisis morfológico",
    "etiquetado pos",
    "tokenización",
    "tokenizacion",
    "lematización",
    "lematizacion",
    "word embedding",
    "word2vec",
    "bert",
    "spacy",
    "nltk",

    # Datos y analítica
    "ciencia de datos",
    "data science",
    "big data",
    "analítica de datos",
    "analitica de datos",
    "analisis de datos",
    "análisis de datos",
    "minería de datos",
    "mineria de datos",
    "data mining",
    "ingeniería de datos",
    "ingenieria de datos",
    "data engineering",
    "etl",
    "elt",
    "data warehouse",
    "data lake",
    "data lakehouse",
    "business intelligence",
    "bi",
    "estadística",
    "estadistica",
    "visualización de datos",
    "visualizacion de datos",

    # Software y arquitectura
    "algoritmo",
    "algoritmos",
    "programación",
    "programacion",
    "software",
    "desarrollo de software",
    "arquitectura de software",
    "ingeniería de software",
    "ingenieria de software",
    "patrones de diseño",
    "patrones de diseno",
    "microservicios",
    "monolito",
    "api",
    "api rest",
    "graphql",
    "backend",
    "frontend",
    "full stack",
    "testing",
    "pruebas unitarias",
    "integración continua",
    "integracion continua",
    "ci/cd",
    "devops",
    "observabilidad",
    "logging",
    "monitorización",
    "monitorizacion",
    "docker",
    "kubernetes",

    # Sistemas, nube y redes
    "informatica",
    "telecomunicaciones",
    "telecomunicacion",
    "bases de datos",
    "base de datos",
    "sql",
    "nosql",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "sistemas distribuidos",
    "sistema distribuido",
    "sistemas operativos",
    "virtualización",
    "virtualizacion",
    "cloud",
    "computación en la nube",
    "nube",
    "aws",
    "azure",
    "gcp",
    "serverless",
    "redes",
    "redes de computadoras",
    "tcp/ip",
    "protocolo",
    "internet de las cosas",
    "iot",

    # Seguridad
    "computación",
    "computacion",
    "ciberseguridad",
    "seguridad",
    "seguridad informática",
    "seguridad informatica",
    "criptografía",
    "criptografia",
    "autenticación",
    "autenticacion",
    "autorización",
    "autorizacion",
    "zero trust",
    "phishing",
    "malware",
    "ransomware",

    # Interacción / multimedia / robótica
    "computación afectiva",
    "computacion afectiva",
    "computo afectivo",
    "cómputo afectivo",
    "interacción humano computadora",
    "interaccion humano computadora",
    "hci",
    "ux",
    "ui",
    "vision por computadora",
    "visión por computadora",
    "computer vision",
    "procesamiento de imagen",
    "reconocimiento de voz",
    "robótica",
    "robotica",
    "automatización",
    "automatizacion",
}

_TECH_KEYWORD_PATTERNS = {
    k: re.compile(rf"\b{re.escape(k)}\b", flags=re.IGNORECASE) for k in TECH_KEYWORDS if len(k) >= 3
}

# Acronimos cortos: evitar falsos positivos por substring ("ia" en astrologia).
_SHORT_ACRONYM_PATTERNS = {
    "ia": re.compile(r"(?<![a-záéíóúñ])ia(?![a-záéíóúñ])", flags=re.IGNORECASE),
    "ai": re.compile(r"(?<![a-záéíóúñ])ai(?![a-záéíóúñ])", flags=re.IGNORECASE),
    "ml": re.compile(r"(?<![a-záéíóúñ])ml(?![a-záéíóúñ])", flags=re.IGNORECASE),
    "dl": re.compile(r"(?<![a-záéíóúñ])dl(?![a-záéíóúñ])", flags=re.IGNORECASE),
    "nlp": re.compile(r"(?<![a-záéíóúñ])nlp(?![a-záéíóúñ])", flags=re.IGNORECASE),
}


@dataclass(frozen=True)
class RAGHit:
    score: float
    text: str
    source: str
    title: str | None = None
    meta: dict[str, Any] | None = None


class RAGIndex:
    def __init__(self, vectorizer: TfidfVectorizer, matrix, docs: list[dict[str, Any]]) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.docs = docs

    def search(self, query: str, top_k: int = 4, min_score: float = 0.12) -> list[RAGHit]:
        query = (query or "").strip()
        if not query:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        if sims.size == 0:
            return []
        idx_sorted = sims.argsort()[::-1][: max(1, top_k)]
        hits: list[RAGHit] = []
        for i in idx_sorted:
            score = float(sims[i])
            if score < min_score:
                continue
            d = self.docs[int(i)]
            hits.append(
                RAGHit(
                    score=score,
                    text=str(d.get("text", "")),
                    source=str(d.get("source", "desconocida")),
                    title=d.get("title"),
                    meta=d.get("meta") if isinstance(d.get("meta"), dict) else None,
                )
            )
        return hits


def build_index(docs: list[dict[str, Any]], max_features: int = 80000, min_df: int = 2) -> RAGIndex:
    """Convierte la lista de fragmentos en matriz TF-IDF (lo llama rag_ingest)."""
    texts = [str(d.get("text", "")).strip() for d in docs if str(d.get("text", "")).strip()]
    clean_docs = [d for d in docs if str(d.get("text", "")).strip()]
    if not texts:
        raise ValueError("No hay documentos válidos para construir índice RAG.")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, min_df=min_df)
    matrix = vectorizer.fit_transform(texts)
    return RAGIndex(vectorizer=vectorizer, matrix=matrix, docs=clean_docs)


def save_index(index: RAGIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"vectorizer": index.vectorizer, "matrix": index.matrix, "docs": index.docs}
    joblib.dump(payload, path)
    logger.info("Índice RAG guardado en %s (%d documentos)", path, len(index.docs))


def load_index(path: Path) -> RAGIndex | None:
    if not path.exists():
        logger.warning("Índice RAG no encontrado en %s", path)
        return None
    payload = joblib.load(path)
    return RAGIndex(vectorizer=payload["vectorizer"], matrix=payload["matrix"], docs=payload["docs"])


def has_tech_keyword(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    for _k, pat in _TECH_KEYWORD_PATTERNS.items():
        if pat.search(t):
            return True
    for _k, pat in _SHORT_ACRONYM_PATTERNS.items():
        if pat.search(t):
            return True
    return False


def is_tech_query(text: str) -> bool:
    return has_tech_keyword(text)


REGISTRO_CIVIL_KEYWORDS = {
    "registro civil",
    "acta",
    "actas",
    "nacimiento",
    "matrimonio",
    "defuncion",
    "defunción",
    "defunciones",
    "divorcio",
    "divorcios",
    "curp",
    "oficialia",
    "oficialía",
    "oficialias",
    "oficialías",
    "correccion",
    "corrección",
    "aclaracion",
    "aclaración",
    "rectificacion",
    "rectificación",
    "reconocimiento",
    "hijos",
    "adopcion",
    "adopción",
    "sentencia",
    "sentencias",
    "registro extemporaneo",
    "registro extemporáneo",
    "copia certificada",
    "certificada",
    "certificado",
    "tramite",
    "trámite",
    "tramites",
    "trámites",
    "requisitos",
    "manual",
    "procedimiento",
    "procedimientos",
    "derechos",
    "costos",
    "citas",
}


_REGISTRO_CIVIL_PATTERNS = {
    k: re.compile(rf"\b{re.escape(k)}\b", flags=re.IGNORECASE) for k in REGISTRO_CIVIL_KEYWORDS
}


def is_registro_civil_query(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(pat.search(t) for pat in _REGISTRO_CIVIL_PATTERNS.values())


def construir_consulta_rag(texto_usuario: str, tema_detectado: str | None) -> str:
    """
    Enriquece la consulta al índice TF-IDF para desambiguar acrónimos (p. ej. «Whats AI» → contexto de IA
    informática frente a coincidencias con «Apo AI» en biomedicina).
    """
    t = (texto_usuario or "").strip()
    hints: list[str] = []
    if tema_detectado:
        ex = expandir_acronimo_tema(tema_detectado)
        if ex and ex.lower() not in t.lower():
            hints.append(ex)
    low = t.lower()
    need_ia_ctx = False
    if re.search(r"(?<![a-záéíóúñ0-9])ai(?![a-záéíóúñ])", low) and "inteligencia" not in low and "artificial" not in low:
        need_ia_ctx = True
    if re.search(r"(?<![a-záéíóúñ0-9])ia(?![a-záéíóúñ])", low) and "inteligencia" not in low:
        need_ia_ctx = True
    if re.search(r"\b(what\s*is|what's|whats)\s+ai\b", low) and "inteligencia" not in low:
        need_ia_ctx = True
    if need_ia_ctx and not any("inteligencia artificial" in x.lower() for x in hints):
        hints.append("inteligencia artificial informática computación definición")
    seen: set[str] = set()
    uniq: list[str] = []
    for h in hints:
        hl = h.lower()
        if hl not in seen:
            seen.add(hl)
            uniq.append(h)
    return (t + " " + " ".join(uniq)).strip() if uniq else t


def _etiqueta_fuente_rag(source: str) -> str:
    s = (source or "").strip() or "referencia"
    head = s.split("_", 1)[0].lower()
    if head == "abscientia":
        return "AbScientia"
    return head


def _consulta_sobre_ia_informatica(tema: str | None, pregunta: str | None) -> bool:
    if tema and expandir_acronimo_tema(tema) == "inteligencia artificial":
        return True
    low = (pregunta or "").lower()
    if re.search(r"(?<![a-záéíóúñ0-9])ia(?![a-záéíóúñ])", low) and "inteligencia" not in low:
        return True
    if re.search(r"(?<![a-záéíóúñ0-9])ai(?![a-záéíóúñ])", low) and "inteligencia" not in low and "artificial" not in low:
        return True
    if re.search(r"\b(what\s*is|what's|whats)\s+ai\b", low):
        return True
    return False


def _fragmento_es_apo_ai_biologia(h: RAGHit) -> bool:
    blob = f"{h.title or ''} {h.text or ''}"
    return bool(re.search(r"\bapo[- ]?ai\b", blob, re.IGNORECASE))


def format_rag_answer(query: str, hits: list[RAGHit], max_chars: int = 1200) -> str:
    if not hits:
        return ""
    chunks: list[str] = []
    total = 0
    for h in hits[:3]:
        title = f" ({h.title})" if h.title else ""
        header = f"- Fuente: {h.source}{title} | score={h.score:.2f}"
        body = h.text.strip().replace("\n", " ")
        piece = f"{header}\n  {body}"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    if not chunks:
        return ""
    return (
        f"No tenía una entrada directa en la base local para «{query}», "
        "pero encontré contenido en el índice RAG local:\n"
        + "\n".join(chunks)
    )


_STOP_TEMA = frozenset(
    {
        "que",
        "quien",
        "quienes",
        "cual",
        "cuales",
        "son",
        "es",
        "las",
        "los",
        "del",
        "una",
        "unos",
        "unas",
        "como",
        "para",
        "por",
        "dime",
        "define",
        "sobre",
        "hacen",
        "hay",
    }
)


def rag_tema_hint(tema: str | None) -> str | None:
    """Último sustantivo / bigrama útil (p. ej. «que son las telecomunicaciones» → «telecomunicaciones»)."""
    if not tema:
        return None
    base = (normalizar_clave_busqueda(tema) or tema).lower().strip()
    parts = [p for p in base.split() if p not in _STOP_TEMA and len(p) > 2]
    if not parts:
        parts = base.split()
    if len(parts) >= 2 and parts[-1] in {
        "artificial",
        "automatico",
        "automatica",
        "supervisado",
        "supervisada",
        "profundo",
        "profunda",
        "natural",
        "informatica",
        "computacion",
    }:
        return f"{parts[-2]} {parts[-1]}"
    return parts[-1] if parts else None


def _rag_rank_for_chat(h: RAGHit, tema_hint: str | None) -> tuple[int, float]:
    """Prioriza fragmentos cuyo título trata el tema (p. ej. «Telecomunicaciones en…») sobre menciones tangenciales."""
    hint = (tema_hint or "").strip().lower()
    if not hint:
        return (0, h.score)
    title_l = (h.title or "").lower()
    text_l = (h.text or "").lower()
    if hint in title_l:
        return (3, h.score)
    if title_l.startswith(hint) or f" {hint}" in title_l:
        return (3, h.score)
    if hint in text_l[:400]:
        return (1, h.score)
    return (0, h.score)


def format_rag_answer_chat(
    hits: list[RAGHit],
    *,
    tema: str | None = None,
    pregunta: str | None = None,
    max_chars: int = 2200,
    max_passages: int = 3,
) -> str:
    """
    Respuesta RAG para Telegram: encabezado fijo + un bloque por fuente (etiqueta + texto), sin scores.
    """
    if not hits:
        return ""
    if _consulta_sobre_ia_informatica(tema, pregunta):
        hits = [h for h in hits if not _fragmento_es_apo_ai_biologia(h)]
    if not hits:
        return ""
    tema_hint = rag_tema_hint(tema)
    ranked = sorted(hits, key=lambda h: _rag_rank_for_chat(h, tema_hint), reverse=True)
    take = max_passages
    if ranked and _rag_rank_for_chat(ranked[0], tema_hint)[0] >= 3:
        take = min(take, 2)
    intro = (
        "No hay una entrada exacta en mi base local para esa redacción; "
        "esto es un resumen tomado de material de referencia (índice RAG / fuentes abiertas):\n"
    )
    blocks: list[str] = []
    seen_prefix: set[str] = set()
    total = len(intro)
    for h in ranked[:take]:
        body = (h.text or "").strip().replace("\n", " ")
        if len(body) > 900:
            body = body[:900].rsplit(" ", 1)[0] + "…"
        if not body:
            continue
        label = _etiqueta_fuente_rag(h.source)
        dedupe_key = f"{label}:{body[:240]}"
        if dedupe_key in seen_prefix:
            continue
        seen_prefix.add(dedupe_key)
        block = f"{label}:\n{body}"
        if total + len(block) + 2 > max_chars:
            break
        blocks.append(block)
        total += len(block) + 2
    if not blocks:
        return ""
    return intro + "\n" + "\n\n".join(blocks)
