"""
CFG sobre la frase: ¿parece una pregunta del tipo «qué es …»?

spaCy nos da palabras; aquí las resumimos en símbolos cortos (QUES, NOM…)
y vemos si encajan en reglas. Sirve para /syntax y para la nota técnica
en consola; el bot puede responder aunque la CFG diga que no cuadra.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import nltk
from nltk import CFG
from nltk.parse import ChartParser

logger = logging.getLogger(__name__)

# Gramática chiquita: no modela todo el español, solo esqueletos de preguntas frecuentes
_CFG_SOURCE = r"""
% start S
S -> 'QUES' 'COPUL' REST
S -> 'QUES' 'SABES' 'DE' REST
S -> 'DEF' REST
S -> 'ACTV' 'QUES' 'COPUL' REST
S -> 'ACTV' REST
S -> 'HABL' 'DE' REST
S -> 'CUENT' 'DE' REST
S -> 'CUENT' 'SOBRE' REST
S -> 'DIME' 'DE' REST
S -> 'DIME' 'SOBRE' REST
S -> 'INFO' 'SOBRE' REST

REST -> 'NOM'
REST -> 'DET' 'NOM'
REST -> 'DET' 'NOM' 'NOM'
"""


def _load_cfg() -> CFG:
    return CFG.fromstring(_CFG_SOURCE)


QU_SET = {"qué", "que", "cuál", "cual", "cuáles", "cuales"}
COP_SET = {"es", "son", "ser", "era", "fue", "está", "están", "estuvo", "estuvieron"}


def _content_tokens(doc) -> Iterator:
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        yield t


def doc_to_coarse_sequence(doc) -> list[str]:
    """Convierte un Doc de spaCy en secuencia de preterminales."""
    if doc is None:
        return []
    tokens = list(_content_tokens(doc))
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        low = t.lower_
        lem = (t.lemma_ or "").lower()

        if i + 2 < n and low in ("qué", "que") and tokens[i + 1].lower_ == "sabes" and tokens[i + 2].lower_ == "de":
            out.extend(["QUES", "SABES", "DE"])
            i += 3
            continue

        if low in QU_SET:
            out.append("QUES")
            i += 1
            continue

        if low in COP_SET or (t.pos_ in ("AUX", "VERB") and low in ("es", "son", "está", "están", "soy", "somos")):
            out.append("COPUL")
            i += 1
            continue

        if low in ("háblame", "hablame"):
            out.append("HABL")
            i += 1
            continue
        if low in ("cuéntame", "cuentame"):
            out.append("CUENT")
            i += 1
            continue
        if low == "dime":
            out.append("DIME")
            i += 1
            continue
        if low in ("información", "informacion"):
            out.append("INFO")
            i += 1
            continue
        if low == "define" or lem == "definir":
            out.append("DEF")
            i += 1
            continue
        if lem == "explicar" or low in ("explícame", "explicame"):
            out.append("ACTV")
            i += 1
            continue

        if low == "de":
            out.append("DE")
            i += 1
            continue
        if low == "sobre":
            out.append("SOBRE")
            i += 1
            continue

        if t.pos_ == "DET":
            out.append("DET")
            i += 1
            continue

        if t.pos_ in ("NOUN", "PROPN", "ADJ", "X", "NUM", "SYM"):
            j = i + 1
            while j < n and tokens[j].pos_ in ("NOUN", "PROPN", "ADJ", "X", "NUM", "SYM"):
                j += 1
            out.append("NOM")
            i = j
            continue

        # Palabras funcionales no modeladas: se omiten de la secuencia CFG.
        if t.is_stop:
            i += 1
            continue
        i += 1
    return out


@dataclass(frozen=True)
class SyntaxValidation:
    accepted: bool
    sequence: tuple[str, ...]
    trees: tuple[str, ...]
    ambiguity: bool


class SyntaxCFGValidator:
    """Pasa el Doc de spaCy por la CFG y devuelve si «cuadra» y el árbol para /syntax."""

    def __init__(self) -> None:
        self._cfg = _load_cfg()
        self._parser = ChartParser(self._cfg)

    def validate_doc(self, doc) -> SyntaxValidation:
        seq = tuple(doc_to_coarse_sequence(doc))
        if not seq:
            return SyntaxValidation(False, seq, (), False)
        trees: list[str] = []
        try:
            parses = list(self._parser.parse(seq))
        except ValueError as e:
            logger.debug("Fallo CFG parse: %s | seq=%s", e, seq)
            return SyntaxValidation(False, seq, (), False)
        for tr in parses[:5]:
            trees.append(tr.pformat(margin=100))
        amb = len(parses) > 1
        return SyntaxValidation(bool(parses), seq, tuple(trees), amb)

    def describe(self, doc, *, max_chars: int | None = None) -> str:
        v = self.validate_doc(doc)
        seq_s = " ".join(v.sequence) if v.sequence else "(vacía)"
        lines = [
            f"Secuencia de preterminales: {seq_s}",
            f"Aceptada por la CFG: {'sí' if v.accepted else 'no'}",
        ]
        if v.accepted and v.ambiguity:
            lines.append("Ambigüedad: se encontró más de un árbol de derivación (típico en CFG).")
        if v.trees:
            lines.append("Primer árbol:")
            lines.append(v.trees[0])
        out = "\n".join(lines)
        if max_chars is not None and len(out) > max_chars:
            return out[: max_chars - 80].rstrip() + "\n\n… (texto acortado: el árbol CFG es muy largo para Telegram; pruebe una frase más corta o use la consola.)"
        return out


def ensure_nltk_cfg_deps() -> None:
    """punkt puede ser requerido por otras rutas; la CFG opera sobre lista ya tokenizada."""
    # No-op documentado: ChartParser no exige punkt para listas de strings.
    return
