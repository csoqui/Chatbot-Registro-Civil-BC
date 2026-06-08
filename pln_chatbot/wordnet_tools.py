"""
WordNet cuando el JSON no tiene la palabra exacta.

Busca sinónimos y definiciones en la red léxica (idioma spa en el JSON).
Se usa en /wordnet y como puente antes de rendirse y pasar al RAG.
"""

from __future__ import annotations

import logging
from typing import Any

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from pln_chatbot.nlp_utils import normalizar_clave_busqueda

logger = logging.getLogger(__name__)


def map_spacy_pos_to_wordnet_pos(spacy_pos: str | None):
    if spacy_pos in ("NOUN", "PROPN"):
        return wordnet.NOUN
    if spacy_pos == "VERB":
        return wordnet.VERB
    if spacy_pos == "ADJ":
        return wordnet.ADJ
    if spacy_pos == "ADV":
        return wordnet.ADV
    return None


def obtener_sinonimos_wn(
    palabra_o_frase: str,
    *,
    nlp,
    lemmatizer: WordNetLemmatizer,
    lang: str = "spa",
    pos_spacy_tag: str | None = None,
    limit: int = 5,
) -> list[str]:
    sinonimos: set[str] = set()
    norm = normalizar_clave_busqueda(palabra_o_frase)
    if not norm:
        return []

    wn_pos = map_spacy_pos_to_wordnet_pos(pos_spacy_tag)
    synsets = wordnet.synsets(norm.replace(" ", "_"), lang=lang, pos=wn_pos)
    if not synsets and wn_pos:
        synsets = wordnet.synsets(norm.replace(" ", "_"), lang=lang)

    if not synsets:
        if " " in norm and nlp is not None:
            doc_frase = nlp(norm)
            palabras_clave = [
                (tok.lemma_, tok.pos_)
                for tok in doc_frase
                if tok.pos_ in ("NOUN", "PROPN", "ADJ", "VERB") and not tok.is_stop
            ]
            if not palabras_clave:
                palabras_clave = [(p, None) for p in norm.split()]
            for p, pos_tag in palabras_clave:
                for s in obtener_sinonimos_wn(p, nlp=nlp, lemmatizer=lemmatizer, lang=lang, pos_spacy_tag=pos_tag, limit=limit):
                    if normalizar_clave_busqueda(s) != norm:
                        sinonimos.add(s)
                if len(sinonimos) >= limit:
                    break
        elif " " in norm:
            for p in norm.split():
                for s in obtener_sinonimos_wn(p, nlp=None, lemmatizer=lemmatizer, lang=lang, pos_spacy_tag=None, limit=limit):
                    if normalizar_clave_busqueda(s) != norm:
                        sinonimos.add(s)
                if len(sinonimos) >= limit:
                    break
        return list(sinonimos)[:limit]

    for synset in synsets:
        for lemma in synset.lemmas(lang=lang):
            nombre = lemma.name().replace("_", " ")
            nombre_n = normalizar_clave_busqueda(nombre)
            if nombre_n and nombre_n != norm and nombre_n != normalizar_clave_busqueda(palabra_o_frase):
                sinonimos.add(nombre_n)
            if len(sinonimos) >= limit:
                break
        if len(sinonimos) >= limit:
            break

    return list(sinonimos)[:limit]


def obtener_sinonimos_definiciones_wn(
    palabra: str,
    *,
    nlp,
    lang: str = "spa",
    max_defs: int = 2,
    max_syns_total: int = 5,
) -> dict[str, Any]:
    resultados: dict[str, Any] = {"definiciones": [], "sinonimos": set()}
    palabra_busq = normalizar_clave_busqueda(palabra)
    if not palabra_busq:
        return resultados

    try:
        synsets_encontrados: list = wordnet.synsets(palabra_busq.replace(" ", "_"), lang=lang)
        if not synsets_encontrados and nlp is not None:
            doc_palabra = nlp(palabra_busq)
            for token in doc_palabra:
                if token.pos_ in ("NOUN", "PROPN", "ADJ", "VERB") and not token.is_stop:
                    pos = map_spacy_pos_to_wordnet_pos(token.pos_)
                    temp = wordnet.synsets(token.lemma_.replace(" ", "_"), lang=lang, pos=pos) or wordnet.synsets(
                        token.lemma_.replace(" ", "_"), lang=lang
                    )
                    synsets_encontrados.extend(temp)
                    if len(synsets_encontrados) > max_defs * 3:
                        break
        elif not synsets_encontrados:
            for p in palabra_busq.split():
                synsets_encontrados.extend(wordnet.synsets(p.replace(" ", "_"), lang=lang))

        defs_count = 0
        for synset in synsets_encontrados:
            if defs_count < max_defs and synset.definition():
                resultados["definiciones"].append(f"- {synset.definition()}")
                defs_count += 1
            for lemma in synset.lemmas(lang=lang):
                if len(resultados["sinonimos"]) < max_syns_total:
                    nombre = lemma.name().replace("_", " ")
                    nn = normalizar_clave_busqueda(nombre)
                    if nn and nn != palabra_busq and nn != normalizar_clave_busqueda(palabra):
                        resultados["sinonimos"].add(nn)
            if defs_count >= max_defs and len(resultados["sinonimos"]) >= max_syns_total:
                break

        if not resultados["definiciones"]:
            resultados["definiciones"] = ["No encontré una definición clara en WordNet para este término (idioma solicitado)."]
    except Exception as e:
        logger.warning("WordNet error: %s", e)
        resultados["definiciones"] = ["Error al consultar WordNet."]

    resultados["sinonimos"] = list(resultados["sinonimos"])
    return resultados
