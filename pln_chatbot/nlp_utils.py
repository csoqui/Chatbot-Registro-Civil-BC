"""
Normalizar texto para que las búsquedas no fallen por tildes o mayúsculas.

También mapea siglas (IA, ML…) a la clave larga del JSON para que
«qué es IA» y «qué es inteligencia artificial» apunten al mismo sitio.
"""

from __future__ import annotations

import unicodedata


def quitar_acentos(texto: str) -> str:
    if not isinstance(texto, str):
        return str(texto)
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizar_clave_busqueda(texto: str) -> str | None:
    """Minúsculas, sin acentos; quita artículo inicial si existe."""
    if not isinstance(texto, str):
        return None
    t = quitar_acentos(texto.lower().strip())
    partes = t.split()
    articulos = {"el", "la", "los", "las", "un", "una", "unos", "unas"}
    if len(partes) > 1 and partes[0] in articulos:
        return " ".join(partes[1:])
    if len(partes) == 1 and partes[0] in articulos:
        return partes[0]
    return t


# Si el usuario dice «IA», buscamos la entrada «inteligencia artificial» en el JSON
_ACRONIMO_A_CLAVE_CANONICA: dict[str, str] = {
    "ia": "inteligencia artificial",
    "ai": "inteligencia artificial",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "nlp",
    "pln": "procesamiento de lenguaje natural",
    "artificial intelligence": "inteligencia artificial",
}


def expandir_acronimo_tema(tema: str) -> str:
    """
    Si el tema (tras normalizar) es un acrónimo o nombre corto conocido, devuelve la clave canónica
    usada en la base local para que «IA», «AI» y «inteligencia artificial» compartan respuesta.
    """
    if not isinstance(tema, str) or not tema.strip():
        return tema
    k = normalizar_clave_busqueda(tema) or tema.strip().lower()
    k = k.strip()
    if k in _ACRONIMO_A_CLAVE_CANONICA:
        return _ACRONIMO_A_CLAVE_CANONICA[k]
    return tema
