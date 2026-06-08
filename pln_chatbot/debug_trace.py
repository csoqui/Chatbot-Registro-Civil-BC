"""
Líneas «DEBUG …» en consola para ver el camino del mensaje.

Útil cuando explicas en vivo: se ve si ganó NLTK, spaCy o el JSON.
En Telegram se apagan a propósito (handle_turn pone suppress) para no
llenar el chat del móvil. Control: /trace, PLN_CHATBOT_TRACE.
"""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from typing import Iterator, Optional

_override: Optional[bool] = None

# Cuando es True, debug_trace no imprime (p. ej. turno procesado para Telegram).
_suppress_stdout: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "pln_chatbot_suppress_pipeline_trace", default=False
)


def trace_enabled() -> bool:
    if _override is not None:
        return _override
    v = os.environ.get("PLN_CHATBOT_TRACE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def set_trace_override(on: Optional[bool]) -> None:
    """None = respetar solo PLN_CHATBOT_TRACE; True/False fuerza trazas en la sesión."""
    global _override
    _override = on


def debug_trace(msg: str) -> None:
    if _suppress_stdout.get():
        return
    if trace_enabled():
        print(msg, flush=True)


@contextmanager
def pipeline_traces_to_stdout(enabled: bool) -> Iterator[None]:
    """
    enabled=True: mismas reglas que siempre (PLN_CHATBOT_TRACE + /trace).
    enabled=False: no se imprimen líneas «DEBUG …» del pipeline (respuesta Telegram).
    """
    token = _suppress_stdout.set(not enabled)
    try:
        yield
    finally:
        _suppress_stdout.reset(token)
