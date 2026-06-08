"""
Consola: el bucle de «escribes → el bot responde».

Este archivo solo arma la sesión (carga JSON, spaCy, NLTK, RAG) y se queda
leyendo líneas. La lógica de cada mensaje está en interaction.py.
"""

from __future__ import annotations
import logging
import sys

from pln_chatbot.config import KNOWLEDGE_PATH, setup_logging
from pln_chatbot.debug_trace import trace_enabled
from pln_chatbot.dialogue import initial_historial
from pln_chatbot.interaction import handle_turn
from pln_chatbot.knowledge import load_knowledge
from pln_chatbot.resources import build_app_resources
from pln_chatbot.voice_input import listen_once


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    setup_logging()
    log = logging.getLogger("pln_chatbot.cli")

    # --- Arranque: una sola vez por sesión ---
    # kb = respuestas del JSON; res = modelos pesados (spaCy, parser NLTK, índice RAG)
    kb = load_knowledge(KNOWLEDGE_PATH)
    res = build_app_resources()
    historial = initial_historial()  # contador de «no entendí» para mensajes de ayuda

    print("=" * 64)
    print(f"{kb.assistant_name} — plantilla PLN (consola)")
    print("=" * 64)
    print(kb.welcome_hint)
    print("Escriba /ayuda para comandos didácticos, /voz para hablar o «salir» para terminar.")
    if trace_enabled():
        print("Trazas «DEBUG …» del pipeline: ACTIVAS en esta consola (/trace off para silenciar).")
    else:
        print(
            "Trazas «DEBUG …» del pipeline: DESACTIVADAS (p. ej. PLN_CHATBOT_TRACE=0 en el entorno)."
        )
        print("  Active con: /trace on   o   $env:PLN_CHATBOT_TRACE=\"1\"   (PowerShell, misma ventana).")
    print("-" * 64)

    # --- Bucle principal: cada línea es un turno de conversación ---
    while True:
        try:
            line = input("\nUsted: ").strip().lstrip("\ufeff")  # quita BOM si pegan desde Word
        except (KeyboardInterrupt, EOFError):
            print("\nAgente: Sesión terminada.")
            return 0

        if not line:
            continue

        if line.lower() == "/voz":
            voice = listen_once()
            if voice.error:
                print(f"Agente: {voice.error}")
                continue
            if not voice.text:
                print("Agente: No recibí texto desde el micrófono.")
                continue
            print(f"Usted dijo: {voice.text}")
            line = voice.text

        turn = handle_turn(line, kb=kb, res=res, historial=historial, log=log)
        for msg in turn.messages:
            print(f"Agente: {msg}")
        if turn.quit_console:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
