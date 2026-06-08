"""
Atajo para consola: hace lo mismo que `python -m pln_chatbot`.

Este archivo existe por comodidad (doble clic o `python chatbot.py`).
La lógica real está en pln_chatbot/ (cli, dialogue, etc.).

Instalación (una vez), desde la raíz del repo:
  python -m venv .venv
  .\\.venv\\Scripts\\Activate.ps1          # PowerShell
  .\\.venv\\Scripts\\activate.bat           # CMD
  pip install -r requirements.txt
  python -m spacy download es_core_news_lg

Ejecutar en consola (venv activado):
  python chatbot.py
  python -m pln_chatbot                     # equivalente

Telegram (token de @BotFather, misma ventana de terminal):
  $env:TELEGRAM_BOT_TOKEN="SU_TOKEN"
  python -m pln_chatbot.telegram_bot

Guía completa (BotFather, RAG, variables): README.md en la raíz del proyecto.
"""

from pln_chatbot.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
