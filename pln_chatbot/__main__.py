"""
Punto de entrada del paquete: chatbot interactivo en consola.

  cd ruta\\al\\proyecto\\Chatbot
  .\\.venv\\Scripts\\Activate.ps1
  python -m pln_chatbot

Atajo equivalente: python chatbot.py

Ver README.md (consola, Telegram, instalación, RAG).
"""

from pln_chatbot.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
