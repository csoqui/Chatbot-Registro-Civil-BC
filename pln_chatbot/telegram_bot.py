"""
Telegram: misma lógica que consola, pero con la API de python-telegram-bot.

Tu PC tiene que estar prendida con este script corriendo (long polling).
Cada handler llama a interaction.handle_turn; no reescribimos el diálogo aquí.
Ver README para BotFather y el token.
"""

from __future__ import annotations

import logging
import os
import sys

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from pln_chatbot.config import KNOWLEDGE_PATH, setup_logging
from pln_chatbot.dialogue import initial_historial
from pln_chatbot.interaction import chunk_for_telegram, handle_turn
from pln_chatbot.knowledge import KnowledgeBase, load_knowledge
from pln_chatbot.resources import AppResources, build_app_resources

log = logging.getLogger(__name__)

# Cargados en main(); los handlers los usan globalmente (un solo bot, un proceso)
_kb: KnowledgeBase | None = None
_res: AppResources | None = None
_hist_by_chat: dict[int, dict] = {}  # historial por chat de Telegram (cada usuario)


def _hist(chat_id: int) -> dict:
    if chat_id not in _hist_by_chat:
        _hist_by_chat[chat_id] = initial_historial()
    return _hist_by_chat[chat_id]


def telegram_welcome() -> str:
    assert _kb is not None
    return (
        f"Hola, soy { _kb.assistant_name }.\n\n"
        f"Puedo orientarte sobre temas de { _kb.domain_label }.\n\n"
        f"{ _kb.welcome_hint }\n\n"
        "La información es únicamente orientativa y no sustituye la atención oficial."
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(telegram_welcome())


async def cmd_ayuda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert _kb is not None and _res is not None
    turn = handle_turn(
        "/ayuda",
        kb=_kb,
        res=_res,
        historial=_hist(update.effective_chat.id),
        log=log,
        ui_channel="telegram",
    )
    for msg in turn.messages:
        for part in chunk_for_telegram(msg):
            if update.message:
                await update.message.reply_text(part)


async def _reply_turn(update: Update, turn) -> None:
    if not update.message:
        return
    for msg in turn.messages:
        for part in chunk_for_telegram(msg):
            await update.message.reply_text(part)


async def cmd_temas(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert _kb is not None and _res is not None
    turn = handle_turn(
        "lista de temas",
        kb=_kb,
        res=_res,
        historial=_hist(update.effective_chat.id),
        log=log,
        ui_channel="telegram",
    )
    await _reply_turn(update, turn)


async def cmd_morfologia(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert _kb is not None and _res is not None
    args = context.args or []
    if not args:
        if update.message:
            await update.message.reply_text(
                "Uso: /morfologia <frase>\nEjemplo: /morfologia Hola como estas amigo"
            )
        return
    line = "/morfologia " + " ".join(args)
    turn = handle_turn(
        line,
        kb=_kb,
        res=_res,
        historial=_hist(update.effective_chat.id),
        log=log,
        ui_channel="telegram",
    )
    await _reply_turn(update, turn)


async def cmd_syntax(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert _kb is not None and _res is not None
    args = context.args or []
    if not args:
        if update.message:
            await update.message.reply_text(
                "Uso: /syntax <frase>\nEjemplo: /syntax qué es inteligencia artificial"
            )
        return
    line = "/syntax " + " ".join(args)
    turn = handle_turn(
        line,
        kb=_kb,
        res=_res,
        historial=_hist(update.effective_chat.id),
        log=log,
        ui_channel="telegram",
    )
    await _reply_turn(update, turn)


async def cmd_wordnet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert _kb is not None and _res is not None
    args = context.args or []
    if not args:
        if update.message:
            await update.message.reply_text("Uso: /wordnet <término>\nEjemplo: /wordnet casa")
        return
    line = "/wordnet " + " ".join(args)
    turn = handle_turn(
        line,
        kb=_kb,
        res=_res,
        historial=_hist(update.effective_chat.id),
        log=log,
        ui_channel="telegram",
    )
    await _reply_turn(update, turn)


async def cmd_salir(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert _kb is not None and _res is not None
    turn = handle_turn(
        "/salir",
        kb=_kb,
        res=_res,
        historial=_hist(update.effective_chat.id),
        log=log,
        ui_channel="telegram",
    )
    await _reply_turn(update, turn)


async def post_init(application: Application) -> None:
    """Registra el menú que ves al pulsar / en Telegram (start, ayuda, morfologia…)."""
    try:
        await application.bot.set_my_commands(
            [
                BotCommand("start", "Bienvenida y objetivo del bot"),
                BotCommand("ayuda", "Lista de comandos y dominio"),
                BotCommand("temas", "Ejemplos de conceptos y cómo preguntar"),
                BotCommand("morfologia", "Morfología spaCy: escriba la frase después"),
                BotCommand("syntax", "Validación CFG: escriba la frase después"),
                BotCommand("wordnet", "WordNet: escriba el término después"),
                BotCommand("salir", "Despedida"),
            ]
        )
    except Exception as e:
        log.warning("No se pudo actualizar el menú de comandos en Telegram: %s", e)


# Mensajes normales (sin /): «qué es IA», «hola», etc.
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    assert _kb is not None and _res is not None
    chat_id = update.effective_chat.id
    turn = handle_turn(
        update.message.text,
        kb=_kb,
        res=_res,
        historial=_hist(chat_id),
        log=log,
        ui_channel="telegram",
    )
    for msg in turn.messages:
        for part in chunk_for_telegram(msg):
            await update.message.reply_text(part)


def main() -> int:
    global _kb, _res

    # Sin token no arrancamos (evita errores crípticos de la API)
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print(
            "Falta TELEGRAM_BOT_TOKEN.\n"
            "1) En Telegram abra @BotFather → /newbot → copie el token.\n"
            "2) PowerShell:\n"
            '   $env:TELEGRAM_BOT_TOKEN="SU_TOKEN_AQUI"\n'
            "   python -m pln_chatbot.telegram_bot\n",
            file=sys.stderr,
        )
        return 1

    setup_logging()
    _kb = load_knowledge(KNOWLEDGE_PATH)
    _res = build_app_resources()

    log.info("Iniciando Telegram (long polling). Dominio: %s", _kb.domain_label)

    app = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .build()
    )
    # Solo chats privados: en grupos el bot no contesta (así lo dejamos a propósito)
    priv = filters.ChatType.PRIVATE
    app.add_handler(CommandHandler("start", cmd_start, filters=priv))
    app.add_handler(CommandHandler("ayuda", cmd_ayuda, filters=priv))
    app.add_handler(CommandHandler("help", cmd_ayuda, filters=priv))
    app.add_handler(CommandHandler("temas", cmd_temas, filters=priv))
    app.add_handler(CommandHandler("morfologia", cmd_morfologia, filters=priv))
    app.add_handler(CommandHandler("syntax", cmd_syntax, filters=priv))
    app.add_handler(CommandHandler("wordnet", cmd_wordnet, filters=priv))
    app.add_handler(CommandHandler("salir", cmd_salir, filters=priv))
    app.add_handler(MessageHandler(filters.TEXT & priv, on_text))

    app.run_polling(allowed_updates=Update.ALL_TYPES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
