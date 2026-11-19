"""The few Telegram Bot API calls we need, over plain HTTPS."""
import html

import httpx

from . import config

API = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}"
FILE_API = f"https://api.telegram.org/file/bot{config.TELEGRAM_BOT_TOKEN}"
_client = httpx.Client(timeout=70)


def call(method: str, **params) -> dict:
    r = _client.post(f"{API}/{method}", json={k: v for k, v in params.items() if v is not None})
    d = r.json()
    if not d.get("ok"):
        raise RuntimeError(f"telegram {method}: {d.get('description')}")
    return d["result"]


def esc(s) -> str:
    return html.escape(str(s or ""), quote=False)


def send(chat_id: int, text: str, buttons: list[list[tuple[str, str]]] | None = None, html_mode: bool = False,
         preview: bool = False) -> dict | None:
    """buttons = rows of (label, callback_data). Long texts are split at Telegram's 4096-char limit."""
    markup = None
    if buttons:
        markup = {"inline_keyboard": [[({"text": lbl, "url": data[4:]} if data.startswith("url:")
                                        else {"text": lbl, "callback_data": data}) for lbl, data in row]
                                      for row in buttons]}
    chunks = [text[i:i + 3900] for i in range(0, len(text), 3900)] or [""]
    last = None
    for i, chunk in enumerate(chunks):
        try:
            last = call("sendMessage", chat_id=chat_id, text=chunk, parse_mode="HTML" if html_mode else None,
                        reply_markup=markup if i == len(chunks) - 1 else None,
                        link_preview_options={"is_disabled": not preview},
                        disable_notification=True if config.QUIET else None)
        except RuntimeError:
            if html_mode:                              # bad markup must never lose the message
                last = call("sendMessage", chat_id=chat_id, text=chunk,
                            reply_markup=markup if i == len(chunks) - 1 else None,
                            disable_notification=True if config.QUIET else None)
            else:
                raise
    return last


def typing(chat_id: int) -> None:
    try:
        call("sendChatAction", chat_id=chat_id, action="typing")
    except Exception:  # noqa: BLE001
        pass


def answer_callback(callback_id: str, text: str | None = None) -> None:
    try:
        call("answerCallbackQuery", callback_query_id=callback_id, text=text)
    except Exception:  # noqa: BLE001
        pass


def edit_buttons(chat_id: int, message_id: int, buttons: list[list[tuple[str, str]]] | None) -> None:
    markup = {"inline_keyboard": [[({"text": lbl, "url": data[4:]} if data.startswith("url:")
                                    else {"text": lbl, "callback_data": data}) for lbl, data in row]
                                  for row in (buttons or [])]}
    try:
        call("editMessageReplyMarkup", chat_id=chat_id, message_id=message_id, reply_markup=markup)
    except Exception:  # noqa: BLE001
        pass


def download(file_id: str, max_bytes: int = 5_000_000) -> bytes:
    info = call("getFile", file_id=file_id)
    if info.get("file_size", 0) > max_bytes:
        raise RuntimeError("file too large")
    r = _client.get(f"{FILE_API}/{info['file_path']}")
    r.raise_for_status()
    return r.content


def get_updates(offset: int | None, timeout: int = 50) -> list[dict]:
    return call("getUpdates", offset=offset, timeout=timeout, allowed_updates=["message", "callback_query"])
