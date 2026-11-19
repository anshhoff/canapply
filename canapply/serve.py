"""One process for a small host: the HTTP API in front, the Telegram bot behind it.

Two ways to receive Telegram updates:
- No PUBLIC_URL (a laptop): long polling in a thread, as `python -m canapply.bot` does.
- PUBLIC_URL set (a host): webhook mode. Telegram posts updates to /telegram/webhook, which also wakes a
  free-tier host that has gone to sleep. A small ticker thread sends the daily picks and pings our own
  public URL so the host does not spin the service down between messages.
"""
import os
import threading
import time
import traceback

import httpx
import uvicorn

from . import bot, config, db
from . import telegram as tg
from .api import app

TICK_SECONDS = 300
KEEPALIVE_SECONDS = 600        # hosts spin an idle free service down after 15 minutes


def ticker() -> None:
    last_ping = 0.0
    while True:
        try:
            bot.run_digests_if_due()
        except Exception:  # noqa: BLE001  - one bad day must not stop tomorrow's picks
            traceback.print_exc()
        if time.time() - last_ping > KEEPALIVE_SECONDS:
            last_ping = time.time()
            try:
                httpx.get(config.PUBLIC_URL + "/ping", timeout=20)
            except Exception as e:  # noqa: BLE001
                print("keep-alive ping failed:", str(e)[:120], flush=True)
        time.sleep(TICK_SECONDS)


def start_webhook_mode() -> None:
    db.init_schema()
    tg.set_webhook(config.PUBLIC_URL + "/telegram/webhook")
    print("CanApply in webhook mode at", config.PUBLIC_URL, flush=True)
    ticker()


def main() -> None:
    target = start_webhook_mode if config.PUBLIC_URL else bot.main
    threading.Thread(target=target, name="bot", daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")), log_level="warning")


if __name__ == "__main__":
    main()
