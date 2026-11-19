"""One process for a small host: the Telegram bot (long polling) in a thread, the HTTP status API in front."""
import os
import threading

import uvicorn

from . import bot
from .api import app


def main() -> None:
    threading.Thread(target=bot.main, name="bot", daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")), log_level="warning")


if __name__ == "__main__":
    main()
