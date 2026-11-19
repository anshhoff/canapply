"""Settings read from the environment (.env locally, real env vars in production)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql:///canapply")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OWNER_CHAT_ID = int(os.getenv("OWNER_CHAT_ID", "0") or 0)
# Set on a host (Render sets RENDER_EXTERNAL_URL itself): Telegram then pushes updates to us instead of being polled.
PUBLIC_URL = (os.getenv("PUBLIC_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").rstrip("/")
QUIET = os.getenv("QUIET", "") == "1"      # send Telegram messages without a notification sound (testing at night)

# Models, each a fallback chain (first that answers wins). Free-tier models are often overloaded, so every
# job has at least two options. On the free tier only the 'lite' models are usable for a live bot: the
# larger flash models allow 20 requests a DAY. With a paid key, set AGENT_MODEL to a stronger model.
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "gemini-3.5-flash-lite,gemini-3.1-flash-lite")
# The overnight bulk run uses its own models so it does not eat the per-minute allowance the live bot needs.
BULK_EXTRACT_MODELS = os.getenv("BULK_EXTRACT_MODELS", "gemini-3.1-flash-lite,gemini-3.1-flash-lite-preview")
AGENT_MODEL = os.getenv("AGENT_MODEL", "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-flash-lite-latest")
RERANK_MODEL = os.getenv("RERANK_MODEL", "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-flash-lite-latest")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite,gemini-3.1-flash-lite")

USER_AGENT = "canapply-indexer/0.1 (job index for students; contact via github.com/anshhoff)"
