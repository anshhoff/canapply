"""Minimal Gemini client over plain HTTPS: JSON-schema output, tool calls, embeddings, retries, usage log.

No SDK on purpose: every request and response here is something I can explain line by line.
"""
import json
import random
import re
import threading
import time

import httpx

from . import config

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_client = httpx.Client(timeout=120)
INTERACTIVE = {"agent", "rerank", "profile", "embed_query"}   # purposes where a person is waiting
_lock = threading.Lock()
_last_call: dict[str, float] = {}
_local = threading.local()


def last_model() -> str | None:
    """Which model in the fallback chain actually answered the most recent call on this thread."""
    return getattr(_local, "model", None)
# Free-tier pacing: the Gemini free tier allows 15 requests a minute per model, so we space calls ~4.3 s apart
# per model instead of bursting and eating 429s. Raised automatically if a 429 still arrives.
DEFAULT_GAP = 4.3
MIN_GAP: dict[str, float] = {}


class LLMError(Exception):
    pass


class QuotaExhausted(LLMError):
    """The daily free quota for this model is used up. Callers fall back or stop."""


def _log(purpose, model, usage, ms, ok=True):
    try:
        from . import db
        db.ex("insert into llm_usage (purpose, model, prompt_tokens, output_tokens, ms, ok) values (%s,%s,%s,%s,%s,%s)",
              (purpose, model, (usage or {}).get("promptTokenCount"), (usage or {}).get("candidatesTokenCount"),
               int(ms), ok))
    except Exception:  # noqa: BLE001  - logging must never break a request
        pass


def _post(model: str, method: str, body: dict, purpose: str, tries: int = 6) -> dict:
    """`model` may be a comma-separated fallback chain, e.g. "gemini-3.8-flash,gemini-3.5-flash-lite".
    A model that is overloaded (503) or out of quota is skipped and the next one is tried."""
    chain = [m.strip() for m in model.split(",") if m.strip()]
    last: Exception | None = None
    live = purpose in INTERACTIVE
    for i, m in enumerate(chain):
        final = i == len(chain) - 1
        try:
            # When a person is waiting: one quick attempt per model, 18 s cap, then move on.
            return _post_one(m, method, body, purpose,
                             tries=(2 if final else 1) if live else (tries if final else 2),
                             timeout=18 if live else 120)
        except (QuotaExhausted, Overloaded) as e:
            last = e
            continue
    raise last or LLMError("no model configured")


class Overloaded(LLMError):
    """The model kept answering 5xx. Callers with a fallback chain move on to the next model."""


def _post_one(model: str, method: str, body: dict, purpose: str, tries: int = 6, timeout: float = 120) -> dict:
    url = f"{BASE}/{model}:{method}"
    delay = 2.0
    for attempt in range(tries):
        gap = MIN_GAP.get(model, DEFAULT_GAP)
        with _lock:                                  # reserve the next free slot for this model
            slot = max(time.time(), _last_call.get(model, 0) + gap)
            _last_call[model] = slot
        if slot > time.time():
            time.sleep(slot - time.time())
        t0 = time.time()
        try:
            r = _client.post(url, json=body, headers={"x-goog-api-key": config.GEMINI_API_KEY}, timeout=timeout)
        except httpx.HTTPError as e:
            _log(purpose, model, None, (time.time() - t0) * 1000, ok=False)
            if attempt == tries - 1:
                raise Overloaded(f"network: {type(e).__name__}") from e
            time.sleep(delay); delay = min(delay * 2, 60)
            continue
        ms = (time.time() - t0) * 1000
        if r.status_code == 200:
            d = r.json()
            _log(purpose, model, d.get("usageMetadata"), ms)
            _local.model = model
            return d
        msg = r.text[:900]
        if r.status_code == 429:
            _log(purpose, model, None, ms, ok=False)
            full = r.text
            if re.search(r"PerDay|per day|daily", full, re.I) or "limit: 0" in full:
                raise QuotaExhausted(f"{model}: daily quota used up")
            msg = full[:3000]
            m = re.search(r'retryDelay"?:\s*"?(\d+(?:\.\d+)?)s', msg) or re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
            pause = float(m.group(1)) + 1 if m else delay
            MIN_GAP[model] = min(max(MIN_GAP.get(model, DEFAULT_GAP) * 1.25, 1.0), 20.0)   # slow down from here on
            if attempt == tries - 1:
                raise QuotaExhausted(f"{model}: still rate limited after {tries} tries")
            time.sleep(min(pause, 75) + random.random()); delay = min(delay * 2, 60)
            continue
        if r.status_code >= 500:
            _log(purpose, model, None, ms, ok=False)
            if attempt == tries - 1:
                raise Overloaded(f"{model} {r.status_code}: {msg[:200]}")
            time.sleep(delay + random.random()); delay = min(delay * 2, 30)
            continue
        _log(purpose, model, None, ms, ok=False)
        raise LLMError(f"{model} {r.status_code}: {msg}")
    raise LLMError("unreachable")


def _text(d: dict) -> str:
    try:
        parts = d["candidates"][0]["content"].get("parts") or []
    except (KeyError, IndexError) as e:
        raise LLMError(f"no candidates: {json.dumps(d)[:300]}") from e
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def generate_json(prompt: str, schema: dict, *, model: str, purpose: str, system: str | None = None,
                  temperature: float = 0.0, max_tokens: int = 2048, thinking: str | None = None) -> dict:
    """One call, answer constrained to a JSON schema. `thinking` = minimal | low | medium | high (None = default)."""
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens,
                             "responseMimeType": "application/json", "responseJsonSchema": schema},
    }
    if thinking:
        body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": thinking}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    raw = ""
    for attempt in range(2):
        d = _post(model, "generateContent", body, purpose)
        raw = _text(d)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Thinking tokens count against the output limit, so a cut-off answer gets one retry with more room.
            if attempt == 0 and (d.get("candidates") or [{}])[0].get("finishReason") == "MAX_TOKENS":
                body["generationConfig"]["maxOutputTokens"] = max_tokens * 3
                continue
            break
    raise LLMError(f"model returned invalid JSON: {raw[:200]}")


def generate_text(prompt: str, *, model: str, purpose: str, system: str | None = None,
                  temperature: float = 0.3, max_tokens: int = 1024) -> str:
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return _text(_post(model, "generateContent", body, purpose))


def chat_with_tools(contents: list[dict], tools: list[dict], *, model: str, purpose: str, system: str,
                    temperature: float = 0.2, max_tokens: int = 2048) -> dict:
    """One step of a tool-using conversation. Returns the model's raw `content` (parts kept verbatim,
    because newer Gemini models attach signatures to tool calls that must be sent back unchanged)."""
    body = {"contents": contents, "systemInstruction": {"parts": [{"text": system}]},
            "tools": [{"functionDeclarations": tools}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
    d = _post(model, "generateContent", body, purpose)
    try:
        content = d["candidates"][0].get("content") or {"role": "model", "parts": []}
    except (KeyError, IndexError) as e:
        raise LLMError(f"no candidates: {json.dumps(d)[:300]}") from e
    content.setdefault("role", "model"); content.setdefault("parts", [])
    return content
