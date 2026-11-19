"""The HTTP surface that a host talks to. No database or network needed."""
from fastapi.testclient import TestClient

from canapply import telegram as tg
from canapply.api import app

client = TestClient(app)


def test_ping_needs_no_database():
    assert client.get("/ping").json() == {"ok": True}


def test_webhook_rejects_a_missing_or_wrong_secret():
    assert client.post("/telegram/webhook", json={"update_id": 1}).status_code == 403
    r = client.post("/telegram/webhook", json={"update_id": 1}, headers={"X-Telegram-Bot-Api-Secret-Token": "nope"})
    assert r.status_code == 403


def test_webhook_accepts_the_right_secret_and_hands_the_update_to_the_bot(monkeypatch):
    seen = []
    import canapply.bot as bot
    monkeypatch.setattr(bot, "dispatch", seen.append)
    r = client.post("/telegram/webhook", json={"update_id": 7},
                    headers={"X-Telegram-Bot-Api-Secret-Token": tg.webhook_secret()})
    assert r.status_code == 200
    from canapply.api import _workers
    _workers.submit(lambda: None).result(timeout=5)      # let the worker thread drain
    assert seen == [{"update_id": 7}]
