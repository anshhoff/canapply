"""Small HTTP surface: a health check for the host, and a status page that shows the index is alive."""
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Header, HTTPException, Query, Request

from . import db
from . import search as S
from . import telegram as tg
from .eligibility import Profile

app = FastAPI(title="CanApply", docs_url="/docs")
_workers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="update")


@app.get("/ping")
def ping() -> dict:
    """Liveness only. Deliberately does not touch the database, so keep-alive pings let a serverless database sleep."""
    return {"ok": True}


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request, x_telegram_bot_api_secret_token: str = Header(default="")) -> dict:
    """Webhook mode: Telegram posts each update here. Answer at once and do the work on a worker thread."""
    if x_telegram_bot_api_secret_token != tg.webhook_secret():
        raise HTTPException(status_code=403, detail="bad secret")
    update = await request.json()
    # Imported late: the bot pulls in the agent and search stack.
    from . import bot

    _workers.submit(bot.dispatch, update)
    return {"ok": True}


@app.get("/health")
def health() -> dict:
    db.q1("select 1 as ok")
    return {"ok": True}


@app.get("/status")
def status() -> dict:
    """Is the index fresh, and is every source still answering?"""
    idx = db.q1("""select count(*) filter (where closed_at is null) open_jobs,
                          count(*) filter (where closed_at is null and is_tech) open_tech_jobs,
                          count(*) filter (where first_seen_at > now() - interval '24 hours') added_24h,
                          count(*) filter (where closed_at > now() - interval '24 hours') closed_24h,
                          max(last_seen_at) last_refresh from jobs""")
    src = db.q("""select source, count(*) filter (where enabled) boards,
                         count(*) filter (where enabled and last_status = 'ok') healthy,
                         count(*) filter (where enabled and fail_streak >= 3) failing,
                         max(last_ok_at) last_ok, sum(last_job_count) filter (where enabled) jobs
                  from boards group by 1 order by 1""")
    read = db.q1("""select count(*) filter (where method='llm') full_readings,
                           count(*) filter (where method='rules') rules_only,
                           coalesce(sum(quotes_total),0) quotes_checked, coalesce(sum(quotes_ok),0) quotes_passed
                    from requirements""")
    return {"index": idx, "sources": src, "requirements": read}


@app.get("/search")
def search(q: str = Query(min_length=2, max_length=200), country: str = "IN", years: float = 0,
           student: bool = True, mode: str = "hybrid", eligible_only: bool = True, limit: int = Query(10, le=25)) -> dict:
    """Debug endpoint: the same search the bot uses, without the LLM re-rank, for any simple profile."""
    profile = Profile(country=country.upper()[:2], work_auth=[country.upper()[:2]], years_experience=years,
                      is_student=student)
    hits = S.search(q, profile=profile, mode=mode if mode in ("keyword", "vector", "hybrid") else "hybrid",
                    eligible_only=eligible_only, limit=limit)
    return {"query": q, "results": [{
        "id": h.id, "title": h.job["title"], "company": h.job["company"], "location": h.job.get("location_raw"),
        "url": h.job["url"], "verdict": h.verdict.bucket if h.verdict else None,
        "reason": h.verdict.headline() if h.verdict else None,
        "reading": h.job.get("req_method"), "keyword_rank": h.kw_rank, "vector_rank": h.vec_rank} for h in hits]}
