"""Small HTTP surface: a health check for the host, and a status page that shows the index is alive."""
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from . import db
from . import search as S
from . import telegram as tg
from .eligibility import Profile

app = FastAPI(title="CanApply", docs_url="/docs")
_workers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="update")


HOME = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CanApply</title>
<style>
:root{--bg:#f7f7f4;--fg:#1b1d1f;--mut:#5d6368;--line:#dcdcd6;--acc:#1f5fae;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#131517;--fg:#eceee9;--mut:#9aa1a6;--line:#2b2f33;--acc:#7fb2f0;--card:#1b1e21}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:680px;margin:0 auto;padding:56px 16px 48px}h1{font-size:34px;margin:0 0 6px;letter-spacing:-.02em}
.lead{font-size:19px;margin:0 0 18px}p{color:var(--mut);margin:0 0 14px}
.row{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0 30px}
a.btn{display:inline-block;padding:10px 16px;border-radius:8px;border:1px solid var(--line);color:var(--fg);text-decoration:none;font-weight:600}
a.btn.primary{background:var(--acc);border-color:var(--acc);color:#fff}
@media (prefers-color-scheme:dark){a.btn.primary{color:#0e1114}}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.n{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums}.l{font-size:13px;color:var(--mut)}
small{color:var(--mut)}small a{color:var(--acc)}
</style></head><body><main>
<h1>CanApply</h1>
<p class="lead">A job search that tells you which jobs you can actually get.</p>
<p>It reads the hard requirements out of every posting (years of experience, work authorisation, where you must
live), checks them against your profile in plain code, and sorts jobs into can apply, stretch and can't, with the
sentence from the posting that decided it. It runs as a Telegram bot.</p>
<div class="row">
<a class="btn primary" href="https://t.me/CanApplyJobsBot">Open the bot on Telegram</a>
<a class="btn" href="https://github.com/anshhoff/canapply">Code and evals on GitHub</a>
</div>
<div class="grid">
<div class="card"><div class="n" id="n-open">…</div><div class="l">open postings</div></div>
<div class="card"><div class="n" id="n-tech">…</div><div class="l">tech roles, searchable</div></div>
<div class="card"><div class="n" id="n-boards">…</div><div class="l">company career pages</div></div>
<div class="card"><div class="n" id="n-read">…</div><div class="l">full LLM readings</div></div>
</div>
<small id="fresh">Loading the live index…</small>
<p style="margin-top:26px"><small>Raw numbers: <a href="/status">/status</a> · API: <a href="/docs">/docs</a>.
The first message to the bot after a quiet spell can take up to a minute while the free instance wakes.</small></p>
</main><script>
const el=id=>document.getElementById(id),f=n=>Number(n||0).toLocaleString('en-IN');
fetch('/status').then(r=>r.json()).then(d=>{
el('n-open').textContent=f(d.index.open_jobs);el('n-tech').textContent=f(d.index.open_tech_jobs);
el('n-boards').textContent=f(d.sources.reduce((a,s)=>a+Number(s.boards||0),0));
el('n-read').textContent=f(d.requirements.full_readings);
el('fresh').textContent='Index last refreshed '+new Date(d.index.last_refresh).toLocaleString();})
.catch(()=>{el('fresh').textContent='Live numbers are unavailable right now.'});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    """A plain landing page. Static: the live numbers are fetched from /status by the page itself."""
    return HOME


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
