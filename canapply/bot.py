"""CanApply on Telegram: long polling, a daily digest, feedback buttons, and a labelling mode for evals.

Run with:  python -m canapply.bot
"""
import datetime as dt
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

from psycopg.types.json import Jsonb

from . import agent, config, db, gap, llm
from . import profile as P
from . import search as S
from . import telegram as tg
from .eligibility import judge
from .requirements import Requirements

IST = ZoneInfo("Asia/Kolkata")
DIGEST_HOUR = 9                 # 09:00 IST
DIGEST_SIZE = 5
BUCKET_LABEL = {"can_apply": "✅ Can apply", "stretch": "🟡 Stretch", "cannot": "⛔ Can't apply"}
_locks: dict[int, threading.Lock] = {}

WELCOME = (
    "Hi, I'm CanApply.\n\n"
    "Most job search shows you jobs that look right but that you can't actually get: \"remote\" that is "
    "US-only, or \"3+ years\" hidden in the text. I read every posting's requirements and sort jobs into "
    "✅ Can apply, 🟡 Stretch (you miss exactly one thing) and ⛔ Can't.\n\n"
    "To start, send me your resume as a PDF. Or just tell me about yourself in one message: where you live, "
    "whether you're a student and when you graduate, your experience, and the roles you want."
)
HELP = (
    "Just talk to me, for example:\n"
    "• remote AI internships I can apply to\n"
    "• backend new grad roles in India\n"
    "• why can't I apply to #123?\n"
    "• what should I learn next?\n"
    "• only show me remote roles from now on\n\n"
    "Commands:\n/jobs - today's picks\n/gap - what blocks you most, and skills to learn\n"
    "/profile - what I know about you\n/pause and /resume - daily picks off or on\n/help - this message"
)


def today() -> str:
    return dt.datetime.now(IST).strftime("%A %d %B %Y")


def log(chat_id: int | None, kind: str, payload: dict | None = None) -> None:
    db.ex("insert into events (chat_id, kind, payload) values (%s,%s,%s)", (chat_id, kind, Jsonb(payload or {})))


# ---- job cards ---------------------------------------------------------------------------------------------
def card_text(h: S.Hit) -> str:
    j, v = h.job, h.verdict
    lines = [f"<b>{tg.esc(j['title'])}</b>", f"{tg.esc(j['company'])} · {tg.esc(j.get('location_raw') or 'location not given')}"]
    if v:
        line = BUCKET_LABEL[v.bucket]
        if v.blockers:
            line += f": {tg.esc(v.blockers[0].reason)}"
        lines.append(line)
        quote = next((c.quote for c in v.blockers if c.quote and c.quote != j.get("location_raw")), None)
        if quote:
            lines.append(f"<i>“{tg.esc(' '.join(quote.split())[:220])}”</i>")
        if v.bucket != "cannot" and v.unverified:
            lines.append(f"Not stated: {tg.esc(v.unverified[0].reason[0].lower() + v.unverified[0].reason[1:])}")
    if h.why:
        lines.append(f"Fit: {tg.esc(h.why)}")
    if j.get("req_method") == "rules":
        lines.append("<i>(quick check only, not a full reading)</i>")
    return "\n".join(lines)


def card_buttons(h: S.Hit, state: str | None = None) -> list[list[tuple[str, str]]]:
    jid = h.id if isinstance(h, S.Hit) else h
    marks = {"up": "👍 Saved", "down": "👎 Hidden", "applied": "✔️ Applied"}
    row1 = [("👍", f"fb:up:{jid}"), ("👎", f"fb:down:{jid}"), ("Applied", f"fb:applied:{jid}"), ("Why?", f"why:{jid}")]
    if state in marks:
        row1 = [(marks[state], f"fb:{state}:{jid}"), ("Why?", f"why:{jid}")]
    url = h.job["url"] if isinstance(h, S.Hit) else None
    return [row1] + ([[("Open posting ↗", f"url:{url}")]] if url else [])


def send_cards(chat_id: int, hits: list[S.Hit]) -> None:
    for h in hits:
        tg.send(chat_id, card_text(h), buttons=card_buttons(h), html_mode=True)
        db.ex("insert into sent_jobs (chat_id, job_id) values (%s,%s) on conflict do nothing", (chat_id, h.id))


# ---- flows -------------------------------------------------------------------------------------------------
def onboard_from_text(chat_id: int, text: str, name: str | None, source: str) -> None:
    tg.typing(chat_id)
    try:
        prof, found_name = P.profile_from_resume(text, today())
    except llm.LLMError:
        tg.send(chat_id, "I couldn't read that just now (the AI service is busy). Please send it again in a minute.")
        return
    P.save_profile(chat_id, prof, resume_text=text, name=found_name or name)
    log(chat_id, "onboarded", {"source": source})
    tg.send(chat_id, "Got it. Here's what I understood:\n\n" + P.describe(prof)
            + "\n\nIf anything is wrong, just tell me (for example: \"I have a UK work visa\" or "
            "\"I have 2 years of experience\"). Finding your first matches now…")
    digest_for(chat_id, announce_empty=True)
    tg.send(chat_id, "Ask me anything, or tap /help to see what I can do. I'll send new picks every morning.")


def digest_query(user: dict) -> str:
    prof = P.user_profile(user)
    prefs = user.get("prefs") or {}
    return str(prefs.get("target_roles") or prof.interests or "software engineer")


def digest_for(chat_id: int, announce_empty: bool = False) -> int:
    user = P.get_user(chat_id)
    if not P.has_profile(user):
        return 0
    prof = P.user_profile(user)
    seen = [r["job_id"] for r in db.q("select job_id from sent_jobs where chat_id=%s", (chat_id,))]
    seen += [r["job_id"] for r in db.q("select job_id from feedback where chat_id=%s", (chat_id,))]
    filters = {"personal": True, "no_senior": prof.years_experience < 4, "exclude_ids": seen or None}
    if str((user.get("prefs") or {}).get("remote_only", "")).lower() in ("true", "yes", "1"):
        filters["remote"] = True
    hits = S.search(digest_query(user), profile=prof, user=user, mode="hybrid", eligible_only=True, do_rerank=True,
                    filters={k: v for k, v in filters.items() if v}, limit=DIGEST_SIZE, upgrade_top=6, patient=True)
    if hits:
        n_can = sum(h.verdict.bucket == "can_apply" for h in hits)
        tg.send(chat_id, f"Today's picks: {n_can} you can apply to" + (f", {len(hits) - n_can} stretch" if len(hits) > n_can else "")
                + ". 👍 / 👎 teaches me what you like.")
        send_cards(chat_id, hits)
    elif announce_empty:
        tg.send(chat_id, "I didn't find new jobs you can get right now. Try telling me which roles you want, or ask /gap to see what is blocking you.")
    log(chat_id, "digest_sent", {"jobs": [h.id for h in hits]})
    return len(hits)


def send_gap(chat_id: int) -> None:
    user = P.get_user(chat_id)
    g = gap.analyse(P.user_profile(user), user)
    b = g["buckets"]
    lines = [f"I checked {g['total']} open jobs close to “{g['query'][:80]}”.",
             f"✅ {b.get('can_apply', 0)} can apply · 🟡 {b.get('stretch', 0)} stretch · ⛔ {b.get('cannot', 0)} can't", ""]
    if g["blockers"]:
        lines.append("What blocks you most:")
        lines += [f"• {name}: {n} jobs" for name, n in g["blockers"]]
    if g["missing_skills"]:
        lines += ["", f"Skills the {g['reach']} jobs within reach ask for that your profile doesn't list:"]
        lines += [f"• {s} ({n} jobs)" for s, n in g["missing_skills"]]
    tg.send(chat_id, "\n".join(lines))
    log(chat_id, "gap", {"buckets": b})


def why(chat_id: int, job_id: int) -> None:
    row = db.q1("""select j.*, r.data as req, r.method as req_method from jobs j
                   left join requirements r on r.job_id = j.id where j.id=%s""", (job_id,))
    user = P.get_user(chat_id)
    if not row or not row.get("req") or not P.has_profile(user):
        tg.send(chat_id, "I don't have the details for that one.")
        return
    prof = P.user_profile(user)
    hit = S.Hit({**row, "snippet": ""}, 0.0, verdict=judge(Requirements.model_validate(row["req"]), prof))
    if row["req_method"] == "rules":
        S.upgrade([hit], prof, n=1)
    icon = {"pass": "✅", "soft": "🟡", "hard": "⛔", "unknown": "❔"}
    lines = [f"<b>{tg.esc(row['title'])}</b> at {tg.esc(row['company'])}", BUCKET_LABEL[hit.verdict.bucket], ""]
    for c in hit.verdict.checks:
        lines.append(f"{icon[c.status]} {tg.esc(c.reason)}")
        if c.quote and c.status in ("soft", "hard"):
            lines.append(f"    <i>“{tg.esc(' '.join(c.quote.split())[:240])}”</i>")
    skills = (hit.job.get("req") or {}).get("skills") or []
    if skills:
        lines += ["", "Must-have skills: " + tg.esc(", ".join(skills))]
    tg.send(chat_id, "\n".join(lines), html_mode=True)
    log(chat_id, "why", {"job": job_id})


# ---- labelling mode (owner only): builds the human-labelled eval set from the phone -------------------------
def next_label(chat_id: int) -> None:
    row = db.q1("""select j.id, j.title, j.company, j.location_raw, j.workplace_raw, j.url, left(j.description, 2600) d
                   from eval_pool e join jobs j on j.id = e.job_id
                   where not exists (select 1 from labels l where l.job_id = j.id and l.labeler = 'ansh')
                   order by e.ord limit 1""")
    done = db.q1("select count(*) n from labels where labeler='ansh'")["n"]
    total = db.q1("select count(*) n from eval_pool")["n"]
    if not row:
        tg.send(chat_id, f"All {total} labelled. Thank you! Run the evals to see how the system did.")
        return
    text = (f"<b>Label {done + 1} of {total}</b>\n\n<b>{tg.esc(row['title'])}</b>\n{tg.esc(row['company'])} · "
            f"{tg.esc(row['location_raw'] or 'no location')} · {tg.esc(row['workplace_raw'] or '')}\n\n"
            f"{tg.esc(row['d'])}…\n\nCould YOU realistically apply to this and be considered?")
    tg.send(chat_id, text, html_mode=True, buttons=[
        [("✅ Can apply", f"lab:can_apply:{row['id']}"), ("🟡 Stretch", f"lab:stretch:{row['id']}"),
         ("⛔ Can't", f"lab:cannot:{row['id']}")],
        [("Skip", f"lab:skip:{row['id']}"), ("Open posting ↗", f"url:{row['url']}")]])


# ---- update handlers ---------------------------------------------------------------------------------------
def handle_message(msg: dict) -> None:
    chat_id = msg["chat"]["id"]
    if msg["chat"].get("type") != "private":
        return
    name = (msg.get("from") or {}).get("first_name")
    user = P.ensure_user(chat_id, name)
    text = (msg.get("text") or "").strip()
    doc = msg.get("document")

    if doc:
        if (doc.get("mime_type") or "") != "application/pdf":
            tg.send(chat_id, "Please send the resume as a PDF.")
            return
        tg.typing(chat_id)
        try:
            resume = P.pdf_to_text(tg.download(doc["file_id"]))
        except Exception:  # noqa: BLE001
            resume = ""
        if len(resume) < 200:
            tg.send(chat_id, "I couldn't read text from that PDF. Is it a scan? You can also just describe yourself in a message.")
            return
        onboard_from_text(chat_id, resume, name, "pdf")
        return
    if not text:
        return

    cmd = text.split()[0].split("@")[0].lower() if text.startswith("/") else None
    if cmd == "/start":
        log(chat_id, "start")
        tg.send(chat_id, WELCOME if not P.has_profile(user) else "Welcome back. " + HELP)
    elif cmd == "/help":
        tg.send(chat_id, HELP)
    elif not P.has_profile(user):
        if cmd or len(text) < 25:
            tg.send(chat_id, WELCOME)
        else:
            onboard_from_text(chat_id, text, name, "text")
    elif cmd == "/profile":
        p = P.user_profile(user)
        tg.send(chat_id, P.describe(p) + f"\nInterests: {p.interests}\nSkills: {', '.join(p.skills[:20])}\n"
                f"Preferences: {user.get('prefs') or 'none yet'}\n\nTell me if anything is wrong, or send a new resume PDF.")
    elif cmd == "/jobs":
        tg.typing(chat_id)
        digest_for(chat_id, announce_empty=True)
    elif cmd == "/gap":
        tg.typing(chat_id)
        send_gap(chat_id)
    elif cmd in ("/pause", "/stop"):
        db.ex("update users set digest_on=false where chat_id=%s", (chat_id,))
        tg.send(chat_id, "Daily picks paused. /resume turns them back on.")
    elif cmd == "/resume":
        db.ex("update users set digest_on=true where chat_id=%s", (chat_id,))
        tg.send(chat_id, "Daily picks are on. You'll hear from me around 9 in the morning.")
    elif cmd == "/label" and chat_id == config.OWNER_CHAT_ID:
        next_label(chat_id)
    elif cmd == "/stats" and chat_id == config.OWNER_CHAT_ID:
        s = db.q1("""select (select count(*) from users) users,
                            (select count(*) from users where profile ? 'country') onboarded,
                            (select count(*) from events where kind='agent_turn' and at > now() - interval '7 days') turns,
                            (select count(*) from feedback) feedback,
                            (select count(*) from jobs where closed_at is null and is_tech) jobs,
                            (select count(*) from requirements where method='llm') read""")
        tg.send(chat_id, f"Users {s['users']} (onboarded {s['onboarded']}) · chat turns last 7 days {s['turns']} · "
                f"feedback taps {s['feedback']}\nOpen tech jobs {s['jobs']} · postings fully read {s['read']}")
    elif cmd:
        tg.send(chat_id, "I don't know that command. " + HELP)
    else:
        tg.typing(chat_id)
        try:
            out = agent.run_turn(chat_id, text, today())
        except llm.LLMError as e:
            log(chat_id, "agent_error", {"error": str(e)[:300]})
            tg.send(chat_id, "The AI service is busy right now. Please try again in a minute. /jobs still works.")
            return
        tg.send(chat_id, out["text"] or "Here's what I found.")
        send_cards(chat_id, out["hits"])


def handle_callback(cb: dict) -> None:
    chat_id = cb["message"]["chat"]["id"]
    data = cb.get("data") or ""
    parts = data.split(":")
    if parts[0] == "fb" and len(parts) == 3:
        action, job_id = parts[1], int(parts[2])
        existing = db.q1("select 1 from feedback where chat_id=%s and job_id=%s and action=%s", (chat_id, job_id, action))
        if existing:                                    # tapping again undoes it
            db.ex("delete from feedback where chat_id=%s and job_id=%s and action=%s", (chat_id, job_id, action))
            tg.answer_callback(cb["id"], "Undone")
            state = None
        else:
            db.ex("delete from feedback where chat_id=%s and job_id=%s", (chat_id, job_id))
            db.ex("insert into feedback (chat_id, job_id, action) values (%s,%s,%s)", (chat_id, job_id, action))
            tg.answer_callback(cb["id"], {"up": "Saved. I'll find more like this.", "down": "Hidden. Fewer like this.",
                                          "applied": "Marked as applied. Good luck!"}[action])
            state = action
        url = (db.q1("select url from jobs where id=%s", (job_id,)) or {}).get("url")
        rows = card_buttons(job_id, state)
        if url:
            rows.append([("Open posting ↗", f"url:{url}")])
        tg.edit_buttons(chat_id, cb["message"]["message_id"], rows)
        log(chat_id, "button", {"action": action, "job": job_id, "undo": bool(existing)})
    elif parts[0] == "why" and len(parts) == 2:
        tg.answer_callback(cb["id"])
        why(chat_id, int(parts[1]))
    elif parts[0] == "lab" and len(parts) == 3 and chat_id == config.OWNER_CHAT_ID:
        bucket, job_id = parts[1], int(parts[2])
        if bucket == "skip":
            db.ex("update eval_pool set ord = ord + 100000 where job_id=%s", (job_id,))
        else:
            db.ex("""insert into labels (job_id, labeler, bucket) values (%s,'ansh',%s)
                     on conflict (job_id, labeler) do update set bucket=excluded.bucket, at=now()""", (job_id, bucket))
        tg.answer_callback(cb["id"], "Saved")
        tg.edit_buttons(chat_id, cb["message"]["message_id"], [[(f"Labelled: {bucket}", "noop")]])
        next_label(chat_id)
    else:
        tg.answer_callback(cb["id"])


def dispatch(update: dict) -> None:
    chat_id = ((update.get("message") or {}).get("chat") or {}).get("id") or \
              (((update.get("callback_query") or {}).get("message") or {}).get("chat") or {}).get("id")
    lock = _locks.setdefault(chat_id or 0, threading.Lock())
    with lock:                                          # one thing at a time per chat, chats in parallel
        try:
            if "message" in update:
                handle_message(update["message"])
            elif "callback_query" in update:
                handle_callback(update["callback_query"])
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            log(chat_id, "error", {"trace": traceback.format_exc()[-800:]})
            if chat_id:
                try:
                    tg.send(chat_id, "Something went wrong on my side. Please try again.")
                except Exception:  # noqa: BLE001
                    pass


def run_digests_if_due() -> None:
    now = dt.datetime.now(IST)
    if now.hour != DIGEST_HOUR:
        return
    day = now.strftime("%Y-%m-%d")
    if db.q1("select 1 from events where kind='digest_run' and payload->>'day' = %s", (day,)):
        return
    log(None, "digest_run", {"day": day})
    for u in db.q("select chat_id from users where digest_on and profile ? 'country'"):
        try:
            digest_for(u["chat_id"])
        except Exception:  # noqa: BLE001
            traceback.print_exc()


def main() -> None:
    db.init_schema()
    print("CanApply bot running. Ctrl+C to stop.", flush=True)
    offset = None
    pool = ThreadPoolExecutor(max_workers=4)
    last_tick = 0.0
    while True:
        try:
            for upd in tg.get_updates(offset, timeout=40):
                offset = upd["update_id"] + 1
                pool.submit(dispatch, upd)
        except KeyboardInterrupt:
            break
        except Exception as e:  # noqa: BLE001  - network blips must not kill the bot
            print("poll error:", str(e)[:200], flush=True)
            time.sleep(5)
        if time.time() - last_tick > 300:
            last_tick = time.time()
            pool.submit(run_digests_if_due)


if __name__ == "__main__":
    main()
