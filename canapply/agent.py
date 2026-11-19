"""The chat agent: a hand-written tool loop. No framework.

  1. Send the conversation + tool descriptions to the model.
  2. If the model asks for tools, run them and send the results back.
  3. Repeat until the model answers in plain text, or MAX_STEPS is reached.

The model never sees the whole database and never decides eligibility itself. It can only ask the tools,
and the tools answer from stored postings and the eligibility code. That is what keeps it from inventing jobs.
"""
import json
import re

from psycopg.types.json import Jsonb

from . import config, db, gap, llm
from . import profile as P
from . import search as S
from .eligibility import Profile, judge
from .requirements import Requirements

MAX_STEPS = 6
HISTORY = 10          # past messages (user + assistant text) carried into each turn

TOOLS = [
    {"name": "search_jobs",
     "description": "Search open job postings. Results are already checked against the user's profile. "
                    "Use short, role-focused queries like 'machine learning intern' or 'backend python new grad'.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "What kind of job. Role words and skills, not a sentence."},
         "show": {"type": "string", "enum": ["can_apply", "can_apply_and_stretch", "all"],
                  "description": "can_apply_and_stretch is the default. 'all' also returns jobs the user cannot get."},
         "remote_only": {"type": "boolean"},
         "country": {"type": "string", "description": "ISO-2 code to only show jobs based in that country, e.g. IN."},
         "levels": {"type": "array", "items": {"type": "string", "enum": ["intern", "new_grad", "junior", "unknown"]},
                    "description": "Only these levels (from the job title)."},
         "company": {"type": "string"},
         "new_in_days": {"type": "integer", "description": "Only postings first seen in the last N days."},
         "limit": {"type": "integer", "description": "How many to return, 1-8. Default 5."}},
         "required": ["query"]}},
    {"name": "explain_job",
     "description": "Full eligibility check for one job: every requirement found, the exact sentence it came from, "
                    "and the verdict for this user. Use when the user asks why, or about a specific job id.",
     "parameters": {"type": "object", "properties": {"job_id": {"type": "integer"}}, "required": ["job_id"]}},
    {"name": "skills_gap",
     "description": "What blocks this user most often and which skills the jobs within reach ask for that the "
                    "user does not list. Use for 'what should I learn', 'why am I not getting matches'.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Optional role focus, e.g. 'ai engineer'."}}}},
    {"name": "remember",
     "description": "Save a lasting preference so future searches and daily digests respect it "
                    "(e.g. roles wanted, 'remote only', companies to avoid, minimum stipend).",
     "parameters": {"type": "object", "properties": {
         "key": {"type": "string", "description": "Short snake_case name, e.g. remote_only, target_roles, avoid_companies."},
         "value": {"type": "string"}}, "required": ["key", "value"]}},
    {"name": "update_profile",
     "description": "Correct a fact in the user's profile when they tell you it is wrong or has changed.",
     "parameters": {"type": "object", "properties": {
         "country": {"type": "string", "description": "ISO-2 country they live in."},
         "work_auth": {"type": "array", "items": {"type": "string"},
                       "description": "ISO-2 countries where they already have the right to work."},
         "open_to_relocate": {"type": "boolean"},
         "years_experience": {"type": "number"},
         "is_student": {"type": "boolean"}, "grad_year": {"type": "integer"},
         "degree": {"type": "string", "enum": ["none", "bachelors", "masters", "phd"]},
         "wants": {"type": "array", "items": {"type": "string", "enum": ["internship", "full_time", "part_time", "contract"]}},
         "interests": {"type": "string"}}}},
]

SYSTEM = """You are CanApply, a job-search assistant on Telegram for students and early-career people.
Your one job: help the user find jobs they can ACTUALLY get, and be honest about the ones they can't.

How you work:
- You know nothing about jobs except what the tools return. Never invent a job, company, requirement or link.
- Eligibility is decided by the tools, not by you. Report their verdicts and reasons; do not overrule them.
- When you mention a job, write its id as #123 right after the title. The app turns each #id into a card with
  the link and buttons, so do NOT paste links or long descriptions yourself.
- Keep replies short and plain: a sentence or two, then the jobs as a short list (title #id - one-line reason).
  No markdown, no asterisks, no headings.
- One search is usually enough. Leave filters off unless the user asked for them: every filter you add hides jobs.
  If a search comes back empty, retry ONCE with fewer filters, then answer with what you have.
- If nothing fits, say so plainly and offer the closest 'stretch' options. Do not pad with jobs marked 'cannot'
  unless asked.
- Only give reasons the tools gave you. Do not add your own guesses about why a job does or does not fit.
- If the user states a lasting preference, call remember. If they correct a fact about themselves, call update_profile.
- If the profile looks wrong for what they ask (e.g. they say they have a US visa), fix it first, then search.

User profile:
{profile}

Saved preferences: {prefs}
Today: {today}"""


def card(h: S.Hit) -> dict:
    """Compact, model-facing view of one search hit."""
    v = h.verdict
    out = {"id": h.id, "title": h.job["title"], "company": h.job["company"],
           "location": h.job.get("location_raw"), "level": h.job.get("level_rule"),
           "verdict": v.bucket if v else "unknown"}
    if v:
        if v.blockers:
            out["blockers"] = [c.reason for c in v.blockers]
        if v.unverified:
            out["not_stated"] = [c.reason for c in v.unverified]
    if h.why:
        out["fit"] = h.why
    if h.job.get("req_method") == "rules":
        out["note"] = "quick rules-based check only"
    return out


class Session:
    """Per-turn state: the user row, their profile, and every job the tools returned (so the bot can show cards)."""

    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self.user = P.ensure_user(chat_id)
        self.profile = P.user_profile(self.user)
        self.hits: dict[int, S.Hit] = {}
        self.tool_log: list[dict] = []

    # ---- tools -------------------------------------------------------------
    def search_jobs(self, query: str, show: str = "can_apply_and_stretch", remote_only: bool = False,
                    country: str | None = None, levels: list[str] | None = None, company: str | None = None,
                    new_in_days: int | None = None, limit: int = 5) -> dict:
        prefs = self.user.get("prefs") or {}
        limit = max(1, min(int(limit or 5), 8))
        # An explicit search is not personalised at retrieval time: the user asked for something specific and a
        # resume-weighted query vector drowned it out ("ios developer" returned backend jobs). The resume still
        # reaches the re-ranker. Daily picks, which have no query, are where personalisation is used.
        filters = {"no_senior": self.profile.years_experience < 4}
        if remote_only or str(prefs.get("remote_only", "")).lower() in ("true", "yes", "1"):
            filters["remote"] = True
        if country:
            filters["country"] = country.upper()[:2]
        if levels:
            filters["levels"] = levels
        if company:
            filters["company"] = company
        if new_in_days:
            filters["since_days"] = int(new_in_days)
        hidden = [r["job_id"] for r in db.q("select job_id from feedback where chat_id=%s and action='down'", (self.chat_id,))]
        if hidden:
            filters["exclude_ids"] = hidden
        buckets = {"can_apply": ("can_apply",), "all": ("can_apply", "stretch", "cannot")}.get(
            show, ("can_apply", "stretch"))
        hits = S.search(query, profile=self.profile, user=self.user, mode="hybrid", eligible_only=True,
                        do_rerank=True, filters=filters, limit=limit, buckets=buckets, upgrade_top=6)
        considered = len(S.candidates(query, "keyword", self.user, filters)) if not hits else None
        for h in hits:
            self.hits[h.id] = h
        db.ex("insert into events (chat_id, kind, payload) values (%s,'search',%s)",
              (self.chat_id, Jsonb({"query": query, "filters": {k: v for k, v in filters.items() if k != "exclude_ids"},
                                    "show": show, "returned": [h.id for h in hits]})))
        out = {"results": [card(h) for h in hits], "count": len(hits)}
        if not hits and show == "can_apply":           # widen once inside the tool instead of costing a model step
            wider = S.search(query, profile=self.profile, user=self.user, mode="hybrid", eligible_only=True,
                             do_rerank=False, filters=filters, limit=limit, buckets=("stretch",), upgrade_top=0)
            for h in wider:
                self.hits[h.id] = h
            if wider:
                return {"results": [], "count": 0, "note": "Nothing the user can apply to outright with these filters. "
                        "Closest stretch options are in stretch_results.", "stretch_results": [card(h) for h in wider]}
        if not hits:
            out["note"] = (f"No matching jobs in the '{show}' bucket"
                           + (f" (keyword matches before the eligibility filter: {considered})" if considered else "")
                           + ". Try show='all' to see what blocks them, or a broader query.")
        return out

    def explain_job(self, job_id: int) -> dict:
        row = db.q1("""select j.*, r.data as req, r.method as req_method, r.quotes_total, r.quotes_ok
                       from jobs j left join requirements r on r.job_id = j.id where j.id = %s""", (job_id,))
        if not row:
            return {"error": f"No job with id {job_id}."}
        hit = S.Hit({**row, "snippet": row["description"][:1200]}, 0.0)
        if row.get("req"):
            hit.verdict = judge(Requirements.model_validate(row["req"]), self.profile)
            if row["req_method"] == "rules":
                S.upgrade([hit], self.profile, n=1)
        self.hits[job_id] = hit
        v = hit.verdict
        return {"id": job_id, "title": row["title"], "company": row["company"], "location": row["location_raw"],
                "open": row["closed_at"] is None, "verdict": v.bucket if v else "unknown",
                "checks": [{"what": c.name, "result": c.status, "reason": c.reason, "quote_from_posting": c.quote}
                           for c in (v.checks if v else [])],
                "must_have_skills": (hit.job.get("req") or {}).get("skills", []),
                "reading": "full reading, quotes verified against the posting" if hit.job.get("req_method") == "llm"
                           else "quick rules-based check",
                "posting_excerpt": " ".join(row["description"].split())[:900]}

    def skills_gap(self, query: str | None = None) -> dict:
        g = gap.analyse(self.profile, self.user, query)
        return {"role_focus": g["query"], "jobs_checked": g["total"], "buckets": g["buckets"],
                "top_blockers": [{"blocker": b, "jobs": n, "example": g["examples"].get(b)} for b, n in g["blockers"]],
                "skills_asked_in_reachable_jobs_you_do_not_list": [{"skill": s, "jobs": n} for s, n in g["missing_skills"]],
                "reachable_jobs": g["reach"]}

    def remember(self, key: str, value: str) -> dict:
        key = re.sub(r"[^a-z0-9_]", "_", key.lower())[:40]
        prefs = P.set_pref(self.chat_id, key, value[:300])
        self.user = P.get_user(self.chat_id)
        return {"saved": {key: value}, "all_preferences": prefs}

    def update_profile(self, **changes) -> dict:
        data = self.profile.model_dump()
        for k, v in changes.items():
            if k in data and v is not None:
                data[k] = [x.upper()[:2] for x in v] if k == "work_auth" else (v.upper()[:2] if k == "country" else v)
        if "country" in changes and "work_auth" not in changes and data["country"] not in data["work_auth"]:
            data["work_auth"].append(data["country"])
        self.profile = Profile.model_validate(data)
        P.save_profile(self.chat_id, self.profile)
        self.user = P.get_user(self.chat_id)
        return {"profile": self.profile.model_dump(exclude={"skills"})}

    def call(self, name: str, args: dict) -> dict:
        fn = {"search_jobs": self.search_jobs, "explain_job": self.explain_job, "skills_gap": self.skills_gap,
              "remember": self.remember, "update_profile": self.update_profile}.get(name)
        if not fn:
            return {"error": f"Unknown tool {name}"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}
        except llm.QuotaExhausted:
            return {"error": "The AI quota for today is used up. Search still works with a simpler ranking; try again."}
        except Exception as e:  # noqa: BLE001  - a tool failure is information for the model, not a crash
            return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def _history(chat_id: int) -> list[dict]:
    rows = db.q("select role, content from messages where chat_id=%s order by id desc limit %s", (chat_id, HISTORY))
    return [{"role": r["role"], "parts": [{"text": r["content"]["text"]}]} for r in reversed(rows)]


def _save(chat_id: int, role: str, text: str) -> None:
    db.ex("insert into messages (chat_id, role, content) values (%s,%s,%s)", (chat_id, role, Jsonb({"text": text})))


def run_turn(chat_id: int, text: str, today: str) -> dict:
    """One user message in, one reply out. Returns {'text', 'job_ids', 'steps', 'tools'}."""
    s = Session(chat_id)
    system = SYSTEM.format(profile=P.describe(s.profile) + f"\nSkills: {', '.join(s.profile.skills[:25]) or 'n/a'}"
                           f"\nInterests: {s.profile.interests or 'n/a'}",
                           prefs=json.dumps(s.user.get("prefs") or {}) or "none", today=today)
    contents = _history(chat_id) + [{"role": "user", "parts": [{"text": text}]}]
    reply, steps = "", 0
    for steps in range(1, MAX_STEPS + 1):
        content = llm.chat_with_tools(contents, TOOLS, model=config.AGENT_MODEL, purpose="agent", system=system)
        calls = [p["functionCall"] for p in content["parts"] if "functionCall" in p]
        if not calls:
            reply = "".join(p.get("text", "") for p in content["parts"] if not p.get("thought")).strip()
            break
        contents.append(content)                       # the model's own turn, verbatim (keeps tool-call signatures)
        responses = []
        for c in calls:
            result = s.call(c.get("name", ""), c.get("args") or {})
            s.tool_log.append({"tool": c.get("name"), "args": c.get("args"), "ok": "error" not in result})
            fr = {"name": c.get("name"), "response": result}
            if c.get("id"):
                fr["id"] = c["id"]
            responses.append({"functionResponse": fr})
        contents.append({"role": "user", "parts": responses})
    if not reply:                                      # step limit hit: force a final answer without tools
        try:
            reply = llm.generate_text(
                "Summarise for the user in 2-3 short sentences what was found, mentioning jobs as #id:\n"
                + json.dumps([card(h) for h in s.hits.values()])[:6000], model=config.AGENT_MODEL, purpose="agent").strip()
        except llm.LLMError:
            reply = "I ran out of steps on that one. Try asking in a simpler way."
    ids = [int(x) for x in dict.fromkeys(re.findall(r"#(\d+)", reply)) if int(x) in s.hits]
    _save(chat_id, "user", text)
    _save(chat_id, "model", reply)
    db.ex("insert into events (chat_id, kind, payload) values (%s,'agent_turn',%s)",
          (chat_id, Jsonb({"steps": steps, "tools": s.tool_log, "jobs": ids})))
    return {"text": reply, "job_ids": ids, "hits": [s.hits[i] for i in ids], "steps": steps, "tools": s.tool_log}
