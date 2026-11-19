"""Read the hard requirements out of a posting. Cheap first, LLM only where it is needed.

  stage 1  rules   - title level, 'N years' patterns, location string, work-authorisation phrases (free)
  stage 2  LLM     - structured output with a quote for every requirement (costs one call)
  check    verify  - every quote must appear verbatim in the posting, or that requirement is dropped

Results are cached by the posting's content hash, so an unchanged posting is never read twice.

Run with:  python -m canapply.extract --limit 500
"""
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from psycopg.types.json import Jsonb

from . import config, db, geo, llm, rules
from .requirements import LLM_SCHEMA, Requirements
from .textutil import quote_in_text

MAX_CHARS = 9000

SYSTEM = """You read job postings and pull out the HARD requirements a candidate must meet to be considered.
Rules:
- Report only what the posting states. Never guess. If something is not stated, use null / "unknown" / "not_mentioned".
- Every *_quote must be copied character for character from the posting text or its Location line. No paraphrasing.
- min_years is the lowest number of years REQUIRED. Ignore "preferred", "bonus", "nice to have", "ideally".
  If the posting only says e.g. "2-4 years", min_years is 2. Internships and new-grad roles usually have null.
- A job listed as "Remote" with a single country next to it is remote only within that country
  (remote_scope "countries") unless the text says candidates from anywhere are welcome.
- grad_years lists every graduation year the posting accepts. "Graduating by June 2028" accepts 2028 and the
  years before it that are still in the future, so list those too.
- Country codes are ISO-2 (US, IN, GB, DE, ...)."""


def header(job: dict) -> str:
    return (f"Title: {job['title']}\nCompany: {job['company']}\nLocation: {job.get('location_raw') or 'not given'}\n"
            f"Workplace type: {job.get('workplace_raw') or 'not given'}\n"
            f"Employment type: {job.get('employment_raw') or 'not given'}")


def job_text(job: dict) -> str:
    """Everything a quote is allowed to come from."""
    return header(job) + "\n\n" + (job["description"] or "")


def _employment(job: dict, level: str) -> str:
    raw = f"{job.get('employment_raw') or ''} {job['title']}".lower()
    if level == "intern" or "intern" in raw:
        return "internship"
    if "contract" in raw or "freelance" in raw:
        return "contract"
    if "part" in raw and "time" in raw:
        return "part_time"
    if "full" in raw or job.get("employment_raw"):
        return "full_time"
    return "unknown"


def by_rules(job: dict) -> Requirements:
    """Stage 1. No LLM. Used on every posting, and as the baseline in the evals."""
    text = job["description"] or ""
    level = rules.level_from_title(job["title"])
    years, yq = rules.years_required(text)
    countries, regions, says_remote = geo.parse_location(job.get("location_raw"))
    wp = (job.get("workplace_raw") or "").lower()
    if "remote" in wp or (says_remote and "hybrid" not in wp and "site" not in wp):
        mode = "remote"
    elif "hybrid" in wp:
        mode = "hybrid"
    elif "site" in wp or "office" in wp:
        mode = "onsite"
    elif regions and not countries:
        mode = "remote"                              # a location given only as a region ("EMEA") means remote within it
    else:
        mode = "onsite" if countries else "unknown"
    scope = "unknown"
    if mode == "remote":
        loc = (job.get("location_raw") or "").lower()
        if re.search(r"\b(anywhere|worldwide|global|any country)\b", loc + " " + text[:1500].lower()) and not countries:
            scope = "anywhere"
        elif countries:
            scope = "countries"
        elif regions:
            scope = "region"
    sig = rules.work_auth_signals(text)
    if "citizenship" in sig:
        auth, aq = "citizens_only", sig["citizenship"]
    elif "no_sponsorship" in sig:
        auth, aq = "must_already_have", sig["no_sponsorship"]
    elif "sponsorship_offered" in sig:
        auth, aq = "sponsorship_offered", sig["sponsorship_offered"]
    elif "auth_required" in sig:
        auth, aq = "must_already_have", sig["auth_required"]
    else:
        auth, aq = "not_mentioned", None
    emp = _employment(job, level)
    return Requirements(
        min_years=years, min_years_quote=yq, level=level, employment=emp, work_mode=mode,
        job_countries=countries, remote_scope=scope, remote_countries=countries if scope == "countries" else [],
        remote_regions=regions if mode == "remote" else [], location_quote=job.get("location_raw"),
        work_auth=auth, work_auth_quote=aq,
        student="must_be_enrolled" if emp == "internship" else "unknown",
    )


def by_llm(job: dict, model: str | None = None) -> Requirements:
    """Stage 2. One structured-output call."""
    desc = job["description"] or ""
    if len(desc) > MAX_CHARS:                      # keep the start and the end; requirements live in both
        desc = desc[: MAX_CHARS - 2500] + "\n[...]\n" + desc[-2500:]
    prompt = f"{header(job)}\n\nPosting text:\n\"\"\"\n{desc}\n\"\"\""
    raw = llm.generate_json(prompt, LLM_SCHEMA, model=model or config.EXTRACT_MODEL, purpose="extract",
                            system=SYSTEM, max_tokens=4000)
    return _clean(raw)


BATCH_SCHEMA = {
    "type": "object",
    "properties": {"postings": {"type": "array", "items": {
        "type": "object",
        "properties": {"n": {"type": "integer", "description": "The posting number this entry is about."},
                       **LLM_SCHEMA["properties"]},
        "required": ["n", *LLM_SCHEMA["required"]]}}},
    "required": ["postings"],
}


def _clean(raw: dict) -> Requirements:
    raw = dict(raw)
    raw.pop("n", None)
    raw["skills"] = [s.lower().strip()[:40] for s in (raw.get("skills") or []) if isinstance(s, str)][:8]
    for k in ("job_countries", "remote_countries"):
        raw[k] = [c.upper().strip() for c in (raw.get(k) or []) if isinstance(c, str) and len(c.strip()) == 2]
    if raw.get("enrollment_country"):
        raw["enrollment_country"] = raw["enrollment_country"].upper().strip()[:2]
    if isinstance(raw.get("min_years"), int) and not (0 <= raw["min_years"] <= 25):
        raw["min_years"] = None
    return Requirements.model_validate(raw)


def by_llm_batch(jobs: list[dict], model: str | None = None, purpose: str = "extract") -> list[Requirements | None]:
    """Several postings in one call. The free tier limits requests per minute, not tokens, so batching is
    what makes thousands of postings affordable. Quotes are still checked per posting afterwards."""
    blocks = []
    for i, job in enumerate(jobs, 1):
        desc = job["description"] or ""
        if len(desc) > 6500:
            desc = desc[:4500] + "\n[...]\n" + desc[-2000:]
        blocks.append(f"=== POSTING {i} ===\n{header(job)}\n\nPosting text:\n\"\"\"\n{desc}\n\"\"\"")
    prompt = ("\n\n".join(blocks) + f"\n\nReturn one entry per posting, {len(jobs)} in total, with n = the posting "
              "number. Treat each posting on its own: a quote must come from that posting's own text.")
    raw = llm.generate_json(prompt, BATCH_SCHEMA, model=model or config.EXTRACT_MODEL, purpose=purpose,
                            system=SYSTEM, max_tokens=2500 * len(jobs) + 2000)
    out: list[Requirements | None] = [None] * len(jobs)
    for item in raw.get("postings", []):
        n = item.get("n")
        if isinstance(n, int) and 1 <= n <= len(jobs) and out[n - 1] is None:
            try:
                out[n - 1] = _clean(item)
            except Exception:  # noqa: BLE001  - one malformed entry must not sink the batch
                pass
    return out


WORLDWIDE = re.compile(r"\b(anywhere|worldwide|world-wide|global(?:ly)?|any country|any location|all countries|"
                       r"international(?:ly)?|across the world|around the world)\b", re.I)
AUTH_WORDS = {
    "citizens_only": re.compile(r"citizen|clearance|national|itar|export", re.I),
    "must_already_have": re.compile(r"authori[sz]|sponsor|eligib|right to work|permit|visa|legally", re.I),
    "sponsorship_offered": re.compile(r"sponsor|visa|relocat|immigration", re.I),
}
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def verify(req: Requirements, job: dict) -> tuple[Requirements, int, int]:
    """Two checks on every requirement the model reports, both in plain code:
      1. the quote exists verbatim in the posting, and
      2. the quote actually contains the evidence (the number of years, a word like "sponsor" or "worldwide").
    A requirement that fails either check is dropped or downgraded. Returns (clean requirements, checked, passed)."""
    text = job_text(job)
    total = ok = 0
    r = req.model_copy(deep=True)

    def check(q: str | None, evidence: bool = True) -> bool:
        nonlocal total, ok
        total += 1
        good = quote_in_text(q, text) and evidence
        ok += good
        return good

    if r.min_years is not None:
        q = r.min_years_quote or ""
        has_number = bool(re.search(rf"(?<!\d){r.min_years}(?!\d)", q)) or _WORDS.get(r.min_years, "\0") in q.lower()
        if not check(q, has_number):
            r.min_years, r.min_years_quote = None, None
    if r.work_auth != "not_mentioned":
        q = r.work_auth_quote or ""
        if not check(q, bool(AUTH_WORDS[r.work_auth].search(q))):
            r.work_auth, r.work_auth_quote = "not_mentioned", None
    if r.work_mode == "remote" and r.remote_scope == "anywhere":
        q = r.location_quote or ""
        if not check(q, bool(WORLDWIDE.search(q))):
            # "Remote" next to a list of places is remote within those places, not worldwide.
            countries, regions, _ = geo.parse_location(f"{job.get('location_raw') or ''} | {q}")
            r.remote_countries = countries or r.job_countries
            r.remote_regions = [x for x in regions if x in ("EMEA", "EUROPE", "APAC", "AMERICAS", "NA", "LATAM", "MENA", "AFRICA")]
            r.remote_scope = "countries" if r.remote_countries else ("region" if r.remote_regions else "unknown")
    elif r.work_mode == "remote" and r.remote_scope in ("countries", "region"):
        if not check(r.location_quote):
            r.remote_scope, r.remote_countries, r.remote_regions, r.location_quote = "unknown", [], [], None
    if r.enrollment_country or r.grad_years:
        if not check(r.student_quote):
            r.enrollment_country, r.grad_years, r.student_quote = None, [], None
    return r, total, ok


def reverify_all() -> dict:
    """Re-run the code checks over every stored LLM reading (no LLM calls). Used after tightening the checks."""
    rows = db.q("""select j.*, r.data as req from jobs j join requirements r on r.job_id = j.id where r.method = 'llm'""")
    changed = 0
    for row in rows:
        before = Requirements.model_validate(row["req"])
        after, _, _ = verify(before, row)
        if after != before:
            db.ex("update requirements set data = %s where job_id = %s", (Jsonb(after.model_dump()), row["id"]))
            changed += 1
    return {"checked": len(rows), "changed": changed}


def merge(rule: Requirements, model: Requirements) -> Requirements:
    """LLM result wins, but facts the board itself gives us (country list, intern in title) fill its gaps."""
    m = model.model_copy(deep=True)
    if not m.job_countries and rule.job_countries:
        m.job_countries = rule.job_countries
    if m.level == "unknown":
        m.level = rule.level
    if m.employment == "unknown":
        m.employment = rule.employment
    if m.work_mode == "unknown":
        m.work_mode = rule.work_mode
    return m


def save(job: dict, req: Requirements, method: str, model: str | None, total: int, ok: int) -> None:
    db.ex("""insert into requirements (job_id, content_hash, method, model, data, quotes_total, quotes_ok)
             values (%s,%s,%s,%s,%s,%s,%s)
             on conflict (job_id) do update set content_hash=excluded.content_hash, method=excluded.method,
               model=excluded.model, data=excluded.data, quotes_total=excluded.quotes_total,
               quotes_ok=excluded.quotes_ok, extracted_at=now()""",
          (job["id"], job["content_hash"], method, model, Jsonb(req.model_dump()), total, ok))


def extract_one(job: dict, use_llm: bool = True) -> tuple[Requirements, str]:
    rule = by_rules(job)
    if not use_llm:
        save(job, rule, "rules", None, 0, 0)
        return rule, "rules"
    raw = by_llm(job)
    clean, total, ok = verify(raw, job)
    final = merge(rule, clean)
    save(job, final, "llm", llm.last_model(), total, ok)
    return final, "llm"


# Which postings deserve an LLM call, and in what order. Senior-titled roles are settled by the title alone.
NEEDS_LLM = """
select j.* from jobs j left join requirements r on r.job_id = j.id
where j.closed_at is null and j.is_tech and j.level_rule <> 'senior'
  and (r.job_id is null or r.content_hash <> j.content_hash or r.method = 'rules')
order by (j.level_rule in ('intern','new_grad')) desc, (j.level_rule = 'junior') desc,
         ('IN' = any(j.countries)) desc, (j.workplace_raw ilike '%%remote%%' or j.location_raw ilike '%%remote%%') desc,
         j.posted_at desc nulls last
limit %s
"""
NEEDS_RULES = """
select j.* from jobs j left join requirements r on r.job_id = j.id
where j.closed_at is null and j.is_tech and (r.job_id is null or r.content_hash <> j.content_hash)
"""


def run_rules() -> int:
    """Give every tech posting a rules-based row, so nothing is ever without requirements."""
    n = 0
    for job in db.q(NEEDS_RULES):
        save(job, by_rules(job), "rules", None, 0, 0)
        n += 1
    return n


def run_llm(limit: int = 500, batch: int = 5, workers: int = 2) -> dict:
    """Bulk run. Batches rotate across the models in EXTRACT_MODEL so each model's per-minute allowance is used."""
    jobs = db.q(NEEDS_LLM, (limit,))
    chunks = [jobs[i:i + batch] for i in range(0, len(jobs), batch)]
    models = [m.strip() for m in config.BULK_EXTRACT_MODELS.split(",") if m.strip()]
    chains = [",".join(models[k:] + models[:k]) for k in range(len(models))]
    numbered = list(enumerate(chunks))
    stats = dict(todo=len(jobs), done=0, failed=0, quotes=0, quotes_ok=0, stopped=None)
    t0 = time.time()

    def work(item):
        idx, chunk = item
        try:
            results = by_llm_batch(chunk, model=chains[idx % len(chains)])
            model = llm.last_model()
        except llm.QuotaExhausted as e:
            return f"quota:{e}", 0, 0, 0
        except Exception as e:  # noqa: BLE001
            return f"err:{type(e).__name__}:{str(e)[:120]}", 0, 0, 0
        done = qt = qo = 0
        for job, raw in zip(chunk, results):
            if raw is None:
                continue
            clean, total, ok = verify(raw, job)
            save(job, merge(by_rules(job), clean), "llm", model, total, ok)
            done += 1; qt += total; qo += ok
        return "ok", done, qt, qo

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (res, done, qt, qo) in enumerate(pool.map(work, numbered)):
            if res == "ok":
                stats["done"] += done; stats["quotes"] += qt; stats["quotes_ok"] += qo
            elif res.startswith("quota"):
                stats["stopped"] = res
                pool.shutdown(wait=False, cancel_futures=True)
                break
            else:
                stats["failed"] += 1
                if stats["failed"] <= 5:
                    print("  fail:", res, flush=True)
            if (i + 1) % 25 == 0:
                print(f"  {stats['done']}/{len(jobs)} postings read in {time.time() - t0:.0f}s "
                      f"(quotes verified {stats['quotes_ok']}/{stats['quotes']})", flush=True)
    stats["seconds"] = round(time.time() - t0, 1)
    db.ex("insert into events (kind, payload) values ('extract', %s)", (Jsonb(stats),))
    return stats


if __name__ == "__main__":
    limit = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--limit=")), 500))
    if "--rules" in sys.argv:
        print("rules rows written:", run_rules())
    else:
        print("rules rows written:", run_rules())
        print(run_llm(limit))
