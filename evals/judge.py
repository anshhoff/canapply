"""Gold labels for the evals.

Two labelers write to the same tables:
  - 'ansh'            a human, through the bot's /label mode (the labels that count)
  - 'judge:<model>'   an LLM that reads the WHOLE posting and applies the rubric directly

The judge is deliberately a different route from the system under test: the system extracts fields and lets
code decide; the judge reads the posting end to end and decides. Where they disagree, one of them is wrong,
and that is what the error analysis looks at. Judge labels are a stand-in until enough human labels exist.
"""
from canapply import config, db, llm
from canapply.eligibility import Profile
from canapply.extract import header

RUBRIC = """Decide whether THIS candidate could realistically apply to each posting and be considered.

HARD miss (cannot be fixed by effort this month):
- the posting needs citizenship, a security clearance, or says the candidate must ALREADY have the right to
  work in a country where this candidate has no work rights, or says no visa sponsorship
- it is remote but limited to countries or regions that do not include the candidate's country
  (a job listed as "Remote" next to one country is remote within that country only)
- the candidate must be enrolled at a university in a different country
- it is an internship based in the United States and the candidate has no US work rights
- a PhD is required
- it requires 2 or more years of experience beyond what the candidate has, or is clearly a senior/staff/lead role

SOFT miss (a strong candidate sometimes gets past it):
- it requires about 1 year more experience than the candidate has
- it is on-site or hybrid in a country where the candidate has no work rights, and the posting does not say
  whether it sponsors visas
- a Master's degree is required
- the stated graduation year is one year off

Verdict:
- can_apply: no hard miss and no soft miss
- stretch: no hard miss and exactly one soft miss
- cannot: any hard miss, or two or more soft misses

Use only what the posting says. If the posting does not state a requirement, do not count it as a miss."""

BUCKET_SCHEMA = {
    "type": "object",
    "properties": {"postings": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "hard_misses": {"type": "array", "items": {"type": "string"}},
            "soft_misses": {"type": "array", "items": {"type": "string"}},
            "bucket": {"type": "string", "enum": ["can_apply", "stretch", "cannot"]},
            "reason": {"type": "string", "description": "One short sentence."}},
        "required": ["n", "hard_misses", "soft_misses", "bucket", "reason"]}}},
    "required": ["postings"],
}
REL_SCHEMA = {
    "type": "object",
    "properties": {"grades": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "integer"}, "grade": {"type": "integer"}},
        "required": ["id", "grade"]}}},
    "required": ["grades"],
}


def candidate_text(p: Profile) -> str:
    return (f"Candidate: lives in {p.country}; has the right to work only in: {', '.join(p.work_auth)}; "
            f"open to relocating: {'yes' if p.open_to_relocate else 'no'}; "
            f"{'currently a student, graduating in ' + str(p.grad_year) if p.is_student else 'not a student'}; "
            f"degree in progress or held: {p.degree}; professional experience: about {p.years_experience:g} years "
            f"(internships and part-time work); looking for internships and full-time roles.")


def labeler_name() -> str:
    return "judge:" + (llm.last_model() or config.JUDGE_MODEL.split(",")[0])


def judge_buckets(job_ids: list[int], profile: Profile, batch: int = 4) -> dict[int, str]:
    """Label every posting not yet labelled by a judge. Returns {job_id: bucket} for all requested ids."""
    have = {r["job_id"]: r["bucket"] for r in db.q(
        "select job_id, bucket from labels where job_id = any(%s) and labeler like 'judge:%%'", (job_ids,))}
    todo = [i for i in job_ids if i not in have]
    rows = {r["id"]: r for r in db.q("select * from jobs where id = any(%s)", (todo,))} if todo else {}
    for k in range(0, len(todo), batch):
        chunk = [rows[i] for i in todo[k:k + batch] if i in rows]
        blocks = []
        for n, job in enumerate(chunk, 1):
            d = job["description"]
            if len(d) > 7000:
                d = d[:5000] + "\n[...]\n" + d[-2000:]
            blocks.append(f"=== POSTING {n} ===\n{header(job)}\n\n{d}")
        prompt = f"{candidate_text(profile)}\n\n" + "\n\n".join(blocks) + f"\n\nGive a verdict for each of the {len(chunk)} postings."
        try:
            out = llm.generate_json(prompt, BUCKET_SCHEMA, model=config.JUDGE_MODEL, purpose="judge", system=RUBRIC,
                                    max_tokens=6000)
        except llm.LLMError as e:
            print("  judge call failed:", str(e)[:120], flush=True)
            continue
        who = labeler_name()
        for item in out.get("postings", []):
            n = item.get("n")
            if isinstance(n, int) and 1 <= n <= len(chunk) and item.get("bucket") in ("can_apply", "stretch", "cannot"):
                jid = chunk[n - 1]["id"]
                db.ex("""insert into labels (job_id, labeler, bucket, reason) values (%s,%s,%s,%s)
                         on conflict (job_id, labeler) do update set bucket=excluded.bucket, reason=excluded.reason""",
                      (jid, who, item["bucket"], (item.get("reason") or "")[:300]))
                have[jid] = item["bucket"]
        print(f"  judged {min(k + batch, len(todo))}/{len(todo)}", flush=True)
    return have


def judge_relevance(query: str, job_ids: list[int], batch: int = 20) -> dict[int, int]:
    have = {r["job_id"]: r["grade"] for r in db.q(
        "select job_id, grade from rel_labels where query=%s and job_id = any(%s) and labeler like 'judge:%%'",
        (query, job_ids))}
    todo = [i for i in job_ids if i not in have]
    rows = {r["id"]: r for r in db.q(
        "select id, title, company, department, left(description, 700) d from jobs where id = any(%s)", (todo,))} if todo else {}
    for k in range(0, len(todo), batch):
        chunk = [rows[i] for i in todo[k:k + batch] if i in rows]
        cards = "\n".join(f"[{j['id']}] {j['title']} | {j['company']} | {j['department'] or ''} | "
                          f"{' '.join(j['d'].split())[:400]}" for j in chunk)
        prompt = (f"Search request: \"{query}\"\n\nFor each job, grade how well the ROLE matches what was asked for. "
                  "Judge the kind of work and level only. Ignore location and whether the person could get it.\n"
                  "2 = this is the kind of role asked for\n1 = related or adjacent role\n0 = a different kind of role\n\n"
                  f"{cards}")
        try:
            out = llm.generate_json(prompt, REL_SCHEMA, model=config.JUDGE_MODEL, purpose="judge", max_tokens=4000)
        except llm.LLMError as e:
            print("  relevance call failed:", str(e)[:120], flush=True)
            continue
        who = labeler_name()
        ids = {j["id"] for j in chunk}
        for item in out.get("grades", []):
            if item.get("id") in ids and item.get("grade") in (0, 1, 2):
                db.ex("""insert into rel_labels (query, job_id, labeler, grade) values (%s,%s,%s,%s)
                         on conflict (query, job_id, labeler) do update set grade=excluded.grade""",
                      (query, item["id"], who, item["grade"]))
                have[item["id"]] = item["grade"]
    return have
