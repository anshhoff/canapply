"""Hybrid search with eligibility on top.

  1. keyword arm   - Postgres full-text search (finds exact words: "python", "kubernetes", a company name)
  2. vector arm    - pgvector cosine search on embeddings (finds meaning: "ML intern" ~ "AI research trainee")
  3. fusion        - Reciprocal Rank Fusion: score = sum over arms of 1 / (60 + rank). Ranks, not raw scores,
                     because the two arms score on scales that cannot be compared.
  4. eligibility   - each candidate is judged against the person's profile -> can_apply / stretch / cannot
  5. re-rank       - optional LLM pass that reorders the survivors against the person's background

`mode` picks how much of that runs, which is what the evals compare.
"""
from dataclasses import dataclass

import numpy as np

from . import config, db, embedder, llm
from .eligibility import REGION_COUNTRIES, Profile, Verdict, judge
from .requirements import Requirements

STATS = {"rerank_ok": 0, "rerank_failed": 0, "upgraded": 0, "upgrade_failed": 0}   # lets the evals see silent fallbacks
RRF_K = 60
POOL = 300          # candidates taken from each arm before fusion
# Relevance floor for the eligibility pipeline: a posting must be within this cosine distance of the 10th-best
# match in the whole index. Without it, a search with few reachable matches ("ios developer") was padded with
# reachable but unrelated jobs. Measured from the 10th-best and not the best, because one unusually close top
# match made the floor too strict. On the labelled set 0.08 kept 78% of relevant and 18% of irrelevant results.
FLOOR = 0.08
RERANK_VETO = 2     # the re-ranker scores fit 0-10; anything at or below this is dropped

KEYWORD_SQL = """
with q as (select nullif(replace(plainto_tsquery('english', %(q)s)::text, '&', '|'), '')::tsquery as tsq)
select j.id, row_number() over (order by ts_rank_cd(j.tsv, q.tsq, 32) desc, j.id) as rnk
from jobs j left join requirements r on r.job_id = j.id, q
where j.closed_at is null and j.is_tech and q.tsq is not null and j.tsv @@ q.tsq {extra}
order by ts_rank_cd(j.tsv, q.tsq, 32) desc, j.id
limit %(pool)s
"""
VECTOR_SQL = """
select j.id, row_number() over (order by j.embedding <=> %(v)s::halfvec) as rnk,
       1 - (j.embedding <=> %(v)s::halfvec) as sim
from jobs j left join requirements r on r.job_id = j.id
where j.closed_at is null and j.is_tech and j.embedding is not null {extra}
order by j.embedding <=> %(v)s::halfvec
limit %(pool)s
"""
LOAD_SQL = """
select j.id, j.title, j.company, j.url, j.location_raw, j.workplace_raw, j.department, j.employment_raw,
       j.level_rule, j.countries, j.posted_at, j.first_seen_at, left(j.description, 1200) as snippet,
       r.data as req, r.method as req_method
from jobs j left join requirements r on r.job_id = j.id
where j.id = any(%s)
"""


@dataclass
class Hit:
    job: dict
    score: float
    verdict: Verdict | None = None
    why: str | None = None          # one line from the re-ranker
    kw_rank: int | None = None
    vec_rank: int | None = None
    sim: float | None = None        # cosine similarity to the query
    fit: int | None = None          # 0-10 from the re-ranker

    @property
    def id(self) -> int:
        return self.job["id"]


def personal_vector(query_vec: list[float], user: dict | None) -> list[float]:
    """Rocchio-style personalisation: nudge the query toward the resume and liked jobs, away from disliked ones."""
    v = np.asarray(query_vec, dtype=np.float32)
    if user:
        if user.get("resume_embedding") is not None:
            v = v + 0.35 * np.asarray(user["resume_embedding"].to_list()
                                      if hasattr(user["resume_embedding"], "to_list") else user["resume_embedding"],
                                      dtype=np.float32)
        fb = db.q("""select f.action, j.embedding from feedback f join jobs j on j.id = f.job_id
                     where f.chat_id = %s and j.embedding is not null order by f.at desc limit 40""",
                  (user["chat_id"],))
        ups = [np.asarray(r["embedding"].to_list(), dtype=np.float32) for r in fb if r["action"] in ("up", "applied")]
        downs = [np.asarray(r["embedding"].to_list(), dtype=np.float32) for r in fb if r["action"] == "down"]
        if ups:
            v = v + 0.25 * np.mean(ups, axis=0)
        if downs:
            v = v - 0.15 * np.mean(downs, axis=0)
    n = float(np.linalg.norm(v))
    return (v / n).tolist() if n else v.tolist()


def rrf(*rankings: dict[int, int]) -> dict[int, float]:
    """Reciprocal Rank Fusion over any number of {id: rank} lists."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for jid, rank in ranking.items():
            scores[jid] = scores.get(jid, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def _extra(filters: dict, params: dict) -> str:
    sql = ""
    if filters.get("no_senior"):
        sql += " and j.level_rule <> 'senior'"
    if filters.get("levels"):
        sql += " and j.level_rule = any(%(levels)s)"; params["levels"] = filters["levels"]
    if filters.get("country"):
        sql += " and %(country)s = any(j.countries)"; params["country"] = filters["country"]
    if filters.get("remote"):
        sql += " and (j.workplace_raw ilike '%%remote%%' or j.location_raw ilike '%%remote%%')"
    if filters.get("company"):
        sql += " and j.company ilike %(company)s"; params["company"] = f"%{filters['company']}%"
    if filters.get("exclude_ids"):
        sql += " and not (j.id = any(%(exclude_ids)s))"; params["exclude_ids"] = filters["exclude_ids"]
    if filters.get("reach"):
        # Eligibility pushed INTO retrieval. Filtering after retrieval is not enough: if 95% of the nearest
        # postings are in countries the person cannot work in, the few they can get never make the top 300.
        # This is a loose SQL version of the geography + years checks; the exact judge still runs afterwards.
        p: Profile = filters["reach"]
        params["auth"] = list(dict.fromkeys([p.country, *p.work_auth]))
        params["regions"] = [reg for reg, cs in REGION_COUNTRIES.items() if p.country in cs] or ["-"]
        params["max_years"] = int(p.years_experience) + 1
        sql += """
          and r.job_id is not null
          and ((r.data->>'min_years') is null or (r.data->>'min_years')::int <= %(max_years)s)
          and ( (coalesce(r.data->>'work_mode','') <> 'remote' and r.data->'job_countries' ?| %(auth)s)
             or (r.data->>'work_mode' = 'remote' and (
                    r.data->>'remote_scope' = 'anywhere'
                 or r.data->'remote_countries' ?| %(auth)s
                 or r.data->'remote_regions' ?| %(regions)s
                 or (r.data->>'remote_scope' = 'unknown' and (r.data->'job_countries' ?| %(auth)s
                                                             or jsonb_array_length(r.data->'job_countries') = 0)))))"""
    if filters.get("since_days"):
        sql += " and j.first_seen_at > now() - make_interval(days => %(since_days)s)"
        params["since_days"] = filters["since_days"]
    return sql


def candidates(query: str, mode: str, user: dict | None, filters: dict) -> list[Hit]:
    params: dict = {"q": query, "pool": POOL}
    extra = _extra(filters, params)
    kw: dict[int, int] = {}
    vec: dict[int, int] = {}
    sims: dict[int, float] = {}
    if mode != "vector":
        kw = {r["id"]: r["rnk"] for r in db.q(KEYWORD_SQL.format(extra=extra), params)}
    if mode != "keyword":
        qv = embedder.embed_query(query)
        if mode.endswith("+personal") or filters.get("personal"):
            qv = personal_vector(qv, user)
        params["v"] = str(qv)
        with db.conn() as c:
            c.execute("set hnsw.ef_search = 400")
            if extra:
                c.execute("set hnsw.iterative_scan = relaxed_order")
            for r in c.execute(VECTOR_SQL.format(extra=extra), params).fetchall():
                vec[r["id"]] = r["rnk"]
                sims[r["id"]] = float(r["sim"])
    scores = rrf(kw, vec)
    ids = sorted(scores, key=lambda i: (-scores[i], i))
    rows = {r["id"]: r for r in db.q(LOAD_SQL, (ids,))} if ids else {}
    return [Hit(rows[i], scores[i], kw_rank=kw.get(i), vec_rank=vec.get(i), sim=sims.get(i)) for i in ids if i in rows]


def attach_verdicts(hits: list[Hit], profile: Profile) -> None:
    for h in hits:
        if h.job.get("req"):
            h.verdict = judge(Requirements.model_validate(h.job["req"]), profile)


RERANK_SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "score": {"type": "integer", "description": "0-10 fit"},
                       "why": {"type": "string", "description": "One short sentence: why this fits or does not."}},
        "required": ["id", "score", "why"]}}},
    "required": ["ranking"],
}


def rerank(query: str, hits: list[Hit], profile: Profile, resume: str | None, top: int = 25,
           patient: bool = False) -> list[Hit]:
    """LLM re-ranker over the top candidates. Anything it does not mention keeps its place after the ranked ones."""
    head, tail = hits[:top], hits[top:]
    if len(head) < 3:
        return hits
    cards = "\n\n".join(
        f"[{h.id}] {h.job['title']} - {h.job['company']} ({h.job.get('location_raw') or 'location n/a'})\n"
        f"{' '.join((h.job.get('snippet') or '').split())[:500]}" for h in head)
    who = (f"Candidate: {profile.years_experience:g} yrs experience, based in {profile.country}, "
           f"{'student graduating ' + str(profile.grad_year) if profile.is_student and profile.grad_year else 'not a student'}. "
           f"Skills: {', '.join(profile.skills[:20]) or 'n/a'}. Interests: {profile.interests or 'n/a'}.")
    if resume:
        who += f"\nResume excerpt: {' '.join(resume.split())[:1500]}"
    prompt = (f"{who}\n\nSearch request: {query}\n\nJobs:\n{cards}\n\n"
              "Score every job 0-10 for how well it fits the search request and this candidate's background "
              "(role type and skills match matter most). Return all of them, best first.")
    try:
        out = llm.generate_json(prompt, RERANK_SCHEMA, model=config.RERANK_MODEL,
                                purpose="rerank_batch" if patient else "rerank", max_tokens=8000, thinking="low")
        STATS["rerank_ok"] += 1
    except llm.LLMError:
        STATS["rerank_failed"] += 1
        return hits                                  # a failed re-rank must never break search
    by_id = {h.id: h for h in head}
    ranked: list[Hit] = []
    for item in sorted(out.get("ranking", []), key=lambda x: -x.get("score", 0)):
        h = by_id.pop(item.get("id"), None)
        if h:
            h.why, h.fit = item.get("why"), item.get("score")
            ranked.append(h)
    return ranked + list(by_id.values()) + tail


def upgrade(hits: list[Hit], profile: Profile, n: int = 6, patient: bool = False) -> int:
    """Lazy stage 2: postings about to be shown that only have a rules-based reading get a proper LLM reading
    now (one batched call), then are judged again. This is how the whole index gets covered without paying
    for an LLM call on postings nobody ever sees."""
    from . import extract  # local import: extract imports nothing from search
    todo = [h for h in hits if h.job.get("req_method") == "rules"][:n]
    if not todo:
        return 0
    full = {r["id"]: r for r in db.q("select * from jobs where id = any(%s)", ([h.id for h in todo],))}
    jobs = [full[h.id] for h in todo if h.id in full]
    try:
        results = extract.by_llm_batch(jobs, purpose="extract" if patient else "rerank")   # live = fail fast
    except llm.LLMError:
        STATS["upgrade_failed"] += 1
        return 0                                      # keep the rules reading; never block a search on this
    done = 0
    for h, job, raw in zip(todo, jobs, results):
        if raw is None:
            continue
        clean, total, ok = extract.verify(raw, job)
        final = extract.merge(extract.by_rules(job), clean)
        extract.save(job, final, "llm", llm.last_model(), total, ok)
        h.job["req"], h.job["req_method"] = final.model_dump(), "llm"
        h.verdict = judge(final, profile)
        done += 1
    STATS["upgraded"] += done
    return done


def search(query: str, *, profile: Profile | None = None, user: dict | None = None, mode: str = "hybrid",
           eligible_only: bool = False, do_rerank: bool = False, filters: dict | None = None,
           limit: int = 10, buckets: tuple[str, ...] = ("can_apply", "stretch"),
           upgrade_top: int = 0, patient: bool = False, floor: float | None = FLOOR) -> list[Hit]:
    """mode: keyword | vector | hybrid.  eligible_only + do_rerank switch on stages 4 and 5.
    upgrade_top > 0 gives the first few results a full LLM reading if they only have a rules-based one.
    patient=True (evals, digests) lets LLM calls retry instead of failing fast.
    floor=None switches the relevance floor off (the evals use this to measure what the floor changes)."""
    filters = dict(filters or {})
    if not (eligible_only and profile):
        hits = candidates(query, mode, user, filters)
        if profile:
            attach_verdicts(hits, profile)
        if do_rerank and profile:
            hits = rerank(query, hits, profile, (user or {}).get("resume_text"), patient=patient)
        return hits[:limit]

    def keep(hs: list[Hit]) -> list[Hit]:
        return [h for h in hs if h.verdict and h.verdict.bucket in buckets]

    # Two retrievals: one restricted to postings in reach (finds the "can apply" jobs), one open (finds stretch).
    near = candidates(query, mode, user, {**filters, "reach": profile})
    wide = candidates(query, mode, user, filters)
    attach_verdicts(near, profile)
    attach_verdicts(wide, profile)
    seen = {h.id for h in near}
    hits = keep(near) + keep([h for h in wide if h.id not in seen])
    if floor is not None and mode != "keyword":
        top = sorted((h.sim for h in wide if h.sim is not None), reverse=True)[:10]
        if top:
            ref = top[-1]                                # 10th-best match in the whole index
            hits = [h for h in hits if h.sim is not None and ref - h.sim <= floor]
    for _ in range(2 if upgrade_top else 0):               # lazy full reading of what is about to be shown
        if not upgrade(hits[:max(limit, upgrade_top)], profile, n=upgrade_top, patient=patient):
            break
        hits = keep(hits)
    order = {"can_apply": 0, "stretch": 1, "cannot": 2}
    hits.sort(key=lambda h: order[h.verdict.bucket])        # stable: relevance order kept inside each bucket
    if do_rerank:
        hits = rerank(query, hits, profile, (user or {}).get("resume_text"), top=min(25, max(limit * 2, 10)),
                      patient=patient)
        hits = [h for h in hits if h.fit is None or h.fit > RERANK_VETO]         # the re-ranker can veto
        head = sorted(hits[:limit * 2], key=lambda h: order[h.verdict.bucket])
        hits = head + hits[limit * 2:]
    return hits[:limit]
