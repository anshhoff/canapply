"""Evals. Two questions, both answered with numbers:

  1. RANKING    Of the top 10 results for a search, how many could this person actually apply to?
                Compared across: keyword, vector, hybrid, hybrid+re-rank, +eligibility, the full pipeline.
  2. READING    How often does the requirements reading get the verdict right?
                Compared across: rules only vs LLM reading with verified quotes.

Run with:  python -m evals.run_evals [--ranking] [--reading] [--pool]
Results are written to evals/results/ and summarised in evals/RESULTS.md.
"""
import datetime as dt
import json
import math
import sys
from collections import Counter
from pathlib import Path

from canapply import config, db, extract
from canapply import profile as P
from canapply import search as S
from canapply.eligibility import judge
from canapply.requirements import Requirements

from . import judge as J

HERE = Path(__file__).parent
K = 10
VARIANTS = {
    "keyword":            dict(mode="keyword"),
    "vector":             dict(mode="vector"),
    "hybrid":             dict(mode="hybrid"),
    "hybrid + rerank":    dict(mode="hybrid", do_rerank=True),
    "hybrid + eligibility": dict(mode="hybrid", eligible_only=True, floor=None),
    "full pipeline":      dict(mode="hybrid", eligible_only=True, do_rerank=True, upgrade_top=6, floor=None),
}
# Added after the first table showed the eligibility variants padding results with unrelated jobs.
FLOOR_VARIANTS = {
    "hybrid + eligibility + relevance floor": dict(mode="hybrid", eligible_only=True),
    "full pipeline + relevance floor": dict(mode="hybrid", eligible_only=True, do_rerank=True, upgrade_top=6),
}


def ranking_add(names: list[str]) -> dict:
    """Run extra variants and add them to the saved runs (so earlier variants are not re-run)."""
    user, profile = owner()
    saved = json.loads(RUNS_FILE.read_text())
    for name in names:
        kw = {**VARIANTS, **FLOOR_VARIANTS}[name]
        saved["runs"][name] = {}
        for q in saved["queries"]:
            before = dict(S.STATS)
            hits = S.search(q, profile=profile, user={"resume_text": user["resume_text"], "chat_id": 0}, limit=K,
                            patient=True, **kw)
            if S.STATS["rerank_failed"] > before["rerank_failed"]:
                print(f"  re-rank failed for '{q}' in {name}: left out", flush=True)
                continue
            saved["runs"][name][q] = [h.id for h in hits]
        print(f"ran {name}: {[len(v) for v in saved['runs'][name].values()]}", flush=True)
    RUNS_FILE.write_text(json.dumps(saved, indent=1))
    pool = sorted({i for v in saved["runs"].values() for ids in v.values() for i in ids})
    return {"pool": len(pool), "new_files": export_for_labelling(pool, "extra")}


def owner():
    user = P.get_user(config.OWNER_CHAT_ID)
    return user, P.user_profile(user)


LABELER_ORDER = ["ansh", "claude"]          # most trusted first; 'judge:<gemini model>' labels come last


def gold_buckets(job_ids: list[int], use_judge: bool = False, profile=None) -> tuple[dict[int, str], str]:
    """Gold verdicts from the labels table. A human label beats a Claude label beats a Gemini-judge label.
    Returns (labels, description of where they came from)."""
    if use_judge:
        J.judge_buckets(job_ids, profile)
    rows = db.q("select job_id, labeler, bucket from labels where job_id = any(%s)", (job_ids,))
    rank = lambda who: LABELER_ORDER.index(who) if who in LABELER_ORDER else len(LABELER_ORDER)
    gold: dict[int, str] = {}
    src: dict[int, str] = {}
    for r in sorted(rows, key=lambda r: rank(r["labeler"]), reverse=True):   # most trusted written last
        gold[r["job_id"]] = r["bucket"]
        src[r["job_id"]] = r["labeler"].split(":")[0]
    mix = Counter(src.values())
    return gold, ", ".join(f"{n} {who}" for who, n in mix.most_common()) or "none"


def gold_relevance(query: str, job_ids: list[int]) -> dict[int, int]:
    rows = db.q("select job_id, labeler, grade from rel_labels where query=%s and job_id = any(%s)", (query, job_ids))
    rank = lambda who: LABELER_ORDER.index(who) if who in LABELER_ORDER else len(LABELER_ORDER)
    out: dict[int, int] = {}
    for r in sorted(rows, key=lambda r: rank(r["labeler"]), reverse=True):
        out[r["job_id"]] = r["grade"]
    return out


def export_for_labelling(job_ids: list[int], name: str, per_file: int = 30) -> list[str]:
    """Write postings that still have no trusted label to evals/labeling/<name>_NN.json for a labeller to read."""
    have = {r["job_id"] for r in db.q("select job_id from labels where labeler = any(%s) and job_id = any(%s)",
                                      (LABELER_ORDER, job_ids))}
    todo = [i for i in job_ids if i not in have]
    rows = {r["id"]: r for r in db.q("select * from jobs where id = any(%s)", (todo,))} if todo else {}
    out_dir = HERE / "labeling"
    out_dir.mkdir(exist_ok=True)
    files = []
    for k in range(0, len(todo), per_file):
        items = []
        for i in todo[k:k + per_file]:
            j = rows[i]
            d = j["description"]
            if len(d) > 6000:
                d = d[:4200] + "\n[...]\n" + d[-1800:]
            items.append({"id": i, "posting": extract.header(j) + "\n\n" + d})
        path = out_dir / f"{name}_{k // per_file + 1:02d}.json"
        path.write_text(json.dumps(items, indent=1))
        files.append(str(path))
    return files


def ndcg(grades: list[int], ideal: list[int], k: int = K) -> float:
    def dcg(gs):
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gs[:k]))
    best = dcg(sorted(ideal, reverse=True))
    return dcg(grades) / best if best else 0.0


# ---------------------------------------------------------------------------------------------- ranking
RUNS_FILE = HERE / "results" / "ranking_runs.json"


def ranking_run() -> dict:
    """Step 1: run every variant on every query and save what each returned."""
    user, profile = owner()
    queries = json.loads((HERE / "queries.json").read_text())
    runs: dict[str, dict[str, list[int]]] = {v: {} for v in VARIANTS}
    for q in queries:
        for name, kw in VARIANTS.items():
            # No personalisation and no senior-title filter for any variant: the comparison is the pipeline only.
            hits = S.search(q, profile=profile, user={"resume_text": user["resume_text"], "chat_id": 0}, limit=K,
                            patient=True, **kw)
            runs[name][q] = [h.id for h in hits]
        print(f"ran '{q}'  {S.STATS}", flush=True)
    RUNS_FILE.write_text(json.dumps({"date": dt.datetime.now().isoformat(timespec="minutes"), "queries": queries,
                                     "llm_calls": dict(S.STATS), "runs": runs}, indent=1))
    pool = sorted({i for v in runs.values() for ids in v.values() for i in ids})
    files = export_for_labelling(pool, "ranking")
    pairs = {q: sorted({i for v in runs.values() for i in v[q]}) for q in queries}
    rows = {r["id"]: r for r in db.q("select id, title, company, department, left(description, 600) d from jobs where id = any(%s)", (pool,))}
    rel = [{"query": q, "jobs": [{"id": i, "title": rows[i]["title"], "company": rows[i]["company"],
                                  "department": rows[i]["department"], "text": " ".join(rows[i]["d"].split())[:420]}
                                 for i in ids]} for q, ids in pairs.items()]
    (HERE / "labeling" / "relevance.json").write_text(json.dumps(rel, indent=1))
    return {"pool": len(pool), "bucket_files": files, "relevance_pairs": sum(len(v) for v in pairs.values())}


def ranking_score() -> dict:
    """Step 2: score the saved runs against the gold labels."""
    saved = json.loads(RUNS_FILE.read_text())
    queries, runs = saved["queries"], saved["runs"]
    pool = sorted({i for v in runs.values() for ids in v.values() for i in ids})
    gold, mix = gold_buckets(pool)
    table = {}
    for name in runs:
        can, reach, useful, nd, n_q, sizes, rel_share = [], [], [], [], 0, [], []
        for q in queries:
            if q not in runs[name]:
                continue
            ids = runs[name][q]
            if any(i not in gold for i in ids):   # an unlabelled result would silently flatter or punish a variant
                continue
            g = gold_relevance(q, sorted({i for v in runs.values() for i in v.get(q, [])}))
            n_q += 1
            sizes.append(len(ids))
            can.append(sum(gold[i] == "can_apply" for i in ids) / K)
            reach.append(sum(gold[i] in ("can_apply", "stretch") for i in ids) / K)
            useful.append(sum(gold[i] == "can_apply" and g.get(i, 0) >= 1 for i in ids) / K)
            nd.append(ndcg([g.get(i, 0) for i in ids], list(g.values())))
            if ids:
                rel_share.append(sum(g.get(i, 0) >= 1 for i in ids) / len(ids))
        avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
        table[name] = {"can_apply@10": avg(can), "can_apply_or_stretch@10": avg(reach),
                       "relevant_and_can_apply@10": avg(useful), "ndcg@10": avg(nd),
                       "relevant_share_of_returned": avg(rel_share),
                       "avg_results_returned": avg(sizes), "queries_scored": n_q}
    return {"run_date": saved["date"], "queries": queries, "pool": len(pool), "gold": mix, "table": table}


# ---------------------------------------------------------------------------------------------- reading
POOL_SQL = {
    "India, not senior":      "is_tech and level_rule <> 'senior' and 'IN' = any(countries)",
    "Remote, not senior":     "is_tech and level_rule <> 'senior' and (workplace_raw ilike '%%remote%%' or location_raw ilike '%%remote%%')",
    "Intern or new grad":     "is_tech and level_rule in ('intern','new_grad')",
    "Other, not senior":      "is_tech and level_rule in ('unknown','junior') and not ('IN' = any(coalesce(countries,'{}')))",
    "Senior title":           "is_tech and level_rule = 'senior'",
}
POOL_SIZES = {"India, not senior": 35, "Remote, not senior": 30, "Intern or new grad": 30, "Other, not senior": 15,
              "Senior title": 10}


def build_pool() -> int:
    """A fixed, stratified random sample of postings. Stratified because a plain random sample would be ~60%
    senior US roles, which every method gets right for the same boring reason."""
    if db.q1("select count(*) n from eval_pool")["n"]:
        return db.q1("select count(*) n from eval_pool")["n"]
    picked: list[tuple[int, str]] = []
    for why, where in POOL_SQL.items():
        rows = db.q(f"""select id from jobs where closed_at is null and {where}
                        and not (id = any(%s)) order by md5(id::text || 'canapply-eval-v1') limit %s""",
                    ([i for i, _ in picked], POOL_SIZES[why]))
        picked += [(r["id"], why) for r in rows]
    order = sorted(range(len(picked)), key=lambda i: (i % 5, i))        # interleave strata for the labeller
    for ord_, i in enumerate(order):
        db.ex("insert into eval_pool (job_id, ord, why) values (%s,%s,%s) on conflict do nothing",
              (picked[i][0], ord_, picked[i][1]))
    return len(picked)


def score(pred: dict[int, str], gold: dict[int, str]) -> dict:
    ids = [i for i in pred if i in gold]
    if not ids:
        return {"n": 0}
    acc = sum(pred[i] == gold[i] for i in ids) / len(ids)
    tp = sum(pred[i] == "can_apply" and gold[i] == "can_apply" for i in ids)
    said = sum(pred[i] == "can_apply" for i in ids)
    real = sum(gold[i] == "can_apply" for i in ids)
    confusion = Counter(f"{gold[i]} -> {pred[i]}" for i in ids if pred[i] != gold[i])
    return {"n": len(ids), "accuracy": round(acc, 3),
            "can_apply_precision": round(tp / said, 3) if said else None,
            "can_apply_recall": round(tp / real, 3) if real else None,
            "said_can_apply": said, "really_can_apply": real,
            "mistakes (gold -> predicted)": dict(confusion.most_common())}


def reading() -> dict:
    user, profile = owner()
    build_pool()
    pool = db.q("select j.*, e.why from eval_pool e join jobs j on j.id = e.job_id order by e.ord")
    ids = [j["id"] for j in pool]
    # Reading A: rules only.  Reading B: the cascade (title rule for senior titles, LLM + verified quotes otherwise).
    by_rules = {j["id"]: judge(extract.by_rules(j), profile).bucket for j in pool}
    need = [j for j in pool if j["level_rule"] != "senior" and not db.q1(
        "select 1 from requirements where job_id=%s and method='llm' and content_hash=%s", (j["id"], j["content_hash"]))]
    print(f"reading: {len(need)} postings still need an LLM reading", flush=True)
    for k in range(0, len(need), 5):
        chunk = need[k:k + 5]
        try:
            # Same models that read the index overnight, so the eval measures what is actually deployed.
            for job, raw in zip(chunk, extract.by_llm_batch(chunk, model=config.BULK_EXTRACT_MODELS)):
                if raw is not None:
                    clean, total, ok = extract.verify(raw, job)
                    extract.save(job, extract.merge(extract.by_rules(job), clean), "llm", extract.llm.last_model(), total, ok)
        except extract.llm.LLMError as e:
            print("  extraction call failed:", str(e)[:100], flush=True)
    stored = {r["job_id"]: r for r in db.q("select job_id, method, data, quotes_total, quotes_ok from requirements where job_id = any(%s)", (ids,))}
    cascade = {i: judge(Requirements.model_validate(stored[i]["data"]), profile).bucket for i in ids if i in stored}
    llm_read = [i for i in ids if stored.get(i, {}).get("method") == "llm"]
    export_for_labelling(ids, "reading")
    judged, mix = gold_buckets(ids)
    human = {r["job_id"]: r["bucket"] for r in db.q("select job_id, bucket from labels where labeler='ansh' and job_id = any(%s)", (ids,))}
    out = {"pool": len(ids), "strata": dict(Counter(j["why"] for j in pool)),
           "llm_readings_in_pool": len(llm_read),
           "quotes": {"checked": sum(stored[i]["quotes_total"] for i in llm_read),
                      "passed": sum(stored[i]["quotes_ok"] for i in llm_read)},
           "vs_gold": {"labels": len(judged), "labelled_by": mix, "gold_mix": dict(Counter(judged.values())),
                        "rules_only": score(by_rules, judged), "cascade": score(cascade, judged)},
           "vs_human": {"labels": len(human), "rules_only": score(by_rules, human), "cascade": score(cascade, human),
                        "claude_agreement_with_human": score(
                            {r["job_id"]: r["bucket"] for r in db.q("select job_id, bucket from labels where labeler='claude' and job_id = any(%s)", (list(human),))}, human)}}
    out["disagreements"] = [
        {"job_id": i, "title": next(j["title"] for j in pool if j["id"] == i), "gold": judged[i], "cascade": cascade.get(i),
         "rules": by_rules[i]} for i in ids if i in judged and cascade.get(i) != judged[i]][:40]
    return out


# ---------------------------------------------------------------------------------------------- index-wide
def index_stats() -> dict:
    user, profile = owner()
    s = db.q1("""select count(*) filter (where closed_at is null) open_jobs,
                        count(*) filter (where closed_at is null and is_tech) tech,
                        count(*) filter (where closed_at is null and is_tech and embedding is not null) embedded,
                        (select count(*) from boards where enabled) boards,
                        (select count(*) from requirements where method='llm') llm_read,
                        (select count(*) from requirements where method='rules') rules_only,
                        (select coalesce(sum(quotes_total),0) from requirements where method='llm') quotes,
                        (select coalesce(sum(quotes_ok),0) from requirements where method='llm') quotes_ok
                 from jobs""")
    rows = db.q("""select j.level_rule, r.method, r.data from jobs j join requirements r on r.job_id=j.id
                   where j.closed_at is null and j.is_tech""")
    buckets = Counter()
    for r in rows:
        buckets[judge(Requirements.model_validate(r["data"]), profile).bucket] += 1
    remote = db.q("""select r.data from jobs j join requirements r on r.job_id=j.id
                     where j.closed_at is null and j.is_tech and r.method='llm' and r.data->>'work_mode'='remote'""")
    remote_loc_fail = sum(any(c.name == "location" and c.status == "hard"
                              for c in judge(Requirements.model_validate(r["data"]), profile).checks) for r in remote)
    return {**{k: int(v) for k, v in s.items()}, "buckets_for_owner_profile": dict(buckets),
            "remote_postings_fully_read": len(remote), "remote_closed_to_owner_country": remote_loc_fail}


def main() -> None:
    which = {a for a in sys.argv[1:] if a.startswith("--")} or {"--stats", "--reading", "--ranking-score"}
    out = {"date": dt.datetime.now().isoformat(timespec="minutes"), "profile": owner()[1].model_dump(exclude={"skills"})}
    if "--pool" in which:
        print("eval pool size:", build_pool())
    if "--stats" in which:
        out["index"] = index_stats(); print(json.dumps(out["index"], indent=1), flush=True)
    if "--reading" in which:
        out["reading"] = reading(); print(json.dumps({k: v for k, v in out["reading"].items() if k != "disagreements"}, indent=1), flush=True)
    if "--ranking-run" in which:
        print(json.dumps(ranking_run(), indent=1), flush=True)
    if "--ranking-floor" in which:
        print(json.dumps(ranking_add(["hybrid + eligibility + relevance floor"]), indent=1), flush=True)
    if "--ranking-floor-full" in which:
        print(json.dumps(ranking_add(["full pipeline + relevance floor"]), indent=1), flush=True)
    if "--ranking-score" in which and RUNS_FILE.exists():
        out["ranking"] = ranking_score(); print(json.dumps(out["ranking"]["table"], indent=1), flush=True)
    path = HERE / "results" / f"{dt.datetime.now():%Y-%m-%d_%H%M}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    (HERE / "results" / "latest.json").write_text(json.dumps(out, indent=1, default=str))
    print("saved", path)


if __name__ == "__main__":
    main()
