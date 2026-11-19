"""Give every open tech posting an embedding. Only postings whose text changed are (re)embedded.

Run with:  python -m canapply.embed --limit=20000
"""
import sys
import time

from . import db, embedder

CHUNK = 128

TODO = """
select id, title, company, location_raw, department, description, content_hash from jobs
where closed_at is null and is_tech and (embedding is null or embedded_hash <> content_hash)
order by (level_rule in ('intern','new_grad','junior')) desc, (level_rule = 'unknown') desc,
         ('IN' = any(countries)) desc, posted_at desc nulls last
limit %s
"""


def run(limit: int = 20000) -> dict:
    jobs = db.q(TODO, (limit,))
    stats = dict(todo=len(jobs), done=0)
    t0 = time.time()
    for i in range(0, len(jobs), CHUNK):
        chunk = jobs[i:i + CHUNK]
        vecs = embedder.embed_docs([embedder.doc_text(j) for j in chunk])
        with db.conn() as c:
            for j, v in zip(chunk, vecs):
                c.execute("update jobs set embedding = %s::halfvec, embedded_hash = %s where id = %s",
                          (str(v), j["content_hash"], j["id"]))
        stats["done"] += len(chunk)
        if (i // CHUNK) % 10 == 9:
            print(f"  {stats['done']}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


if __name__ == "__main__":
    limit = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--limit=")), 20000))
    print(run(limit))
