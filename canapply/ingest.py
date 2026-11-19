"""Keep the index fresh: upsert what each board lists now, close what it no longer lists.

Run with:  python -m canapply.ingest            (all enabled boards)
           python -m canapply.ingest --discover  (try every candidate in data/boards.txt first)
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from . import config, db, geo, rules
from .sources.boards import FETCHERS, BoardError
from .textutil import content_hash

SEED = config.ROOT / "data" / "boards.txt"


def load_seed() -> list[tuple[str, str]]:
    out = []
    for line in Path(SEED).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            src, tok = line.split()[:2]
            out.append((src, tok))
    return out


def sync_seed() -> int:
    n = 0
    for src, tok in load_seed():
        n += db.ex("insert into boards (source, token) values (%s,%s) on conflict do nothing", (src, tok))
    return n


def _fetch(args):
    src, tok = args
    t0 = time.time()
    try:
        with httpx.Client(timeout=45, follow_redirects=True, headers={"User-Agent": config.USER_AGENT}) as c:
            company, jobs = FETCHERS[src](c, tok)
        return src, tok, company, jobs, "ok", time.time() - t0
    except BoardError as e:
        return src, tok, None, None, str(e), time.time() - t0
    except Exception as e:  # noqa: BLE001
        return src, tok, None, None, f"error:{type(e).__name__}", time.time() - t0


UPSERT = """
insert into jobs (source, board_token, ext_id, company, title, url, location_raw, workplace_raw, department,
                  employment_raw, description, content_hash, posted_at, is_tech, level_rule, countries)
values (%(source)s, %(board_token)s, %(ext_id)s, %(company)s, %(title)s, %(url)s, %(location_raw)s,
        %(workplace_raw)s, %(department)s, %(employment_raw)s, %(description)s, %(content_hash)s, %(posted_at)s,
        %(is_tech)s, %(level_rule)s, %(countries)s)
on conflict (source, board_token, ext_id) do update set
  company = excluded.company, title = excluded.title, url = excluded.url, location_raw = excluded.location_raw,
  workplace_raw = excluded.workplace_raw, department = excluded.department,
  employment_raw = excluded.employment_raw, description = excluded.description,
  content_hash = excluded.content_hash, posted_at = coalesce(excluded.posted_at, jobs.posted_at),
  is_tech = excluded.is_tech, level_rule = excluded.level_rule, countries = excluded.countries,
  last_seen_at = now(), closed_at = null
returning (xmax = 0) as inserted
"""


def store(src: str, tok: str, company: str | None, jobs: list[dict]) -> tuple[int, int]:
    """Upsert one board's postings and close the ones that disappeared. Returns (added, closed)."""
    company = company or (db.q1("select company from boards where source=%s and token=%s", (src, tok)) or {}
                          ).get("company") or tok.replace("-", " ").title()
    added, seen = 0, []
    with db.conn() as c:
        for j in jobs:
            if not j["title"] or not j["url"] or len(j["description"]) < 80:
                continue
            countries, _, _ = geo.parse_location(j["location_raw"])
            row = dict(j, source=src, board_token=tok, company=company,
                       content_hash=content_hash(j["title"], j["location_raw"], j["description"]),
                       is_tech=rules.is_tech(j["title"], j["department"]),
                       level_rule=rules.level_from_title(j["title"]), countries=countries or None)
            r = c.execute(UPSERT, row).fetchone()
            added += 1 if r["inserted"] else 0
            seen.append(j["ext_id"])
        closed = c.execute(
            "update jobs set closed_at = now() where source=%s and board_token=%s and closed_at is null "
            "and not (ext_id = any(%s))", (src, tok, seen)).rowcount
    db.ex("update boards set company=%s where source=%s and token=%s", (company, src, tok))
    return added, closed


def run(discover: bool = False, only: str | None = None, workers: int = 6) -> dict:
    sync_seed()
    where = "true" if discover else "enabled"
    boards = db.q(f"select source, token from boards where {where} order by source, token")
    if only:
        boards = [b for b in boards if b["token"] == only or b["source"] == only]
    stats = dict(boards=len(boards), ok=0, failed=0, jobs=0, added=0, closed=0)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for src, tok, company, jobs, status, secs in pool.map(_fetch, [(b["source"], b["token"]) for b in boards]):
            if status == "ok":
                added, closed = store(src, tok, company, jobs)
                db.ex("update boards set last_run_at=now(), last_ok_at=now(), last_status='ok', "
                      "last_job_count=%s, last_added=%s, last_closed=%s, fail_streak=0, enabled=true "
                      "where source=%s and token=%s", (len(jobs), added, closed, src, tok))
                stats["ok"] += 1; stats["jobs"] += len(jobs); stats["added"] += added; stats["closed"] += closed
            else:
                # A board that has never worked is switched off; one that used to work is kept and flagged.
                db.ex("update boards set last_run_at=now(), last_status=%s, fail_streak=fail_streak+1, "
                      "enabled = (last_ok_at is not null) where source=%s and token=%s", (status, src, tok))
                stats["failed"] += 1
    stats["seconds"] = round(time.time() - t0, 1)
    db.ex("insert into events (kind, payload) values ('ingest', %s)", (db_json(stats),))
    return stats


def db_json(obj):
    from psycopg.types.json import Jsonb
    return Jsonb(obj)


if __name__ == "__main__":
    only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
    print(run(discover="--discover" in sys.argv, only=only))
