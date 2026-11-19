"""Load label files written by a labeller (evals/labeling/out_*.json) into the labels tables.

  python -m evals.import_labels claude            bucket labels from out_reading_*.json and out_ranking_*.json
  python -m evals.import_labels claude --relevance  graded relevance from out_relevance*.json
"""
import json
import sys
from pathlib import Path

from canapply import db

DIR = Path(__file__).parent / "labeling"


def buckets(labeler: str) -> int:
    n = 0
    for path in sorted(DIR.glob("out_reading_*.json")) + sorted(DIR.glob("out_ranking_*.json")):
        for item in json.loads(path.read_text()):
            if item.get("bucket") not in ("can_apply", "stretch", "cannot"):
                continue
            misses = "; ".join((item.get("hard_misses") or []) + (item.get("soft_misses") or []))
            reason = (item.get("reason") or "") + (f" [{misses}]" if misses else "")
            n += db.ex("""insert into labels (job_id, labeler, bucket, reason) values (%s,%s,%s,%s)
                          on conflict (job_id, labeler) do update set bucket=excluded.bucket, reason=excluded.reason""",
                       (item["id"], labeler, item["bucket"], reason[:400]))
    return n


def relevance(labeler: str) -> int:
    n = 0
    for path in sorted(DIR.glob("out_relevance*.json")):
        for block in json.loads(path.read_text()):
            for g in block["grades"]:
                if g.get("grade") in (0, 1, 2):
                    n += db.ex("""insert into rel_labels (query, job_id, labeler, grade) values (%s,%s,%s,%s)
                                  on conflict (query, job_id, labeler) do update set grade=excluded.grade""",
                               (block["query"], g["id"], labeler, g["grade"]))
    return n


if __name__ == "__main__":
    who = sys.argv[1]
    print("relevance grades:" if "--relevance" in sys.argv else "bucket labels:",
          relevance(who) if "--relevance" in sys.argv else buckets(who))
