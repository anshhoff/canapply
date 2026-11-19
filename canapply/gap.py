"""'What should I learn next?' answered from the index instead of from opinion.

Take the jobs that match the person's interests, judge each one, then count:
  - what blocks them most often (experience, location, work authorisation, ...)
  - which must-have skills appear most in the roles within reach that their profile does not list
"""
from collections import Counter

from . import search as S
from .eligibility import Profile

ALIASES = {"js": "javascript", "ts": "typescript", "node": "node.js", "nodejs": "node.js", "postgres": "postgresql",
           "k8s": "kubernetes", "golang": "go", "py": "python", "reactjs": "react", "react.js": "react",
           "ml": "machine learning", "llms": "llm", "large language models": "llm", "gcp": "google cloud",
           "amazon web services": "aws", "ci/cd": "ci/cd", "rest apis": "rest", "restful apis": "rest"}


def canon(skill: str) -> str:
    s = skill.lower().strip(" .")
    return ALIASES.get(s, s)


def analyse(profile: Profile, user: dict | None, query: str | None = None, pool: int = 400) -> dict:
    query = query or profile.interests or "software engineer"
    filters = {"personal": True}
    hits = S.candidates(query, "hybrid", user, filters)[:pool]
    near = S.candidates(query, "hybrid", user, {**filters, "reach": profile})[:pool]   # for the skills count
    S.attach_verdicts(hits, profile)
    S.attach_verdicts(near, profile)
    judged = [h for h in hits if h.verdict]
    buckets = Counter(h.verdict.bucket for h in judged)
    blockers = Counter()
    examples: dict[str, str] = {}
    for h in judged:
        for c in h.verdict.blockers:
            key = {"experience": "Years of experience", "location": "Location (remote limited to other countries)",
                   "work_auth": "Work authorisation / visa", "student": "Student or graduation-year rules",
                   "degree": "Degree level"}.get(c.name, c.name)
            blockers[key] += 1
            examples.setdefault(key, f"{h.job['title']} at {h.job['company']}: {c.reason}")
    have = {canon(s) for s in profile.skills}
    wanted = Counter()
    reach = [h for h in near if h.verdict and h.verdict.bucket in ("can_apply", "stretch")]
    for h in reach:
        for s in {canon(x) for x in (h.job["req"].get("skills") or [])}:
            wanted[s] += 1
    missing = [(s, n) for s, n in wanted.most_common(60) if s not in have and not any(s in x or x in s for x in have)]
    return dict(query=query, total=len(judged), buckets=dict(buckets), blockers=blockers.most_common(5),
                examples=examples, reach=len(reach), missing_skills=missing[:8],
                asked_skills=wanted.most_common(8))
