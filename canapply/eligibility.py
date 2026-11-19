"""Pure code, no LLM: a person's profile + a posting's requirements -> Can apply / Stretch / Can't, with reasons.

The LLM only reads the posting. The decision is made here, so it is testable and explainable.

  hard miss  = something effort can't fix this month (work authorisation, wrong country, PhD, 2+ years short)
  soft miss  = a gap a strong candidate sometimes gets past (about a year short, visa not mentioned, Masters)

  no misses            -> can_apply
  exactly one soft miss -> stretch (and we say which one)
  anything else         -> cannot
"""
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from .requirements import Requirements

Bucket = Literal["can_apply", "stretch", "cannot"]

REGION_COUNTRIES = {
    "APAC": {"IN", "SG", "JP", "KR", "CN", "HK", "TW", "AU", "NZ", "ID", "MY", "TH", "VN", "PH", "PK", "BD", "LK"},
    "EUROPE": {"GB", "IE", "DE", "FR", "NL", "ES", "PT", "IT", "SE", "DK", "NO", "FI", "PL", "CH", "AT", "BE",
               "CZ", "RO", "HU", "GR", "EE", "LT", "LV", "UA", "RS", "BG", "HR", "LU"},
    "NA": {"US", "CA"}, "AMERICAS": {"US", "CA", "MX", "BR", "AR", "CO", "CL"},
    "LATAM": {"MX", "BR", "AR", "CO", "CL"}, "MENA": {"AE", "SA", "QA", "EG", "IL", "TR"},
    "AFRICA": {"ZA", "NG", "KE", "EG"},
}
REGION_COUNTRIES["EMEA"] = REGION_COUNTRIES["EUROPE"] | REGION_COUNTRIES["MENA"] | REGION_COUNTRIES["AFRICA"]

LEVEL_YEARS = {"intern": 0, "new_grad": 0, "junior": 1, "mid": 3, "senior": 5}
# Countries where an internship practically requires being enrolled locally (or already holding work rights):
# a US internship needs CPT/OPT through a US university, or a J-1 that employers rarely arrange.
INTERN_NEEDS_LOCAL_STATUS = {"US"}
DEGREE_RANK = {"none": 0, "unknown": 0, "bachelors": 1, "masters": 2, "phd": 3}


class Profile(BaseModel):
    country: str = "IN"                               # ISO-2, where the person lives
    work_auth: list[str] = Field(default_factory=lambda: ["IN"])   # countries they can already work in
    open_to_relocate: bool = True
    years_experience: float = 0                       # professional experience incl. substantial internships
    is_student: bool = True
    grad_year: int | None = None
    degree: str = "bachelors"                         # highest degree held or in progress
    wants: list[str] = Field(default_factory=lambda: ["internship", "full_time"])
    skills: list[str] = Field(default_factory=list)
    interests: str = ""                               # free text: what kind of roles


@dataclass
class Check:
    name: str                 # experience | location | work_auth | student | degree
    status: str               # pass | soft | hard | unknown
    reason: str
    quote: str | None = None


@dataclass
class Verdict:
    bucket: Bucket
    checks: list[Check] = field(default_factory=list)

    @property
    def blockers(self) -> list[Check]:
        return [c for c in self.checks if c.status in ("soft", "hard")]

    @property
    def unverified(self) -> list[Check]:
        return [c for c in self.checks if c.status == "unknown"]

    def headline(self) -> str:
        if self.bucket == "can_apply":
            return "You meet every stated requirement."
        b = self.blockers
        return b[0].reason if b else ""


def _experience(r: Requirements, p: Profile) -> Check:
    need = r.min_years
    source = "posting"
    if need is None and r.level == "senior":          # a senior title settles it even with no number given
        need, source = LEVEL_YEARS["senior"], "title"
    if need is None:
        if r.level in ("intern", "new_grad", "junior"):
            return Check("experience", "pass", f"{r.level.replace('_', ' ')} role, no years of experience required")
        if r.level == "mid":
            return Check("experience", "unknown", "Reads like a mid-level role, but the posting gives no number of years")
        return Check("experience", "unknown", "The posting does not say how much experience it needs")
    gap = need - p.years_experience
    where = "asks for" if source == "posting" else "is a senior-titled role, which usually means"
    if gap <= 0.5:                                    # within half a year counts as meeting it
        return Check("experience", "pass", f"Needs {need} yr, you have about {p.years_experience:g}", r.min_years_quote)
    if gap < 2:                                       # "about a year short": a real gap, but people get past it
        return Check("experience", "soft", f"It {where} {need}+ years; you have about {p.years_experience:g}",
                     r.min_years_quote)
    return Check("experience", "hard", f"It {where} {need}+ years of experience; you have about {p.years_experience:g}",
                 r.min_years_quote)


def _in_regions(country: str, regions: list[str]) -> bool:
    return any(country in REGION_COUNTRIES.get(reg, set()) for reg in regions)


def _location(r: Requirements, p: Profile) -> list[Check]:
    q = r.location_quote
    if r.work_mode == "remote":
        if r.remote_scope == "anywhere":
            return [Check("location", "pass", "Remote, open worldwide", q)]
        if r.remote_scope in ("countries", "region") and (r.remote_countries or r.remote_regions):
            places = ", ".join(r.remote_countries + r.remote_regions)
            if (p.country in r.remote_countries or any(c in p.work_auth for c in r.remote_countries)
                    or _in_regions(p.country, r.remote_regions)):
                return [Check("location", "pass", f"Remote, and your country is covered ({places})", q)]
            return [Check("location", "hard", f"Remote, but only for people in {places}", q)]
        # remote with no stated scope: fall back to the countries the board lists
        if r.job_countries:
            if p.country in r.job_countries or any(c in p.work_auth for c in r.job_countries):
                return [Check("location", "pass", f"Remote, listed for {', '.join(r.job_countries)}", q)]
            return [Check("location", "hard",
                          f"Listed as remote in {', '.join(r.job_countries)} only; nothing says other countries are accepted", q)]
        return [Check("location", "unknown", "Remote, but the posting does not say from which countries", q)]

    # on-site or hybrid (or unknown mode with a known country)
    if not r.job_countries:
        return [Check("location", "unknown", "The posting does not make the location clear", q)]
    if any(c in p.work_auth for c in r.job_countries):
        return [Check("location", "pass", f"Based in {', '.join(r.job_countries)}, where you can already work", q)]
    where = ", ".join(r.job_countries)
    if not p.open_to_relocate:
        return [Check("location", "hard", f"On-site in {where} and you are not open to relocating", q)]
    if r.work_auth == "citizens_only":
        return [Check("work_auth", "hard", f"Based in {where} and needs citizenship or a security clearance",
                      r.work_auth_quote)]
    if r.work_auth == "must_already_have":
        return [Check("work_auth", "hard", f"Based in {where} and you must already have the right to work there",
                      r.work_auth_quote)]
    if r.work_auth == "sponsorship_offered":
        return [Check("work_auth", "pass", f"Based in {where}; the posting says they sponsor visas", r.work_auth_quote)]
    if r.employment == "internship" and all(c in INTERN_NEEDS_LOCAL_STATUS for c in r.job_countries):
        return [Check("work_auth", "hard", f"Internship based in {where}: interns there normally need local work "
                      "rights through a local university, and the posting offers no visa route", q)]
    return [Check("work_auth", "soft", f"Based in {where}; you would need a visa and the posting does not say they sponsor",
                  q)]


def _student(r: Requirements, p: Profile) -> Check | None:
    q = r.student_quote
    if r.enrollment_country and r.enrollment_country != p.country and r.enrollment_country not in p.work_auth:
        return Check("student", "hard", f"You must be enrolled at a university in {r.enrollment_country}", q)
    if r.grad_years and p.grad_year and p.grad_year not in r.grad_years:
        near = min(abs(p.grad_year - y) for y in r.grad_years)
        return Check("student", "soft" if near <= 1 else "hard",
                     f"It is for people graduating in {', '.join(map(str, r.grad_years))}; you graduate in {p.grad_year}", q)
    if r.student == "must_be_enrolled" or r.employment == "internship":
        if p.is_student:
            return Check("student", "pass", "Open to current students", q)
        return Check("student", "hard", "It is only open to current students", q)
    return None


def _degree(r: Requirements, p: Profile) -> Check | None:
    need, have = DEGREE_RANK.get(r.degree, 0), DEGREE_RANK.get(p.degree, 1)
    if need <= have or need <= 1:
        return None
    if r.degree == "phd":
        return Check("degree", "hard", "A PhD is required")
    return Check("degree", "soft", "A Master's degree is required")


def judge(r: Requirements, p: Profile) -> Verdict:
    checks: list[Check] = [_experience(r, p), *_location(r, p)]
    for extra in (_student(r, p), _degree(r, p)):
        if extra:
            checks.append(extra)
    hard = sum(c.status == "hard" for c in checks)
    soft = sum(c.status == "soft" for c in checks)
    if hard == 0 and soft == 0:
        bucket: Bucket = "can_apply"
    elif hard == 0 and soft == 1:
        bucket = "stretch"
    else:
        bucket = "cannot"
    order = {"hard": 0, "soft": 1, "unknown": 2, "pass": 3}
    checks.sort(key=lambda c: order[c.status])
    return Verdict(bucket, checks)
