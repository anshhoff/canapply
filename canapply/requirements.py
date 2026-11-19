"""The shape of what we pull out of a posting, and the JSON schema the LLM must fill.

Every hard requirement carries a quote. A requirement whose quote is not found verbatim in the posting
is thrown away (see extract.verify) - the model is not trusted on its own word.
"""
from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["intern", "new_grad", "junior", "mid", "senior", "unknown"]
Employment = Literal["internship", "full_time", "part_time", "contract", "unknown"]
WorkMode = Literal["remote", "hybrid", "onsite", "unknown"]
RemoteScope = Literal["anywhere", "countries", "region", "unknown"]
WorkAuth = Literal["citizens_only", "must_already_have", "sponsorship_offered", "not_mentioned"]
Student = Literal["must_be_enrolled", "recent_grad", "none", "unknown"]
Degree = Literal["none", "bachelors", "masters", "phd", "unknown"]


class Requirements(BaseModel):
    min_years: int | None = None            # minimum REQUIRED years of professional experience
    min_years_quote: str | None = None
    level: Level = "unknown"
    employment: Employment = "unknown"
    work_mode: WorkMode = "unknown"
    job_countries: list[str] = Field(default_factory=list)      # ISO-2: where the job is based
    remote_scope: RemoteScope = "unknown"                       # only meaningful when work_mode == remote
    remote_countries: list[str] = Field(default_factory=list)   # ISO-2: where a remote hire may live
    remote_regions: list[str] = Field(default_factory=list)     # e.g. EMEA, APAC, EUROPE, AMERICAS, LATAM
    location_quote: str | None = None
    work_auth: WorkAuth = "not_mentioned"
    work_auth_quote: str | None = None
    student: Student = "unknown"
    enrollment_country: str | None = None   # ISO-2 if the posting demands study in a specific country
    grad_years: list[int] = Field(default_factory=list)         # accepted graduation years, if stated
    student_quote: str | None = None
    degree: Degree = "unknown"
    skills: list[str] = Field(default_factory=list)             # must-have skills, lower-case, max 8


_N = lambda t: {"type": [t, "null"]}  # noqa: E731

LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "min_years": {**_N("integer"), "description": "Minimum years of professional experience the posting "
                      "REQUIRES. Use the lowest required number, ignore 'preferred'/'nice to have'. null if none stated."},
        "min_years_quote": {**_N("string"), "description": "Exact sentence copied from the posting that states it."},
        "level": {"type": "string", "enum": ["intern", "new_grad", "junior", "mid", "senior", "unknown"]},
        "employment": {"type": "string", "enum": ["internship", "full_time", "part_time", "contract", "unknown"]},
        "work_mode": {"type": "string", "enum": ["remote", "hybrid", "onsite", "unknown"]},
        "job_countries": {"type": "array", "items": {"type": "string"},
                          "description": "ISO-2 codes of countries where the job is based."},
        "remote_scope": {"type": "string", "enum": ["anywhere", "countries", "region", "unknown"],
                         "description": "For remote jobs: 'anywhere' only if the posting really says worldwide/any "
                         "country. 'countries' or 'region' if remote is limited. 'unknown' if not remote or unclear."},
        "remote_countries": {"type": "array", "items": {"type": "string"},
                             "description": "ISO-2 codes a remote hire must live in, if limited."},
        "remote_regions": {"type": "array", "items": {"type": "string",
                           "enum": ["EMEA", "EUROPE", "APAC", "AMERICAS", "NA", "LATAM", "MENA", "AFRICA"]}},
        "location_quote": {**_N("string"), "description": "Exact text copied from the posting (or its Location "
                           "line) that shows where the hire must be."},
        "work_auth": {"type": "string", "enum": ["citizens_only", "must_already_have", "sponsorship_offered",
                                                 "not_mentioned"],
                      "description": "citizens_only: citizenship or security clearance needed. must_already_have: "
                      "must already be authorised to work there / no visa sponsorship. sponsorship_offered: the "
                      "posting says visas are sponsored. not_mentioned otherwise."},
        "work_auth_quote": _N("string"),
        "student": {"type": "string", "enum": ["must_be_enrolled", "recent_grad", "none", "unknown"],
                    "description": "must_be_enrolled: applicant must currently be a student. recent_grad: aimed at "
                    "people who just graduated or are about to. none: no student condition."},
        "enrollment_country": {**_N("string"), "description": "ISO-2 code if the posting requires being enrolled "
                               "at a university in a specific country, else null."},
        "grad_years": {"type": "array", "items": {"type": "integer"},
                       "description": "Graduation years the posting accepts, if it names any."},
        "student_quote": _N("string"),
        "degree": {"type": "string", "enum": ["none", "bachelors", "masters", "phd", "unknown"],
                   "description": "Minimum degree REQUIRED (not preferred)."},
        "skills": {"type": "array", "items": {"type": "string"},
                   "description": "Up to 8 must-have technical skills or tools, lower-case, short names."},
    },
    "required": ["min_years", "min_years_quote", "level", "employment", "work_mode", "job_countries",
                 "remote_scope", "remote_countries", "remote_regions", "location_quote", "work_auth",
                 "work_auth_quote", "student", "enrollment_country", "grad_years", "student_quote", "degree",
                 "skills"],
}
