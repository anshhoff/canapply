"""Turn a resume into a structured profile (one structured-output call) and keep users in the database."""
import io

from psycopg.types.json import Jsonb
from pypdf import PdfReader

from . import config, db, embedder, llm
from .eligibility import Profile

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": ["string", "null"]},
        "country": {"type": "string", "description": "ISO-2 code of the country the person lives in. "
                    "Infer from address, phone code or university. Use 'IN' style codes."},
        "is_student": {"type": "boolean", "description": "True if still studying (degree end date is in the future)."},
        "grad_year": {"type": ["integer", "null"], "description": "Year the current or latest degree ends."},
        "degree": {"type": "string", "enum": ["none", "bachelors", "masters", "phd"],
                   "description": "Highest degree held or in progress."},
        "years_experience": {"type": "number", "description": "Years of professional experience. Count full-time "
                             "work fully. Count internships and part-time work at half. Do not count projects."},
        "skills": {"type": "array", "items": {"type": "string"}, "description": "Up to 25 technical skills, lower-case."},
        "interests": {"type": "string", "description": "Role titles and focus areas this person fits, as a short "
                      "comma-separated list a job search would use, e.g. 'ai engineer, backend engineer, llm agents, "
                      "full-stack developer'. Max 12 words, no sentences."},
    },
    "required": ["name", "country", "is_student", "grad_year", "degree", "years_experience", "skills", "interests"],
}


def pdf_to_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def profile_from_resume(resume_text: str, today: str) -> tuple[Profile, str | None]:
    raw = llm.generate_json(
        f"Today is {today}.\n\nResume:\n\"\"\"\n{resume_text[:12000]}\n\"\"\"",
        PROFILE_SCHEMA, model=config.AGENT_MODEL, purpose="profile",
        system="Read the resume and fill the profile. Only use what the resume says.", max_tokens=4000, thinking="low")
    country = (raw.get("country") or "IN").upper()[:2]
    p = Profile(country=country, work_auth=[country], is_student=bool(raw.get("is_student")),
                grad_year=raw.get("grad_year"), degree=raw.get("degree") or "bachelors",
                years_experience=round(min(float(raw.get("years_experience") or 0), 40), 1),
                skills=[s.lower()[:40] for s in (raw.get("skills") or [])][:25],
                interests=(raw.get("interests") or "")[:300])
    return p, raw.get("name")


def get_user(chat_id: int) -> dict | None:
    return db.q1("select * from users where chat_id = %s", (chat_id,))


def ensure_user(chat_id: int, name: str | None = None) -> dict:
    db.ex("insert into users (chat_id, name) values (%s, %s) on conflict (chat_id) do update set last_seen_at = now()",
          (chat_id, name))
    return get_user(chat_id)


def has_profile(user: dict | None) -> bool:
    return bool(user and user.get("profile") and user["profile"].get("country"))


def user_profile(user: dict) -> Profile:
    return Profile.model_validate(user.get("profile") or {})


def save_profile(chat_id: int, profile: Profile, resume_text: str | None = None, name: str | None = None) -> None:
    emb = None
    if resume_text:
        emb = str(embedder.embed_docs([resume_text[:3000]])[0])
    db.ex("""update users set profile = %s, resume_text = coalesce(%s, resume_text),
             resume_embedding = coalesce(%s::halfvec, resume_embedding), name = coalesce(%s, name)
             where chat_id = %s""", (Jsonb(profile.model_dump()), resume_text, emb, name, chat_id))


def set_pref(chat_id: int, key: str, value) -> dict:
    db.ex("update users set prefs = prefs || %s where chat_id = %s", (Jsonb({key: value}), chat_id))
    return get_user(chat_id)["prefs"]


def describe(p: Profile) -> str:
    who = f"student, graduating {p.grad_year}" if p.is_student and p.grad_year else ("student" if p.is_student else "not a student")
    return (f"Based in {p.country} · {who} · {p.degree} · about {p.years_experience:g} yr experience\n"
            f"Can already work in: {', '.join(p.work_auth)} · open to relocating: {'yes' if p.open_to_relocate else 'no'}\n"
            f"Looking for: {', '.join(w.replace('_', ' ') for w in p.wants)}")
