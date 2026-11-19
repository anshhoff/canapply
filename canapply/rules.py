"""Stage 1 of the cascade: free, instant rules that run on every posting at ingest time.

They answer three things without an LLM call:
  - is this a tech role at all? (we only spend LLM calls on tech roles)
  - what level does the TITLE claim? (a 'Staff Engineer' needs no further reading)
  - how many years does the text ask for, by pattern?
"""
import re

TECH_RE = re.compile(
    r"\b(engineer|engineering|developer|software|swe|sde|sdet|programmer|backend|back-end|frontend|front-end|"
    r"full[- ]?stack|data scien|data analy|data engineer|analytics|machine learning|ml\b|ai\b|a\.i\.|llm|nlp|"
    r"deep learning|research scientist|research engineer|researcher|applied scientist|scientist|devops|sre|"
    r"site reliability|platform|infrastructure|security engineer|ios|android|mobile|web developer|qa\b|"
    r"quality assurance|test engineer|solutions engineer|forward deployed|fde\b|technical|technology|"
    r"product manager|product engineer|firmware|embedded|robotics|cloud|database|systems|architect|"
    r"computer vision|prompt|agent|automation|member of technical staff|mts\b|intern\b|internship|"
    r"new grad|graduate|apprentice|trainee|fellow)\b", re.I)

NON_TECH_RE = re.compile(
    r"\b(account executive|sales|recruit|talent|attorney|counsel|legal|paralegal|accountant|accounting|"
    r"payroll|finance manager|controller|tax\b|marketing manager|brand|copywriter|content writer|events|"
    r"office manager|executive assistant|workplace|facilities|customer success manager|account manager|"
    r"business development|bdr\b|sdr\b|partnerships|procurement|hr\b|people partner|people operations|"
    r"communications|public relations|policy|warehouse|driver|nurse|physician|chef|barista)\b", re.I)

SENIOR_RE = re.compile(
    r"\b(senior|sr\.?|staff|principal|lead|leader|head of|director|vp\b|vice president|manager|managing|"
    r"architect|distinguished|fellow engineer|chief|expert|specialist iii|iii\b|iv\b|l5\b|l6\b|l7\b|"
    r"engineering manager|tech lead|team lead)\b", re.I)
INTERN_RE = re.compile(r"\b(intern|interns|internship|co-?op|apprentice|apprenticeship|trainee|working student|"
                       r"werkstudent|summer analyst|student researcher|stagiaire|praktik)\b", re.I)
NEW_GRAD_RE = re.compile(r"\b(new grad|new graduate|university grad|graduate (?:software|engineer|program|scheme)|"
                         r"campus hire|early career|early-career|entry[- ]level|recent grad|grad 202\d|"
                         r"class of 202\d|graduate engineer|associate engineer|engineer i\b|sde[- ]?1|sde i\b|"
                         r"software engineer 1\b|level 1\b)\b", re.I)
JUNIOR_RE = re.compile(r"\b(junior|jr\.?|associate|engineer ii\b|sde[- ]?2)\b", re.I)


def is_tech(title: str, department: str | None = None) -> bool:
    t = title or ""
    if NON_TECH_RE.search(t) and not re.search(r"\b(engineer|developer|scientist|technical)\b", t, re.I):
        return False
    if TECH_RE.search(t):
        return True
    return bool(department and re.search(r"\b(engineering|data|research|machine learning|ai|product|technology|"
                                         r"infrastructure|security)\b", department, re.I)
                and not NON_TECH_RE.search(t))


def level_from_title(title: str) -> str:
    t = title or ""
    if INTERN_RE.search(t):
        return "intern"
    if NEW_GRAD_RE.search(t):
        return "new_grad"
    if SENIOR_RE.search(t):
        return "senior"
    if JUNIOR_RE.search(t):
        return "junior"
    return "unknown"


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
              "nine": 9, "ten": 10}
_NUM = r"(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"
YEARS_RE = re.compile(
    rf"(?:(?:at least|minimum(?: of)?|min\.?|over|more than)\s+)?{_NUM}\s*(?:\+|plus)?\s*"
    rf"(?:(?:-|–|to)\s*{_NUM}\s*)?\+?\s*(?:years?|yrs?)\b(?P<tail>[^.\n;]{{0,80}})", re.I)
_EXP_WORDS = re.compile(r"experience|exp\b|background|track record|industry|professional|working|hands-on|"
                        r"building|developing|in a |as a |of software|in software|of relevant|post", re.I)
_NOT_REQ = re.compile(r"\b(old|ago|founded|history|anniversary|vesting|cliff|warranty|year plan|years of "
                      r"growth|in business|over the (?:last|past)|for the (?:last|past)|in the (?:last|past)|"
                      r"next \d|within)\b", re.I)


def _to_int(s: str) -> int:
    return _NUM_WORDS.get(s.lower(), None) or int(s)


def years_required(text: str) -> tuple[int | None, str | None]:
    """Smallest 'N years of experience' the posting asks for, with the sentence it came from.

    Smallest, because postings often list '3+ years' as required and '5+ years' as preferred.
    """
    best: tuple[int, str] | None = None
    for m in YEARS_RE.finditer(text or ""):
        tail = m.group("tail") or ""
        start = max(0, m.start() - 60)
        window = text[start:m.end()]
        if _NOT_REQ.search(window):
            continue
        if not _EXP_WORDS.search(tail) and not _EXP_WORDS.search(text[start:m.start()]):
            continue
        n = _to_int(m.group(1))
        if n > 20:
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        quote = text[line_start: line_end if line_end != -1 else len(text)].strip(" -•*\t")[:300]
        if best is None or n < best[0]:
            best = (n, quote)
    return best if best else (None, None)


WORK_AUTH_PATTERNS = [
    ("citizenship", re.compile(r"\b(u\.?s\.? citizen(?:ship)?|must be a (?:us|u\.s\.|united states) citizen|"
                               r"security clearance|ts/sci|secret clearance|itar|export control)", re.I)),
    ("no_sponsorship", re.compile(r"(?:not|unable to|cannot|can't|won't|will not|do not|does not|don't|doesn't)"
                                  r"[^.\n]{0,60}(?:sponsor|sponsorship)|without (?:the need for |requiring )?"
                                  r"(?:visa )?sponsorship|no (?:visa )?sponsorship", re.I)),
    ("auth_required", re.compile(r"(?:authori[sz]ed|authori[sz]ation|eligible|eligibility|right|legally "
                                 r"(?:authori[sz]ed|permitted|entitled))\s+to\s+work\s+in\s+(?:the\s+)?"
                                 r"([A-Z][A-Za-z. ]{1,30})", re.I)),
    ("sponsorship_offered", re.compile(r"(?:we (?:do |can |will )?(?:offer|provide|support)|offers?|provides?)"
                                       r"[^.\n]{0,40}(?:visa )?sponsorship|visa sponsorship (?:is )?available|"
                                       r"sponsor(?:s|ing)? (?:work )?visas?|relocation and visa", re.I)),
]


def work_auth_signals(text: str) -> dict[str, str]:
    """Phrases about work authorisation, each with the line it came from."""
    out: dict[str, str] = {}
    for name, rx in WORK_AUTH_PATTERNS:
        m = rx.search(text or "")
        if m:
            ls = text.rfind("\n", 0, m.start()) + 1
            le = text.find("\n", m.end())
            out[name] = text[ls: le if le != -1 else len(text)].strip(" -•*\t")[:300]
    return out
