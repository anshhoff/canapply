"""Fetch + normalise postings from the three public job-board APIs. No scraping, no logins."""
from datetime import datetime, timezone

import httpx

from ..textutil import html_to_text


class BoardError(Exception):
    pass


def _get(client: httpx.Client, url: str):
    r = client.get(url)
    if r.status_code == 404:
        raise BoardError("http_404")
    if r.status_code != 200:
        raise BoardError(f"http_{r.status_code}")
    try:
        return r.json()
    except Exception as e:  # noqa: BLE001
        raise BoardError("bad_json") from e


def _ts(s) -> datetime | None:
    if not s:
        return None
    try:
        if isinstance(s, (int, float)):
            return datetime.fromtimestamp(s / 1000, tz=timezone.utc)
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:  # noqa: BLE001
        return None


def greenhouse(client: httpx.Client, token: str) -> tuple[str | None, list[dict]]:
    d = _get(client, f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true")
    out, company = [], None
    for j in d.get("jobs", []):
        company = j.get("company_name") or company
        meta = {m.get("name"): m.get("value") for m in (j.get("metadata") or []) if isinstance(m, dict)}
        workplace = next((str(v) for k, v in meta.items() if k and "location type" in k.lower() and v), None)
        depts = [x.get("name") for x in (j.get("departments") or []) if x.get("name")]
        emp = next((str(v) for k, v in meta.items() if k and "employment" in k.lower() and v), None)
        out.append(dict(
            ext_id=str(j["id"]), title=(j.get("title") or "").strip(), url=j.get("absolute_url") or "",
            location_raw=((j.get("location") or {}).get("name") or "").strip() or None,
            workplace_raw=workplace, department=", ".join(depts) or None, employment_raw=emp,
            description=html_to_text(j.get("content")),
            posted_at=_ts(j.get("first_published") or j.get("updated_at")),
        ))
    return company, out


def ashby(client: httpx.Client, token: str) -> tuple[str | None, list[dict]]:
    d = _get(client, f"https://api.ashbyhq.com/posting-api/job-board/{token}?includeCompensation=true")
    out = []
    for j in d.get("jobs", []):
        if j.get("isListed") is False:
            continue
        locs = [j.get("location")] + [s.get("location") for s in (j.get("secondaryLocations") or [])]
        country = (((j.get("address") or {}).get("postalAddress") or {}).get("addressCountry"))
        loc = " | ".join(x for x in locs if x)
        if country and country.lower() not in loc.lower():
            loc = f"{loc}, {country}" if loc else country
        workplace = j.get("workplaceType") or ("Remote" if j.get("isRemote") else None)
        out.append(dict(
            ext_id=str(j["id"]), title=(j.get("title") or "").strip(), url=j.get("jobUrl") or "",
            location_raw=loc or None, workplace_raw=workplace,
            department=j.get("department") or j.get("team"), employment_raw=j.get("employmentType"),
            description=(j.get("descriptionPlain") or html_to_text(j.get("descriptionHtml"))).strip(),
            posted_at=_ts(j.get("publishedAt")),
        ))
    return None, out


def lever(client: httpx.Client, token: str) -> tuple[str | None, list[dict]]:
    d = _get(client, f"https://api.lever.co/v0/postings/{token}?mode=json")
    if not isinstance(d, list):
        raise BoardError("bad_json")
    out = []
    for j in d:
        cat = j.get("categories") or {}
        parts = [j.get("descriptionPlain") or html_to_text(j.get("description"))]
        for lst in j.get("lists") or []:
            parts.append(f"{lst.get('text', '')}\n{html_to_text(lst.get('content'))}")
        parts.append(j.get("additionalPlain") or html_to_text(j.get("additional")))
        locs = cat.get("allLocations") or ([cat.get("location")] if cat.get("location") else [])
        out.append(dict(
            ext_id=str(j["id"]), title=(j.get("text") or "").strip(), url=j.get("hostedUrl") or "",
            location_raw=" | ".join(x for x in locs if x) or None, workplace_raw=j.get("workplaceType"),
            department=cat.get("department") or cat.get("team"), employment_raw=cat.get("commitment"),
            description="\n\n".join(p.strip() for p in parts if p and p.strip()),
            posted_at=_ts(j.get("createdAt")),
        ))
    return None, out


FETCHERS = {"greenhouse": greenhouse, "ashby": ashby, "lever": lever}
