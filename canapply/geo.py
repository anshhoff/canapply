"""Tiny gazetteer: turn a board's location string into ISO-2 country codes.

Deliberately small and explicit. Anything it cannot place stays unknown and is left to the LLM stage.
"""
import re

COUNTRY_NAMES = {
    "united states": "US", "usa": "US", "u.s.": "US", "us": "US", "united states of america": "US", "america": "US",
    "canada": "CA", "united kingdom": "GB", "uk": "GB", "england": "GB", "scotland": "GB", "ireland": "IE",
    "india": "IN", "germany": "DE", "france": "FR", "netherlands": "NL", "spain": "ES", "portugal": "PT",
    "italy": "IT", "sweden": "SE", "denmark": "DK", "norway": "NO", "finland": "FI", "poland": "PL",
    "switzerland": "CH", "austria": "AT", "belgium": "BE", "czech republic": "CZ", "czechia": "CZ",
    "romania": "RO", "hungary": "HU", "greece": "GR", "estonia": "EE", "lithuania": "LT", "latvia": "LV",
    "ukraine": "UA", "serbia": "RS", "bulgaria": "BG", "croatia": "HR", "turkey": "TR", "israel": "IL",
    "united arab emirates": "AE", "uae": "AE", "saudi arabia": "SA", "qatar": "QA", "egypt": "EG",
    "singapore": "SG", "japan": "JP", "south korea": "KR", "korea": "KR", "china": "CN", "hong kong": "HK",
    "taiwan": "TW", "australia": "AU", "new zealand": "NZ", "indonesia": "ID", "malaysia": "MY",
    "thailand": "TH", "vietnam": "VN", "philippines": "PH", "pakistan": "PK", "bangladesh": "BD",
    "sri lanka": "LK", "brazil": "BR", "mexico": "MX", "argentina": "AR", "colombia": "CO", "chile": "CL",
    "south africa": "ZA", "nigeria": "NG", "kenya": "KE", "luxembourg": "LU",
}

CITIES = {
    # India
    "bengaluru": "IN", "bangalore": "IN", "mumbai": "IN", "delhi": "IN", "new delhi": "IN", "gurugram": "IN",
    "gurgaon": "IN", "noida": "IN", "hyderabad": "IN", "pune": "IN", "chennai": "IN", "kolkata": "IN",
    "ahmedabad": "IN", "jaipur": "IN", "kochi": "IN", "indore": "IN", "chandigarh": "IN", "thiruvananthapuram": "IN",
    # US
    "san francisco": "US", "sf": "US", "new york": "US", "nyc": "US", "new york city": "US", "seattle": "US",
    "austin": "US", "boston": "US", "los angeles": "US", "chicago": "US", "denver": "US", "atlanta": "US",
    "palo alto": "US", "mountain view": "US", "sunnyvale": "US", "san jose": "US", "menlo park": "US",
    "washington": "US", "washington dc": "US", "miami": "US", "san diego": "US", "bellevue": "US",
    "pittsburgh": "US", "portland": "US", "dallas": "US", "houston": "US", "phoenix": "US", "redwood city": "US",
    "san mateo": "US", "santa clara": "US", "cupertino": "US", "brooklyn": "US", "salt lake city": "US",
    "raleigh": "US", "philadelphia": "US", "bay area": "US", "south san francisco": "US", "oakland": "US",
    "cambridge, ma": "US", "irvine": "US", "nashville": "US", "minneapolis": "US", "detroit": "US",
    # Europe / UK
    "london": "GB", "manchester": "GB", "edinburgh": "GB", "cambridge": "GB", "oxford": "GB", "bristol": "GB",
    "dublin": "IE", "berlin": "DE", "munich": "DE", "hamburg": "DE", "frankfurt": "DE", "cologne": "DE",
    "paris": "FR", "amsterdam": "NL", "rotterdam": "NL", "madrid": "ES", "barcelona": "ES", "lisbon": "PT",
    "porto": "PT", "milan": "IT", "rome": "IT", "stockholm": "SE", "copenhagen": "DK", "oslo": "NO",
    "helsinki": "FI", "warsaw": "PL", "krakow": "PL", "zurich": "CH", "geneva": "CH", "vienna": "AT",
    "brussels": "BE", "prague": "CZ", "bucharest": "RO", "budapest": "HU", "athens": "GR", "tallinn": "EE",
    "vilnius": "LT", "belgrade": "RS", "sofia": "BG", "istanbul": "TR", "tel aviv": "IL",
    # Rest of world
    "toronto": "CA", "vancouver": "CA", "montreal": "CA", "waterloo": "CA", "ottawa": "CA", "calgary": "CA",
    "dubai": "AE", "abu dhabi": "AE", "riyadh": "SA", "doha": "QA", "tokyo": "JP", "osaka": "JP",
    "seoul": "KR", "beijing": "CN", "shanghai": "CN", "shenzhen": "CN", "taipei": "TW", "sydney": "AU",
    "melbourne": "AU", "brisbane": "AU", "auckland": "NZ", "jakarta": "ID", "kuala lumpur": "MY",
    "bangkok": "TH", "ho chi minh city": "VN", "hanoi": "VN", "manila": "PH", "sao paulo": "BR",
    "são paulo": "BR", "mexico city": "MX", "buenos aires": "AR", "bogota": "CO", "bogotá": "CO",
    "santiago": "CL", "cape town": "ZA", "johannesburg": "ZA", "lagos": "NG", "nairobi": "KE", "cairo": "EG",
}

US_STATES = set("AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM "
                "NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split())
CA_PROVINCES = {"ON", "BC", "QC", "AB"}

REGIONS = {  # region words -> a marker, not a country list; handled by eligibility as "region restricted"
    "emea": "EMEA", "europe": "EUROPE", "eu": "EUROPE", "apac": "APAC", "asia": "APAC", "latam": "LATAM",
    "latin america": "LATAM", "north america": "NA", "namer": "NA", "americas": "AMERICAS", "amer": "AMERICAS",
    "middle east": "MENA", "mena": "MENA", "africa": "AFRICA",
}

_SPLIT = re.compile(r"\s*(?:\||;|/|\bor\b|\band\b|•|·|\n)\s*", re.I)


def parse_location(raw: str | None) -> tuple[list[str], list[str], bool]:
    """Return (countries, regions, says_remote) for a location string like 'Remote - US' or 'Bengaluru, India'."""
    if not raw:
        return [], [], False
    s = raw.strip()
    low = s.lower()
    says_remote = bool(re.search(r"\b(remote|anywhere|work from home|wfh|distributed)\b", low))
    countries: list[str] = []
    regions: list[str] = []

    def add(code: str) -> None:
        if code not in countries:
            countries.append(code)

    for part in _SPLIT.split(s):
        p = part.strip()
        if not p:
            continue
        pl = re.sub(r"[()\[\]]", " ", p.lower())
        pl = re.sub(r"\b(remote|hybrid|on-?site|in office|office|hq|headquarters|greater|area|metro|based)\b", " ", pl)
        pl = re.sub(r"\s+", " ", pl).strip(" ,-:")
        if not pl:
            continue
        # region words
        for name, code in REGIONS.items():
            if re.search(rf"\b{re.escape(name)}\b", pl) and code not in regions:
                regions.append(code)
        tokens = [t.strip(" .-:") for t in re.split(r",| - | – ", pl) if t.strip(" .-:")]
        raw_tokens = [t.strip(" .-:()") for t in re.split(r",| - | – ", p) if t.strip(" .-:()")]
        hit = False
        # country name anywhere in the tokens wins
        for t in tokens:
            if t in COUNTRY_NAMES and not (t == "us" and len(tokens) == 1 and "us" not in low.split()):
                add(COUNTRY_NAMES[t]); hit = True
        if not hit:
            for t in tokens:
                if t in CITIES:
                    add(CITIES[t]); hit = True; break
        if not hit:
            for t in raw_tokens[1:]:                 # 'Austin, TX' style
                tt = t.strip().upper()
                if tt in US_STATES:
                    add("US"); hit = True; break
                if tt in CA_PROVINCES:
                    add("CA"); hit = True; break
        if not hit:
            for name, code in COUNTRY_NAMES.items():
                if len(name) > 3 and re.search(rf"\b{re.escape(name)}\b", pl):
                    add(code); hit = True
            if not hit:
                for name, code in CITIES.items():
                    if len(name) > 3 and re.search(rf"\b{re.escape(name)}\b", pl):
                        add(code); break
    return countries, regions, says_remote
