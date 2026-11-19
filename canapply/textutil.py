"""HTML -> clean text, hashing, and whitespace-insensitive quote matching."""
import hashlib
import html
import re

from selectolax.lexbor import LexborHTMLParser as HTMLParser

_BLOCK = {"p", "div", "br", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section"}


def html_to_text(raw: str | None) -> str:
    if not raw:
        return ""
    s = raw
    if "&lt;" in s and "<" not in s[:200]:      # Greenhouse sends escaped HTML
        s = html.unescape(s)
    tree = HTMLParser(s)
    for n in tree.css("script,style"):
        n.decompose()
    for n in tree.css("li"):
        n.insert_before("\n- ")
    for tag in _BLOCK:
        for n in tree.css(tag):
            n.insert_after("\n")
    text = tree.body.text(separator="") if tree.body else tree.text(separator="")
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_hash(*parts: str | None) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def norm(s: str) -> str:
    """Lowercase, straighten quotes/dashes, collapse whitespace. Used to compare a quote with its source."""
    s = s.lower()
    s = s.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\xa0": " "}))
    return re.sub(r"\s+", " ", s).strip()


def quote_in_text(quote: str | None, text: str) -> bool:
    """True only if the quote appears verbatim in the text (ignoring case, whitespace and bullet marks)."""
    if not quote or len(quote.strip()) < 6:
        return False
    q = norm(quote).strip(" .-*•\"'")
    if not q:
        return False
    return q in norm(text)
