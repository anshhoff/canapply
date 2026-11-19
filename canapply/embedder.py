"""Local embeddings (BAAI/bge-small-en-v1.5, 384 dims, runs on CPU through ONNX).

Why local and not an API: the index is ~16k postings and changes every few hours. A free embedding API
allowed ~45 postings a minute; this model does the whole index in minutes on a laptop, costs nothing,
and search does not stop working when an API quota runs out.
"""
import re
import threading

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
_model = None
_lock = threading.Lock()


def model():
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(MODEL_NAME, threads=4)
        return _model


def embed_docs(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    return [v.tolist() for v in model().embed(texts, batch_size=batch_size)]


def embed_query(text: str) -> list[float]:
    """Queries get the model's retrieval prefix (fastembed adds it in query_embed)."""
    return next(iter(model().query_embed([text]))).tolist()


_ROLE_START = re.compile(
    r"(about the (?:role|job|position|team|opportunity)|the (?:role|opportunity|job)\b|what you(?:'|’)?ll (?:do|be doing)|"
    r"what you will do|responsibilities|your (?:role|impact|mission)|in this role|role overview|job description|"
    r"position (?:summary|overview)|you will\b|what we(?:'|’)?re looking for|requirements|qualifications|who you are)",
    re.I)


def role_section(description: str, max_chars: int = 1400) -> str:
    """Postings usually open with a company blurb that is identical across all of that company's jobs.
    Embedding it makes every job at a company look alike, so start from the first role heading instead."""
    d = description or ""
    m = _ROLE_START.search(d, 150) if len(d) > 600 else None
    start = m.start() if m and m.start() < len(d) * 0.6 else 0
    return d[start:start + max_chars]


def doc_text(job: dict) -> str:
    return (f"{job['title']} at {job['company']}. {job.get('department') or ''}. "
            f"Location: {job.get('location_raw') or 'unknown'}.\n{role_section(job['description'])}")


def unit(v) -> list[float]:
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return (a / n).tolist() if n else a.tolist()
