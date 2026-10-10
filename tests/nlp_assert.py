"""NLP-assisted test assertions — robust "is this reply on-topic?" checks without brittle substrings.

The chat's prose is nondeterministic, so asserting an exact word (`"greet" in out`) is fragile. This
uses the BM25 index the harness already ships (Groundwire) to score the reply against an EXPECTED
topic vs a DECOY topic — deterministic, offline, no new dependency. `about(text, expected, decoy)`
is true when the reply is lexically closer to the expected topic than the decoy.
"""
from __future__ import annotations

import re


def topic_score(text: str, query: str) -> float:
    """Max BM25 score of `query` against the sentences of `text` (0.0 when nothing matches)."""
    from groundwire import Groundwire

    gw = Groundwire(memory="bm25", k=5, max_words=40)
    ingested = 0
    for i, s in enumerate(re.split(r"(?<=[.!?])\s+|\n+", text)):
        if s.strip():
            gw.ingest(s.strip(), title=str(i))
            ingested += 1
    if not ingested:
        return 0.0
    try:
        hits = gw.retrieve(query, k=3)
    except Exception:
        return 0.0
    return max((score for _cid, _text, score in hits), default=0.0)


def about(text: str, expected: str, decoy: str) -> bool:
    """True when `text` is more about `expected` than about `decoy` (both are term strings)."""
    return topic_score(text, expected) > topic_score(text, decoy)
