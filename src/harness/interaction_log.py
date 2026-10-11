"""Log EVERY model interaction (prompt + response + timing/stats) for debugging.

A standalone module (no deps on lab/repo) so any generator — the chat brain (`lab`), repo Q&A
(`repo`), vision — can log without an import cycle. One JSON record per line in
logs/model-interactions.jsonl. On by default; disable with HARNESS_LOG_INTERACTIONS=0.

Read it back:  tail -f logs/model-interactions.jsonl        (live)
               python -m json.tool < logs/model-interactions.jsonl | less
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime

_lock = threading.Lock()


def log_interaction(kind: str, messages, response: str, meta: dict | None = None) -> None:
    """Append one interaction. Best-effort — never raises into the generation path."""
    if os.environ.get("HARNESS_LOG_INTERACTIONS", "1") == "0":
        return
    try:
        from .download import PROJECT_ROOT

        logdir = PROJECT_ROOT / "logs"
        logdir.mkdir(exist_ok=True)
        rec = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "kind": kind,
            "messages": messages,
            "response": response,
        }
        if meta:
            rec.update(meta)
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with _lock:
            with open(logdir / "model-interactions.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass
