"""Benchmark a model behind the harness on the jobs a real offline user relies on.

Runs a fixed battery of tasks through the SAME harness brain (lab._chat / vision), against whatever
model server HARNESS_MODEL_SERVER points at, and scores the OUTCOME (does the code run / behave /
render in the right mode / stay on-topic) — not the wording. Also records end-to-end latency per
task (what the user feels) and the server's peak memory. So two models are compared apples-to-apples:
same prompts, same routing, same checks — only the weights differ.

Usage:
    HARNESS_MODEL_SERVER=http://127.0.0.1:7777 PYTHONPATH=src \\
        .venv-notebook/bin/python scripts/benchmark.py <label> [--json out.json]

Prints a per-task table and a summary; with --json also writes the raw results.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))  # for nlp_assert

from harness import lab  # noqa: E402
from harness.backends.remote_backend import reachable  # noqa: E402

try:
    from nlp_assert import about  # BM25 topic gate (robust, offline)
except Exception:  # pragma: no cover
    about = None

GOOD = 'def greet(name):\n    return "Hello " + name'
BUGGY = ("def total(items):\n    t = 0\n    for i in items:\n"
         "        t += i['price'] * i['qty']\n    return T")
LOOP = "squares = []\nfor i in range(10):\n    squares.append(i * i)\nprint(squares)"
GREET = 'def greet(name: str) -> str:\n    """Greet someone."""\n    return f"Hello, {name}!"'
PYIMG = str(ROOT / "notebooks" / "python.png")


def _mode(raw: str) -> str:
    return ("code" if raw.startswith("@@HARNESS_CODE@@")
            else "ask" if raw.startswith("@@HARNESS_ASK@@")
            else "fix" if raw.startswith("@@HARNESS_FIX@@")
            else "other")


def _clean(raw: str) -> str:
    return re.sub(r"__H(STATUS|STREAM)__\S*|@@HARNESS_(ASK|CODE|FIX|CHOICES)@@\S*\n?", "", raw)


def _code(out: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", out, re.S)
    return blocks[-1].strip() if blocks else ""


def _blocks(out: str) -> list[str]:
    return [b.strip() for b in re.findall(r"```(?:python)?\s*\n(.*?)```", out, re.S)]


def _valid(code: str) -> bool:
    return bool(code) and lab._valid_python(code)[0]


# --- the battery: (name, run, check) -----------------------------------------------------------
# run() -> (mode, text, extra); check(mode, text, extra) -> (ok: bool, note: str)

def _chat(msg, **kw):
    kw.setdefault("think", "brief")
    raw = lab._chat([{"role": "user", "content": msg}], **kw)
    return _mode(raw), _clean(raw)


def t_fix_error():
    mode, out = _chat("fix this", context=BUGGY, output="NameError: name 'T' is not defined",
                      intent="code")
    c = _code(out)
    ok = _valid(c) and not re.search(r"return\s+T\b", c)
    return ok, "runnable fix, undefined name gone" if ok else "fix invalid or still uses T"


def t_write_function():
    mode, out = _chat("write a function that counts the words in a sentence", intent="code")
    c = _code(out)
    if not (mode == "code" and _valid(c)):
        return False, "no valid code / wrong mode"
    m = re.search(r"def\s+(\w+)\s*\(", c)
    if not m:
        return False, "no function defined"
    r = lab._run_in_kernel(c + f'\nprint({m.group(1)}("one two three four"))')
    ok = "4" in (r.get("out", "") + str(r.get("value")))
    return ok, "counts words correctly" if ok else "ran but wrong count"


def t_explain_concept():
    mode, out = _chat("what is a decorator?", intent="ask")
    ok = mode == "ask" and "decorator" in out.lower() and len(out) < 1600
    return ok, f"ask, on-topic, {len(out)} chars" if ok else f"mode={mode}, len={len(out)}"


def t_explain_code():
    mode, out = _chat("what does this do?", context=GOOD, intent="ask")
    ok = mode == "ask" and bool(re.search(r"greet|hello", out, re.I))
    return ok, "explains the greet function" if ok else f"mode={mode}, off-topic"


def t_add_comments():
    mode, out = _chat("add comments to this", context=GOOD, intent="code")
    c = _code(out)
    documented = "#" in c or '"""' in c or "'''" in c  # a # comment OR a docstring both count
    ok = _valid(c) and "def greet" in c and documented
    return ok, "documented, still valid" if ok else "lost function or no comment/docstring"


def t_add_type_hints():
    mode, out = _chat("add type hints to this function", context=GOOD, intent="code")
    c = _code(out)
    ok = _valid(c) and "def greet" in c and "->" in c
    return ok, "typed, still valid" if ok else "no hints / invalid"


def t_remove_print():
    mode, out = _chat("remove the print statement", context=LOOP, intent="code")
    c = _code(out)
    ok = _valid(c) and not re.search(r"\bprint\s*\(", c) and "squares" in c
    return ok, "print gone, logic kept" if ok else "print left or logic lost"


def t_convert_loop():
    mode, out = _chat("convert this loop to a list comprehension", context=LOOP, intent="code")
    c = _code(out)
    ok = _valid(c) and bool(re.search(r"\[.*for .* in .*\]", c))
    return ok, "comprehension produced" if ok else "no comprehension"


def t_call_function():
    mode, out = _chat("call this function with an example value", context=GOOD, intent="code")
    c = _code(out)
    ok = _valid(c) and "greet(" in c
    return ok, "calls greet()" if ok else "no call"


def t_write_test():
    mode, out = _chat("write a test for this function", context=GREET, intent="code")
    c = _code(out)
    ok = _valid(c) and ("assert" in c.lower() or "def test" in c.lower())
    return ok, "assert-based test" if ok else "no test"


def t_multi_cell():
    mode, out = _chat("add 2 cells - one for code and one to test the code, write me a method to "
                      "add 2 numbers together", intent="code")
    bl = _blocks(out)
    if mode != "code" or len(bl) != 2:
        return False, f"mode={mode}, blocks={len(bl)}"
    fn, test = bl[0], bl[1]
    if not (_valid(fn) and "def " in fn and _valid(test) and "assert" in test and "def " not in test):
        return False, "cells malformed (redef or no assert)"
    r = lab._run_in_kernel(fn + "\n" + test)
    ok = not r.get("error")
    return ok, "two cells, test passes" if ok else f"test failed: {r.get('error')}"


def t_vision_read():
    if not Path(PYIMG).exists():
        return None, "no python.png"
    r = lab._vision(PYIMG, "What text is written in this comic? List the exact phrases.")
    low = (r or "").lower()
    hits = sum(k in low for k in ["antigravity", "python", "flying", "hello world", "dynamic typing"])
    ok = hits >= 2
    return ok, f"read {hits}/5 key phrases" if ok else f"only {hits}/5 phrases"


TASKS = [
    ("fix-error-from-traceback", t_fix_error),
    ("write-function-that-runs", t_write_function),
    ("explain-concept-brief", t_explain_concept),
    ("explain-code", t_explain_code),
    ("add-comments", t_add_comments),
    ("add-type-hints", t_add_type_hints),
    ("remove-print", t_remove_print),
    ("convert-loop-to-comprehension", t_convert_loop),
    ("call-function", t_call_function),
    ("write-test", t_write_test),
    ("multi-cell-code+test", t_multi_cell),
    ("vision-read-comic", t_vision_read),
]


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "model"
    out_json = None
    if "--json" in sys.argv:
        out_json = sys.argv[sys.argv.index("--json") + 1]

    server = os.environ.setdefault("HARNESS_MODEL_SERVER", "http://127.0.0.1:7777")
    if not reachable(server):
        print(f"model server {server} is not reachable — start it first", file=sys.stderr)
        return 2

    import urllib.request
    def mem():
        try:
            return json.load(urllib.request.urlopen(server + "/memory", timeout=5))
        except Exception:
            return {}

    info = {}
    try:
        info = json.load(urllib.request.urlopen(server + "/info", timeout=5))
    except Exception:
        pass

    print(f"\n=== benchmark: {label}  ({server})  model={info.get('repo_id','?')} ===", flush=True)
    results = []
    t_all = time.time()
    for name, fn in TASKS:
        t0 = time.time()
        try:
            ok, note = fn()
        except Exception as e:
            ok, note = False, f"EXC {type(e).__name__}: {e}"
        dt = time.time() - t0
        status = "—" if ok is None else ("PASS" if ok else "FAIL")
        results.append({"task": name, "ok": ok, "note": note, "seconds": round(dt, 1)})
        print(f"  [{status:4}] {name:32} {dt:6.1f}s  {note}", flush=True)
    total = time.time() - t_all
    scored = [r for r in results if r["ok"] is not None]
    passed = sum(1 for r in scored if r["ok"])
    m = mem()
    peak = m.get("peak_gb")
    summary = {
        "label": label, "server": server, "model": info.get("repo_id", "?"),
        "passed": passed, "scored": len(scored), "total_seconds": round(total, 1),
        "avg_seconds": round(total / len(results), 1), "peak_gb": peak,
        "weights_gb": m.get("weights_gb"), "results": results,
    }
    print(f"\n  SCORE {passed}/{len(scored)}   total {total:.0f}s   "
          f"avg {summary['avg_seconds']}s/task   peak {peak} GB   weights {m.get('weights_gb')} GB\n")
    if out_json:
        Path(out_json).write_text(json.dumps(summary, indent=2))
        print(f"  wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
