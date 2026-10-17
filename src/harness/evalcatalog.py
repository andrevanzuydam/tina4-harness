"""Render the harness's eval suites as a human-readable catalog — "here's how we prove it works".

The harness has two eval suites, and together they ARE the specification of what it can do:

  * tests/test_scenarios.py  — LIVE outcome-evals. Each one hits the real local model and checks the
    OUTCOME (valid runnable code / correct behaviour / right render mode), not exact wording, so it
    survives model variance. This is the "what can a user actually get done offline" gate.
  * tests/test_chat_logic.py — DETERMINISTIC logic-evals. No model, no network, no mocks: real exec,
    real temp images, a real BM25 index. These pin the routing / fix-validation / grounding helpers.

This module reads those files and emits a catalog grouped by their `# --- section ---` banners, with
one plain-English line per eval (from its leading comment or docstring) plus the key assertion. It
parses the source (it does not run it), so the catalog can never drift from the tests it describes.

    python -m harness.evalcatalog            # writes reports/evals.md
    python -m harness.evalcatalog -o x.md
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

_BANNER = re.compile(r"#\s*-+\s*(.*?)\s*-+\s*$")

_SUITES = [
    ("tests/test_scenarios.py", "Live outcome-evals",
     "Each eval hits the **real local model** and checks the outcome — runnable code, correct "
     "behaviour, the right answer mode — never exact wording. This is the gate for *what a user can "
     "actually get done offline*."),
    ("tests/test_chat_logic.py", "Deterministic logic-evals",
     "No model, no network, **no mocks**: real `exec`, real temp images, a real BM25 index. These "
     "pin the deterministic helpers — routing, fix-validation, assertions, grounding — that decide "
     "how the harness behaves before the model is ever called."),
]


def _leading_note(src_lines: list[str], node: ast.FunctionDef) -> str:
    """One plain-English line for a test: its docstring, else the comment just above the assertion."""
    doc = ast.get_docstring(node)
    if doc:
        return re.sub(r"\s+", " ", doc.strip().split("\n\n")[0])
    # else: first inline comment inside the body, often the "# this proves X" line
    for ln in src_lines[node.lineno: getattr(node, "end_lineno", node.lineno + 20)]:
        s = ln.strip()
        if s.startswith("#") and len(s) > 3:
            return s.lstrip("# ").strip()
    return ""


def _first_assert(node: ast.FunctionDef, src: str) -> str:
    for n in ast.walk(node):
        if isinstance(n, ast.Assert):
            seg = ast.get_source_segment(src, n.test)
            if seg:
                return re.sub(r"\s+", " ", seg.strip())
    return ""


def _title(fn_name: str) -> str:
    return fn_name[len("test_"):].replace("_", " ").strip().capitalize() if fn_name.startswith("test_") else fn_name


def build_catalog(root: Path) -> str:
    out: list[str] = []
    out.append("# Harness evals — how we prove it works")
    out.append("")
    out.append("_Generated from the test sources; it parses them, so it can't drift from the suites._")
    out.append("")
    out.append("The harness is a small **local** model (Qwen3.8-2B-Distill or Gemma-4-E2B, 4-bit MLX, "
               "fully offline) wrapped in a self-checking harness. These two suites are the spec of "
               "what that combination can do, and the proof it still does it.")
    out.append("")

    total = 0
    for rel, heading, blurb in _SUITES:
        path = root / rel
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        src_lines = src.splitlines()
        tree = ast.parse(src)
        tests = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        total += len(tests)

        out.append(f"## {heading}  ·  `{rel}`")
        out.append("")
        out.append(blurb)
        out.append("")
        out.append(f"**{len(tests)} evals.**")
        out.append("")

        # Map each test to the nearest preceding `# --- section ---` banner.
        banners: dict[int, str] = {}
        for i, ln in enumerate(src_lines, 1):
            m = _BANNER.match(ln.strip())
            if m and m.group(1):
                banners[i] = m.group(1)

        def section_for(lineno: int) -> str:
            best = ""
            for bl, name in banners.items():
                if bl < lineno:
                    best = name
            return best or "General"

        current = None
        for t in tests:
            sec = section_for(t.lineno)
            if sec != current:
                current = sec
                out.append(f"### {sec}")
                out.append("")
            note = _leading_note(src_lines, t)
            assertion = _first_assert(t, src)
            out.append(f"- **{_title(t.name)}** — {note}" if note else f"- **{_title(t.name)}**")
            if assertion:
                out.append(f"  `{assertion[:140]}`")
        out.append("")

    out.insert(4, f"**{total} evals across {len(_SUITES)} suites.**  Run them with:")
    out.insert(5, "")
    out.insert(6, "```bash\nPYTHONPATH=src .venv-notebook/bin/python -m pytest tests/ -q\n```")
    out.insert(7, "")
    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str]) -> int:
    from .download import PROJECT_ROOT

    out = PROJECT_ROOT / "reports" / "evals.md"
    args = list(argv)
    for i, a in enumerate(args):
        if a in ("-o", "--out"):
            out = Path(args[i + 1])
    report = build_catalog(PROJECT_ROOT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
