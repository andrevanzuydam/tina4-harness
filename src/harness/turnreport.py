"""Turn the raw interaction log into a human-readable report — with the SYSTEM PROMPT of each turn.

`interaction_log.py` writes one JSON record per model call to logs/model-interactions.jsonl. That
file is for machines (tail -f, jq). This module turns it into something a person reads: it pulls the
**system prompt** out of every turn, de-duplicates them (the harness reuses a handful of system
prompts across many turns, so printing all ~5 KB each time would be noise), and lays the turns out
as a table that says — per turn — which system prompt drove it, what the user asked, what mode the
harness answered in, and how fast.

    python -m harness.turnreport                         # reads logs/model-interactions.jsonl
    python -m harness.turnreport logs/x.jsonl -o out.md  # explicit in/out

No third-party deps (stdlib only), so it runs anywhere the harness does.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

_THINK = [
    re.compile(r"<think>.*?</think>", re.S),                         # Qwen
    re.compile(r"<\|channel>thought.*?<channel\|>", re.S),           # Gemma (closed)
]
_SENTINEL = re.compile(r"@@HARNESS_([A-Z]+)@@")
_HSTATUS = re.compile(r"__H(?:STATUS|STREAM)__\S*")


def load_turns(path: str | Path) -> list[dict]:
    """Read the JSONL log into a list of turn records (skips blank/corrupt lines)."""
    turns = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            turns.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return turns


def _system_of(turn: dict) -> str:
    for m in turn.get("messages", []) or []:
        if isinstance(m, dict) and m.get("role") == "system":
            return str(m.get("content", "")).strip()
    return ""


def _user_of(turn: dict) -> str:
    """The last user message — what the person actually asked on this turn."""
    users = [m for m in (turn.get("messages") or [])
             if isinstance(m, dict) and m.get("role") == "user"]
    return str(users[-1].get("content", "")).strip() if users else ""


def _form_of(turn: dict) -> str:
    """The OUTPUT FORM of the reply — code vs prose — judged from the model's raw text.

    (The log stores the model's raw output, before the harness adds its @@HARNESS_x@@ routing
    sentinels, so form is read from the text itself: a reply that opens with a code fence is a code
    reply, otherwise it's prose.)"""
    text = _strip_think(turn.get("response", "") or "").lstrip()
    return "code" if text.startswith("```") or re.match(r"(?s)^(?:def |class |import |from )", text) else "prose"


def _strip_think(text: str) -> str:
    for rx in _THINK:
        text = rx.sub("", text)
    # An unclosed Gemma thought block (hit the token cap mid-think) — drop the dangling opener.
    text = re.sub(r"<\|channel>thought.*\Z", "«reasoning truncated at token cap»", text, flags=re.S)
    text = _SENTINEL.sub("", text)
    text = _HSTATUS.sub("", text)
    return text.strip()


def _oneline(text: str, limit: int = 90) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _esc(text: str) -> str:
    return text.replace("|", "\\|")


def build_report(turns: list[dict], title: str = "Harness turn log") -> str:
    # De-duplicate system prompts → stable labels S1, S2, … in first-seen order.
    labels: dict[str, str] = {}
    label_kinds: dict[str, set] = {}
    for t in turns:
        sysp = _system_of(t)
        if sysp and sysp not in labels:
            labels[sysp] = f"S{len(labels) + 1}"
        if sysp:
            label_kinds.setdefault(labels[sysp], set()).add(t.get("kind", "?"))

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Generated {datetime.now().astimezone().isoformat(timespec='seconds')} "
                 f"from {len(turns)} logged model turns._")
    lines.append("")
    lines.append(f"Every turn is one real call to the local model. The harness drives it with one of "
                 f"**{len(labels)} distinct system prompts** (shown in full below); the table maps each "
                 f"turn to the system prompt that shaped it, what was asked, the mode the harness "
                 f"answered in, and the timing.")
    lines.append("")

    # --- timing / throughput summary ---
    secs = [t.get("seconds") for t in turns if isinstance(t.get("seconds"), (int, float))]
    tps = [t.get("stats", {}).get("tokens_per_second") for t in turns
           if isinstance(t.get("stats"), dict) and t["stats"].get("tokens_per_second")]
    if secs:
        lines.append("## At a glance")
        lines.append("")
        lines.append(f"- **Turns:** {len(turns)}")
        lines.append(f"- **Total model time:** {sum(secs):.1f}s "
                     f"(avg {sum(secs)/len(secs):.2f}s/turn, slowest {max(secs):.2f}s)")
        if tps:
            lines.append(f"- **Throughput:** {sum(tps)/len(tps):.0f} tokens/s average")
        kinds = {}
        for t in turns:
            kinds[t.get("kind", "?")] = kinds.get(t.get("kind", "?"), 0) + 1
        lines.append("- **Kinds:** " + ", ".join(f"{k} ×{v}" for k, v in sorted(kinds.items())))
        lines.append("")

    # --- per-turn table ---
    # Token stats are only shown when the log actually captured them (the streaming path records
    # them; the non-streaming /generate path currently reports zeros — don't print fake numbers).
    has_tokens = any((t.get("stats") or {}).get("generation_tokens") for t in turns)
    lines.append("## Turns")
    lines.append("")
    tok_head = " Gen tok | tok/s |" if has_tokens else ""
    tok_sep = "--------:|------:|" if has_tokens else ""
    lines.append(f"| # | Time | Kind | System | Asked | Form | Secs |{tok_head}")
    lines.append(f"|--:|------|------|:------:|-------|:----:|-----:|{tok_sep}")
    for i, t in enumerate(turns, 1):
        label = labels.get(_system_of(t), "—")
        stats = t.get("stats") or {}
        ts = (t.get("ts", "") or "")[11:19]  # HH:MM:SS
        secs_v = t.get("seconds")
        secs_cell = f"{secs_v:.2f}" if isinstance(secs_v, (int, float)) else ""
        tok_cell = (f" {stats.get('generation_tokens','')} | {stats.get('tokens_per_second','')} |"
                    if has_tokens else "")
        lines.append(f"| {i} | {ts} | {t.get('kind','?')} | {label} | "
                     f"{_esc(_oneline(_user_of(t), 60))} | {_form_of(t)} | {secs_cell} |{tok_cell}")
    lines.append("")
    if not has_tokens:
        lines.append("> _Per-call token counts aren't shown: the non-streaming `/generate` path the "
                     "suite uses doesn't yet report them back to the log (a known logging gap). "
                     "Wall-clock seconds are real._")
        lines.append("")

    # --- the system prompts, in full ---
    lines.append("## System prompts (in full)")
    lines.append("")
    lines.append("These are the instructions the harness prepends to the model on each turn — the "
                 "'operating manual' it hands the model before the user's message. This is where the "
                 "harness's behaviour actually lives.")
    lines.append("")
    for sysp, label in labels.items():
        used = ", ".join(sorted(label_kinds.get(label, set())))
        lines.append(f"### {label} — used by: {used} ({len(sysp)} chars)")
        lines.append("")
        lines.append("```text")
        lines.append(sysp)
        lines.append("```")
        lines.append("")

    # --- a couple of worked turns (prompt → stripped answer) ---
    lines.append("## Sample turns (what the model actually returned)")
    lines.append("")
    lines.append("A few turns end to end, with the model's `<think>`/`thought` reasoning stripped the "
                 "way the chat UI strips it before display.")
    lines.append("")
    for i, t in enumerate(turns, 1):
        asked = _user_of(t)
        if not asked:
            continue
        answer = _strip_think(t.get("response", "") or "")
        lines.append(f"**Turn {i} · {t.get('kind','?')} · {_form_of(t)} reply** — system `{labels.get(_system_of(t),'—')}`")
        lines.append("")
        lines.append(f"> **Asked:** {_oneline(asked, 160)}")
        lines.append("")
        lines.append("> **Answered:**")
        lines.append("")
        for ln in _oneline(answer, 600).splitlines() or [""]:
            lines.append(f"> {ln}")
        lines.append("")
        if i >= 6:  # first six is enough for a readable sample
            break

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str]) -> int:
    from .download import PROJECT_ROOT

    inp = PROJECT_ROOT / "logs" / "model-interactions.jsonl"
    out = None
    args = list(argv)
    for i, a in enumerate(args):
        if a in ("-o", "--out"):
            out = Path(args[i + 1])
        elif not a.startswith("-") and (i == 0 or args[i - 1] not in ("-o", "--out")):
            inp = Path(a)
    turns = load_turns(inp)
    report = build_report(turns)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"wrote {out} ({len(turns)} turns)")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
