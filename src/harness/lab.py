"""Kernel-side bridge for the JupyterLab extension (tina4-harness-lab).

The right-click menu items (Check Code / Create Cell / Fix Code) call ONE function here —
``harness_edit`` — over the notebook's kernel. It uses the model the notebook already loaded,
and **loads one itself if the kernel doesn't have one yet**, so the menus work on a clean
notebook with nothing but a local LLM.

Concurrency: a single local model (MLX on the Metal GPU, or one torch model) cannot truly run
several generations at once, so requests are submitted to a small thread pool (``HARNESS_THREADS``,
default 4) and serialised on a lock. You can fire several requests; they queue safely rather than
corrupting the model's state. True parallelism would need several model copies.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

# Prompts + protocol sentinels live in chatprompts.py; re-exported here so `lab._CHAT_SYSTEM`,
# `lab._MODE_CODE`, etc. keep working for callers and tests.
from . import chatprompts as _prompts  # situational instruction constants (_prompts.HELP, …)
from .chatprompts import (  # noqa: F401
    _APP_SENTINEL,
    _CHAT_SYSTEM,
    _CHOICES_SENTINEL,
    _FIX_SENTINEL,
    _INSERT_SENTINEL,
    _MODE_ASK,
    _MODE_CODE,
    _NORUN,
    _RUN_SENTINEL,
    _THINK_MODES,
    _WROTE_SENTINEL,
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Some models reason in a different wrapper than Qwen's <think>…</think>: Gemma 4 uses a channel
# format, <|channel>thought … <channel|>answer. Strip either so routing sees the real answer.
_CHANNEL = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)

_lock = threading.Lock()
_model = None
_pool: ThreadPoolExecutor | None = None
_vision_fallback = None


def _status(msg: str) -> None:
    """Emit a live status line the chat shows while it works (read from the kernel's stdout)."""
    try:
        print("__HSTATUS__" + msg, flush=True)
    except Exception:
        pass


def _threads() -> int:
    try:
        return max(1, int(os.environ.get("HARNESS_THREADS", "4")))
    except ValueError:
        return 4


def get_model():
    """Reuse an already-loaded model (demo via agent.setup(), or a teaching notebook's `model`),
    else load one now. Validates it is a real model (has .generate), not e.g. agent.model()."""
    global _model

    # 1) The packaged agent's model (clean demo notebooks call agent.setup()).
    try:
        from . import agent

        if getattr(agent, "_model", None) is not None:
            _model = agent._model
            return _model
    except Exception:
        pass

    # 2) A model instance in the notebook namespace (teaching notebook: `model = load_model()`).
    ip = None
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            existing = ip.user_ns.get("model")
            if existing is not None and hasattr(existing, "generate"):
                _model = existing
                return _model
    except Exception:
        ip = None

    # 3) Load one ourselves and share it back.
    if _model is None:
        from harness import load_model

        _model = load_model()
        try:
            if ip is not None and not hasattr(ip.user_ns.get("model"), "generate"):
                ip.user_ns["harness_model"] = _model
        except Exception:
            pass
    return _model


def _executor() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=_threads(), thread_name_prefix="harness")
    return _pool


def _stream(chunk: str) -> None:
    """Emit a generation delta the chat renders live (base64 so newlines survive)."""
    try:
        print("__HSTREAM__" + base64.b64encode(chunk.encode("utf-8")).decode(), flush=True)
    except Exception:
        pass


def _locked_generate(model, messages, max_tokens, on_text=None, enable_thinking=True, temperature=None):
    with _lock:  # one generation at a time on the single local model
        return model.generate(
            messages, max_tokens=max_tokens, on_text=on_text, enable_thinking=enable_thinking,
            temperature=temperature,
        )


# Set when the last generation was interrupted (Stop button → kernel interrupt), so the chat
# flow can skip its retry instead of launching a second generation.
_stopped = False

from .interaction_log import log_interaction as _log_interaction  # noqa: E402  (logs all generations)


def _generate(messages, max_tokens=700, stream: bool = False, enable_thinking: bool = True,
              temperature=None) -> str:
    """Generate, honouring a mid-run kernel interrupt (Stop button) by ending early.

    The Stop button interrupts the kernel; that raises KeyboardInterrupt in this (main) thread
    while the generation runs in the pool. We ask the model to stop, then collect the partial
    text the worker produced so far.
    """
    global _stopped
    _stopped = False
    model = get_model()
    on_text = _stream if stream else None
    import time as _time

    t0 = _time.time()
    fut = _executor().submit(_locked_generate, model, messages, max_tokens, on_text,
                             enable_thinking, temperature)
    try:
        out = fut.result()
    except KeyboardInterrupt:
        _stopped = True
        try:
            model.request_stop()
        except Exception:
            pass
        try:
            out = fut.result()  # completes quickly now, with whatever was generated
        except Exception:
            out = ""
    stats = getattr(model, "last_stats", None)
    _log_interaction("generate", messages, out, {
        "max_tokens": max_tokens, "enable_thinking": enable_thinking, "temperature": temperature,
        "stream": stream, "stopped": _stopped, "seconds": round(_time.time() - t0, 2),
        "stats": vars(stats) if stats is not None else None,
    })
    return out


def _strip_think(text: str) -> str:
    # Models reason first and close with a marker (the opening tag comes from the chat template, so
    # it is not in the generated text). Keep only what follows the LAST reasoning block, whichever
    # marker the model uses — Qwen's </think> or Gemma's channel close <channel|>.
    cut = -1
    for marker in ("</think>", "<channel|>"):
        i = text.rfind(marker)
        if i != -1:
            cut = max(cut, i + len(marker))
    if cut != -1:
        text = text[cut:]
    text = _THINK.sub("", text)
    text = _CHANNEL.sub("", text)
    # Drop any stray channel tag left when the reasoning ran to the end without a clean close.
    text = re.sub(r"<\|?channel\|?>\w*", "", text)
    return text.replace("<think>", "").strip()


def _code(text: str) -> str:
    text = _strip_think(text)
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip()
    # No clean fenced block — drop any stray ``` fence lines so the cell is pure code.
    lines = [ln for ln in text.strip().splitlines() if not ln.strip().startswith("```")]
    return "\n".join(lines).strip()


# --- the three actions ---------------------------------------------------------------
def _create(prompt: str) -> str:
    messages = [
        {"role": "system", "content":
            "You write ONE Python notebook cell. Output ONLY runnable Python in a single ``` "
            "block, with no prose. Prefer the standard library. Do NOT write files to disk unless "
            "the request explicitly asks for it."},
        {"role": "user", "content": prompt},
    ]
    return _codegen(messages, 4096)


def _check(code: str, notebook_code: str = "") -> str:
    ctx = ""
    if notebook_code:
        ctx = ("\n\nThese names are ALREADY DEFINED elsewhere in the notebook (harness helpers such "
               "as @tool, tool, run_agent) — treat them as valid and defined; do NOT claim they are "
               "undefined or 'not valid Python':\n```python\n" + notebook_code[-2000:] + "\n```")
    messages = [
        {"role": "system", "content":
            "You are a concise Python code reviewer. List at most 5 DISTINCT real issues as short "
            "bullets. Valid Python (decorators, function calls, comprehensions) is runnable — never "
            "call it 'not runnable' or 'not valid Python'. Assume names defined elsewhere exist. "
            "Plain text only, no code fences."},
        {"role": "user", "content": f"Review this code:\n```\n{code}\n```" + ctx},
    ]
    _status("🔎 reviewing the code…")
    review = _strip_think(_generate(messages, 1536))
    # Dedupe repeated bullet lines (a small model tends to repeat itself).
    seen, out = set(), []
    for ln in review.splitlines():
        key = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(ln)
    return "\n".join(out).strip()


def _fix(code: str, instruction: str) -> str:
    # Known non-Python syntax (??, &&, ...) is fixed deterministically — always correct.
    det = _deterministic_fix(code)
    if det != code and _valid_python(det)[0]:
        return det
    messages = [
        {"role": "system", "content":
            "You edit Python code. Apply the instruction and return ONLY the corrected, valid Python "
            "in one ``` block. No explanation. Python has no ?? (use `x if x is not None else y`), "
            "no && (use `and`), no || (use `or`), no === (use ==)."},
        {"role": "user", "content": f"Instruction: {instruction}\n\nCode:\n```\n{code}\n```"},
    ]
    return _codegen(messages, 4096)




def _detable_block(s: str) -> str:
    """Convert Markdown tables in a (fence-free) chunk to bullet lists — a table renders as cramped,
    unreadable columns in the narrow chat panel. Header cells become labels on each row's bullet."""
    lines = s.split("\n")
    out: list[str] = []
    i = 0
    row_re = re.compile(r"^\s*\|.*\|\s*$")
    sep_cell = re.compile(r"^:?-{1,}:?$")
    while i < len(lines):
        if row_re.match(lines[i]):
            tbl = []
            while i < len(lines) and row_re.match(lines[i]):
                tbl.append(lines[i])
                i += 1
            rows = []
            for r in tbl:
                cells = [c.strip() for c in r.strip().strip("|").split("|")]
                if cells and all(c == "" or sep_cell.match(c) for c in cells):
                    continue  # the |---|---| separator row
                rows.append(cells)
            if not rows:
                continue
            header = rows[0] if len(rows) > 1 else None
            data = rows[1:] if header else rows
            for r in data:
                if header and len(header) == len(r):
                    out.append("- " + "; ".join(f"**{h}**: {c}" for h, c in zip(header, r) if c))
                else:
                    out.append("- " + " · ".join(c for c in r if c))
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _demote_headings(text: str) -> str:
    """Make a reply render cleanly in a chat bubble, OUTSIDE code fences: Markdown headings (# ..
    ######) become bold (a heading renders as oversized text), and Markdown TABLES become bullet
    lists (they render as cramped columns in the narrow panel). `# comments` inside ```code``` and
    code tables are left alone."""
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    for i in range(0, len(parts), 2):  # even indices are outside fenced code
        parts[i] = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*$", r"**\1**", parts[i])
        parts[i] = _detable_block(parts[i])
    return "".join(parts)


def _cap_prose(text: str, limit: int = 700) -> str:
    """Trim a long prose answer at a sentence/line boundary — WITHOUT breaking Markdown. Never cuts
    inside a fenced code block (that would leave an unclosed ``` and render the whole reply as raw
    text), closes any dangling fence, and demotes chat-breaking headings to bold."""
    text = _demote_headings(text.strip())
    if len(text) > limit:
        cut = text[:limit]
        # If the cut lands inside a code fence, back up to before that fence opened.
        if cut.count("```") % 2 == 1:
            cut = cut[: cut.rfind("```")].rstrip()
        dot = max(cut.rfind(". "), cut.rfind("\n"))
        text = (cut[: dot + 1] if dot > limit // 2 else cut).rstrip() + "\n\n…"
    # Always leave Markdown valid — close a fence the model left open (else the chat renders raw).
    if text.count("```") % 2 == 1:
        text += "\n```"
    return text


def _starts_with_code(text: str, strict: bool = True) -> bool:
    """The reply is a CODE answer (not an explanation) — i.e. the model chose to WRITE CODE. This is
    the routing signal: the model reliably picks the right FORM, so its own output decides ask-vs-code
    (no brittle classifier needed).

    ``strict`` (the default, used when the caller asked a QUESTION): the reply must OPEN with a code
    block after at most a short lead-in, so a prose explanation that merely contains a code example
    stays an explanation. Non-strict (the caller already intended CODE): a single-line caption lead
    ("Here is the code, split into two cells:") is allowed when the reply is predominantly code — some
    models (e.g. Gemma) caption their code where the terser 2B just emits it."""
    s = (text or "").lstrip()
    f = s.find("```")
    if f == -1:
        return False
    lead = s[:f].strip()
    if len(lead) <= 60:
        return True
    if strict:
        return False
    # Non-strict: a one-line caption (no paragraph break) in front of code that dominates the reply.
    code_chars = sum(len(m) for m in re.findall(r"```.*?```", s, re.S))
    prose_chars = max(1, len(s) - code_chars)
    return len(lead) <= 140 and "\n\n" not in lead and code_chars >= prose_chars


def _with_marker(mark: str, text: str) -> str:
    """Prepend the 🤖 marker WITHOUT breaking Markdown. A heading/list/quote/fence/table must stay at
    the start of its line, so put the marker on its own line in that case; otherwise inline it."""
    if not mark:
        return text
    if re.match(r"\s*(#{1,6}\s|[-*+]\s|>\s|\||```|~~~|\d+[.)]\s)", text):
        return mark.rstrip() + "\n\n" + text
    return mark + text


_PLACEHOLDER = re.compile(
    r"\[\s*(complete|full|your|the|insert|add)\b[^\]]*\]"
    r"|\[[^\]]*\b(solution|code|implementation|logic|here)\b[^\]]*\]"
    r"|#\s*(your code|todo|implement|complete this)",
    re.I,
)


_NONPY = [
    (r"\?\?", "Python has no `??` — use `x if x is not None else y`"),
    (r"&&", "Python uses `and`, not `&&`"),
    (r"\|\|", "Python uses `or`, not `||`"),
    (r"===", "Python uses `==`, not `===`"),
    (r"!==", "Python uses `!=`, not `!==`"),
    (r"\+\+", "Python has no `++` — use `x += 1`"),
]


def _py_hint(code: str) -> str:
    """Deterministic hints for common non-Python syntax a beginner (or the model) may use."""
    return "; ".join(msg for pat, msg in _NONPY if re.search(pat, code))


def _deterministic_fix(code: str) -> str:
    """Rewrite common non-Python syntax to valid Python — no model needed, always correct."""
    out = code
    # null-coalescing: X ?? Y  ->  X if X is not None else Y
    out = re.sub(r"([\w.\[\]()]+)\s*\?\?\s*([\w.\[\]()'\".+\- ]+?)(\s*(?:#.*)?)$",
                 r"\1 if \1 is not None else \2\3", out, flags=re.M)
    out = re.sub(r"([\w.\[\]()]+)\s*\?\?\s*([\w.\[\]()'\".]+)", r"\1 if \1 is not None else \2", out)
    out = out.replace("===", "==").replace("!==", "!=")
    out = re.sub(r"\s*&&\s*", " and ", out)
    out = re.sub(r"\s*\|\|\s*", " or ", out)
    return out


def _valid_python(code: str):
    """(ok, message). ok is False only for a real SyntaxError."""
    try:
        compile(code, "<cell>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"{e.msg} (line {e.lineno})"
    except Exception:
        return True, ""  # runtime/name issues aren't a syntax problem


def _codegen(messages: list, max_tokens: int) -> str:
    """Generate code, then syntax-check it and retry ONCE with the error fed back."""
    _status("✍️ writing code…")
    code = _code(_generate(messages, max_tokens))
    if _is_placeholder(code):  # let the model try again for real code, don't give up
        _status("↻ asking for real code…")
        retry = list(messages) + [
            {"role": "assistant", "content": "```python\n" + code + "\n```"},
            {"role": "user", "content": "Do NOT use a placeholder like '[complete solution]'. "
                                        "Write the ACTUAL, complete, runnable Python code now."},
        ]
        code = _code(_generate(retry, max_tokens))
    _status("🔎 checking syntax…")
    ok, err = _valid_python(code)
    if ok:
        return code
    _status("↻ fixing a syntax error…")
    retry = list(messages) + [
        {"role": "assistant", "content": "```python\n" + code + "\n```"},
        {"role": "user", "content": f"That has a Python syntax error: {err}. {_py_hint(code)}. "
                                    "Return ONLY the corrected, valid Python code."},
    ]
    fixed = _code(_generate(retry, max_tokens))
    return fixed if _valid_python(fixed)[0] else code


def _is_placeholder(code: str) -> bool:
    """True if the model emitted a stub instead of real code (e.g. '[complete solution]')."""
    s = code.strip()
    if not s:
        return True
    if _PLACEHOLDER.search(s):
        return True
    if s.startswith("[") and s.endswith("]") and "\n" not in s:
        return True
    return False


def _is_prose_placeholder(text: str) -> bool:
    """A PROSE reply that is a TEMPLATE of an answer rather than an answer — every content line is a
    bracketed meta-instruction, e.g. "[Analyze the situation…]" / "[Clear, concise final answer]".
    Distinct from `_is_placeholder` (code stubs): a real explanation that merely CONTAINS a bracket
    like `[Name]` is NOT a placeholder, so this checks the whole reply, not an incidental match."""
    s = text.strip()
    if not s:
        return True
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return bool(lines) and all(ln.startswith("[") and ln.endswith("]") for ln in lines)


def _is_noop_code(code: str) -> bool:
    """The 'code' does NOTHING runnable — only a module docstring / comments / `pass` (a DESCRIPTION
    of what it would do, not actual code). This is what let "demo the classes" paste a cell that was
    just a docstring. Returns False for any real statement (call, assignment, def, class, import)."""
    import ast

    try:
        tree = ast.parse(code)
    except Exception:
        return False
    for node in tree.body:
        if isinstance(node, ast.Pass):
            continue
        if (isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant)
                and isinstance(node.value.value, str)):
            continue  # a bare string literal (docstring)
        return False  # a real statement exists
    return True  # nothing but docstrings/comments/pass


def _strip_script_scaffold(code: str) -> str:
    """Make generated code safe to drop in a cell WITHOUT losing examples. Only remove the
    `if __name__ == "__main__":` block when it would actually hang the cell (it reads input());
    a harmless demo (e.g. `if __name__ == "__main__": draw_tree("*")`) is KEPT — in a notebook
    __name__ is "__main__", so it runs and shows the example. Also drop a trailing bare `main()`
    call only when main() itself reads input()."""
    has_input = "input(" in code  # a program that reads stdin would hang on auto-run
    lines = code.splitlines()
    if has_input:
        for k, ln in enumerate(lines):
            if re.match(r"\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", ln):
                lines = lines[:k]  # drop the interactive entry point and everything after it
                break
        # No __main__ wrapper: drop a trailing bare main() that would run on cell exec.
        while lines and re.match(r"\s*main\s*\(\s*\)\s*$", lines[-1]):
            lines.pop()
    return "\n".join(lines).strip() or code.strip()


def _tidy_chat(reply: str, as_tool: bool) -> str:
    """Keep a chat reply notebook-friendly: at most one clean ```python block, one line of prose.
    Drops REPL (>>> / ...) example lines, and adds @tool when the user asked for a tool."""
    blocks = [m.group(1).strip() for m in _FENCE.finditer(reply)]
    prose = _FENCE.sub("", reply).strip()
    if not blocks:
        # Tolerate an UNCLOSED fence (rare, if generation still hit a cap): take what's after it.
        opened = re.search(r"```[a-zA-Z0-9_+-]*\n(.*)$", reply, re.DOTALL)
        if opened:
            blocks = [opened.group(1).strip()]
            prose = reply[: opened.start()].strip()
        else:
            # Pure prose (e.g. a 'what is' answer or an unwanted plan): cap it so it can't
            # overflow the chat with a wall of text.
            return _cap_prose(reply)
    code = "\n".join(
        ln for ln in blocks[0].splitlines()
        if not ln.strip().startswith(">>>") and not ln.strip().startswith("...")
    ).strip()
    code = _strip_script_scaffold(code)
    if as_tool and code.startswith("def ") and "@tool" not in code:
        code = "@tool\n" + code
    first_line = prose.splitlines()[0].strip() if prose else ""
    head = (first_line + "\n\n") if first_line else ""
    return head + "```python\n" + code + "\n```"


def _defined_names(code: str) -> set:
    """Top-level names a chunk of code defines (functions, classes, simple assignments)."""
    names = set()
    for m in re.finditer(r"^(?:def|class)\s+([A-Za-z_]\w*)", code, re.M):
        names.add(m.group(1))
    for m in re.finditer(r"^([A-Za-z_]\w*)\s*=", code, re.M):
        names.add(m.group(1))
    return names


def _drop_redefs(code: str, existing: set) -> str:
    """Remove top-level def/class blocks that redefine a name already in the notebook."""
    lines = code.split("\n")
    out: list = []
    i = 0
    while i < len(lines):
        m = re.match(r"^(?:def|class)\s+([A-Za-z_]\w*)", lines[i])
        if m and m.group(1) in existing:
            i += 1
            while i < len(lines) and (lines[i].strip() == "" or lines[i][:1] in (" ", "\t")):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out).strip()


def _strip_redefs_in_reply(reply: str, existing: set) -> str:
    """Within a tidied reply, drop code that redefines existing notebook names."""
    m = _FENCE.search(reply)
    if not m:
        return reply
    cleaned = _drop_redefs(m.group(1), existing)
    if not cleaned:
        return reply.replace(m.group(0), "").strip() or "(nothing new to add)"
    return reply[: m.start()] + "```python\n" + cleaned + "\n```" + reply[m.end():]


_GIT_URL = re.compile(r"(git@[\w.-]+:[\w./-]+?\.git|https?://[\w.-]+/[\w./-]+?(?:\.git)?)(?=\s|$)")

# Words in a "call X" message that are NOT the method name, so we don't look them up as definitions.
_CALL_STOPWORDS = {
    "call", "invoke", "run", "execute", "the", "function", "method", "func", "def", "please",
    "and", "with", "for", "use", "using", "that", "this", "it", "cell", "notebook", "print",
    "show", "result", "output", "example", "code", "file", "from", "into", "new", "make",
}

# Common words that also happen to be file names in sample repos — they must NOT route a general
# concept question ("how do I read a file?", "what is a websocket?") to repo Q&A. An IR stopword
# list for the filename match, not intent routing.
_COMMON_WORDS = {
    "file", "files", "read", "write", "text", "data", "code", "app", "main", "test", "tests",
    "input", "output", "print", "string", "number", "numbers", "list", "lists", "time", "date",
    "line", "lines", "open", "close", "loop", "error", "image", "images", "json", "html", "css",
    "server", "client", "request", "response", "websocket", "socket", "api", "http", "https",
    "class", "object", "value", "values", "index", "array", "dict", "table", "database", "query",
    "function", "method", "variable", "example", "sample", "program", "script", "module", "package",
    "python", "work", "works", "working", "about", "stuff", "thing", "things", "stream", "buffer",
}

# File extensions a "load this file" request may name (kept in sync with repo.CODE_EXT spirit).
_FILE_EXT = (r"py|js|ts|tsx|jsx|java|go|rs|rb|php|c|cc|cpp|h|hpp|cs|swift|kt|scala|"
             r"md|rst|txt|json|ya?ml|toml|cfg|ini|sh|sql|html?|css")
_FILE_RE = re.compile(r"([\w./-]+\.(?:" + _FILE_EXT + r"))\b", re.I)
_LOAD_VERB = r"load|open|read|show|view|cat|insert|paste|pull up|bring up|import"


def _is_load_request(text: str) -> bool:
    """A request to LOAD a file into a cell (not a question about code)."""
    t = text.strip()
    # Must not be a question like "how do I open a file?"
    if t.endswith("?") and re.search(r"\bhow\b|\bwhat\b|\bwhy\b", t, re.I):
        return False
    if _FILE_RE.search(t) and re.search(r"\b(" + _LOAD_VERB + r")\b", t, re.I):
        return True
    # "load the code / load it" is a deliberate load-from-context — but ONLY the explicit verb
    # "load/open/read/import/paste". "show me the code" / "write the code" are DISPLAY/WRITE requests,
    # not file opens (and in an image context mean the code shown in the image) — let them fall through.
    return bool(re.search(r"\b(load|open|read|import|paste|pull up|bring up)\s+(it|this|that|the\s+(code|file|source|script))\b", t, re.I))


def _recent_file_hint(messages: list) -> str:
    """When the user says 'load the code' with no name, find the most recently mentioned file."""
    from . import repo

    for m in reversed(messages or []):
        c = m.get("content", "")
        fm = _FILE_RE.search(c)
        if fm:
            return fm.group(1)
        # a bare stem that resolves to a real file (e.g. 'stone_paper_scissor')
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", c):
            if tok.lower() in ("code", "file", "this", "that", "load", "open", "show", "explain"):
                continue
            try:
                if repo.find_file(tok) is not None:
                    return tok
            except Exception:
                pass
    return ""




def _functions_in(code: str) -> list[tuple[str, list[str]]]:
    """TOP-LEVEL functions only (name, params) — a method INSIDE a class is not a standalone function,
    so "use/demo X" must not try to call `bark()` directly (it belongs to an instance)."""
    import ast

    try:
        tree = ast.parse(code or "")
    except Exception:
        # Fallback for unparseable code: regex, but only NON-indented (module-level) defs.
        out = []
        for m in re.finditer(r"^(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)[ \t]*\(([^)]*)\)", code or "", re.M):
            params = [p.strip() for p in m.group(2).split(",") if p.strip() and p.strip() != "self"]
            out.append((m.group(1), params))
        return out
    out = []
    for node in tree.body:  # module body only — ClassDef methods are skipped
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [a.arg for a in (list(getattr(node.args, "posonlyargs", [])) + node.args.args)
                      if a.arg != "self"]
            out.append((node.name, params))
    return out


def _simple_arg(param: str, i: int):
    """A plain example value for a positional parameter (skip ones with defaults)."""
    if "=" in param or param.strip().startswith("*"):
        return None
    base = param.split(":")[0].strip().lower()
    ann = param.split(":", 1)[1].strip().lower() if ":" in param else ""
    if "str" in ann or base in ("name", "text", "s", "word", "msg", "message", "title"):
        return '"example"'
    if "bool" in ann:
        return "True"
    if "list" in ann:
        return "[1, 2, 3]"
    if "dict" in ann:
        return "{}"
    if "float" in ann:
        return str(round(1.5 + i, 1))
    return str(2 + i)  # default: small distinct integers (add(2, 3), etc.)


def _example_call(name: str, params: list, text: str) -> str:
    """Build a runnable call to an EXISTING function. The model picks example values (grounded in the
    real signature); falls back to simple ones so it always runs."""
    sig = f"{name}({', '.join(params)})"
    sys = (f"The function {sig} already exists. Write ONLY ONE line that CALLS it with sensible "
           f"example argument values and prints the result, like print({name}(...)). "
           "Output ONLY that one line — no prose, no explanation, no code fence, no def.")
    raw = _strip_think(_generate([{"role": "system", "content": sys},
                                  {"role": "user", "content": text}], 120)).strip()
    for ln in [l.strip().strip("`") for l in raw.splitlines()]:
        if "def " in ln or (name + "(") not in ln:
            continue
        # Must actually SHOW output: a print(...) or a bare call expression (Jupyter displays it).
        # Reject an assignment like `result = add(3, 5)` — it runs but prints nothing.
        shows = ln.startswith("print(") or re.match(r"^" + re.escape(name) + r"\s*\(", ln)
        if not shows:
            continue
        try:
            compile(ln, "<e>", "exec")
            return ln
        except Exception:
            break
    args = [a for k, p in enumerate(params) if (a := _simple_arg(p, k)) is not None]
    return f"print({name}({', '.join(args)}))"


def _maybe_example(last_user: str, notebook_code: str, open_files: str, messages: list):
    """'make me an example' / 'how do I call X' / 'use X' about a function that EXISTS in the
    notebook → build a NEW cell that calls it (and runs it). 'Ask one short question' when the
    intent clearly wants an example but no function can be identified."""
    # A request to WRITE/CREATE a new thing is NEVER an example of an existing function. This also
    # stops an incidental word (e.g. "the char to use for drawing") from hijacking a create request.
    if re.search(r"\b(write|create|make|build|implement|generat\w*|draw|design|code)\b.*\b"
                 r"(method|function|class|script|program|tool|route|page|def|algorithm|tree|game|app)\b",
                 last_user, re.I):
        return None

    pool = (notebook_code or "") + "\n" + (open_files or "")
    funcs = _functions_in(pool)
    order = [n for n, _ in funcs]
    funcmap = {n: p for n, p in funcs}
    real = [n for n in order if n not in ("main", "call")]  # candidates to disambiguate over

    named = next((n for n in funcmap if re.search(r"\b" + re.escape(n) + r"\b", last_user)), None)
    # Verb that, with a named function, clearly means "demonstrate this one".
    verb_on_named = bool(re.search(r"\b(use|call|run|invoke|try|example|demo)\b", last_user, re.I))
    # Phrasing that clearly asks for an example even without naming a function.
    explicit = bool(re.search(r"\b(example|usage|demo|how\s+(do|to|can)\b.*\b(use|call|run)|invoke)\b",
                              last_user, re.I))

    target = None
    if named and verb_on_named:
        target = named                            # "use add" / "call add" / "example of add"
    elif explicit:
        if re.search(r"\b(it|this|that|the (function|method|code))\b", last_user, re.I) and real:
            target = real[-1]                     # anaphora → the most recently defined
        elif len(real) == 1:
            target = real[0]                      # only one candidate → obvious
    if target is None:
        if explicit and real:  # clearly wants an example but which? ask ONE short question
            return f"Which function should I write an example for? I can see: {', '.join(real[:6])}."
        return None  # not an example request → let normal handling take it
    call = _example_call(target, funcmap[target], last_user)
    note = f"Example calling `{target}`:"
    return f"{_RUN_SENTINEL}{note}\n\n```python\n{call}\n```"


def _maybe_load(text: str, messages: list):
    """If this is a 'load/open/show <file>' request, read the file (notebook space OR open repo)
    and return its source tagged for direct cell insertion — NO model, NO comprehension, just the
    file. Returns None when it isn't a load (so the caller falls through to Q&A).

    Two cases:
      • explicit   — a real filename ('open x.py') or 'load the code' → act, and if nothing is
                     found say so rather than silently falling through.
      • bare verb  — 'open stone_paper_scissor' → only loads if the word EXACTLY names a file,
                     otherwise returns None so a genuine question still reaches the repo Q&A.
    """
    from . import repo

    explicit = _is_load_request(text)
    try:
        repo.ensure_indexed()  # make the clone's files findable even after a kernel restart
    except Exception:
        pass

    fm = _FILE_RE.search(text)
    hint = fm.group(1) if fm else ""
    bare = False
    if not hint:
        bm = re.match(r"\s*(?:" + _LOAD_VERB + r")\s+(?:the\s+)?([A-Za-z0-9_./-]{2,})", text, re.I)
        if bm and bm.group(1).lower() not in ("code", "file", "source", "script", "it", "this", "that"):
            hint, bare = bm.group(1), True
    if not hint and explicit:  # "load the code" / "load it" → resolve from the conversation
        hint = _recent_file_hint(messages)

    if not hint:
        if explicit:
            return ("Which file should I load? Name it — e.g. `load stone_paper_scissor.py` — or "
                    "from a folder in your notebook, e.g. `load scripts/clean.py`.")
        return None

    # A bare 'open <word>' must EXACTLY name a file; otherwise it is probably a question → fall through.
    found = repo.find_file(hint, exact=(bare and not fm))
    if found is None:
        if explicit:
            return (f"I couldn't find a file matching **{hint}** in your notebook folder or the open "
                    "repo. Check the name, or clone/open the repo first.")
        return None

    _status(f"📂 opening {found.name}…")
    try:
        name, lang, code = repo.load_file(str(found))
    except Exception as exc:
        return f"Could not open {hint}: {exc}"
    # Leading sentinel → the extension drops this straight into a new cell (no pills, any mode).
    note = f"Opened **{name}** ({len(code.splitlines())} lines) into a new cell."
    return f"{_INSERT_SENTINEL}{note}\n\n```{lang}\n{code.rstrip()}\n```"


def _with_referenced(open_files: str) -> str:
    """Append the content of LOCAL modules the open files import but that aren't themselves open,
    so the model can follow the code's dependencies. Best-effort; bounded in size."""
    try:
        from . import repo

        repo.ensure_indexed()
        already = set(re.findall(r"# ===== (.+?) =====", open_files))
        refs = repo.referenced_files(open_files)
        extra = []
        for path, src in refs.items():
            if path in already or path.split("/")[-1] in {a.split("/")[-1] for a in already}:
                continue
            extra.append(f"# ===== {path}  (referenced, not open) =====\n{src}")
        if extra:
            return open_files + "\n\n" + "\n\n".join(extra)
    except Exception:
        pass
    return open_files


def _detect_tina4(open_files: str, last_user: str = "", preview_urls: list | None = None):
    """Automatically tell which Tina4 project we're working in. Priority: a project NAMED in the
    message, else the project of an OPEN file, else the project served by an open App Preview.
    Returns the project Path or None. No toggle — it just knows."""
    try:
        from . import tina4
    except Exception:
        return None
    from pathlib import Path

    # 1) A Tina4 project named in the message, found in the notebook area (cwd and its parent).
    if last_user:
        projects = {}
        for area in (Path.cwd(), Path.cwd().parent):
            projects.update(tina4.projects_in(area))
        for pname, ppath in projects.items():
            if re.search(r"\b" + re.escape(pname) + r"\b", last_user, re.I):
                return ppath

    def _walk_up(start: Path):
        """Nearest enclosing Tina4 project, stopping at the first repo (.git) boundary so one
        repo's files never resolve into a DIFFERENT enclosing project (e.g. the harness repo)."""
        d = start.resolve()
        for _ in range(12):
            if tina4.is_project(d):
                return d
            if (d / ".git").exists():  # repo boundary — a cloned repo here is not the harness
                return None
            if d.parent == d:
                return None
            d = d.parent
        return None

    # 2) The file you're working on being inside a Tina4 project.
    for rel in re.findall(r"# ===== (.+?) =====", open_files or ""):
        rel = rel.split("  (")[0].strip()  # drop a "(referenced…)" suffix
        p = Path(rel)
        cand = p if p.is_absolute() else (Path.cwd() / rel)
        proj = _walk_up(cand.parent if cand.suffix else cand)
        if proj is not None:
            return proj

    # 3) The project served by an open App Preview tab (the thing you're looking at).
    for url in preview_urls or []:
        m = re.search(r":(\d+)", url or "")
        if m:
            proj = tina4.project_for_port(int(m.group(1)))
            if proj is not None:
                return proj
    return None




def _as_choices(reply: str) -> dict | None:
    """If an ask reply is a clarifying question offering a SHORT list of options (e.g. 'A programming
    language? / A specific code snippet? …'), return {lead, options} so the UI can show clickable
    pills instead of making the user type. Returns None for ordinary explanatory lists."""
    raw, other = [], []
    for ln in reply.splitlines():
        m = re.match(r"^\s*(?:[-*•‣▪]|\d+[.)])\s+(.+?)\s*$", ln)
        if m:
            raw.append(m.group(1).strip())
        elif ln.strip() and not ln.strip().startswith("```"):
            other.append(ln.strip())
    if not (2 <= len(raw) <= 6):
        return None
    # A bullet may carry trailing lead-out prose after the choice ("A problem you're working on? Just
    # let me know…") — keep only the choice itself (up to the first '?').
    q_count, choices = 0, []
    for o in raw:
        if "?" in o:
            q_count += 1
            choices.append(o.split("?", 1)[0].strip())
        else:
            choices.append(o.strip())
    if any(len(c) > 60 or not c for c in choices):
        return None
    lead = " ".join(other).strip()
    # Only an explicit "pick one" lead-in counts as a clarification cue — NOT a bare "what/which"
    # (an overview like "What is a Queue?" with bullets must stay prose, not become pills).
    cue = re.search(
        r"\b(which (?:one|of these|of the following|would)|did you mean|do you mean|are you asking|"
        r"would you like (?:me )?to|do you want (?:me )?to|pick one|choose (?:one|from|between)|"
        r"let me know which|are you looking for|which of)\b", lead, re.I)
    # A clarification: MOST options are themselves questions, OR the lead explicitly asks to choose.
    questiony = len(choices) >= 2 and q_count >= max(2, (len(choices) + 1) // 2)
    if not (questiony or (cue and "?" in lead)):
        return None
    clean = [re.sub(r"^(an?|the)\s+", "", c, flags=re.I).strip() for c in choices]
    clean = [c for c in clean if c]
    if len(clean) < 2:
        return None
    return {"lead": lead or "Which did you mean?", "options": clean[:6]}


def _is_new_app_request(text: str) -> bool:
    """'new tina4 app called X' / 'create a tina4 project X' → provision + serve + preview."""
    return bool(
        re.search(r"\b(new|create|make|provision|scaffold|spin up|start)\b", text, re.I)
        and re.search(r"\btina4\b", text, re.I)
        and re.search(r"\b(app|application|project|site|server)\b", text, re.I)
    )


def _app_name(text: str) -> str:
    m = re.search(r"\b(?:called|named)\s+([A-Za-z][\w-]*)", text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"\btina4\s+(?:app|application|project|site)\s+([A-Za-z][\w-]*)", text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-Za-z][\w-]*)\s+(?:app|application|project|site)\b", text, re.I)
    if m and m.group(1).lower() not in ("new", "tina4", "a", "the", "my"):
        return m.group(1)
    return "myapp"


def _gitignore_path(path) -> None:
    """Add a provisioned app to the enclosing repo's .gitignore so it doesn't pollute the repo."""
    from pathlib import Path

    p = Path(path).resolve()
    root = p
    for _ in range(12):
        if (root / ".git").exists():
            break
        if root.parent == root:
            return
        root = root.parent
    gi = root / ".gitignore"
    try:
        rel = str(p.relative_to(root)) + "/"
    except Exception:
        return
    try:
        existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
        if rel not in existing:
            with gi.open("a", encoding="utf-8") as f:
                f.write(f"\n# Provisioned Tina4 app (local artifact)\n{rel}\n")
    except Exception:
        pass


_TINA4_MISSING = (
    "To spin up a web app I use the **tina4** client (the Rust CLI) — `tina4 init` scaffolds the "
    "project and `tina4 serve` runs it. It isn't installed / on PATH here, so I can't provision one "
    "yet.\n\nInstall it (needs internet):\n```bash\ncurl -fsSL https://tina4.com/install.sh | sh\n```\n"
    "Then ask me again. Heads-up: the *first* `tina4 init` also downloads the Python runtime and the "
    "tina4 framework package, so initial setup needs a connection — after that, serving and adding "
    "routes/pages work fully offline."
)


def _looks_offline(msg: str) -> bool:
    return bool(re.search(
        r"could not resolve|name or service not known|temporary failure in name resolution|"
        r"no address associated|failed to (?:download|fetch|resolve)|network is unreachable|"
        r"connection (?:refused|timed out|reset)|timed out|offline|proxyerror|"
        r"max retries exceeded|getaddrinfo|ssl|certificate|resolve host|no route to host|"
        r"pypi|crates\.io|tina4\.com",
        msg, re.I,
    ))


_IMPORT_ERR = re.compile(
    r"cannot import name ['\"](?P<name>\w+)['\"] from ['\"](?P<mod>[\w.]+)['\"]"
    r"|No module named ['\"](?P<mod2>[\w.]+)['\"]"
    r"|module ['\"](?P<mod3>[\w.]+)['\"] has no attribute ['\"](?P<attr>\w+)['\"]",
    re.I,
)


def _ground_import_error(output: str) -> str:
    """When an error is an import/attribute failure, introspect the REAL module (by importing it) so
    the model fixes against reality instead of inventing an API — e.g. 'from antigravity import
    Client' (antigravity has no Client; it is a joke module). Returns a GROUNDED FACT to add to the
    prompt, or '' when the error isn't an import/attribute error."""
    m = _IMPORT_ERR.search(output or "")
    if not m:
        return ""
    mod = m.group("mod") or m.group("mod2") or m.group("mod3")
    name = m.group("name") or m.group("attr")
    if not mod:
        return ""
    probe = (
        "import importlib\n"
        f"try:\n    _m = importlib.import_module({mod!r})\n"
        "    print('OK', sorted(n for n in dir(_m) if not n.startswith('_'))[:40])\n"
        "except Exception as _e:\n    print('MISSING', type(_e).__name__, str(_e)[:120])\n"
    )
    out = (_run_in_kernel(probe, timeout=6.0).get("out") or "").strip()
    if out.startswith("OK"):
        names = out[2:].strip()
        truth = f"The module `{mod}` IS importable here. Its REAL public names are: {names}."
        if name:
            truth += (f" It has NO `{name}` — that name does not exist in `{mod}`. Do NOT import or "
                      f"call `{name}`, do NOT invent an API around it, and do NOT suggest installing "
                      f"`{mod}` (it is already present). Use only the real names above, or remove the "
                      "fabricated usage.")
        return "\n\nGROUNDED FACT (from actually importing the module): " + truth
    if out.startswith("MISSING"):
        return (f"\n\nGROUNDED FACT: `{mod}` is NOT available in this environment ({out[7:].strip()}). "
                "The user is OFFLINE — do NOT suggest pip/uv/conda install. Use the standard library "
                "instead, or tell the user plainly that the module isn't installed.")
    return ""


def _new_tina4_app(text: str) -> str:
    """Provision a new Tina4 Python app in the notebook area, start it, and open the preview."""
    from pathlib import Path

    from . import tina4

    # Tell the user about tina4 up front instead of surfacing a raw FileNotFoundError.
    if not tina4.available():
        return _TINA4_MISSING

    name = tina4._slug(_app_name(text))
    dest = Path.cwd() / name
    if not dest.exists() or not tina4.is_project(dest):
        _status(f"⬇️ provisioning tina4 app '{name}' (uv sync)…")
        try:
            tina4.init(dest)
        except Exception as exc:
            if _looks_offline(str(exc)):
                return (
                    f"I couldn't provision **{name}** offline. `tina4 init` runs `uv sync`, which "
                    "reuses the uv cache (`~/.cache/uv`) when it can — so with the client installed and "
                    "the tina4 packages already cached, provisioning works offline. This time it "
                    "needed something not in the cache (a runtime or a framework version not synced "
                    "yet) and couldn't reach the network.\n\nDo one online `tina4 init` (or `uv sync`) "
                    "to warm the cache; after that, provisioning, `tina4 serve`, and adding "
                    f"routes/pages all work offline.\n\nDetail: {exc}"
                )
            return f"Could not provision the Tina4 app **{name}**: {exc}"
        _gitignore_path(dest)  # keep the harness repo clean — provisioned apps are local artifacts
    port = tina4.free_port(7146)
    _status(f"🚀 starting tina4 serve on :{port}…")
    url = tina4.serve(dest, port=port)
    note = (f"🤖 Provisioned and started **{name}** at {url} — opening the preview. "
            f"It's a standard Tina4 app in `{name}/` (routes in `src/routes/`). "
            "Ask me to add routes/pages and they'll land in the project.")
    return f"{_APP_SENTINEL}|{url}\n{note}"


def _is_tina4_file_request(text: str) -> bool:
    """A request to add a real file to the Tina4 project (route/page/endpoint), not a notebook cell."""
    if re.search(r"\b(in|into|as|to)\s+a?\s*(new\s+)?cell\b", text, re.I):
        return False
    return bool(
        re.search(r"\b(add|create|make|new|build|scaffold|generate)\b", text, re.I)
        and re.search(r"\b(route|page|endpoint|api|view|template|home ?page|handler)\b", text, re.I)
    )


def _tina4_target(text: str):
    """Work out (name, url_path, is_page) for a route from the request."""
    mpath = re.search(r"(/[A-Za-z0-9_\-/]*)", text)
    url = mpath.group(1) if mpath else None
    is_page = bool(re.search(r"\bpage|home ?page|view|html|template\b", text, re.I)) and \
        not re.search(r"\b(json|api|endpoint)\b", text, re.I)
    _STOP = {"a", "an", "the", "new", "another", "some", "simple", "basic"}
    name = None
    if re.search(r"\bhome ?page\b", text, re.I):
        name, url = "home", url or "/"
    else:
        mname = re.search(r"\b([a-z][a-z0-9_]*)\s+(?:page|route|endpoint|view|handler|api)\b", text, re.I)
        if mname and mname.group(1).lower() not in _STOP:
            name = mname.group(1).lower()
    if name is None and url:  # fall back to the URL's last segment
        name = url.strip("/").split("/")[-1] or "index"
    if name is None:
        name = "index"
    if url is None:
        url = "/" if name in ("home", "index") else "/" + name
    return name, url, is_page


def _tina4_title(text: str, name: str) -> str:
    m = re.search(r"\b(?:titled|title|says?|saying|called)\s+['\"]?([^'\".]{2,50})", text, re.I)
    if m:
        return m.group(1).strip().title()
    return name.replace("_", " ").title()


def _tina4_endpoint_expr(text: str, name: str):
    """Model generates ONLY the expression to return from response(...). Validated; safe fallback.
    Returns (expr, extra_imports)."""
    sys = ("You write ONE Python expression (a dict or value) — no statements, no code fence, no "
           "assignment, no function. It will be placed inside `response(<expr>)` in an async web "
           "handler. You MAY use datetime.now() and json. Examples:\n"
           '  {"time": datetime.now().isoformat()}\n  {"items": [1, 2, 3]}\n'
           "Return ONLY the expression.")
    raw = _strip_think(_generate([{"role": "system", "content": sys},
                                  {"role": "user", "content": f"Endpoint for: {text}"}], 200)).strip()
    raw = raw.strip("`").splitlines()[0].strip() if raw else ""
    raw = re.sub(r"^return\s+", "", raw)
    try:
        compile(f"({raw})", "<expr>", "eval")
        expr = raw
    except Exception:
        expr = f'{{"{name}": "ok"}}'
    imports = ""
    if "datetime" in expr:
        imports += "from datetime import datetime\n"
    if re.search(r"\bjson\.", expr):
        imports += "import json\n"
    return expr, imports


def _tina4_write(project, text: str, messages: list) -> str:
    """Scaffold a Tina4 route DETERMINISTICALLY (correct API, always valid) and let the model fill
    only the content (a page's template, or an endpoint's return value). Writes into the project."""
    from . import tina4

    name, url, is_page = _tina4_target(text)
    _status(f"🛠 writing src/routes/{name}.py in {project.name}…")
    written = []
    if is_page:
        title = _tina4_title(text, name)
        tpl_sys = ("Write ONLY the HTML for a small, self-contained web page (a Twig template). Use "
                   "{{ title }} where the title goes. Plain inline HTML/CSS only — no external "
                   "scripts/fonts/CDNs. No code fence.")
        tpl = _code(_generate([{"role": "system", "content": tpl_sys},
                               {"role": "user", "content": f"Page titled '{title}': {text}"}], 700))
        if not tpl.strip() or "<" not in tpl:
            tpl = f"<!doctype html>\n<html><head><title>{{{{ title }}}}</title></head>\n<body>\n  <h1>{{{{ title }}}}</h1>\n</body></html>"
        written.append(tina4.add_template(project, name, tpl))
        # Correct Tina4 pattern: @template BELOW @get; the handler returns a dict (the context).
        route = (
            "from tina4_python.core.router import get, template\n\n\n"
            f'@get("{url}")\n'
            f'@template("{name}.twig")\n'
            f"async def {name}(request, response):\n"
            f'    return {{"title": {title!r}}}\n'
        )
    else:
        expr, imports = _tina4_endpoint_expr(text, name)
        route = (
            "from tina4_python.core.router import get\n"
            + imports
            + f'\n\n@get("{url}")\n'
            f"async def {name}(request, response):\n"
            f"    return response({expr})\n"
        )
    # Guarantee the route file is valid Python before writing (never break the server).
    if not _valid_python(route)[0]:
        route = (
            "from tina4_python.core.router import get\n\n\n"
            f'@get("{url}")\n'
            f"async def {name}(request, response):\n"
            f'    return response({{"{name}": "ok"}})\n'
        )
    written.insert(0, tina4.add_route(project, name, route))
    files = ", ".join(f"`{w}`" for w in written)
    note = (f"🤖 Wrote {files} in **{project.name}** — route `{url}`. "
            "tina4's hot-reload picks it up and the App Preview auto-refreshes.")
    return f"{_WROTE_SENTINEL}|{project}|{url}\n{note}\n\n```python\n{route.strip()}\n```"




def _evolving_context(messages: list, query: str, keep_recent: int = 5, k: int = 8,
                      max_chars: int = 60000) -> list:
    """Endless chat via Groundwire: keep the last ``keep_recent`` turns verbatim (recency) and
    RETRIEVE the most relevant EARLIER turns with BM25, so the prompt stays bounded no matter how
    long the conversation grows. This is the 'rotating / evolving' context — old turns don't pile up
    linearly; they're recalled only when relevant to the current question."""
    def _cap(msgs: list) -> list:
        out, used = [], 0
        for m in reversed(msgs):
            c = len(m.get("content", ""))
            if out and used + c > max_chars:
                break
            out.append(m)
            used += c
        out.reverse()
        return out

    if len(messages) <= keep_recent + 3:
        return _cap(messages)  # short conversation — send it all (within the cap)

    recent = messages[-keep_recent:]
    older = messages[:-keep_recent]
    relevant: list = []
    try:
        from groundwire import Groundwire

        gw = Groundwire(memory="bm25", k=k, max_words=160)
        for i, m in enumerate(older):
            txt = (m.get("role", "") + ": " + m.get("content", "")).strip()
            if txt:
                gw.ingest_code(txt, title=str(i))
        idxs: set[int] = set()
        for item in gw.retrieve(query, k=k):
            text = item[1] if isinstance(item, (list, tuple)) and len(item) >= 2 else str(item)
            mm = re.search(r"#\s*file:\s*(\d+)", text)
            if mm:
                idxs.add(int(mm.group(1)))
        relevant = [older[i] for i in sorted(idxs) if 0 <= i < len(older)]  # chronological order
    except Exception:
        relevant = []
    return _cap(relevant + recent)


def _one_line(s: str, n: int = 120) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def _answer_gist(content: str, n: int = 150) -> str:
    """A one-line gist of an assistant reply — code fences, sentinels, the 🤖 marker and heading
    markers stripped, first real sentence kept. Used in the running conversation summary."""
    c = re.sub(r"```.*?```", "", content or "", flags=re.S)
    c = re.sub(r"@@HARNESS_\w+@@\S*", "", c)
    c = re.sub(r"(?m)^\s*#{1,6}\s*", "", c).replace("🤖", "").strip()
    first = re.split(r"(?<=[.!?])\s", c, 1)[0] if c else ""
    return _one_line(first or c, n)


def _conversation_digest(messages: list, keep_recent: int = 5, max_chars: int = 1600) -> str:
    """A compact, model-free running summary of the turns BEFORE the last ``keep_recent`` — each
    exchange as the user's line + a one-line gist of my answer. This is the persisting thread memory:
    a follow-up like 'why not dependencies?' can see the conversation was about tina4, even once the
    original turn has scrolled out of the verbatim window."""
    older = messages[:-keep_recent] if len(messages) > keep_recent else []

    def _verb(text: str) -> str:
        # A question vs a statement/fact the user told us ("my favourite number is 42").
        return "You asked" if text.rstrip().endswith("?") or re.match(
            r"\s*(what|why|how|which|who|when|where|can|does|is|are|do)\b", text, re.I) else "You told me"

    lines, pending = [], None
    for m in older:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            if pending:
                lines.append(f"- {_verb(pending)}: {pending}")
            pending = _one_line(content, 140)
        elif role == "assistant":
            gist = _answer_gist(content)
            lines.append(f"- {_verb(pending)}: {pending} → I answered: {gist}" if pending
                         else f"- I said: {gist}")
            pending = None
    if pending:
        lines.append(f"- {_verb(pending)}: {pending}")
    # Keep the most recent exchanges within the cap (drop oldest first).
    kept, used = [], 0
    for ln in reversed(lines):
        if kept and used + len(ln) > max_chars:
            break
        kept.append(ln)
        used += len(ln)
    kept.reverse()
    return "\n".join(kept)


def _strip_docstrings(tree):
    """Drop the leading bare-string docstring of every module/class/function node (in place)."""
    import ast

    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
               and isinstance(first.value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _behaviour_equiv(a: str, b: str) -> bool:
    """True when two snippets parse to the SAME AST once docstrings are removed — i.e. they do the
    exact same thing and differ only in names-in-comments, docstrings, comments or formatting. Used
    to reject a bogus 'fix' for code that is not actually broken (e.g. a mislabelled 'broken_add')."""
    import ast

    try:
        da = ast.dump(_strip_docstrings(ast.parse(a)))
        db = ast.dump(_strip_docstrings(ast.parse(b)))
        return da == db
    except Exception:
        return False


def _run_in_kernel(code: str, timeout: float = 8.0) -> dict:
    """Actually RUN a snippet so the agent can judge by reality, not by names/comments. Executes in a
    COPY of the live kernel namespace (so it sees prior cells' imports/defs but cannot clobber them),
    captures stdout+stderr and the value of a trailing expression, and never raises — a crash in the
    snippet is returned as data. Bounded by a timeout; a snippet that reads stdin is skipped."""
    import ast, io, contextlib, threading, traceback

    res = {"ran": False, "out": "", "value": None, "error": None, "timeout": False, "skipped": None}
    if not code.strip():
        res["skipped"] = "empty"
        return res
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        res["error"] = f"SyntaxError: {e.msg} (line {e.lineno})"
        res["ran"] = True
        return res
    # A top-level input() would block until the timeout — don't run those.
    if re.search(r"(?<![\w.])input\s*\(", code):
        res["skipped"] = "reads stdin"
        return res
    try:
        from IPython import get_ipython

        ip = get_ipython()
        ns = dict(ip.user_ns) if ip is not None else {}
    except Exception:
        ns = {}
    ns["__name__"] = "__main__"  # so `if __name__ == "__main__":` blocks execute
    buf = io.StringIO()

    def _work():
        try:
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(tree, "<cell>", "exec"), ns)  # noqa: S102 — running the user's own cell
                if last is not None:
                    val = eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns)  # noqa: S307
                    if val is not None:
                        res["value"] = repr(val)[:400]
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc().strip().splitlines()
            if tb:
                res["error_tb"] = tb[-1]
        finally:
            res["out"] = buf.getvalue()[:1500]

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(timeout)
    res["ran"] = True
    if t.is_alive():
        res["timeout"] = True
        res["out"] = buf.getvalue()[:1500]
    return res


_BUG_CLAIM = re.compile(
    r"\b(bugs?|broken|errors?|fix(?:ed|es)?|fixing|wrong|incorrect|fails?|failing|failure|"
    r"missing|doesn'?t|does not|won'?t|will not|never\s+returns?|syntaxerror|typeerror|"
    r"nameerror|valueerror|exception|raise[sd]?|should\s+(?:be|return|instead|have)|"
    r"problem|issue|mistake|typo|defect|flaw|correct(?:ed|ion|ions|ing)|"
    r"would\s+have|instead\s+of|rather\s+than)\b",
    re.I,
)


def _drop_bug_claims(text: str) -> str:
    """Deterministically remove sentences/bullets that ASSERT a bug — for use only after the code has
    been proven clean, so a stubborn model can't keep insisting on a bug that isn't there."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    kept = [p.strip() for p in parts if p.strip() and not _BUG_CLAIM.search(p)]
    return " ".join(kept).strip()


def _run_summary(res: dict) -> str:
    """A one-line, human-readable summary of a _run_in_kernel result."""
    if res.get("skipped"):
        return f"not run ({res['skipped']})"
    if res.get("timeout"):
        return f"ran longer than the timeout and was left running; partial output: {res.get('out','').strip()[:200]!r}"
    if res.get("error"):
        out = res.get("out", "").strip()
        return f"raised {res['error']}" + (f" (after printing {out[:120]!r})" if out else "")
    out = res.get("out", "").strip()
    val = res.get("value")
    if out and val is not None:
        return f"printed {out[:200]!r} and the last expression returned {val}"
    if out:
        return f"printed {out[:300]!r}"
    if val is not None:
        return f"the last expression returned {val}"
    return "ran with no output or error"


def _safe_to_run(code: str) -> bool:
    """Safe to AUTO-run for self-validation (before the user has chosen to run it): no stdin, and no
    destructive / filesystem-writing / network / process side effects. We would rather skip
    validating risky model output than execute it silently — the goal is to catch hallucinated
    imports/APIs in ordinary code, not to run `os.remove` or a network call on the user's behalf."""
    if re.search(r"(?<![\w.])input\s*\(", code):
        return False
    return not re.search(
        r"\bos\.(remove|unlink|rmdir|removedirs|rename|replace|system|truncate)\b"
        r"|\bshutil\.(rmtree|move|copy\w*)\b|\bsubprocess\b|\bPopen\b|\bos\.popen\b"
        r"|\bopen\s*\([^)]*['\"][wax]\+?['\"]|\bpathlib\b.*\.(write_\w+|unlink|rmdir)"
        r"|\.write_(text|bytes)\s*\(|\bsocket\b|\brequests\.|\burllib\b|\bhttpx\b|\bsmtplib\b"
        r"|\bsys\.exit\b|\bexit\s*\(|\bquit\s*\(|\b__import__\b",
        code,
    )


def _probe_calls(code: str, max_calls: int = 4) -> str:
    """Build a snippet that CALLS each top-level function with simple, distinct arguments and prints
    the result — so two versions of a cell can be compared by what they actually produce, not by how
    they read. Distinct primes per position make e.g. a+b (=5) differ from a+1 (=3). Returns "" when
    there is nothing callable (then the comparison falls back to the cell's own output)."""
    import ast

    try:
        tree = ast.parse(code)
    except Exception:
        return ""
    # Don't actively CALL functions that look like they touch files/network/processes with made-up
    # args — only compare the cell's own output in that case.
    if re.search(r"\b(open|os|shutil|subprocess|sys|socket|requests|urllib|httpx|pathlib|Path|"
                 r"remove|unlink|rmtree|rename|write|send|connect|popen|system|exec|eval|input)\b", code):
        return ""
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    lines = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name.startswith("_"):
            continue
        a = node.args
        if a.vararg or a.kwarg or a.kwonlyargs:
            continue  # keep probes to plain positional signatures
        pos = list(getattr(a, "posonlyargs", [])) + list(a.args)
        required = pos[: len(pos) - len(a.defaults)] if a.defaults else pos
        vals = []
        for i, arg in enumerate(required):
            ann = getattr(arg, "annotation", None)
            t = ann.id if isinstance(ann, ast.Name) else None
            if t == "str":
                vals.append(repr("abcde"[: i + 1]))
            elif t == "float":
                vals.append(str(primes[i % len(primes)]) + ".5")
            elif t == "bool":
                vals.append("True")
            elif t in ("list", "List"):
                vals.append("[1, 2, 3]")
            elif t in ("dict", "Dict"):
                vals.append("{'k': 1}")
            else:  # int or unannotated → a distinct small number
                vals.append(str(primes[i % len(primes)]))
        call = f"{node.name}({', '.join(vals)})"
        lines.append(
            f"try:\n    print({node.name!r}, '->', repr({call}))\n"
            f"except Exception as _e:\n    print({node.name!r}, 'raised', type(_e).__name__, _e)"
        )
        if len(lines) >= max_calls:
            break
    return "\n".join(lines)


def _characterization_test(fn_code: str) -> str:
    """Build a test cell that is GUARANTEED to pass and run: call each top-level function with simple
    distinct args, capture what it REALLY returns by executing it, and pin that as an assertion. A
    characterization test grounded in real execution — never a guessed value that fails (e.g. the 2B
    reaching for `assert add(-3.7, -4.2) == -7.9`, which floating point makes false). Returns "" when
    nothing safe is callable, so the caller can fall back."""
    import ast

    try:
        tree = ast.parse(fn_code)
    except Exception:
        return ""
    if re.search(r"\b(open|os|shutil|subprocess|sys|socket|requests|urllib|httpx|pathlib|Path|"
                 r"remove|unlink|rmtree|rename|write|send|connect|popen|system|exec|eval|input)\b", fn_code):
        return ""
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    calls = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name.startswith("_"):
            continue
        a = node.args
        if a.vararg or a.kwarg or a.kwonlyargs:
            continue
        pos = list(getattr(a, "posonlyargs", [])) + list(a.args)
        required = pos[: len(pos) - len(a.defaults)] if a.defaults else pos
        vals = []
        for i, arg in enumerate(required):
            ann = getattr(arg, "annotation", None)
            t = ann.id if isinstance(ann, ast.Name) else None
            if t == "str":
                vals.append(repr("abcde"[: i + 1]))
            elif t == "bool":
                vals.append("True")
            elif t in ("list", "List"):
                vals.append("[1, 2, 3]")
            elif t in ("dict", "Dict"):
                vals.append("{'k': 1}")
            else:  # int or unannotated — exact under ==, unlike a float
                vals.append(str(primes[i % len(primes)]))
        calls.append(f"{node.name}({', '.join(vals)})")
        if len(calls) >= 3:
            break
    if not calls:
        return ""
    # Execute to capture the REAL return values, then pin them as assertions.
    probe = "\n".join(f"print(repr({c}))" for c in calls)
    r = _run_in_kernel(fn_code + "\n" + probe)
    outs = (r.get("out") or "").splitlines()
    if r.get("error") or len(outs) < len(calls):
        return ""
    lines = [f"assert {c} == {outs[i]}" for i, c in enumerate(calls)]
    lines.append('print("tests passed!")')
    test = "\n".join(lines)
    if not _valid_python(test)[0] or _run_in_kernel(fn_code + "\n" + test).get("error"):
        return ""
    return test


def _pytest_from_expected(fn_code: str) -> str:
    """Build a pytest test by asking the model ONLY for the value each sample call SHOULD return, then
    asserting it. The 2B can't reliably write a whole pytest file (it reaches for fixtures/parametrize
    that don't run when called directly), but it can name an expected value — so we keep the EXPECTED
    value (a buggy function still fails) while guaranteeing a valid, runnable pytest test."""
    import ast

    try:
        tree = ast.parse(fn_code)
    except Exception:
        return ""
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and not n.name.startswith("_")), None)
    if fn is None:
        return ""
    a = fn.args
    if a.vararg or a.kwarg or a.kwonlyargs:
        return ""
    pos = list(getattr(a, "posonlyargs", [])) + list(a.args)
    required = pos[: len(pos) - len(a.defaults)] if a.defaults else pos
    primes = [2, 3, 5, 7]

    def _mkargs(shift: int) -> str:
        vals = []
        for i, arg in enumerate(required):
            ann = getattr(arg, "annotation", None)
            t = ann.id if isinstance(ann, ast.Name) else None
            vals.append(repr("abc"[: i + 1]) if t == "str" else str(primes[(i + shift) % 4]))
        return ", ".join(vals)

    calls = [f"{fn.name}({_mkargs(0)})", f"{fn.name}({_mkargs(2)})"]
    asserts, uses_float = [], False
    for call in calls:
        q = ("Given the function below, what value SHOULD this call return (the CORRECT result for the "
             f"function's purpose, judging by its NAME)?\nCall: {call}\nReply with ONLY the Python "
             "value — e.g. 5 or 'hi' or 3.14 — nothing else.")
        try:
            raw = _strip_think(_generate(
                [{"role": "system", "content": "Reply with ONLY a single Python literal value — no prose, no code fence."},
                 {"role": "user", "content": "```python\n" + fn_code + "\n```\n\n" + q}],
                40, enable_thinking=False, temperature=0.0)).strip()
        except Exception:
            return ""
        m = re.search(r"-?\d+\.\d+|-?\d+|'[^']*'|\"[^\"]*\"|True|False|None", raw)
        if not m:
            return ""
        expected = m.group(0)
        if re.fullmatch(r"-?\d+\.\d+", expected):
            uses_float = True
            asserts.append(f"    assert {call} == pytest.approx({expected})")
        else:
            asserts.append(f"    assert {call} == {expected}")
    head = "import pytest\n\n" if uses_float else ""
    body = (head + f"def test_{fn.name}():\n" + "\n".join(asserts)
            + f"\n\ntest_{fn.name}()\nprint('✓ test_{fn.name} passed')")
    return body if _valid_python(body)[0] else ""


def _valid_expr(src: str) -> bool:
    import ast

    try:
        ast.parse(src, mode="eval")
        return True
    except Exception:
        return False


def _extract_assertion(text: str) -> list:
    """Pull expectations the user pinned in their message so they can act as a test oracle, e.g.
    'add(3,4)==7', 'add(3,4) should be 7', 'greet(\"Al\") returns \"Hi Al\"', or 'assert add(3,4)==7'.
    Returns a list of Python boolean-expression strings (at most 3)."""
    conds = []
    for m in re.finditer(r"\bassert\s+(.+?)(?:[.;\n]|$)", text, re.I):
        conds.append(m.group(1).strip())
    # a call compared directly: name(args) ==|!=|>=|<=|>|< value
    for m in re.finditer(
        r"([A-Za-z_]\w*\s*\([^()]*\)\s*(?:==|!=|>=|<=|>|<)\s*[^,;\n]+?)(?=\s+(?:and|or)\s+|[,;.\n]|$)", text):
        conds.append(m.group(1).strip())
    # a call with phrasing: name(args) should be / returns / gives / equals / -> value
    for m in re.finditer(
        r"([A-Za-z_]\w*\s*\([^()]*\))\s*(?:should\s+(?:be|return|equal|give|output)s?|returns?|gives?|"
        r"outputs?|equals?|is|=>|->)\s*"
        r"(-?\d+(?:\.\d+)?|'[^']*'|\"[^\"]*\"|True|False|None|\[[^\]]*\]|\{[^}]*\}|\([^()]*\))",
        text, re.I):
        conds.append(f"{m.group(1).strip()} == {m.group(2).strip()}")
    seen, out = set(), []
    for c in conds:
        if c not in seen and _valid_expr(c):
            seen.add(c)
            out.append(c)
    return out[:3]


def _assert_report(code: str, conds: list) -> dict:
    """Run `code` then evaluate each pinned condition against it. Returns status (pass/fail/error) and
    a human line per condition, including the ACTUAL value on failure."""
    import ast

    lines, status = [], "pass"
    for c in conds:
        r = _run_in_kernel(code + "\n(" + c + ")")
        if r.get("error"):
            lines.append(f"`{c}` → could not run ({r['error']})")
            status = "error"
            continue
        passed = r.get("value") == "True"
        actual = ""
        if not passed:
            try:
                node = ast.parse(c, mode="eval").body
                if isinstance(node, ast.Compare) and len(node.comparators) == 1:
                    lhs = ast.unparse(node.left)
                    ar = _run_in_kernel(code + "\n(" + lhs + ")")
                    if not ar.get("error") and ar.get("value") is not None:
                        actual = f" (got {ar['value']})"
            except Exception:
                pass
            if status != "error":
                status = "fail"
        lines.append(f"`{c}` → {'PASS' if passed else 'FAIL' + actual}")
    return {"status": status, "lines": lines}


def _run_sig(res: dict):
    """A comparable signature of a run: (did it error, error type, printed text, returned value)."""
    err = res.get("error") or ""
    etype = err.split(":", 1)[0] if err else ""
    return (bool(err), etype, res.get("out", "").strip(), res.get("value"))


def _defs_only(code: str) -> str:
    """Keep only imports and function/class definitions — drop top-level statements (calls, prints,
    `if __name__ == '__main__':` blocks). Lets us compare FUNCTION behaviour without example code the
    model adds."""
    import ast

    try:
        tree = ast.parse(code)
    except Exception:
        return code
    tree.body = [n for n in tree.body if isinstance(
        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    try:
        return ast.unparse(tree)
    except Exception:
        return code


def _fix_verdict(orig: str, fixed: str) -> str:
    """Decide whether a proposed 'fix' is real by RUNNING both and comparing outputs:
      'cosmetic'          — identical observable behaviour (not a fix at all)
      'fixes_error'       — original errored, proposal runs clean
      'fix_breaks'        — original ran clean, proposal errors (reject it)
      'changes_behaviour' — both run but produce different output (a real candidate)
    """
    if _behaviour_equiv(orig, fixed):
        return "cosmetic"  # identical once docstrings/comments are ignored — no need to run
    # 1) Run the cells AS WRITTEN to catch a top-level ERROR being fixed (or introduced).
    a_full = _run_sig(_run_in_kernel(orig))
    b_full = _run_sig(_run_in_kernel(fixed))
    if a_full[0] and not b_full[0]:
        return "fixes_error"
    if b_full[0] and not a_full[0]:
        return "fix_breaks"
    # 2) Otherwise compare the FUNCTIONS' behaviour in isolation (defs + probe calls), so merely
    #    ADDING example usage (prints, a __main__ block) is NOT mistaken for a behaviour change.
    probe = _probe_calls(orig)
    if probe:
        a = _run_sig(_run_in_kernel(_defs_only(orig) + "\n" + probe))
        b = _run_sig(_run_in_kernel(_defs_only(fixed) + "\n" + probe))
        return "cosmetic" if a == b else "changes_behaviour"
    return "cosmetic" if a_full == b_full else "changes_behaviour"


def _chat(messages: list, context: str = "", notebook_code: str = "", output: str = "",
          intent: str = "code", open_files: str = "", preview_urls: list | None = None,
          think: str = "full") -> str:
    enable_thinking, chat_cap = _THINK_MODES.get(think, _THINK_MODES["full"])
    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    tina4_project = _detect_tina4(open_files, last_user, preview_urls or [])

    # "snapshot" / "look at the screen" / "what's on screen" → screenshot the harness window and let
    # the vision model SEE what's happening (the notebook or the tina4 App Preview).
    if re.search(r"\b(snapshot|screenshot)\b", last_user, re.I) or \
       re.search(r"\b(see|look at|check|view|what(?:'s| is| do you see)?)\b.{0,20}\bscreen\b", last_user, re.I):
        try:
            from . import snapshot as _snap

            _status("📸 taking a screenshot…")
            return _snap.snapshot(last_user, show=False)
        except Exception as exc:
            return f"Could not snapshot the screen: {exc}"

    # In a Tina4 project + a "add a route/page" request → WRITE the file(s) into the project
    # (not a notebook cell). The extension shows the code and refreshes the open App Preview.
    if tina4_project is not None and _is_tina4_file_request(last_user):
        try:
            return _tina4_write(tina4_project, last_user, messages)
        except Exception as exc:
            return f"Could not write the Tina4 files: {exc}"

    # An open file may import/reference LOCAL modules that aren't open — pull those in too, so the
    # answer can follow the code's own dependencies.
    if open_files.strip():
        open_files = _with_referenced(open_files)

    # "clone <git url>" → actually clone + index it with the harness (not just print a command).
    if re.search(r"\b(clone|open|load|index)\b", last_user, re.I):
        gm = _GIT_URL.search(last_user)
        if gm and ("git@" in gm.group(1) or "github.com" in gm.group(1) or gm.group(1).endswith(".git")):
            url = gm.group(1)
            _status("⬇️ cloning the repo…")
            try:
                from . import repo

                dest, n = repo.open_repo(url)
                _status("📚 indexing the code…")
                return (f"Cloned and indexed **{dest.name}** — {n} files. Now you can ask about the "
                        f"code, search it, or review a file, e.g.\n\n```python\n"
                        f"from harness.repo import ask_repo, list_files, explain_file\n"
                        f"list_files('*.py')\n"
                        f"print(explain_file('Factorial.py'))\n```")
            except Exception as exc:
                return f"Could not clone that repo: {exc}"

    # "new tina4 app called X" → provision + serve + open the preview, all from the chat.
    if _is_new_app_request(last_user):
        try:
            return _new_tina4_app(last_user)
        except Exception as exc:
            return f"Could not provision the Tina4 app: {exc}"

    # "load / open / show <file>" → read the file (notebook space OR open repo) and return its
    # source for direct cell insertion. No model, no comprehension — just the file. Returns None
    # (falls through to Q&A) when the message isn't actually naming a file.
    loaded = _maybe_load(last_user, messages)
    if loaded is not None:
        return loaded

    # "example of / how do I call X" about a function in the notebook → build a NEW cell calling it.
    example = _maybe_example(last_user, notebook_code, open_files, messages)
    if example is not None:
        return example

    # Route to repo Q&A ONLY for code/repo questions — NOT general conversation. Because grounding is
    # always on (the whole workspace), a bare "what is…" must not hijack a chat question like
    # "what is my favourite number?". Trigger on an explicit code/repo signal, OR an exploration
    # question that names a real file/function in the workspace.
    try:
        from . import repo as _repo

        strong = re.search(
            r"\b(repos?|repositor\w*|codebase|folders?|director\w*|structure|organi[sz]\w*|overview|"
            r"layout|which files?|the code|this (?:repo|project|codebase|file)|"
            r"in (?:the|this) (?:repo|project|codebase))\b"
            r"|\blist\b.{0,25}\b(?:files?|folders?|programs?|scripts?|modules?|functions?|classes)\b"
            r"|\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|php|c|cpp|cs|swift|kt|html|css|sql|sh)\b",
            last_user, re.I,
        )
        explore = re.search(r"\b(what|how|where|explain|summar|describe|review|does|show)\b", last_user, re.I)
        refs_code = False
        if explore and not strong:
            # Only treat it as a REPO question when the user names a DISTINCTIVE file in the workspace.
            # Common words ("file", "read", "websocket", "server") happen to match sample files and
            # hijack general concept questions — those belong in the normal (stdlib-grounded) answer.
            toks = [t for t in re.findall(r"[A-Za-z_]\w{2,}", last_user)
                    if t.lower() not in _CALL_STOPWORDS and t.lower() not in _COMMON_WORDS
                    and len(t) >= 5][:6]
            try:
                refs_code = any(_repo.find_file(t, exact=True) or _repo.find_file(t + ".py", exact=True)
                                for t in toks)
            except Exception:
                refs_code = False
        pinned = bool(context.strip() and intent == "ask" and _extract_assertion(last_user))
        # With a cell SELECTED, "explain the code" / "what does this do" means THAT cell — not the
        # repo. Only an EXPLICIT repo/file/folder signal (a filename, 'repo', 'codebase', 'list the
        # files', …) routes to repo Q&A then; the generic 'the code'/'this' stays on the selection.
        has_sel = bool(context.strip()) and intent == "ask"
        explicit_repo = bool(re.search(
            r"\b(repos?|repositor\w*|codebase|folders?|director\w*|structure|layout|overview|"
            r"organi[sz]\w*)\b"
            r"|\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|php|c|cpp|cs|swift|kt|html|css|sql|sh)\b"
            r"|\blist\b.{0,25}\b(?:files?|folders?|programs?|scripts?|modules?|functions?|classes)\b"
            r"|\bin (?:the|this) (?:repo|project|codebase)\b",
            last_user, re.I,
        ))
        # A question that names tina4 is a FRAMEWORK question → tina4 docs, never repo Q&A (unless an
        # explicit repo/file signal is present).
        tina4_q = bool(re.search(r"\btina4\b", last_user, re.I))
        # A request to AUTHOR code ("write me a method", "create a function to test the code") is a
        # generation task, not a repository question — an incidental "the code"/"this file" in it must
        # not hijack it to repo Q&A. Only an EXPLICIT repo/file signal still routes a writing request.
        writing_code = bool(re.search(
            r"\b(write|create|make|build|implement|generat\w*|code)\b.{0,40}?\b"
            r"(method|function|class|script|program|def|algorithm|snippet|test|cell|code)\b",
            last_user, re.I))
        route_repo = explicit_repo or (
            not has_sel and not tina4_q and not writing_code and (strong or (explore and refs_code)))
        if route_repo and not pinned and _repo.ensure_indexed():
            _status("🔎 searching the repo…")
            # history → keeps follow-ups in context; open_files → aware of what's open on screen.
            return _MODE_ASK + _repo.ask_repo(last_user, max_tokens=600, history=messages,
                                              open_code=open_files)
    except Exception:
        pass

    # Known non-Python syntax (??, &&, ||, ===) in the code being fixed → fix it deterministically,
    # no model guessing. This is always correct, which matters for beginners.
    if intent == "code" and context.strip():
        det = _deterministic_fix(context)
        if det != context and _valid_python(det)[0] and not _valid_python(context)[0]:
            _status("🔧 rewriting non-Python syntax…")
            return "Rewrote the non-Python syntax to valid Python:\n\n```python\n" + det.strip() + "\n```"
    # Qwen's template allows only ONE system message, so fold everything into it.
    system_text = _CHAT_SYSTEM
    # Give the model the real "now" so time-related code/answers aren't hallucinated dates.
    from datetime import datetime

    system_text += (
        f"\n\nThe current date and time (the user's 'now') is "
        f"{datetime.now().astimezone().isoformat(timespec='seconds')}. Use it when the request "
        "involves the current time/date; prefer computing it in code (e.g. datetime.now()) over "
        "hardcoding a literal."
    )
    # A multi-cell request ("add 2 cells — one for code and one to test it") → emit ONE code block
    # PER cell; the notebook turns each block into its own cell.
    multi_cell = bool(re.search(r"\b(\d+|two|three|four|several|multiple|a couple of)\s+cells?\b", last_user, re.I)) or bool(
        re.search(r"\bone\s+(?:cell\s+)?(?:for|to)\b.+\b(?:and|,|then|plus|another)\b.+"
                  r"(?:one|another|a\s+second|the\s+other|to\s+test|for\s+test|test)\b", last_user, re.I))
    if multi_cell:
        system_text += _prompts.MULTI_CELL
    # Ground in Tina4 when we're in a Tina4 project OR the question is explicitly ABOUT tina4 (so
    # "tell me about tina4" is answered from the bundled docs, not the 2B's guesswork). LEAN ON the
    # docs only when the request needs guidance — not on every coding task — so documentation prose
    # never gets mistaken for code to write. The 🤖 marker signals "Tina4 mode" in the chat.
    # Sticky topic: tina4-mode holds across the conversation, so a follow-up that doesn't repeat the
    # word "tina4" ("why not dependencies?", "what port does it use?") still grounds in the tina4
    # docs. Include BOTH roles in the recent window — my own answers keep naming Tina4, so the topic
    # persists until the user clearly moves on (several non-tina4 turns).
    recent_text = " ".join((m.get("content") or "") for m in messages[-12:])
    asks_tina4 = bool(re.search(r"\btina4\b", recent_text, re.I))
    explicit_tina4 = bool(re.search(r"\btina4\b", last_user, re.I)) or tina4_project is not None
    # A clearly PERSONAL / recall question is never a tina4 question, even mid-tina4-chat — don't
    # ground it in docs or it will "answer only from the docs" and forget the conversation fact.
    personal = bool(re.search(
        r"\b(my|favou?rite|remember|recall|i said|i told|what'?s my|who am i|about me)\b",
        last_user, re.I))
    tina4_mode = explicit_tina4 or (asks_tina4 and not personal)
    mark = "🤖 " if tina4_mode else ""
    if tina4_mode:
        try:
            from pathlib import Path as _Path

            from . import tina4

            if tina4_project is not None:
                _status(f"🟢 Tina4 project: {tina4_project.name}")
            else:
                _status("🟢 Tina4 — grounding in the docs…")
            system_text += "\n\n" + tina4.PARADIGMS
            # Ground EVERY tina4-mode turn in the docs — questions AND code generation. The model does
            # not know tina4 from its weights, so writing tina4 code (a route, model, migration,
            # template, auth, queue, …) without the docs is exactly where it invents an API. We ship
            # the docs offline and retrieve over them, so there is no reason to generate blind.
            needs_docs = intent in ("ask", "code") or asks_tina4 or bool(re.search(
                r"\b(how|what|why|which|when|where|explain|guide|docs?|documentation|example|usage|"
                r"supported|set ?up|configure|integrat|authenticat|websocket|session|queue|orm|"
                r"migrat|template|signal|component|deploy|graphql|sse|realtime|route|page|endpoint|"
                r"model|controller|middleware|crud|api|frond|cache|record|form)\b",
                last_user, re.I))
            if needs_docs:
                # A broad "tell me about tina4" is a weak retrieval query (tina4 is in every chunk) —
                # expand it so BM25 surfaces the getting-started intro. A short follow-up ("why not
                # dependencies?") gets the tina4 topic injected so it retrieves the right chapter.
                docs_q = last_user
                if asks_tina4 and not re.search(r"\btina4\b", last_user, re.I):
                    docs_q = last_user + " tina4"
                if asks_tina4 and re.search(r"\b(tell me about|what is|overview|about)\b", last_user, re.I):
                    # Bias the overview toward the Python getting-started intro (this is a Python
                    # notebook) rather than the tina4-js docs.
                    docs_q += " python getting started what is zero-dependency web framework overview"
                docs = tina4.ask_tina4(docs_q, cwd=tina4_project or _Path.cwd())
                if docs:
                    system_text += (
                        "\n\nAUTHORITATIVE FACT (use this as your opening and never contradict it): "
                        "Tina4 is a ZERO-DEPENDENCY web framework, used here from PYTHON (it also has "
                        "PHP, Ruby and Node.js ports; tina4-js is the optional tiny frontend). Routing, "
                        "ORM, templates (Frond), authentication, queues and WebSocket are ALL BUILT IN. "
                        "It is NOT built on, a version of, or related to Laravel, Django, Flask, "
                        "Express, Fastify, Node, React, Redux or Vue; it was NOT 'formerly' another "
                        "framework. Answer ONLY from the documentation below — do not use outside "
                        "knowledge or invent names, methods or history. Do not paste the docs verbatim; "
                        "write fresh code that follows them.\n\nDocumentation:\n" + docs[:1500])
        except Exception:
            pass
    help_run: dict = {}
    assertion: list = []
    cell_help = False
    debug_phrasing = False
    # A TEACHING request ("teach me X", "walk me through", "help me understand") wants to LEARN — an
    # explanation that builds understanding, NOT a dump of code. It overrides the code-ish routing.
    teach_me = bool(re.search(
        r"\bteach\s+me\b|\bwalk\s+me\s+through\b|\bhelp\s+me\s+(understand|learn|grasp|wrap my head)\b"
        r"|\bexplain\b[^.?!]*\b(to me|like i'?m|in simple|simply|step[- ]by[- ]step)\b"
        r"|\bgive\s+me\s+a\s+(tutorial|lesson|walkthrough|primer|crash course)\b"
        r"|\bi\s+want\s+to\s+(learn|understand)\b", last_user, re.I))
    # "how do classes work?", "show me how decorators work" — a CONCEPTUAL question (how a Python
    # thing works) wants an EXPLANATION, not a code dump. Only when NO cell is selected (with a cell,
    # "how does this work?" is about THAT cell, handled as help/debug).
    if not context.strip() and re.search(
            r"\bshow me how\b.{0,40}\bworks?\b|\bhow\s+(?:do|does)\b[^?]{0,40}\bworks?\b"
            r"|\bhow\s+(?:do|does)\s+(?:a|an|the)?\s*\w+s?\s+work\b", last_user, re.I) and not re.search(
            r"\b(this|it|that|the\s+(?:code|cell|function|method|loop|script))\b", last_user, re.I):
        teach_me = True
    if intent == "ask":
        # A bare "help" (with a selected cell) means: explain THIS cell and flag any problems, and —
        # if a change is warranted — hand back the COMPLETE corrected cell so the UI can apply it.
        is_help = bool(context.strip() and re.match(
            r"^\s*(?:please\s+|pls\s+|plz\s+)?(?:can\s+you\s+|could\s+you\s+|would\s+you\s+)?h[ae]lp\b",
            last_user, re.I,
        ))
        # A DEBUGGING question ("what's wrong?", "why doesn't this work?", "is this right?", "find the
        # bug") is help too — EXPLAIN the problem and offer the fix, never generate a new cell. (With a
        # cell selected it validates + offers a fix; without one it still just explains.)
        debug_phrasing = bool(re.search(
            r"\bwhat'?s?\s+(wrong|the\s+(bug|problem|issue|mistake))\b|\bwhat\s+is\s+wrong\b"
            r"|\bwhy\s+(is|isn'?t|does|does\s*n'?t|wo\s*n'?t|doesnt|isnt)\b"
            r"|\b(spot|find|catch|see)\s+the\s+(bug|problem|issue|mistake)\b"
            r"|\bis\s+(this|that|it|my\s+code)\s+(right|correct|wrong|broken|ok|okay|a\s+bug)\b"
            r"|\b(debug|diagnose)\b|\b(not|n'?t)\s+working\b|\bdoes\s*n'?t\s+work\b|\bbroken\b",
            last_user, re.I))
        is_debug_q = debug_phrasing and bool(context.strip())
        # A pinned expectation ("add(3,4)==7", "assert …") turns help into a real test oracle.
        assertion = _extract_assertion(last_user) if context.strip() else []
        cell_help = is_help or is_debug_q or bool(assertion)
        if cell_help:
            # Run the cell so the model's explanation is grounded in reality. The real decision about
            # whether there's a bug is made AFTER generation by running the proposal (and the pinned
            # assertion) — validation, not prompting.
            _status("▶ running the cell…")
            # Run the cell to ground the explanation in reality — WITH the notebook's other defs in
            # scope. For a cell that only DEFINES functions (no output alone), also CALL them with
            # sample inputs so the model sees what they ACTUALLY return — that's how "what's wrong?"
            # gets diagnosed (e.g. add_numbers(2, 3) returns 3, not 5).
            try:
                nb_defs = _defs_only(notebook_code).strip() if notebook_code else ""
            except Exception:
                nb_defs = ""
            pre = (nb_defs + "\n") if nb_defs else ""
            help_run = _run_in_kernel(pre + context)
            probe = _probe_calls(context)
            probe_run = _run_in_kernel(pre + context + "\n" + probe) if probe else None
            system_text += (
                _prompts.HELP
            )
            if assertion:
                base0 = _assert_report(context, assertion)
                spec = "; ".join("assert " + c for c in assertion)
                system_text += (
                    f"\n\nThe cell MUST satisfy this check that the user pinned: {spec}\n"
                    f"Running the cell now, the check currently {base0['status'].upper()} "
                    f"({'; '.join(base0['lines'])}). If it does NOT all pass, return the COMPLETE "
                    "corrected cell as ONE ```python block that makes every check pass (keep the public "
                    "function names the same). If it already passes, output no code block."
                )
            if help_run.get("ran") and help_run.get("skipped") is None:
                system_text += "\n\nFor reference, running this cell just now: " + _run_summary(help_run)
            if probe_run and (probe_run.get("out") or "").strip():
                system_text += (
                    "\n\nI CALLED its function(s) with sample inputs and got these ACTUAL results — use "
                    "them to judge whether it is correct (e.g. a result that isn't the real sum/answer "
                    "means a bug):\n" + probe_run["out"].strip()[:600]
                )
        else:
            overview = bool(re.search(
                r"\b(tell me about|overview|what(?:'s| is| are)|give me a rundown|rundown|summar\w*|"
                r"introduc\w*|explain .{0,20}\b(framework|library|project|tina4))\b", last_user, re.I))
            if teach_me:
                system_text += _prompts.TEACH
            elif overview:
                system_text += _prompts.OVERVIEW
            else:
                system_text += _prompts.QUESTION
    # "call / invoke / run the function" → just CALL an already-defined function, don't define a new one.
    if intent != "ask" and re.search(r"\b(call|invoke|run)\b", last_user, re.I):
        system_text += _prompts.CALL
        # Ground the model in the REAL definition of whatever method the user named — any name, any
        # file — so it calls it with the correct signature instead of guessing.
        seen_here = (notebook_code or "") + "\n" + (open_files or "")
        names = [t for t in re.findall(r"[A-Za-z_]\w{2,}", last_user)
                 if t.lower() not in _CALL_STOPWORDS and ("def " + t) not in seen_here
                 and ("class " + t) not in seen_here]
        if names:
            try:
                from . import repo as _repo

                _repo.ensure_indexed()
                defs = _repo.find_definitions(names)
                if defs:
                    blocks = "\n\n".join(f"# {path}\n{snip}" for _n, (path, snip) in defs.items())
                    system_text += ("\n\nThese are the ACTUAL definitions to call (from the project) — "
                                    "use the real name and parameters:\n```python\n" + blocks + "\n```")
            except Exception:
                pass
    # Ground a BUILD request in what ALREADY exists in the WORKING FOLDER — look at the project's real
    # code (existing models, routes, helpers) and reuse it, instead of inventing. Only for a FRESH
    # build (no cell selected): when a cell IS selected the user is editing THAT code, and other
    # project files are noise that can derail the edit (e.g. 'remove the print' must not re-add one).
    # Retrieval-gated: only inject chunks that share a meaningful term with the request.
    if intent == "code" and not context.strip():
        try:
            from . import repo as _repo

            if _repo.ensure_indexed():
                q_terms = {w for w in re.findall(r"[A-Za-z_]\w{3,}", last_user.lower())
                           if w not in _COMMON_WORDS and w not in _CALL_STOPWORDS}
                seen = (notebook_code or "") + (open_files or "")
                picked = []
                for text, _score in _repo.search_repo(last_user, k=4):
                    body = (text or "").strip()
                    if not body or body[:160] in seen:
                        continue
                    if q_terms and not (q_terms & set(re.findall(r"[A-Za-z_]\w{3,}", body.lower()))):
                        continue  # no shared term → not actually relevant
                    picked.append(body[:800])
                    if len(picked) >= 3:
                        break
                if picked:
                    system_text += (
                        "\n\nRELEVANT CODE ALREADY IN THE WORKING FOLDER — reuse these names, models and "
                        "patterns and follow the project's conventions; do NOT redefine or reinvent what "
                        "already exists, and do NOT invent names that aren't here:\n```python\n"
                        + "\n\n# ---\n".join(picked) + "\n```")
        except Exception:
            pass
    # A CONCEPTUAL question (an ask that isn't help/fix and doesn't point at code) is answered from
    # the CONVERSATION + docs — not from stale notebook/cell/open-file code that would hijack it
    # (e.g. "why not dependencies?" must not get answered with the selected greet cell).
    # Explicit code nouns ALWAYS point at the cell ("explain the code"). Bare pronouns ("it", "this")
    # only do so OUTSIDE tina4-mode — in a tina4 chat "why IT has no dependencies" means the
    # framework, not the selected cell.
    strong_code_ref = bool(re.search(
        r"\b(the code|the cell|the function|the method|the class|the selection|my code|the error|"
        r"the output|the loop|the variable|the snippet|the file)\b", last_user, re.I))
    weak_ref = bool(re.search(r"\b(this|that|it|here|above|below)\b", last_user, re.I))
    references_code = strong_code_ref or (weak_ref and not asks_tina4)
    # A question asked while the cell shows an ERROR, or a "what's going on / not working / why isn't
    # this working" question with a cell selected, is about THAT cell/error — NOT a concept. Withhold
    # the code only for a genuinely general question (no error, no cell reference).
    output_err = bool(output) and bool(re.search(r"error|traceback|exception", output, re.I))
    situational = bool(re.search(
        r"\bwhat(?:'?s| is| are)?\s+(going on|happening|wrong|the problem|the issue)\b"
        r"|\b(not|n'?t)\s+work(ing|s)?\b|\bdoes\s*n'?t\s+work\b|\bwhy (is|isn'?t|does|does\s*n'?t|won'?t)\b"
        r"|\bbroken\b|\bfailing\b|\bwrong with\b", last_user, re.I))
    conceptual = (intent == "ask" and not cell_help and not references_code
                  and not output_err and not (situational and context.strip()))
    # Persisting thread memory: a compact running summary of earlier turns so a follow-up stays on
    # topic after the original turn scrolls out of the verbatim window.
    digest = _conversation_digest(messages)
    if digest:
        system_text += ("\n\nConversation so far — use it to resolve short references ('it', 'that') "
                        "and to RECALL anything the user told you earlier (e.g. a fact or preference). "
                        "If they ask you to recall it, answer from here:\n" + digest)
    # ALWAYS let the model SEE what the user is looking at — the cell, its error, the notebook, open
    # files — and let IT determine whether the message is about them. We never WITHHOLD the cell on a
    # keyword guess (that made "what is going on?" get a generic non-answer); for a general question we
    # just FRAME the code as reference so it isn't dragged into the answer. `about_code` (= not
    # conceptual) decides the framing, not whether the code is present.
    about_code = not conceptual
    if output:
        if intent == "ask":
            system_text += (
                "\n\nThe active cell's latest OUTPUT / ERROR is below. The user is ASKING about it — "
                "EXPLAIN in plain words what it means and what causes it; LEAD WITH PROSE, not a code "
                "block (a short corrected snippet may come after). The code is already in this message, "
                "so NEVER ask the user to share it:\n" + output[-1800:]
            )
        else:
            system_text += (
                "\n\nThe active cell's latest OUTPUT / ERROR is below. If it is an error, use it to find "
                "the cause and FIX it — the code is already in this message, so NEVER ask the user to "
                "share it again:\n" + output[-1800:]
            )
        # Ground an import/attribute error against the REAL module so the model stops inventing an API
        # (e.g. 'from antigravity import Client') or suggesting an install for a module already present.
        system_text += _ground_import_error(output)
    if open_files.strip():
        # Files the user has OPEN in editor/notebook tabs — be aware of them when answering.
        system_text += (
            "\n\nThe user currently has these FILES OPEN on screen. When they ask about 'the file', "
            "'this code', or a function/name that appears here, they mean THIS — use it to answer:\n"
            "```python\n" + open_files.strip()[:8000] + "\n```"
        )
        # "use / run / call / demo the open file" → adapt it into runnable cell code, don't paste
        # an interactive script (a bare input()/main() just hangs a cell).
        if intent != "ask" and re.search(r"\b(use|run|call|demo|try|execute|exercise|invoke)\b", last_user, re.I):
            system_text += (
                "\n\nThe user wants to USE the open file's logic in this cell. Write a SELF-CONTAINED, "
                "runnable cell that defines whatever functions it needs (adapt them from the open file) "
                "and then CALLS them with a concrete example value, printing the result. Do NOT call "
                "input() — substitute a literal (e.g. 27). Do NOT paste the file's interactive main()/"
                "__main__ block verbatim, and do NOT try to import a file whose name isn't a valid "
                "Python module (e.g. one with a hyphen)."
            )
    # An EDIT request (remove / strip / delete / simplify / change … the selected code) must REWRITE
    # the selected cell, not write a new addition — so it overrides the "don't redefine" guidance.
    is_edit = intent == "code" and bool(context.strip()) and bool(re.search(
        r"\b(remov\w*|delet\w*|strip\w*|get rid|take (out|away)|clean\s*up|tidy|simplif\w*|drop\w*|"
        r"without|fix\w*|edit\w*|updat\w*|chang\w*|refactor\w*|rewrit\w*|modif\w*|replace|rename|"
        r"add\w*|insert\w*|include|wrap|annotat\w*)\b", last_user, re.I))
    if is_edit:
        system_text += (
            "\n\nThe user wants to EDIT the selected code below — apply exactly the change they asked "
            "for (e.g. REMOVE the part they named, such as the example usage / `if __name__` block / "
            "tests / comments) and return the COMPLETE edited code, ready to run. Do NOT add anything "
            "they did not ask for, and do NOT keep the part they asked to remove. Reply with ONE "
            "```python block containing the whole edited cell:\n```python\n" + context.strip() + "\n```"
        )
    elif context.strip():
        if about_code:
            system_text += (
                "\n\nThe user has SELECTED this code in the notebook; when they say 'this', 'it', "
                "'the selection', or 'the error', they mean it — answer about it:\n```python\n"
                + context + "\n```"
            )
        else:
            system_text += (
                "\n\nFOR REFERENCE: the user has this cell selected, but their message reads like a "
                "GENERAL question. Answer the question directly; do NOT use or mention this code unless "
                "it is clearly what they are asking about:\n```python\n" + context + "\n```"
            )
    if notebook_code and not is_edit:
        system_text += (
            "\n\nThe notebook ALREADY contains this code — reuse these names and do NOT rewrite or "
            "redefine them. Write ONLY the new code the user asks for (e.g. just the call, not the "
            "function again). If the user's question is about it, use it to answer:\n```python\n"
            + notebook_code[-4000:] + "\n```"
        )
    # Ground questions AND error-fixes in the stdlib docs (help beginners get correct Python).
    if _looks_like_question(last_user) or output:
        try:
            from . import pyhelp

            _status("📚 grounding in the Python docs…")
            query = (last_user + " " + output.splitlines()[0]) if output else last_user
            docs = "\n\n".join(t[:600] for t, _s in pyhelp.search(query, k=2))
            if docs:
                system_text += "\n\nRelevant Python documentation (use if helpful, ignore if not):\n" + docs
        except Exception:
            pass
    # Endless chat with a BOUNDED prompt: recency (last turns) + Groundwire-retrieved relevant older
    # turns ("rotating/evolving" context). Tune the overall char cap with HARNESS_CONTEXT_CHARS.
    budget = int(os.environ.get("HARNESS_CONTEXT_CHARS", str(60_000)))
    recent = _evolving_context(messages, last_user, max_chars=budget)
    # Ground the CURRENT turn in the code the user is pointing at, so a poisoned history (e.g. earlier
    # "please share the code" deflections) can't make the model claim no code was provided. The code
    # goes INTO the latest user message — but only when the message is about the code (about_code), so
    # a general question keeps the code as system-prompt reference without it being forced into the ask.
    if context.strip() and about_code:
        recent = [dict(m) for m in recent]  # copy so we never mutate the caller's messages
        for m in reversed(recent):
            if m.get("role") == "user" and "```" not in m.get("content", ""):
                m["content"] = (m.get("content", "")
                                + "\n\nThe code I'm referring to (already provided — do NOT ask me to "
                                "share it again; use it directly):\n```python\n"
                                + context.strip()[:4000] + "\n```")
                break
    # A plain explanation is kept SHORT by a small budget (no thinking) — the user asks for more if
    # they want depth. Code and help/fix keep the full budget (they need room for the code). This
    # also starves the small model's tendency to ramble into a repetition loop.
    gen_cap, gen_think = chat_cap, enable_thinking
    if intent == "ask" and not cell_help:
        # A plain explanation is short; a "teach me" explanation gets a little more room to build it.
        gen_cap, gen_think = min(chat_cap, 380 if teach_me else 240), False
    elif cell_help:
        gen_think = False   # help: skip the reasoning phase so it stays terse (the fix code still fits)
    _status("💭 thinking…")
    reply = _strip_think(_generate([{"role": "system", "content": system_text}] + recent, gen_cap,
                                    stream=True, enable_thinking=gen_think))
    # NLP routing by OUTPUT FORM: the model reliably writes CODE for a code request and PROSE for a
    # question, so its own answer decides ask-vs-code — a misrouted intent from the UI self-corrects
    # here. (Help/fix/assertion already fixed intent to 'ask' via cell_help; leave those.)
    if not cell_help:
        if debug_phrasing or teach_me:
            intent = "ask"  # a debugging question / a "teach me" request EXPLAINS — never a new cell
        else:
            # Strict when the caller asked a QUESTION (keep explanations as prose); looser when they
            # already intended code, so a captioned code reply (common on Gemma) still renders as code.
            intent = "code" if _starts_with_code(reply, strict=(intent == "ask")) else "ask"
    if intent == "code":
        _status("🔎 checking syntax…")
    if intent == "ask":
        # A bare "help" on a selected cell: VALIDATE, don't guess. Run the proposed change and compare
        # its output to the original. The execution verdict — not the model's prose — decides whether
        # a bug is real and whether to keep the fix (so the UI offers it) or drop it (no offer).
        if cell_help and context.strip():
            blocks = [b.strip() for b in re.findall(r"```(?:python)?\s*\n(.*?)```", reply, re.S) if b.strip()]
            cand = next((b for b in reversed(blocks) if _valid_python(b)[0]), None)  # proposed cell
            ran_error = bool(help_run.get("error"))
            offer_fix = False  # only a VALIDATED fix sets this → the extension offers Apply then

            # A pinned assertion is the oracle: it, not the model, decides pass/fail and the fix.
            if assertion:
                reply = re.sub(r"```(?:python)?\s*\n.*?```", "", reply, flags=re.S).strip()
                reply = _drop_bug_claims(reply)
                base = _assert_report(context, assertion)
                fix = _assert_report(cand, assertion) if cand else None
                # The code fails the check but no proposal passes it → ask the model ONCE more, aimed
                # squarely at making the assertion pass (fulfilling the user's spec, not coaxing).
                if base["status"] != "pass" and (fix is None or fix["status"] != "pass") and not _stopped:
                    _status("▶ fixing to satisfy the check…")
                    spec = "; ".join("assert " + c for c in assertion)
                    aim = system_text + (
                        f"\n\nReturn ONLY the complete corrected cell as one ```python block so that "
                        f"{spec} all pass. Keep the public function names the same. No prose."
                    )
                    r2 = _strip_think(_generate([{"role": "system", "content": aim}] + recent, chat_cap,
                                                enable_thinking=enable_thinking))
                    b2 = re.findall(r"```(?:python)?\s*\n(.*?)```", r2, re.S)
                    c2 = next((b.strip() for b in reversed(b2) if b.strip() and _valid_python(b.strip())[0]), None)
                    if c2:
                        f2 = _assert_report(c2, assertion)
                        if f2["status"] == "pass":
                            cand, fix = c2, f2
                if base["status"] == "pass":
                    reply = (reply + "\n\n" if reply else "") + \
                        f"_✅ Checked your assertion by running it: {'; '.join(base['lines'])} — it holds._"
                elif fix and fix["status"] == "pass":
                    _status("▶ validated against your check…")
                    reply = (reply + "\n\n" if reply else "") + (
                        f"_▶ Validated against your check — **before:** {'; '.join(base['lines'])} · "
                        f"**after:** {'; '.join(fix['lines'])}._"
                    )
                    reply += "\n\n```python\n" + cand + "\n```"
                    offer_fix = True
                else:
                    reply = (reply + "\n\n" if reply else "") + (
                        f"_❌ Your check fails: {'; '.join(base['lines'])}. I couldn't produce a change "
                        "that makes it pass — check the expected value, or tell me more._"
                    )
                return (_FIX_SENTINEL if offer_fix else "") + _cap_prose(reply, 8000)

            verdict = _fix_verdict(context, cand) if cand else None
            # A fix is offered ONLY when running it proves it helps: it clears a real error, or it
            # changes the output. "cosmetic"/"fix_breaks" are never offered.
            real_fix = verdict in ("fixes_error", "changes_behaviour")

            reply = re.sub(r"```(?:python)?\s*\n.*?```", "", reply, flags=re.S).strip()
            # For a plain "help" we suppress unproven bug claims (don't invent bugs). But when the user
            # EXPLICITLY asks "what's wrong?", keep the model's assessment — they want it.
            if not is_debug_q:
                reply = _drop_bug_claims(reply)
            if real_fix:
                _status("▶ validated the fix…")
                # Keep the proposal as THE code block so the extension offers to apply it, with the
                # before/after evidence from actually running both.
                probe = _probe_calls(context)
                before = _run_summary(_run_in_kernel(context + ("\n" + probe if probe else "")))
                after = _run_summary(_run_in_kernel(cand + ("\n" + probe if probe else "")))
                reply = (reply + "\n\n" if reply else "") + (
                    f"_▶ Validated by running both — **before:** {before} · **after:** {after}._"
                )
                reply += "\n\n```python\n" + cand + "\n```"
                offer_fix = True
            elif ran_error:
                # Genuinely broken (it raised), but no suggestion was validated to resolve it — report
                # the real error honestly instead of offering a fix that doesn't actually fix it.
                extra = " The suggested change didn't resolve it when run." if cand else ""
                reply = (reply + "\n\n" if reply else "") + (
                    f"_▶ Running it raised: {help_run.get('error')}.{extra} Tell me the behaviour you "
                    "want and I'll fix it._"
                )
            elif is_debug_q:
                # The user asked "what's wrong?" but nothing was PROVEN broken by running it. Show the
                # ACTUAL behaviour and let them judge against intent — don't claim "nothing to fix".
                probe = _probe_calls(context)
                shown = ""
                if probe:
                    pr = _run_in_kernel((nb_defs + "\n" if nb_defs else "") + context + "\n" + probe)
                    if (pr.get("out") or "").strip():
                        shown = pr["out"].strip()[:300].replace("\n", " · ")
                tail = (f"_▶ What it actually does: {shown}. I couldn't prove a bug by running it — "
                        "check that's the behaviour you intended; if not, tell me the expected result._"
                        if shown else "_▶ It runs without error — tell me the behaviour you expected and "
                        "I'll pin down the problem._")
                reply = (reply + "\n\n" if reply else "") + tail
            else:
                # Ran clean and nothing changes the output → there is no bug.
                ran = _run_summary(help_run) if (help_run.get("ran") and not help_run.get("skipped")) else "it is valid Python"
                if verdict in ("cosmetic", "fix_breaks"):
                    detail = "a suggested rewrite produces the exact same output, so there's nothing to fix"
                else:  # the model proposed no change at all
                    detail = "I didn't find a behaviour problem — if it should do something different, tell me"
                reply = (reply + "\n\n" if reply else "") + f"_✅ Validated by running it: {ran} — {detail}._"
            return (_FIX_SENTINEL if offer_fix else "") + _cap_prose(reply, 8000)
        # The small model sometimes answers a vague prompt with a TEMPLATE of an answer in brackets
        # ("[Analyze the situation, consider multiple hypotheses…]") instead of answering. Don't show
        # that — retry once, firmly, for a direct answer; if it still stalls, ask the user to be
        # specific rather than printing a placeholder.
        if _is_prose_placeholder(reply) and not _stopped:
            _status("↻ answering directly…")
            firm = system_text + ("\n\nAnswer the user's question DIRECTLY in plain prose. Do NOT "
                "describe what a good answer would contain, do NOT output a template or bracketed "
                "placeholder, do NOT use square brackets around meta-instructions — just answer.")
            reply2 = _strip_think(_generate([{"role": "system", "content": firm}] + recent, gen_cap,
                                            enable_thinking=False))
            if reply2.strip() and not _is_prose_placeholder(reply2):
                reply = reply2
            else:
                reply = ("I need a bit more to go on — tell me what you'd like (e.g. a specific "
                         "question, a cell to look at, or 'describe the open image').")
        # A clarifying question with a short option list → let the UI show clickable pills.
        choices = _as_choices(reply)
        if choices:
            import json as _json
            return _CHOICES_SENTINEL + _json.dumps(choices)
        # Keep explanations short (the user can ask for more); code/help kept full above.
        return _MODE_ASK + _with_marker(mark, _cap_prose(reply, 1200 if teach_me else 700))
    tidied = _tidy_chat(reply, as_tool="tool" in last_user.lower())
    # If the model stubbed or wrote invalid Python, retry once (ask the model — no canned reply).
    m = _FENCE.search(tidied)
    if m:
        bad_placeholder = _is_placeholder(m.group(1))
        ok, err = _valid_python(m.group(1))
        if (bad_placeholder or not ok) and not _stopped:
            _status("↻ fixing the code…")
            why = "it was a placeholder" if bad_placeholder else f"it had a syntax error: {err}"
            retry = recent + [
                {"role": "assistant", "content": tidied},
                {"role": "user", "content": f"That is not acceptable ({why}). {_py_hint(m.group(1))}. "
                                            "Return ONLY the actual, complete, valid Python code."},
            ]
            reply2 = _strip_think(_generate([{"role": "system", "content": system_text}] + retry, chat_cap,
                                            enable_thinking=enable_thinking))
            tidied2 = _tidy_chat(reply2, as_tool="tool" in last_user.lower())
            m2 = _FENCE.search(tidied2)
            if m2 and not _is_placeholder(m2.group(1)) and _valid_python(m2.group(1))[0]:
                tidied = tidied2
    # A fix/edit request WANTS to redefine (the UI replaces the existing cell). A plain "add/use"
    # request should NOT redefine — strip any duplicate definition of names already in the notebook.
    is_revision = is_edit or bool(re.search(
        r"\b(fix|edit|updat|chang|refactor|improv|rewrit|modif)\w*", last_user, re.I))
    if notebook_code and not is_revision:
        tidied = _strip_redefs_in_reply(tidied, _defined_names(notebook_code))
    # SELF-CHECK: run newly generated code BEFORE it is pasted. If it crashes (a hallucinated import
    # or API is the common cause), ground the error against reality and retry; after a couple of
    # honest tries that still fail, DON'T pretend it works — hand the attempt back as a chat message
    # with the real error and ask the user, instead of auto-inserting broken code. (The user is
    # working WITH the harness toward a solution.) Edits/fixes are skipped (they lean on prior cells
    # and are validated separately); risky side-effecting code is skipped rather than run silently.
    norun = False
    if intent == "code" and not is_revision and not _stopped:
        mv = _FENCE.search(tidied)
        code0 = mv.group(1) if mv else ""
        if code0 and _valid_python(code0)[0] and not _safe_to_run(code0):
            # Can't validate it by running (it makes a network call, writes a file, reads stdin, …).
            # Honour "check before run": insert it, but DON'T auto-run code we never checked.
            norun = True
        elif code0 and _valid_python(code0)[0] and _safe_to_run(code0):
            # Run the check WITH the notebook's AND the selected cell's definitions in scope
            # (imports/defs/classes only, so we don't re-run side-effecting top-level code), so
            # generated code that CALLS a function defined elsewhere — e.g. "write a test for this
            # function" / "test my add_numbers function" — resolves it instead of a false NameError
            # when that cell hasn't been run yet.
            try:
                nb_defs = _defs_only((notebook_code or "") + "\n" + (context or "")).strip()
            except Exception:
                nb_defs = ""
            _check = lambda c: _run_in_kernel((nb_defs + "\n" + c) if nb_defs else c)  # noqa: E731
            run = _check(code0)
            err = run.get("error") if not run.get("skipped") else None
            # A cell that is ONLY a docstring/comments runs clean but DOES nothing — treat it as a
            # failure so the retry writes real code ("demo the classes" must instantiate + call them).
            if not err and _is_noop_code(code0):
                err = ("the code does NOTHING — it is only a docstring/comments. Write ACTUAL runnable "
                       "code that does what was asked (e.g. instantiate the classes and CALL their "
                       "methods, printing results)")
            best, attempts = tidied, 0
            while err and attempts < 2 and not _stopped:
                attempts += 1
                _status("↻ checking my code by running it…")
                fixsys = system_text + (
                    f"\n\nYour previous code was NOT acceptable: {err}\nReturn the COMPLETE code as ONE "
                    "```python block that RUNS with no error AND actually does the task. Use ONLY real, "
                    "existing names — do NOT invent imports, classes, methods or attributes, and do NOT "
                    "write only a docstring or a description."
                    + _ground_import_error(err))
                try:
                    rt = _tidy_chat(_strip_think(_generate(
                        [{"role": "system", "content": fixsys}] + recent, chat_cap,
                        enable_thinking=enable_thinking)), as_tool="tool" in last_user.lower())
                except Exception:
                    break
                mr = _FENCE.search(rt)
                if (not mr or _is_placeholder(mr.group(1)) or not _valid_python(mr.group(1))[0]
                        or _is_noop_code(mr.group(1))):
                    continue
                best = rt
                r2 = _check(mr.group(1))
                if not (r2.get("error") and not r2.get("skipped")):
                    tidied, err = rt, None   # verified: runs clean and does something
                    break
                err = r2.get("error")
            if err:
                # Honest deferral: show the best attempt + the real error, ask the user — don't paste.
                last = err.strip().splitlines()[-1][:200]
                ground = _ground_import_error(err)
                why = ""
                if "GROUNDED FACT" in ground:
                    why = " " + ground.split("GROUNDED FACT", 1)[1].split(":", 1)[-1].strip()[:260]
                b = _FENCE.search(best)
                attempt = b.group(1) if b else code0
                note = (f"I checked this by running it and it still fails: `{last}`.{why}\n\nI couldn't "
                        "get it to run cleanly on my own — here's my best attempt. Can you point me at "
                        "the right approach (or confirm the API)?")
                return _MODE_ASK + _with_marker(mark, note + f"\n\n```python\n{attempt}\n```")
    # "add 2 cells — one for code and one to test it": the 2B won't emit two blocks, so split the
    # function into its own cell and generate a REAL test (assert) for it as a second cell.
    if multi_cell and re.search(r"\btest", last_user, re.I):
        m = _FENCE.search(tidied)
        if m:
            fn_code = _defs_only(m.group(1)).strip() or m.group(1).strip()
            # Show the model a COMPACT version (docstrings stripped) so it's less tempted to re-emit a
            # long function and overflow the token budget — the full function is already in cell 1.
            try:
                import ast as _ast
                fn_for_prompt = _ast.unparse(_strip_docstrings(_ast.parse(fn_code)))
            except Exception:
                fn_for_prompt = fn_code
            base_sys = _CHAT_SYSTEM + "\n\n" + _prompts.PYTEST_FOR_FN + fn_for_prompt + "\n```"
            # A tolerant runner so the pytest-style `def test_*` functions EXECUTE when the cell runs
            # (pytest isn't invoked here): skip any that need fixtures/params, and report ✓/✗ per test
            # without aborting the cell.
            runner = ("\n\nimport inspect as _inspect\n"
                      "for _n, _f in list(globals().items()):\n"
                      "    if _n.startswith('test_') and callable(_f) and not _inspect.signature(_f).parameters:\n"
                      "        try:\n"
                      "            _f()\n            print('\u2713', _n)\n"
                      "        except AssertionError as _e:\n            print('\u2717', _n, _e)")
            test_only, err, last_cand = "", "", ""
            for attempt in range(3):
                sys_prompt = base_sys if attempt == 0 else base_sys + (
                    f"\n\nYour previous attempt was WRONG: {err}\nWrite SIMPLE `def test_x():` functions "
                    "with plain `assert` lines — NO decorators, NO @pytest.mark, NO fixtures, NO params.")
                try:
                    # temperature 0 → follow the rigid template instead of wandering into fixtures.
                    tr = _tidy_chat(_strip_think(_generate(
                        [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": "write the test"}], 512, enable_thinking=False,
                        temperature=0.0)), as_tool=False)
                except Exception:
                    break
                tm = _FENCE.search(tr)
                cand = (tm.group(1).strip() if tm else "")
                if not (cand and _valid_python(cand)[0]):
                    continue
                cand = _drop_redefs(cand, _defined_names(fn_code)).strip() or cand
                last_cand = cand
                # The 2B loves fixtures/parametrize that don't run when called directly — reject and retry.
                if re.search(r"@\s*pytest\.(fixture|mark)", cand):
                    err = "used @pytest.fixture / @pytest.mark — not allowed"
                    continue
                if "pytest" in cand and not re.search(r"(?m)^\s*import pytest\b", cand):
                    cand = "import pytest\n\n" + cand
                r = _run_in_kernel(fn_code + "\n" + cand + runner)
                if not (r.get("error") and not r.get("skipped")) and "\u2713" in (r.get("out") or ""):
                    test_only = cand + runner  # at least one test ran green
                    break
                err = (r.get("error") or "no test executed").strip().splitlines()[-1][:160]
            if not test_only:
                # Robust fallback: the 2B can't be trusted to write a whole pytest file, but it CAN
                # tell us the value a call SHOULD return. Ask for that (easy to parse) and BUILD a clean
                # pytest test ourselves — valid every time, with the model-reasoned EXPECTED value (so a
                # buggy function still FAILS the test — never characterization of the actual output).
                test_only = _pytest_from_expected(fn_code)
            if test_only:
                tidied = f"```python\n{fn_code}\n```\n\n```python\n{test_only}\n```"
    return _MODE_CODE + (_NORUN if norun else "") + _with_marker(mark, tidied)


def _looks_like_question(text: str) -> bool:
    t = text.lower().strip()
    return ("?" in t) or t.startswith(
        ("what", "how", "why", "which", "when", "where", "explain", "is ", "does ", "can ", "tell me")
    )


def _resolve_image(image: str):
    """Find an image whose path is given relative to the Jupyter ROOT, from a kernel whose cwd may
    be a sub-folder (so 'notebooks/x.jpeg' must not become 'notebooks/notebooks/x.jpeg')."""
    from pathlib import Path

    p = Path(image)
    if p.is_absolute():
        return p if p.is_file() else None
    name = p.name
    bases = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    # Try the given relative path against each base, then just the basename.
    for base in bases:
        for cand in (base / image, base / name):
            if cand.is_file():
                return cand
    # Last resort: search for the basename nearby.
    for base in (Path.cwd(), Path.cwd().parent):
        try:
            hit = next((h for h in base.rglob(name) if h.is_file()), None)
        except Exception:
            hit = None
        if hit:
            return hit
    return None


def _vision(image: str, question: str) -> str:
    """Answer a question about an image — using the SAME model as text (Qwen3.8-2B-Distill is a VLM)."""
    path = _resolve_image(image)
    if path is None:
        return f"Could not find the image '{image}'."

    model = get_model()
    prompt = question or "Describe this image."
    _status("🖼 looking at the image…")
    with _lock:
        if hasattr(model, "describe"):          # the MLX model does text AND vision
            out = model.describe(str(path), prompt, max_tokens=512)
        else:
            from .vision import VisionModel      # torch fallback
            global _vision_fallback
            if _vision_fallback is None:
                _vision_fallback = VisionModel()
            out = _vision_fallback.describe(str(path), prompt, max_tokens=512)
    _log_interaction("vision", [{"role": "user", "content": prompt, "image": str(path)}], out)
    return out


def _chat_file(key: str):
    import hashlib
    import re as _re
    from .download import PROJECT_ROOT

    d = PROJECT_ROOT / ".tina4" / "chat"
    d.mkdir(parents=True, exist_ok=True)
    safe = _re.sub(r"[^A-Za-z0-9._-]", "_", key)[-50:] if key else "scratch"
    h = hashlib.sha1((key or "scratch").encode()).hexdigest()[:8]
    return d / f"{safe}.{h}.json"


def chat_fetch(key: str) -> str:
    """Return the saved conversation as a JSON array string for a document key, or '[]'."""
    p = _chat_file(key)
    try:
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            msgs = obj.get("messages", obj) if isinstance(obj, dict) else obj
            return json.dumps(msgs)
    except Exception:
        pass
    return "[]"


def chat_store(key: str, messages: list) -> str:
    """Persist a conversation under .tina4/chat/ keyed by the document path."""
    try:
        _chat_file(key).write_text(json.dumps({"key": key, "messages": messages}, indent=1), encoding="utf-8")
    except Exception as exc:
        return f"error: {exc}"
    return "ok"


def harness_edit(kind: str, b64: str) -> str:
    """Entry point the extension calls. Returns a single ``__HARNESS__{json}`` line to stdout."""
    try:
        data = json.loads(base64.b64decode(b64).decode("utf-8"))
        if kind == "create":
            text = _create(data["prompt"])
        elif kind == "check":
            text = _check(data["code"], data.get("notebook", ""))
        elif kind == "fix":
            text = _fix(data["code"], data["instruction"])
        elif kind == "chat":
            text = _chat(
                data["messages"], data.get("context", ""), data.get("notebook", ""),
                data.get("output", ""), data.get("intent", "code"), data.get("open_files", ""),
                data.get("preview_urls", []), data.get("think", "full"),
            )
        elif kind == "vision":
            text = _vision(data["image"], data.get("question", ""))
        elif kind == "chat_load":
            text = chat_fetch(data["key"])
        elif kind == "chat_save":
            text = chat_store(data["key"], data.get("messages", []))
        else:
            raise ValueError(f"unknown action '{kind}'")
        return "__HARNESS__" + json.dumps({"text": text})
    except Exception as exc:  # report back to the UI instead of raising
        return "__HARNESS__" + json.dumps({"error": str(exc)})


def warm_up() -> str:
    """Preload the model so the first right-click is fast. Safe to call repeatedly."""
    model = get_model()
    return f"Harness ready — right-click a cell for Check / Create / Fix. Model: {model.repo_id}"
