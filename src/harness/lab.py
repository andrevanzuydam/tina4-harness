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

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
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


def _locked_generate(model, messages, max_tokens, on_text=None):
    with _lock:  # one generation at a time on the single local model
        return model.generate(messages, max_tokens=max_tokens, on_text=on_text)


def _generate(messages, max_tokens=700, stream: bool = False) -> str:
    model = get_model()
    on_text = _stream if stream else None
    return _executor().submit(_locked_generate, model, messages, max_tokens, on_text).result()


def _strip_think(text: str) -> str:
    # This distill reasons first and closes with </think> (the opening tag comes from its
    # chat template, so it is not in the generated text). Keep only what follows the reasoning.
    idx = text.rfind("</think>")
    if idx != -1:
        text = text[idx + len("</think>"):]
    return _THINK.sub("", text).replace("<think>", "").strip()


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


_CHAT_SYSTEM = (
    "You are a PYTHON coding assistant in a live JupyterLab notebook. ASSUME every question is about "
    "Python and Python coding unless the user clearly says otherwise — e.g. 'antigravity' means the "
    "Python module (`import antigravity`), not anything else. Code you write is inserted and run as a "
    "NEW NOTEBOOK CELL automatically.\n\n"
    "RULES:\n"
    "- Reply with AT MOST ONE ```python code block plus one short sentence of prose. ALWAYS write the "
    "real, complete, runnable code — NEVER a placeholder like '[your code here]' or '[complete solution]'.\n"
    "- For a build/create/write-code request, go STRAIGHT to the code block. Do NOT write a plan, an "
    "outline, or a bulleted list of what the code will contain.\n"
    "- To add a TOOL, define a function decorated with @tool whose one-line docstring ends with "
    "'args: <names>' and returns a string; do NOT call run_agent yourself. Example:\n"
    "```python\n@tool\ndef shout(text: str) -> str:\n    \"Uppercase some text. args: text\"\n"
    "    return text.upper()\n```\n"
    "- For a 'what is / how do I' question, answer briefly in Python terms, using the documentation "
    "context below if given.\n"
    "- If code or an error is provided (a selection, the notebook, or the cell output), USE it "
    "directly. NEVER ask the user to paste or share the code again."
)


def _cap_prose(text: str, limit: int = 700) -> str:
    """Trim a long prose answer at a sentence/line boundary so the chat doesn't overflow."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = max(cut.rfind(". "), cut.rfind("\n"))
    return (cut[: dot + 1] if dot > limit // 2 else cut).rstrip() + " …"


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


def _strip_script_scaffold(code: str) -> str:
    """Make generated code safe to drop in a cell: remove the `if __name__ == "__main__":` block
    (and any trailing bare `main()` call) so running the cell can't kick off an interactive script
    that blocks on input(). Function/class definitions and example CALLS are kept."""
    lines = code.splitlines()
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if re.match(r"\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", ln):
            break  # drop this block and everything after it
        out.append(ln)
        i += 1
    # Drop a trailing bare `main()` / `main( )` call left at top level (it would run on cell exec).
    while out and re.match(r"\s*main\s*\(\s*\)\s*$", out[-1]):
        out.pop()
    return "\n".join(out).strip() or code.strip()


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

# File extensions a "load this file" request may name (kept in sync with repo.CODE_EXT spirit).
_FILE_EXT = (r"py|js|ts|tsx|jsx|java|go|rs|rb|php|c|cc|cpp|h|hpp|cs|swift|kt|scala|"
             r"md|rst|txt|json|ya?ml|toml|cfg|ini|sh|sql|html?|css")
_FILE_RE = re.compile(r"([\w./-]+\.(?:" + _FILE_EXT + r"))\b", re.I)
_LOAD_VERB = r"load|open|read|show|view|cat|insert|paste|pull up|bring up|import"
# Reply prefix that tells the extension to insert the code block directly into a new cell.
_INSERT_SENTINEL = "@@HARNESS_INSERT@@\n"


def _is_load_request(text: str) -> bool:
    """A request to LOAD a file into a cell (not a question about code)."""
    t = text.strip()
    # Must not be a question like "how do I open a file?"
    if t.endswith("?") and re.search(r"\bhow\b|\bwhat\b|\bwhy\b", t, re.I):
        return False
    if _FILE_RE.search(t) and re.search(r"\b(" + _LOAD_VERB + r")\b", t, re.I):
        return True
    if re.search(r"\b(" + _LOAD_VERB + r")\b.*\b(the\s+)?(code|file|source|script)\b", t, re.I):
        return True
    return bool(re.search(r"\bload\s+(it|this|that|the code|the file)\b", t, re.I))


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


_RUN_SENTINEL = "@@HARNESS_RUN@@"  # extension: insert into a new cell AND run it (if safe)


def _functions_in(code: str) -> list[tuple[str, list[str]]]:
    out = []
    for m in re.finditer(r"^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)[ \t]*\(([^)]*)\)", code or "", re.M):
        params = [p.strip() for p in m.group(2).split(",") if p.strip() and p.strip() != "self"]
        out.append((m.group(1), params))
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


_WROTE_SENTINEL = "@@HARNESS_WROTE@@"  # tells the extension: files written, show + refresh preview
_APP_SENTINEL = "@@HARNESS_APP@@"      # tells the extension: open the App Preview tab at <url>


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


def _new_tina4_app(text: str) -> str:
    """Provision a new Tina4 Python app in the notebook area, start it, and open the preview."""
    from pathlib import Path

    from . import tina4

    name = tina4._slug(_app_name(text))
    dest = Path.cwd() / name
    if not dest.exists() or not tina4.is_project(dest):
        _status(f"⬇️ provisioning tina4 app '{name}' (uv sync)…")
        tina4.init(dest)
        _gitignore_path(dest)  # keep the harness repo clean — provisioned apps are local artifacts
    port = tina4.free_port(7146)
    _status(f"🚀 starting tina4 serve on :{port}…")
    url = tina4.serve(dest, port=port)
    note = (f"Provisioned and started **{name}** at {url} — opening the preview. "
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
    note = (f"Wrote {files} in **{project.name}** — route `{url}`. "
            "tina4's hot-reload picks it up and the App Preview auto-refreshes.")
    return f"{_WROTE_SENTINEL}|{project}|{url}\n{note}\n\n```python\n{route.strip()}\n```"


def _chat(messages: list, context: str = "", notebook_code: str = "", output: str = "",
          intent: str = "code", open_files: str = "", preview_urls: list | None = None) -> str:
    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    tina4_project = _detect_tina4(open_files, last_user, preview_urls or [])

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

    # If a repo is open, answer code/exploration questions FROM the repo (not the active cell).
    try:
        from . import repo as _repo

        if re.search(
            r"\b(repo|repositor|codebase|folder|directory|director|structure|organi[sz]|"
            r"what (does|is)|how (does|is)|summar|overview|explain|where|which file|list)\b",
            last_user, re.I,
        ) and _repo.ensure_indexed():  # re-index the on-disk clone if this kernel has none
            _status("🔎 searching the repo…")
            # history → keeps follow-ups in context; open_files → aware of what's open on screen.
            return _repo.ask_repo(last_user, max_tokens=600, history=messages, open_code=open_files)
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
    # Working inside a Tina4 project? Ground the model in Tina4 paradigms automatically — no toggle.
    if tina4_project is not None:
        try:
            from . import tina4

            _status(f"🟢 Tina4 project: {tina4_project.name}")
            system_text += "\n\n" + tina4.PARADIGMS
            docs = tina4.ask_tina4(last_user, cwd=tina4_project)
            if docs:
                system_text += "\n\nRelevant Tina4 documentation:\n" + docs[:1500]
        except Exception:
            pass
    if intent == "ask":
        system_text += (
            "\n\nThis is a QUESTION, not a build request. EXPLAIN clearly in a few sentences; show at "
            "most a tiny corrected snippet inline. Do NOT output a full program or a cell to run."
        )
    # "call / invoke / run the function" → just CALL an already-defined function, don't define a new one.
    if intent != "ask" and re.search(r"\b(call|invoke|run)\b", last_user, re.I):
        system_text += (
            "\n\nThe user wants to CALL a function that is ALREADY defined (in the notebook or an open "
            "file) — do NOT define a new function, and do NOT invent one named 'call'. Write just ONE "
            "or two lines that invoke the EXISTING function by its real name with a sensible example "
            "argument and print the result. For example, if the defined function were "
            "`def area(w, h): ...`, you would write `print(area(3, 4))`. Use the function's REAL name "
            "and arguments that match its parameters."
        )
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
    if output:
        system_text += (
            "\n\nThe active cell's latest OUTPUT / ERROR is below. If it is an error, use it to find "
            "the cause and FIX it. The code is already provided in this message — do NOT ask the user "
            "to share the code again; just return the corrected code:\n" + output[-1800:]
        )
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
    if notebook_code:
        system_text += (
            "\n\nThe notebook ALREADY contains this code — reuse these names and do NOT rewrite or "
            "redefine them. Write ONLY the new code the user asks for (e.g. just the call, not the "
            "function again):\n```python\n" + notebook_code[-4000:] + "\n```"
        )
    if context:
        system_text += (
            "\n\nThe user has SELECTED this code in the notebook; when they say 'this', 'it', or "
            "'the selection', they mean it:\n```python\n" + context + "\n```"
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
    # Context budget: default ~128K tokens (≈ 512K chars). The model supports up to 256K; a 2B is
    # sharpest with less, so keep only as much recent history as fits. Tune with HARNESS_CONTEXT_CHARS.
    budget = int(os.environ.get("HARNESS_CONTEXT_CHARS", str(512_000)))
    recent: list = []
    used = 0
    for m in reversed(messages):
        c = len(m.get("content", ""))
        if recent and used + c > budget:
            break
        recent.append(m)
        used += c
    recent.reverse()
    _status("💭 thinking…")
    reply = _strip_think(_generate([{"role": "system", "content": system_text}] + recent, 4096, stream=True))
    if intent == "code":
        _status("🔎 checking syntax…")
    if intent == "ask":
        return _cap_prose(reply, 1000)  # an explanation to read in the chat — nothing inserted
    tidied = _tidy_chat(reply, as_tool="tool" in last_user.lower())
    # If the model stubbed or wrote invalid Python, retry once (ask the model — no canned reply).
    m = _FENCE.search(tidied)
    if m:
        bad_placeholder = _is_placeholder(m.group(1))
        ok, err = _valid_python(m.group(1))
        if bad_placeholder or not ok:
            _status("↻ fixing the code…")
            why = "it was a placeholder" if bad_placeholder else f"it had a syntax error: {err}"
            retry = recent + [
                {"role": "assistant", "content": tidied},
                {"role": "user", "content": f"That is not acceptable ({why}). {_py_hint(m.group(1))}. "
                                            "Return ONLY the actual, complete, valid Python code."},
            ]
            reply2 = _strip_think(_generate([{"role": "system", "content": system_text}] + retry, 4096))
            tidied2 = _tidy_chat(reply2, as_tool="tool" in last_user.lower())
            m2 = _FENCE.search(tidied2)
            if m2 and not _is_placeholder(m2.group(1)) and _valid_python(m2.group(1))[0]:
                tidied = tidied2
    # A fix/edit request WANTS to redefine (the UI replaces the existing cell). A plain "add/use"
    # request should NOT redefine — strip any duplicate definition of names already in the notebook.
    is_revision = bool(re.search(r"\b(fix|edit|updat|chang|refactor|improv|rewrit|modif)\w*", last_user, re.I))
    if notebook_code and not is_revision:
        tidied = _strip_redefs_in_reply(tidied, _defined_names(notebook_code))
    return tidied


def _looks_like_question(text: str) -> bool:
    t = text.lower().strip()
    return ("?" in t) or t.startswith(
        ("what", "how", "why", "which", "when", "where", "explain", "is ", "does ", "can ", "tell me")
    )


def _vision(image: str, question: str) -> str:
    """Answer a question about an image — using the SAME model as text (Qwen3.8-2B-Distill is a VLM)."""
    from pathlib import Path

    path = Path(image)
    if not path.is_absolute():
        path = Path.cwd() / image      # the kernel's cwd is the notebook folder
    if not path.is_file():
        return f"Could not find the image at {path}."

    model = get_model()
    prompt = question or "Describe this image."
    _status("🖼 looking at the image…")
    with _lock:
        if hasattr(model, "describe"):          # the MLX model does text AND vision
            return model.describe(str(path), prompt, max_tokens=512)
        from .vision import VisionModel          # torch fallback
        global _vision_fallback
        if _vision_fallback is None:
            _vision_fallback = VisionModel()
        return _vision_fallback.describe(str(path), prompt, max_tokens=512)


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
                data.get("preview_urls", []),
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
