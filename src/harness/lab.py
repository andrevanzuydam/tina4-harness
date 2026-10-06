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


def _locked_generate(model, messages, max_tokens):
    with _lock:  # one generation at a time on the single local model
        return model.generate(messages, max_tokens=max_tokens)


def _generate(messages, max_tokens=700) -> str:
    model = get_model()
    return _executor().submit(_locked_generate, model, messages, max_tokens).result()


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


def _check(code: str) -> str:
    messages = [
        {"role": "system", "content":
            "You are a concise code reviewer. List at most 5 DISTINCT issues as short bullet points. "
            "Do NOT repeat the same point. Plain text only, no code fences."},
        {"role": "user", "content": f"Review this code:\n```\n{code}\n```"},
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


def _chat(messages: list, context: str = "", notebook_code: str = "", output: str = "", intent: str = "code") -> str:
    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    # Known non-Python syntax (??, &&, ||, ===) in the code being fixed → fix it deterministically,
    # no model guessing. This is always correct, which matters for beginners.
    if intent == "code" and context.strip():
        det = _deterministic_fix(context)
        if det != context and _valid_python(det)[0] and not _valid_python(context)[0]:
            _status("🔧 rewriting non-Python syntax…")
            return "Rewrote the non-Python syntax to valid Python:\n\n```python\n" + det.strip() + "\n```"
    # Qwen's template allows only ONE system message, so fold everything into it.
    system_text = _CHAT_SYSTEM
    if intent == "ask":
        system_text += (
            "\n\nThis is a QUESTION, not a build request. EXPLAIN clearly in a few sentences; show at "
            "most a tiny corrected snippet inline. Do NOT output a full program or a cell to run."
        )
    if output:
        system_text += (
            "\n\nThe active cell's latest OUTPUT / ERROR is below. If it is an error, use it to find "
            "the cause and FIX it. The code is already provided in this message — do NOT ask the user "
            "to share the code again; just return the corrected code:\n" + output[-1800:]
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
    reply = _strip_think(_generate([{"role": "system", "content": system_text}] + recent, 4096))
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
            text = _check(data["code"])
        elif kind == "fix":
            text = _fix(data["code"], data["instruction"])
        elif kind == "chat":
            text = _chat(
                data["messages"], data.get("context", ""), data.get("notebook", ""),
                data.get("output", ""), data.get("intent", "code"),
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
