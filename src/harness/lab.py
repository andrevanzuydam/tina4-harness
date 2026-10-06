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
    return _code(_generate(messages, 800))


def _check(code: str) -> str:
    messages = [
        {"role": "system", "content":
            "You are a concise code reviewer. List issues and concrete suggestions as short "
            "bullet points. Plain text only, no code fences."},
        {"role": "user", "content": f"Review this code:\n```\n{code}\n```"},
    ]
    return _strip_think(_generate(messages, 500))


def _fix(code: str, instruction: str) -> str:
    # Reuse the notebook's own edit_code() if the reader defined one; otherwise do it here.
    try:
        from IPython import get_ipython

        ip = get_ipython()
        edit_code = ip.user_ns.get("edit_code") if ip is not None else None
        if callable(edit_code):
            return edit_code(code, instruction)
    except Exception:
        pass
    messages = [
        {"role": "system", "content":
            "You edit code. Apply the instruction and return ONLY the edited code in one ``` "
            "block. No explanation."},
        {"role": "user", "content": f"Instruction: {instruction}\n\nCode:\n```\n{code}\n```"},
    ]
    return _code(_generate(messages, 800))


_CHAT_SYSTEM = (
    "You are a PYTHON coding assistant in a live JupyterLab notebook. ASSUME every question is about "
    "Python and Python coding unless the user clearly says otherwise — e.g. 'antigravity' means the "
    "Python module (`import antigravity`), not anything else. Code you write is inserted and run as a "
    "NEW NOTEBOOK CELL automatically.\n\n"
    "RULES:\n"
    "- Reply with AT MOST ONE ```python code block plus one short sentence of prose. ALWAYS write the "
    "real, complete, runnable code — NEVER a placeholder like '[your code here]' or '[complete solution]'.\n"
    "- To add a TOOL, define a function decorated with @tool whose one-line docstring ends with "
    "'args: <names>' and returns a string; do NOT call run_agent yourself. Example:\n"
    "```python\n@tool\ndef shout(text: str) -> str:\n    \"Uppercase some text. args: text\"\n"
    "    return text.upper()\n```\n"
    "- For a 'what is / how do I' question, answer briefly in Python terms, using the documentation "
    "context below if given."
)


def _tidy_chat(reply: str, as_tool: bool) -> str:
    """Keep a chat reply notebook-friendly: at most one clean ```python block, one line of prose.
    Drops REPL (>>> / ...) example lines, and adds @tool when the user asked for a tool."""
    blocks = [m.group(1).strip() for m in _FENCE.finditer(reply)]
    prose = _FENCE.sub("", reply).strip()
    if not blocks:
        return reply
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


def _chat(messages: list, context: str = "", notebook_code: str = "") -> str:
    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    # Qwen's template allows only ONE system message, so fold everything into it.
    system_text = _CHAT_SYSTEM
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
    if _looks_like_question(last_user):  # ground factual Python questions in the stdlib docs
        try:
            from . import pyhelp

            docs = "\n\n".join(t[:600] for t, _s in pyhelp.search(last_user, k=2))
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
    reply = _strip_think(_generate([{"role": "system", "content": system_text}] + recent, 900))
    tidied = _tidy_chat(reply, as_tool="tool" in last_user.lower())
    if notebook_code:  # deterministically drop any re-definition of names already in the notebook
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
            text = _chat(data["messages"], data.get("context", ""), data.get("notebook", ""))
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
