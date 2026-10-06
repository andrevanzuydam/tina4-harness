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


def _threads() -> int:
    try:
        return max(1, int(os.environ.get("HARNESS_THREADS", "4")))
    except ValueError:
        return 4


def get_model():
    """The notebook's `model` if present, else one we load now (and share back to the notebook)."""
    global _model
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            existing = ip.user_ns.get("model")
            if existing is not None:
                _model = existing
                return _model
    except Exception:
        ip = None

    if _model is None:
        from harness import load_model

        _model = load_model()
        try:
            if ip is not None and "model" not in ip.user_ns:
                ip.user_ns["model"] = _model  # let notebook cells use the same instance
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
    "You are an assistant driving a live JupyterLab notebook that runs a local agent 'harness'. "
    "Code you write is inserted and run as a NEW NOTEBOOK CELL automatically.\n\n"
    "RULES:\n"
    "- Reply with AT MOST ONE ```python code block, and one short sentence of prose. Never repeat "
    "the code in words. For pure ideation, use no code block.\n"
    "- To add a TOOL, ONLY define a function decorated with @tool whose one-line docstring ends "
    "with 'args: <names>' and which returns a string. Do NOT call run_agent yourself — the user "
    "does that. Example:\n"
    "```python\n@tool\ndef shout(text: str) -> str:\n    \"Uppercase some text. args: text\"\n"
    "    return text.upper()\n```\n"
    "- run_agent(task) takes ONE natural-language task string, e.g. run_agent(\"shout 'hi'\")."
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


def _chat(messages: list, context: str = "") -> str:
    # Qwen's template allows only ONE system message, so fold the selection context into it.
    system_text = _CHAT_SYSTEM
    if context:
        system_text += (
            "\n\nThe user has SELECTED this code in the notebook; when they say 'this', 'it', or "
            "'the selection', they mean it:\n```python\n" + context + "\n```"
        )
    reply = _strip_think(_generate([{"role": "system", "content": system_text}] + list(messages), 700))
    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    return _tidy_chat(reply, as_tool="tool" in last_user.lower())


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
            text = _chat(data["messages"], data.get("context", ""))
        else:
            raise ValueError(f"unknown action '{kind}'")
        return "__HARNESS__" + json.dumps({"text": text})
    except Exception as exc:  # report back to the UI instead of raising
        return "__HARNESS__" + json.dumps({"error": str(exc)})


def warm_up() -> str:
    """Preload the model so the first right-click is fast. Safe to call repeatedly."""
    model = get_model()
    return f"Harness ready — right-click a cell for Check / Create / Fix. Model: {model.repo_id}"
