"""A ready-to-use harness — import it, call ``setup()``, add a ``@tool``, run the agent.

This is the *packaged* version of the mechanism the teaching notebook builds by hand
(``notebooks/harness.ipynb``). It exists so a clean demo notebook can hide the busy setup and
leave you a single place to add a tool:

    from harness.agent import *
    setup()                                   # loads the local model, registers the default tools

    @tool
    def reverse(text: str) -> str:
        "Reverse a string. args: text"
        return text[::-1]

    run_agent("Use reverse on 'hello' and give the final answer.")

Everything runs on the local model (``harness.load_model``), fully offline.
"""
from __future__ import annotations

import json
import re
import subprocess
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path

from .download import PROJECT_ROOT
from .model import load_model
from .pyhelp import ask_python  # grounded Python Q&A, available after `from harness.agent import *`

WORKSPACE = PROJECT_ROOT / "workspace"
TOOLS: dict = {}
_model = None


def tool(fn):
    """Register a function as a tool (its docstring is shown to the model)."""
    TOOLS[fn.__name__] = fn
    return fn


def _safe(path: str) -> Path:
    target = (WORKSPACE / path).resolve()
    if WORKSPACE.resolve() not in target.parents and target != WORKSPACE.resolve():
        raise ValueError(f"path '{path}' is outside the workspace")
    return target


# --- default tools -------------------------------------------------------------------
@tool
def write_file(path: str, content: str) -> str:
    "Write text to a file (creates folders). args: path, content"
    t = _safe(path)
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"


@tool
def read_file(path: str) -> str:
    "Read a file and return its text. args: path"
    t = _safe(path)
    return t.read_text(encoding="utf-8") if t.is_file() else f"ERROR: {path} is not a file"


@tool
def list_dir(path: str = ".") -> str:
    "List files in a folder. args: path"
    t = _safe(path)
    if not t.exists():
        return f"ERROR: {path} does not exist"
    return "\n".join(sorted(p.name + ("/" if p.is_dir() else "") for p in t.iterdir())) or "(empty)"


@tool
def run_shell(command: str) -> str:
    "Run a shell command in the workspace (60s). args: command"
    try:
        p = subprocess.run(command, shell=True, cwd=WORKSPACE, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "ERROR: timed out"
    return f"exit {p.returncode}\n{(p.stdout + p.stderr).strip()}"


def _http_get(url: str, timeout: int = 12) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "tina4-harness/0.1 (demo)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


@tool
def search_web(query: str) -> str:
    "Search the internet (Wikipedia) for a short fact. args: query"
    url = "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=1&srsearch=" + urllib.parse.quote(query)
    hits = json.loads(_http_get(url))["query"]["search"]
    if not hits:
        return "no results"
    top = hits[0]
    return f"{top['title']}: {re.sub('<.*?>', '', top['snippet'])}"


@tool
def star_wars(resource: str) -> str:
    "Look up Star Wars data from SWAPI, e.g. 'people/1'. args: resource"
    data = json.loads(_http_get(f"https://swapi.info/api/{resource.strip('/')}"))
    if isinstance(data, dict):
        return json.dumps({k: data[k] for k in list(data)[:6]})
    return json.dumps(data[:3])


@tool
def python_help(query: str) -> str:
    "Look up Python standard-library documentation (grounded via Groundwire). args: query"
    from . import pyhelp
    hits = pyhelp.search(query, k=3)
    return "\n\n".join(text[:700] for text, _s in hits) or "no matching docs"


@tool
def final_answer(answer: str) -> str:
    "Give the final answer and stop. args: answer"
    return answer


# --- prompts -------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a small, careful agent in a harness. You work in single steps.

On each turn: think in AT MOST two short sentences, then output exactly ONE fenced JSON action
and stop. Use the real tool result to choose the next step. When done, call final_answer.

<think>I need the file's contents, so I will read it.</think>
```json
{"tool": "read_file", "args": {"path": "notes.txt"}}
```
"""


def tools_prompt() -> str:
    catalogue = "\n".join(f"- {n}: {(fn.__doc__ or '').strip()}" for n, fn in TOOLS.items())
    return (
        "You can use these tools, one at a time:\n\n" + catalogue + "\n\n"
        "End your reply with ONE fenced JSON block:\n\n"
        '```json\n{"tool": "<name>", "args": { ... }}\n```\n\n'
        "One tool per reply. When done, call final_answer."
    )


def build_system() -> str:
    return SYSTEM_PROMPT.rstrip() + "\n\n" + tools_prompt()


# --- parser --------------------------------------------------------------------------
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def split_thinking(text: str):
    thoughts = "\n".join(m.group(1).strip() for m in _THINK.finditer(text))
    rest = _THINK.sub("", text)
    if "</think>" in rest:
        head, _, tail = rest.rpartition("</think>")
        if not thoughts:
            thoughts = head.strip()
        rest = tail
    elif not thoughts and "<think>" in rest:
        head, _, tail = rest.partition("<think>")
        thoughts, rest = tail.strip(), head
    return thoughts.strip(), rest.replace("<think>", "").strip()


def _iter_json(text: str):
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


def parse_action(reply: str):
    thinking, rest = split_thinking(reply)
    for chunk in list(_iter_json(rest)) + list(_iter_json(reply)):
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            args = obj.get("args", {})
            if isinstance(args, str) and obj["tool"] == "final_answer":
                args = {"answer": args}
            if isinstance(args, dict):
                return thinking, obj["tool"], args
    return thinking, None, {}


# --- loop ----------------------------------------------------------------------------
def run_agent(task: str, max_steps: int = 6, verbose: bool = True):
    """Run the agent loop on the local model until final_answer (or the step limit)."""
    if _model is None:
        raise RuntimeError("Call setup() first.")
    messages = [{"role": "system", "content": build_system()}, {"role": "user", "content": task}]
    last = None
    last_result = None  # once a tool gives us something, we can finish with it
    for step in range(1, max_steps + 1):
        reply = _model.generate(messages)
        thinking, name, args = parse_action(reply)
        messages.append({"role": "assistant", "content": json.dumps({"tool": name, "args": args}) if name else "(no action)"})
        if verbose:
            print(f"\n── STEP {step} " + "─" * 46)
            if thinking:
                print("🧠", textwrap.shorten(thinking, 200))

        if name == "final_answer":
            answer = args.get("answer", last_result or "")
            if verbose:
                print("✅", answer)
            return answer

        # No action, or a repeat of the last call: if we already have a result, finish with it.
        sig = (name, json.dumps(args, sort_keys=True)) if name else None
        if name is None or sig == last:
            if last_result is not None:
                if verbose:
                    print("✅ (final answer from the last result)", last_result)
                return last_result
            messages.append({"role": "user", "content":
                             'Reply with ONE fenced json action, e.g. {"tool":"final_answer","args":{"answer":"..."}}.'})
            continue
        last = sig

        try:
            result = TOOLS[name](**args) if name in TOOLS else f"ERROR: no tool '{name}'"
        except Exception as exc:
            result = f"ERROR: {name} failed: {exc}"
        last_result = result
        if verbose:
            print(f"🔧 {name}({args})")
            print("📤", textwrap.shorten(str(result), 200))
        # Strongly steer toward finishing now that there is a result.
        messages.append({"role": "user", "content":
                         f"Result of {name}: {result}\nIf this answers the task, reply ONLY with "
                         f'{{"tool":"final_answer","args":{{"answer":"{result}"}}}}.'})
    return last_result


def model():
    return _model


def setup(verbose: bool = True):
    """Load the local model and get the harness ready. Returns the model."""
    global _model
    WORKSPACE.mkdir(exist_ok=True)
    _model = load_model()
    if verbose:
        print(f"Harness ready · model {_model.repo_id} · tools: {', '.join(TOOLS)}")
        print("Add a tool with @tool, then call run_agent('…'). Right-click a cell for Check/Create/Fix; chat is in the right sidebar.")
    return _model
