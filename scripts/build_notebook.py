"""Build notebooks/harness.ipynb — the teaching notebook, with the whole harness in cells."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

nb = new_notebook()
cells = []
def md(s): cells.append(new_markdown_cell(s.strip("\n")))
def code(s): cells.append(new_code_cell(s.strip("\n")))

# ------------------------------------------------------------------ intro
md(r"""
# tina4-harness — how an agent harness works

A **large language model** does only one thing: given some text, it predicts the next words.
It cannot open a file, run a command, or look something up. So how does a model become an
"agent" that *does* things?

The answer is the **harness** — a small, ordinary program that wraps the model and runs this
loop:

```
            ┌──────────────────────────────────────────────┐
            │                                                │
   task ──▶ build transcript ──▶ ask the model ──▶ read the reply
            │                                        │       │
            │                                 is it a tool?   │
            │                                        │ yes    │
            └──────────── put result back ◀── run the tool ◀──┘
                                                     │ "final_answer"
                                                     ▼
                                                  answer
```

That loop is the whole idea. This notebook builds it **from scratch, one cell at a time**,
driven by a small local reasoning model — `DeepSeek-R1-Distill-Qwen-1.5B` (a *distill*: a tiny
model taught by a big one). It runs fully offline on your own machine.

> **No miracles.** A 1.5B model is not going to behave like a frontier agent. That is the
> point: with the model this weak, you can *see* every job the harness does to keep it on the
> rails — the format contract, the parser, the step limit, the repeat-guard. Watch the
> **mechanism**, not the IQ.

**The layout mirrors the parts of a harness:**
1. load the model · 2. watch its memory · 3. the **tools** (one per cell) · 4. the **tools
prompt** · 5. the **system prompt** · 6. the **parser** · 7. the **loop** · 8. run it ·
9. add your own tools (**internet search** + a real **API**) · 10. **RAG** (unlimited context,
off the GPU) · 11. a **highlight-and-prompt** cell editor.

Everything that matters is in the cells below — read them, change them, re-run them.
""")

# ------------------------------------------------------------------ setup
md(r"""
## 0. Setup

The only thing we hide is the boring, platform-specific job of *loading a model* — because
that is not the lesson. It lives in the small `harness` helper (`src/harness/`), which picks
**MLX** on Apple Silicon and **PyTorch/transformers** on Windows/Linux, and downloads the
weights into this project's `models/` folder on first use. Everything else — tools, prompts,
parser, loop — we write here, in the open.
""")
code(r"""
import sys, json, re, subprocess, time, textwrap, urllib.request, urllib.parse
from pathlib import Path

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from harness import load_model, format_memory, Settings   # the ONLY hidden part: model loading
print("project root:", ROOT)
""")

# ------------------------------------------------------------------ 1 load
md(r"""
## 1. Load the model

`load_model()` returns an object with what a harness needs: `generate(messages)`, `memory()`,
and knobs (`settings.temperature`, `set_cache_limit_gb`). The first run downloads ~1 GB into
`models/`; later runs are instant and offline.
""")
code(r"""
model = load_model()                       # defaults: temperature=0.2, max_tokens=1024
print("loaded:", format_memory(model.memory()))
print("repo:", model.repo_id, "| backend:", model.backend, "| device:", model.device)
""")

# ------------------------------------------------------------------ 2 telemetry
md(r"""
## 2. Watch the memory — and change the knobs

A harness manages the model's memory. Two different "caches" matter, and people mix them up:

* **Buffer cache** — freed GPU buffers the engine keeps to reuse. Bigger = faster but holds more
  memory. On MLX you can cap it live (`set_cache_limit_gb`); `clear_cache()` frees it. (On the
  PyTorch backend this is informational.)
* **KV cache** — the attention keys/values that grow with the **length of the conversation**.
  Not a setting — it grows as the transcript grows. That is *why* long context is expensive, and
  why RAG (section 10) exists.

Sliders change **temperature** (how random the model is; low = reliable tool use) and the **cache
limit**. Plain functions are there too if `ipywidgets` is missing.
""")
code(r"""
def show_memory():
    print(format_memory(model.memory()))
    print(f"temperature={model.settings.temperature}  max_tokens={model.settings.max_tokens}")

def set_temperature(x):
    model.settings.temperature = float(x)

def set_cache_gb(x):
    model.set_cache_limit_gb(None if x in (None, 0) else float(x))   # None = engine default
    model.clear_cache()

try:
    import ipywidgets as W
    from IPython.display import display

    temp = W.FloatSlider(value=model.settings.temperature, min=0.0, max=1.5, step=0.05, description="temp")
    cache = W.FloatSlider(value=0.0, min=0.0, max=4.0, step=0.25, description="cache GB")
    out = W.Output()

    def refresh(_=None):
        out.clear_output()
        with out:
            show_memory()

    temp.observe(lambda c: set_temperature(c["new"]), names="value")
    cache.observe(lambda c: (set_cache_gb(c["new"]), refresh()), names="value")
    btn = W.Button(description="refresh / clear cache")
    btn.on_click(lambda _: (model.clear_cache(), refresh()))
    display(W.VBox([W.HBox([temp, cache, btn]), out]))
    refresh()
except Exception as e:
    print("(no ipywidgets — call the functions directly)", e)
    show_memory()
""")

# ------------------------------------------------------------------ 3 tools
md(r"""
## 3. The tools — the hands of the agent

A tool is just a Python function plus a one-line docstring. We keep a tiny registry so we can
(a) describe the tools to the model and (b) look one up by name and run it. Everything runs
inside a **workspace** folder, so the agent can't touch the rest of your disk.

Each tool gets **its own cell** so you can read them one at a time.
""")
code(r"""
WORKSPACE = ROOT / "workspace"
WORKSPACE.mkdir(exist_ok=True)

TOOLS = {}                       # name -> function
def tool(fn):                    # a 3-line decorator: register and return unchanged
    TOOLS[fn.__name__] = fn
    return fn

def _safe(path: str) -> Path:
    "Keep every path inside the workspace - a tool must not escape its sandbox."
    target = (WORKSPACE / path).resolve()
    if WORKSPACE.resolve() not in target.parents and target != WORKSPACE.resolve():
        raise ValueError(f"path '{path}' is outside the workspace")
    return target

print("workspace:", WORKSPACE)
""")
code(r"""
@tool
def write_file(path: str, content: str) -> str:
    "Write text to a file (creates folders as needed). args: path, content"
    target = _safe(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {path}"
""")
code(r"""
@tool
def read_file(path: str) -> str:
    "Read a file and return its text. args: path"
    target = _safe(path)
    if not target.is_file():
        return f"ERROR: {path} is not a file"
    return target.read_text(encoding="utf-8")
""")
code(r"""
@tool
def list_dir(path: str = ".") -> str:
    "List the files in a folder (use '.' for the top). args: path"
    target = _safe(path)
    if not target.exists():
        return f"ERROR: {path} does not exist"
    return "\n".join(sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())) or "(empty)"
""")
code(r"""
@tool
def run_shell(command: str) -> str:
    "Run a shell command inside the workspace and return its output (60s limit). args: command"
    # shell=True uses the platform shell: /bin/sh on macOS/Linux, cmd.exe on Windows.
    try:
        proc = subprocess.run(command, shell=True, cwd=WORKSPACE,
                              capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 60s"
    return f"exit {proc.returncode}\n{(proc.stdout + proc.stderr).strip()}"
""")
code(r"""
@tool
def final_answer(answer: str) -> str:
    "Give the final answer and stop. Call this when the task is done. args: answer"
    return answer

print("tools:", list(TOOLS))
""")

# ------------------------------------------------------------------ 4 tools prompt
md(r"""
## 4. The tools prompt

The model was never trained on *our* tools, so we describe them and pin down an exact format for
asking. We build that text from the registry — docstrings become the catalogue. Add a tool in a
cell, call `tools_prompt(TOOLS)` again, and the model learns about it.
""")
code(r"""
def tools_prompt(tools) -> str:
    catalogue = "\n".join(f"- {name}: {(fn.__doc__ or '').strip()}" for name, fn in tools.items())
    return (
        "You can use these tools, one at a time:\n\n"
        f"{catalogue}\n\n"
        "To use a tool, end your reply with ONE fenced JSON block and nothing after it:\n\n"
        '```json\n{\"tool\": \"<tool name>\", \"args\": { ... }}\n```\n\n'
        "Use exactly one tool per reply. When the task is done, call final_answer."
    )

TOOLS_PROMPT = tools_prompt(TOOLS)
print(TOOLS_PROMPT)
""")

# ------------------------------------------------------------------ 5 system prompt
md(r"""
## 5. The system prompt

The tools prompt says *what* the model can do; the system prompt says *how to behave*. For a
small reasoning distill the two rules that matter most: **think briefly** (or it rambles forever
and never acts) and **one action per turn**. The worked example at the end is worth a paragraph
of instructions.
""")
code(r'''
SYSTEM_PROMPT = """You are a small, careful agent running inside a harness. You work in single steps.

On each turn:
1. Think in AT MOST two short sentences about the ONE next step.
2. Then output exactly ONE fenced JSON action and stop. Never output more than one action.
3. You will be shown the real tool result. Use it to choose the next step.
4. When the task is done, call the final_answer tool.

Do not imagine tool results - wait for the real one. Do not repeat a tool you already ran. Here
is the shape of one good turn:

<think>I need the file's contents, so I will read it.</think>
```json
{"tool": "read_file", "args": {"path": "notes.txt"}}
```
"""

def build_system():
    "The full system message each turn: behaviour + the tool catalogue."
    return SYSTEM_PROMPT.rstrip() + "\n\n" + TOOLS_PROMPT

print(build_system()[:380], "...")
''')

# ------------------------------------------------------------------ 6 parser
md(r"""
## 6. Reading the reply — the parser

The model only ever emits text. The parser turns that text into a decision. It copes with how a
small distill really behaves: it writes its reasoning in `<think> … </think>` first (we split
that off, and keep it — watching it think is the lesson); it often **plans several steps** at
once (we take the **first**); and `final_answer` sometimes arrives as `"args": "the answer"` (we
accept that too). We scan for balanced `{…}` by hand, because a regex can't — braces nest, and
`}` appears inside strings.
""")
code(r'''
THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

def split_thinking(text):
    "Return (thinking, rest). Handles an unclosed <think> that ran out of budget."
    thoughts = "\n".join(m.group(1).strip() for m in THINK.finditer(text))
    rest = THINK.sub("", text)
    if not thoughts and "<think>" in rest.lower() and "</think>" not in rest.lower():
        head, _, tail = rest.partition("<think>")
        thoughts, rest = tail.strip(), head
    return thoughts.strip(), rest.strip()

def iter_json_objects(text):
    "Yield every top-level {...} substring, honouring strings and nesting."
    depth = 0; start = None; in_str = False; esc = False
    for i, ch in enumerate(text):
        if in_str:                       # inside a "string": ignore braces until it closes
            if esc:          esc = False
            elif ch == "\\": esc = True
            elif ch == '"':  in_str = False
            continue
        if ch == '"': in_str = True
        elif ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0: yield text[start:i+1]

def parse_action(reply):
    "Turn a model reply into (thinking, tool, args). tool is None if no action was found."
    thinking, rest = split_thinking(reply)
    for chunk in list(iter_json_objects(rest)) + list(iter_json_objects(reply)):
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

parse_action('<think>read it</think>\n```json\n{"tool":"read_file","args":{"path":"a"}}\n```')
''')

# ------------------------------------------------------------------ 7 loop
md(r"""
## 7. The agent loop — the heart of the harness

Now put it together. Two small jobs here are what a real harness does and a naive script does
not:

* **It curates the transcript.** It feeds back a *compact record* of each step (the action it
  took), not the model's raw rambling — so the context stays clean and the model doesn't spiral.
* **It guards against loops.** If the model asks for the exact same tool call twice, the harness
  nudges it to move on. A step limit ends the run no matter what.

Read it top to bottom — this is the whole thing.
""")
code(r'''
def run_agent(task, max_steps=6, verbose=True):
    messages = [
        {"role": "system", "content": build_system()},
        {"role": "user", "content": task},
    ]
    last_call = None
    for step in range(1, max_steps + 1):
        reply = model.generate(messages)                     # 1. ask the model
        thinking, name, args = parse_action(reply)           # 2. read the reply
        # curate: record the chosen step, not the raw musing, so context stays clean
        messages.append({"role": "assistant",
                         "content": json.dumps({"tool": name, "args": args}) if name else "(no action)"})
        if verbose:
            print(f"\n── STEP {step} " + "─" * 48)
            if thinking: print("🧠", textwrap.shorten(thinking, 220))

        if name is None:                                     # no action — remind and retry
            if verbose: print("⚠️  no tool call; reminding the model of the format")
            messages.append({"role": "user",
                             "content": 'Reply with ONE fenced json: {"tool":...,"args":{...}}'})
            continue

        if name == "final_answer":                           # 3a. done
            if verbose: print("✅ final:", args.get("answer", ""))
            return args.get("answer", "")

        signature = (name, json.dumps(args, sort_keys=True))
        if signature == last_call:                           # repeat-guard
            if verbose: print(f"↻ repeat of {name}; nudging it to move on")
            messages.append({"role": "user",
                             "content": f"You already ran {name} with those args. Use the result, or call final_answer."})
            continue
        last_call = signature

        try:                                                 # 3b. act — a bad tool call must
            result = TOOLS[name](**args) if name in TOOLS else f"ERROR: no tool '{name}'"
        except Exception as exc:                             #     not crash the harness; report it
            result = f"ERROR: {name} failed: {exc}"
        if verbose:
            print(f"🔧 {name}({args})")
            print("📤", textwrap.shorten(str(result), 220))
        messages.append({"role": "user", "content": f"Result of {name}:\n{result}"})   # 4. feed back

    if verbose: print("\n(stopped: hit the step limit)")
    return None
''')

# ------------------------------------------------------------------ 8 run
md(r"""
## 8. Run it!

Watch the loop think, pick a tool, see the result, decide the next step. A 1.5B often stumbles —
repeats a step, forgets to finish — and you will *see* the harness catch it with the repeat-guard
and step limit. That visible scaffolding is the lesson. Re-run it a few times; lower the
temperature in §2 for steadier tool use.
""")
code(r'''
answer = run_agent(
    "Create a file greeting.txt containing exactly 'hello harness', "
    "then read it back to confirm, then give the final answer with its contents."
)
print("\nANSWER:", answer)
print("workspace now:", [p.name for p in WORKSPACE.iterdir()])
''')

# ------------------------------------------------------------------ 9 add tools
md(r"""
## 9. Add your own tools: the internet, and a real API

Adding a capability is just: write a function, give it a docstring, decorate it with `@tool`, and
rebuild the tools prompt. Let's add two that reach the outside world — **internet search** (via
Wikipedia) and a **real API** (SWAPI, the Star Wars API). (Swap in any open API you like — a snake
facts API, a weather API — the pattern is identical.)

These cells need an internet connection.
""")
code(r'''
def _http_get(url, timeout=12):
    "Tiny HTTP GET with a User-Agent (many public APIs reject the default one)."
    req = urllib.request.Request(url, headers={"User-Agent": "tina4-harness/0.1 (teaching demo)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()

@tool
def search_web(query: str) -> str:
    "Search the internet (Wikipedia) and return the best matching snippet. args: query"
    url = ("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=1"
           "&srsearch=" + urllib.parse.quote(query))
    hits = json.loads(_http_get(url))["query"]["search"]
    if not hits:
        return "no results"
    top = hits[0]
    return f"{top['title']}: {re.sub('<.*?>', '', top['snippet'])}"

@tool
def star_wars(resource: str) -> str:
    "Look up Star Wars data from SWAPI. args: resource (e.g. 'people/1', 'planets/1', 'films/1')"
    data = json.loads(_http_get(f"https://swapi.info/api/{resource.strip('/')}"))
    if isinstance(data, dict):
        return json.dumps({k: data[k] for k in list(data)[:6]})   # first few fields
    return json.dumps(data[:3])

TOOLS_PROMPT = tools_prompt(TOOLS)   # rebuild so the model learns the new tools
print("tools now:", list(TOOLS))
print("\nsearch_web demo:", search_web("Python programming language"))
print("star_wars demo:", star_wars("people/1"))
''')
md(r"""
Now let the agent choose a tool to answer a question. (It is still a 1.5B — it may need the
repeat-guard; you are watching the *tool call* happen, which is the point.)
""")
code(r'''
answer = run_agent("Use the star_wars tool to look up people/1, then tell me that person's name.",
                   max_steps=4)
print("\nANSWER:", answer)
''')

# ------------------------------------------------------------------ 10 RAG
md(r"""
## 10. RAG — a memory bigger than the context window

Our model's context is small and slow. Paste a 50-page manual into the prompt and it is
expensive, and the real fact gets "lost in the middle". **Groundwire** (`tina4stack/groundwire`)
keeps the whole corpus **off the GPU** (an on-disk/in-RAM index) and hands the model only the few
chunks that match the question. The model's footprint stays constant no matter how big the corpus
is — *retrieval substitutes for attention*.

First, a long "needle in a haystack" document: one real fact buried in a lot of filler.
""")
code(r'''
def make_haystack(needle, paragraphs=90):
    filler = ("This section of the operations manual describes routine procedures and "
              "escalation paths. Follow the standard checklist and record each step.")
    body = []
    for i in range(paragraphs):
        body.append(f"section {i+1}. {filler}")
        if i == paragraphs // 2:
            body.append(f"section {i+1}a. IMPORTANT: {needle}")
    return "\n\n".join(body)

NEEDLE = "The vault access code for the Rondebosch branch is GRM-2026-XT-5590."
QUESTION = "What is the vault access code for the Rondebosch branch?"
document = make_haystack(NEEDLE)
print(f"document is {len(document)} chars (~{len(document)//4} tokens), one needle inside")

def ask(question, context, max_tokens=120):
    msgs = [
        {"role": "system", "content": "Answer using ONLY the context. Be brief. "
                                       "If it is not there, say you could not find it."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    t0 = time.time()
    out = model.generate(msgs, max_tokens=max_tokens)
    return split_thinking(out)[1] or out.strip(), time.time() - t0
''')
md(r"""
### 10a. WITHOUT RAG — stuff the whole document into the prompt
The entire manual rides into the model on every call. Note the context size and the time.
""")
code(r'''
text, secs = ask(QUESTION, document)
print(f"context sent: {len(document)} chars | {secs:.1f}s")
print("answer:", textwrap.shorten(text, 300))
''')
md(r"""
### 10b. WITH RAG — retrieve only the matching chunks
Groundwire indexes the document, then returns just the top few small chunks. An exact identifier
like `GRM-2026-XT-5590` comes back byte-for-byte from a lexical index, so a weak reader finds it.
""")
code(r'''
from groundwire import Groundwire

docs = Groundwire(memory="bm25", k=3, max_words=40)   # bm25 = in-RAM lexical; small chunks
docs.ingest(document)

hits = docs.retrieve(QUESTION)
retrieved = "\n\n".join(text for _cid, text, _score in hits)
print(f"retrieved {len(hits)} chunks = {len(retrieved)} chars (document was {len(document)}):\n")
print(textwrap.shorten(retrieved, 300), "\n")

text, secs = ask(QUESTION, retrieved)
print(f"context sent: {len(retrieved)} chars | {secs:.1f}s")
print("answer:", textwrap.shorten(text, 300))
''')
md(r"""
**Same model, same question.** With RAG the context is a fraction of the size and the call is
faster, because retrieval substitutes for attention. The gap only grows with the corpus: stuffing
scales with the document; retrieval stays flat.

### 10c. RAG as a *tool* the agent can choose
Better still: give the agent a `search_docs` tool and let the loop decide when to look things up.
This is retrieval *inside* the harness — the §7 loop, with one more tool.
""")
code(r'''
@tool
def search_docs(query: str) -> str:
    "Search the manual corpus and return the passages that best match a query. args: query"
    return "\n\n".join(f"[{i+1}] {t}" for i, (_c, t, _s) in enumerate(docs.retrieve(query))) or "no matches"

TOOLS_PROMPT = tools_prompt(TOOLS)
answer = run_agent(f"Use search_docs to answer: {QUESTION}", max_steps=4)
print("\nANSWER:", answer)
''')

# ------------------------------------------------------------------ 11 cell editor
md(r"""
## 11. A highlight-and-prompt cell editor

Here the harness earns its keep on *your* work: paste a snippet, type a one-line instruction, and
the small model makes the little change. **You do the thinking** — it does the typing. This is a
single call (no loop): the model returns only the edited code, and we strip the `<think>` and pull
out the code block.
""")
code(r'''
def edit_code(code: str, instruction: str, max_tokens: int = 700) -> str:
    "Apply a small instruction to a snippet and return ONLY the edited code."
    msgs = [
        {"role": "system", "content": "You are a code editor. Apply the instruction to the code "
                                       "and return ONLY the full edited code in one ``` fenced block. "
                                       "No explanation."},
        {"role": "user", "content": f"Instruction: {instruction}\n\nCode:\n```\n{code}\n```"},
    ]
    _, body = split_thinking(model.generate(msgs, max_tokens=max_tokens, temperature=0.0))
    m = re.search(r"```[a-zA-Z0-9]*\n(.*?)```", body, re.DOTALL)
    return (m.group(1) if m else body).strip()

SAMPLE = "def greet(name):\n    print('hi ' + name)"
print(edit_code(SAMPLE, "rename the function to say_hello and use an f-string"))
''')
md(r"""
And the interactive version — paste code on the left, type the change, press **Apply**:
""")
code(r'''
try:
    import ipywidgets as W
    from IPython.display import display

    code_box = W.Textarea(value=SAMPLE, layout=W.Layout(width="60%", height="120px"))
    instr = W.Text(value="add a docstring", description="change:", layout=W.Layout(width="60%"))
    result = W.Output()
    apply = W.Button(description="Apply", button_style="primary")

    def on_apply(_):
        result.clear_output()
        with result:
            print(edit_code(code_box.value, instr.value))
    apply.on_click(on_apply)
    display(W.VBox([code_box, instr, apply, result]))
except Exception as e:
    print("(no ipywidgets — call edit_code(code, instruction) directly)", e)
''')

# ------------------------------------------------------------------ recap
md(r"""
## Recap — what a harness is

You built every part:

| part | cell | what it does |
|------|------|--------------|
| model | §1 | predicts text; knows nothing about tools |
| telemetry | §2 | manages memory; temperature & cache knobs |
| tools | §3, §9 | the only way the model touches the world (files, shell, internet, an API) |
| tools prompt | §4 | tells the model the tools exist, and the exact format |
| system prompt | §5 | how to behave: think briefly, one step at a time |
| parser | §6 | turns the model's text into an action |
| loop | §7 | ask → act → feed back → repeat; curates context, guards loops, limits steps |
| RAG | §10 | unlimited context, off the GPU, via Groundwire |
| editor | §11 | highlight-and-prompt small edits — you drive, it types |

Nothing here is specific to one model or vendor. Swap `load_model()` for any chat model and the
same loop makes it an agent. That is all a harness is.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (tina4-harness)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path("/Users/mac/projects/tina4-harness/notebooks/harness.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out, "with", len(cells), "cells")
