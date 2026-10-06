# tina4-harness

**A teaching notebook that shows how an agent *harness* works — built from scratch, one cell at
a time, around a small local model you can run offline.**

A large language model only predicts text. It cannot open a file, run a command, or search the
web. The **harness** is the small program that wraps the model and turns it into an agent:

```
   task ──▶ build transcript ──▶ ask the model ──▶ read the reply
            ▲                                        │
            │                                 is it a tool call?
            │                                        │ yes
            └──────────── put result back ◀── run the tool ◀──┘
                                                     │ "final_answer"
                                                     ▼
                                                  answer
```

This repo builds that loop in the open, driven by **Qwen3.8-2B-Distill** — a tiny *distill* (a
small model taught by a big one) whose chat template natively supports tool calling. It is
deliberately small, so you can *see* every job the harness does to keep it on the rails. (The
notebook still teaches *prompted* JSON tool-calling, because it shows the mechanism and works
with any model.)

Part of the [tina4stack](https://github.com/tina4stack). Scaffolded as a `tina4-python` project,
so the harness has a real app to act on.

---

## What's inside the notebook (`notebooks/harness.ipynb`)

The cells hold the real code — read them, change them, re-run them:

1. **Load the model** — fetched once into `models/`, then offline.
2. **Memory telemetry** — live GPU memory with sliders for **temperature** and the **cache limit**.
3. **Tools** — one per cell: `write_file`, `read_file`, `list_dir`, `run_shell`, `final_answer`.
4. **The tools prompt** — how the tools are described to the model.
5. **The system prompt** — how the agent should behave.
6. **The parser** — turns the model's text into an action (strips `<think>`, finds the JSON).
7. **The loop** — ask → act → feed back → repeat; curates context, guards loops, limits steps.
8. **Run it** — watch the agent work (and watch the guardrails catch a 1.5B stumbling).
9. **Add your own tools** — an **internet search** tool (Wikipedia) and a real **API** tool (SWAPI,
   the Star Wars API). Swap in any open API (snake facts, weather, …) — the pattern is identical.
10. **RAG with Groundwire** — give the model a memory far bigger than its context window: a
    with/without comparison, then retrieval as a tool the agent can call.
11. **A highlight-and-prompt cell editor** — paste a snippet, type a one-line instruction, get the
    small edit. You do the thinking; it does the typing.
12. **Vision** — the same model reads an image (upload/drag, or a path), and as a tool the agent
    can call. (Qwen3.8-2B-Distill is vision-capable.)
13. **Right-click menus** — a JupyterLab extension (`labext/`) adds **Check Code**, **Create Cell**
    and **Fix Code** to a cell's context menu, all powered by the notebook's own in-kernel model —
    no server, no API key. "Create Cell" writes code (e.g. a SWAPI lookup) straight into a new cell.

---

## Run it

### Requirements
- **Python 3.12**
- **macOS (Apple Silicon)** → runs on **MLX** (Metal GPU, 4-bit weights, ~1.6 GB). Fast.
- **Windows / Linux / Intel Mac** → runs on **PyTorch + transformers** (CPU, or NVIDIA CUDA if
  present). The 2B model works on CPU; it is just slower.
- **Node.js 18+** (optional) — only for the §13 right-click menus. Without it the notebook still
  works fully; `install.sh` just skips building the extension.

### macOS / Linux
```bash
./install.sh      # creates .venv-notebook and installs everything (uses uv if present)
./run.sh          # starts JupyterLab with the login lock OFF, opens the notebook
```

### Windows (PowerShell)
```powershell
.\install.ps1
.\run.ps1
```

Then open the printed link — **http://127.0.0.1:8888/lab** (no token, no password; local use
only). Change the port with `PORT=8899 ./run.sh`.

> The 4-bit weights live in `models/qwen38-2b-distill-4bit/` (~1.6 GB) — plain files you can copy
> to another Mac to run without any download. On a fresh clone without that folder, the MLX
> backend fetches the unquantised weights from Hugging Face; the PyTorch backend always does.

### Just the engine, from Python
The notebook is the lesson, but the model-loading helper is importable:
```python
import sys; sys.path.insert(0, "src")
from harness import load_model, format_memory

model = load_model()                 # MLX on Apple Silicon, torch elsewhere
print(format_memory(model.memory()))
print(model.generate([{"role": "user", "content": "hello"}]))
```

---

## Layout

```
notebooks/harness.ipynb   the teaching notebook — the main thing
notebooks/demo.ipynb      a clean demo: setup hidden & auto-run, chat + right-click refactoring
src/harness/              the engine helper (the ONLY hidden part: model loading)
  model.py                backend-agnostic adapter + Settings + telemetry
  download.py             fetch weights into models/ (portable, Windows-safe)
  vision.py               read-an-image helper (mlx-vlm / transformers)
  pyhelp.py               grounded Python help: pydoc → help file → Groundwire → ask_python()
  agent.py                packaged harness (tools, loop, setup()) for the clean demo
  lab.py                  kernel bridge the menus + chat call (loads the model if needed)
  backends/mlx_backend.py Apple Silicon (mlx_lm)
  backends/torch_backend.py  Windows / Linux / CPU / CUDA (transformers)
labext/                   the JupyterLab extension (TypeScript) for the right-click menus
tests/                    real tests for the engine (no model needed): pytest tests/ -q
requirements-notebook.txt one file, cross-platform (pip environment markers pick the engine)
install.sh / run.sh       macOS & Linux
install.ps1 / run.ps1     Windows
```

## Configuration
- `HARNESS_BACKEND=mlx|torch` — force a backend (otherwise auto-detected).
- Weights live in `models/`; delete that folder to re-download.

## Credits
- Model: **Qwen3.8-2B-Distill** — a local 4-bit MLX conversion in `models/qwen38-2b-distill-4bit/`,
  or `empero-ai/Qwen3.8-2B-Distill` from Hugging Face (transformers / MLX fallback).
- RAG: [`tina4stack/groundwire`](https://github.com/tina4stack/groundwire).
- Ideas and tool patterns lean on `tina4-simple-agent`.
