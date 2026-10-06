# Master Plan — tina4-harness

**Outcome:** A single Jupyter notebook that teaches *how an agent harness works* by building one
from scratch, cell by cell, driven by a small local reasoning **distill** model
(`mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit`, runs offline on Apple Silicon via `mlx_lm`).
The notebook is laid out didactically — **each tool in its own cell → the tools prompt in its own
cell → the main system prompt in its own cell → then the agent loop run live** — and includes a
**telemetry cell** that shows the model's GPU memory with interactive controls for the KV-cache
limit and sampling temperature. It leans on the tool set and conventions of `tina4-simple-agent`
and acts on this scaffolded tina4-python project.

This is a teaching demo, so the whole plan lives here (single file).

## Scope
- [x] Scaffold tina4-python project (`tina4 init python`) as the app the harness acts on
- [x] `src/harness/download.py` — resolve/fetch the model into the project's `models/` folder
      (auto-download via huggingface_hub if missing; real files, no symlinks → copyable & Windows-safe)
- [x] Pluggable, auto-selected backends (loop/tools/prompts stay backend-agnostic):
      - [x] `src/harness/backends/mlx_backend.py` — Apple Silicon (mlx_lm); the primary path
      - [x] `src/harness/backends/torch_backend.py` — Windows/Linux/CPU/CUDA (transformers) fallback
- [x] `src/harness/model.py` — adapter factory: pick backend by platform, load the distill model,
      stream generate, telemetry (active/peak GPU memory, weights) + settings (temperature,
      KV-cache limit, max_tokens). Cross-platform (pathlib, no POSIX-only calls)
- [x] `src/harness/loop.py` — tool-agnostic agent loop: render messages → call model → parse a
      tool call → execute → feed the result back → repeat until `final_answer`; yields trace events
- [x] `src/harness/parse.py` — robust tool-call parser (strips `<think>`, finds the last JSON block)
- [x] `src/harness/trace.py` — pretty, notebook-friendly printing of each loop step
- [x] `notebooks/harness.ipynb` — the didactic notebook (the deliverable)
  - [x] Intro + what a harness is (markdown)
  - [x] Load the model cell
  - [x] Telemetry cell: GPU memory + interactive cache-limit & temperature controls
  - [x] One cell per tool (write_file, read_file, list_dir, run_shell, final_answer)
  - [x] Tools-prompt cell (renders the tool catalogue the model sees)
  - [x] System-prompt cell (the harness's main instructions)
  - [x] The agent-loop cell — run a task live and watch tool-use happen
  - [x] A second task that acts on the tina4 project (writes a route) — optional stretch
- [x] `requirements-notebook.txt` + `install.sh` — a `.venv-notebook` with mlx_lm + JupyterLab
- [x] `README.md` — what it is, how to run
- [x] Verify: the loop actually runs end-to-end against the local MLX model

## Tests (real — no mocks)
- [x] `tests/test_harness.py` — real engine tests: repo→folder mapping, weight detection,
      symlink materialisation, offline-missing error, backend pick, Settings (no model needed)
- [x] Live verification: the whole `notebooks/harness.ipynb` executes end-to-end on the real MLX
      model (`jupyter nbconvert --execute`) — model load, tool loop, network tools, RAG, editor

## Decisions
- Mechanism (tools, prompts, parser, loop, RAG, editor) lives INLINE in the notebook cells — it
  is the lesson. `src/harness/` is engine-only (model load + telemetry + cross-platform backends).
- Model: local MLX `DeepSeek-R1-Distill-Qwen-1.5B-4bit`; torch/transformers fallback off Mac.
- Added per request: internet-search + SWAPI API tools, Groundwire RAG A/B, highlight-and-prompt
  cell editor, telemetry with temperature + cache controls, no-auth JupyterLab launch.

## Bugs
- [x] 1.5B distill rambled past the token budget and never emitted JSON — fixed with brief-think
      system prompt + one-shot example + curated transcript + repeat-guard
- [x] Inline tools could raise and crash the loop — loop now catches and feeds the error back
- [x] JupyterLab showed a login despite empty token — saved password hash; run.sh overrides it

## Commits
- (hash  description — one line per landed change)

## Status: Complete (2026-10-06)
