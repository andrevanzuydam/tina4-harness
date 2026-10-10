# CLAUDE.md — tina4-harness

A teaching harness: a JupyterLab prebuilt extension (`labext/`, TypeScript) + a kernel-side
Python harness (`src/harness/`) that drives a small **local** model (Qwen3.8-2B-Distill, 4-bit,
MLX) fully offline, to show how an agent harness works. It also integrates **Tina4**: the chat can
provision tina4 apps, write routes/pages into them, and ground its answers in bundled tina4 docs.

## No-hack principle (this is production, not a prototype)

Solve problems at the **root**, the way an intelligent system should — never with band-aids:

- **Route by understanding, not keyword regex.** Intent (ask vs code vs help) is decided by an NLP
  classifier (the local model), not an ever-growing list of `if "tell me" in text` patterns.
  Growing a regex to catch each new phrasing is the anti-pattern this rule exists to stop.
- **Shape output by prompting, not post-processing.** If the model does something unwanted (ASCII-art
  diagrams, walls of text), fix the **prompt** so it doesn't do it — don't strip/patch the output
  after the fact. A person wouldn't suddenly emit ASCII art; the model shouldn't either.
- **Ground, don't hardcode.** Facts come from the bundled docs / retrieval, not canned strings.
- **Prove it with real tests** (no mocks), and verify end to end before calling it done.

If a fix feels like whack-a-mole, stop and solve the real cause.

## Load the Tina4 skills — even though this is NOT a tina4 app

This repo is a harness, not a tina4-python application, so do **not** impose the full tina4 app
layout (`plan/`, `src/routes/`, `src/orm/`, `TINA4.md`) on it. But it uses tina4 throughout, so
**load the tina4 skills and apply their conventions**:

- `tina4-cli` — for anything touching the `tina4` client (provisioning, `tina4 serve`, docs).
- `tina4-developer-python` — for the working method and discipline.

Conventions that DO apply here (the spirit, not the scaffolding):

- **🤖 marker** — begin replies with 🤖 while a tina4 skill is active (and the harness chat prints
  🤖 when it's in "tina4 mode" — a tina4 project is detected). 💥 **Bazinga!** on an *earned* win
  (a green test run on a real dependency, a feature verified end to end) — never faked.
- **Real tests, no mocks** — the deterministic harness logic is unit-tested in `tests/` against
  real execution (real `exec`, real temp files, real BM25 index). No stubs/fakes/"returned 200".
  The model-dependent paths (full `_chat` generation) are exercised live, not mocked.
- **Commit trailers** — work done under a tina4 skill co-authors Tina4:
  `Co-Authored-By: Tina4 <82961293+tina4stack@users.noreply.github.com>` (in addition to the
  Claude trailer). Add it to this repo's commits; never back-fill.

## Run it

```bash
./scripts/start.sh          # shared model server (:7777, 8 threads) + JupyterLab (:8890, root notebooks/)
```

- One **shared** model process; each notebook kernel is a thin `RemoteModel` proxy over HTTP.
- Vision + text use the same MLX weights (`describe()` normalises images to ~1.2M px first).
- Grounding (Groundwire BM25) is offline: stdlib docs (`data/python-help.txt`), repo code, chat
  history, and bundled **tina4 docs** (`data/tina4-docs/{python,js}/`).

## Build the JupyterLab extension

```bash
cd labext && jlpm build              # or: tsc --sourceMap && jupyter labextension build --development True .
# then copy labext/tina4_harness_lab/labextension/ → .venv-notebook/share/jupyter/labextensions/tina4-harness-lab/
```
A **kernel** change (`src/harness/*.py`) needs a kernel/lab restart; a **frontend** change needs a
browser hard-refresh (Cmd-Shift-R) after redeploy.

## Tests

```bash
PYTHONPATH=src .venv-notebook/bin/python -m pytest tests/ -q
```

- `tests/test_harness.py` — model/download plumbing.
- `tests/test_chat_logic.py` — the chat/help/fix/assertion/choices/offline/vision/tina4-docs logic
  (all deterministic, no model, no network).

## Offline note

The harness is fully offline (local model + bundled docs). The one thing that needs a network
*once* is a cold `tina4 init` (`uv sync` pulls the framework package); after the uv cache is warm,
provisioning, `tina4 serve`, routes/pages, and doc grounding all work offline.
