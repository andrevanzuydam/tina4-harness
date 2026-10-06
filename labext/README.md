# tina4-harness-lab

A small **JupyterLab 4** extension that puts a local LLM to work *inside* the notebook — using
the model the notebook already loaded in its kernel. No server, no API key.

It is built from scratch for this repo (nothing is reused from other projects); the only shared
thing is the model weights.

## What it adds

**Right-click a cell:**
- **Harness: Check Code** — reviews the active cell; the review renders as Markdown.
- **Harness: Create Cell…** — describe a cell; the model writes it; you **Insert & run**.
- **Harness: Fix Code** — rewrites your **highlighted selection** (whole cell only if nothing is
  selected) and offers **Replace / Insert below / Cancel**.

**Right sidebar (💬):**
- **Chat** for ideation. It is *notebook-aware*: code it drafts is offered as **Insert / Insert &
  run / Replace** pills and goes into **cells** (never shown in the chat), it writes `@tool`
  functions for the harness, and it reads your current **selection** as context (“fix *this*”).
- **Chat to an image** — when an image document is the active tab, your question is answered by the
  **vision** model (via `harness.lab`'s `vision` action → `harness.vision.VisionModel`).

**Clean demos:** a notebook with metadata `{"harness": {"autostart": true}}` runs its cells tagged
`startup` automatically on open — so setup can be hidden and the surface starts ready
(see `notebooks/demo.ipynb`).

## How it talks to the model

The extension never ships a model. On each action it runs one line in the notebook's kernel:

```python
import harness.lab as _hl; print(_hl.harness_edit("<action>", "<base64 payload>"))
```

`harness.lab` (in `../src/harness/lab.py`) uses the notebook's `model` — **loading one if the
kernel has none** — and serialises requests across `HARNESS_THREADS` (default 4) workers. Replies
come back as a single `__HARNESS__{json}` line on stdout.

## Build & install

Needs **Node 18+**. The project's `./install.sh` does all of this for you; to do it by hand:

```bash
# from the repo root, with the notebook venv active on PATH
cd labext
jlpm install                       # yarn, node-modules linker (see .yarnrc.yml)
jlpm run build:lib:prod            # tsc → lib/
jupyter labextension build .       # @jupyterlab/builder → tina4_harness_lab/labextension/

# install it where JupyterLab looks (or use `jupyter labextension develop .`)
cp -R tina4_harness_lab/labextension/* \
  "$VIRTUAL_ENV/share/jupyter/labextensions/tina4-harness-lab/"
```

Restart JupyterLab (or reload the page) and confirm:

```bash
jupyter labextension list | grep harness     # tina4-harness-lab vX enabled OK
```

## Develop

```bash
jlpm run build:lib    # tsc with sourcemaps
# re-run `jupyter labextension build .` and reload the JupyterLab page to pick up changes
```

Source is one file, `src/index.ts`. Styling is `style/base.css`. The kernel side is
`../src/harness/lab.py` (pure Python — a kernel restart or fresh import picks up edits).

## Requirements

- JupyterLab `^4`
- Node `>=18` (build only)
- The notebook kernel must be able to `import harness` (the extension adds `src`/`../src` to
  `sys.path` automatically, so running from the repo works out of the box).
