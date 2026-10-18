# Tuning & runtime knobs

Every operational lever for the harness, with its **real default read from the code** (not from
memory). All are environment variables unless noted; set them before `./scripts/start.sh`.

## Model & interpreter

| Knob | Default | What it does |
|------|---------|--------------|
| `HARNESS_MLX_REPO` | `empero-ai/Qwen3.8-2B-Distill` | Which MLX model to load. A non-default repo downloads to its own `models/<repo with / → __>/` folder, so it never collides with the Qwen 4-bit weights. Swap with no code change — see [gemma-model-swap.md](gemma-model-swap.md). |
| `HARNESS_MLX_LOCAL_DIR` | `qwen38-2b-distill-4bit` (for the default repo) | Override the local weights folder name under `models/`. |
| `HARNESS_TORCH_REPO` | = `HARNESS_MLX_REPO` | The HF repo for the PyTorch backend (Windows / Linux / CPU / CUDA). |
| `HARNESS_BACKEND` | auto-detect | Force `mlx` or `torch` instead of auto-choosing by platform. |
| `HARNESS_VENV` | `.venv-notebook` | Which virtualenv `scripts/start.sh` runs from. Set `HARNESS_VENV=.venv-315` to run the whole harness on **Python 3.15** (the 3.12 `.venv-notebook` stays as the fallback). |

```bash
# run Gemma 4 E2B on Python 3.15:
HARNESS_VENV=.venv-315 HARNESS_MLX_REPO=mlx-community/gemma-4-e2b-it-4bit ./scripts/start.sh
```

The default model's chat template and Gemma's are documented in [model-templates.md](model-templates.md).

## Temperature

**`temperature = 0.2`** for all generation — set in `Settings` (`src/harness/model.py`), **not**
env-overridable today. Low but not zero: reliable, with a little variation. The comment flags the
tradeoff: *0 = deterministic; higher = more varied, less reliable tool use.*

Two paths force **`temperature = 0.0`** (determinism matters more than variety there):
- the pytest **expected-value** probe — ask the model for one literal value (`lab.py` ~L1651);
- **multi-cell test** generation — follow the rigid pytest template, don't wander into fixtures (`lab.py` ~L2640).

## Vision — image size

**Optimal: ~1.2 million total pixels** (any aspect ratio — it's the pixel *count* that matters).

`backends/mlx_backend.py::_prep_image` normalises every image **up or down** with a sharp LANCZOS
filter to `HARNESS_VISION_PIXELS` (default **`1200000`**) before the vision model sees it, keeping
aspect ratio and writing a lossless WebP. Empirically where the 2B reads small text best — **bigger
regresses** (the `import antigravity` garbled→correct case). Images already within ±8% of target are
left untouched. File **format is irrelevant** (PNG/WebP/JPEG are pixel-identical to the model); only
the pixel count matters.

- `HARNESS_VISION_PIXELS` (env) — retune globally. A different VLM may have a different sweet spot.
- `describe(..., pixels=N)` — per-request override.

## Repetition penalty (stops the small model looping)

| Knob | Default | What it does |
|------|---------|--------------|
| `HARNESS_REPETITION_PENALTY` | `1.18` | Penalises repeats so the 2B doesn't degenerate into a loop ("No ORM, No routing, …" forever). |
| `HARNESS_REPETITION_CONTEXT` | `64` | How many recent tokens the penalty considers. |

## Shared server

| Knob | Default | What it does |
|------|---------|--------------|
| `HARNESS_MODEL_PORT` | `7777` | Port for the one shared model server. |
| `HARNESS_MODEL_SERVER` | `http://127.0.0.1:7777` | Where each kernel's `RemoteModel` proxy sends generate/describe. |
| `HARNESS_THREADS` | `8` | Concurrent in-flight requests the server queues (generation is still serialised on one lock). |

## Context, paths, logging

| Knob | Default | What it does |
|------|---------|--------------|
| `HARNESS_CONTEXT_CHARS` | `60000` | Overall char cap for the bounded conversation context (recency + BM25-retrieved older turns). |
| `HARNESS_REPOS_DIR` / `HARNESS_NOTEBOOK_DIR` | `notebooks/` | The workspace/grounding root (clones land here; grounding indexes it). |
| `HARNESS_LOG_INTERACTIONS` | `1` (on) | Log every model interaction to `logs/model-interactions.jsonl`. Set `0` to disable. |

## In-UI sliders (not env)

The teaching notebook exposes two live knobs via widgets: **temperature** and the MLX **cache
limit** (`set_cache_limit_gb` — freed GPU buffers MLX keeps to reuse; `clear_cache()` frees it).
