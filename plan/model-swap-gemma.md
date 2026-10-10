# Feature: Model-agnostic swap + Gemma proof-of-use

**Outcome:** the harness can drive a different MLX model than the default 2B, end to end (text +
vision), proving the routing/grounding is model-agnostic — not a replacement for the 2B, proof of
use. A benchmark quantifies the trade-off so the default choice is evidence-based.

## Scope
- [x] Make the MLX model swappable via env (`HARNESS_MLX_REPO` / `HARNESS_MLX_LOCAL_DIR` /
      `HARNESS_TORCH_REPO`); non-default repo downloads to its own `models/<repo>/` folder.
- [x] Model-agnostic reasoning strip: `lab._strip_think` handles Qwen `<think>…</think>` AND Gemma
      `<|channel>thought…<channel|>`.
- [x] Per-request vision pixel target: `describe(..., pixels=)` through mlx/remote backends + server,
      so the resolution sweet spot is tunable per model without a restart.
- [x] Output-form router loosened only when code was already intended (`_starts_with_code(strict=)`)
      so a captioned code reply (Gemma's style) renders as code; questions stay strict.
- [x] `scripts/benchmark.py` — model-vs-model battery scoring outcome + latency + peak memory.
- [x] `docs/gemma-model-swap.md` — the Gemma template format + the comparison.

## Tests (real — no mocks)
- [x] Full suite passes on BOTH models: Qwen 64/64; Gemma 64/64 (run-to-run temp-0.2 variance on 1
      item in the full suite, near-100% standalone).
- [x] Benchmark (3 runs each): Qwen-2B 12/12 ~1.7s/task peak 2.99GB; Gemma-4-E4B 12/12 ~2.7s/task
      peak 5.67GB — parity on quality at ~3× weights / ~1.9× memory.
- [x] Vision sweet spot measured: Gemma reads python.png flawlessly native→2.6M px; floor ~120k px.

## Bugs
- [x] Gemma's reasoning leaked into answers (different token format) → router misread code as prose;
      3 spurious failures. Fixed by format-agnostic `_strip_think` (41c955e).
- [x] `add comments` scored as fail on Gemma — it adds a docstring, not `#`; test was 2B-specific.
      Broadened to accept either (54f1557).
- [x] Full-suite flakiness on both models — cross-file state leakage. Fixed with a global isolation
      fixture `tests/conftest.py` (54f1557).

## Commits
- 41c955e  model swappable + model-agnostic reasoning/vision + benchmark (proof of use)
- 54f1557  cross-model routing (`_starts_with_code(strict=)`) + test isolation (conftest)

## Status: Complete
Default stays the 2B (lighter/faster, native fit); Gemma is the "swap in when you have the memory"
option, now a one-liner: `HARNESS_MLX_REPO=mlx-community/gemma-4-e4b-it-4bit ./scripts/start.sh`.
