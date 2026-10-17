# Feature: Python 3.15

**Outcome:** The harness runs on Python 3.15, every new 3.15 language feature is proven by a real
executable test (version-guarded, run on the actual 3.15 interpreter — no mocks), and the 3.15
"what's new" paradigms are grounded into the offline retrieval corpus so the harness can answer
questions about them.

Driven by three user asks (2026-10-17): "groundwire the latest python 3.15 with its new paradigms —
write tests", "we need to test for all the new python features", "Restart on Python 3.15".

## Feasibility (verified 2026-10-17)
- `uv python install 3.15` → **cpython-3.15.0rc2** (also a free-threaded build). Installed.
- Dry-run resolve on 3.15rc2: **mlx==0.32.4, mlx-metal==0.32.4, mlx-vlm==0.7.8, numpy==2.5.4** all
  have real wheels → the FULL harness (model server + lab) can run on 3.15, not just the kernels.
- Current harness interpreter: 3.12.12 (`.venv-notebook`).

## Scope
- [x] Make `scripts/start.sh` interpreter env-overridable (`HARNESS_VENV`, default `.venv-notebook`)
- [x] Build `.venv-315` on 3.15rc2 with the full stack (reversible: keep `.venv-notebook` 3.12 as fallback)
      — needed a `cmake` wheel to source-build sentencepiece + miniaudio + pyyaml (no rc2 wheels)
- [x] Copy the built labext + line-number overrides into `.venv-315/share/jupyter/`
- [x] Restart server + lab on `.venv-315` (Gemma E2B); verified /info, /memory (3.307 GB), /lab 200, cells run on 3.15.0rc2
- [x] Run the deterministic suite on 3.15 — 38 passed
- [x] Run the live scenario suite on 3.15 — green (full suite 69 passed, `-W error::DeprecationWarning`)
- [x] Fix the one 3.15 deprecation in our code (`re.split(..., 1)` → `maxsplit=1`, lab.py)
- [ ] Enumerate ALL 3.15 features from the full whats-new (213 KB; fetch in offset chunks)
- [ ] Groundwire: distil the 3.15 paradigms into `data/` and wire into the grounding corpus
- [ ] Grounding retrieval tests (query "lazy import" → PEP 810 chunk, "frozendict" → PEP 814, …)

## Tests (written first, real — no mocks)
Execution tests run on the real 3.15 interpreter, each guarded `@pytest.mark.skipif(sys.version_info < (3,15))`:
- [ ] PEP 810 `lazy import` defers loading
- [ ] PEP 814 `frozendict` immutable + hashable + order-independent equality
- [ ] PEP 798 `*`/`**` unpacking in list/set/dict/generator comprehensions
- [ ] PEP 661 `sentinel` identity preserved across copy
- [ ] `match` unary-plus literal patterns (`case +1:`)
- [ ] `slice[int]` subscription (slice is generic)
- [ ] `bytearray.take_bytes`, `Counter.__xor__`, improved AttributeError suggestions
- [ ] PEP 686 UTF-8 default encoding for I/O
- [ ] grounding: 3.15 queries retrieve the right whats-new chunk (offline, no model)

## Bugs
- (none yet)

## Commits
- (this milestone) HARNESS_VENV override + re.split fix + .venv-315 + plan + turn/eval report tools

## Status: Platform DONE — harness runs on 3.15.0rc2, 69/69 green. Remaining: feature tests + groundwire.
