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
- [x] Enumerate ALL 3.15 features from the full whats-new (213 KB; fetched in 3 offset chunks) and
      VERIFY each on the real 3.15.0rc2 interpreter before asserting (sentinel sig, exact error wording)
- [x] Groundwire: distilled the 3.15 paradigms into `data/python-3.15/*.md` (9 chunks) + `pyhelp.search_py315`
      (clearable cache, BM25, `# file:` tags), wired into the lab.py grounding block (gated on 3.15 topics)
- [x] Grounding retrieval tests (query "lazy import" → PEP 810 chunk, "frozendict" → PEP 814, …) — 8 tests

## Tests (written first, real — no mocks) — `tests/test_py315_features.py`
Execution tests run on the real 3.15 interpreter, each guarded `@pytest.mark.skipif(sys.version_info < (3,15))`.
All values/signatures/error wording captured from 3.15.0rc2 BEFORE asserting (the harness thesis: models hallucinate APIs).
- [x] PEP 810 `lazy import` defers loading (real module run in subprocess) + runtime API + inside-function SyntaxError
- [x] PEP 814 `frozendict` builtin, not a `dict` subclass, immutable (exact TypeError), hashable, order-independent eq
- [x] PEP 798 `*`/`**` unpacking in list/set/dict/generator comprehensions (run via `eval` of a source string — the
      literal new syntax would break *parsing* the module on the 3.12 fallback; `eval` still really executes it)
- [x] PEP 661 `sentinel` — signature `(name, /, *, repr=None)` verified from the builtin, identity across copy, custom repr
- [x] `match` unary-plus literal patterns (`case +1:`)
- [x] `slice[int]` subscription (slice is generic)
- [x] `bytearray.take_bytes` (n + all), `Counter.__xor__`/`__ixor__`, improved AttributeError suggestions
      (list.push→.append, dict.put→d[k]=v — suggestion is in the rendered traceback, not `str(e)`)
- [x] `math.integer` (PEP 791) + `math.isnormal/signbit/fmax/fmin`, `re.prefixmatch`, `asyncio.TaskGroup.cancel`, tomllib 1.1
- [x] PEP 686 UTF-8 default encoding for I/O (`open()` without `encoding=` is utf-8), `__cached__` gone → `__spec__.cached`
- [x] grounding: 3.15 queries retrieve the right whats-new chunk (offline, no model) — 8 deterministic tests

Counts: 30 feature tests + 8 grounding tests. 3.15 → 38/38 pass; 3.12 fallback → 30 skip + 8 pass.
Full suite: 3.15 **107 passed**; 3.12 **77 passed, 30 skipped**.
Only parse-only (no execution) test: PEP 810 lazy-import-inside-function (the point is it must NOT compile).

## Bugs
- (none yet)

## Commits
- (platform) HARNESS_VENV override + re.split fix + .venv-315 + plan + turn/eval report tools — 9ae2eeb
- (this milestone) tests/test_py315_features.py (38 tests) + data/python-3.15/*.md corpus +
  pyhelp.search_py315 grounding wired into lab.py

## Status: DONE — every 3.15 feature proven by a real test (107/107 on 3.15), paradigms groundwired offline.
