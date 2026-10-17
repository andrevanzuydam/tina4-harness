# Feature: Semantic typing eval

**Outcome:** The harness can prove that type annotations it writes or edits are *semantically*
correct — that the code actually **type-checks** — not merely that a `->` or `: int` is present. The
proof is a static type checker run over the code, the same way the harness already proves *behaviour*
by running code. A deliberately wrong annotation must be caught; a correct one must pass.

Motivation: today `tests/test_scenarios.py::test_add_type_hints` only asserts `"->" in code` — that is
**syntactic**. A model can emit `def add(a: int, b: int) -> str: return a + b` and pass that check
while being type-incorrect. This feature closes that gap, and it's a direct application of the
harness's core thesis (models hallucinate; the harness must self-check against a real tool).

## How it works

The harness has three "oracles" that ground the model in reality. Semantic typing adds the third:

| Oracle | Question it answers | Mechanism (existing / new) |
|--------|---------------------|----------------------------|
| `_run_in_kernel` | Does the code **run**? | real `exec` in an isolated namespace |
| `_assert_report` | Does it produce the **right value**? | runs pinned `fn(x)==y` assertions |
| **`_typecheck`** (new) | Do the **types check out**? | runs a static type checker as a subprocess |

**The checker.** `mypy` — pure-Python, pip-installable, and **fully offline** after install (it ships
its own typeshed stubs; no network at check time). It's a *dev/test* dependency, not a notebook
runtime requirement, so the harness still runs offline on a machine without it (tests `skipif` absent).
`pyright` is rejected (needs a Node runtime); `ty` is too young to pin. Pin `mypy` in
`requirements-notebook.txt`.

**The helper** `src/harness/lab.py::_typecheck(code: str) -> dict` (mirrors `_run_in_kernel`'s shape):
1. Write `code` to an isolated temp file (`tempfile`, unique dir), never the notebook cwd.
2. Run `python -m mypy <file> --no-incremental --cache-dir=/dev/null --follow-imports=silent
   --no-error-summary` as a subprocess. No-cache + no-incremental = deterministic; `follow-imports=silent`
   keeps it to the snippet, not the whole workspace. `--strict` is opt-in per call (default off — a
   strict run flags *missing* annotations, which is a different question from *wrong* annotations).
3. Parse mypy's `file:line: error: <msg>` lines into a list; return
   `{"checked": bool, "ok": bool, "errors": [..], "skipped": "mypy not installed"?}`.
4. Best-effort: if mypy isn't importable, return `{"checked": False, "skipped": ...}` — never raise into
   the chat path.

**Two eval directions** (the negative case is what proves the check has teeth):
- **Positive:** correctly-annotated code → `ok is True`, `errors == []`.
- **Negative:** a known type error (`-> str` on `return a + b` where `a,b: int`; passing `str` to an
  `int` param) → `ok is False` and the expected error substring is present. A checker that passes the
  negative case is broken; the test asserts it fails.

**Harness integration (stretch, behind deliverable 1).** Once `_typecheck` exists, wire it as a
*validation gate* on the "add type hints" / "add types" path, exactly like code is run before it's
pasted: generate → `_typecheck` → if it doesn't check, do one grounded retry (feed mypy's error back
to the model) → if still failing, insert but flag honestly ("added types, but they don't type-check:
<error>") rather than pretending. This makes typing a first-class self-check, not a syntactic gesture.

**Why not just ask the model?** Same reason we run code instead of trusting it: the model asserting
"these types are correct" is exactly the hallucination the harness exists to catch. The type checker
is the ground truth.

## Scope
- [x] Add `mypy` to `requirements-notebook.txt` (dev/test dep; offline after install) + installed in both venvs (mypy 2.4.0)
- [x] `_typecheck(code, strict=False)` + `_typecheck_summary(res)` in `src/harness/lab.py` (tmp file + mypy subprocess + parse; never raises; skips if mypy absent)
- [x] Deterministic unit tests for `_typecheck` (positive clean, 2 negatives, skip-if-mypy-absent, summary) — 5 tests in test_chat_logic.py
- [x] Added a sibling live scenario `test_added_type_hints_actually_typecheck` asserting the model's hints TYPE-CHECK, not just `->`
- [ ] (Stretch) wire `_typecheck` as a validation gate on the add-type-hints path: grounded retry + honest flag on failure
- [x] (Stretch, positive half) live scenario: "add type hints to this" → result type-checks clean  — negative probe is the deterministic suite

## Tests (written first, real — no mocks) — asserts grounded in mypy 2.4.0's ACTUAL output
- [x] `_typecheck` ok on `def add(a: int, b: int) -> int: return a + b`
- [x] `_typecheck` NOT ok + 'return value type' on `-> str` over `return a + b` (real wording: `Incompatible return value type (got "int", expected "str")`)
- [x] `_typecheck` flags bad call arg `add("x", 1)` ('incompatible type'; `Argument 1 to "add" has incompatible type "str"; expected "int"`)
- [x] `_typecheck` degrades gracefully (skipped) when mypy isn't installed — monkeypatched find_spec
- [x] `_typecheck_summary` reads human ('types check out' / 'does not type-check: …')
- [x] live: the model's added hints TYPE-CHECK (`test_added_type_hints_actually_typecheck`)

## Bugs
- (none)

## Commits
- (this milestone) _typecheck + _typecheck_summary in lab.py; 5 deterministic + 1 live test; mypy dep

## Status: Core DONE — the eval ships (3.15 113 passed; 3.12 83 passed, 30 skipped). Remaining: the stretch self-check GATE on the add-type-hints path (generate → type-check → grounded retry → honest flag).
