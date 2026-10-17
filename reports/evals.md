# Harness evals — how we prove it works

_Generated from the test sources; it parses them, so it can't drift from the suites._

**51 evals across 2 suites.**  Run them with:

```bash
PYTHONPATH=src .venv-notebook/bin/python -m pytest tests/ -q
```

The harness is a small **local** model (Qwen3.8-2B-Distill or Gemma-4-E2B, 4-bit MLX, fully offline) wrapped in a self-checking harness. These two suites are the spec of what that combination can do, and the proof it still does it.

## Live outcome-evals  ·  `tests/test_scenarios.py`

Each eval hits the **real local model** and checks the outcome — runnable code, correct behaviour, the right answer mode — never exact wording. This is the gate for *what a user can actually get done offline*.

**13 evals.**

### the #1 offline need: fix the error from the traceback

- **Fix error from traceback**
  `valid(code)`
- **Explain an error**
  `mode == "ask"`
### write code

- **Write a function that runs**
  `mode == "code" and valid(code)`
### explain

- **Explain this code**
  `mode == "ask"`
- **Explain a concept concisely**
  `mode == "ask" and "decorator" in out.lower()`
### edit existing code (the big category)

- **Add comments edits the cell** — "comments" = any added documentation: an inline # comment OR a docstring (both are valid, and
  `valid(code) and "def greet" in code and documented`
- **Add type hints**
  `valid(code) and "def greet" in code and "->" in code`
- **Add a docstring**
  `valid(code) and "def greet" in code and '"""' in code`
- **Remove the print**
  `valid(code) and not re.search(r"\bprint\s*\(", code) and "squares" in code`
- **Convert loop to comprehension**
  `valid(code) and re.search(r"\[.*for .* in .*\]", code)`
### call / test

- **Call an existing function**
  `valid(code) and "greet(" in code`
- **Write a test**
  `valid(code) and ("assert" in code.lower() or "def test" in code.lower())`
### multi-cell: one cell for the code, one for the test

- **Add two cells code and test**
  `mode == "code"`

## Deterministic logic-evals  ·  `tests/test_chat_logic.py`

No model, no network, **no mocks**: real `exec`, real temp images, a real BM25 index. These pin the deterministic helpers — routing, fix-validation, assertions, grounding — that decide how the harness behaves before the model is ever called.

**38 evals.**

### running code & summarising it (the grounding under help/fix)

- **Run in kernel returns trailing expression value**
  `r["ran"] and r["error"] is None`
- **Run in kernel captures stdout**
  `"hello world" in r["out"] and r["error"] is None`
- **Run in kernel captures error without raising**
  `r["error"] and "ZeroDivisionError" in r["error"]`
- **Run in kernel skips stdin**
  `r["skipped"] == "reads stdin" and not r["ran"]`
- **Starts with code routes by output form** — opens with a code block (optionally a short lead-in) → the model chose to write CODE
  `lab._starts_with_code("```python\ndef f():\n return 1\n```") is True`
- **Run summary shapes**
  `"ZeroDivisionError" in lab._run_summary({"ran": True, "error": "ZeroDivisionError: x", "out": ""})`
### behaviour equivalence & fix validation

- **Behaviour equiv ignores docstrings and comments**
  `lab._behaviour_equiv(a, b) is True`
- **Behaviour equiv detects real change**
  `lab._behaviour_equiv("def add(a, b):\n return a + 1", "def add(a, b):\n return a + b") is False`
- **Probe calls targets functions with distinct args**
  `"add(" in probe and "2" in probe and "3" in probe`
- **Probe calls skips side effecting code**
  `lab._probe_calls("def wipe(p):\n import os\n os.remove(p)") == ""`
- **Fix verdict cosmetic when only docstring changes**
  `lab._fix_verdict(orig, fixed) == "cosmetic"`
- **Fix verdict changes behaviour for real logic bug** — the probe calls add(2,3): a+1 -> 3, a+b -> 5  => observably different
  `lab._fix_verdict("def add(a, b):\n return a + 1", "def add(a, b):\n return a + b") == "changes_behaviour"`
- **Fix verdict fixes error when original crashes at top level**
  `lab._fix_verdict(orig, fixed) == "fixes_error"`
- **Fix verdict fix breaks when proposal errors**
  `lab._fix_verdict(orig, fixed) == "fix_breaks"`
### pinned assertions (the test-oracle path)

- **Extract assertion direct comparison**
  `lab._extract_assertion("help add(3,4)==7") == ["add(3,4)==7"]`
- **Extract assertion phrasing forms**
  `lab._extract_assertion("help double(5) should be 10") == ["double(5) == 10"]`
- **Extract assertion explicit assert keyword**
  `lab._extract_assertion("assert add(3,4)==7") == ["add(3,4)==7"]`
- **Extract assertion none for plain help**
  `lab._extract_assertion("help") == []`
- **Assert report passes on correct code**
  `rep["status"] == "pass"`
- **Assert report fails and shows actual**
  `rep["status"] == "fail"`
### clarifying-question pills

- **As choices detects clarification and truncates trailing prose**
  `c is not None`
- **As choices ignores explanatory lists**
  `lab._as_choices(EXPLANATORY) is None`
### offline classification & bug-claim filtering

- **Looks offline true for network errors**
  `lab._looks_offline("error: Could not resolve host: pypi.org") is True`
- **Looks offline false for unrelated errors**
  `lab._looks_offline("SyntaxError: invalid syntax") is False`
- **Drop bug claims removes bug sentences keeps description**
  `"bug" not in out.lower()`
### grounding an import error against the REAL module (no hallucinated APIs)

- **Ground import error names real module members and denies fake symbol** — 'os' is importable but has no 'Client' — grounding must say so from reality, not guess.
  `"importable" in g.lower()`
- **Ground import error missing module says offline no install**
  `"not available" in g.lower() and "offline" in g.lower()`
- **Ground import error ignores non import errors**
  `lab._ground_import_error("ZeroDivisionError: division by zero") == ""`
### vision image normalisation (the real quality lever)

- **Prep image upscales small image toward target**
  `out != str(src)`
- **Prep image downscales large image toward target**
  `px <= 1.5e6`
- **Prep image leaves in band image untouched**
  `out == str(src)`
### tina4: route targeting & request detection

- **Tina4 target home page**
  `(name, url, is_page) == ("home", "/", True)`
- **Tina4 target api endpoint with path**
  `name == "status" and url == "/status" and is_page is False`
- **Is tina4 file request vs cell**
  `lab._is_tina4_file_request("add a route") is True`
### tina4: offline doc grounding over the bundled docs

- **Bundled docs present for python and js**
  `"python" in dirs and "js" in dirs`
- **Ask tina4 grounds offline and tags source**
  `out.strip()`
### persisting conversation memory (digest + evolving context)

- **Conversation digest preserves an early fact**
  `"42" in digest`
- **Evolving context retrieves the relevant old turn** — 30 turns of filler, with the fact planted at the very start; a recall query must pull it back.
  `"42" in joined`
