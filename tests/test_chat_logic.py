"""Real, deterministic tests for the harness chat logic — no model, no network, no mocks.

Everything here runs against the real thing: real `exec` in _run_in_kernel, real temp image files
for the vision resize, a real Groundwire BM25 index over the bundled tina4 docs. The model-dependent
paths (full _chat generation) are exercised live elsewhere; these pin the deterministic helpers that
decide routing, fix-validation, assertions, choices, offline messaging, image normalisation and doc
grounding.

Run:  PYTHONPATH=src .venv-notebook/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # import harness without installing

from harness import lab  # noqa: E402
from harness import tina4  # noqa: E402
from harness.backends import mlx_backend  # noqa: E402


# --- running code & summarising it (the grounding under help/fix) --------------------------------

def test_run_in_kernel_returns_trailing_expression_value():
    r = lab._run_in_kernel("a = 20\nb = 22\na + b")
    assert r["ran"] and r["error"] is None
    assert r["value"] == "42"


def test_run_in_kernel_captures_stdout():
    r = lab._run_in_kernel("print('hello world')")
    assert "hello world" in r["out"] and r["error"] is None


def test_run_in_kernel_captures_error_without_raising():
    r = lab._run_in_kernel("1 / 0")
    assert r["error"] and "ZeroDivisionError" in r["error"]


def test_run_in_kernel_skips_stdin():
    r = lab._run_in_kernel("x = input('name? ')")
    assert r["skipped"] == "reads stdin" and not r["ran"]


def test_starts_with_code_routes_by_output_form():
    # opens with a code block (optionally a short lead-in) → the model chose to write CODE
    assert lab._starts_with_code("```python\ndef f():\n    return 1\n```") is True
    assert lab._starts_with_code("Here's the function:\n```python\ndef f(): ...\n```") is True
    # prose first (even with a code example later) → an explanation, not a code reply
    assert lab._starts_with_code(
        "# Tina4\nTina4 is a zero-dependency web framework for Python with routing, an ORM, "
        "templates and auth all built in.\n\nExample:\n```python\n@get('/')\n```") is False
    assert lab._starts_with_code("A list comprehension builds a list in one line.") is False


def test_run_summary_shapes():
    assert "ZeroDivisionError" in lab._run_summary({"ran": True, "error": "ZeroDivisionError: x", "out": ""})
    assert "42" in lab._run_summary({"ran": True, "error": None, "out": "", "value": "42"})
    assert "printed" in lab._run_summary({"ran": True, "error": None, "out": "hi", "value": None})


# --- behaviour equivalence & fix validation ------------------------------------------------------

def test_behaviour_equiv_ignores_docstrings_and_comments():
    a = 'def add(a, b):\n    return a + b'
    b = 'def add(a, b):\n    """sum"""\n    return a + b  # adds'
    assert lab._behaviour_equiv(a, b) is True


def test_behaviour_equiv_detects_real_change():
    assert lab._behaviour_equiv("def add(a, b):\n    return a + 1",
                                "def add(a, b):\n    return a + b") is False


def test_probe_calls_targets_functions_with_distinct_args():
    probe = lab._probe_calls("def add(a, b):\n    return a + b")
    assert "add(" in probe and "2" in probe and "3" in probe  # distinct primes


def test_probe_calls_skips_side_effecting_code():
    assert lab._probe_calls("def wipe(p):\n    import os\n    os.remove(p)") == ""


def test_fix_verdict_cosmetic_when_only_docstring_changes():
    orig = "def add(a, b):\n    return a + b"
    fixed = "def add(a, b):\n    '''adds two numbers'''\n    return a + b"
    assert lab._fix_verdict(orig, fixed) == "cosmetic"


def test_fix_verdict_changes_behaviour_for_real_logic_bug():
    # the probe calls add(2,3): a+1 -> 3, a+b -> 5  => observably different
    assert lab._fix_verdict("def add(a, b):\n    return a + 1",
                            "def add(a, b):\n    return a + b") == "changes_behaviour"


def test_fix_verdict_fixes_error_when_original_crashes_at_top_level():
    orig = "def f(n):\n    return n / 0\n\nprint(f(2))"
    fixed = "def f(n):\n    return n / 1\n\nprint(f(2))"
    assert lab._fix_verdict(orig, fixed) == "fixes_error"


def test_fix_verdict_fix_breaks_when_proposal_errors():
    orig = "x = 1\nprint(x)"
    fixed = "x = 1\nprint(y)"  # NameError at run time (valid syntax)
    assert lab._fix_verdict(orig, fixed) == "fix_breaks"


# --- pinned assertions (the test-oracle path) ----------------------------------------------------

def test_extract_assertion_direct_comparison():
    assert lab._extract_assertion("help add(3,4)==7") == ["add(3,4)==7"]


def test_extract_assertion_phrasing_forms():
    assert lab._extract_assertion("help double(5) should be 10") == ["double(5) == 10"]
    assert lab._extract_assertion("does greet('Al') returns 'Hi'") == ["greet('Al') == 'Hi'"]


def test_extract_assertion_explicit_assert_keyword():
    assert lab._extract_assertion("assert add(3,4)==7") == ["add(3,4)==7"]


def test_extract_assertion_none_for_plain_help():
    assert lab._extract_assertion("help") == []


def test_assert_report_passes_on_correct_code():
    rep = lab._assert_report("def add(a, b):\n    return a + b", ["add(3,4)==7"])
    assert rep["status"] == "pass"


def test_assert_report_fails_and_shows_actual():
    rep = lab._assert_report("def add(a, b):\n    return a + 1", ["add(3,4)==7"])
    assert rep["status"] == "fail"
    assert any("FAIL" in ln and "got 4" in ln for ln in rep["lines"])


# --- clarifying-question pills -------------------------------------------------------------------

CLARIFY = (
    "What would you like me to explain?\n"
    "- A programming language?\n"
    "- A programming concept?\n"
    "- A specific code snippet?\n"
    "- A problem you're working on? Just let me know and I'll explain!"
)
EXPLANATORY = (
    "This function does the following:\n"
    "- It takes a string argument name.\n"
    "- It returns a greeting string.\n"
    "- It uses an f-string to insert the name."
)


def test_as_choices_detects_clarification_and_truncates_trailing_prose():
    c = lab._as_choices(CLARIFY)
    assert c is not None
    assert c["options"] == ["programming language", "programming concept",
                            "specific code snippet", "problem you're working on"]


def test_as_choices_ignores_explanatory_lists():
    assert lab._as_choices(EXPLANATORY) is None


# --- offline classification & bug-claim filtering ------------------------------------------------

def test_looks_offline_true_for_network_errors():
    assert lab._looks_offline("error: Could not resolve host: pypi.org") is True
    assert lab._looks_offline("Temporary failure in name resolution") is True


def test_looks_offline_false_for_unrelated_errors():
    assert lab._looks_offline("SyntaxError: invalid syntax") is False


def test_drop_bug_claims_removes_bug_sentences_keeps_description():
    out = lab._drop_bug_claims("The bug is a missing return. This function returns the sum of a and b.")
    assert "bug" not in out.lower()
    assert "returns the sum" in out


# --- grounding an import error against the REAL module (no hallucinated APIs) --------------------

def test_ground_import_error_names_real_module_members_and_denies_fake_symbol():
    # 'os' is importable but has no 'Client' — grounding must say so from reality, not guess.
    g = lab._ground_import_error("ImportError: cannot import name 'Client' from 'os'")
    assert "importable" in g.lower()
    assert "NO `Client`" in g
    assert "REAL public names are:" in g and "DirEntry" in g  # grounded from real dir(os)
    assert "install" in g.lower()  # ... specifically telling the model NOT to suggest installing


def test_ground_import_error_missing_module_says_offline_no_install():
    g = lab._ground_import_error("ModuleNotFoundError: No module named 'tensorflow_xyz'")
    assert "not available" in g.lower() and "offline" in g.lower()


def test_ground_import_error_ignores_non_import_errors():
    assert lab._ground_import_error("ZeroDivisionError: division by zero") == ""


# --- vision image normalisation (the real quality lever) -----------------------------------------

def _make_png(tmp_path, w, h):
    from PIL import Image

    p = tmp_path / f"img_{w}x{h}.png"
    Image.new("RGB", (w, h), "white").save(p, "PNG")
    return p


def test_prep_image_upscales_small_image_toward_target(tmp_path):
    from PIL import Image

    src = _make_png(tmp_path, 300, 300)  # 90k px — well under target
    out = mlx_backend._prep_image(str(src))
    assert out != str(src)
    px = Image.open(out).size[0] * Image.open(out).size[1]
    assert 0.9e6 <= px <= 1.5e6  # normalised into the ~1.2M band


def test_prep_image_downscales_large_image_toward_target(tmp_path):
    from PIL import Image

    src = _make_png(tmp_path, 2000, 2000)  # 4M px — over target
    out = mlx_backend._prep_image(str(src))
    px = Image.open(out).size[0] * Image.open(out).size[1]
    assert px <= 1.5e6


def test_prep_image_leaves_in_band_image_untouched(tmp_path):
    src = _make_png(tmp_path, 1100, 1100)  # ~1.21M px — already in band
    out = mlx_backend._prep_image(str(src))
    assert out == str(src)


# --- tina4: route targeting & request detection --------------------------------------------------

def test_tina4_target_home_page():
    name, url, is_page = lab._tina4_target("add a home page")
    assert (name, url, is_page) == ("home", "/", True)


def test_tina4_target_api_endpoint_with_path():
    name, url, is_page = lab._tina4_target("add a /status endpoint")
    assert name == "status" and url == "/status" and is_page is False


def test_is_tina4_file_request_vs_cell():
    assert lab._is_tina4_file_request("add a route") is True
    assert lab._is_tina4_file_request("add two numbers in a cell") is False


# --- tina4: offline doc grounding over the bundled docs ------------------------------------------

def test_bundled_docs_present_for_python_and_js():
    dirs = dict((label, p) for label, p in tina4._doc_dirs(ROOT))
    assert "python" in dirs and "js" in dirs
    assert any(dirs["python"].glob("*.md")) and any(dirs["js"].glob("*.md"))


def test_ask_tina4_grounds_offline_and_tags_source():
    tina4._docs_index_cache.clear()  # fresh index for a deterministic run
    out = tina4.ask_tina4("how do I define a route", cwd=ROOT, k=3)
    assert out.strip()
    assert "# file:" in out  # chunks are tagged with their source file


# --- persisting conversation memory (digest + evolving context) ----------------------------------

def _mk(*pairs):
    msgs = []
    for role, content in pairs:
        msgs.append({"role": role, "content": content})
    return msgs


def test_conversation_digest_preserves_an_early_fact():
    msgs = _mk(
        ("user", "my favourite number is 42, remember it"),
        ("assistant", "Got it — 42."),
        ("user", "tell me about tina4"),
        ("assistant", "Tina4 is a zero-dependency web framework."),
        ("user", "how does routing work?"),
        ("assistant", "File-based routes in src/routes/."),
        ("user", "what about the orm?"),
        ("assistant", "Built-in ORM with models."),
    )
    digest = lab._conversation_digest(msgs, keep_recent=2)
    assert "42" in digest                         # the planted fact survives in the running summary
    assert "You told me" in digest                # a statement is labelled as told, not asked


def test_evolving_context_retrieves_the_relevant_old_turn():
    # 30 turns of filler, with the fact planted at the very start; a recall query must pull it back.
    msgs = _mk(("user", "my favourite number is 42"), ("assistant", "noted"))
    for i in range(30):
        msgs.append({"role": "user", "content": f"unrelated question number {i} about lists"})
        msgs.append({"role": "assistant", "content": f"here is answer {i}"})
    sel = lab._evolving_context(msgs, "what is my favourite number?", keep_recent=5)
    joined = " ".join(m["content"] for m in sel)
    assert "42" in joined                         # BM25 recalled the early fact turn


# --- semantic typing: does the code actually TYPE-CHECK, not just have a `->`? --------------------
# The third grounding oracle. _run_in_kernel proves behaviour, _assert_report proves the value,
# _typecheck proves the TYPES — by running real mypy, grounded in its actual output (not guessed).

import importlib.util as _ilu  # noqa: E402

import pytest  # noqa: E402

needs_mypy = pytest.mark.skipif(_ilu.find_spec("mypy") is None, reason="[needs:mypy] pip install mypy")


@needs_mypy
def test_typecheck_passes_on_correct_annotations():
    r = lab._typecheck("def add(a: int, b: int) -> int:\n    return a + b\n")
    assert r["checked"] and r["ok"] and r["errors"] == []


@needs_mypy
def test_typecheck_catches_wrong_return_type():
    # the arrow is PRESENT and the code RUNS — only the type checker exposes the lie
    r = lab._typecheck("def add(a: int, b: int) -> str:\n    return a + b\n")
    assert r["checked"] and not r["ok"]
    assert any("return value type" in e.lower() for e in r["errors"])  # real mypy wording


@needs_mypy
def test_typecheck_catches_incompatible_call_argument():
    code = 'def add(a: int, b: int) -> int:\n    return a + b\n\nadd("x", 1)\n'
    r = lab._typecheck(code)
    assert r["checked"] and not r["ok"]
    assert any("incompatible type" in e.lower() for e in r["errors"])


def test_typecheck_degrades_gracefully_without_mypy(monkeypatch):
    # with no checker installed it must SKIP, never crash the chat path
    real = _ilu.find_spec
    monkeypatch.setattr(_ilu, "find_spec",
                        lambda name, *a, **k: None if name == "mypy" else real(name, *a, **k))
    r = lab._typecheck("def f(x: int) -> int:\n    return x\n")
    assert r["checked"] is False and r["skipped"] == "mypy not installed"


@needs_mypy
def test_typecheck_summary_reads_like_a_human():
    assert lab._typecheck_summary(lab._typecheck("def f(x: int) -> int:\n    return x\n")) == "types check out"
    bad = lab._typecheck_summary(lab._typecheck("def f(x: int) -> str:\n    return x\n"))
    assert bad.startswith("does not type-check:")
