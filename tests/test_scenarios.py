"""Offline Python/Jupyter scenario suite — the jobs a user offline (no internet) relies on this tool
for: fix an error, write code, explain, edit (add comments/types/docstring, remove, refactor,
convert), call a function, a how-to, a test. Hits the REAL shared model and checks the OUTCOME
(valid code / behaviour / render mode / on-topic), not exact wording, so it survives model variance.

This is the quality gate for "what the user can actually get done". Skips when the model server is
down (start ./scripts/start.sh).

Run:  PYTHONPATH=src .venv-notebook/bin/python -m pytest tests/test_scenarios.py -q
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harness import lab  # noqa: E402
from harness.backends.remote_backend import reachable  # noqa: E402

SERVER = os.environ.setdefault("HARNESS_MODEL_SERVER", "http://127.0.0.1:7777")
pytestmark = pytest.mark.skipif(
    not reachable(SERVER),
    reason="[needs:model-server] start it with ./scripts/start.sh, then rerun",
)

GOOD = 'def greet(name):\n    return "Hello " + name'
BUGGY = ('def total(items):\n    t = 0\n    for i in items:\n'
         '        t += i["price"] * i["qty"]\n    return T')
LOOP = 'squares = []\nfor i in range(10):\n    squares.append(i * i)\nprint(squares)'


def chat(msg, intent="code", context="", output=""):
    raw = lab._chat([{"role": "user", "content": msg}], context=context, output=output,
                    intent=intent, think="brief")
    mode = ("code" if raw.startswith("@@HARNESS_CODE@@")
            else "ask" if raw.startswith("@@HARNESS_ASK@@")
            else "fix" if raw.startswith("@@HARNESS_FIX@@")
            else "other")
    out = re.sub(r"__H(STATUS|STREAM)__\S*|@@HARNESS_(ASK|CODE|FIX|CHOICES)@@\S*\n?", "", raw)
    return mode, out


def code_of(out):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", out, re.S)
    return blocks[-1].strip() if blocks else ""


def valid(code):
    return bool(code) and lab._valid_python(code)[0]


# --- the #1 offline need: fix the error from the traceback -------------------------------------

def test_fix_error_from_traceback():
    mode, out = chat("fix this", context=BUGGY, output="NameError: name 'T' is not defined")
    code = code_of(out)
    assert valid(code)                       # the fix is runnable
    assert not re.search(r"return\s+T\b", code)   # the undefined name is gone


def test_explain_an_error():
    mode, out = chat("what does this error mean?", intent="ask",
                     output="NameError: name 'T' is not defined")
    assert mode == "ask"
    assert re.search(r"not defined|nameerror", out, re.I)


# --- write code -------------------------------------------------------------------------------

def test_write_a_function_that_runs():
    mode, out = chat("write a function that counts the words in a sentence")
    code = code_of(out)
    assert mode == "code" and valid(code)
    fn = re.search(r"def\s+(\w+)\s*\(", code)
    assert fn
    r = lab._run_in_kernel(code + f'\nprint({fn.group(1)}("one two three four"))')
    assert "4" in (r.get("out", "") + str(r.get("value")))   # it actually counts words


# --- explain ----------------------------------------------------------------------------------

def test_explain_this_code():
    mode, out = chat("what does this do?", intent="ask", context=GOOD)
    assert mode == "ask"
    assert re.search(r"greet|hello", out, re.I)


def test_explain_a_concept_concisely():
    mode, out = chat("what is a decorator?", intent="ask")
    assert mode == "ask" and "decorator" in out.lower()
    assert len(out) < 1600                    # short — the user asks for more if they want it


# --- edit existing code (the big category) -----------------------------------------------------

def test_add_comments_edits_the_cell():
    mode, out = chat("add comments to this", context=GOOD)
    code = code_of(out)
    assert valid(code) and "def greet" in code and "#" in code


def test_add_type_hints():
    mode, out = chat("add type hints to this function", context=GOOD)
    code = code_of(out)
    assert valid(code) and "def greet" in code and "->" in code


def test_add_a_docstring():
    mode, out = chat("add a docstring to this", context=GOOD)
    code = code_of(out)
    assert valid(code) and "def greet" in code and '"""' in code


def test_remove_the_print():
    mode, out = chat("remove the print statement", context=LOOP)
    code = code_of(out)
    assert valid(code) and not re.search(r"\bprint\s*\(", code) and "squares" in code


def test_convert_loop_to_comprehension():
    mode, out = chat("convert this loop to a list comprehension", context=LOOP)
    code = code_of(out)
    assert valid(code) and re.search(r"\[.*for .* in .*\]", code)


# --- call / test ------------------------------------------------------------------------------

def test_call_an_existing_function():
    mode, out = chat("call this function with an example value", context=GOOD)
    code = code_of(out)
    assert valid(code) and "greet(" in code


def test_write_a_test():
    mode, out = chat("write a test for this function", context=GOOD)
    code = code_of(out)
    assert valid(code) and ("assert" in code.lower() or "def test" in code.lower())


# --- multi-cell: one cell for the code, one for the test -------------------------------------

def test_add_two_cells_code_and_test():
    mode, out = chat("add 2 cells - one for code and one to test the code, "
                     "write me a method to add 2 numbers together")
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", out, re.S)
    assert mode == "code"
    assert len(blocks) == 2                        # one cell for the function, one for the test
    fn, test = blocks[0].strip(), blocks[1].strip()
    assert valid(fn) and "def " in fn              # cell 1 defines the function
    assert valid(test) and "assert" in test        # cell 2 is a real assert-based test
    assert "def " not in test                       # the test cell does NOT redefine the function
    r = lab._run_in_kernel(fn + "\n" + test)        # both cells run in sequence and the test passes
    assert not r.get("error"), r
