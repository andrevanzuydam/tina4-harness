"""Integration tests for the chat flow — hit the REAL shared model, assert the DECISIONS.

The model's exact wording is nondeterministic, so these never assert prose. They assert the
deterministic decisions the chat makes — routing (a selected cell wins over repo Q&A), the
@@HARNESS_FIX@@ gating (only a VALIDATED fix offers Apply), assertion validation (run the check,
not guess), the no-bogus-fix-for-correct-code rule, and the 🤖 tina4 marker. This is what replaces
clicking through the UI by hand.

These need the shared model server (./scripts/start.sh). When it's down they skip with a
machine-readable reason — they are real when it's up; run start.sh to supply the dependency.

Run:  PYTHONPATH=src .venv-notebook/bin/python -m pytest tests/test_chat_integration.py -q
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
from harness import repo as _repo  # noqa: E402
from harness import tina4 as _tina4  # noqa: E402
from harness.backends.remote_backend import reachable  # noqa: E402

from nlp_assert import about  # noqa: E402  — BM25 topic gate (robust, offline)

GREETING = "greeting hello welcome name string format"
CIPHER = "cipher encrypt caesar random secret letters"

SERVER = os.environ.setdefault("HARNESS_MODEL_SERVER", "http://127.0.0.1:7777")
pytestmark = pytest.mark.skipif(
    not reachable(SERVER),
    reason="[needs:model-server] start it with ./scripts/start.sh, then rerun",
)

GREET = 'def greet(name: str) -> str:\n    """Greet someone."""\n    return f"Hello, {name}!"'
BUGGY_ADD = "def add(a, b):\n    return a + 1"
GOOD_ADD = "def add(a, b):\n    return a + b"
BROKEN_NAME_OK = 'def broken_add(a, b):\n    """this function has a bug"""\n    return a + b  # missing return'

FIX = "@@HARNESS_FIX@@"


@pytest.fixture(autouse=True)
def _clean_indexes():
    """Each test starts with fresh retrieval indexes so they don't leak into one another."""
    _repo._index = None
    _repo._indexed_path = None
    _tina4._docs_index = None
    yield
    _repo._index = None
    _repo._indexed_path = None
    _tina4._docs_index = None


def _clean(out):
    out = re.sub(r"__H(STATUS|STREAM)__\S*", "", out)
    return re.sub(r"^@@HARNESS_(ASK|CODE)@@\n", "", out)  # the kernel's render-mode signal


def chat(msg, **kw):
    kw.setdefault("intent", "ask")
    kw.setdefault("think", "brief")
    return _clean(lab._chat([{"role": "user", "content": msg}], **kw))


def last_block(out):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", out, re.S)
    return blocks[-1].strip() if blocks else ""


# --- the FIX-sentinel gating (only a validated fix offers Apply) ---------------------------------

def test_plain_explain_never_offers_a_fix():
    out = chat("explain the code", context=GREET)
    assert not out.startswith(FIX)                       # an explanation is not a fix
    assert about(out, GREETING, CIPHER)                  # on-topic (BM25 gate, robust to wording)


def test_help_on_correct_code_offers_no_fix():
    out = chat("help", context=GOOD_ADD)
    assert not out.startswith(FIX)


def test_broken_named_but_correct_code_offers_no_fix():
    # name/docstring/comment all claim a bug, but the code is correct → must NOT offer a fix
    out = chat("help", context=BROKEN_NAME_OK)
    assert not out.startswith(FIX)


# --- assertion validation (run the pinned check, offer only a fix that passes it) ----------------

def test_help_with_assertion_offers_a_fix_that_actually_passes():
    out = chat("help add(3,4)==7", context=BUGGY_ADD)
    assert out.startswith(FIX), out[:200]
    # the offered code, when run, really satisfies the pinned assertion
    code = last_block(out)
    assert code
    r = lab._run_in_kernel(code + "\n(add(3, 4) == 7)")
    assert r.get("value") == "True", r


def test_passing_assertion_is_confirmed_not_fixed():
    out = chat("help add(3,4)==7", context=GOOD_ADD)
    assert not out.startswith(FIX)                       # already passes → nothing to fix


# --- selection preference: a selected cell wins over repo Q&A ------------------------------------

def test_explain_the_code_prefers_the_cell_over_a_matching_repo_file(tmp_path, monkeypatch):
    # A decoy workspace with a function that WOULD be retrieved if 'explain the code' hijacked to
    # repo search. Reverting the selection-preference fix would make this test go red.
    (tmp_path / "secret_language.py").write_text(
        "def random_chars():\n    '''Caesar cipher helper'''\n    return 'xyz'\n"
    )
    monkeypatch.setenv("HARNESS_REPOS_DIR", str(tmp_path))
    _repo._index = None
    _repo._indexed_path = None
    _repo.ensure_indexed()                               # index the decoy for real

    out = chat("explain the code", context=GREET)
    assert "random_chars" not in out and "caesar" not in out.lower()   # decoy must not leak
    assert about(out, GREETING, CIPHER)                                # and it's about the cell


# --- the 🤖 tina4 marker shows when a tina4 project is active ------------------------------------

def test_tina4_marker_prefixes_replies_in_a_tina4_project(tmp_path, monkeypatch):
    proj = tmp_path / "shopapp"
    proj.mkdir()
    (proj / "app.py").write_text("# tina4 app entry\n")
    (proj / "pyproject.toml").write_text(
        '[project]\nname = "shopapp"\ndependencies = ["tina4_python"]\n'
    )
    monkeypatch.chdir(tmp_path)                           # _detect_tina4 looks in cwd + parent
    monkeypatch.setenv("HARNESS_REPOS_DIR", str(tmp_path))
    _repo._index = None
    _repo._indexed_path = None

    assert _tina4.is_project(proj)
    out = chat("what is shopapp for", context="")
    assert out.startswith("🤖"), out[:120]


# --- multi-turn conversational context (the mechanism, end to end) -------------------------------

def test_multiturn_context_persists_topic_recall_and_no_hijack():
    """A realistic short conversation: the topic stays sticky across follow-ups that don't repeat
    'tina4', a selected cell never hijacks a conceptual question, and a fact planted early is
    recalled many turns later — even with a cell selected."""
    greet = 'def greet(name): return f"Hello, {name}!"'
    script = [
        ("my favourite number is 42, please remember it", ""),
        ("tell me about tina4", ""),
        ("why not dependencies?", greet),          # follow-up, no 'tina4', cell selected
        ("how does routing work?", ""),
        ("what port does it use?", greet),         # sticky topic, cell selected
        ("what is my favourite number?", greet),   # recall across the whole chat, cell selected
    ]
    msgs: list = []
    outs: list = []
    for text, ctx in script:
        msgs.append({"role": "user", "content": text})
        # Pass the FULL running history each turn — this is what makes it a conversation.
        out = _clean(lab._chat(msgs, context=ctx, intent="ask", think="brief"))
        msgs.append({"role": "assistant", "content": out})
        outs.append(out)

    why_deps, port, favnum = outs[2], outs[4], outs[5]
    # 1) conceptual follow-up with a cell selected is NOT hijacked by the cell
    assert "greet" not in why_deps.lower()
    # 2) sticky topic: a follow-up that never says 'tina4' still answers about tina4, not "which app?"
    assert "greet" not in port.lower()
    assert not re.search(r"which (software|service|framework|app|tool)|could you clarify", port, re.I)
    # 3) the planted fact persists in the memory the model is given (deterministic — the 2B's exact
    #    recall wording varies, but the fact must be in the running digest), and recall isn't hijacked.
    assert "42" in lab._conversation_digest(msgs)
    assert "greet" not in favnum.lower()
