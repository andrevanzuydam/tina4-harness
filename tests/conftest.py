"""Shared test setup: make every test start from clean module state.

The full suite was order-dependent — only test_chat_integration.py reset the retrieval indexes, so
when it ran before test_scenarios.py a stale repo index / changed cwd leaked across and destabilised
unrelated scenarios (on BOTH models, not just one). This autouse fixture isolates every test: fresh
repo + tina4-docs indexes, and the working directory restored. State, not the model, was the flake.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def _isolate_harness_state():
    from harness import repo as _repo
    from harness import tina4 as _tina4

    cwd = os.getcwd()

    def _reset() -> None:
        _repo._index = None
        _repo._indexed_path = None
        _tina4._docs_index_cache.clear()

    _reset()
    yield
    _reset()
    try:
        os.chdir(cwd)
    except Exception:
        pass
