# Master Plan — tina4-harness

The teaching harness: a JupyterLab extension (`labext/`) + a kernel-side Python harness
(`src/harness/`) driving a small **local** model (default Qwen3-2B-distill, 4-bit MLX) fully
offline, to show how an agent harness works, with Tina4 integration throughout.

This plan folder was established mid-project (it does not back-document every earlier commit); it is
the live dashboard from here forward. Work is plan-driven per the tina4-developer-python method:
every feature has a Scope checklist, real Tests (no mocks), a Bugs section, and a Commit log, and the
plan file is updated in the SAME turn progress is claimed.

| Feature | Plan | Status |
|---------|------|--------|
| Notebook chat UX (multi-cell, bulk ops, edit-in-place, decoupling) | [notebook-chat-ux.md](notebook-chat-ux.md) | ✅ Complete |
| Model-agnostic swap + Gemma proof-of-use | [model-swap-gemma.md](model-swap-gemma.md) | ✅ Complete |
| Honest, self-checking, collaborative harness | [harness-honesty.md](harness-honesty.md) | ✅ Complete (living) |
| Offline-user scenario quality gate | [scenario-quality-gate.md](scenario-quality-gate.md) | 🟡 Ongoing (living suite) |
| Python 3.15 (run on it, test features, groundwire) | [python-315.md](python-315.md) | 🟡 Platform done; feature tests + groundwire in progress |
| Semantic typing eval (type-check, not just `->`) | [semantic-typing-eval.md](semantic-typing-eval.md) | 📋 Planned |

## Conventions (tina4-developer-python)

- 🤖 marker on every reply while a tina4 skill is active; 💥 **Bazinga!** only on an earned win
  (a green suite on the real model, a feature verified end to end) — never faked.
- Real tests, no mocks: deterministic logic in `tests/test_chat_logic.py`; model-dependent paths
  exercised live in `tests/test_scenarios.py` + `tests/test_chat_integration.py` against the shared
  server (`./scripts/start.sh`).
- No-hack principle (see CLAUDE.md): route by understanding, shape output by prompting, ground don't
  hardcode, prove with real tests.
- Commit trailers co-author Tina4 + Claude.
