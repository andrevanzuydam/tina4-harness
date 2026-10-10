# Feature: Offline-user scenario quality gate

**Outcome:** a living, execution-based test suite that pins "what the user can actually get done"
offline — so quality is proven by real outcomes (code runs / behaves / renders right / on-topic),
not by wording, and survives model variance and model swaps.

## Scope
- [x] `tests/test_scenarios.py` — the 12+ offline jobs checked by OUTCOME against the real model.
- [x] `tests/test_chat_integration.py` — the deterministic DECISIONS (FIX gating, assertion
      validation, selection preference, 🤖 marker, multi-turn context).
- [x] `tests/conftest.py` — global isolation so the full suite is order-independent.
- [ ] Extend as new scenarios surface in testing (this is a living suite, never "done").

## Tests
- [x] Green on the default 2B (64/64) and on Gemma-4-E4B (64/64), proving the gate is model-agnostic.

## Bugs
- [ ] `test_add_two_cells_code_and_test` is the flakiest item on Gemma in the FULL suite
      (temp-0.2 variance; near-100% standalone). Watch; tighten the split if it recurs on the 2B.

## Commits
- effdca3  offline-user scenario suite (the basics quality gate)
- 54f1557  test isolation (conftest) so the suite is order-independent

## Status: Ongoing (living suite)
