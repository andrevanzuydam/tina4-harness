# Feature: Notebook chat UX

**Outcome:** the chat does what a user working in Python + Jupyter actually needs, offline — write
and edit code in the right place, run it, manage cells, and converse — without fighting the tool.

## Scope
- [x] Multi-cell requests ("add 2 cells — one for code, one to test it") emit one block per cell;
      the test is a REAL assert, proven to pass by running it (characterization-test fallback when
      the model guesses a wrong value); the test cell never redefines the function.
- [x] Bulk notebook ops (Jupyter commands, not model work): clear all cells (with Undo), clear all
      outputs, run all cells.
- [x] Edit-in-place vs create: modifying existing code edits the cell; cell-number targeting; delete
      cell N.
- [x] Decouple the brain from an open notebook: the chat panel runs on its own kernel when no
      notebook is open, so conversation + image questions work with nothing open; a scratch notebook
      is created only when there's code to place.
- [x] Vision intercept gated: a bare greeting with an image tab open is NOT a vision query; only a
      message that refers to the image routes to the vision model.
- [x] A code-authoring request ("write me a method … to test the code") is not hijacked to repo Q&A
      by an incidental "the code".

## Tests (real — no mocks)
- [x] `tests/test_scenarios.py::test_add_two_cells_code_and_test` — two cells, real passing test, no
      redefinition, runs clean.
- [x] Scenario suite covers fix-from-traceback, write/explain/edit/convert/call/test, vision-read.
- [x] Deterministic routing/validation helpers in `tests/test_chat_logic.py`.

## Bugs
- [x] "write me a method … test the code" hijacked to repo Q&A by "the code" — added a code-authoring
      signal that excludes it (d9679f7).
- [x] `_tidy_chat` missing `as_tool` arg swallowed the multi-cell split (TypeError) — fixed (d9679f7).
- [x] `hi` with an image open → forced vision — gated to real image questions (d9679f7).
- [x] "How about in a new notebook" (image open) → mis-routed to vision and HALLUCINATED an image
      description. The gate triggered on a bare question word ("how") / deictic. Tightened: vision
      needs an affirmative image signal (image noun, perception verb, "what's in the picture", or the
      filename) AND must not be a notebook/code action.
- [x] "…in a new notebook" did not create a notebook. Added `wantsNewNotebook`: open a fresh notebook
      and route as code so the brain writes the code into it.
- [x] "show me the code" (image context) → LOADED a random venv file (.venv-notebook/share/.../
      about.json). Root causes: (1) `_is_load_request` treated any load-verb + "the code" as a file
      open — tightened so only the explicit verbs load, and only with a real filename or "load it/the
      code"; (2) `.venv-notebook` was not skipped — added `_vendored()` that skips any `.venv*`.
- [x] Image context did not flow into the ACTION. Added an image→code path: with an image open,
      "show/write/run the code" transcribes the code FROM the image (vision) and places+runs it (new
      notebook if asked). Prompt tuned ("list every line of code visible, verbatim") + placeholder
      guard, since the 2B emits a template placeholder for "transcribe as runnable Python".
- [x] "What is happening?" (image open) → text brain returned a BRACKETED PLACEHOLDER
      ("[Analyze the situation, consider multiple hypotheses…]"). Two fixes: (1) a situational
      question with an image open now routes to vision (describes the image), unlike a concept
      question ("what is a decorator?"); (2) the ask path now detects a placeholder reply and retries
      firmly for a direct answer, else asks the user to be specific — never prints the template.

## Commits
- d9679f7  multi-cell code+test, bulk notebook ops, repo-routing + vision-intercept fixes
- a3d9c1e  decouple the harness brain from an open notebook (own kernel + scratch notebook on demand)
- (pending) vision gate tightened + "in a new notebook" creates & fills a fresh notebook

## Status: Complete
