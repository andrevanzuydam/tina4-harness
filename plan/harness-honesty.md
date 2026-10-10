# Feature: Honest, self-checking, collaborative harness

**Outcome:** the harness works WITH the user toward a solution — it checks its own output by running
it, it asks when it is unsure instead of guessing, and when it gets stuck it defers honestly rather
than pasting broken code or inventing an API. (User's stated principles.)

## Scope
- [x] Ground import/attribute errors against the REAL module (`_ground_import_error`): on an
      ImportError/ModuleNotFound/AttributeError, import the real module, list its actual members, and
      tell the model the symbol doesn't exist + don't invent an API or suggest installing a present
      module (or, if truly missing, that we're offline — no install).
- [x] Self-check generated code by RUNNING it before pasting: if it crashes, ground the error and
      retry (up to twice); `_safe_to_run` skips stdin/destructive/network code rather than running it.
- [x] Honest deferral: after retries still fail, return the best attempt as a CHAT message with the
      real error and a question ("can you point me at the right approach?") — never auto-insert broken
      code. Edits/fixes lean on prior cells and are validated separately (not pre-run in isolation).
- [x] Ask when unsure: a clarifying rule at the top of the system prompt — ambiguous/underspecified
      requests get ONE short question; clear requests are acted on directly (no needless questions).
- [x] PROACTIVE grounding (ground before generating, not only after). The model doesn't know the long
      tail from its weights, so hand it the facts first:
      - tina4: ground EVERY tina4-mode turn in the bundled docs — questions AND code generation
        (previously code slipped through ungrounded, which is where it invented tina4 APIs).
      - working folder: a BUILD request pulls RELEVANT existing project code (models/routes/helpers)
        from the workspace index and tells the model to reuse it — relevance-gated (shared term) so a
        self-contained ask doesn't get unrelated code dumped on it.

## Tests (real — no mocks)
- [x] `_ground_import_error` over os / a missing module / a non-import error (deterministic).
- [x] `_safe_to_run` allows defs/compute, blocks os.remove / file-write / requests / input.
- [x] Live: ambiguous ("fix it", "do that thing") → asks; clear ("write a function…") → code;
      guaranteed-error code → self-check fixes or defers (verified pasted code runs clean).

## Bugs
- [ ] Still 2B-limited: it hallucinates APIs on first generation; the self-check/grounding catches it
      on the run, but a stronger model (Gemma) hallucinates less. Full-suite flakes ~1–2 items/run at
      temp 0.2 (not these changes — same before).

## Commits
- 9f0b781  ground import/attribute errors against the real module
- 325471c  self-check generated code by running it + honest deferral + ask-when-unsure

## Status: Complete (living — tighten as testing surfaces gaps)
