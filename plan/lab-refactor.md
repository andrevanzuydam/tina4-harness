# Feature: Break up the monolithic lab.py

**Outcome:** `src/harness/lab.py` (~2550 lines doing a dozen jobs) is split into cohesive, focused
modules so future edits are small and local. Behaviour is IDENTICAL — verified by the suite staying
green at every step. `lab.py` stays the public entry (`harness_edit`, `_chat`, the model runtime) and
RE-EXPORTS every helper, so `tests/` (`lab._xxx`) and the extension bridge need NO changes.

## Target modules (DAG — leaf first)

- `harness/chatprompts.py` — `_CHAT_SYSTEM`, all `@@HARNESS_*@@` sentinels, `_THINK_MODES`,
  `_CALL_STOPWORDS`, `_COMMON_WORDS`, `_FILE_EXT`. (pure constants)
- `harness/coderun.py` — the Python domain: `_run_in_kernel`/`_run_summary`/`_run_sig`,
  `_valid_python`/`_valid_expr`, `_is_placeholder`/`_is_prose_placeholder`, `_safe_to_run`,
  `_defs_only`/`_defined_names`/`_drop_redefs`/`_strip_redefs_in_reply`/`_strip_script_scaffold`/
  `_strip_docstrings`, `_behaviour_equiv`/`_probe_calls`/`_characterization_test`/`_fix_verdict`,
  `_deterministic_fix`/`_py_hint`, `_functions_in`/`_simple_arg`/`_example_call`.
- `harness/chattext.py` — text/markdown shaping + routing signals: `_strip_think`, `_cap_prose`,
  `_demote_headings`/`_detable_block`, `_with_marker`, `_starts_with_code`, `_tidy_chat`,
  `_as_choices`, `_drop_bug_claims`, `_one_line`/`_answer_gist`, `_looks_like_question`,
  `_is_load_request`/`_recent_file_hint`.
- `harness/chatmemory.py` — `_evolving_context`, `_conversation_digest`.
- `harness/chatground.py` — grounding/assertions/files/tina4: `_looks_offline`, `_ground_import_error`,
  `_extract_assertion`/`_assert_report`, `_maybe_example`/`_maybe_load`/`_with_referenced`,
  `_detect_tina4` + the tina4-app helpers (`_new_tina4_app`, `_tina4_write`, …).
- `harness/lab.py` — the model runtime (`get_model`, `_generate`, `_status`, `_stream`, `_executor`),
  `_chat`, `_vision`/`_resolve_image`, `harness_edit`/`chat_fetch`/`chat_store`/`warm_up`, and the
  re-exports.

To avoid an import cycle, the shared runtime (`_status`, `_generate`, `get_model`, `_FENCE`) that a
few helpers need lives in a small `harness/runtime.py` that both `lab.py` and the modules import.

## Scope
- [ ] runtime.py (model/stream core) — re-exported by lab; suite green.
- [ ] chatprompts.py — constants.
- [ ] coderun.py — python domain.
- [ ] chattext.py — text/markdown/routing.
- [ ] chatmemory.py — conversation memory.
- [ ] chatground.py — grounding/assertions/files/tina4.
- [ ] lab.py slimmed to orchestrator + runtime + re-exports.

## Tests
- [ ] The FULL suite (deterministic + live) is green after EACH module extraction (behaviour-preserving).

## Commits
- (per stage)

## Status: In progress
