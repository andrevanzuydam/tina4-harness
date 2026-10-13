# Dev Admin Gemma backend

## Scope
Use the existing shared MLX model server with the loaded model's own bundled chat template for text and multimodal calls. Serve Ollama-compatible chat/generate through the authenticated Tina4 gateway. Keep streamed reasoning separate from answer output, with token-limit and generation-error metadata.

## Tests
- Incremental response-channel parsing across all chunk boundaries for Qwen and Gemma: 2 unittest tests pass.
- Live Gemma text request produced complete JSON and separate reasoning, done_reason stop.
- Dev Admin real writes: heading, SCSS colour, contextual colour follow-up all passed (9.431s, 24.307s, 34.837s). No scaffold or unrelated declaration changes.
- Split editor preview/reload checks tracked in tina4-dev-admin/plan/editor-testing-ux.md.

## Operations
`scripts/start-devadmin-model.sh` starts Gemma E4B on port11438. Set HARNESS_PYTHON to the installed MLX-capable Python environment. Start gateway with `tina4 ai serve --backend http://127.0.0.1:11438`.

## Commits
Uncommitted local work; no release.
