# Swapping the model: Gemma 4 E4B, its chat template, and the 2B comparison

This note records how to run the harness on a different MLX model, the **Gemma 4
chat-template format** (which differs from Qwen's and is why the reasoning-strip had
to be made format-agnostic), and the measured Gemma-4-E4B vs Qwen3-2B-distill
comparison. Verified on Apple Silicon (metal-gpu) with `mlx_vlm` 0.7.6.

---

## TL;DR

- The MLX model is **swappable via env** — no code edit:
  `HARNESS_MLX_REPO=mlx-community/gemma-4-e4b-it-4bit ./scripts/start.sh`
  (a non-default repo downloads to its own `models/<repo>/` folder, so it never
  collides with the Qwen 4-bit weights).
- Gemma reasons in a **channel format** `<|channel>thought … <channel|>`, *not*
  Qwen's `<think>…</think>`. `lab._strip_think` strips both now; without it, Gemma's
  reasoning leaked into answers and the output-form router misread valid code as prose.
- **The 2B stays the default.** Gemma matches it on task quality but costs ~3× the
  weights, ~1.9× peak memory, and ~1.6× latency — too much for an offline laptop tool.

---

## Gemma 4 chat-template format

Rendered for a system+user prompt (`add_generation_prompt=True`):

```
<bos><|turn>system
You are a helpful coding assistant.<turn|>
<|turn>user
Write a function to add two numbers.<turn|>
<|turn>model
```

With `enable_thinking=True`, a `<|think|>` token is injected at the top of the first
system turn.

| Element | Token(s) |
|---|---|
| Sequence start | `<bos>` |
| Turn wrapper | `<|turn>ROLE\n … <turn|>\n` (close tag flips the pipe) |
| Roles | `system`, `user`, `model` (**`assistant` → `model`**), `tool` |
| Thinking enable | `<|think|>` at top of first system turn |
| **Reasoning channel** | `<|channel>thought\n …reasoning… \n<channel|>` |
| Tool definition | `<|tool>…<tool|>` |
| Tool call | `<|tool_call>call:NAME{args}<tool_call|>` |
| Tool response | `<|tool_response>…<tool_response|>` |
| Multimodal | `<|image|>` · `<|audio|>` · `<|video|>` |

**Gotcha:** the reasoning delimiters are asymmetric — open is `<|channel>`, close is
`<channel|>` (the pipe switches sides). The template even ships a `strip_thinking()`
macro that removes those markers from *prior* assistant turns; `lab._strip_think` does
the same for the *live* reply. Full Jinja lives in the model dir at
`models/mlx-community__gemma-4-e4b-it-4bit/chat_template.jinja`.

Contrast — Qwen 2B (the default): `<|im_start|>role … <|im_end|>` turns and
`<think>…</think>` reasoning.

## Benchmark — Qwen-2B vs Gemma-4-E4B

Same battery through the same harness brain (`scripts/benchmark.py`), scoring the
outcome (runs / behaves / right render mode / on-topic), 3 runs each:

| | Qwen3-2B-distill | Gemma-4-E4B |
|---|---|---|
| Quality (full battery) | 12/12 | 12/12 (after the strip fix) |
| Latency | ~1.7 s/task | ~2.7 s/task (1.6×) |
| Peak memory | 2.99 GB | 5.67 GB (1.9×) |
| Weights | 1.6 GB | 4.8 GB (3×) |

**Vision sweet spot:** Gemma reads `python.png` flawlessly from native 304k px up to
2.6M (its SigLIP encoder normalizes to 896² ≈ 800k, so no gain above that); it only
degrades below ~120k px. Unlike the 2B — which needs the LANCZOS upscale to ~1.2M to
make fine text legible — Gemma needs no upscaling. `describe()` takes a per-request
`pixels` target so the sweet spot is tunable per model without a restart.

Net: Gemma is **proof the harness is model-agnostic, not a replacement**. The 2B is the
offline default; Gemma is the "swap in when you have the memory" option.
