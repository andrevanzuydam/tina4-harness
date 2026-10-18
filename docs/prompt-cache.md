# Prompt caching — investigated, not adopted

**Does the model server cache prompts (reuse a KV prefix across requests)? No — and prototyping
showed it would not help on the harness's current models.** This note records the measurement so
nobody re-investigates, and so it can be revisited if the model/backend changes.

Reproduce any time with [`scripts/bench_prompt_cache.py`](../scripts/bench_prompt_cache.py).

## What was tried

mlx_vlm 0.7.6 has **native** prompt-prefix caching: `stream_generate(..., prompt_cache_state=…)`
with a `PromptCacheState()` that finds the common token prefix with the previous turn, trims the KV
cache back to it, reuses it, and updates itself. The server serialises generation on one lock, so a
single shared state would be safe to reuse across requests. The harness prepends a large, mostly
fixed system prompt to every turn, so in principle that prefix never needs re-prefilling.

## What was measured (qwen38-2b-distill-4bit, Apple Silicon, 2026-10)

| Scenario | Baseline TTFT | Cached TTFT | Speedup |
|---|---|---|---|
| Realistic system prompt (~900 tok) | 0.270s | 0.272s | **0.99×** |
| Large system prompt (~4500 tok) | 1.338s | 1.326s | **1.01×** |
| Pure append, no trim needed | 1.220s | 1.252s | **~1.0×** |

- The **prefix matcher works**: with a 4508-token system prompt, `find_prefix_length` returned 4516
  — the whole prefix is detected as reusable.
- **Cached output is byte-identical** to the uncached output (so it is not harmful — just useless).
- `prompt_tokens` stays at the full count in every cached run, and TTFT does not drop.

## Why it doesn't help (root cause)

```
cache type: ArraysCache | can_trim_prompt_cache: False
```

1. The Qwen3-VL model's KV cache is an **`ArraysCache` that cannot be trimmed**. The harness's
   cross-request pattern — consecutive turns from different conversations that share only the system
   prefix — **requires** trimming the cache back to the shared prefix. It can't, so the cache is
   dropped and the prompt is fully re-prefilled.
2. Even the **pure-append** case (a strict extension of the previous turn, no trim) shows no speedup,
   so mlx_vlm 0.7.6's VLM generate path re-prefills regardless for this model.

## Decision

**Not wired in.** It would add a shared-mutable-cache correctness risk (across kernels/conversations,
and vision requests) for **zero measured gain**. Prefill on a 2B 4-bit MLX model is already cheap.

**Revisit if:** the model is swapped for one whose `make_prompt_cache` yields a trimmable
`KVCache`/`RotatingKVCache` (`can_trim_prompt_cache` → `True`), or generation moves to a text-only
`mlx-lm` path where prefix reuse is effective. Re-run `scripts/bench_prompt_cache.py` to check.
