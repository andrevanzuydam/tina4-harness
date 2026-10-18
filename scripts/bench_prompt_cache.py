"""Benchmark mlx_vlm's native prompt-prefix cache against the harness's models.

Does reusing a `PromptCacheState` across requests (so the shared system-prompt prefix is not
re-prefilled) make generation faster? This measures time-to-first-token (TTFT) baseline vs cached,
on a realistic and a large system prompt, verifies the cached output is byte-identical, and reports
whether the model's KV cache can even be trimmed (our cross-request pattern needs trimming).

    PYTHONPATH=src .venv-notebook/bin/python scripts/bench_prompt_cache.py [model_dir]

Finding on qwen38-2b-distill-4bit (2026-10): no speedup — the model uses an ArraysCache with
can_trim_prompt_cache=False, so the prefix is detected but never reused. See docs/prompt-cache.md.
Re-run this if the model or mlx_vlm version changes.
"""
import statistics as st
import sys
import time

sys.path.insert(0, "src")
from mlx_vlm import PromptCacheState, load, stream_generate  # noqa: E402
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache  # noqa: E402

from harness.chatprompts import _CHAT_SYSTEM  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/qwen38-2b-distill-4bit"
USERS = ["add two numbers", "reverse a string", "what is a list comprehension",
         "sort a dict by value", "read a file line by line"]


def main() -> None:
    print(f"loading {MODEL} …")
    model, processor = load(MODEL)
    tok = getattr(processor, "tokenizer", processor)

    lm = getattr(model, "language_model", model)
    cache = make_prompt_cache(lm)
    print(f"cache type: {type(cache[0]).__name__} | can_trim_prompt_cache: {can_trim_prompt_cache(cache)}")

    def render(sys_text, user):
        return tok.apply_chat_template(
            [{"role": "system", "content": sys_text}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def run(sys_text, user, pc=None, mt=50):
        kw = {"prompt_cache_state": pc} if pc is not None else {}
        t0 = time.time(); ttft = None; last = None; text = []
        for r in stream_generate(model, processor, render(sys_text, user), image=None,
                                 max_tokens=mt, temperature=0.0, **kw):
            if ttft is None:
                ttft = time.time() - t0
            text.append(r.text); last = r
        return ttft, time.time() - t0, getattr(last, "prompt_tokens", None), "".join(text)

    for label, sys_text in [("realistic system prompt", _CHAT_SYSTEM),
                            ("large system prompt", _CHAT_SYSTEM + "\n```python\n"
                             + "def f(x):\n    return x + 1\n" * 300 + "\n```")]:
        ntok = len(tok.encode(sys_text))
        print(f"\n=== {label} ({ntok} tokens) ===")
        run(sys_text, "warm up")  # compile; excluded
        base = [run(sys_text, u) for u in USERS]
        pc = PromptCacheState()
        cached = [run(sys_text, u, pc=pc) for u in USERS]
        b = st.mean(x[0] for x in base)
        c = st.mean(x[0] for x in cached[1:])  # skip cold first cached call
        print(f"  baseline      TTFT avg {b:.3f}s")
        print(f"  cached(2..N)  TTFT avg {c:.3f}s   → speedup {b / c:.2f}x")

    # correctness: cache must not change the output
    pc = PromptCacheState()
    run(_CHAT_SYSTEM, "prime with something else", pc=pc)
    a = run(_CHAT_SYSTEM, "multiply two numbers")[3]
    d = run(_CHAT_SYSTEM, "multiply two numbers", pc=pc)[3]
    print(f"\ncorrectness — cached output identical to baseline: {a == d}")


if __name__ == "__main__":
    main()
