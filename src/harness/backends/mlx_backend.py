"""MLX backend — the fast path on Apple Silicon (M-series) using the Metal GPU.

MLX gives us three things that make it a great teaching backend:
    * tiny 4-bit weights (~1 GB for a 1.5B model),
    * real GPU-memory counters we can print, and
    * an explicit *buffer-cache* knob (``set_cache_limit``) we can turn live in a cell.

A note on the two caches, because the demo talks about both:
    * **Buffer cache** (what ``set_cache_limit`` controls): MLX keeps freed GPU buffers
      around to reuse them instead of asking Metal for new ones. A bigger limit = fewer
      allocations = faster, at the cost of holding more memory. ``clear_cache()`` frees it.
    * **KV cache** (not this knob): the attention keys/values that grow with the length of
      the conversation. That one scales with context, not with a setting.
"""
from __future__ import annotations

import time
from pathlib import Path

from ..download import MODELS_DIR, _has_weights, disk_size_gb, ensure_model
from ..model import GB, MLX_LOCAL_DIR, MLX_REPO, GenStats, HarnessModel, Settings


class MLXModel(HarnessModel):
    backend = "mlx"

    def __init__(self, models_dir=None, allow_download=True, settings: Settings | None = None):
        import mlx.core as mx
        from mlx_lm import load

        self._mx = mx
        self.device = "metal-gpu"
        self.settings = settings or Settings()
        self.last_stats = GenStats()

        # Prefer the project-local 4-bit folder; otherwise fetch the HF weights into models/.
        base = Path(models_dir) if models_dir else MODELS_DIR
        local = base / MLX_LOCAL_DIR
        if _has_weights(local):
            self.path = local
            self.repo_id = MLX_LOCAL_DIR
        else:
            self.path = ensure_model(MLX_REPO, base, allow_download=allow_download)
            self.repo_id = MLX_REPO
        self.disk_gb = disk_size_gb(self.path)

        mx.reset_peak_memory()
        before = mx.get_active_memory()
        t0 = time.time()
        self.model, self.tokenizer = load(str(self.path))
        self._warmup()  # realise the (lazily loaded) weights so memory figures are honest
        self.weights_gb = max(0.0, (mx.get_active_memory() - before) / GB)
        self.load_seconds = time.time() - t0
        mx.reset_peak_memory()

        if self.settings.cache_limit_gb is not None:
            self.set_cache_limit_gb(self.settings.cache_limit_gb)

    # --- helpers ---------------------------------------------------------------------
    def _warmup(self) -> None:
        from mlx_lm import generate

        generate(self.model, self.tokenizer, "hi", max_tokens=1, verbose=False)

    def _render(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # --- generation ------------------------------------------------------------------
    def generate(self, messages, on_text=None, max_tokens=None, temperature=None):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        temp = self.settings.temperature if temperature is None else temperature
        limit = max_tokens or self.settings.max_tokens
        sampler = make_sampler(temp=temp)

        prompt = self._render(messages)
        chunks: list[str] = []
        last = None
        for resp in stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=limit, sampler=sampler
        ):
            chunks.append(resp.text)
            if on_text:
                on_text(resp.text)
            last = resp

        if last is not None:
            self.last_stats = GenStats(
                prompt_tokens=getattr(last, "prompt_tokens", 0),
                generation_tokens=getattr(last, "generation_tokens", 0),
                tokens_per_second=round(getattr(last, "generation_tps", 0.0), 1),
                peak_gb=round(self._mx.get_peak_memory() / GB, 3),
            )
        return "".join(chunks)

    # --- telemetry -------------------------------------------------------------------
    def memory(self) -> dict:
        mx = self._mx
        return {
            "backend": self.backend,
            "device": self.device,
            "weights_gb": round(self.weights_gb, 3),
            "disk_gb": round(self.disk_gb, 3),
            "active_gb": round(mx.get_active_memory() / GB, 3),
            "peak_gb": round(mx.get_peak_memory() / GB, 3),
            "cache_gb": round(mx.get_cache_memory() / GB, 3),
            "cache_limit_gb": self.settings.cache_limit_gb,
        }

    def set_cache_limit_gb(self, gb):
        # None restores MLX's own default (a large limit); a number caps the buffer cache.
        self.settings.cache_limit_gb = gb
        limit_bytes = (2 ** 62) if gb is None else int(gb * GB)
        self._mx.set_cache_limit(limit_bytes)

    def clear_cache(self):
        self._mx.clear_cache()

    def reset_peak(self):
        self._mx.reset_peak_memory()
