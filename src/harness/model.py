"""The model adapter — the *only* part of the harness that knows which engine runs the model.

A harness does not care how tokens are produced. It needs three things from a model:

    1. ``generate(messages)``        turn a chat transcript into the next assistant message
    2. ``memory()``                  how much GPU/CPU memory it is using right now
    3. a couple of knobs            temperature and a cache limit

So we hide the engine behind one small interface, ``HarnessModel``, and pick a backend
automatically:

    * **Apple Silicon**  -> the ``mlx`` backend (fast, Metal GPU, tiny 4-bit weights).
    * **everything else** (Windows, Linux, CPU or NVIDIA) -> the ``transformers`` backend.

The agent loop, the tools and the prompts never import a backend. Swap the model and the
rest of the notebook is unchanged — that separation is the whole point of a harness.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

GB = 1024 ** 3

# Two faces of the same distilled model. MLX hosts a 4-bit port; transformers reads the
# original weights. Both are DeepSeek-R1-Distill-Qwen-1.5B — a 1.5B reasoning *distill*.
MLX_REPO = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
TORCH_REPO = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@dataclass
class Settings:
    """The live knobs a reader can change from a notebook cell."""

    temperature: float = 0.2          # 0 = deterministic; higher = more varied, less reliable tool use
    max_tokens: int = 1024            # cap on a single reply (a reasoning distill needs room to think)
    cache_limit_gb: float | None = None  # engine buffer-cache ceiling; None = engine default


@dataclass
class GenStats:
    """What the last generation cost — shown in the trace so the mechanism is visible."""

    prompt_tokens: int = 0
    generation_tokens: int = 0
    tokens_per_second: float = 0.0
    peak_gb: float = 0.0


class HarnessModel:
    """Interface every backend implements. Not abstract on purpose — the backends fill it in."""

    repo_id: str = ""
    backend: str = ""
    device: str = ""
    settings: Settings
    last_stats: GenStats

    # --- generation -------------------------------------------------------------------
    def generate(
        self,
        messages: list[dict],
        on_text: Callable[[str], None] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return the next assistant message for ``messages`` (a list of {role, content}).

        If ``on_text`` is given, it is called with each streamed chunk so a notebook can
        print the model thinking in real time.
        """
        raise NotImplementedError

    # --- telemetry --------------------------------------------------------------------
    def memory(self) -> dict:
        """A dict of memory figures in GB: weights, active, peak, cache (keys vary by backend)."""
        raise NotImplementedError

    def set_cache_limit_gb(self, gb: float | None) -> None:
        raise NotImplementedError

    def clear_cache(self) -> None:
        raise NotImplementedError

    def reset_peak(self) -> None:
        raise NotImplementedError


def pick_backend() -> str:
    """Choose a backend for this machine. Override with the HARNESS_BACKEND env var."""
    import os

    forced = os.environ.get("HARNESS_BACKEND")
    if forced:
        return forced.lower()
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "torch"


def load_model(
    backend: str | None = None,
    models_dir: Path | str | None = None,
    allow_download: bool = True,
    settings: Settings | None = None,
) -> HarnessModel:
    """Load the distilled model with the right backend for this machine.

    The weights are fetched into the project's ``models/`` folder on first use (see
    ``download.ensure_model``), so a second run — and an offline run — is instant.
    """
    backend = (backend or pick_backend()).lower()
    settings = settings or Settings()

    if backend == "mlx":
        from .backends.mlx_backend import MLXModel

        return MLXModel(models_dir=models_dir, allow_download=allow_download, settings=settings)
    if backend in ("torch", "transformers"):
        from .backends.torch_backend import TorchModel

        return TorchModel(models_dir=models_dir, allow_download=allow_download, settings=settings)
    raise ValueError(f"Unknown backend '{backend}'. Use 'mlx' or 'torch'.")


def format_memory(mem: dict) -> str:
    """A one-line, human-readable summary of ``model.memory()`` for the telemetry cell."""
    parts = [f"backend={mem.get('backend')}", f"device={mem.get('device')}"]
    for key in ("weights_gb", "active_gb", "peak_gb", "cache_gb", "reserved_gb", "disk_gb"):
        if mem.get(key) is not None:
            parts.append(f"{key}={mem[key]}")
    if mem.get("cache_limit_gb") is not None:
        parts.append(f"cache_limit_gb={mem['cache_limit_gb']}")
    return " · ".join(parts)
