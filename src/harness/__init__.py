"""tina4-harness — the *engine* helper.

This package is deliberately tiny. It holds only the part of running a local model that is
noisy and platform-specific, and **not** the lesson: downloading the weights into the
project's ``models/`` folder, and loading them on whichever engine this machine has
(MLX on Apple Silicon, PyTorch/transformers everywhere else), with memory telemetry.

The actual harness — the tools, the prompts, the parser and the agent loop — lives in the
notebook (``notebooks/harness.ipynb``), in plain cells you can read and change. That is the
whole point of the demo, so it is not hidden in here.

    from harness import load_model, format_memory

    model = load_model()                 # MLX on a Mac, torch on Windows/Linux
    print(format_memory(model.memory()))
    reply = model.generate([{"role": "user", "content": "hi"}])
"""
from .download import disk_size_gb, ensure_model, local_path_for
from .model import (
    MLX_REPO,
    TORCH_REPO,
    GenStats,
    HarnessModel,
    Settings,
    format_memory,
    load_model,
    pick_backend,
)

__all__ = [
    "load_model",
    "pick_backend",
    "Settings",
    "HarnessModel",
    "GenStats",
    "format_memory",
    "MLX_REPO",
    "TORCH_REPO",
    "ensure_model",
    "local_path_for",
    "disk_size_gb",
]
