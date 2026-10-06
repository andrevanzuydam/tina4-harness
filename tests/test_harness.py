"""Real tests for the engine helper — no mocks, no model download.

These cover the deterministic plumbing: how a repo id maps to a project-local folder, how
weights are detected, how symlinks are materialised (so a model folder is copyable to
Windows), and how the backend is chosen. The model-in-the-loop part is proven by running
the notebook itself (``notebooks/harness.ipynb``) or ``scripts/smoke.py``.

Run:  python -m pytest tests/ -q      (from the project root)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # import harness without installing

from harness import local_path_for, disk_size_gb  # noqa: E402
from harness.download import _dereference_symlinks, _has_weights, ensure_model  # noqa: E402
from harness.model import Settings, pick_backend  # noqa: E402


def test_repo_id_maps_to_flat_folder():
    path = local_path_for("mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit", "/tmp/models")
    assert path.name == "mlx-community__DeepSeek-R1-Distill-Qwen-1.5B-4bit"
    assert "/" not in path.name


def test_has_weights_detects_safetensors(tmp_path):
    assert not _has_weights(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"\x00\x01")
    assert _has_weights(tmp_path)


def test_ensure_model_uses_existing_without_download(tmp_path):
    repo = "fake/model"
    dst = local_path_for(repo, tmp_path)
    dst.mkdir(parents=True)
    (dst / "model.safetensors").write_bytes(b"\x00")
    # allow_download=False must NOT raise when the weights are already present
    out = ensure_model(repo, tmp_path, allow_download=False)
    assert out == dst


def test_ensure_model_raises_clearly_when_missing_and_offline(tmp_path):
    try:
        ensure_model("fake/missing", tmp_path, allow_download=False)
    except FileNotFoundError as exc:
        assert "not in" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError when weights are missing and offline")


def test_dereference_symlinks_materialises_real_files(tmp_path):
    real = tmp_path / "real.bin"
    real.write_bytes(b"hello-weights")
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    _dereference_symlinks(tmp_path)
    assert not link.is_symlink()
    assert link.read_bytes() == b"hello-weights"


def test_disk_size_counts_files(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 1024)
    assert disk_size_gb(tmp_path) > 0


def test_backend_pick_respects_env_override():
    os.environ["HARNESS_BACKEND"] = "torch"
    try:
        assert pick_backend() == "torch"
    finally:
        del os.environ["HARNESS_BACKEND"]


def test_settings_defaults_are_sane():
    s = Settings()
    assert 0 <= s.temperature <= 2
    assert s.max_tokens >= 256
