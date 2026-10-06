"""Vision — let the model *read an image*.

Qwen3.8-2B-Distill is vision-capable (it ships an image processor and vision tokens), so the
same small model that runs the text harness can also look at a picture. The text path uses
``mlx_lm``; the vision path needs the image processor, so it goes through **mlx-vlm** on Apple
Silicon (and transformers' image-text-to-text class on Windows/Linux).

This loads a *second* copy of the weights (~1 GB) because the two loaders are different — fine on
an M3 Max, but that is why vision is its own optional helper rather than folded into every run.

    from harness.vision import VisionModel
    vision = VisionModel()
    print(vision.describe("photo.png", "What is in this image?"))
"""
from __future__ import annotations

from pathlib import Path

from .download import MODELS_DIR, _has_weights, ensure_model
from .model import MLX_LOCAL_DIR, MLX_REPO, Settings, pick_backend


def _resolve_model_path(models_dir=None) -> str:
    """The local 4-bit folder if present, else fetch the HF weights into models/."""
    base = Path(models_dir) if models_dir else MODELS_DIR
    local = base / MLX_LOCAL_DIR
    if _has_weights(local):
        return str(local)
    return str(ensure_model(MLX_REPO, base))


class VisionModel:
    """Reads an image and answers a question about it. Backend auto-selected like the text model."""

    def __init__(self, backend: str | None = None, models_dir=None, settings: Settings | None = None):
        self.backend = (backend or pick_backend()).lower()
        self.settings = settings or Settings()
        self.path = _resolve_model_path(models_dir)

        if self.backend == "mlx":
            from mlx_vlm import load
            from mlx_vlm.prompt_utils import apply_chat_template

            self._apply_chat_template = apply_chat_template
            self.model, self.processor = load(self.path)
            self.config = getattr(self.model, "config", None)
        else:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self._torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.path, torch_dtype=dtype, trust_remote_code=True
            ).to(self.device)
            self.model.eval()

    def describe(self, image, prompt: str = "Describe this image.", max_tokens: int = 200) -> str:
        """Answer ``prompt`` about ``image`` (a file path). Returns the model's text."""
        image = str(image)
        if self.backend == "mlx":
            from mlx_vlm import generate

            full_prompt = self._apply_chat_template(self.processor, self.config, prompt, num_images=1)
            result = generate(
                self.model, self.processor, full_prompt, image=[image],
                max_tokens=max_tokens, temperature=self.settings.temperature, verbose=False,
            )
            return result.text if hasattr(result, "text") else str(result)

        # transformers image-text-to-text path (Windows / Linux / CUDA)
        from PIL import Image

        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[Image.open(image).convert("RGB")],
                                return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        trimmed = out[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()
