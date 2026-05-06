"""BLIP image captioning — local, no API required."""
from __future__ import annotations

from PIL.Image import Image as PILImage


class BLIPModel:
    """Wraps Salesforce BLIP for unconditional and conditional image captioning."""

    def __init__(self, model_id: str, cache_dir: str) -> None:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self._torch = torch
        self.processor = BlipProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def caption(self, image: PILImage, max_new_tokens: int = 100) -> str:
        """Generate an unconditional caption for the given PIL image."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def caption_conditional(
        self,
        image: PILImage,
        prompt: str,
        max_new_tokens: int = 60,
    ) -> str:
        """Generate a caption conditioned on a text prompt."""
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(out[0], skip_special_tokens=True)
