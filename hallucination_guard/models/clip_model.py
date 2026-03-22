"""CLIP ViT-L/14 — text and image encoding for cross-modal comparison."""
from __future__ import annotations

import numpy as np
import torch
from PIL.Image import Image as PILImage


class CLIPModel:
    """Wraps HuggingFace CLIP for image-text similarity scoring."""

    def __init__(self, model_id: str, cache_dir: str) -> None:
        from transformers import CLIPProcessor, CLIPModel as HFCLIPModel

        self.processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = HFCLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def encode_text(self, text: str) -> np.ndarray:
        """Return L2-normalised text embedding as a 1-D numpy array."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().squeeze()

    def encode_image(self, image: PILImage) -> np.ndarray:
        """Return L2-normalised image embedding as a 1-D numpy array."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().squeeze()

    def similarity(self, image: PILImage, text: str) -> float:
        """Cosine similarity between image and text embeddings in [−1, 1]."""
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(text)
        return float(np.dot(img_emb, txt_emb))
