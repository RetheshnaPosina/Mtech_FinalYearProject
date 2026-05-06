"""Florence-2 unified vision extractor — OCR + Object Detection + Dense Region Captions.
Single 230M model replaces BLIP (caption-only). CPU-feasible, MIT license.
Handles: plain text extraction, scene understanding, spatial grounding.
Reference: Florence-2 (Microsoft 2024), AVerImaTeC NeurIPS 2025 pipeline.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any

from PIL import Image

# Module-level globals for singleton model (double-checked locking, Fix #22)
_FLORENCE_MODEL = None
_FLORENCE_PROCESSOR = None

import threading as _threading
_FLORENCE_LOAD_LOCK = _threading.Lock()

# Numeric value extraction pattern
_NUM_RE = re.compile(
    r"(\$[\d,\.]+\s*(?:billion|million|trillion|B|M|T|K)?|[\d,\.]+\s*%|\b\d{4}\b|[\d,\.]+\s*(?:billion|million|trillion))",
    re.IGNORECASE,
)


@dataclass
class FlorenceResult:
    ocr_text: str = ""
    ocr_regions: List[Dict] = field(default_factory=list)
    objects_detected: List[str] = field(default_factory=list)
    dense_captions: List[str] = field(default_factory=list)
    scene_caption: str = ""
    detailed_caption: str = ""
    numeric_values: List[str] = field(default_factory=list)
    extracted_claims: List[str] = field(default_factory=list)


def _patch_config(cfg) -> None:
    """Add missing attributes for transformers >=4.45 compatibility."""
    for attr in ["vision_config", "text_config"]:
        if not hasattr(cfg, attr):
            setattr(cfg, attr, None)


def _load_florence():
    """Return cached Florence-2 model+processor. Loads once per process (~30s first call).
    Fix #22: double-checked locking prevents race condition on first load.
    """
    global _FLORENCE_MODEL, _FLORENCE_PROCESSOR

    # Fast path: already loaded
    if _FLORENCE_MODEL is not None and _FLORENCE_PROCESSOR is not None:
        return _FLORENCE_MODEL, _FLORENCE_PROCESSOR

    with _FLORENCE_LOAD_LOCK:
        # Re-check after acquiring lock (double-checked locking)
        if _FLORENCE_MODEL is not None and _FLORENCE_PROCESSOR is not None:
            return _FLORENCE_MODEL, _FLORENCE_PROCESSOR

        import transformers
        import torch
        from hallucination_guard.config import settings

        AutoProcessor = transformers.AutoProcessor
        AutoModelForCausalLM = transformers.AutoModelForCausalLM

        _m = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            cache_dir=str(settings.model_cache_dir),
            trust_remote_code=True,
            attn_implementation="eager",
        )
        _p = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base",
            cache_dir=str(settings.model_cache_dir),
            trust_remote_code=True,
        )
        _m.eval()
        # Patch for transformers compatibility
        if hasattr(_m, "config"):
            for sub in [_m.config] + list(getattr(_m.config, "_modules", {}).values()):
                if hasattr(sub, "model_type"):
                    _patch_config(sub)

        _FLORENCE_MODEL = _m
        _FLORENCE_PROCESSOR = _p

    return _FLORENCE_MODEL, _FLORENCE_PROCESSOR


def _run_task(model, processor, image: Image.Image, task: str, max_tokens: int = 512) -> Any:
    """Run a single Florence-2 task and return parsed result."""
    import torch
    with torch.no_grad():
        inputs = processor(text=task, images=image, return_tensors="pt")
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
        )
        raw = processor.batch_decode(generated_ids, skip_special_tokens=False)
        return processor.post_process_generation(
            raw[0],
            task=task,
            image_size=(image.width, image.height),
        )


def extract_all(image_path: str) -> FlorenceResult:
    """Full Florence-2 extraction: OCR + OD + Dense Captions + Scene.
    All tasks run sequentially on same model instance — no repeated loading.
    """
    result = FlorenceResult()
    try:
        image = Image.open(image_path).convert("RGB")
        model, processor = _load_florence()

        # Plain OCR text (fast path — skip region bounding boxes for speed)
        try:
            ocr_plain_data = _run_task(model, processor, image, "<OCR>", max_tokens=256)
            ocr_plain = ocr_plain_data.get("<OCR>", "")
            result.ocr_text = _clean_ocr(ocr_plain.strip())
            result.ocr_regions = []
        except Exception:
            result.ocr_text = ""
            result.ocr_regions = []

        # Object detection
        try:
            od_data = _run_task(model, processor, image, "<OD>", max_tokens=128)
            od = od_data.get("<OD>", {})
            result.objects_detected = list(set(od.get("labels", [])))
        except Exception:
            result.objects_detected = []

        # Scene caption
        try:
            cap_data = _run_task(model, processor, image, "<CAPTION>", max_tokens=64)
            result.scene_caption = cap_data.get("<CAPTION>", "").strip()
        except Exception:
            result.scene_caption = ""

        # Detailed caption
        try:
            dcap_data = _run_task(model, processor, image, "<DETAILED_CAPTION>", max_tokens=128)
            result.detailed_caption = dcap_data.get("<DETAILED_CAPTION>", "").strip()
        except Exception:
            result.detailed_caption = ""

        result.dense_captions = []  # Skip DENSE_REGION_CAPTION for speed

        # Extract numeric values from OCR text
        result.numeric_values = _NUM_RE.findall(result.ocr_text)

        # Build verifiable claims from OCR + dense captions
        result.extracted_claims = _build_claims(result.ocr_text, result.dense_captions)

    except Exception as e:
        result.scene_caption = f"florence_error:{e}"

    return result


def _clean_ocr(text: str) -> str:
    """Remove Florence-2 special tokens, social media UI noise, keep only message content."""
    import re as _re
    # Remove special tokens like <loc_0123>, <OD>, etc.
    noise_patterns = [
        r"</?[a-zA-Z_]\w*>",     # XML/special tags
        r"<loc_\d+>",             # Florence location tokens
        r"\b\d{3,6}[KkMm]?\b",   # Engagement counts (e.g. "123K likes")
        r"\b\d{1,2}[hHdDmM]\b",  # Timestamps (e.g. "3h", "2d")
        r"\b[A-Z0-9]{1,3}\b",    # Short UI labels ("RT", "DM", etc.)
    ]
    for n in noise_patterns:
        text = re.sub(n, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_claims(ocr_text: str, dense_captions: List[str]) -> List[str]:
    """Convert raw OCR text into verifiable claim sentences using spaCy NLP."""
    claims: List[str] = []
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(ocr_text[:2000])
        for sent in doc.sents:
            s = sent.text.strip()
            if len(s) >= 10:
                claims.append(s)
    except Exception:
        # Fallback: simple sentence splitting
        parts = re.split(r"[.!?\n|]+", ocr_text)
        claims = [p.strip() for p in parts if len(p.strip()) >= 10]

    # Add dense captions that look like complete sentences
    for cap in dense_captions:
        cap = cap.strip()
        if len(cap) >= 20:
            claims.append(cap)

    return claims[:10]
