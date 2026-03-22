"""Singleton lazy-loading registry for all local HuggingFace models."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Sentinel object — distinct from None so that a failed load is not retried
# every call, but also does not mask a legitimately None model slot.
_LOAD_FAILED = object()


class ModelHub(object):
    """Thread-safe singleton that lazily loads each model on first access."""

    _instance: Optional["ModelHub"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ModelHub":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._nli = None
                inst._clip = None
                inst._blip = None
                inst._sentence = None
                inst._cnn = None
                inst._florence = None
                cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    # NLI
    # ------------------------------------------------------------------

    @property
    def nli(self) -> Any:
        if self._nli is None:
            try:
                from hallucination_guard.config import settings
                from hallucination_guard.models.nli_model import NLIModel
                loaded = NLIModel(
                    model_id=str(settings.nli_model_id),
                    cache_dir=str(settings.model_cache_dir),
                )
                self._nli = loaded
            except Exception as e:
                logger.warning("NLI model load failed (will retry next call): %s", e)
                self._nli = _LOAD_FAILED
        if self._nli is _LOAD_FAILED:
            self._nli = None
            return None
        return self._nli

    # ------------------------------------------------------------------
    # CLIP
    # ------------------------------------------------------------------

    @property
    def clip(self) -> Any:
        if self._clip is None:
            try:
                from hallucination_guard.config import settings
                from hallucination_guard.models.clip_model import CLIPModel
                loaded = CLIPModel(
                    model_id=str(settings.clip_model_id),
                    cache_dir=str(settings.model_cache_dir),
                )
                self._clip = loaded
            except Exception as e:
                logger.warning("CLIP model load failed (will retry next call): %s", e)
                self._clip = _LOAD_FAILED
        if self._clip is _LOAD_FAILED:
            self._clip = None
            return None
        return self._clip

    # ------------------------------------------------------------------
    # BLIP
    # ------------------------------------------------------------------

    @property
    def blip(self) -> Any:
        if self._blip is None:
            try:
                from hallucination_guard.config import settings
                from hallucination_guard.models.blip_model import BLIPModel
                loaded = BLIPModel(
                    model_id=str(settings.blip_model_id),
                    cache_dir=str(settings.model_cache_dir),
                )
                self._blip = loaded
            except Exception as e:
                logger.warning("BLIP model load failed (will retry next call): %s", e)
                self._blip = _LOAD_FAILED
        if self._blip is _LOAD_FAILED:
            self._blip = None
            return None
        return self._blip

    # ------------------------------------------------------------------
    # Sentence
    # ------------------------------------------------------------------

    @property
    def sentence(self) -> Any:
        if self._sentence is None:
            try:
                from hallucination_guard.config import settings
                from hallucination_guard.models.sentence_model import SentenceModel
                loaded = SentenceModel(
                    model_id=str(settings.sentence_model_id),
                    cache_dir=str(settings.model_cache_dir),
                )
                self._sentence = loaded
            except Exception as e:
                logger.warning("Sentence model load failed (will retry next call): %s", e)
                self._sentence = _LOAD_FAILED
        if self._sentence is _LOAD_FAILED:
            self._sentence = None
            return None
        return self._sentence

    # ------------------------------------------------------------------
    # CNN (TensorFlow / Keras)
    # ------------------------------------------------------------------

    @property
    def cnn(self) -> Any:
        if self._cnn is None:
            try:
                from hallucination_guard.config import settings
                cnn_path = Path(settings.cnn_model_path)
                if not cnn_path.exists():
                    raise FileNotFoundError(f"CNN model not found at {cnn_path}")
                # TODO: verify SHA-256 hash of .h5 file before loading
                import tensorflow as tf
                loaded = tf.keras.models.load_model(str(cnn_path))
                self._cnn = loaded
            except Exception as e:
                logger.warning("CNN model load failed (will retry next call): %s", e)
                self._cnn = _LOAD_FAILED
        if self._cnn is _LOAD_FAILED:
            self._cnn = None
            return None
        return self._cnn

    # ------------------------------------------------------------------
    # Florence-2 (unified OCR + OD + Dense Caption, 230M, CPU-feasible)
    # ------------------------------------------------------------------

    @property
    def florence(self) -> Any:
        """Florence-2-base: unified OCR + OD + Dense Caption (230M, CPU-feasible)."""
        if self._florence is None:
            try:
                from hallucination_guard.image.florence_extractor import _load_florence
                loaded = _load_florence()
                self._florence = loaded
            except Exception as e:
                logger.warning("Florence model load failed (will retry next call): %s", e)
                self._florence = _LOAD_FAILED
        if self._florence is _LOAD_FAILED:
            self._florence = None
            return None
        return self._florence


# Module-level singleton instance
hub = ModelHub()


# Convenience accessor functions
def get_nli_model():
    return hub.nli


def get_clip_model():
    return hub.clip


def get_blip_model():
    return hub.blip


def get_sentence_model():
    return hub.sentence


def get_cnn_model():
    return hub.cnn
