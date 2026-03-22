"""FORENSICS Agent — ELA + FFT + CNN + Florence-2 (OCR+OD+DenseCaptions) + CLIP.
Florence-2 replaces BLIP: reads text on image, detects objects, describes regions.
Exports visual_facts dict including ocr_text + extracted_claims for AWP debate.
Reference: Florence-2 (Microsoft 2024), AVerImaTeC NeurIPS 2025.
"""
from __future__ import annotations

import asyncio
import logging

from hallucination_guard.agents.base_agent import BaseAgent
from hallucination_guard.trust_score import AgentMessage, Verdict
from hallucination_guard.text.evidence_pool import EvidencePool
from hallucination_guard.config import settings

logger = logging.getLogger(__name__)


class ForensicsAgent(BaseAgent):
    name = "forensics"

    async def run(
        self,
        claim: str,
        evidence_pool: EvidencePool,
        visual_facts: dict | None = None,
    ) -> AgentMessage:
        """Text-only path: ForensicsAgent returns a neutral message when no image is provided."""
        return self._make_message(
            claim=claim,
            verdict=Verdict.NOT_ENOUGH_INFO,
            confidence=0.5,
            evidence=[],
            reasoning="no_image",
        )

    async def run_with_image(
        self,
        image_path: str,
        caption: str,
        evidence_pool: EvidencePool | None = None,
    ) -> tuple[AgentMessage, dict]:
        """Full forensic analysis: ELA + FFT + CNN + Florence-2 + Deepfake + Watermark.

        Fix #12: all sub-tasks run via asyncio.gather(return_exceptions=True) with
        per-task fallback defaults, preventing a single tool failure from aborting
        the full pipeline.

        Image fusion: 0.5*CNN + 0.25*ELA + 0.25*FFT

        Returns
        -------
        (AgentMessage, visual_facts dict) where visual_facts contains all
        forensic signals needed by InvestigatorAgent and the debate orchestrator.
        """
        loop = asyncio.get_event_loop()

        # --- Fix #12: parallel sub-tasks with per-task exception return ---
        raw_results = await asyncio.gather(
            loop.run_in_executor(None, self._ela, image_path),
            loop.run_in_executor(None, self._fft, image_path),
            loop.run_in_executor(None, self._cnn, image_path),
            loop.run_in_executor(None, self._florence, image_path),
            loop.run_in_executor(None, self._deepface, image_path),
            loop.run_in_executor(None, self._watermark, image_path),
            return_exceptions=True,
        )

        # Import result types for isinstance checks
        from hallucination_guard.image.florence_extractor import FlorenceResult
        from hallucination_guard.image.deepfake_detector import DeepfakeResult
        from hallucination_guard.image.watermark_detector import WatermarkResult

        ela_r_raw, fft_r_raw, cnn_raw, florence_raw, deepfake_raw, watermark_raw = raw_results

        # Per-task fallback defaults (Fix #12)
        defaults = {
            "ela": {"mean_energy": 0.0, "anomaly_regions": [], "has_anomaly": False},
            "fft": {"spectral_score": 0.0, "has_periodic": False},
            "cnn": 0.5,
            "florence": FlorenceResult(),
            "deepfake": DeepfakeResult(),
            "watermark": WatermarkResult(),
        }

        labels = ["ela", "fft", "cnn", "florence", "deepfake", "watermark"]
        processed = []
        for label, exc in zip(labels, raw_results):
            if isinstance(exc, Exception):
                logger.warning("Forensics sub-task '%s' failed: %s", label, exc)
                processed.append(defaults[label])
            else:
                processed.append(exc)

        ela_r, fft_r, cnn_prob, florence_r, deepfake_res, watermark_res = processed

        # --- Image fusion score: 0.5*CNN + 0.25*ELA + 0.25*FFT ---
        ela_energy = ela_r.get("mean_energy", 0.0) if isinstance(ela_r, dict) else 0.0
        ela_norm = min(1.0, ela_energy / 20.0)
        fft_score = fft_r.get("spectral_score", 0.0) if isinstance(fft_r, dict) else 0.0
        has_periodic = fft_r.get("has_periodic", False) if isinstance(fft_r, dict) else False

        if not isinstance(cnn_prob, (int, float)):
            cnn_prob = 0.5

        fusion = (
            settings.cnn_weight * float(cnn_prob)
            + settings.ela_weight * ela_norm
            + settings.fft_weight * fft_score
        )
        fusion = min(1.0, max(0.0, fusion))

        # --- Context checker (out-of-context detection) ---
        context_r = await loop.run_in_executor(
            None,
            self._check_context,
            image_path,
            florence_r.ocr_text if isinstance(florence_r, FlorenceResult) else "",
            caption,
            florence_r.objects_detected if isinstance(florence_r, FlorenceResult) else [],
            florence_r.dense_captions if isinstance(florence_r, FlorenceResult) else [],
        )

        # --- Per-claim CLIP scoring ---
        claims_list = florence_r.extracted_claims if isinstance(florence_r, FlorenceResult) else []
        per_claim_clip = await loop.run_in_executor(
            None,
            self._score_claims,
            image_path,
            claims_list,
        )

        # --- Build visual_facts dict for downstream agents ---
        if isinstance(florence_r, FlorenceResult):
            ocr_text = florence_r.ocr_text
            objects_detected = florence_r.objects_detected
            dense_captions = florence_r.dense_captions
            extracted_claims = florence_r.extracted_claims
            detailed_caption = florence_r.detailed_caption
            ocr_regions = florence_r.ocr_regions
            numeric_values = florence_r.numeric_values
            scene_caption = florence_r.scene_caption
        else:
            ocr_text = ""
            objects_detected = []
            dense_captions = []
            extracted_claims = []
            detailed_caption = ""
            ocr_regions = []
            numeric_values = []
            scene_caption = ""

        if isinstance(deepfake_res, DeepfakeResult):
            faces_found = deepfake_res.faces_found > 0
            deepfake_probability = deepfake_res.deepfake_probability
        else:
            faces_found = False
            deepfake_probability = 0.0

        if isinstance(watermark_res, WatermarkResult):
            watermark_present = watermark_res.watermark_present
            watermark_type = watermark_res.watermark_type
        else:
            watermark_present = False
            watermark_type = ""

        from hallucination_guard.image.context_checker import ContextCheckResult
        if isinstance(context_r, ContextCheckResult):
            clip_score = context_r.clip_score
            is_out_of_context = context_r.is_out_of_context
            context_trust = context_r.context_trust
            mismatched_entities = context_r.mismatched_entities
        else:
            clip_score = 0.5
            is_out_of_context = False
            context_trust = 1.0
            mismatched_entities = []

        # --- Image verdict ---
        if fusion > 0.65:
            verdict = Verdict.REFUTED
        elif fusion < 0.35:
            verdict = Verdict.SUPPORTED
        else:
            verdict = Verdict.NOT_ENOUGH_INFO

        confidence = fusion

        visual_facts = {
            "mean_energy": ela_energy,
            "has_periodic": has_periodic,
            "caption": scene_caption,
            "ela_energy": ela_energy,
            "ela_anomaly_regions": ela_r.get("anomaly_regions", []) if isinstance(ela_r, dict) else [],
            "anomaly_regions": ela_r.get("anomaly_regions", []) if isinstance(ela_r, dict) else [],
            "fft_score": fft_score,
            "spectral_score": fft_score,
            "has_periodic_artifacts": has_periodic,
            "cnn_probability": float(cnn_prob),
            "fusion_score": fusion,
            "dominant_colors": [],
            "ocr_text": ocr_text,
            "ocr_regions": ocr_regions,
            "objects_detected": objects_detected,
            "dense_captions": dense_captions,
            "detailed_caption": detailed_caption,
            "numeric_values": numeric_values,
            "extracted_claims": extracted_claims,
            "context_clip_score": clip_score,
            "is_out_of_context": is_out_of_context,
            "context_trust": context_trust,
            "mismatched_entities": mismatched_entities,
            "faces_found": faces_found,
            "deepfake_probability": deepfake_probability,
            "watermark_present": watermark_present,
            "watermark_type": watermark_type,
            "per_claim_clip": per_claim_clip,
        }

        reasoning = (
            f"Image: cnn={float(cnn_prob):.3f}, ela={ela_energy:.2f}, "
            f"fft={fft_score:.3f}, fusion={fusion:.3f}, "
            f"ocr_chars={len(ocr_text)}, objects={len(objects_detected)}, "
            f"claims_extracted={len(extracted_claims)}, "
            f"out_of_context={is_out_of_context}, context_trust={context_trust:.3f}, "
            f"faces={faces_found}, deepfake_prob={deepfake_probability:.3f}, "
            f"watermark={watermark_present}"
        )

        msg = self._make_message(
            claim=caption,
            verdict=verdict,
            confidence=confidence,
            evidence=[],
            reasoning=reasoning,
        )
        return msg, visual_facts

    def _ela(self, path: str):
        from hallucination_guard.image.ela_processor import compute_ela
        return compute_ela(path)

    def _fft(self, path: str):
        from hallucination_guard.image.fft_processor import compute_fft
        return compute_fft(path)

    def _cnn(self, path: str) -> float:
        import hallucination_guard.models.model_hub as hub_module
        hub = hub_module.hub
        try:
            import numpy
            from PIL import Image
            from hallucination_guard.image.heuristic_fallback import heuristic_score
            img = Image.open(path).convert("RGB").resize((512, 512))
            arr = numpy.array(img, dtype=numpy.float32) / 255.0
            arr = arr[numpy.newaxis, ...]
            return float(hub.cnn.predict(arr)[0])
        except Exception:
            from hallucination_guard.image.heuristic_fallback import heuristic_score
            return heuristic_score(path)

    def _florence(self, path: str):
        """Florence-2: OCR + OD + Dense Captions + Scene. Replaces BLIP."""
        from hallucination_guard.image.florence_extractor import extract_all
        try:
            return extract_all(path)
        except Exception as e:
            from hallucination_guard.image.florence_extractor import FlorenceResult
            r = FlorenceResult()
            r.scene_caption = f"florence_unavailable:{e}"
            return r

    def _deepface(self, path: str):
        from hallucination_guard.image.deepfake_detector import detect_deepface
        return detect_deepface(path)

    def _watermark(self, path: str):
        from hallucination_guard.image.watermark_detector import detect_watermark
        return detect_watermark(path)

    def _score_claims(self, path: str, claims: list):
        from hallucination_guard.image.claim_image_matcher import score_claims_against_image
        return score_claims_against_image(path, claims)

    def _check_context(
        self,
        image_path: str,
        ocr_text: str,
        caption: str,
        objects: list,
        dense_caps: list,
    ):
        from hallucination_guard.image.context_checker import check_context, ContextCheckResult
        try:
            return check_context(image_path, ocr_text, caption, objects, dense_caps)
        except Exception:
            from hallucination_guard.image.context_checker import ContextCheckResult
            return ContextCheckResult(
                clip_score=0.5,
                context_trust=1.0,
                is_out_of_context=False,
                reasoning="context_check_failed",
            )
