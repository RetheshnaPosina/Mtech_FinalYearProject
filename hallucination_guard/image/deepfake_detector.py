"""DeepFace-based face detection and AI-face probability estimation.
Uses face detection confidence as a proxy for synthetic/manipulated faces.
Falls back gracefully if deepface is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DeepfakeResult:
    faces_found: int = 0
    deepfake_probability: float = 0.0
    face_details: List[dict] = field(default_factory=list)
    error: str = ""


def detect_deepface(image_path: str) -> DeepfakeResult:
    """Detect faces and estimate deepfake probability via DeepFace confidence scores.

    Strategy: DeepFace.extract_faces() returns confidence scores per detected face.
    Authentic real faces from high-resolution photos typically score 0.9+.
    AI-generated or manipulated faces often score below 0.7 due to facial
    geometry artifacts. We use (1 - avg_confidence) as deepfake probability proxy.

    Falls back to DeepfakeResult(faces_found=0, error='deepface_not_installed')
    when the deepface package is not installed.

    Parameters
    ----------
    image_path : Path to the image file.

    Returns
    -------
    DeepfakeResult dataclass with face detection results.
    """
    result = DeepfakeResult()
    try:
        import deepface
        from deepface import DeepFace

        faces = DeepFace.extract_faces(
            img_path=image_path,
            enforce_detection=False,
        )
        if not faces:
            return result

        confidences = [f.get("confidence", 0.0) for f in faces]
        avg_conf = round(sum(confidences) / len(confidences), 4)

        result.faces_found = len(faces)
        result.deepfake_probability = round(max(0.0, 1.0 - avg_conf), 4)
        result.face_details = [{"confidence": f.get("confidence", 0.0)} for f in faces]

    except ImportError:
        result.error = "deepface_not_installed"
    except Exception as e:
        result.error = str(e)[:100]

    return result
