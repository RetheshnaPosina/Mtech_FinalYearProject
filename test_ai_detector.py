"""Quick test: download 2 images and run AI-image detector on each."""
import sys
import os
import urllib.request
import tempfile
import pathlib

# Ensure project root is on path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

# ── Download helpers ─────────────────────────────────────────────────────────

def download(url: str, dest: str) -> bool:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"  Downloaded → {dest}")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


# ── Test images ──────────────────────────────────────────────────────────────
# Image 1: real photograph (Lenna — classic test image, no EXIF camera tag
#           so heuristics may flag it; use it to show score, not as ground truth)
# Image 2: AI-generated sample from Wikimedia (DALL-E / diffusion output)

IMAGES = [
    {
        "name": "real_photo.jpg",
        # Public-domain real photograph (Lenna, USC SIPI)
        "url": "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
        "expected": "real",
    },
    {
        "name": "ai_generated.jpg",
        # Wikimedia AI-generated image (clearly labeled as AI art)
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Stable_Diffusion_image_of_a_dog_on_a_rocket.jpg/512px-Stable_Diffusion_image_of_a_dog_on_a_rocket.jpg",
        "expected": "artificial",
    },
]

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    from hallucination_guard.image.ai_image_detector import detect_ai_generated

    tmpdir = tempfile.mkdtemp(prefix="ai_detector_test_")
    print(f"\nTemp dir: {tmpdir}\n")

    for img in IMAGES:
        dest = os.path.join(tmpdir, img["name"])
        print(f"[{img['expected'].upper()}] {img['name']}")
        if not download(img["url"], dest):
            print("  SKIP — download failed\n")
            continue

        result = detect_ai_generated(dest)
        match = "✓" if result.label == img["expected"] else "✗"
        print(f"  ai_probability : {result.ai_probability:.4f}")
        print(f"  label          : {result.label}  {match} (expected: {img['expected']})")
        print(f"  model_used     : {'HuggingFace ViT' if result.model_available else 'heuristics'}")
        print(f"  signals        : {result.signals}")
        print()


if __name__ == "__main__":
    main()
