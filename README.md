# HallucinationGuard AMADA v6.0

**Adversarial Multi-Agent Debate Architecture for Multimodal Hallucination Detection**

---

## Overview

HallucinationGuard is a production-grade hallucination detection system built on the AMADA pattern — five specialist AI agents debate claims in adversarial rounds, then a Judge produces a calibrated trust score. Supports text-only, image-only, and full multimodal (text + image) verification.

---

## Architecture

```
Input
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Sentinel (Tier 0)                                  │
│  InputValidator → RiskEstimator → TierRouter        │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────────┐
          ▼            ▼                ▼
     Tier 1       Tier 2           Tier 3
   Text AWP    + Image Forensics  + OCR claims
    Debate       (ELA/FFT/CNN/     + CMCD cross-
                  Florence-2)       modal check
          │            │                │
          └────────────┴────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Cascade Executor│
              │  BudgetManager  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────────────────────────────┐
              │  Debate Orchestrator                     │
              │                                          │
              │  ① Neutral Seed Retrieval (unframed)    │
              │                                          │
              │  Round 1 (parallel asyncio.gather):      │
              │    Prosecutor ──┐                        │
              │    Defender   ──┼──► EvidencePool        │
              │    Investigator ┘                        │
              │                                          │
              │  If gap ≥ 0.15 → Round 2:               │
              │    Prosecutor counters Defender's best   │
              │    Defender counters Prosecutor's best   │
              │    (Argument-Graph Memory)               │
              └────────────────┬────────────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────────────┐
              │  Judge Agent                             │
              │  gap < 0.15 → local DeBERTa vote        │
              │  gap ≥ 0.15 → Claude API (primary)      │
              │             → Gemini API (fallback)      │
              │             → local DeBERTa (fallback)  │
              └────────────────┬────────────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────────────┐
              │  VCADE Calibration                       │
              │  5 difficulty dimensions → calibrated    │
              │  trust + SuspicionFlag                   │
              └────────────────┬────────────────────────┘
                               │
                               ▼
                          TrustScore
                    (policy: PUBLISH/FLAG/REJECT)
```

---

## Agents

| Agent | Role | Key output |
|-------|------|------------|
| **Prosecutor** | Challenges claims; generates adversarial hypotheses; AWP scoring | `best_alt_support`, `adversarial_hypotheses`, `strongest_point` |
| **Defender** | Supports claims; retrieves corroborating evidence | `confidence`, `evidence_used`, `strongest_point` |
| **Investigator** | Neutral analysis; entity disambiguation; temporal checks | `suspicion_flags` |
| **Forensics** | Image authenticity: ELA + FFT + CNN + Florence-2 + Deepfake + Watermark | `visual_facts` dict |
| **Judge** | Final arbiter; receives only structured messages (never raw content) | `Verdict`, `confidence`, `reasoning` |

All agents share a common `EvidencePool`. Evidence is pre-seeded with neutral unframed queries before any agent runs, preventing query-framing bias.

---

## Novel Algorithms

### 1. AWP — Adversarial Weight Perturbation Scoring

Measures how robust a claim is against the best adversarial alternative:

```
adversarial_score = original_support / (original_support + best_alt_support)
```

**Data-driven thresholds** (`learn_awp_thresholds()`): instead of hand-tuned cutoffs, the system grid-searches the audit log for the (refuted_th, supported_th) pair that maximises macro-F1 over labelled entries. Falls back to defaults (0.35 / 0.72) when fewer than 20 labelled entries exist.

Configurable via `.env`:
```
AWP_REFUTED_THRESHOLD=0.35
AWP_SUPPORTED_THRESHOLD=0.72
```

---

### 2. VCADE — Versatile Calibration and Difficulty Estimation

Calibrates raw NLI trust scores using five difficulty dimensions:

| Dimension | Formula | Meaning |
|-----------|---------|---------|
| `d_retrieval` | `1 − avg_relevance` | Poor evidence retrieval |
| `d_consensus` | `min(1, stdev(relevances) × 2.5)` | Sources disagree |
| `d_adversarial` | `best_alt_support` | Strong alternative hypothesis exists |
| `d_entity` | `min(1, entity_count × 0.25)` | Entity confusion risk |
| `d_temporal` | `0.65 if temporal else 0.1` | Time-sensitive claim |

**Calibration formulas:**
- `SUPPORTED`: `raw × (1 − 0.2 × (1 − difficulty))`
- `NOT_ENOUGH_INFO`: `raw × (1 − 0.3 × (1 − difficulty))`
- `REFUTED`: `raw × difficulty` (easy fact failed = high suspicion)

**Isotonic enhancement** (`calibrate_from_logs()`): when sufficient labelled audit data exists (≥10 entries), fits an `IsotonicRegression` on `(raw_score, ground_truth)` pairs and replaces the formula entirely.

**Suspicion flags raised by VCADE:**
- `HIGH_SUSPICION_EASY_FACT_FAILED` — difficulty < 0.3 but not SUPPORTED
- `TEMPORAL_STALENESS` — temporal claim was REFUTED
- `ENTITY_CONFUSION` — >4 entities and not SUPPORTED

---

### 3. CMCD — Cross-Modal Contradiction Detection

Detects mismatches between image content and accompanying text (Tier 3):

```
1. Extract visual facts from image (color, scene, objects, BLIP caption)
2. Extract text facts from claim/caption (entities, attributes, values)
3. Detect attribute-level contradictions (e.g., text says "red car", image shows blue)
4. Compute CLIP cosine similarity (image, text)
5. cross_modal_trust = clip_similarity − penalty_weight × avg_contradiction_severity
```

Out-of-context penalty: if `is_out_of_context=True`, `overall_trust` is capped at 0.25 (forces REJECT).

---

## Debate Flow

### Round 1 (always)

All three agents run in parallel via `asyncio.gather()`:

```
Neutral seed retrieval → EvidencePool pre-populated (unframed query)
    ↓
Prosecutor.run()  ──┐
Defender.run()    ──┤ (parallel)
Investigator.run()──┘
```

### Round 2 (when confidence gap ≥ 0.15)

**Argument-Graph Memory** — each agent receives the opposing agent's `strongest_point` as a counter-argument to rebut, rather than repeating independent inference:

```
Prosecutor Round 2: counter_argument = Defender's strongest_point
Defender Round 2:   counter_argument = Prosecutor's strongest_point
```

This produces true adversarial rebuttal. Round 2 Prosecutor results are used for final AWP scoring.

---

## Cascade Tiers

| Tier | Trigger | Operations | Budget |
|------|---------|------------|--------|
| **0** | Always | Input validation, reject malformed input | 10 ms |
| **1** | Any text | AWP debate (all 3 agents + Judge) | 200 ms |
| **2** | Has image | Tier 1 + ELA/FFT/CNN/Florence-2/Deepfake/Watermark | 3 s |
| **3** | Image + high risk | Tier 2 + OCR claim fact-check + CMCD | 8 s |

---

## Image Forensics Pipeline

All sub-tasks run in parallel via `asyncio.gather(return_exceptions=True)`. Individual tool failures use per-task defaults — no single failure aborts the pipeline.

| Tool | Signal | Weight in fusion |
|------|--------|-----------------|
| **ELA** (Error Level Analysis) | `ela_energy` (normalised 0–1) | 0.25 |
| **FFT** | `fft_score`, `has_periodic_artifacts` | 0.25 |
| **CNN** | `cnn_probability` (authenticity classifier) | 0.50 |
| **Florence-2** | OCR text, objects, dense captions, extracted claims, scene caption | — |
| **Deepfake detector** | `deepfake_probability`, `faces_found` | (penalty) |
| **Watermark detector** | `watermark_type` | — |
| **Context checker** | `is_out_of_context`, `context_trust`, `mismatched_entities` | (penalty) |
| **CLIP claim scorer** | `per_claim_clip` (per-claim image–text match) | — |

**Fusion score:** `0.5 × CNN + 0.25 × ELA + 0.25 × FFT`

**Image verdict thresholds:**
- `fusion > 0.65` → REFUTED (likely manipulated)
- `fusion < 0.35` → SUPPORTED (likely authentic)
- otherwise → NOT_ENOUGH_INFO

**Florence-2** (Microsoft 2024) replaces BLIP as the captioning/OCR backbone. Outputs: scene caption, OCR text + regions, detected objects, dense region captions, extracted factual claims, numeric values.

---

## Judge Decision Logic

```
1. Compute confidence gap = |prosecutor_conf − defender_conf|
2. gap < 0.15  →  local DeBERTa weighted vote (no API call)
3. gap ≥ 0.15 + use_api=True:
     a. Claude API (claude-sonnet-4-6, timeout 30 s)
     b. Gemini API (gemini-2.0-flash, fallback)
     c. local DeBERTa weighted vote (final fallback)
```

**Local vote weights:** Prosecutor 0.4, Defender 0.35, Investigator 0.25

**Security:** Judge receives only structured `AgentMessage` objects — never raw social media content or user input. Prompt-injection is architecturally prevented.

**API response validation:** all JSON keys (`verdict`, `confidence`, `reasoning`) are validated; verdict must be one of `SUPPORTED/REFUTED/NOT_ENOUGH_INFO`; confidence must be in `[0, 1]`.

---

## Policy Engine

| `overall_trust` | Policy |
|----------------|--------|
| ≥ 0.7 | `PUBLISH` |
| 0.4 – 0.7 | `FLAG` |
| < 0.4 | `REJECT` |

---

## Models

| Model | Purpose | ID |
|-------|---------|-----|
| DeBERTa-v3 (NLI) | Entailment scoring, local Judge vote | `cross-encoder/nli-deberta-v3-base` |
| CLIP ViT-L/14 | Image–text similarity, CMCD | `openai/clip-vit-large-patch14` |
| BLIP | Fallback image captioning | `Salesforce/blip-image-captioning-base` |
| Florence-2 | OCR, object detection, dense captions | Microsoft Florence-2 |
| all-MiniLM-L6-v2 | Sentence embeddings, evidence retrieval | `sentence-transformers/all-MiniLM-L6-v2` |
| Custom CNN | Image authenticity classifier | `./model_cache/my_model.h5` |
| Claude (API) | Primary Judge for ambiguous cases | `claude-sonnet-4-6` |
| Gemini (API) | Secondary Judge fallback | `models/gemini-2.0-flash` |

**CLIP upgrade note:** ViT-L/14 replaces ViT-B/32. ViT-B/32 treats "a man biting a dog" and "a dog biting a man" as nearly identical embeddings. ViT-L/14 significantly improves relationship and composition sensitivity (Cherti et al., 2023).

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/verify/text` | Text-only claim verification |
| `POST` | `/api/v1/verify/image` | Image verification (path-based) |
| `POST` | `/api/v1/verify/image/upload` | Image verification (multipart file upload) |
| `POST` | `/api/v1/verify/full` | Multimodal text + image (path-based) |
| `POST` | `/api/v1/verify/full/upload` | Multimodal text + image (file upload) |
| `GET`  | `/api/v1/metrics` | Prometheus-style system metrics |
| `GET`  | `/health` | Health check |

**Rate limiting:** 30 requests/min/IP (sliding window). Excess returns HTTP 429.

### Example: text verification

```bash
curl -X POST http://127.0.0.1:8000/api/v1/verify/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple was founded in 1976 by Steve Jobs.", "use_api_judge": false}'
```

**Response fields:** `overall_trust`, `policy`, `tier_used`, `latency_ms`, `claims[]` (each with `verdict`, `calibrated_trust`, `difficulty_score`, `suspicion_flag`, `correction_suggestion`, `evidence_snippets`, `debate_rounds`, `api_judge_used`), `awp_fact_score`, `adversarial_detection_rate`, image forensics fields.

---

## Security

| Fix | Description |
|-----|-------------|
| Path traversal | All image paths resolved + checked for `..` before use |
| API key storage | Fernet symmetric encryption + PBKDF2-HMAC-SHA256 (100k iterations) key derivation |
| CORS | Restricted to `GET`, `POST`; `Content-Type` and `Authorization` headers only |
| Rate limiting | Sliding-window 30 req/min/IP |
| Error leakage | Raw exceptions never returned to client; `"Verification failed"` generic message only |
| Input bounds | Claims truncated to 10,000 chars; max 20 claims per request |
| Prompt injection | Judge is architecturally isolated from raw content |
| Model load safety | Models loaded to temp variable first; only written to cache after success |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Configure environment
cp .env.example .env
# Edit .env — encrypt API keys with:
python -m hallucination_guard.security.key_manager encrypt ANTHROPIC_API_KEY <your_key>
python -m hallucination_guard.security.key_manager encrypt GEMINI_API_KEY <your_key>

# 3. Start server
python -m hallucination_guard.api.app
# → http://127.0.0.1:8000

# Or use the helper script:
python run_server.py
```

---

## Project Structure

```
hallucination_guard/
├── agents/
│   ├── base_agent.py          # Shared agent base class
│   ├── prosecutor_agent.py    # Challenges claims, AWP scoring
│   ├── defender_agent.py      # Supports claims
│   ├── investigator_agent.py  # Neutral analysis, entity/temporal checks
│   ├── forensics_agent.py     # Image forensics pipeline
│   ├── judge_agent.py         # Claude → Gemini → local verdict
│   └── debate_orchestrator.py # 2-round debate + argument-graph memory
├── text/
│   ├── awp_scorer.py          # AWP metric + data-driven threshold learning
│   ├── vcade_calibrator.py    # VCADE calibration + isotonic enhancement
│   ├── evidence_pool.py       # Shared credibility-weighted evidence store
│   ├── evidence_retriever.py  # Async Wikipedia/search retrieval
│   ├── claim_extractor.py     # spaCy claim segmentation
│   ├── entailment_matrix.py   # NLI entailment matrix builder
│   ├── adversarial_generator.py # Hypothesis generation for AWP
│   └── policy_engine.py       # PUBLISH/FLAG/REJECT thresholds
├── image/
│   ├── ela_processor.py       # Error Level Analysis
│   ├── fft_processor.py       # FFT spectral analysis
│   ├── florence_extractor.py  # Florence-2: OCR, OD, dense captions
│   ├── deepfake_detector.py   # Face-based deepfake probability
│   ├── watermark_detector.py  # AI-generated watermark detection
│   ├── context_checker.py     # Out-of-context detection (CLIP)
│   ├── claim_image_matcher.py # Per-claim CLIP scoring
│   └── heuristic_fallback.py  # CNN fallback when model unavailable
├── consistency/
│   ├── pipeline.py            # CMCD orchestration
│   ├── visual_fact_extractor.py
│   ├── text_fact_extractor.py
│   ├── contradiction_detector.py
│   └── clip_scorer.py
├── cascade/
│   ├── cascade_executor.py    # Tier 0→3 execution pipeline
│   ├── router.py              # Risk → tier routing
│   └── budget_manager.py      # Per-tier time budgets
├── sentinel/
│   ├── input_validator.py     # Input sanitisation
│   ├── risk_estimator.py      # Risk score for routing
│   ├── query_classifier.py
│   └── confidence_heuristics.py
├── models/
│   ├── model_hub.py           # Lazy singleton model loader
│   ├── nli_model.py           # DeBERTa-v3 NLI
│   ├── clip_model.py          # CLIP ViT-L/14
│   ├── blip_model.py          # BLIP captioning fallback
│   └── sentence_model.py      # all-MiniLM-L6-v2
├── security/
│   └── key_manager.py         # Fernet + PBKDF2 key management
├── audit/
│   ├── logger.py              # JSONL audit log (every request)
│   └── metrics.py             # Prometheus-style metrics
├── api/
│   ├── app.py                 # FastAPI application + CORS
│   ├── routes.py              # All endpoint handlers + rate limiter
│   └── schemas.py             # Pydantic request/response models
├── trust_score.py             # Core dataclasses: TrustScore, ClaimResult, AgentMessage
└── config.py                  # Pydantic Settings (all config in one place)
```

---

## Audit Logging

Every verification run is appended to `audit_logs/audit.jsonl`:

```json
{
  "request_id": "ffd274d3",
  "timestamp": 1773084943.2,
  "tier_used": 2,
  "latency_ms": 414610.72,
  "overall_trust": 0.5,
  "policy": "FLAG",
  "api_calls_made": 0,
  "claims_count": 2,
  "awp_fact_score": 0.5,
  "claims": [
    {
      "claim": "...",
      "verdict": "SUPPORTED",
      "calibrated_trust": 0.82,
      "difficulty_score": 0.29,
      "api_judge_used": false,
      "suspicion_flag": "NONE"
    }
  ]
}
```

Add `"ground_truth"` fields to claim entries to enable data-driven AWP threshold learning and VCADE isotonic calibration.

---

## Configuration Reference

All settings configurable via `.env` or environment variables:

```ini
# Server
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO

# Judge
JUDGE_MODEL=claude-sonnet-4-6
GEMINI_JUDGE_MODEL=models/gemini-2.0-flash
JUDGE_API_DISAGREEMENT_THRESHOLD=0.15
JUDGE_API_TIMEOUT_S=30.0
MONTHLY_API_BUDGET_USD=10.0

# AWP thresholds (data-driven via learn_awp_thresholds)
AWP_REFUTED_THRESHOLD=0.35
AWP_SUPPORTED_THRESHOLD=0.72

# Rate limiting
API_RATE_LIMIT_RPM=30

# Models
NLI_MODEL_ID=cross-encoder/nli-deberta-v3-base
CLIP_MODEL_ID=openai/clip-vit-large-patch14

# Fusion weights
CNN_WEIGHT=0.5
ELA_WEIGHT=0.25
FFT_WEIGHT=0.25

# Policy thresholds
PUBLISH_THRESHOLD=0.7
FLAG_THRESHOLD=0.4
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU | Not required (CPU inference) | NVIDIA GPU 8 GB VRAM (speeds Florence-2 / CLIP) |
| Disk | 6 GB free (model cache) | 10 GB free |
| Python | 3.10+ | 3.10+ |

**Model download sizes (auto-downloaded on first run):**

| Model | Size |
|-------|------|
| Florence-2-base | ~1.5 GB |
| CLIP ViT-L/14 | ~900 MB |
| DeBERTa-v3-base | ~700 MB |
| BLIP captioning | ~500 MB |
| CLIP ViT-B/32 | ~350 MB |
| all-MiniLM-L6-v2 | ~90 MB |
| **Total** | **~4 GB** |

First cold-start takes 5–15 minutes to download all models. Subsequent warm starts load from `model_cache/` in ~10–30 seconds. **All benchmark latency figures below are warm-start numbers.**

---

## Latency

| Tier | Operations | Typical warm-start latency |
|------|-----------|--------------------------|
| Tier 0 | Input validation only | < 5 ms |
| Tier 1 | AWP debate (text, local judge) | 0.5 – 3 s |
| Tier 2 | Tier 1 + image forensics | 3 – 15 s |
| Tier 3 | Tier 2 + OCR claims + CMCD | 8 – 30 s |

> **Note on cold-start:** The audit log contains entries with `latency_ms ~414,000` (~414 seconds). These reflect model-loading time on first run, not inference time. Warm-start latency (models already in memory) is shown in the table above. Production deployments should pre-load all models at server startup.

---

## Web UI (Frontend)

A browser-based frontend is included at `index.html`. Open it directly or serve via the running FastAPI server.

**Features:**
- **Text tab** — paste a claim or paragraph; get per-claim verdicts, trust scores, evidence snippets, suspicion flags, debate rounds
- **Image tab** — upload an image or provide a path; shows ELA energy, FFT score, CNN probability, deepfake probability, OCR text, extracted claims, out-of-context detection
- **Full tab** — multimodal: text + image together; shows CLIP similarity, CMCD cross-modal trust, contradiction details

**To open:**
```bash
# Option 1: open directly in browser (file:// mode, no server needed for UI)
start index.html

# Option 2: served via FastAPI at http://127.0.0.1:8000
python run_server.py
```

---

## API Response Fields — `correction_suggestion`

Each claim result includes a `correction_suggestion` field:

```json
{
  "claim": "The speed of light is 150,000 km/s.",
  "verdict": "REFUTED",
  "correction_suggestion": "Consider: The speed of light is approximately 300,000 km/s."
}
```

**How it is generated:**
1. The `ProsecutorAgent` generates adversarial hypotheses via `AdversarialGenerator` (negation, numeric substitution, entity swap, temporal shift, citation check)
2. AWP scoring ranks hypotheses by entailment strength against retrieved evidence
3. The highest-scoring adversarial hypothesis (`best_alt_hypothesis`) becomes the `correction_suggestion`
4. Format: `"Consider: {best_alt_hypothesis}"` — suggests a factually grounded alternative

This creates a built-in fact-correction loop: when a claim is REFUTED, the system proposes what the correct statement likely is.

---

## Tests

Unit tests cover the three core algorithms and audit logger:

```bash
# Run all tests
pytest tests/ -v

# Run specific module
pytest tests/test_awp.py -v
pytest tests/test_vcade.py -v
pytest tests/test_policy.py -v
pytest tests/test_audit_logger.py -v
```

| Test file | What it covers |
|-----------|---------------|
| `test_awp.py` | AWP score computation, boundary cases, empty input |
| `test_vcade.py` | VCADE difficulty dimensions, suspicion flags, calibrated_trust bounds |
| `test_policy.py` | PUBLISH/FLAG/REJECT thresholds and boundary values |
| `test_audit_logger.py` | `calibrated_trust` in [0,1] (not percentage), suspicion_flag serialized as string, request_id recorded |

---

## Evaluation

### AWP Strategy Ablation (Offline)

Run the ablation study to measure each adversarial strategy's contribution:

```bash
python benchmarks/run_ablation.py --verbose
```

Measures per-strategy impact rate and correctness contribution across 20 labelled claims covering numeric, temporal, entity, citation, and plain fact types.

### FEVER Benchmark

Text fact-verification against the FEVER dataset (Thorne et al., 2018):

```bash
# Offline micro-sample (25 claims, no download needed)
python benchmarks/fever_benchmark.py --sample

# Full FEVER dev set (download from https://fever.ai/dataset/fever.html)
python benchmarks/fever_benchmark.py --fever-path paper_dev.jsonl --limit 500
```

**Published baselines for comparison:**

| System | Label Accuracy | Macro-F1 |
|--------|---------------|----------|
| FEVER TF-IDF baseline | 52.1% | 0.490 |
| MultiFC (Augenstein 2019) | 60.2% | 0.570 |
| FEVEROUS (Aly et al. 2021) | 67.0% | 0.630 |
| GPT-4 zero-shot | ~72.0% | 0.680 |

### Self-Improving Calibration

The audit log feeds back into the system:

1. Add `"ground_truth"` fields to labelled claims in `audit_logs/audit.jsonl`
2. Run `learn_awp_thresholds()` — grid-searches for optimal (refuted_th, supported_th) via macro-F1
3. Run `calibrate_from_logs()` — fits isotonic regression on raw scores → replaces hand-tuned VCADE formula

This creates a **self-improving loop**: the more labelled data accumulated, the better the system calibrates itself over time — a compelling contribution for thesis write-up.

---

## References

- Florence-2 (Microsoft, 2024) — OCR + object detection + dense captioning
- AVerITeC (NeurIPS 2025) — multimodal claim verification benchmark
- Cherti et al. (2023) — "Reproducible Scaling Laws for Contrastive Language-Image Learning"
- Platt, J. (1999) — "Probabilistic Outputs for SVMs" (isotonic calibration principle)
- Shafahi et al. (2019) — adversarial weight perturbation (AWP)
- Thorne et al. (2018) — FEVER: a large-scale dataset for fact extraction and verification
- Aly et al. (2021) — FEVEROUS: fact extraction and verification over unstructured and structured information
