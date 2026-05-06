# CLAUDE.md — HallucinationGuard AMADA v6.0

## Security Fix Status

> Full fix details: `~/.claude/projects/.../memory/fixes.md`
> Last updated: 2026-03-23

| # | Severity | Issue | Status |
|---|----------|-------|--------|
| 1 | CRITICAL | Path traversal in `routes.py` `_validate_image_path()` | ✅ FIXED |
| 2 | HIGH | Silent key decryption failure in `key_manager.py` | ✅ FIXED |
| 3 | HIGH | Weak key derivation (bare SHA256) in `key_manager.py` | ✅ FIXED |
| 4 | HIGH | Missing image path validation in `cascade_executor.py` | ✅ FIXED |
| 5 | HIGH | No JSON validation on API judge response in `judge_agent.py` | ✅ FIXED |
| 6 | HIGH | Missing timeout on Anthropic API calls in `judge_agent.py` | ✅ FIXED |
| 7 | HIGH | Blocking HTTP in async context in `evidence_retriever.py` | ✅ FIXED |
| 8 | HIGH | Bare `except: pass` swallowing errors across modules | ✅ FIXED |
| 9 | MEDIUM | Overly permissive CORS in `app.py` | ✅ FIXED |
| 10 | MEDIUM | No rate limiting in `routes.py` | ✅ FIXED |
| 11 | MEDIUM | Hardcoded model name in `judge_agent.py` | ✅ FIXED |
| 12 | MEDIUM | Missing exception handling in `forensics_agent.py` | ✅ FIXED |
| 13 | MEDIUM | Unbounded input text in `claim_extractor.py` | ✅ FIXED |
| 14 | MEDIUM | Float overflow risk in `numerical_verifier.py` | ✅ FIXED |
| 15 | MEDIUM | Unsafe cache assignment on model load failure in `model_hub.py` | ✅ FIXED |
| 16 | MEDIUM | Unpinned dependencies in `requirements.txt` | ✅ FIXED |
| 17 | LOW | No model file integrity check in `model_hub.py` | ✅ FIXED |
| 18 | LOW | Redundant HTTP connections in `evidence_retriever.py` | ✅ FIXED |
| 19 | LOW | Image path exposed via query param in `routes.py` | ✅ FIXED |
| 20 | LOW | Raw exception message leaked to client in `routes.py` | ✅ FIXED |
| 21 | LOW | Race condition on concurrent evidence fetch in `evidence_pool.py` | ✅ FIXED |
| 22 | LOW | Race condition on Florence model load in `florence_extractor.py` | ✅ FIXED |
| 23 | LOW | `.env` not in `.gitignore` | ✅ FIXED |
| 24 | LOW | Missing module-level loggers in 8 modules | ✅ FIXED |
| 25 | LOW | Request ID not propagated to audit log in `routes.py` | ✅ FIXED |
| 26 | LOW | TF 2.18.0 CVE unverified in `requirements.txt` | ✅ FIXED |

**Total: 26/26 FIXED** (1 Critical, 7 High, 8 Medium, 10 Low)

---

## Architecture Summary

- **Pattern**: AMADA — Adversarial Multi-Agent Debate Architecture
- **Agents**: Prosecutor, Defender, Investigator, Forensics, Judge (5 total)
- **Algorithms**: AWP (Adversarial Weight Perturbation), CMCD, VCADE
- **API Judge**: Claude → Gemini fallback → local fallback
- **Convergence threshold**: gap < 0.15 → local; gap ≥ 0.15 → API

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env        # fill in encrypted API keys
python -m hallucination_guard.api.app   # → http://127.0.0.1:8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/verify/text` | Text-only verification |
| POST | `/api/v1/verify/image` | Image verification |
| POST | `/api/v1/verify/full` | Multimodal (text + image) |
| GET | `/metrics` | Prometheus metrics |
| GET | `/health` | Health check |
