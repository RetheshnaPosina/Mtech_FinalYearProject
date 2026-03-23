"""Pydantic Settings for AMADA v6.0 — all config in one place.

Novelty improvements in this version
--------------------------------------
1. CLIP model upgraded from ViT-B/32 → ViT-L/14:
   clip_model_id = "openai/clip-vit-large-patch14"
   ViT-B/32 (2021) has well-documented limitations with fine-grained text
   understanding — it treats "a man biting a dog" and "a dog biting a man"
   as nearly identical embeddings.  ViT-L/14 significantly improves
   relationship and composition sensitivity (Cherti et al., 2023,
   "Reproducible Scaling Laws for CLIP").  Configurable via .env so
   resource-constrained deployments can fall back to the base model.

2. Learnable AWP thresholds:
   awp_refuted_threshold   = 0.35  (was hard-coded in prosecutor_agent)
   awp_supported_threshold = 0.72  (was hard-coded in prosecutor_agent)
   Externalising these values lets learn_awp_thresholds() (awp_scorer.py)
   write back optimised values without touching agent source code.
   Set via .env or pass --awp-refuted-threshold CLI arg to run_ablation.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Paths
    audit_dir: Path = Path("./audit_logs")
    model_cache_dir: Path = Path("./model_cache")

    # API keys (encrypted via Fernet in security/key_manager.py)
    anthropic_api_key_enc: str = ""
    google_search_api_key_enc: str = ""
    google_search_engine_id: str = ""
    anthropic_api_key: str = ""
    google_search_api_key: str = ""
    gemini_api_key_enc: str = ""
    gemini_api_key: str = ""
    gemini_judge_model: str = "models/gemini-2.0-flash"

    # Judge settings
    judge_api_disagreement_threshold: float = 0.15
    monthly_api_budget_usd: float = 10.0
    judge_model: str = "claude-sonnet-4-6"
    judge_api_timeout_s: float = 30.0

    # Rate limiting
    api_rate_limit_rpm: int = 30

    # Claim processing
    max_claims_per_request: int = 20
    evidence_top_k: int = 5
    adversarial_top_k: int = 3
    nli_batch_size: int = 8
    debate_max_rounds: int = 3

    # AWP thresholds (learnable)
    awp_refuted_threshold: float = 0.35
    awp_supported_threshold: float = 0.72

    # ELA settings
    ela_quality: int = 90
    ela_resize_w: int = 512
    ela_resize_h: int = 512

    # FFT settings
    fft_sigma_threshold: float = 6.0

    # Fusion weights
    cnn_weight: float = 0.5
    ela_weight: float = 0.25
    fft_weight: float = 0.25

    # CLIP / CMCD
    clip_contradiction_threshold: float = 0.5
    contradiction_penalty_weight: float = 0.4

    # Policy thresholds
    publish_threshold: float = 0.7
    flag_threshold: float = 0.4

    # Tier budgets (seconds)
    tier0_budget_s: float = 0.01
    tier1_budget_s: float = 0.2
    tier2_budget_s: float = 3.0
    tier3_budget_s: float = 8.0

    # VCADE weights
    vcade_w_retrieval: float = 0.3
    vcade_w_consensus: float = 0.3
    vcade_w_adversarial: float = 0.3
    vcade_w_entity: float = 0.1
    vcade_w_temporal: float = 0.1

    # Model IDs
    nli_model_id: str = "cross-encoder/nli-deberta-v3-base"
    blip_model_id: str = "Salesforce/blip-image-captioning-base"
    clip_model_id: str = "openai/clip-vit-large-patch14"
    sentence_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    cnn_model_path: str = "./model_cache/my_model.h5"


settings = Settings()
