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
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    audit_dir: Path = Field(default=Path("./audit_logs"))
    model_cache_dir: Path = Field(default=Path("./model_cache"))

    # ------------------------------------------------------------------
    # API keys (Fernet-encrypted; decrypt via key_manager.py)
    # ------------------------------------------------------------------
    anthropic_api_key_enc: str = Field(default="")
    google_search_api_key_enc: str = Field(default="")
    google_search_engine_id: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    google_search_api_key: str = Field(default="")
    gemini_api_key_enc: str = Field(default="")
    gemini_api_key: str = Field(default="")

    # ------------------------------------------------------------------
    # Judge / API settings
    # ------------------------------------------------------------------
    gemini_judge_model: str = Field(default="models/gemini-2.0-flash")
    judge_api_disagreement_threshold: float = Field(default=0.15)
    monthly_api_budget_usd: float = Field(default=10.0)
    judge_model: str = Field(default="claude-sonnet-4-6")
    judge_api_timeout_s: float = Field(default=30.0)
    api_rate_limit_rpm: int = Field(default=30)

    # ------------------------------------------------------------------
    # Debate / claim settings
    # ------------------------------------------------------------------
    max_claims_per_request: int = Field(default=20)
    evidence_top_k: int = Field(default=5)
    adversarial_top_k: int = Field(default=3)
    nli_batch_size: int = Field(default=8)
    debate_max_rounds: int = Field(default=2)

    # ------------------------------------------------------------------
    # AWP thresholds  (novelty: externalised so learn_awp_thresholds()
    # can write back data-driven values; see awp_scorer.py)
    # ------------------------------------------------------------------
    awp_refuted_threshold: float = Field(default=0.35)
    awp_supported_threshold: float = Field(default=0.72)

    # ------------------------------------------------------------------
    # Image forensics
    # ------------------------------------------------------------------
    ela_quality: int = Field(default=90)
    ela_resize_w: int = Field(default=512)
    ela_resize_h: int = Field(default=512)
    fft_sigma_threshold: float = Field(default=6.0)
    cnn_weight: float = Field(default=0.5)
    ela_weight: float = Field(default=0.25)
    fft_weight: float = Field(default=0.25)

    # ------------------------------------------------------------------
    # Cross-modal consistency (CMCD)
    # ------------------------------------------------------------------
    clip_contradiction_threshold: float = Field(default=0.5)
    contradiction_penalty_weight: float = Field(default=0.4)

    # ------------------------------------------------------------------
    # Policy engine
    # ------------------------------------------------------------------
    publish_threshold: float = Field(default=0.7)
    flag_threshold: float = Field(default=0.4)

    # ------------------------------------------------------------------
    # Cascade tier budgets (seconds)
    # ------------------------------------------------------------------
    tier0_budget_s: float = Field(default=0.01)
    tier1_budget_s: float = Field(default=0.2)
    tier2_budget_s: float = Field(default=3.0)
    tier3_budget_s: float = Field(default=8.0)

    # ------------------------------------------------------------------
    # VCADE weights  (d1..d5; sum ≈ 1.0)
    # d1 retrieval=0.25, d2 consensus=0.25, d3 adversarial=0.30,
    # d4 entity=0.10, d5 temporal=0.10
    # ------------------------------------------------------------------
    vcade_w_retrieval: float = Field(default=0.25)
    vcade_w_consensus: float = Field(default=0.25)
    vcade_w_adversarial: float = Field(default=0.30)
    vcade_w_entity: float = Field(default=0.10)
    vcade_w_temporal: float = Field(default=0.10)

    # ------------------------------------------------------------------
    # Model IDs
    # ------------------------------------------------------------------
    nli_model_id: str = Field(default="cross-encoder/nli-deberta-v3-base")
    blip_model_id: str = Field(default="Salesforce/blip-image-captioning-base")

    # Novelty: upgraded from ViT-B/32 to ViT-L/14 for better relationship
    # and composition sensitivity (Cherti et al., 2023).
    # Override in .env: CLIP_MODEL_ID=openai/clip-vit-base-patch32
    clip_model_id: str = Field(default="openai/clip-vit-large-patch14")

    sentence_model_id: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    cnn_model_path: str = Field(default="./model_cache/my_model.h5")


settings = Settings()
