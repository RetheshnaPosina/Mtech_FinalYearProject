"""AMADA FastAPI application entry point."""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from hallucination_guard.api.routes import router
from hallucination_guard.config import settings

app = FastAPI(
    title="HallucinationGuard AMADA v6.0",
    description=(
        "Adversarial Multi-Agent Debate Architecture for AI-generated misinformation detection. "
        "Three novel algorithms: AWP, CMCD, VCADE."
    ),
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(router)

# Serve static UI if present
_index = Path(__file__).parent.parent.parent / "index.html"


@app.get("/")
def serve_ui() -> FileResponse:
    if _index.exists():
        return FileResponse(str(_index))
    from fastapi.responses import JSONResponse
    return JSONResponse({"message": "AMADA v6.0 API running. See /docs for endpoints."})


def main() -> None:
    settings.audit_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    uvicorn.run(
        "hallucination_guard.api.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
