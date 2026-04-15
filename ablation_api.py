"""
Ablation study API — Context-Adaptive Evidence Weighting Framework.

Tests the impact of bias-aware weighting by comparing two conditions:

  FULL SYSTEM  — weight = credibility × NLI_conf × (1 − bias_penalty) × authority × recency
  BASELINE     — weight = credibility × NLI_conf × authority × recency  (no bias penalty)

Imports _pipeline_steps from api.py so the pipeline logic is NOT duplicated.
Models are loaded into a separate _models dict so this server is fully standalone.

Run on a different port to the main API:
    uvicorn ablation_api:app --port 8001 --reload

Endpoints:
    GET  /health
    POST /verify/baseline  — baseline condition (disable_bias=True always)
    POST /verify/compare   — runs both conditions, returns side-by-side comparison + delta
"""

import asyncio
import sys

sys.path.append("src")

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import Config

# _pipeline_steps takes `m` (models dict) as a parameter — not a global — so we
# can import the function and drive it with our own _models dict without touching api.py.
from api import _pipeline_steps

# ---------------------------------------------------------------------------
# Global model registry — populated once during startup, separate from api.py
# ---------------------------------------------------------------------------
_models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy models once at startup (mirrors the main api.py lifespan)."""
    print("=" * 60)
    print("ABLATION API — LOADING MODELS")
    print("=" * 60)

    from src.evidence_retrieval.google_ai_mode_retrieval import GoogleAIModeRetriever
    from src.evidence_retrieval.tavily_evidence_retrieval import TavilyEvidenceRetriever
    from src.retrieval.qdrant_hybrid_retriever import QdrantHybridRetriever as HybridRetriever
    from src.verification import ClaimVerifier
    from src.verification.gemini_nli_verification import GeminiNLIVerifier
    from src.bias_detection import BiasDetector
    from src.bias_detection.claim_bias_analyzer import ClaimBiasAnalyzer
    from src.evidence_weighting import EvidenceWeighter, VerdictGenerator

    _models["hybrid_retriever"] = HybridRetriever()
    print("  HybridRetriever ready")

    if Config.SERP_API_KEY:
        _models["web_retriever"] = GoogleAIModeRetriever(
            serp_api_key=Config.SERP_API_KEY,
            groq_key=Config.GROQ_API_KEY,
            tavily_key=Config.TAVILY_API_KEY,
        )
        print("  Web retriever: Google AI Mode")
    else:
        _models["web_retriever"] = TavilyEvidenceRetriever(
            tavily_key=Config.TAVILY_API_KEY,
            groq_key=Config.GROQ_API_KEY,
        )
        print("  Web retriever: Tavily (fallback)")

    _models["nli_verifier"] = GeminiNLIVerifier()
    print("  GeminiNLIVerifier ready")

    _models["verifier"] = ClaimVerifier()
    print("  ClaimVerifier (DeBERTa) ready")

    _models["bias_detector"] = BiasDetector()
    print("  BiasDetector ready")

    _models["claim_bias_analyzer"] = ClaimBiasAnalyzer(gemini_api_key=Config.GEMINI_API_KEY)
    print("  ClaimAwareBiasAnalyzer ready")

    _models["weighter"] = EvidenceWeighter()
    _models["verdict_generator"] = VerdictGenerator()
    print("  EvidenceWeighter + VerdictGenerator ready")

    try:
        from sentence_transformers import CrossEncoder
        _models["relevance_filter"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("  CrossEncoder relevance filter ready")
    except Exception as e:
        _models["relevance_filter"] = None
        print(f"  CrossEncoder unavailable — relevance filtering disabled: {e}")

    # Gemini explainer not needed for ablation — mark as unavailable so
    # _pipeline_steps skips the optional explanation step cleanly.
    _models["gemini_available"] = False
    _models["gemini_explainer"] = None

    print("=" * 60)
    print("ALL MODELS LOADED — ablation server ready")
    print("=" * 60)

    yield

    _models.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Ablation Study API — Context-Adaptive Evidence Weighting Framework",
    description=(
        "Baseline (no bias-aware weighting) and side-by-side comparison endpoints "
        "for evaluating the impact of the bias penalty on verdict and confidence."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AblationRequest(BaseModel):
    claim: str


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------
async def _run_pipeline(claim: str, disable_bias: bool) -> dict:
    """Drive _pipeline_steps to completion and return the result dict."""
    result_data = None
    async for event in _pipeline_steps(claim, _models, disable_bias=disable_bias):
        if event["type"] == "error":
            raise HTTPException(status_code=422, detail=event["message"])
        if event["type"] == "result":
            result_data = event["data"]
            # discard _cache — not needed for ablation
    if result_data is None:
        raise HTTPException(status_code=500, detail="Pipeline produced no result")
    return result_data


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(_models)}


@app.post("/verify/baseline")
async def verify_baseline(request: AblationRequest):
    """
    Full pipeline with bias-aware weighting DISABLED.

    The weight formula becomes:
        weight = credibility × NLI_confidence × authority × recency

    The bias_penalty term is zeroed out regardless of the source's political alignment
    with the claim. Use this as the comparison baseline in the ablation study.
    """
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="claim must not be empty")
    return await _run_pipeline(request.claim.strip(), disable_bias=True)


@app.post("/verify/compare")
async def verify_compare(request: AblationRequest):
    """
    Run the full pipeline under BOTH conditions and return a side-by-side comparison.

    The response contains:
      full_system  — result with bias-aware weighting enabled
      baseline     — result with bias weighting disabled
      delta        — verdict change flag, confidence shift, and the sources most affected
                     by the bias penalty (sorted by |weight_delta|)

    NOTE: this runs the full pipeline twice — retrieval and NLI are NOT shared between
    runs. Intended for offline evaluation on a curated set of test claims, not live use.
    """
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="claim must not be empty")

    claim = request.claim.strip()

    full_result = await _run_pipeline(claim, disable_bias=False)

    # Brief pause between the two runs so both don't hammer Gemini simultaneously.
    # Reduces the chance of one run hitting a transient error while the other succeeds.
    await asyncio.sleep(10)

    baseline_result = await _run_pipeline(claim, disable_bias=True)

    # --- delta: verdict and scores ---
    verdict_changed    = full_result["verdict"] != baseline_result["verdict"]
    confidence_delta   = round(full_result["confidence"]    - baseline_result["confidence"],    4)
    support_delta      = round(full_result["support_score"] - baseline_result["support_score"], 4)
    refute_delta       = round(full_result["refute_score"]  - baseline_result["refute_score"],  4)

    # --- delta: per-source weight shift ---
    # Match sources across the two runs by URL (link field).
    # Sources that only appear in one run (different retrieval results) are skipped.
    full_by_link = {
        s["link"]: s
        for s in full_result["sources"]
        if s.get("link") and not s.get("excluded")
    }
    most_affected = []
    for src in baseline_result["sources"]:
        if src.get("excluded") or not src.get("link"):
            continue
        link = src["link"]
        if link not in full_by_link:
            continue
        full_src  = full_by_link[link]
        fw = full_src.get("weight") or 0.0
        bw = src.get("weight")      or 0.0
        delta = round(fw - bw, 4)
        if abs(delta) < 0.001:
            continue  # unchanged — skip
        most_affected.append({
            "source":           src.get("source", ""),
            "title":            src.get("title", ""),
            "link":             link,
            "full_weight":      fw,
            "baseline_weight":  bw,
            "weight_delta":     delta,          # positive = full system weighted it LOWER
            "bias_alignment":   full_src.get("bias_alignment", 0.0),
            "verdict":          src.get("verdict", ""),
        })
    most_affected.sort(key=lambda x: abs(x["weight_delta"]), reverse=True)

    return {
        "claim":       claim,
        "full_system": full_result,
        "baseline":    baseline_result,
        "delta": {
            "verdict_changed":     verdict_changed,
            "full_verdict":        full_result["verdict"],
            "baseline_verdict":    baseline_result["verdict"],
            "confidence_delta":    confidence_delta,
            "full_confidence":     full_result["confidence"],
            "baseline_confidence": baseline_result["confidence"],
            "support_score_delta": support_delta,
            "refute_score_delta":  refute_delta,
            # Top 5 sources whose weight shifted most due to the bias penalty
            "most_affected_sources": most_affected[:5],
        },
    }
