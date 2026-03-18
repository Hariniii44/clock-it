"""
FastAPI wrapper for the Context-Adaptive Evidence Weighting Framework.

Models are loaded ONCE at startup and reused for every request.
core_pipeline.py is not modified — all logic is imported from it.

Run with:
    uvicorn api:app --reload
"""

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

sys.path.append('src')

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import Config
from verification.gemini_verification import GeminiClaimVerifier

from core_pipeline import (
    detect_temporal_claim,
    analyze_evidence_temporal_distribution,
    analyze_content_freshness,
    calculate_temporal_confidence,
    detect_temporal_mismatch,
    assess_evidence_quality,
    _compute_divergence,
    _divergence_summary,
)


# ---------------------------------------------------------------------------
# Global model registry — populated once during startup
# ---------------------------------------------------------------------------
_models: dict[str, Any] = {}

# In-memory result cache keyed by claim string.
# Enables /explain to call Gemini without re-running the pipeline.
_result_cache: dict[str, dict] = {}
_MAX_CACHE = 10


def _cache_store(claim: str, data: dict) -> None:
    if len(_result_cache) >= _MAX_CACHE:
        oldest = next(iter(_result_cache))
        del _result_cache[oldest]
    _result_cache[claim] = data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy models once when the server starts."""
    print("=" * 60)
    print("LOADING MODELS — this happens only once")
    print("=" * 60)

    from src.evidence_retrieval.tavily_evidence_retrieval import TavilyEvidenceRetriever
    from src.evidence_retrieval.google_ai_mode_retrieval import GoogleAIModeRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.verification import ClaimVerifier
    from src.verification.groq_verification import GroqVerifier
    from src.bias_detection import BiasDetector
    from src.bias_detection.claim_aware_bias_analyzer import ClaimAwareBiasAnalyzer
    from src.evidence_weighting import EvidenceWeighter, VerdictGenerator

    _models["hybrid_retriever"] = HybridRetriever()
    print("  HybridRetriever ready")

    WEB_RETRIEVER = 'google_ai_mode'
    if WEB_RETRIEVER == 'google_ai_mode' and Config.SERP_API_KEY:
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
        print("  Web retriever: Tavily")

    _models["groq_verifier"] = GroqVerifier(groq_api_key=Config.GROQ_API_KEY)
    _models["verifier"] = ClaimVerifier()
    print("  ClaimVerifier (DeBERTa) ready")

    _models["bias_detector"] = BiasDetector()
    print("  BiasDetector ready")

    _models["claim_bias_analyzer"] = ClaimAwareBiasAnalyzer(groq_api_key=Config.GROQ_API_KEY)

    _models["weighter"] = EvidenceWeighter()
    _models["verdict_generator"] = VerdictGenerator()
    print("  EvidenceWeighter + VerdictGenerator ready")

    try:
        _models["gemini_explainer"] = GeminiClaimVerifier()
        _models["gemini_available"] = True
        print("  Gemini explainer ready")
    except Exception as e:
        _models["gemini_explainer"] = None
        _models["gemini_available"] = False
        print(f"  Gemini unavailable: {e}")

    print("=" * 60)
    print("ALL MODELS LOADED — server is ready")
    print("=" * 60)

    yield  # server runs here

    _models.clear()
    _result_cache.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Context-Adaptive Evidence Weighting Framework",
    description="Bias-aware fact-checking API for Sri Lankan political claims",
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
# Request / Response schemas
# ---------------------------------------------------------------------------
class VerifyRequest(BaseModel):
    claim: str
    skip_gemini: bool = True  # Gemini skipped by default; use /explain for on-demand


class ExplainRequest(BaseModel):
    claim: str


class VerifyResponse(BaseModel):
    claim: str
    verdict: str
    confidence: float
    support_score: float
    refute_score: float
    neutral_score: float
    total_sources: int
    db_sources: int
    web_sources: int
    temporal_status: str
    is_breaking_news: bool
    temporal_warnings: list[str]
    uncertainty: dict
    evidence_quality: dict
    claim_bias: dict
    divergence_level: str
    sources: list[dict]
    gemini_explanation: str | None


# ---------------------------------------------------------------------------
# Shared pipeline — async generator
#
# Both /verify and /verify/stream drive the same logic here.
# Yields dicts:
#   {"type": "status", "message": "..."}               — progress update
#   {"type": "error",  "message": "..."}               — fatal, stop
#   {"type": "result", "data": {...}, "_cache": {...}}  — final result
# ---------------------------------------------------------------------------
async def _pipeline_steps(claim: str, m: dict):
    loop = asyncio.get_running_loop()

    # ------------------------------------------------------------------
    # STEP 0: Temporal claim analysis (fast, no I/O)
    # ------------------------------------------------------------------
    yield {"type": "status", "message": "Analyzing temporal context..."}
    claim_temporal = detect_temporal_claim(claim)

    # ------------------------------------------------------------------
    # STEP 1: Dual evidence retrieval — DB and web run in parallel
    # ------------------------------------------------------------------
    yield {"type": "status", "message": "Retrieving evidence from database and web..."}

    def _retrieval():
        def _db():
            try:
                results = m["hybrid_retriever"].hybrid_search(
                    query=claim, claim_types=None, total_results=10, use_query_expansion=True
                )
                return [
                    {
                        "source": r["source"], "title": r["title"],
                        "snippet": r.get("passage", r["text"]), "link": r["url"],
                        "dataset_source": r["dataset"], "authority": r["authority"],
                        "relevance_score": r["similarity_score"], "evidence_type": "database",
                    }
                    for r in results
                ]
            except Exception as e:
                print(f"  Database retrieval failed: {e}")
                return []

        def _web():
            try:
                results = m["web_retriever"].retrieve_hybrid_serper_decomposition(
                    claim, num_results=15, results_per_query=8
                )
                for ev in results:
                    ev["evidence_type"] = "web"
                return results
            except Exception as e:
                print(f"  Web retrieval failed: {e}")
                return []

        with ThreadPoolExecutor(max_workers=2) as pool:
            db_f = pool.submit(_db)
            web_f = pool.submit(_web)
            return db_f.result(), web_f.result()

    db_formatted, web_evidence = await loop.run_in_executor(None, _retrieval)
    all_evidence = db_formatted + web_evidence

    if not all_evidence:
        yield {"type": "error", "message": "No evidence found for this claim."}
        return

    # ------------------------------------------------------------------
    # Temporal evidence analysis + stale-source filter (fast)
    # ------------------------------------------------------------------
    evidence_temporal = analyze_evidence_temporal_distribution(all_evidence)
    content_freshness = analyze_content_freshness(all_evidence, claim)

    _now = datetime.now()
    _current_ym = f"/{_now.year}/{_now.month:02d}/"
    _url_recent = sum(1 for ev in all_evidence if _current_ym in ev.get("link", ""))
    _has_url_recency = _url_recent >= 1
    _has_structured_recency = (
        evidence_temporal.get("recent_evidence_ratio", 0) > 0.6
        and evidence_temporal.get("avg_evidence_age_hours") is not None
        and evidence_temporal["avg_evidence_age_hours"] < 48
    )
    if not claim_temporal["is_temporal"] and (_has_structured_recency or _has_url_recency):
        claim_temporal = {
            **claim_temporal, "is_temporal": True, "recency_level": "recent",
            "analysis_note": "Recency inferred from evidence dates; claim text had no temporal markers",
        }

    temporal_confidence = calculate_temporal_confidence(claim_temporal, evidence_temporal, len(all_evidence))
    if "analysis_note" in claim_temporal and not temporal_confidence["temporal_warnings"]:
        temporal_confidence["temporal_warnings"].append(
            "Recency inferred from evidence — this may be a developing story; verdict may change as more information emerges"
        )
        temporal_confidence["is_breaking_news"] = True
        temporal_confidence["temporal_confidence_multiplier"] = 0.85

    temporal_mismatch = detect_temporal_mismatch(all_evidence)
    evidence_quality = assess_evidence_quality(all_evidence, claim_temporal)

    # Only remove stale sources when the claim itself had explicit temporal markers.
    # If recency was inferred from evidence (analysis_note present), the claim is
    # likely historical — removing older sources would delete the most relevant evidence.
    _claim_had_explicit_temporal = claim_temporal.get("is_temporal") and "analysis_note" not in claim_temporal
    if _claim_had_explicit_temporal and temporal_mismatch["has_mismatch"] and temporal_mismatch.get("recent_source_count", 0) >= 3:
        stale_indices = set(temporal_mismatch["stale_source_indices"])
        all_evidence = [e for i, e in enumerate(all_evidence) if i not in stale_indices]

    db_sources_count = sum(1 for e in all_evidence if e.get("evidence_type") == "database")
    web_sources_count = sum(1 for e in all_evidence if e.get("evidence_type") == "web")
    total_sources = len(all_evidence)

    # ------------------------------------------------------------------
    # STEP 2+3: NLI, per-source bias, claim-level bias — all concurrent
    # ------------------------------------------------------------------
    yield {"type": "status", "message": f"Found {total_sources} sources. Running verification and bias analysis..."}

    def _analysis():
        def _nli():
            return m["groq_verifier"].verify_batch(claim, all_evidence)

        def _bias_det():
            results = []
            for ev in all_evidence:
                try:
                    results.append(
                        m["bias_detector"].analyze_source(ev.get("snippet", ""), ev.get("link", ""))
                    )
                except Exception:
                    results.append({"overall_bias": 0.0, "confidence": 0.0})
            return results

        def _claim_bias():
            return m["claim_bias_analyzer"].analyze(claim, all_evidence)

        with ThreadPoolExecutor(max_workers=3) as pool:
            nli_f       = pool.submit(_nli)
            bias_det_f  = pool.submit(_bias_det)
            claim_bias_f = pool.submit(_claim_bias)
            return nli_f.result(), bias_det_f.result(), claim_bias_f.result()

    groq_batch, bias_analyses, bias_result = await loop.run_in_executor(None, _analysis)

    use_groq = len(groq_batch) == len(all_evidence)
    verification_results = []
    for i, ev in enumerate(all_evidence):
        try:
            vr = groq_batch[i] if use_groq else m["verifier"].verify_claim(claim, ev.get("snippet", ""))
            if not isinstance(vr, dict) or "label" not in vr or "confidence" not in vr:
                verification_results.append({"label": "error", "confidence": 0.0})
            else:
                verification_results.append(vr)
        except Exception as e:
            print(f"  NLI source {i} error: {e}")
            verification_results.append({"label": "error", "confidence": 0.0})

    # ------------------------------------------------------------------
    # STEP 4: Bias-aware weighting + verdict (fast, no I/O)
    # ------------------------------------------------------------------
    yield {"type": "status", "message": "Generating verdict..."}

    claim_bias = bias_result["claim_bias"]
    framing_result = {
        "sources": [
            {
                "index": s["index"],
                "source": (all_evidence[s["index"]].get("source", "") if s["index"] < len(all_evidence) else ""),
                "framing_direction": s["framing_type"],
                "consistency_with_profile": "unknown",
                "loaded_phrases": s["loaded_phrases"],
                "explanation": s["explanation"],
                "emotional_tone": s["emotional_tone"],
                "political_direction": s["political_direction"],
                "alignment": s["alignment"],
            }
            for s in bias_result["sources"]
        ],
        "divergence_level": _compute_divergence(bias_result["sources"]),
        "cross_source_divergence": _divergence_summary(bias_result["sources"], claim_bias),
        "contested_entities": [],
    }
    precomputed_alignments = [s["alignment"] for s in bias_result["sources"]]
    weighted_evidence = m["weighter"].weight_all_evidence(
        claim, all_evidence, verification_results, bias_analyses,
        framing_analyses=framing_result, precomputed_alignments=precomputed_alignments,
    )
    base_verdict = m["verdict_generator"].generate_verdict(weighted_evidence)
    temporal_adjusted_confidence = (
        base_verdict["confidence"] * temporal_confidence["temporal_confidence_multiplier"]
    )
    final_verdict = {
        **base_verdict,
        "confidence": temporal_adjusted_confidence,
        "base_confidence": base_verdict["confidence"],
        "temporal_adjustment": temporal_confidence["temporal_confidence_multiplier"],
        "temporal_status": claim_temporal["recency_level"],
        "temporal_warnings": temporal_confidence["temporal_warnings"],
    }
    verdict_label = final_verdict["verdict"]
    if temporal_confidence["is_breaking_news"]:
        verdict_label += " (PRELIMINARY)"

    sources_out = []
    for item in weighted_evidence:
        ev = item["evidence"]
        vr = item["verification"]
        sources_out.append({
            "title": ev.get("title", ""), "link": ev.get("link", ""),
            "source": ev.get("source", ""), "evidence_type": ev.get("evidence_type", ""),
            "verdict": vr.get("label", ""), "confidence": vr.get("confidence", 0.0),
            "weight": item.get("weight", 0.0), "bias_alignment": item.get("bias_alignment", 0.0),
        })

    uncertainty = final_verdict.get("uncertainty_decomposition", {})

    result_data = {
        "claim": claim,
        "verdict": verdict_label,
        "confidence": final_verdict["confidence"],
        "support_score": final_verdict.get("support_score", 0.0),
        "refute_score": final_verdict.get("refute_score", 0.0),
        "neutral_score": final_verdict.get("neutral_score", 0.0),
        "total_sources": total_sources,
        "db_sources": db_sources_count,
        "web_sources": web_sources_count,
        "temporal_status": claim_temporal["recency_level"],
        "is_breaking_news": temporal_confidence["is_breaking_news"],
        "temporal_warnings": temporal_confidence["temporal_warnings"],
        "uncertainty": {
            "epistemic": uncertainty.get("epistemic", 0.0),
            "aleatoric": uncertainty.get("aleatoric", 0.0),
            "bias_induced": uncertainty.get("bias_induced", 0.0),
        },
        "evidence_quality": {
            "level": evidence_quality["quality_level"],
            "note": evidence_quality.get("quality_note"),
            "established_count": evidence_quality["established_count"],
            "unverified_social_count": evidence_quality["unverified_social_count"],
        },
        "claim_bias": {
            "framing_type": claim_bias.get("framing_type", ""),
            "emotional_tone": claim_bias.get("emotional_tone", 0.0),
            "political_direction": claim_bias.get("political_direction", ""),
            "explanation": claim_bias.get("explanation", ""),
        },
        "divergence_level": framing_result["divergence_level"],
        "sources": sources_out,
        "gemini_explanation": None,
    }

    cache_data = {
        "weighted_evidence": weighted_evidence,
        "final_verdict": final_verdict,
        "temporal_context": {
            "is_breaking_news": temporal_confidence.get("is_breaking_news", False),
            "recency_level": claim_temporal["recency_level"],
            "estimated_hours_old": claim_temporal["estimated_hours_old"],
            "content_freshness_score": content_freshness["freshness_score"],
            "breaking_indicators": content_freshness.get("breaking_indicators", []),
            "temporal_warnings": temporal_confidence["temporal_warnings"],
        },
    }

    yield {"type": "result", "data": result_data, "_cache": cache_data}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(_models)}


@app.post("/verify", response_model=VerifyResponse)
async def verify_claim(request: VerifyRequest):
    """Blocking endpoint — waits for the full result before responding."""
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="claim must not be empty")

    claim = request.claim.strip()
    result_data = None

    async for event in _pipeline_steps(claim, _models):
        if event["type"] == "error":
            raise HTTPException(status_code=422, detail=event["message"])
        if event["type"] == "result":
            _cache_store(claim, event["_cache"])
            result_data = event["data"]

    if result_data is None:
        raise HTTPException(status_code=500, detail="Pipeline produced no result")

    # Optional Gemini explanation (off by default; use /explain for on-demand)
    if not request.skip_gemini and _models.get("gemini_available"):
        cached = _result_cache.get(claim)
        if cached:
            try:
                result_data["gemini_explanation"] = _models["gemini_explainer"].explain_weighting_decisions(
                    claim, cached["weighted_evidence"], cached["final_verdict"], cached["temporal_context"]
                )
            except Exception as e:
                print(f"  Gemini explanation failed: {e}")

    return VerifyResponse(**result_data)


@app.post("/verify/stream")
async def verify_claim_stream(request: VerifyRequest):
    """
    Streaming endpoint — emits Server-Sent Events as each pipeline step completes.
    The frontend receives status messages immediately and the result when ready.
    """
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="claim must not be empty")

    claim = request.claim.strip()

    async def generate():
        try:
            async for event in _pipeline_steps(claim, _models):
                cache = event.pop("_cache", None)
                if cache:
                    _cache_store(claim, cache)
                yield f"data: {json.dumps(event)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/explain")
def get_explanation(request: ExplainRequest):
    """Blocking Gemini explanation (kept for compatibility)."""
    claim = request.claim.strip()
    cached = _result_cache.get(claim)
    if not cached:
        raise HTTPException(status_code=404, detail="No cached result found. Please verify the claim first.")
    if not _models.get("gemini_available"):
        raise HTTPException(status_code=503, detail="Gemini is not available.")
    try:
        explanation = _models["gemini_explainer"].explain_weighting_decisions(
            claim,
            cached["weighted_evidence"],
            cached["final_verdict"],
            cached["temporal_context"],
        )
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini failed: {e}")


@app.post("/explain/stream")
async def stream_explanation(request: ExplainRequest):
    """
    Stream Gemini explanation token-by-token as Server-Sent Events.
    The frontend receives text chunks immediately as Gemini generates them.
    """
    claim = request.claim.strip()
    cached = _result_cache.get(claim)
    if not cached:
        raise HTTPException(status_code=404, detail="No cached result found. Please verify the claim first.")
    if not _models.get("gemini_available"):
        raise HTTPException(status_code=503, detail="Gemini is not available.")

    import queue as q
    import threading

    async def generate():
        chunk_queue: q.Queue = q.Queue()
        loop = asyncio.get_running_loop()

        def _produce():
            try:
                for chunk in _models["gemini_explainer"].stream_explanation(
                    claim,
                    cached["weighted_evidence"],
                    cached["final_verdict"],
                    cached["temporal_context"],
                ):
                    chunk_queue.put({"type": "chunk", "text": chunk})
                chunk_queue.put({"type": "done"})
            except Exception as e:
                chunk_queue.put({"type": "error", "message": str(e)})

        threading.Thread(target=_produce, daemon=True).start()

        while True:
            try:
                msg = await loop.run_in_executor(None, lambda: chunk_queue.get(timeout=120))
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("done", "error"):
                    break
            except Exception:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Timed out waiting for Gemini'})}\n\n"
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
