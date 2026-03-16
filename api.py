"""
FastAPI wrapper for the Context-Adaptive Evidence Weighting Framework.

Models are loaded ONCE at startup and reused for every request.
core_pipeline.py is not modified — all logic is imported from it.

Run with:
    uvicorn api:app --reload
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

sys.path.append('src')

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import Config
from verification.gemini_verification import GeminiClaimVerifier

# Import pipeline helper functions directly from core_pipeline
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

    weighter = EvidenceWeighter()
    _models["weighter"] = weighter
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
    skip_gemini: bool = False


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
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(_models)}


@app.post("/verify", response_model=VerifyResponse)
def verify_claim(request: VerifyRequest):
    if not request.claim.strip():
        raise HTTPException(status_code=400, detail="claim must not be empty")

    claim = request.claim.strip()
    skip_gemini = request.skip_gemini

    m = _models  # shorthand

    # ------------------------------------------------------------------
    # STEP 0: Temporal claim analysis
    # ------------------------------------------------------------------
    claim_temporal = detect_temporal_claim(claim)

    # ------------------------------------------------------------------
    # STEP 1: Dual evidence retrieval — DB and web run in parallel
    # ------------------------------------------------------------------
    def _fetch_db():
        try:
            results = m["hybrid_retriever"].hybrid_search(
                query=claim,
                claim_types=None,
                total_results=10,
                use_query_expansion=True,
            )
            return [
                {
                    "source": r["source"],
                    "title": r["title"],
                    "snippet": r.get("passage", r["text"]),
                    "link": r["url"],
                    "dataset_source": r["dataset"],
                    "authority": r["authority"],
                    "relevance_score": r["similarity_score"],
                    "evidence_type": "database",
                }
                for r in results
            ]
        except Exception as e:
            print(f"  Database retrieval failed: {e}")
            return []

    def _fetch_web():
        try:
            results = m["web_retriever"].retrieve_hybrid_serper_decomposition(
                claim,
                num_results=10,
                results_per_query=8,
            )
            for ev in results:
                ev["evidence_type"] = "web"
            return results
        except Exception as e:
            print(f"  Web retrieval failed: {e}")
            return []

    with ThreadPoolExecutor(max_workers=2) as pool:
        db_future = pool.submit(_fetch_db)
        web_future = pool.submit(_fetch_web)
        db_formatted = db_future.result()
        web_evidence = web_future.result()

    all_evidence = db_formatted + web_evidence
    total_sources = len(all_evidence)

    if total_sources == 0:
        raise HTTPException(status_code=422, detail="No evidence found for this claim.")

    # ------------------------------------------------------------------
    # Temporal evidence analysis + pre-filter stale sources
    # ------------------------------------------------------------------
    evidence_temporal = analyze_evidence_temporal_distribution(all_evidence)
    content_freshness = analyze_content_freshness(all_evidence, claim)

    # If the claim text had no temporal markers but the evidence is fresh,
    # upgrade the classification so it is reported as "recent" (not "historical").
    # Three independent signals are checked as fallbacks because web evidence
    # often lacks a structured date field:
    #   1. evidence_temporal: structured dates present and mostly <48 h old
    #   2. URL dates: news sites embed publication date in the path (e.g. /2026/03/16/)
    #   3. content freshness: sources use "breaking", "developing", "latest" language
    _now = datetime.now()
    _current_ym = f"/{_now.year}/{_now.month:02d}/"
    _url_recent = sum(1 for ev in all_evidence if _current_ym in ev.get("link", ""))
    _url_recent_ratio = _url_recent / len(all_evidence) if all_evidence else 0.0

    _has_structured_recency = (
        evidence_temporal.get("recent_evidence_ratio", 0) > 0.6
        and evidence_temporal.get("avg_evidence_age_hours") is not None
        and evidence_temporal["avg_evidence_age_hours"] < 48
    )
    _has_url_recency = _url_recent >= 1  # even one URL with this month's date is conclusive
    _has_content_freshness = content_freshness.get("total_freshness_signals", 0) >= 1

    if not claim_temporal["is_temporal"] and (_has_structured_recency or _has_url_recency or _has_content_freshness):
        claim_temporal = {
            **claim_temporal,
            "is_temporal": True,
            "recency_level": "recent",
            "analysis_note": "Recency inferred from evidence (URL dates or breaking-news language); claim text had no temporal markers",
        }

    temporal_confidence = calculate_temporal_confidence(claim_temporal, evidence_temporal, total_sources)

    # If recency was inferred from URL dates or breaking-news language (not from
    # structured date fields), calculate_temporal_confidence won't generate a
    # warning because breaking_count == 0.  Inject the warning and penalty here.
    if "analysis_note" in claim_temporal and not temporal_confidence["temporal_warnings"]:
        temporal_confidence["temporal_warnings"].append(
            "Recency inferred from evidence — this may be a developing story; verdict may change as more information emerges"
        )
        temporal_confidence["is_breaking_news"] = True
        temporal_confidence["temporal_confidence_multiplier"] = 0.85

    temporal_mismatch = detect_temporal_mismatch(all_evidence)
    evidence_quality = assess_evidence_quality(all_evidence, claim_temporal)

    if temporal_mismatch["has_mismatch"] and temporal_mismatch.get("recent_source_count", 0) >= 3:
        stale_indices = set(temporal_mismatch["stale_source_indices"])
        all_evidence = [e for i, e in enumerate(all_evidence) if i not in stale_indices]
        total_sources = len(all_evidence)

    # Recount per-type after filtering so response counts stay consistent
    db_sources_count = sum(1 for e in all_evidence if e.get("evidence_type") == "database")
    web_sources_count = sum(1 for e in all_evidence if e.get("evidence_type") == "web")

    # ------------------------------------------------------------------
    # STEP 2 + 3 setup: NLI, per-source bias, and claim-level bias run concurrently
    #
    # groq_verifier.verify_batch  — single Groq API call (I/O bound)
    # bias_detector per source    — RoBERTa + keyword fallback (CPU, fast)
    # claim_bias_analyzer.analyze — Groq API call (I/O bound)
    #
    # The two Groq calls and the bias loop are independent, so we fire all
    # three at once and join when done.
    # ------------------------------------------------------------------
    def _run_nli():
        return m["groq_verifier"].verify_batch(claim, all_evidence)

    def _run_bias_detector():
        results = []
        for evidence in all_evidence:
            try:
                results.append(
                    m["bias_detector"].analyze_source(
                        evidence.get("snippet", ""), evidence.get("link", "")
                    )
                )
            except Exception:
                results.append({"overall_bias": 0.0, "confidence": 0.0})
        return results

    def _run_claim_bias():
        return m["claim_bias_analyzer"].analyze(claim, all_evidence)

    with ThreadPoolExecutor(max_workers=3) as pool:
        nli_future          = pool.submit(_run_nli)
        bias_det_future     = pool.submit(_run_bias_detector)
        claim_bias_future   = pool.submit(_run_claim_bias)

        groq_batch    = nli_future.result()
        bias_analyses = bias_det_future.result()
        bias_result   = claim_bias_future.result()

    use_groq = len(groq_batch) == len(all_evidence)

    verification_results = []
    for i, evidence in enumerate(all_evidence):
        try:
            if use_groq:
                vr = groq_batch[i]
            else:
                vr = m["verifier"].verify_claim(claim, evidence.get("snippet", ""))

            if not isinstance(vr, dict) or "label" not in vr or "confidence" not in vr:
                verification_results.append({"label": "error", "confidence": 0.0})
            else:
                verification_results.append(vr)
        except Exception as e:
            print(f"  NLI source {i} error: {e}")
            verification_results.append({"label": "error", "confidence": 0.0})

    # ------------------------------------------------------------------
    # STEP 3: Bias-aware evidence weighting
    # ------------------------------------------------------------------
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
        claim,
        all_evidence,
        verification_results,
        bias_analyses,
        framing_analyses=framing_result,
        precomputed_alignments=precomputed_alignments,
    )

    # ------------------------------------------------------------------
    # STEP 4: Verdict generation
    # ------------------------------------------------------------------
    base_verdict = m["verdict_generator"].generate_verdict(weighted_evidence)
    temporal_adjusted_confidence = base_verdict["confidence"] * temporal_confidence["temporal_confidence_multiplier"]

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

    # ------------------------------------------------------------------
    # STEP 5: Optional Gemini explanation
    # ------------------------------------------------------------------
    gemini_explanation = None
    if m["gemini_available"] and not skip_gemini:
        try:
            temporal_context = {
                "is_breaking_news": temporal_confidence.get("is_breaking_news", False),
                "recency_level": claim_temporal["recency_level"],
                "estimated_hours_old": claim_temporal["estimated_hours_old"],
                "content_freshness_score": content_freshness["freshness_score"],
                "breaking_indicators": content_freshness.get("breaking_indicators", []),
                "temporal_warnings": temporal_confidence["temporal_warnings"],
            }
            gemini_explanation = m["gemini_explainer"].explain_weighting_decisions(
                claim, weighted_evidence, final_verdict, temporal_context
            )
        except Exception as e:
            print(f"  Gemini explanation failed: {e}")

    # ------------------------------------------------------------------
    # Build response
    # ------------------------------------------------------------------
    sources_out = []
    for item in weighted_evidence:
        ev = item["evidence"]
        vr = item["verification"]
        sources_out.append({
            "title": ev.get("title", ""),
            "link": ev.get("link", ""),
            "source": ev.get("source", ""),
            "evidence_type": ev.get("evidence_type", ""),
            "verdict": vr.get("label", ""),
            "confidence": vr.get("confidence", 0.0),
            "weight": item.get("weight", 0.0),
            "bias_alignment": item.get("bias_alignment", 0.0),
        })

    uncertainty = final_verdict.get("uncertainty_decomposition", {})

    return VerifyResponse(
        claim=claim,
        verdict=verdict_label,
        confidence=final_verdict["confidence"],
        support_score=final_verdict.get("support_score", 0.0),
        refute_score=final_verdict.get("refute_score", 0.0),
        neutral_score=final_verdict.get("neutral_score", 0.0),
        total_sources=total_sources,
        db_sources=db_sources_count,
        web_sources=web_sources_count,
        temporal_status=claim_temporal["recency_level"],
        is_breaking_news=temporal_confidence["is_breaking_news"],
        temporal_warnings=temporal_confidence["temporal_warnings"],
        uncertainty={
            "epistemic": uncertainty.get("epistemic", 0.0),
            "aleatoric": uncertainty.get("aleatoric", 0.0),
            "bias_induced": uncertainty.get("bias_induced", 0.0),
        },
        evidence_quality={
            "level": evidence_quality["quality_level"],
            "note": evidence_quality.get("quality_note"),
            "established_count": evidence_quality["established_count"],
            "unverified_social_count": evidence_quality["unverified_social_count"],
        },
        claim_bias={
            "framing_type": claim_bias.get("framing_type", ""),
            "emotional_tone": claim_bias.get("emotional_tone", 0.0),
            "political_direction": claim_bias.get("political_direction", ""),
            "explanation": claim_bias.get("explanation", ""),
        },
        divergence_level=framing_result["divergence_level"],
        sources=sources_out,
        gemini_explanation=gemini_explanation,
    )
