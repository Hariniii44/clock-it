"""
FastAPI wrapper for the Context-Adaptive Evidence Weighting Framework.

Models are loaded ONCE at startup and reused for every request.
core_pipeline.py is not modified — all logic is imported from it.

Run with:
    uvicorn api:app --reload
"""

import asyncio
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
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
    from src.retrieval.qdrant_hybrid_retriever import QdrantHybridRetriever as HybridRetriever
    from src.verification.gemini_nli_verification import GeminiNLIVerifier
    from src.bias_detection.claim_bias_analyzer import ClaimBiasAnalyzer
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

    _models["nli_verifier"] = GeminiNLIVerifier()
    print("  GeminiNLIVerifier ready")


    _models["claim_bias_analyzer"] = ClaimBiasAnalyzer(gemini_api_key=Config.GEMINI_API_KEY)
    print("  ClaimBiasAnalyzer (Gemini 2.5 Pro) ready")

    _models["weighter"] = EvidenceWeighter()
    _models["verdict_generator"] = VerdictGenerator()
    print("  EvidenceWeighter + VerdictGenerator ready")

    try:
        from sentence_transformers import CrossEncoder
        _models["relevance_filter"] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("  CrossEncoder relevance filter ready (ms-marco-MiniLM-L-6-v2)")
    except Exception as e:
        _models["relevance_filter"] = None
        print(f"  CrossEncoder unavailable — relevance filtering disabled: {e}")

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
    disable_bias: bool = False  # Set True to disable bias-aware weighting (ablation study)


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
    bias_sources: list[dict]
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
async def _pipeline_steps(claim: str, m: dict, disable_bias: bool = False):
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
                formatted = [
                    {
                        "source": r["source"], "title": r["title"],
                        "snippet": r.get("passage", r["text"]), "link": r["url"],
                        "dataset_source": r["dataset"], "authority": r["authority"],
                        "relevance_score": r["similarity_score"], "evidence_type": "database",
                    }
                    for r in results
                ]
                # Qdrant already ranks by semantic similarity — skip the
                # cross-encoder filter here so high-quality chunks aren't
                # incorrectly dropped by the MS-MARCO-trained model.
                return formatted
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

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_db  = ex.submit(_db)
            fut_web = ex.submit(_web)
            return fut_db.result(), fut_web.result()

    db_formatted, web_evidence = await loop.run_in_executor(None, _retrieval)
    all_evidence = db_formatted + web_evidence

    if not all_evidence:
        yield {"type": "error", "message": "No evidence found for this claim."}
        return

    # Drop sources where the snippet is predominantly non-Latin script
    # (YouTube pages where trafilatura extracted Sinhala/Tamil UI text instead
    # of video content — these contain no verifiable English text).
    def _is_latin_content(text: str) -> bool:
        if not text:
            return False
        latin = sum(1 for c in text if c.isascii() and c.isalpha())
        total = sum(1 for c in text if c.isalpha())
        return total == 0 or (latin / total) >= 0.6

    before = len(all_evidence)
    all_evidence = [e for e in all_evidence if _is_latin_content(e.get("snippet", ""))]
    dropped = before - len(all_evidence)
    if dropped:
        print(f"  Non-Latin filter: {dropped} source(s) dropped (Sinhala/Tamil UI text)")

    # ------------------------------------------------------------------
    # Temporal evidence analysis + stale-source filter (fast)
    # ------------------------------------------------------------------
    evidence_temporal = analyze_evidence_temporal_distribution(all_evidence)
    content_freshness = analyze_content_freshness(all_evidence, claim)

    _now = datetime.now()
    _two_weeks_ago = _now - timedelta(days=14)

    def _url_within_two_weeks(url: str) -> bool:
        m = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
        if m:
            try:
                pub = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                return _two_weeks_ago <= pub <= _now
            except ValueError:
                pass
        return False

    _url_recent = sum(1 for ev in all_evidence if _url_within_two_weeks(ev.get("link", "")))
    _has_url_recency = _url_recent > len(all_evidence) / 2  # majority of sources must be recent
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

    # Only flag as developing story if a majority of sources are actually
    # dated within the last 2 weeks — not just because the claim lacks time markers.
    _recent_source_count = sum(
        1 for ev in all_evidence
        if _url_within_two_weeks(ev.get("link", "")) or _url_within_two_weeks(ev.get("url", ""))
    )
    _majority_recent = len(all_evidence) > 0 and (_recent_source_count / len(all_evidence)) >= 0.5

    if "analysis_note" in claim_temporal and _majority_recent and not temporal_confidence["temporal_warnings"]:
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

    # ------------------------------------------------------------------
    # STAGE 1 RELEVANCE FILTER: Cross-encoder pre-filter for DB sources
    # ------------------------------------------------------------------
    # DB sources come from Qdrant keyword/semantic search and can include
    # topically-adjacent but claim-irrelevant documents 
    # Web sources skip this filter — Serper already used the claim as query.
    ce_model = m.get("relevance_filter")
    if ce_model is not None:
        db_indices  = [i for i, e in enumerate(all_evidence) if e.get("evidence_type") == "database"]
        web_indices = [i for i, e in enumerate(all_evidence) if e.get("evidence_type") != "database"]

        if db_indices:
            pairs  = [(claim, all_evidence[i].get("snippet", "") or all_evidence[i].get("content", "")) for i in db_indices]
            scores = ce_model.predict(pairs)

            # Keep DB sources scoring above CE_KEEP_THRESHOLD (clearly topical).
            # Guarantee at least MIN_DB_KEEP sources survive, cap at MAX_DB_KEEP
            # to prevent Qdrant flooding the pipeline with marginally-relevant docs.
            # Attach scores to evidence items so Stage 2 can protect high-scorers.
            CE_KEEP_THRESHOLD = 1.0   # ms-marco: >1 = strong topical match
            MIN_DB_KEEP = 3
            MAX_DB_KEEP = 5
            for score, idx in zip(scores, db_indices):
                all_evidence[idx]["_ce_score"] = float(score)
            scored = sorted(zip(scores, db_indices), key=lambda x: x[0], reverse=True)
            # Strong keepers: above threshold, capped at MAX_DB_KEEP
            strong = [idx for score, idx in scored if score >= CE_KEEP_THRESHOLD]
            keep_db = set(strong[:MAX_DB_KEEP])
            # Top-up to MIN_DB_KEEP if not enough strong keepers
            if len(keep_db) < MIN_DB_KEEP:
                keep_db = {idx for _, idx in scored[:MIN_DB_KEEP]}

            dropped = len(db_indices) - len(keep_db)
            print(f"  [CrossEncoder] DB sources: {len(keep_db)} kept, {dropped} dropped (threshold={CE_KEEP_THRESHOLD})")
            for score, idx in scored:
                tag = "KEEP" if idx in keep_db else "DROP"
                print(f"    [{tag}] score={score:+.2f}  {all_evidence[idx].get('source', '')[:60]}")

            keep_indices = sorted(keep_db | set(web_indices))
            all_evidence = [all_evidence[i] for i in keep_indices]

    db_sources_count = sum(1 for e in all_evidence if e.get("evidence_type") == "database")
    web_sources_count = sum(1 for e in all_evidence if e.get("evidence_type") == "web")
    total_sources = len(all_evidence)

    # ------------------------------------------------------------------
    # SOCIAL MEDIA FILTER: exclude social media sources when sufficient
    # real news/db sources exist with substantive content.
    # Kept in response as excluded_sources (greyed out in UI) so users
    # can see them but know they weren't used in the verdict.
    # Threshold: ≥4 non-social sources with snippet > 100 chars.
    # ------------------------------------------------------------------
    SOCIAL_DOMAINS = {'facebook.com', 'x.com', 'twitter.com', 'instagram.com',
                      'tiktok.com', 'linkedin.com', 'threads.net'}

    def _is_social(ev):
        domain = ev.get('source', '') or ''
        try:
            from urllib.parse import urlparse
            domain = urlparse(ev.get('link', '')).netloc.lower() if not domain else domain
        except Exception:
            pass
        return any(s in domain for s in SOCIAL_DOMAINS)

    non_social_with_content = sum(
        1 for e in all_evidence
        if not _is_social(e) and len(e.get('snippet', '')) > 100
    )
    excluded_social = []
    if non_social_with_content >= 4:
        remaining, excluded_social = [], []
        for e in all_evidence:
            (excluded_social if _is_social(e) else remaining).append(e)
        if excluded_social:
            print(f"  [SocialFilter] Excluding {len(excluded_social)} social media source(s) "
                  f"({non_social_with_content} substantive non-social sources available)")
            all_evidence = remaining
            db_sources_count  = sum(1 for e in all_evidence if e.get("evidence_type") == "database")
            web_sources_count = sum(1 for e in all_evidence if e.get("evidence_type") == "web")
            total_sources     = len(all_evidence)

    # ------------------------------------------------------------------
    # STEP 2+3: NLI, per-source bias, claim-level bias — all concurrent
    # ------------------------------------------------------------------
    yield {"type": "status", "message": f"Found {total_sources} sources. Running verification and bias analysis..."}

    def _analysis():
        def _nli():
            return m["nli_verifier"].verify_batch(claim, all_evidence)

        def _claim_bias():
            return m["claim_bias_analyzer"].analyze(claim, all_evidence)

        with ThreadPoolExecutor(max_workers=2) as pool:
            nli_f        = pool.submit(_nli)
            claim_bias_f = pool.submit(_claim_bias)
            return nli_f.result(), claim_bias_f.result()

    groq_batch, bias_result = await loop.run_in_executor(None, _analysis)
    # bias_analyses is a placeholder — alignment and framing come entirely from
    # ClaimBiasAnalyzer (Gemini) via precomputed_alignments / framing_result.
    bias_analyses = [{"overall_bias": 0.0, "confidence": 0.0}] * len(all_evidence)

    # ------------------------------------------------------------------
    # STAGE 2 RELEVANCE FILTER: Groq relevance flag
    # ------------------------------------------------------------------
    # Groq read each source in full during NLI — it can definitively
    # judge relevance.
    # DB sources: always apply.
    # Web sources: apply only when the snippet is long enough (>100 chars)
    #   for Groq to make a reliable call. Short Serper snippets can look
    #   off-topic even when the full page is useful, so we skip those.
    if len(groq_batch) == len(all_evidence):
        irrelevant_indices = {
            i for i, gr in enumerate(groq_batch)
            if not gr.get('relevant', True)
            and (
                all_evidence[i].get('evidence_type') == 'database'
                or len(all_evidence[i].get('snippet', '')) > 100
            )
        }
        if irrelevant_indices:
            print(f"  [NLIFilter] Dropping {len(irrelevant_indices)} source(s) flagged IRRELEVANT by Gemini NLI:")
            for i in sorted(irrelevant_indices):
                print(f"    - {all_evidence[i].get('source', '') or all_evidence[i].get('link', '')[:60]}")
            keep            = [i for i in range(len(all_evidence)) if i not in irrelevant_indices]
            all_evidence    = [all_evidence[i] for i in keep]
            groq_batch      = [groq_batch[i]   for i in keep]
            bias_analyses   = [{"overall_bias": 0.0, "confidence": 0.0}] * len(keep)
            if bias_result and 'sources' in bias_result:
                bias_result['sources'] = [bias_result['sources'][i] for i in keep]

    if not groq_batch:
        yield {"type": "error", "message": "NLI verification failed — all Gemini models unavailable. Please try again later."}
        return

    verification_results = []
    for i, ev in enumerate(all_evidence):
        try:
            vr = groq_batch[i]
            if not isinstance(vr, dict) or "label" not in vr or "confidence" not in vr:
                verification_results.append({"label": "NEUTRAL", "confidence": 0.5, "reason": "Parse error"})
            else:
                verification_results.append(vr)
        except Exception as e:
            print(f"  NLI source {i} error: {e}")
            verification_results.append({"label": "NEUTRAL", "confidence": 0.5, "reason": "Parse error"})

    # ------------------------------------------------------------------
    # STEP 4: Bias-aware weighting + verdict (fast, no I/O)
    # ------------------------------------------------------------------
    yield {"type": "status", "message": "Generating verdict..."}

    claim_bias = bias_result["claim_bias"]
    framing_result = {
        "sources": [
            {
                "index": i,
                "source": (all_evidence[i].get("source", "") if i < len(all_evidence) else ""),
                "title":  (all_evidence[i].get("title",  "") if i < len(all_evidence) else ""),
                "framing_type": s["framing_type"],
                "consistency_with_profile": "unknown",
                "loaded_phrases": s["loaded_phrases"],
                "explanation": s["explanation"],
                "emotional_tone": s["emotional_tone"],
                "political_direction": s["political_direction"],
                "alignment": s["alignment"],
                "alignment_breakdown": s.get("alignment_breakdown", {}),
                # Sri Lankan-specific bias flags
                "gender_bias_signal":    s.get("gender_bias_signal", False),
                "ethnic_bias_signal":    s.get("ethnic_bias_signal", False),
                "trauma_trivialization": s.get("trauma_trivialization", False),
                "source_type":           s.get("source_type", "unknown"),
            }
            for i, s in enumerate(bias_result["sources"])
        ],
        "divergence_level": _compute_divergence(bias_result["sources"]),
        "cross_source_divergence": _divergence_summary(bias_result["sources"], claim_bias),
        "contested_entities": [],
    }
    precomputed_alignments = [s["alignment"] for s in bias_result["sources"]]
    weighted_evidence = m["weighter"].weight_all_evidence(
        claim, all_evidence, verification_results, bias_analyses,
        framing_analyses=framing_result, precomputed_alignments=precomputed_alignments,
        disable_bias=disable_bias,
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
        fe = item.get("framing_entry", {})
        sources_out.append({
            "title": ev.get("title", ""), "link": ev.get("link", ""),
            "source": ev.get("source", ""), "evidence_type": ev.get("evidence_type", ""),
            "snippet": ev.get("snippet", ""),
            "date": ev.get("date", "") or ev.get("date_raw", ""),
            "verdict": vr.get("label", ""), "confidence": vr.get("confidence", 0.0),
            "verdict_reason": vr.get("reason", ""),
            "weight": item.get("weight", 0.0), "bias_alignment": item.get("bias_alignment", 0.0),
            "weight_explanation": item.get("explanation", ""),
            "gender_bias_signal":    fe.get("gender_bias_signal", False),
            "ethnic_bias_signal":    fe.get("ethnic_bias_signal", False),
            "trauma_trivialization": fe.get("trauma_trivialization", False),
            "source_type":           fe.get("source_type", "unknown"),
        })

    # Append excluded social media sources (greyed out in UI — not used in verdict)
    for ev in excluded_social:
        sources_out.append({
            "title":         ev.get("title", ""),
            "link":          ev.get("link", ""),
            "source":        ev.get("source", ""),
            "evidence_type": ev.get("evidence_type", "web"),
            "snippet":       ev.get("snippet", ""),
            "date":          ev.get("date", "") or ev.get("date_raw", ""),
            "excluded":      True,
            "exclusion_reason": "Social media — not evaluated (sufficient news sources available)",
            # No verdict/weight fields — not processed through NLI or weighting
            "verdict": None, "confidence": None, "weight": None,
            "bias_alignment": None, "verdict_reason": "", "weight_explanation": "",
            "gender_bias_signal": False, "ethnic_bias_signal": False,
            "trauma_trivialization": False, "source_type": "social_media",
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
            "claim_type":          claim_bias.get("claim_type", "general"),
            "framing_type":        claim_bias.get("framing_type", ""),
            "emotional_tone":      claim_bias.get("emotional_tone", 0.0),
            "political_direction": claim_bias.get("political_direction", ""),
            "explanation":         claim_bias.get("explanation", ""),
            "loaded_phrases":      claim_bias.get("loaded_phrases", []),
            "gender_bias_signal":    claim_bias.get("gender_bias_signal", False),
            "ethnic_bias_signal":    claim_bias.get("ethnic_bias_signal", False),
            "trauma_trivialization": claim_bias.get("trauma_trivialization", False),
        },
        "divergence_level": framing_result["divergence_level"],
        "bias_sources": [
            {
                "source": s.get("source", ""),
                "title": s.get("title", ""),
                "framing_type": s.get("framing_type", ""),
                "emotional_tone": s.get("emotional_tone", 0.0),
                "political_direction": s.get("political_direction", ""),
                "loaded_phrases": s.get("loaded_phrases", []),
                "explanation": s.get("explanation", ""),
                "alignment": s.get("alignment", 0.0),
                "alignment_breakdown": s.get("alignment_breakdown", {}),
                "gender_bias_signal":    s.get("gender_bias_signal", False),
                "ethnic_bias_signal":    s.get("ethnic_bias_signal", False),
                "trauma_trivialization": s.get("trauma_trivialization", False),
                "source_type":           s.get("source_type", "unknown"),
                "profile_score": (
                    m["weighter"].bias_profiles.get(s.get("source", ""), {}).get("bias_score")
                ),
                "profile_interpretation": (
                    m["weighter"].bias_profiles.get(s.get("source", ""), {}).get("interpretation")
                ),
            }
            for s in framing_result["sources"]
        ],
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
        "bias_context": {
            "claim_bias": result_data["claim_bias"],
            "bias_sources": result_data["bias_sources"],
            "divergence_level": framing_result["divergence_level"],
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

    async for event in _pipeline_steps(claim, _models, disable_bias=request.disable_bias):
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
                    claim, cached["weighted_evidence"], cached["final_verdict"], cached["temporal_context"],
                    cached.get("bias_context", {}).get("claim_bias"),
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
            async for event in _pipeline_steps(claim, _models, disable_bias=request.disable_bias):
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
            cached.get("bias_context", {}).get("claim_bias"),
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
                    cached.get("bias_context", {}).get("claim_bias"),
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


@app.post("/explain/bias")
async def stream_bias_explanation(request: ExplainRequest):
    """
    Stream a cross-source bias pattern explanation using the cached bias dimensions.
    No re-running the pipeline — uses what was already computed during /verify.
    """
    claim = request.claim.strip()
    cached = _result_cache.get(claim)
    if not cached or "bias_context" not in cached:
        raise HTTPException(status_code=404, detail="No cached result found. Please verify the claim first.")

    import queue as q
    import threading

    bias_ctx = cached["bias_context"]

    async def generate():
        chunk_queue: q.Queue = q.Queue()
        loop = asyncio.get_running_loop()

        def _produce():
            try:
                from groq import Groq
                client = Groq(api_key=Config.GROQ_API_KEY)

                claim_bias  = bias_ctx["claim_bias"]
                sources     = bias_ctx["bias_sources"]
                divergence  = bias_ctx["divergence_level"]

                # Build a readable summary of each source's bias dimensions
                source_lines = []
                for i, s in enumerate(sources, 1):
                    phrases = '", "'.join(s["loaded_phrases"]) if s["loaded_phrases"] else "none"
                    title = s.get("title", "") or ""
                    source_lines.append(
                        f"  [{i}] {s['source'] or 'unknown'}"
                        + (f' — "{title}"' if title else "") +
                        f"\n      Framing: {s['framing_type']} | "
                        f"Political: {s['political_direction']} | "
                        f"Tone: {s['emotional_tone']:+.2f}"
                        + (f'\n      Loaded phrases: "{phrases}"' if s["loaded_phrases"] else "") +
                        f"\n      Bias note: {s['explanation']}"
                    )

                prompt = (
                    f'Claim: "{claim}"\n\n'
                    f"Overall claim framing: {claim_bias['framing_type']} "
                    f"({claim_bias['political_direction']}, tone {claim_bias['emotional_tone']:+.2f})\n"
                    f"Source divergence level: {divergence}\n\n"
                    f"Per-source bias data:\n"
                    + "\n\n".join(source_lines) +
                    "\n\nIdentify 2-4 cross-source bias PATTERNS. "
                    "A pattern is a group of sources that share similar framing, OR two sources "
                    "with contrasting framing of the same fact.\n\n"
                    "For EACH pattern, output EXACTLY this format:\n\n"
                    "• <short pattern title>\n"
                    "  Source [N] (<domain>) said: '<quote the actual title or a loaded phrase from it>'\n"
                    "  Source [M] (<domain>) said: '<quote the actual title or a loaded phrase from it>'\n"
                    "  Analysis: <2-3 sentences explaining what this pattern reveals about bias, "
                    "why the specific language is loaded, and what effect it has on the reader>\n\n"
                    "Rules:\n"
                    "- ALWAYS quote directly from the title or loaded phrases provided above\n"
                    "- Name the domain (e.g. adaderana.lk) in every source reference\n"
                    "- Do not write any text before the first bullet or after the last pattern\n"
                    "- Use plain English suitable for a general audience"
                )

                with client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1024,
                    stream=True,
                ) as stream:
                    for chunk in stream:
                        text = chunk.choices[0].delta.content or ""
                        if text:
                            chunk_queue.put({"type": "chunk", "text": text})

                chunk_queue.put({"type": "done"})

            except Exception as e:
                chunk_queue.put({"type": "error", "message": str(e)})

        threading.Thread(target=_produce, daemon=True).start()

        while True:
            try:
                msg = await loop.run_in_executor(None, lambda: chunk_queue.get(timeout=60))
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("done", "error"):
                    break
            except Exception:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Timed out'})}\n\n"
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
