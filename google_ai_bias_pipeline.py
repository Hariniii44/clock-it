"""
Google AI Mode + Bias-Aware Analysis Pipeline
----------------------------------------------
Step 1  : Google AI Mode retrieves and synthesises evidence (via SerpAPI)
Steps 2-5: Same bias-aware NLI → weighting → verdict as core_pipeline.py

Demonstrates the research contribution on top of Google's best-in-class
retrieval — no temporal pollution, no stale sources.

Usage:
    python google_ai_bias_pipeline.py
    python google_ai_bias_pipeline.py --skip-gemini
"""

import sys
import os
import re
sys.path.append('src')

from dotenv import load_dotenv
load_dotenv()

from src.evidence_retrieval.google_ai_mode_simple import query_google_ai_mode


def _domain(url: str) -> str:
    return url.split("://", 1)[-1].split("/")[0].replace("www.", "")


def _refs_to_evidence(references: list) -> list:
    """Map Google AI Mode references → evidence dicts."""
    evidence = []
    for ref in references:
        snippet = ref.get("snippet", "").strip()
        if not snippet:
            continue
        link = ref.get("link", "")
        date = ""
        m = re.search(
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(20\d{2})\b',
            snippet, re.IGNORECASE
        )
        if m:
            date = m.group(0)
        evidence.append({
            "title":         ref.get("title", ""),
            "snippet":       snippet,
            "link":          link,
            "source":        ref.get("source", _domain(link)),
            "date":          date,
            "evidence_type": "web",
        })
    return evidence


def run(claim: str, skip_gemini: bool = False):
    SERP_API_KEY = os.getenv("SERP_API_KEY", "")
    if not SERP_API_KEY:
        print("ERROR: SERP_API_KEY not set in .env")
        return

    print("\n" + "=" * 70)
    print("GOOGLE AI MODE + BIAS-AWARE PIPELINE")
    print("=" * 70)
    print(f"Claim: {claim}\n")

    # ── STEP 1: Google AI Mode retrieval ───────────────────────────────────
    print("STEP 1: GOOGLE AI MODE RETRIEVAL")
    print("-" * 70)
    print("Querying Google AI Mode...")
    google_result = query_google_ai_mode(claim, SERP_API_KEY)

    synthesis  = google_result["synthesis"]
    references = google_result["references"]
    evidence   = _refs_to_evidence(references)

    print(f"  Google returned {len(references)} references → {len(evidence)} usable snippets\n")
    print("GOOGLE SYNTHESIS:")
    print(synthesis if synthesis else "(no synthesis)")
    print(f"\nCITED SOURCES ({len(references)}):")
    for i, ref in enumerate(references, 1):
        print(f"  [{i}] {ref.get('source','?'):20s} | {ref.get('title','')[:60]}")

    if not evidence:
        print("\nNo usable snippets — cannot run bias analysis.")
        return

    # ── STEP 2: NLI + bias detection ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: NLI VERIFICATION + BIAS DETECTION")
    print("-" * 70)

    from verification.verification import ClaimVerifier
    from bias_detection.bias_detection import BiasDetector

    print("Loading NLI model...")
    verifier      = ClaimVerifier()
    print("Loading bias detection models...")
    bias_detector = BiasDetector()

    verification_results = []
    bias_analyses        = []

    for i, ev in enumerate(evidence, 1):
        print(f"\n  Source {i}/{len(evidence)}: {_domain(ev['link'])}")
        print(f"    {ev['snippet'][:150]}{'...' if len(ev['snippet']) > 150 else ''}")

        vr = verifier.verify_claim(claim, ev["snippet"])
        if not isinstance(vr, dict) or 'label' not in vr:
            vr = {'label': 'neutral', 'confidence': 0.5}
        verification_results.append(vr)

        ba = bias_detector.analyze_source(ev["snippet"], ev["link"])
        bias_analyses.append(ba)

        prof    = ba.get("source_profile", {})
        combined = ba.get("combined_score", {})
        print(f"    NLI: {vr['label'].upper()} ({vr['confidence']:.1%})  |  "
              f"Profile: {prof.get('bias_interpretation','unknown')} "
              f"(score {prof.get('bias_score',0):+.1f})  |  "
              f"Combined bias: {combined.get('overall_bias', 0):.3f}")

    # Trim lists to matched length (safety)
    n = min(len(evidence), len(verification_results), len(bias_analyses))
    evidence             = evidence[:n]
    verification_results = verification_results[:n]
    bias_analyses        = bias_analyses[:n]

    # ── STEP 3: Bias-aware weighting ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: BIAS-AWARE EVIDENCE WEIGHTING  (core research contribution)")
    print("-" * 70)
    print("   • Source bias profiles (99k Sri Lankan articles)")
    print("   • Dynamic bias-claim alignment calculation")
    print("   • Authority & recency weighting")
    print("   • Multi-dimensional uncertainty quantification")

    from evidence_weighting.evidence_weighting import EvidenceWeighter, VerdictGenerator
    from bias_detection.framing_analyzer import FramingAnalyzer
    from config import Config

    weighter         = EvidenceWeighter()
    verdict_gen      = VerdictGenerator()
    framing_analyzer = FramingAnalyzer(
        groq_key=Config.GROQ_API_KEY,
        bias_profiles=weighter.bias_profiles,
    )

    print("\nRunning framing analysis (entity-level + cross-source LLM)...")
    framing_result = framing_analyzer.analyze(claim, evidence)
    print(f"  Framing divergence: {framing_result.get('divergence_level','unknown').upper()}")

    weighted_evidence = weighter.weight_all_evidence(
        claim, evidence, verification_results, bias_analyses,
        framing_analyses=framing_result,
    )

    print("\nWEIGHTING RESULTS:")
    print("─" * 70)
    for i, item in enumerate(weighted_evidence, 1):
        ev      = item["evidence"]
        vr      = item["verification"]
        w       = item["weight"]
        align   = item["bias_alignment"]
        framing = item.get("framing_entry", {})
        domain  = _domain(ev["link"])
        print(f"Source {i:2d} | {vr['label']:8s} | Weight: {w:.3f} | Bias-align: {align:.3f}")
        print(f"         └─ {domain:30s} | "
              f"Framing: {framing.get('framing_direction','unknown')} | "
              f"Consistency: {framing.get('consistency_with_profile','unknown')}")

    # ── STEP 4: Verdict ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: WEIGHTED VERDICT")
    print("-" * 70)

    verdict_result = verdict_gen.generate_verdict(weighted_evidence)

    verdict    = verdict_result.get("verdict", "UNCERTAIN")
    confidence = verdict_result.get("confidence", 0)
    support    = verdict_result.get("support_score", 0)
    refute     = verdict_result.get("refute_score", 0)
    neutral    = verdict_result.get("neutral_score", 0)
    uncertainty = verdict_result.get("uncertainty_decomposition", {})

    print(f"\nFINAL VERDICT  : {verdict}")
    print(f"Confidence     : {confidence:.1%}")
    print(f"Vote split     : Support {support:.1%} | Refute {refute:.1%} | Neutral {neutral:.1%}")
    print(f"\nUncertainty breakdown:")
    print(f"  Epistemic (model variance) : {uncertainty.get('epistemic', 0):.3f}")
    print(f"  Aleatoric (evidence scarcity): {uncertainty.get('aleatoric', 0):.3f}")
    print(f"  Bias-induced (source conflicts): {uncertainty.get('bias_induced', 0):.3f}")

    # ── Bias framing per source ────────────────────────────────────────────
    print("\nBIAS FRAMING ANALYSIS (per source):")
    print("─" * 70)
    for i, item in enumerate(weighted_evidence, 1):
        ev      = item["evidence"]
        framing = item.get("framing_entry", {})
        domain  = _domain(ev["link"])
        print(f"\n  Source {i} | {domain}")
        print(f"    Framing direction : {framing.get('framing_direction','unknown')}")
        print(f"    Profile consistency: {framing.get('consistency_with_profile','unknown')}")
        phrases = framing.get("loaded_phrases", [])
        if phrases:
            print(f"    Loaded phrases    : {', '.join(str(p) for p in phrases[:3])}")
        expl = framing.get("explanation", "") or item.get("explanation", "")
        if expl:
            print(f"    Explanation       : {expl}")

    cross = framing_result.get("cross_source_divergence", "")
    if cross:
        print(f"\nCROSS-SOURCE DIVERGENCE: {framing_result.get('divergence_level','').upper()}")
        print(f"  {cross}")

    # ── Comparison summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON: GOOGLE SYNTHESIS vs BIAS-AWARE VERDICT")
    print("=" * 70)
    first_para = (synthesis or "").split("\n")[0]
    print(f"\nGoogle AI Mode says:")
    print(f"  \"{first_para}\"")
    print(f"\nBias-aware verdict : {verdict} ({confidence:.1%} confidence)")
    print(f"  Based on {len(evidence)} sources weighted by political framing alignment")

    # ── STEP 5: Gemini explanations (optional) ─────────────────────────────
    if not skip_gemini:
        print("\n" + "=" * 70)
        print("STEP 5: GEMINI NATURAL LANGUAGE EXPLANATIONS")
        print("-" * 70)
        try:
            from verification.gemini_verification import GeminiClaimVerifier
            gemini       = GeminiClaimVerifier()
            gemini_result = gemini.generate_weighted_explanation(
                claim, weighted_evidence, verdict_result, framing_result
            )
            if gemini_result.get("weighting_explanation"):
                print("\nWEIGHTING EXPLANATION:")
                print(gemini_result["weighting_explanation"])
            if gemini_result.get("bias_explanation"):
                print("\nBIAS PATTERN EXPLANATION:")
                print(gemini_result["bias_explanation"])
        except Exception as e:
            print(f"  Gemini unavailable: {e}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    skip_gemini = "--skip-gemini" in sys.argv

    print("=" * 70)
    print("GOOGLE AI MODE + BIAS-AWARE FACT-CHECKING PIPELINE")
    print("=" * 70)
    print("Retrieval : Google AI Mode (SerpAPI)")
    print("Analysis  : NLI + Bias-aware weighting (core research contribution)")
    print()

    claim = input("Enter claim:\n>> ").strip()
    if claim:
        run(claim, skip_gemini=skip_gemini)
