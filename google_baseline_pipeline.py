"""
Google AI Mode Baseline Pipeline
---------------------------------
Minimal pipeline: user enters a claim → Google AI Mode returns its
synthesised answer and cited sources.

No NLI, no bias analysis, no weighting.
Used to compare Google's plain retrieval+synthesis against the full
bias-aware pipeline in core_pipeline.py.

Usage:
    python google_baseline_pipeline.py
"""

import os
from dotenv import load_dotenv
from src.evidence_retrieval.google_ai_mode_simple import query_google_ai_mode

load_dotenv()

SERP_API_KEY = os.getenv("SERP_API_KEY", "")


def main():
    print("=" * 60)
    print("GOOGLE AI MODE — BASELINE PIPELINE")
    print("=" * 60)
    print("No bias analysis. Pure Google synthesis.")
    print()

    if not SERP_API_KEY:
        print("ERROR: SERP_API_KEY not set in .env")
        return

    claim = input("Enter claim or query:\n>> ").strip()
    if not claim:
        return

    print("\nQuerying Google AI Mode...")
    result = query_google_ai_mode(claim, SERP_API_KEY)

    synthesis  = result["synthesis"]
    references = result["references"]

    # ── Synthesised answer ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GOOGLE AI SYNTHESIS")
    print("=" * 60)
    if synthesis:
        print(synthesis)
    else:
        print("(No synthesis returned)")

    # ── Cited sources ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"CITED SOURCES ({len(references)})")
    print("=" * 60)
    for i, ref in enumerate(references, 1):
        print(f"\n[{i}] {ref['title']}")
        print(f"    Source  : {ref['source']}")
        print(f"    URL     : {ref['link']}")
        if ref["snippet"]:
            print(f"    Snippet : {ref['snippet'][:200]}"
                  f"{'...' if len(ref['snippet']) > 200 else ''}")

    print("\n" + "=" * 60)
    print("NOTE: This is Google's answer with no bias analysis.")
    print("Run core_pipeline.py to see the bias-aware verdict")
    print("on the same claim for comparison.")
    print("=" * 60)


if __name__ == "__main__":
    main()
