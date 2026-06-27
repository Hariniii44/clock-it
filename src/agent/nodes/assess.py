import asyncio
import json
from typing import List

from google import genai
from google.genai import types as genai_types

from config import Config
from ..state import FactCheckState, SubClaimResult


SYSTEM_PROMPT = """You are a fact-checking analyst. Assess a specific sub-claim based solely
on the independent evidence sources provided."""

ASSESS_PROMPT = """Assess the following sub-claim using only the independent sources listed.

SUB-CLAIM: {sub_claim_text}
SUB-CLAIM TYPE: {sub_claim_type}

INDEPENDENT SOURCES (derivative/echo sources already removed):
{sources_json}

Rules for verdict:
- SUPPORTED:           ALL of these must be true: (1) independent original sources confirm the
                       sub-claim, (2) precision_match is true for those sources, (3) no source
                       contradicts it. The number of sources does not matter if precision is wrong.
- PARTIALLY_SUPPORTED: sources confirm the general direction but not the specific detail.
                       Use this when sources confirm "tens of thousands" but the claim says "40,000",
                       or when sources say "up to 40,000" but the claim treats it as a confirmed count.
- NOT_VERIFIED:        no independent source confirms this sub-claim (absence of evidence).
- REFUTED:             independent sources directly contradict the sub-claim with evidence.
- UNCERTAIN:           sources exist but are too vague, inaccessible, or conflicting to assess.

Critical precision rules for NUMBER sub-claims:
- Check the precision_level of EACH source. If all sources have precision_level of "estimate",
  "upper_bound", or "vague_magnitude" — the verdict is PARTIALLY_SUPPORTED, NOT SUPPORTED.
  A figure that is an estimate or upper bound does NOT confirm a specific count.
- precision_gap must be true if any source uses hedging language ("approximately", "up to",
  "as many as", "estimated", "may have", "could have") for the specific number in the sub-claim.
- The number of independent sources is irrelevant to precision — 20 sources all citing an estimate
  still does not confirm a specific count.

Return JSON with:
{{
  "verdict": "SUPPORTED|PARTIALLY_SUPPORTED|NOT_VERIFIED|REFUTED|UNCERTAIN",
  "reasoning": "2-3 sentences. State how many independent sources you have, what precision_level
                each source uses for the key claim, and what specifically is or is not confirmed.",
  "precision_gap": true/false
}}

precision_gap is true if the sub-claim is more precise than what any independent source states,
OR if any source uses estimate/upper_bound/vague_magnitude for a figure stated as fact in the claim."""


def _call_gemini(sub_claim_text: str, sub_claim_type: str, sources: list) -> dict:
    sources_summary = [
        {
            "url": s["url"][:80],
            "what_it_says": s["what_claim_does_source_make"],
            "is_original": s["is_original_or_citing"] == "original",
            "precision_level": s["precision_level"],
            "precision_match": s["precision_match"],
            "nli_label": s["nli_label"],
            "confidence": round(s["confidence"], 2),
        }
        for s in sources
    ]

    client = genai.Client(
        vertexai=True,
        project=Config.GCP_PROJECT,
        location=Config.GCP_LOCATION,
    )

    prompt = ASSESS_PROMPT.format(
        sub_claim_text=sub_claim_text,
        sub_claim_type=sub_claim_type,
        sources_json=json.dumps(sources_summary, indent=2),
    )

    models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    for model_name in models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            parsed = json.loads(response.text)
            valid_verdicts = {"SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_VERIFIED", "REFUTED", "UNCERTAIN"}
            if parsed.get("verdict") not in valid_verdicts:
                parsed["verdict"] = "UNCERTAIN"
            return parsed
        except json.JSONDecodeError:
            continue
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            break

    return {"verdict": "UNCERTAIN", "reasoning": "Assessment failed — Gemini unavailable.", "precision_gap": False}


async def assess_sub_claim(state: FactCheckState) -> dict:
    idx = state["current_sub_claim_index"]
    sub_claims = state.get("sub_claims", [])
    independent = state.get("current_independent", [])

    if not sub_claims or idx >= len(sub_claims):
        return {
            "sub_claim_results": state.get("sub_claim_results", []),
            "current_sub_claim_index": idx + 1,
            "current_raw_sources": [],
            "current_investigated": [],
            "current_independent": [],
            "retry_count": 0,
        }

    sub_claim = sub_claims[idx]
    print(f"\n[assess_sub_claim] '{sub_claim['text'][:60]}' — {len(independent)} independent sources")

    loop = asyncio.get_running_loop()
    assessment = await loop.run_in_executor(
        None, _call_gemini, sub_claim["text"], sub_claim["type"], independent
    )

    result = SubClaimResult(
        sub_claim=sub_claim,
        all_sources=state.get("current_investigated", []),
        independent_sources=independent,
        verdict=assessment["verdict"],
        reasoning=assessment["reasoning"],
        precision_gap=bool(assessment.get("precision_gap", False)),
    )

    print(f"  [assess_sub_claim] verdict={result['verdict']} precision_gap={result['precision_gap']}")
    print(f"  reasoning: {result['reasoning'][:120]}")

    return {
        "sub_claim_results": state.get("sub_claim_results", []) + [result],
        "current_sub_claim_index": idx + 1,
        "current_raw_sources": [],
        "current_investigated": [],
        "current_independent": [],
        "retry_count": 0,
    }
