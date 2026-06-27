import asyncio
import json

from google import genai
from google.genai import types as genai_types

from config import Config
from ..state import FactCheckState


SYSTEM_PROMPT = """You are a fact-checking analyst. Compose a final verdict from sub-claim assessments."""

SYNTHESISE_PROMPT = """Compose a final fact-check verdict from these sub-claim assessments.

ORIGINAL CLAIM: {claim}

SUB-CLAIM RESULTS:
{results_json}

Rules for final_verdict:
- VERIFIED:          ALL sub-claims are SUPPORTED with no precision gaps
- MOSTLY_TRUE:       most sub-claims SUPPORTED, minor precision gaps or one NOT_VERIFIED
- PARTIALLY_TRUE:    some sub-claims SUPPORTED, some NOT_VERIFIED or PARTIALLY_SUPPORTED
- NOT_VERIFIED:      key sub-claims lack independent confirmation
- UNCERTAIN:         evidence is genuinely scarce, inaccessible, or conflicting
- REFUTED:           key sub-claims are directly REFUTED

Critical rule: if ANY sub-claim has precision_gap=true, the final verdict CANNOT be VERIFIED.
The claim cannot be confirmed to the level of specificity it asserts.

Return JSON:
{{
  "final_verdict": "...",
  "final_confidence": 0.0,
  "final_explanation": "3-4 sentences. State what IS confirmed, what is NOT confirmed,
                        and where exactly the claim exceeds the evidence.",
  "precision_issues": ["list of strings describing precision gaps, empty if none"],
  "geography_issues": ["list of strings describing location mismatches, empty if none"]
}}"""


def _call_gemini(claim: str, results: list) -> dict:
    results_summary = [
        {
            "sub_claim": r["sub_claim"]["text"],
            "type": r["sub_claim"]["type"],
            "verdict": r["verdict"],
            "reasoning": r["reasoning"],
            "precision_gap": r["precision_gap"],
            "independent_sources": len(r["independent_sources"]),
        }
        for r in results
    ]

    client = genai.Client(
        vertexai=True,
        project=Config.GCP_PROJECT,
        location=Config.GCP_LOCATION,
    )

    prompt = SYNTHESISE_PROMPT.format(
        claim=claim,
        results_json=json.dumps(results_summary, indent=2),
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
            valid = {"VERIFIED", "MOSTLY_TRUE", "PARTIALLY_TRUE", "NOT_VERIFIED", "UNCERTAIN", "REFUTED"}
            if parsed.get("final_verdict") not in valid:
                parsed["final_verdict"] = "UNCERTAIN"
            return parsed
        except json.JSONDecodeError:
            continue
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            break

    return {
        "final_verdict": "UNCERTAIN",
        "final_confidence": 0.0,
        "final_explanation": "Synthesis failed — Gemini unavailable.",
        "precision_issues": [],
        "geography_issues": [],
    }


async def synthesise_verdict(state: FactCheckState) -> dict:
    claim = state["claim"]
    results = state.get("sub_claim_results", [])

    print(f"\n[synthesise_verdict] '{claim[:60]}' — {len(results)} sub-claim results")

    if not results:
        return {
            "final_verdict": "UNCERTAIN",
            "final_confidence": 0.0,
            "final_explanation": "No sub-claims could be assessed.",
            "precision_issues": [],
            "geography_issues": [],
        }

    loop = asyncio.get_running_loop()
    synthesis = await loop.run_in_executor(None, _call_gemini, claim, results)

    print(f"  [synthesise_verdict] verdict={synthesis['final_verdict']} confidence={synthesis['final_confidence']}")
    print(f"  explanation: {synthesis['final_explanation'][:120]}")

    return {
        "final_verdict": synthesis["final_verdict"],
        "final_confidence": float(synthesis["final_confidence"]),
        "final_explanation": synthesis["final_explanation"],
        "precision_issues": synthesis.get("precision_issues", []),
        "geography_issues": synthesis.get("geography_issues", []),
    }
