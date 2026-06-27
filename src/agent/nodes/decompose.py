import asyncio
import json
from typing import List

from google import genai
from google.genai import types as genai_types

from config import Config
from ..state import FactCheckState, SubClaim


SYSTEM_PROMPT = """You are a fact-checking analyst specialising in Sri Lankan political, economic, and social claims.
Your job is to break a claim into its smallest independently verifiable sub-claims."""

DECOMPOSE_PROMPT = """Break the following claim into its smallest independently verifiable sub-claims.

Claim: {claim}

For each sub-claim identify:
- text: the full assertion being made — a complete sentence, not just a fragment.
    Good: "approximately 40,000 people were killed"
    Bad:  "40,000"
- type: classify as exactly one of:
    "number"      — tests a specific figure, statistic, or count
    "category"    — tests what kind of entity is involved (e.g. civilian vs combatant, minister vs MP)
    "geography"   — tests where something happened or applies
    "attribution" — tests who did something or who is responsible
    "event"       — tests whether something happened at all (no sub-type above fits better)
- search_query: the single best web search query to find independent evidence for this sub-claim.
    Make the query specific and direct — include the key entities, numbers, or location.
    Do NOT make it a question. Write it as a search string.

Rules:
- Split only on MEANINGFUL distinctions. "40,000 Tamil civilians were killed in Mullivaikkal"
  should produce sub-claims like: the specific count (number), the identity of victims as
  Tamil civilians (one single category sub-claim — do NOT split "Tamil" and "civilians"
  into separate sub-claims), and the location (geography).
- Keep compound descriptors TOGETHER. "Tamil civilians" is ONE category sub-claim.
  "Tamil" alone and "civilians" alone are not independently useful to verify.
- Do NOT create trivially confirmable sub-claims. If the claim is about 40,000 deaths,
  do NOT add a sub-claim "people were killed" — that adds nothing to fact-checking.
  Every sub-claim must be a meaningful, potentially falsifiable assertion.
- For number sub-claims: include the geographic or temporal context in the text.
  Good: "approximately 40,000 people were killed in Mullivaikkal"
  Bad:  "approximately 40,000 people were killed" (stripped of context)
- If the claim is simple and indivisible, return it as a single sub-claim of type "event".
- Maximum 5 sub-claims. If there are more, merge the least important ones.
- Sub-claims about the same fact from different angles count as one — do not duplicate.

Return ONLY a valid JSON array. No markdown, no explanation.
Format: [{{"text": "...", "type": "...", "search_query": "..."}}]"""


def _call_gemini(claim: str) -> List[SubClaim]:
    client = genai.Client(
        vertexai=True,
        project=Config.GCP_PROJECT,
        location=Config.GCP_LOCATION,
    )

    models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=DECOMPOSE_PROMPT.format(claim=claim),
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()
            parsed = json.loads(raw)

            # Validate each item has required fields
            sub_claims: List[SubClaim] = []
            valid_types = {"number", "category", "geography", "attribution", "event"}
            for item in parsed:
                if not all(k in item for k in ("text", "type", "search_query")):
                    continue
                if item["type"] not in valid_types:
                    item["type"] = "event"
                sub_claims.append(SubClaim(
                    text=str(item["text"]),
                    type=str(item["type"]),
                    search_query=str(item["search_query"]),
                ))

            print(f"  [decompose_claim] {len(sub_claims)} sub-claims from {model_name}")
            for sc in sub_claims:
                print(f"    [{sc['type']}] {sc['text']}")
            return sub_claims

        except json.JSONDecodeError as e:
            print(f"  [decompose_claim] JSON parse error from {model_name}: {e}")
            continue
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print(f"  [decompose_claim] Rate limit on {model_name}, trying next")
                continue
            print(f"  [decompose_claim] Error from {model_name}: {e}")
            continue

    return []


async def decompose_claim(state: FactCheckState) -> dict:
    claim = state["claim"]
    print(f"\n[decompose_claim] '{claim}'")

    loop = asyncio.get_running_loop()
    sub_claims = await loop.run_in_executor(None, _call_gemini, claim)

    if not sub_claims:
        # Treat the whole claim as a single event sub-claim so the graph can still run
        print("  [decompose_claim] Gemini failed — falling back to single sub-claim")
        sub_claims = [SubClaim(text=claim, type="event", search_query=claim)]

    return {
        "sub_claims": sub_claims,
        "current_sub_claim_index": 0,
        "sub_claim_results": [],
        "retry_count": 0,
        "current_raw_sources": [],
        "current_investigated": [],
        "current_independent": [],
    }
