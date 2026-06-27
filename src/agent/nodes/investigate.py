import asyncio
import json
from typing import List, Optional

from google import genai
from google.genai import types as genai_types

from config import Config
from ..state import FactCheckState, InvestigatedSource, RawSource


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an investigative fact-checker. Analyse a source document and answer
specific questions about what it claims and how it relates to a sub-claim being investigated."""

INVESTIGATE_PROMPT = """Analyse the following source document in relation to a specific sub-claim.

SUB-CLAIM BEING CHECKED: {sub_claim_text}
SUB-CLAIM TYPE: {sub_claim_type}

SOURCE URL: {url}
SOURCE TEXT:
{full_text}

Answer ALL of the following in JSON:

1. what_claim_does_source_make (string):
   What exact claim does this source make about this topic?
   Quote the specific relevant sentence if possible. Keep under 200 characters.

2. is_original_or_citing (string — MUST be "original" or "citing"):
   This question is about the SPECIFIC CLAIM OR FIGURE being checked, not the source in general.
   - "original": this source independently generated the specific data or claim being checked.
     It conducted its own investigation, collected its own data, or made its own primary finding.
     Examples: the UN Panel report that first produced the 40,000 estimate; a government census;
     a court judgment; a hospital that counted bodies; a journalist who witnessed events directly.
   - "citing": this source is repeating or referencing a specific figure or claim that
     originated elsewhere — even if the source is otherwise reputable or did independent work
     on other aspects of the story.
     Examples: a BBC article saying "according to the UN, 40,000 died" (citing for the number,
     even though BBC did original reporting on other things); Wikipedia; a blog; an advocacy
     group repeating a UN figure; any source that attributes the specific claim to another source.
   KEY TEST: ask "did this source produce this specific figure/claim itself, or did it get it
   from someone else?" If it got it from someone else, it is "citing".

3. citation_url (string or null):
   If "citing" — the URL of the original source being referenced, if the text contains one.
   Extract the actual hyperlink or mentioned URL. null if not found or if "original".

4. citation_name (string or null):
   If "citing" — the name of the original source (e.g. "UN Panel of Experts 2011",
   "OISL Report 2015", "Darusman Report"). null if not found or if "original".

5. precision_level (string — MUST be one of these exactly):
   How precisely does this source state the claim?
   - "exact_count"     : gives an exact verified number as confirmed fact ("40,000 died")
   - "estimate"        : explicitly says it is an estimate ("estimated 40,000")
   - "upper_bound"     : gives a maximum possible figure ("up to 40,000", "as many as 40,000")
   - "vague_magnitude" : says "thousands" or "tens of thousands" without a specific number
   - "recognition"     : political or memorial recognition language, not a factual count
   - "no_number"       : confirms the event but gives no figure at all

6. precision_match (boolean):
   Does the source's precision level actually match the precision in the sub-claim?
   If the sub-claim says "40,000" but the source says "up to 40,000" or "tens of thousands",
   this is FALSE. Only TRUE if the precision levels genuinely agree.

7. nli_label (string — MUST be "SUPPORTED", "REFUTED", or "NEUTRAL"):
   - SUPPORTED: source directly confirms the sub-claim AND precision_match is true.
     If precision_match is false, do NOT use SUPPORTED — use NEUTRAL instead.
     For number sub-claims: "up to 40,000", "approximately 40,000", "40,000 to 70,000",
     "tens of thousands" do NOT support a claim of "40,000" — they are NEUTRAL.
   - REFUTED:   source directly contradicts the sub-claim with evidence
   - NEUTRAL:   source is about the topic but does not confirm the sub-claim with
     matching precision. Use this when the source reports a related but imprecise figure.

8. confidence (float 0.0 to 1.0):
   Your confidence in the nli_label assessment.

9. reasoning (string):
   One sentence explaining your assessment, citing specific language from the source.

Return ONLY valid JSON with these exact keys. No markdown.
{{
  "what_claim_does_source_make": "...",
  "is_original_or_citing": "original" or "citing",
  "citation_url": "..." or null,
  "citation_name": "..." or null,
  "precision_level": "...",
  "precision_match": true or false,
  "nli_label": "...",
  "confidence": 0.0,
  "reasoning": "..."
}}"""


# ---------------------------------------------------------------------------
# Page fetching — same 4-strategy pattern as GoogleAIModeRetriever
# ---------------------------------------------------------------------------

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/122.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Referer': 'https://www.google.com/',
}

_MAX_TEXT = 8000  # chars fed to Gemini per source


def _fetch_page_text(url: str) -> str:
    """Fetch full article text. Returns empty string on failure."""
    import trafilatura
    import requests

    def _extract(html: str) -> str:
        text = trafilatura.extract(html, include_comments=False, include_tables=False, no_fallback=False)
        return (text or '').strip()

    clean_url = url.split('#')[0]  # strip text-fragment anchors

    # Strategy 1: AMP version
    try:
        resp = requests.get(clean_url.rstrip('/') + '/amp/', headers=_HEADERS, timeout=6)
        if resp.status_code == 200 and len(resp.text) > 500:
            text = _extract(resp.text)
            if text and len(text) > 200:
                return text[:_MAX_TEXT]
    except Exception:
        pass

    # Strategy 2: requests with browser headers
    try:
        resp = requests.get(clean_url, headers=_HEADERS, timeout=8)
        if resp.status_code == 200:
            text = _extract(resp.text)
            if text and len(text) > 200:
                return text[:_MAX_TEXT]
    except Exception:
        pass

    # Strategy 3: trafilatura built-in fetch
    try:
        downloaded = trafilatura.fetch_url(clean_url)
        if downloaded:
            text = _extract(downloaded)
            if text and len(text) > 200:
                return text[:_MAX_TEXT]
    except Exception:
        pass

    # Strategy 4: Jina AI Reader (handles JS-rendered pages)
    try:
        resp = requests.get(
            f'https://r.jina.ai/{clean_url}',
            headers={'Accept': 'text/plain', 'User-Agent': 'Mozilla/5.0'},
            timeout=12,
        )
        if resp.status_code == 200 and len(resp.text) > 200:
            return resp.text.strip()[:_MAX_TEXT]
    except Exception:
        pass

    return ''


# ---------------------------------------------------------------------------
# Gemini analysis for a single source
# ---------------------------------------------------------------------------

_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

_FALLBACK_ANALYSIS = {
    "what_claim_does_source_make": "Could not analyse — fetch or parse failed.",
    "is_original_or_citing": "original",
    "citation_url": None,
    "citation_name": None,
    "precision_level": "no_number",
    "precision_match": False,
    "nli_label": "NEUTRAL",
    "confidence": 0.1,
    "reasoning": "Source could not be fetched or analysed.",
}


def _analyse_source(
    sub_claim_text: str,
    sub_claim_type: str,
    source: RawSource,
    full_text: str,
) -> Optional[InvestigatedSource]:
    """Synchronous — call inside run_in_executor."""

    # Use snippet as fallback if full text fetch failed
    text_for_gemini = full_text or source.get("snippet", "")
    if not text_for_gemini:
        return None  # nothing to analyse

    client = genai.Client(
        vertexai=True,
        project=Config.GCP_PROJECT,
        location=Config.GCP_LOCATION,
    )

    prompt = INVESTIGATE_PROMPT.format(
        sub_claim_text=sub_claim_text,
        sub_claim_type=sub_claim_type,
        url=source["url"],
        full_text=text_for_gemini[:_MAX_TEXT],
    )

    analysis = None
    for model_name in _MODELS:
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
            # Enforce valid enum values
            if parsed.get("is_original_or_citing") not in ("original", "citing"):
                parsed["is_original_or_citing"] = "original"
            valid_precision = {"exact_count", "estimate", "upper_bound", "vague_magnitude", "recognition", "no_number"}
            if parsed.get("precision_level") not in valid_precision:
                parsed["precision_level"] = "no_number"
            if parsed.get("nli_label") not in ("SUPPORTED", "REFUTED", "NEUTRAL"):
                parsed["nli_label"] = "NEUTRAL"
            analysis = parsed
            break
        except json.JSONDecodeError:
            continue
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                continue
            break

    if analysis is None:
        analysis = _FALLBACK_ANALYSIS.copy()

    return InvestigatedSource(
        url=source["url"],
        title=source.get("title", ""),
        full_text=full_text,
        what_claim_does_source_make=analysis.get("what_claim_does_source_make", ""),
        is_original_or_citing=analysis.get("is_original_or_citing", "original"),
        citation_url=analysis.get("citation_url"),
        citation_name=analysis.get("citation_name"),
        precision_level=analysis.get("precision_level", "no_number"),
        precision_match=bool(analysis.get("precision_match", False)),
        nli_label=analysis.get("nli_label", "NEUTRAL"),
        confidence=float(analysis.get("confidence", 0.1)),
        reasoning=analysis.get("reasoning", ""),
        depth=0,
    )


def _fetch_and_analyse_source(
    sub_claim_text: str,
    sub_claim_type: str,
    source: RawSource,
) -> Optional[InvestigatedSource]:
    """Fetch full page text then analyse. Synchronous — called in thread pool."""
    url = source.get("url", "")
    if not url:
        return None

    full_text = _fetch_page_text(url)
    result = _analyse_source(sub_claim_text, sub_claim_type, source, full_text)

    if result:
        status = "original" if result["is_original_or_citing"] == "original" else f"citing->{result.get('citation_name', '?')}"
        print(f"  [investigate] [{result['nli_label']} {result['confidence']:.2f}] [{status}] [{result['precision_level']}] {url[:60]}")

    return result


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

async def investigate_source(state: FactCheckState) -> dict:
    idx = state["current_sub_claim_index"]
    sub_claims = state.get("sub_claims", [])
    raw_sources = state.get("current_raw_sources", [])

    if not sub_claims or idx >= len(sub_claims) or not raw_sources:
        return {"current_investigated": []}

    sub_claim = sub_claims[idx]
    print(f"\n[investigate_source] '{sub_claim['text'][:60]}' — {len(raw_sources)} sources")

    loop = asyncio.get_running_loop()

    # Run fetch+analyse for each source concurrently via run_in_executor.
    # Each call is independent (fetch + one Gemini call), so they parallelise well.
    semaphore = asyncio.Semaphore(6)  # cap concurrent Gemini calls

    async def _bounded(source: RawSource) -> Optional[InvestigatedSource]:
        async with semaphore:
            return await loop.run_in_executor(
                None, _fetch_and_analyse_source,
                sub_claim["text"], sub_claim["type"], source,
            )

    results = await asyncio.gather(*[_bounded(s) for s in raw_sources], return_exceptions=False)
    investigated = [r for r in results if r is not None]

    print(f"  [investigate_source] {len(investigated)} sources analysed "
          f"({sum(1 for s in investigated if s['nli_label']=='SUPPORTED')} SUPPORTED, "
          f"{sum(1 for s in investigated if s['nli_label']=='REFUTED')} REFUTED, "
          f"{sum(1 for s in investigated if s['nli_label']=='NEUTRAL')} NEUTRAL)")

    return {"current_investigated": investigated}
