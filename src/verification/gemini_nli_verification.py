"""
gemini_nli_verification.py
--------------------------
Batch NLI verification using Google Gemini.

verify_batch(claim, sources) -> List[{label, confidence, reason, relevant}]

Fixes applied over the original Groq verifier:
  1. Simplified, non-contradictory prompt (~300 words vs 800)
  2. JSON output mode  — Gemini returns valid JSON directly
  3. Superlative handling via clear prompt rules, no brittle regex post-processing
  4. Specific exception handling (JSON, rate-limit, network, unexpected)
  5. Truncation recovery is logged so failures are visible
  6. Confidence scores are never overridden with hardcoded values
  7. Proper text sanitization (no wholesale backslash removal)
  8. Context-budget-aware snippet limits
  9. IRRELEVANT is a first-class label — relevant flag derived from it, not a separate model output
"""

import json
import os
import time
import random
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Prompt — deliberately concise, non-contradictory
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a fact-checking assistant specialising in Sri Lankan political, economic, and social claims.

Classify each numbered evidence source using EXACTLY ONE of four labels relative to the given claim.

DEFINITIONS:
  SUPPORTED   — The source contains verified factual evidence the claim is TRUE.
                Qualifies: court ruling, official government investigation, authoritative statistics,
                OR an established fact-checker (AFP Fact Check, BBC Verify, Reuters Fact Check,
                FactCrescendo, Newschecker, Alt News) explicitly confirming the claim.

  REFUTED     — The source contains verified factual evidence the claim is FALSE.
                Qualifies: court ruling or official report directly contradicting the claim,
                established fact-checker explicitly labelling it false/debunked,
                OR a confirmed alternative cause that contradicts a causal claim
                (e.g. "died of natural causes" refutes "was killed by").

  NEUTRAL     — The source addresses the claim's specific question but stops short of
                confirming or denying it. Use NEUTRAL when the source directly engages
                with what the claim asks but gives an inconclusive answer:
                accusations, allegations, opinions, or the underlying event is confirmed
                but a key qualifier (e.g. "first ever") is not addressed.
                Do NOT use NEUTRAL as a default when the source simply does not mention
                what the claim asks about.

  IRRELEVANT  — The source is about a DIFFERENT question than the claim, even if it shares
                the same general domain or subject matter.

                Use IRRELEVANT when the source answers a different question, for example:
                • Claim asks about enforcement/legal action → source explains how permits
                  are issued or the regulatory framework (different question)
                • Claim asks about Event A → source covers related Event B in the same field
                • Source is background/context that has no bearing on whether the claim
                  is true or false

                The decision test — ask yourself:
                  "Could reading the full source help confirm or refute this specific claim?"
                  YES, even partially → NEUTRAL
                  NO, it answers a different question entirely → IRRELEVANT

RULES — apply in this exact order:

1. ACCUSATIONS = NEUTRAL.
   "X claims / alleges / says that Y did Z" is evidence only that X made the claim.
   Classify as NEUTRAL regardless of how senior or credible the accuser is.

2. SUPERLATIVE CLAIMS (claim contains "first", "only", "never", "largest", "youngest", etc.):
   a. Source says "first in N years" / "Nth time" / "since [year]" but claim says "first ever"
      → REFUTED. The superlative contradiction wins even if the event itself is confirmed.
   b. Source confirms the event but says nothing about the superlative qualifier
      → NEUTRAL. Do NOT mark SUPPORTED just because the underlying event is confirmed.
   c. Source explicitly confirms the superlative ("first female", "first in history",
      "inaugural", "first time ever") → eligible for SUPPORTED.

3. HISTORICAL FACTS.
   A later event (replacement, successor appointment) does NOT undo a past record.
   "X was replaced by Y" when the claim is "X was the first Z appointed" → NEUTRAL, not REFUTED.
   Only REFUTE if the source directly contradicts the original fact
   (e.g. "X was never appointed" or "the appointment was cancelled").

4. TIME PERIODS.
   A source from a different time period is NEUTRAL if it still addresses the claim's
   specific question (e.g. confirms the same type of action occurred, even if not at
   the exact time the claim implies). It is IRRELEVANT if it only covers background
   context that does not bear on whether the claim is true or false.

OTHER:
  "Rs." = Sri Lankan Rupees. Use today's date (provided in the user message) for temporal reasoning.

REASON FIELD RULES:
  - The reason for index N must describe ONLY the text shown under [N] above.
  - Begin with a verbatim quote (3-8 words) copied exactly from that source's quoted text.
  - NEVER mention any other source index in a reason (no "Source [2]", no "[4] states", nothing).
  - NEVER invent a date, quote, or fact not present in that source's text.
  - One sentence total.

Return ONLY a valid JSON array — no markdown, no text outside the array.
Format: [{"index": <int>, "label": "SUPPORTED"|"REFUTED"|"NEUTRAL"|"IRRELEVANT",
           "confidence": <float 0.0-1.0>, "reason": "<one sentence>"}, ...]"""


# ---------------------------------------------------------------------------
# Verifier class
# ---------------------------------------------------------------------------

class GeminiNLIVerifier:
    """
    Batch NLI verifier backed by Google Gemini.

    Parameters
    ----------
    api_key      : Gemini API key. Falls back to GEMINI_API_KEY env var.
    model_names  : Ordered list of Gemini model IDs to try (rate-limit fallback).
    max_retries  : Per-model retry attempts on rate-limit errors.
    """

    DEFAULT_MODELS = [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-lite",
        "models/gemini-3-flash-preview",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        max_retries: int = 3,
    ):
        if not _GENAI_AVAILABLE:
            raise ImportError(
                "google-genai package not installed. Run: pip install google-genai"
            )

        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set GEMINI_API_KEY."
            )

        self.client = genai.Client(api_key=resolved_key)
        self.model_names = model_names or self.DEFAULT_MODELS
        self.max_retries = max_retries
        self._model_usage: Dict[str, int] = {m: 0 for m in self.model_names}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def verify_batch(self, claim: str, sources: List[Dict]) -> List[Dict]:
        """
        Verify all sources against the claim in one Gemini call.

        Returns
        -------
        List of dicts in the same order as `sources`:
            [{'label': 'SUPPORTED'|'REFUTED'|'NEUTRAL',
              'confidence': float,
              'reason': str,
              'relevant': bool}, ...]
        Returns [] on failure so the caller can fall back to DeBERTa.
        """
        if not sources:
            return []

        user_prompt = self._build_user_prompt(claim, sources)

        print("\n" + "=" * 70)
        print("GEMINI NLI VERIFIER — FULL PROMPT")
        print("=" * 70)
        print(f"SYSTEM:\n{SYSTEM_PROMPT}")
        print(f"\nUSER:\n{user_prompt}")
        print("=" * 70 + "\n")

        try:
            raw = self._call_gemini(user_prompt)
        except Exception as e:
            print(f"  [GeminiNLI] All models failed: {e}")
            return []

        return self._parse_response(raw, sources)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_user_prompt(self, claim: str, sources: List[Dict]) -> str:
        """Build the user-turn prompt with date, claim, and numbered sources."""
        # Dynamic snippet limit: keep total source text under ~60k chars
        # (Gemini 2.5 Flash has 1M token context, so this is very conservative)
        max_snippet = max(600, 8000 // max(len(sources), 1))

        source_lines = []
        for i, src in enumerate(sources, 1):
            snippet = _sanitize(src.get("snippet") or src.get("content") or "", max_snippet)
            title   = _sanitize(src.get("title") or "", 120)
            date    = src.get("date") or src.get("date_raw") or ""
            domain  = (src.get("source") or src.get("link") or "")[:60]
            date_str = f" [{date}]" if date else ""
            source_lines.append(
                f"[{i}] {domain}{date_str} — {title}\n    \"{snippet}\""
            )

        today = datetime.now().strftime("%B %d, %Y")
        return (
            f"Today's date: {today}\n"
            f"Claim: \"{claim}\"\n\n"
            f"Sources:\n" + "\n\n".join(source_lines) +
            "\n\nReturn the JSON array now."
        )

    # ------------------------------------------------------------------
    # API call with model fallback and rate-limit retry
    # ------------------------------------------------------------------

    def _call_gemini(self, user_prompt: str) -> str:
        """
        Call Gemini with model fallback and exponential backoff.
        Returns the raw text response.
        Raises Exception if all models/retries are exhausted.
        """
        last_error: Optional[Exception] = None

        for model_name in self.model_names:
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=user_prompt,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.0,
                            max_output_tokens=8192,
                            response_mime_type="application/json",
                        ),
                    )
                    self._model_usage[model_name] = self._model_usage.get(model_name, 0) + 1
                    print(f"  [GeminiNLI] Response from {model_name.split('/')[-1]}")
                    return response.text

                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    err_str = str(e)
                    is_daily_quota = (
                        "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                    ) and (
                        "quota" in err_str.lower()
                        or "daily" in err_str.lower()
                        or "per day" in err_str.lower()
                    )
                    is_rate_limit = (
                        ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str)
                        and not is_daily_quota
                    )
                    is_connection_drop = (
                        "RemoteProtocolError" in err_str
                        or "Server disconnected" in err_str
                        or "ConnectionError" in err_str
                        or "ConnectTimeout" in err_str
                    )

                    if is_daily_quota:
                        # RPD exhausted — retrying won't help, move to next model immediately
                        print(f"  [GeminiNLI] {model_name.split('/')[-1]} daily quota exhausted, "
                              f"switching model")
                        break

                    if (is_rate_limit or is_connection_drop) and attempt < self.max_retries:
                        if is_rate_limit:
                            delay = self._parse_retry_delay(err_str) or (2 ** attempt + random.uniform(0, 1))
                            reason = "rate limit"
                        else:
                            delay = 3.0 + attempt * 2 + random.uniform(0, 1)
                            reason = "connection drop"
                        print(f"  [GeminiNLI] {reason} on {model_name.split('/')[-1]}, "
                              f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(delay)
                        continue

                    # Permanent error or retries exhausted — try next model
                    reason = f"error: {type(e).__name__}"
                    print(f"  [GeminiNLI] {model_name.split('/')[-1]} failed ({reason}), trying next model")
                    break

        raise Exception(f"All Gemini models failed. Last error: {last_error}")

    @staticmethod
    def _parse_retry_delay(error_message: str) -> Optional[float]:
        """Extract retry-after seconds from a Gemini rate-limit error string."""
        import re
        match = re.search(r"retry[_\s]after[:\s]+(\d+(?:\.\d+)?)", error_message, re.IGNORECASE)
        return float(match.group(1)) if match else None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str, sources: List[Dict]) -> List[Dict]:
        """
        Parse the JSON array returned by Gemini into a normalised result list.
        Falls back gracefully if JSON is malformed or truncated.
        """
        # Strip accidental markdown fences (shouldn't happen with response_mime_type=json)
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        # Parse — with logged truncation recovery
        parsed: Optional[list] = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as primary_err:
            # Attempt recovery: close the array after the last complete object
            last_brace = raw.rfind("}")
            if last_brace != -1:
                candidate = raw[: last_brace + 1] + "]"
                try:
                    parsed = json.loads(candidate)
                    recovered = len(parsed)
                    missed = len(sources) - recovered
                    print(
                        f"  [GeminiNLI] WARNING: Output was truncated — recovered {recovered} results, "
                        f"{missed} source(s) defaulted to NEUTRAL"
                    )
                except json.JSONDecodeError:
                    pass

            if parsed is None:
                print(
                    f"  [GeminiNLI] JSON parse failed — returning [] for DeBERTa fallback.\n"
                    f"    Primary error: {primary_err}\n"
                    f"    Raw output (first 300 chars): {raw[:300]}"
                )
                return []

        # Build a 0-based index map from the parsed list
        result_map: Dict[int, Dict] = {}
        for item in parsed:
            try:
                idx = int(item["index"]) - 1  # convert 1-based to 0-based
            except (KeyError, ValueError, TypeError):
                continue

            label = str(item.get("label", "NEUTRAL")).upper()
            if label not in ("SUPPORTED", "REFUTED", "NEUTRAL", "IRRELEVANT"):
                label = "NEUTRAL"

            try:
                confidence = float(item.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]
            except (TypeError, ValueError):
                confidence = 0.5

            result_map[idx] = {
                "label":      label,
                "confidence": confidence,
                "reason":     str(item.get("reason", "")),
                "relevant":   label != "IRRELEVANT",  # derived — no longer a model output
            }

        # Reconstruct list in original source order; missing entries → NEUTRAL
        results = []
        for i in range(len(sources)):
            if i in result_map:
                results.append(result_map[i])
            else:
                results.append({
                    "label":      "NEUTRAL",
                    "confidence": 0.5,
                    "reason":     "No response from model (truncation or parse error)",
                    "relevant":   True,
                })

        print(f"  [GeminiNLI] Batch complete — {len(results)} verdicts from 1 API call")
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(text: str, max_len: int) -> str:
    """
    Clean source text for safe inclusion in the prompt.

    - Removes non-BMP / emoji characters (can confuse JSON generation)
    - Replaces smart quotes and typographic characters with ASCII equivalents
    - Does NOT remove backslashes (they are valid in source text)
    - Truncates to max_len characters
    """
    # Drop emoji and surrogate characters
    text = "".join(
        c for c in text
        if unicodedata.category(c) not in ("So", "Cs")
    )
    # Normalise typographic characters to ASCII equivalents
    replacements = {
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "\u2013": "-", "\u2014": "-",
        "\u2026": "...", "\u00a0": " ",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    # Replace literal double-quotes with single-quotes so they don't break
    # the JSON that Gemini generates (it may echo snippets into reason fields)
    text = text.replace('"', "'")
    return text[:max_len].strip()
