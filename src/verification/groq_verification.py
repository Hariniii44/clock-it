"""
groq_verification.py
--------------------
Batch NLI verification using Groq LLM.

Replaces per-source DeBERTa calls with a single Groq API call that analyses
all evidence sources simultaneously.  This solves three failure modes of the
small DeBERTa model:
  1. Currency notation  ("Rs. 25" ≠ "25 rupees" to a 180M-param model)
  2. List disambiguation ("rice and kottu by Rs. 25" — isolating just kottu)
  3. Implicit dates     ("yesterday (March 11)" needing date context)

Interface: drop-in compatible with ClaimVerifier — returns the same
{label, confidence} dict format per source.
"""

import json
import os
import re
from typing import Dict, List


SYSTEM_PROMPT = """You are a strict fact-checking assistant specialising in Sri Lankan news.

For each numbered evidence source, decide whether it SUPPORTS, REFUTES, or is NEUTRAL
to the given claim.

Definitions:
  SUPPORTED – The source contains VERIFIED FACTUAL EVIDENCE that the claim is true.
              Only qualifies if an official body, court, government investigation, or
              authoritative statistical source has formally established the fact.
  REFUTED   – The source contains VERIFIED FACTUAL EVIDENCE that the claim is false.
              Official clearances, court rulings of innocence, or authoritative
              factual corrections qualify.
  NEUTRAL   – Everything else: insufficient information, allegations, opinions,
              accusations, political statements, or truncated snippets.

THE MOST IMPORTANT RULE — accusations and allegations are ALWAYS NEUTRAL:
  A quote where person X accuses, claims, says, alleges, or charges that Y did Z
  is NOT evidence that Y did Z. It is evidence only that X made an accusation.
  Classify such sources as NEUTRAL regardless of how directly or confidently
  the accusation is stated, and regardless of how senior or credible the accuser is.

Examples (study these carefully):

  Claim: "Ranil Wickremesinghe was involved in the bond scam"

  Source: "Former CBSL Governor Ajith Nivard Cabraal claims Ranil is the brains
           behind the Central Bank bond scam."
  → NEUTRAL  (an individual making an accusation — not a verified finding)

  Source: "Opposition leader Anura Kumara says Ranil Wickremesinghe was the chief
           planner of the bond scam."
  → NEUTRAL  (a politician's allegation — not a verified finding)

  Source: "The Presidential Commission of Inquiry report cleared Prime Minister
           Ranil Wickremesinghe of direct involvement in the bond scam."
  → REFUTED  (official investigation finding)

  Source: "The High Court indicted Ranil Wickremesinghe on charges relating to
           the Central Bank bond fraud."
  → SUPPORTED  (official court proceeding)

FACT-CHECKER EXCEPTION — established fact-checking organisations can qualify as REFUTED:
  If the source is an established fact-checking outlet (AFP Fact Check, Reuters Fact Check,
  BBC Verify, Snopes, FactCrescendo, Newschecker, Alt News, or similar) AND it explicitly
  labels the claim as "false", "bogus", "misleading", "debunked", or "unverified", classify
  as REFUTED. The fact-checker's investigation (contacting sources, checking records, verifying
  documents) constitutes authoritative factual correction even without a court ruling.

  Example:
    Claim: "Politician X was caught accepting a bribe"
    Source (AFP Fact Check): "This claim is false. AFP reporters contacted the politician's
      office and reviewed official documents — no evidence of bribery was found."
    → REFUTED  (authoritative fact-checker explicitly debunking the claim)

ALTERNATIVE CAUSE EXCEPTION — a credible alternative factual explanation refutes causal claims:
  If a claim asserts that person X caused event E (e.g. "killed", "poisoned", "caused the crash"),
  and multiple independent credible sources confirm a different, natural or unrelated cause for E
  (e.g. "died after a prolonged illness", "death by natural causes", "accident"), classify those
  sources as REFUTED. The confirmed alternative cause directly contradicts the claimed cause.

  Example:
    Claim: "Journalist Y was killed by politician Z"
    Source: "Journalist Y, 55, died at the National Hospital Colombo after receiving
             treatment for a prolonged illness." (from a credible news outlet)
    → REFUTED  (confirmed natural cause of death directly contradicts "killed by")

Other rules:
  • "Rs." means Sri Lankan Rupees.
  • If the source covers a different time period than the claim, treat as NEUTRAL.
  • If the snippet cuts off before the key fact, treat as NEUTRAL not REFUTED.
  • HISTORICAL FACTS — a claim about a past event (appointment, election, ruling, record)
    cannot be REFUTED by a source describing a LATER event. If the claim says "X was the
    first Y appointed" and a source says "X was later replaced by Z", that is NEUTRAL —
    it does not undo the original appointment. Only REFUTE if the source directly contradicts
    the original fact (e.g. "X was never appointed" or "the appointment was cancelled").

  Example:
    Claim: "Sri Lanka appointed its first woman CID Director"
    Source: "SSP Abeysekara replaced SSP Muthumala as CID Director."
    → NEUTRAL  (replacement does not negate the historic first appointment)

  • SUPERLATIVE / UNIQUENESS CLAIMS — if the claim contains words like "first", "only",
    "never", "largest", "youngest", "only ever", you MUST evaluate whether the source
    confirms OR contradicts that specific superlative, not just the underlying event.

    A source can SUPPORT the main event but also REFUTE the superlative. In that case,
    classify as REFUTED — factual accuracy of the superlative is part of the claim.

    SUPERLATIVE OVERRIDE RULE — if ANY part of a source contains phrases like:
      "first in X years", "first since [year]", "X-th time", "second time", "third time",
      "in two decades", "in nearly a century", "for the first time since"
    when the claim asserts "first ever" or "first [descriptor]" — that phrase ALONE is
    enough to classify the entire source as REFUTED. Do not let the primary topic of the
    source (e.g., appointment confirmed) override this. The superlative contradiction wins.

  Examples for claim: "Harini Amarasuriya became the first female Prime Minister of Sri Lanka"

    Source: "Harini Amarasuriya was sworn in as Sri Lanka's prime minister, the first
            woman to hold the office in 24 years."
    → REFUTED  ("first in 24 years" directly contradicts "first female PM ever")

    Source: "Sri Lanka's new president reappoints Amarasuriya as PM ... making her the
            first woman to head the national government in 24 years."
    → REFUTED  ("first woman ... in 24 years" directly contradicts "first female PM" —
                even though the rest of the article confirms the appointment, the
                superlative override rule applies: classify as REFUTED)

    Source: "The president selected his ally Harini Amarasuriya as prime minister,
            choosing a woman for the third time in the country's history."
    → REFUTED  ("third time" explicitly means she is NOT the first — superlative override)

    Source: "Sirimavo Bandaranaike became the world's first female prime minister on
            July 21, 1960."
    → REFUTED  (establishes that a woman held the PM role before Harini)

    Source: "Harini Amarasuriya was sworn in as prime minister on September 24, 2024."
    → NEUTRAL  (confirms appointment but says NOTHING about whether she is the first —
                do NOT classify as SUPPORTED; the superlative is unverified)

    Source (from parliamentary records): "Cabinet: Prime Minister Dr. Harini Amarasuriya,
            Minister of Education..."
    → NEUTRAL  (lists her as PM but does not verify "first female" — do not classify
                as SUPPORTED just because it confirms the appointment)

    Claim: "Sri Lanka appointed its first female CID Director"
    Source: "SSP Muthumala was appointed Sri Lanka's first female CID Director."
    → SUPPORTED  (directly and explicitly confirms the superlative "first female")

Return ONLY a valid JSON array. No markdown, no explanation outside the array.
Each element: {"index": <int>, "label": "SUPPORTED"|"REFUTED"|"NEUTRAL",
               "confidence": <float 0.0-1.0 representing YOUR CERTAINTY in the classification, e.g. 0.9 if clearly NEUTRAL, NOT how much the source supports the claim>,
               "reason": "<one sentence>",
               "relevant": <true|false — ONLY mark false if the source is about completely different entities, people, or topics with NO connection to the claim (e.g. claim is about NPP reforms but source is about a mosque attack). A source that covers the same topic/entity but a different time period (e.g. claim says "100 days" but source covers "one year") is still relevant=true — it provides useful context even if the timeframe differs. When in doubt, default to true.>}"""


class GroqVerifier:
    """
    Single-call batch verifier using Groq.
    Falls back to empty list on any API / parsing failure so the pipeline
    can fall through to DeBERTa.
    """

    def __init__(self, groq_api_key: str = ''):
        self.api_key = groq_api_key or os.getenv('GROQ_API_KEY', '')
        self.model   = 'llama-3.3-70b-versatile'

    # ------------------------------------------------------------------

    def verify_batch(self, claim: str, sources: List[Dict]) -> List[Dict]:
        """
        Verify all sources against the claim in one Groq call.

        Parameters
        ----------
        claim   : original claim string (question or statement)
        sources : list of evidence dicts, each with at least 'snippet', 'link', 'title'

        Returns
        -------
        List of dicts in the same order as `sources`:
            [{'label': 'SUPPORTED'|'REFUTED'|'NEUTRAL', 'confidence': float}, ...]
        Returns [] on failure so the caller can fall back to DeBERTa.
        """
        if not self.api_key:
            print("  [GroqVerifier] No API key — skipping batch verification")
            return []

        if not sources:
            return []

        def _sanitize(text: str, max_len: int) -> str:
            """Strip characters that can break JSON output from Groq."""
            import unicodedata
            # Remove emoji and non-BMP characters
            text = ''.join(c for c in text if unicodedata.category(c) not in ('So', 'Cs'))
            # Replace curly quotes, em-dashes, and other fancy punctuation with ASCII equivalents
            replacements = {
                '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
                '\u2013': '-', '\u2014': '-', '\u2026': '...', '\u00a0': ' ',
            }
            for orig, repl in replacements.items():
                text = text.replace(orig, repl)
            # Replace double-quotes with single-quotes so the plain-text prompt
            # doesn't confuse Groq's JSON generation when it copies snippet text
            # into its "reason" fields.
            text = text.replace('\\', '').replace('"', "'")
            return text[:max_len].strip()

        # Build the source listing for the prompt
        source_lines = []
        for i, src in enumerate(sources, 1):
            snippet  = _sanitize(src.get('snippet') or '', 1200)
            title    = _sanitize(src.get('title')   or '', 120)
            date     = src.get('date', '') or src.get('date_raw', '')
            domain   = src.get('source', src.get('link', ''))[:60]
            date_str = f" [{date}]" if date else ""
            source_lines.append(
                f"[{i}] {domain}{date_str} — {title}\n    \"{snippet}\""
            )

        from datetime import datetime
        today = datetime.now().strftime('%B %d, %Y')

        user_prompt = (
            f'Today\'s date: {today}\n'
            f'Claim: "{claim}"\n\n'
            f'Sources:\n' + '\n\n'.join(source_lines) +
            '\n\nReturn the JSON array now.'
        )

        # --- DEBUG: print full prompt being sent to Groq ---
        print("\n" + "="*70)
        print("GROQ VERIFIER — FULL PROMPT")
        print("="*70)
        print(f"SYSTEM:\n{SYSTEM_PROMPT}")
        print(f"\nUSER:\n{user_prompt}")
        print("="*70 + "\n")
        # --- END DEBUG ---

        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': user_prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content.strip()

            # Strip accidental markdown fences
            if raw.startswith('```'):
                raw = raw.split('```')[1]
                if raw.startswith('json'):
                    raw = raw[4:]
            raw = raw.strip()

            # If the output was truncated, recover by closing the array after
            # the last complete object so json.loads doesn't fail outright.
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                last_brace = raw.rfind('}')
                if last_brace != -1:
                    raw = raw[:last_brace + 1] + ']'
                    parsed = json.loads(raw)
                else:
                    raise

            # Normalise and reorder to match input order
            result_map = {}
            for item in parsed:
                idx  = int(item['index']) - 1          # convert to 0-based
                label = item.get('label', 'NEUTRAL').upper()
                if label not in ('SUPPORTED', 'REFUTED', 'NEUTRAL'):
                    label = 'NEUTRAL'
                conf     = float(item.get('confidence', 0.5))
                reason   = item.get('reason', '')
                relevant = bool(item.get('relevant', True))  # default True so unknown → kept
                result_map[idx] = {'label': label, 'confidence': conf, 'reason': reason, 'relevant': relevant}

            # Build output list in original source order
            results = []
            for i in range(len(sources)):
                if i in result_map:
                    results.append(result_map[i])
                else:
                    results.append({'label': 'NEUTRAL', 'confidence': 0.5, 'reason': 'No response', 'relevant': True})

            results = self._apply_superlative_overrides(claim, results, sources)
            print(f"  [GroqVerifier] Batch complete — {len(results)} verdicts from 1 API call")
            return results

        except Exception as e:
            print(f"  [GroqVerifier] Batch verification failed: {e}")
            return []

    # ------------------------------------------------------------------

    @staticmethod
    def _apply_superlative_overrides(claim: str, results: List[Dict], sources: List[Dict]) -> List[Dict]:
        """
        Deterministic post-processing layer applied after Groq's NLI output.

        Motivation: Groq's temperature-0 output is non-deterministic on compound
        superlative claims (e.g. "first female PM") because long source articles
        have a strong primary-topic signal (appointment confirmed) that sometimes
        drowns out a buried superlative-contradicting phrase ("in 24 years",
        "third time"). Two deterministic override rules address this:

        Override 1 — Contradiction detection (any label → REFUTED):
            If the claim contains a superlative ("first", "only", "never") and the
            source text contains an explicit phrase that contradicts "first ever"
            ("first X in N years", "third time", "since YYYY", etc.), force REFUTED.

        Override 2 — False SUPPORTED detection (SUPPORTED → NEUTRAL):
            If the claim asserts a superlative and the source is SUPPORTED but
            the source text does not contain any "first ever" confirmation
            ("first female", "first time ever", "inaugural", etc.), downgrade
            to NEUTRAL. Sources confirming the underlying event (e.g. "she was
            sworn in as PM") without confirming the "first" qualifier should not
            count as evidence that the superlative is true.
        """
        claim_lower = claim.lower()
        superlative_triggers = ['first', 'only', 'never', 'largest', 'youngest', 'only ever']
        if not any(w in claim_lower for w in superlative_triggers):
            return results

        # ── Override 1 patterns ──────────────────────────────────────────────
        # Each pattern catches an explicit contradiction of "first ever":
        #   • "first/only X in N years"            → not first ever
        #   • "for the Nth time / second/third time"→ not first ever
        #   • "since YYYY" (historical year)        → someone came before
        #   • "world's first" in a source about a DIFFERENT person → predecessor exists
        contradiction_patterns = [
            re.compile(r'\bfirst\b.{3,100}?\bin\s+\d+\s+years?\b',  re.IGNORECASE),
            re.compile(r'\bonly\b.{3,80}?\bin\s+\d+\s+years?\b',    re.IGNORECASE),
            re.compile(r'\b(?:second|third|fourth|fifth)\s+time\b',  re.IGNORECASE),
            re.compile(r'\bfor\s+the\s+\w+\s+time\s+in\b',          re.IGNORECASE),
            re.compile(r'\bsince\s+(?:19|20)\d{2}\b',               re.IGNORECASE),
            re.compile(r"\bworld'?s?\s+first\b",                     re.IGNORECASE),
            re.compile(r'\bin\s+(?:nearly\s+)?\d+\s+(?:years?|decades?)\b', re.IGNORECASE),
        ]

        # ── Override 2 patterns ──────────────────────────────────────────────
        # Phrases that confirm a "first ever" superlative (any of these present →
        # Groq's SUPPORTED verdict is reasonable; leave it alone).
        first_ever_confirmers = [
            re.compile(r'\bfirst\s+(?:ever|female|woman|man|person|time\b)', re.IGNORECASE),
            re.compile(r'\bfirst\s+in\s+(?:the\s+)?(?:history|country|nation)',    re.IGNORECASE),
            re.compile(r'\bhistoric(?:al)?\s+first\b',                             re.IGNORECASE),
            re.compile(r'\binaugural\b',                                            re.IGNORECASE),
            re.compile(r'\bpioneer(?:ing)?\b',                                      re.IGNORECASE),
        ]

        overrides_applied = 0
        for i, result in enumerate(results):
            src_idx = i  # results are already in source order (0-based)
            if src_idx >= len(sources):
                continue

            src = sources[src_idx]
            text = (
                src.get('snippet', '')
                or src.get('content', '')
                or src.get('passage', '')
                or ''
            )

            # Override 1: explicit contradiction → REFUTED
            overridden = False
            for pat in contradiction_patterns:
                if pat.search(text):
                    if result['label'] != 'REFUTED':
                        result['label']      = 'REFUTED'
                        result['confidence'] = max(result.get('confidence', 0.85), 0.85)
                        result['reason']     = '[Override-1] Superlative contradiction phrase detected in source text.'
                        overrides_applied   += 1
                    overridden = True
                    break

            # Override 2: SUPPORTED but no "first ever" confirmation → NEUTRAL
            if not overridden and result['label'] == 'SUPPORTED':
                has_first_ever = any(pat.search(text) for pat in first_ever_confirmers)
                if not has_first_ever:
                    result['label']      = 'NEUTRAL'
                    result['confidence'] = max(result.get('confidence', 0.85), 0.85)
                    result['reason']     = '[Override-2] Source confirms event but contains no explicit superlative confirmation.'
                    overrides_applied   += 1

        if overrides_applied:
            print(f"  [GroqVerifier] Superlative overrides applied to {overrides_applied} source(s).")
        return results
