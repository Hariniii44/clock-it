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

Other rules:
  • "Rs." means Sri Lankan Rupees.
  • If the source covers a different time period than the claim, treat as NEUTRAL.
  • If the snippet cuts off before the key fact, treat as NEUTRAL not REFUTED.

Return ONLY a valid JSON array. No markdown, no explanation outside the array.
Each element: {"index": <int>, "label": "SUPPORTED"|"REFUTED"|"NEUTRAL",
               "confidence": <float 0.0-1.0>, "reason": "<one sentence>"}"""


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
                conf  = float(item.get('confidence', 0.5))
                reason = item.get('reason', '')
                result_map[idx] = {'label': label, 'confidence': conf, 'reason': reason}

            # Build output list in original source order
            results = []
            for i in range(len(sources)):
                if i in result_map:
                    results.append(result_map[i])
                else:
                    results.append({'label': 'NEUTRAL', 'confidence': 0.5, 'reason': 'No response'})

            print(f"  [GroqVerifier] Batch complete — {len(results)} verdicts from 1 API call")
            return results

        except Exception as e:
            print(f"  [GroqVerifier] Batch verification failed: {e}")
            return []
