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
  SUPPORTED – The source directly and explicitly confirms the claim is true.
  REFUTED   – The source directly and explicitly contradicts the claim.
  NEUTRAL   – The source does not contain sufficient information to confirm or deny.

Rules:
  • "Rs." means Sri Lankan Rupees.
  • If the source is about a *different* time period (e.g. 2022 or 2024 when the
    claim is about 2026), treat it as NEUTRAL — it describes a separate event.
  • If the source is a truncated snippet that cuts off before the key fact,
    treat it as NEUTRAL rather than REFUTED.
  • Only mark SUPPORTED when the source clearly states the claimed fact.
  • Only mark REFUTED when the source explicitly contradicts the claimed fact.

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

        # Build the source listing for the prompt
        source_lines = []
        for i, src in enumerate(sources, 1):
            snippet = (src.get('snippet') or '')[:1200].strip()
            title   = (src.get('title')   or '')[:120].strip()
            date    = src.get('date', '') or src.get('date_raw', '')
            domain  = src.get('source', src.get('link', ''))[:60]
            date_str = f" [{date}]" if date else ""
            source_lines.append(
                f"[{i}] {domain}{date_str} — {title}\n    \"{snippet}\""
            )

        user_prompt = (
            f'Claim: "{claim}"\n\n'
            f'Sources:\n' + '\n\n'.join(source_lines) +
            '\n\nReturn the JSON array now.'
        )

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
                max_tokens=1024,
            )
            raw = response.choices[0].message.content.strip()

            # Strip accidental markdown fences
            if raw.startswith('```'):
                raw = raw.split('```')[1]
                if raw.startswith('json'):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)

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
