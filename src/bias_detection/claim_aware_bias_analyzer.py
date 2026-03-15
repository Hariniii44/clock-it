"""
claim_aware_bias_analyzer.py
----------------------------
Claim-relative bias analysis using a single Groq call.

Core idea
---------
Bias alignment should be measured RELATIVE TO THE CLAIM, not on an absolute
political axis.  A source that mirrors the claim's own framing (same emotional
tone, same political direction, same rhetorical style) is *aligned* — it is
amplifying the claim's bias rather than independently verifying the facts.
Aligned sources receive a lower weight.  Sources that frame the same facts
differently are more independent and receive a higher weight.

Two-stage design
----------------
  Stage 1 — Groq returns RAW DIMENSIONS only (no alignment score):
              emotional_tone, political_direction, framing_type, explanation,
              loaded_phrases  for the claim AND each source.

  Stage 2 — Python computes alignment from those dimensions using an explicit,
              fully-explainable weighted formula:

              alignment = emotional_proximity × 0.4
                        + political_match     × 0.3
                        + framing_match       × 0.3

              Where:
                emotional_proximity = 1 - |claim_tone - source_tone| / 2
                political_match     = 1.0 if directions match else 0.0
                framing_match       = 1.0 if framing types match else 0.0

This means the alignment calculation is fully deterministic, reproducible,
and can be described precisely in a dissertation — Groq only handles the
language interpretation, not the scoring arithmetic.
"""

import json
import os
from typing import Dict, List


SYSTEM_PROMPT = """You are a media bias analyst specialising in Sri Lankan news and politics.

For each piece of text measure THREE bias dimensions and return them.
Do NOT compute or return any alignment score — that is calculated separately.

Dimensions to measure:

  emotional_tone      : float  -1.0 (very alarming / fear-inducing / negative)
                                 0.0 (neutral, factual)
                               +1.0 (very positive / celebratory)

  political_direction : "pro_government" | "opposition" | "neutral"
                        (relative to the Sri Lankan government in power)

  framing_type        : "alarmist"   — emphasises danger, crisis, urgency
                        "critical"   — questions or challenges the subject
                        "neutral"    — factual, balanced reporting
                        "supportive" — endorses or defends the subject
                        "dismissive" — downplays significance

  loaded_phrases      : list of specific words or phrases in the text that
                        reveal the bias (empty list if none found)

  explanation         : one sentence describing the bias detected

Return ONLY valid JSON — no markdown fences, no explanation outside the JSON."""


class ClaimAwareBiasAnalyzer:
    """
    Single Groq call for bias dimension extraction; Python computes alignment.
    """

    def __init__(self, groq_api_key: str = ''):
        self.api_key = groq_api_key or os.getenv('GROQ_API_KEY', '')
        self.model   = 'llama-3.3-70b-versatile'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, claim: str, sources: List[Dict]) -> Dict:
        """
        Analyse the claim and all sources.

        Returns
        -------
        {
          "claim_bias": {
              "emotional_tone": float,
              "political_direction": str,
              "framing_type": str,
              "loaded_phrases": [str, ...],
              "explanation": str
          },
          "sources": [
              {
                  "index": int,               # 0-based
                  "emotional_tone": float,
                  "political_direction": str,
                  "framing_type": str,
                  "loaded_phrases": [str, ...],
                  "explanation": str,
                  "alignment": float,         # computed in Python, not by Groq
                  "alignment_breakdown": {    # for dissertation explainability
                      "emotional_proximity": float,
                      "political_match": float,
                      "framing_match": float
                  }
              },
              ...
          ]
        }
        On failure returns safe defaults so the pipeline continues uninterrupted.
        """
        if not self.api_key or not sources:
            return self._default(sources)

        raw_result = self._call_groq(claim, sources)
        if raw_result is None:
            return self._default(sources)

        claim_bias = raw_result['claim_bias']

        sources_out = []
        for i, src_raw in enumerate(raw_result['sources']):
            alignment, breakdown = self._compute_alignment(claim_bias, src_raw)
            sources_out.append({
                'index':               i,
                'emotional_tone':      src_raw['emotional_tone'],
                'political_direction': src_raw['political_direction'],
                'framing_type':        src_raw['framing_type'],
                'loaded_phrases':      src_raw['loaded_phrases'],
                'explanation':         src_raw['explanation'],
                'alignment':           alignment,
                'alignment_breakdown': breakdown,
            })

        print(f"  [ClaimAwareBiasAnalyzer] Done — claim: {claim_bias['framing_type']} "
              f"({claim_bias['emotional_tone']:+.2f} / {claim_bias['political_direction']})")

        return {'claim_bias': claim_bias, 'sources': sources_out}

    # ------------------------------------------------------------------
    # Alignment formula (Python, fully explainable)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_alignment(claim_bias: Dict, source_bias: Dict):
        """
        Compute claim-relative alignment from raw bias dimensions.

        Formula (weighted sum of three components):
          emotional_proximity = 1 - |claim_tone - source_tone| / 2
              Normalises the 0-2 absolute difference to a 0-1 similarity score.
              Weight: 0.4

          political_match = 1.0 if both share the same political direction else 0.0
              Weight: 0.3

          framing_match = 1.0 if both use the same framing type else 0.0
              Weight: 0.3

          alignment = emotional_proximity × 0.4
                    + political_match     × 0.3
                    + framing_match       × 0.3

        Returns (alignment: float, breakdown: dict)
        """
        claim_tone      = float(claim_bias.get('emotional_tone', 0.0))
        source_tone     = float(source_bias.get('emotional_tone', 0.0))
        claim_political = claim_bias.get('political_direction', 'neutral')
        source_political = source_bias.get('political_direction', 'neutral')
        claim_framing   = claim_bias.get('framing_type', 'neutral')
        source_framing  = source_bias.get('framing_type', 'neutral')

        emotional_proximity = 1.0 - abs(claim_tone - source_tone) / 2.0
        political_match     = 1.0 if claim_political == source_political else 0.0
        framing_match       = 1.0 if claim_framing   == source_framing   else 0.0

        alignment = (
            emotional_proximity * 0.4 +
            political_match     * 0.3 +
            framing_match       * 0.3
        )

        breakdown = {
            'emotional_proximity': round(emotional_proximity, 3),
            'political_match':     political_match,
            'framing_match':       framing_match,
        }

        return round(alignment, 3), breakdown

    # ------------------------------------------------------------------
    # Groq call — returns raw dimensions only
    # ------------------------------------------------------------------

    def _call_groq(self, claim: str, sources: List[Dict]):
        """
        Single Groq call.  Returns parsed dict with claim_bias + sources[],
        each containing raw dimensions only (no alignment).
        Returns None on any failure.
        """
        def _sanitize(text: str, max_len: int) -> str:
            import unicodedata
            text = ''.join(c for c in text if unicodedata.category(c) not in ('So', 'Cs'))
            replacements = {
                '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
                '\u2013': '-', '\u2014': '-', '\u2026': '...', '\u00a0': ' ',
            }
            for orig, repl in replacements.items():
                text = text.replace(orig, repl)
            text = text.replace('\\', '').replace('"', "'")
            return text[:max_len].strip()

        source_lines = []
        for i, src in enumerate(sources, 1):
            snippet  = _sanitize(src.get('snippet') or '', 800)
            domain   = src.get('source', '') or src.get('link', '')[:60]
            date     = src.get('date', '')
            date_str = f" [{date}]" if date else ""
            source_lines.append(f"[{i}] {domain}{date_str}\n    \"{snippet}\"")

        user_prompt = (
            f'Claim: "{claim}"\n\n'
            f'Sources:\n' + '\n\n'.join(source_lines) +
            '\n\nReturn this JSON structure (nothing else):\n'
            '{\n'
            '  "claim_bias": {\n'
            '    "emotional_tone": <float -1.0 to 1.0>,\n'
            '    "political_direction": "<pro_government|opposition|neutral>",\n'
            '    "framing_type": "<alarmist|critical|neutral|supportive|dismissive>",\n'
            '    "loaded_phrases": ["<phrase>", ...],\n'
            '    "explanation": "<one sentence>"\n'
            '  },\n'
            '  "sources": [\n'
            '    {\n'
            '      "index": <int 1-based>,\n'
            '      "emotional_tone": <float -1.0 to 1.0>,\n'
            '      "political_direction": "<pro_government|opposition|neutral>",\n'
            '      "framing_type": "<alarmist|critical|neutral|supportive|dismissive>",\n'
            '      "loaded_phrases": ["<phrase>", ...],\n'
            '      "explanation": "<one sentence>"\n'
            '    }\n'
            '  ]\n'
            '}'
        )

        # ── DEBUG: show exactly what is sent to Groq ─────────────────────
        print("\n" + "=" * 60)
        print("BIAS ANALYSIS — GROQ INPUT PROMPT")
        print("=" * 60)
        print(user_prompt)
        print("=" * 60 + "\n")
        # ─────────────────────────────────────────────────────────────────

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
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()

            if raw.startswith('```'):
                raw = raw.split('```')[1]
                if raw.startswith('json'):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)

            # Normalise claim_bias
            cb = parsed.get('claim_bias', {})
            claim_bias = {
                'emotional_tone':      float(cb.get('emotional_tone', 0.0)),
                'political_direction': cb.get('political_direction', 'neutral'),
                'framing_type':        cb.get('framing_type', 'neutral'),
                'loaded_phrases':      cb.get('loaded_phrases', []),
                'explanation':         cb.get('explanation', ''),
            }

            # Normalise sources, map to 0-based index order
            src_map = {}
            for item in parsed.get('sources', []):
                idx = int(item.get('index', 0)) - 1
                src_map[idx] = {
                    'emotional_tone':      float(item.get('emotional_tone', 0.0)),
                    'political_direction': item.get('political_direction', 'neutral'),
                    'framing_type':        item.get('framing_type', 'neutral'),
                    'loaded_phrases':      item.get('loaded_phrases', []),
                    'explanation':         item.get('explanation', ''),
                }

            sources_raw = [
                src_map.get(i, self._default_source_raw())
                for i in range(len(sources))
            ]

            return {'claim_bias': claim_bias, 'sources': sources_raw}

        except Exception as e:
            print(f"  [ClaimAwareBiasAnalyzer] Groq call failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _default_source_raw() -> Dict:
        return {
            'emotional_tone':      0.0,
            'political_direction': 'neutral',
            'framing_type':        'neutral',
            'loaded_phrases':      [],
            'explanation':         'Bias analysis unavailable',
        }

    def _default_source(self, idx: int) -> Dict:
        raw = self._default_source_raw()
        alignment, breakdown = self._compute_alignment(
            {'emotional_tone': 0.0, 'political_direction': 'neutral', 'framing_type': 'neutral'},
            raw
        )
        return {
            'index':               idx,
            **raw,
            'alignment':           alignment,
            'alignment_breakdown': breakdown,
        }

    def _default(self, sources: List[Dict]) -> Dict:
        claim_bias = {
            'emotional_tone':      0.0,
            'political_direction': 'neutral',
            'framing_type':        'neutral',
            'loaded_phrases':      [],
            'explanation':         'Bias analysis unavailable',
        }
        return {
            'claim_bias': claim_bias,
            'sources':    [self._default_source(i) for i in range(len(sources))],
        }
