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

For the CLAIM, also identify its type. For ALL texts (claim and sources) measure THREE
bias dimensions. Do NOT compute or return any alignment score — that is calculated separately.

Claim type (return for the claim_bias object only):

  claim_type : "political"  — involves government, ministers, parliament, parties,
                              elections, political figures, or political decisions
               "economic"   — involves prices, GDP, inflation, budgets, financial
                              statistics, trade, employment numbers
               "legal"      — involves court rulings, indictments, investigations,
                              convictions, official inquiries
               "general"    — everything else (social, sports, science, etc.)

Dimensions to measure for every text:

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

  loaded_phrases      : list of emotionally charged, politically loaded, or
                        rhetorically manipulative words/phrases found in the text.
                        These must be QUOTED DIRECTLY from the text.
                        DO NOT include neutral topic descriptors — the subject
                        name or event name alone (e.g. "bond scam", "election")
                        is NOT a loaded phrase. Only include language that reveals
                        editorial bias: e.g. "brains behind", "chief planner",
                        "alleged mastermind", "blatant corruption", "witch hunt".
                        Return empty list [] if no genuinely loaded language exists.

  explanation         : 2-3 sentences explaining the bias. You MUST:
                        (a) quote at least one specific phrase directly from the
                            text that reveals the bias,
                        (b) explain WHY that phrasing is biased rather than neutral,
                        (c) state the likely editorial intent or effect on the reader.
                        Do NOT write generic summaries like "critical report with
                        negative tone" — be specific about the language used.

Return ONLY valid JSON — no markdown fences, no explanation outside the JSON."""


class ClaimAwareBiasAnalyzer:
    """
    Single Groq call for bias dimension extraction; Python computes alignment.
    """

    # Sensitivity weights per claim type — how much each bias dimension
    # matters when computing claim-relative alignment.
    # Weights must sum to 1.0 within each row.
    _SENSITIVITY = {
        'political': {'emotional': 0.20, 'political': 0.50, 'framing': 0.30},
        'economic':  {'emotional': 0.30, 'political': 0.10, 'framing': 0.60},
        'legal':     {'emotional': 0.20, 'political': 0.20, 'framing': 0.60},
        'general':   {'emotional': 0.50, 'political': 0.05, 'framing': 0.45},
    }

    def __init__(self, groq_api_key: str = ''):
        self.api_key = groq_api_key or os.getenv('GROQ_API_KEY', '')
        self.model   = 'llama-3.3-70b-versatile'
        self.bias_profiles = self._load_bias_profiles()

    @staticmethod
    def _load_bias_profiles() -> Dict:
        try:
            import json as _json
            with open('data/bias_profiles.json', 'r', encoding='utf-8') as f:
                return _json.load(f)
        except Exception:
            return {}

    def _domain_from_url(self, url: str) -> str:
        try:
            u = url.split('://', 1)[-1]
            d = u.split('/')[0]
            return d[4:] if d.startswith('www.') else d
        except Exception:
            return ''

    def _profile_political_direction(self, url: str) -> str:
        """Derive political direction from pre-computed bias profile score."""
        domain = self._domain_from_url(url)
        profile = self.bias_profiles.get(domain)
        if not profile:
            return 'unknown'
        score = profile.get('bias_score', 0.0)
        if score > 10:
            return 'pro_government'
        if score < -10:
            return 'opposition'
        return 'neutral'

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
        claim_type = claim_bias.get('claim_type', 'general')
        sensitivity = self._SENSITIVITY.get(claim_type, self._SENSITIVITY['general'])

        sources_out = []
        for i, src_raw in enumerate(raw_result['sources']):
            url = sources[i].get('link', '') or sources[i].get('url', '') if i < len(sources) else ''
            profile_direction = self._profile_political_direction(url)
            alignment, breakdown = self._compute_alignment(
                claim_bias, src_raw, sensitivity, profile_direction
            )
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

        print(f"  [ClaimAwareBiasAnalyzer] Done — claim type: {claim_type} | "
              f"framing: {claim_bias['framing_type']} "
              f"({claim_bias['emotional_tone']:+.2f} / {claim_bias['political_direction']})")

        return {'claim_bias': claim_bias, 'sources': sources_out}

    # ------------------------------------------------------------------
    # Alignment formula (Python, fully explainable)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_alignment(
        claim_bias: Dict,
        source_bias: Dict,
        sensitivity: Dict = None,
        profile_political_direction: str = 'unknown',
    ):
        """
        Compute claim-relative alignment from raw bias dimensions.

        Formula
        -------
          alignment = emotional_proximity × w_emotional
                    + political_match     × w_political
                    + framing_match       × w_framing

        Weights (w_*) come from the claim-type sensitivity table so that
        bias dimensions that are irrelevant to the claim type contribute
        minimally to the penalty.  E.g. for an economic statistics claim
        the political weight drops to 0.10, so a source's political lean
        barely affects its credibility score.

        Political match blending
        ------------------------
        When a pre-computed bias profile exists for the source, the
        political match is a 50/50 blend of:
          - LLM real-time direction (what this specific snippet says)
          - Historical profile direction (what this outlet typically says)
        This prevents a chronically biased source from "passing" because
        a single neutral snippet hides its systematic lean.

        Returns (alignment: float, breakdown: dict)
        """
        if sensitivity is None:
            sensitivity = {'emotional': 0.40, 'political': 0.30, 'framing': 0.30}

        claim_tone       = float(claim_bias.get('emotional_tone', 0.0))
        source_tone      = float(source_bias.get('emotional_tone', 0.0))
        claim_political  = claim_bias.get('political_direction', 'neutral')
        source_political = source_bias.get('political_direction', 'neutral')
        claim_framing    = claim_bias.get('framing_type', 'neutral')
        source_framing   = source_bias.get('framing_type', 'neutral')

        emotional_proximity = 1.0 - abs(claim_tone - source_tone) / 2.0

        # LLM-based political match (real-time, this snippet)
        llm_political_match = 1.0 if claim_political == source_political else 0.0

        # Profile-based political match (historical, 50/50 blend when available)
        if profile_political_direction not in ('unknown', ''):
            profile_political_match = 1.0 if claim_political == profile_political_direction else 0.0
            political_match = 0.5 * llm_political_match + 0.5 * profile_political_match
        else:
            political_match = llm_political_match

        framing_match = 1.0 if claim_framing == source_framing else 0.0

        w_e = sensitivity['emotional']
        w_p = sensitivity['political']
        w_f = sensitivity['framing']

        raw_alignment = (
            emotional_proximity * w_e +
            political_match     * w_p +
            framing_match       * w_f
        )

        # Gate: only apply a penalty if the source shows genuine bias signal.
        # A neutral source that happens to match the claim's neutral framing
        # should NOT be penalised — tonal consistency ≠ editorial bias.
        # Bias signal requires at least one of:
        #   - emotionally charged language quoted from the text
        #   - meaningfully non-neutral emotional tone (|tone| > 0.3)
        #   - a political lean (not neutral)
        has_loaded_phrases  = bool(source_bias.get('loaded_phrases'))
        has_emotional_bias  = abs(source_tone) > 0.3
        has_political_bias  = source_bias.get('political_direction', 'neutral') != 'neutral'
        has_bias_signal     = has_loaded_phrases or has_emotional_bias or has_political_bias

        alignment = raw_alignment if has_bias_signal else 0.0

        breakdown = {
            'claim_type_weights':       sensitivity,
            'emotional_proximity':      round(emotional_proximity, 3),
            'political_match_llm':      round(llm_political_match, 3),
            'political_match_profile':  round(
                0.5 * llm_political_match + 0.5 * (
                    1.0 if claim_political == profile_political_direction else 0.0
                ) if profile_political_direction not in ('unknown', '') else llm_political_match,
                3
            ),
            'political_match_blended':  round(political_match, 3),
            'framing_match':            framing_match,
            'profile_direction':        profile_political_direction,
            'has_bias_signal':          has_bias_signal,
            'bias_signal_reason':       (
                'loaded phrases detected' if has_loaded_phrases
                else 'emotional tone non-neutral' if has_emotional_bias
                else 'political lean detected' if has_political_bias
                else 'no bias signal — penalty suppressed'
            ),
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
            '    "claim_type": "<political|economic|legal|general>",\n'
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

        # # ── DEBUG: show exactly what is sent to Groq ─────────────────────
        # print("\n" + "=" * 60)
        # print("BIAS ANALYSIS — GROQ INPUT PROMPT")
        # print("=" * 60)
        # print(user_prompt)
        # print("=" * 60 + "\n")
        # # ─────────────────────────────────────────────────────────────────

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
            raw_claim_type = cb.get('claim_type', 'general').lower()
            if raw_claim_type not in ('political', 'economic', 'legal', 'general'):
                raw_claim_type = 'general'
            claim_bias = {
                'claim_type':          raw_claim_type,
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
            'claim_type':          'general',
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
