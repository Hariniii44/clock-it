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

              alignment = emotional_proximity × w_e
                        + political_match     × w_p
                        + framing_match       × w_f
                        + gender_bias_match   × w_g
                        + ethnic_bias_match   × w_et
                        + trauma_bias_match   × w_t

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

For the CLAIM, also identify its type. For ALL texts (claim and sources) measure SIX
bias dimensions. Do NOT compute or return any alignment score — that is calculated separately.

Claim type (return for the claim_bias object only):

  claim_type : "political"    — involves government, ministers, parliament, parties,
                                elections, political figures, or political decisions
               "economic"    — involves prices, GDP, inflation, budgets, financial
                                statistics, trade, employment numbers
               "legal"       — involves court rulings, indictments, investigations,
                                convictions, official inquiries
               "milestone"   — involves a specific appointment, election to a post,
                                historic first, milestone, record, or achievement
                                (e.g. "first woman X", "appointed as Y", "elected to Z")
               "general"     — everything else (social, sports, science, etc.)

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

  gender_bias_signal  : boolean (true / false)
                        true if text exhibits ANY of the following gender bias patterns:

                        (A) GENDER-FIRST FRAMING — foregrounding gender as the
                            primary descriptor for a professional role when neutral
                            framing would convey the same information.
                            Examples:
                              "first woman to head the government in 24 years"
                              "first female PM in 25 years" (as a headline or
                               primary descriptor, not historical context)
                              "female lawmaker", "woman politician"
                            Note: "first woman PM" IS gender-first framing even
                            if factually accurate — a gender-neutral equivalent
                            would be "first PM from NPP" or "first PM since 2000".
                            The test: would this descriptor appear if the subject
                            were male? If not, it is gender-first framing.

                        (B) STEREOTYPING — gendered stereotypes or assumptions
                            about capability, temperament, or social role.
                            Examples: "emotional response", "gentle approach"
                            attributed to gender.

                        (C) APPEARANCE / FAMILY ROLE EMPHASIS — describing a
                            woman's appearance, family, or personal life in a
                            professional context where it would not be mentioned
                            for a male counterpart.
                            Examples: "working mother", "attractive lawmaker",
                            "wife of", "mother of two".

                        (D) PATRONIZING LANGUAGE — diminishing, condescending,
                            or infantilizing language.
                            Examples: "lady politician", "girls in government".

                        false if framing is gender-neutral or focuses purely
                        on competence, policy, or professional role.

  ethnic_bias_signal  : boolean (true / false)
                        true if text contains ethnic stereotyping, "us vs them" framing
                        along Sinhala/Tamil/Muslim lines, differential treatment of
                        communities, or polarizing language that emphasises ethnic division.
                        false if ethnically neutral framing.

  trauma_trivialization : boolean (true / false)
                        true if text minimizes serious harm or suffering, uses casual
                        language for traumatic events, sensationalizes tragedy without
                        context, or lacks empathy toward victims.
                        Examples: "just a domestic issue", "minor incident",
                        downplaying war crimes or discrimination.
                        false if appropriate sensitivity is shown.

  source_type         : classify the source by its nature:
                        "news_media"    — traditional news outlets (newspapers, TV, online news)
                        "official_data" — government agencies, central bank, statistics dept,
                                          court records, parliamentary hansard
                        "expert_opinion"— academic papers, think tanks, expert commentary
                        "social_media"  — Facebook, Twitter, WhatsApp, YouTube content
                        "unknown"       — cannot determine from URL or content style

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
                        (c) state the likely editorial intent or effect on the reader,
                        (d) if any bias is present (gender, ethnic,
                            or trauma trivialization), name it and explain
                            how it manifests in this text.
                        Do NOT write generic summaries — be specific about the language used.

Return ONLY valid JSON — no markdown fences, no explanation outside the JSON."""


class ClaimAwareBiasAnalyzer:
    """
    Single Groq call for bias dimension extraction; Python computes alignment.
    """

    # Sensitivity weights per claim type — how much each bias dimension
    # matters when computing claim-relative alignment.
    # Weights must sum to 1.0 within each row.
    # Sensitivity weights per claim type — how much each bias dimension matters
    # when computing claim-relative alignment. All rows sum to 1.0 (milestone
    # intentionally small to cap its max penalty at ~6%).
    # Three classic dimensions + three Sri Lankan-specific dimensions (interview findings):
    #   gender  — Deepanjalie: gendered framing is endemic in SL political reporting
    #   ethnic  — Deepanjalie: Sinhala/Tamil/Muslim polarisation is a primary bias vector
    #   trauma  — Deepanjalie: trivialization of war-related suffering distorts coverage
    _SENSITIVITY = {
        'political': {
            'emotional': 0.15, 'political': 0.40, 'framing': 0.25,
            'gender': 0.05, 'ethnic': 0.10, 'trauma': 0.05,
        },
        'economic': {
            'emotional': 0.25, 'political': 0.10, 'framing': 0.50,
            'gender': 0.05, 'ethnic': 0.05, 'trauma': 0.05,
        },
        'legal': {
            'emotional': 0.15, 'political': 0.15, 'framing': 0.50,
            'gender': 0.05, 'ethnic': 0.05, 'trauma': 0.10,
        },
        'milestone': {
            # Positive framing of a milestone is expected accurate reporting, not bias.
            # All weights tiny → max possible alignment ≈ 0.12 → max penalty ≈ 6%.
            'emotional': 0.03, 'political': 0.03, 'framing': 0.03,
            'gender': 0.01, 'ethnic': 0.01, 'trauma': 0.01,
        },
        'general': {
            'emotional': 0.35, 'political': 0.05, 'framing': 0.35,
            'gender': 0.10, 'ethnic': 0.10, 'trauma': 0.05,
        },
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
                  "index": int,                  # 0-based
                  "source_type": str,            # news_media|official_data|expert_opinion|...
                  "emotional_tone": float,
                  "political_direction": str,
                  "framing_type": str,
                  "gender_bias_signal": bool,    # Sri Lankan-specific
                  "ethnic_bias_signal": bool,    # Sri Lankan-specific
                  "trauma_trivialization": bool, # Sri Lankan-specific
                  "loaded_phrases": [str, ...],
                  "explanation": str,
                  "alignment": float,            # computed in Python, not by Groq
                  "alignment_breakdown": dict    # for dissertation explainability
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

            # Adjust sensitivity weights based on source type (Mahoshadi interview finding)
            source_type = src_raw.get('source_type', 'unknown')
            adjusted_sensitivity = self._adjust_sensitivity_by_source_type(sensitivity, source_type)

            alignment, breakdown = self._compute_alignment(
                claim_bias, src_raw, adjusted_sensitivity, profile_direction
            )
            sources_out.append({
                'index':                 i,
                'source_type':           source_type,
                'emotional_tone':        src_raw['emotional_tone'],
                'political_direction':   src_raw['political_direction'],
                'framing_type':          src_raw['framing_type'],
                'gender_bias_signal':    src_raw.get('gender_bias_signal', False),
                'ethnic_bias_signal':    src_raw.get('ethnic_bias_signal', False),
                'trauma_trivialization': src_raw.get('trauma_trivialization', False),
                'loaded_phrases':        src_raw['loaded_phrases'],
                'explanation':           src_raw['explanation'],
                'alignment':             alignment,
                'alignment_breakdown':   breakdown,
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

        Formula (6 dimensions)
        ----------------------
          alignment = emotional_proximity × w_emotional
                    + political_match     × w_political
                    + framing_match       × w_framing
                    + gender_bias_match   × w_gender
                    + ethnic_bias_match   × w_ethnic
                    + trauma_bias_match   × w_trauma

        Weights (w_*) come from the claim-type sensitivity table.

        Sri Lankan-specific dimensions (gender, ethnic, trauma) are binary
        flags: they contribute to alignment only when BOTH claim AND source
        exhibit the same bias pattern, meaning the source is echoing the
        same problematic framing as the claim itself.

        Political match blending
        ------------------------
        When a pre-computed bias profile exists for the source, the
        political match is a 50/50 blend of LLM real-time direction and
        historical profile direction. This prevents a chronically biased
        source from "passing" on a single neutral snippet.

        Returns (alignment: float, breakdown: dict)
        """
        if sensitivity is None:
            sensitivity = {
                'emotional': 0.30, 'political': 0.25, 'framing': 0.25,
                'gender': 0.07, 'ethnic': 0.08, 'trauma': 0.05,
            }

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

        # Sri Lankan-specific bias dimensions — binary signal matching
        # Alignment increases only when both claim and source share the same flag,
        # i.e., the source reinforces the claim's problematic framing.
        claim_gender  = bool(claim_bias.get('gender_bias_signal', False))
        source_gender = bool(source_bias.get('gender_bias_signal', False))
        gender_bias_match = 1.0 if (claim_gender and source_gender) else 0.0

        claim_ethnic  = bool(claim_bias.get('ethnic_bias_signal', False))
        source_ethnic = bool(source_bias.get('ethnic_bias_signal', False))
        ethnic_bias_match = 1.0 if (claim_ethnic and source_ethnic) else 0.0

        claim_trauma  = bool(claim_bias.get('trauma_trivialization', False))
        source_trauma = bool(source_bias.get('trauma_trivialization', False))
        trauma_bias_match = 1.0 if (claim_trauma and source_trauma) else 0.0

        w_e  = sensitivity.get('emotional', 0.30)
        w_p  = sensitivity.get('political', 0.25)
        w_f  = sensitivity.get('framing',   0.25)
        w_g  = sensitivity.get('gender',    0.07)
        w_et = sensitivity.get('ethnic',    0.08)
        w_t  = sensitivity.get('trauma',    0.05)

        raw_alignment = (
            emotional_proximity * w_e +
            political_match     * w_p +
            framing_match       * w_f +
            gender_bias_match   * w_g +
            ethnic_bias_match   * w_et +
            trauma_bias_match   * w_t
        )

        # Gate: only apply a penalty if the source shows a genuine bias signal.
        # A neutral source that happens to match neutral framing is NOT penalised.
        has_loaded_phrases  = bool(source_bias.get('loaded_phrases'))
        has_emotional_bias  = abs(source_tone) > 0.3
        has_political_bias  = source_bias.get('political_direction', 'neutral') != 'neutral'
        has_gender_bias     = bool(source_bias.get('gender_bias_signal', False))
        has_ethnic_bias     = bool(source_bias.get('ethnic_bias_signal', False))
        has_trauma_bias     = bool(source_bias.get('trauma_trivialization', False))
        has_bias_signal     = (
            has_loaded_phrases or has_emotional_bias or has_political_bias
            or has_gender_bias or has_ethnic_bias or has_trauma_bias
        )

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
            'framing_match':            round(framing_match, 3),
            'gender_bias_match':        round(gender_bias_match, 3),
            'ethnic_bias_match':        round(ethnic_bias_match, 3),
            'trauma_bias_match':        round(trauma_bias_match, 3),
            'profile_direction':        profile_political_direction,
            'sri_lankan_bias_flags': {
                'gender': has_gender_bias,
                'ethnic': has_ethnic_bias,
                'trauma': has_trauma_bias,
            },
            'has_bias_signal':          has_bias_signal,
            'bias_signal_reason':       (
                'loaded phrases detected'         if has_loaded_phrases
                else 'emotional tone non-neutral' if has_emotional_bias
                else 'political lean detected'    if has_political_bias
                else 'gender bias detected'       if has_gender_bias
                else 'ethnic bias detected'       if has_ethnic_bias
                else 'trauma trivialization'      if has_trauma_bias
                else 'no bias signal — penalty suppressed'
            ),
        }

        return round(alignment, 3), breakdown

    # ------------------------------------------------------------------
    # Source-type sensitivity adjustment
    # ------------------------------------------------------------------

    @staticmethod
    def _adjust_sensitivity_by_source_type(
        base_sensitivity: Dict,
        source_type: str,
    ) -> Dict:
        """
        Scale bias dimension weights based on the source's type.

        Rationale (Mahoshadi interview):
        - news_media:    Political and framing bias most relevant — ownership and
                         editorial stance shape every story.
        - official_data: Methodology transparency matters most. Political/emotional
                         signals are weak in official documents; acknowledged as a
                         limitation (methodology bias not captured in text dimensions).
        - expert_opinion: Framing reflects theoretical framework — the dimension most
                          worth amplifying here.
        - social_media:  Emotional and ethnic/tribal signals dominate; amplify both.
        - unknown:       No adjustment — use base weights unchanged.

        Multipliers are applied then renormalised so the weights still sum to the
        original total (preserving the overall penalty magnitude scale).
        """
        _MULTIPLIERS = {
            'news_media':     {'political': 1.3, 'framing': 1.2, 'emotional': 0.9},
            'official_data':  {'political': 0.5, 'emotional': 0.5, 'framing': 0.7},
            'expert_opinion': {'framing': 1.4, 'political': 0.8, 'emotional': 0.7},
            'social_media':   {'emotional': 1.4, 'ethnic': 1.3, 'political': 1.2},
        }
        multipliers = _MULTIPLIERS.get(source_type)
        if not multipliers:
            return base_sensitivity

        adjusted = {k: v * multipliers.get(k, 1.0) for k, v in base_sensitivity.items()}
        # Renormalise to the same sum as the original weights
        original_total = sum(base_sensitivity.values()) or 1.0
        adjusted_total = sum(adjusted.values()) or 1.0
        scale = original_total / adjusted_total
        return {k: round(v * scale, 4) for k, v in adjusted.items()}

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
            '    "claim_type": "<political|economic|legal|milestone|general>",\n'
            '    "emotional_tone": <float -1.0 to 1.0>,\n'
            '    "political_direction": "<pro_government|opposition|neutral>",\n'
            '    "framing_type": "<alarmist|critical|neutral|supportive|dismissive>",\n'
            '    "gender_bias_signal": <true|false>,\n'
            '    "ethnic_bias_signal": <true|false>,\n'
            '    "trauma_trivialization": <true|false>,\n'
            '    "loaded_phrases": ["<phrase>", ...],\n'
            '    "explanation": "<2-3 sentences>"\n'
            '  },\n'
            '  "sources": [\n'
            '    {\n'
            '      "index": <int 1-based>,\n'
            '      "emotional_tone": <float -1.0 to 1.0>,\n'
            '      "political_direction": "<pro_government|opposition|neutral>",\n'
            '      "framing_type": "<alarmist|critical|neutral|supportive|dismissive>",\n'
            '      "gender_bias_signal": <true|false>,\n'
            '      "ethnic_bias_signal": <true|false>,\n'
            '      "trauma_trivialization": <true|false>,\n'
            '      "source_type": "<news_media|official_data|expert_opinion|social_media|unknown>",\n'
            '      "loaded_phrases": ["<phrase>", ...],\n'
            '      "explanation": "<2-3 sentences>"\n'
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

        import time
        last_error = None
        for attempt in range(2):
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
                    max_tokens=3072,
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
                if raw_claim_type not in ('political', 'economic', 'legal', 'milestone', 'general'):
                    raw_claim_type = 'general'
                claim_bias = {
                    'claim_type':            raw_claim_type,
                    'emotional_tone':        float(cb.get('emotional_tone', 0.0)),
                    'political_direction':   cb.get('political_direction', 'neutral'),
                    'framing_type':          cb.get('framing_type', 'neutral'),
                    'gender_bias_signal':    bool(cb.get('gender_bias_signal', False)),
                    'ethnic_bias_signal':    bool(cb.get('ethnic_bias_signal', False)),
                    'trauma_trivialization': bool(cb.get('trauma_trivialization', False)),
                    'loaded_phrases':        cb.get('loaded_phrases', []),
                    'explanation':           cb.get('explanation', ''),
                }

                # Normalise sources, map to 0-based index order
                _valid_source_types = {'news_media', 'official_data', 'expert_opinion', 'social_media', 'unknown'}
                src_map = {}
                for item in parsed.get('sources', []):
                    idx = int(item.get('index', 0)) - 1
                    raw_source_type = item.get('source_type', 'unknown')
                    if raw_source_type not in _valid_source_types:
                        raw_source_type = 'unknown'
                    src_map[idx] = {
                        'emotional_tone':        float(item.get('emotional_tone', 0.0)),
                        'political_direction':   item.get('political_direction', 'neutral'),
                        'framing_type':          item.get('framing_type', 'neutral'),
                        'gender_bias_signal':    bool(item.get('gender_bias_signal', False)),
                        'ethnic_bias_signal':    bool(item.get('ethnic_bias_signal', False)),
                        'trauma_trivialization': bool(item.get('trauma_trivialization', False)),
                        'source_type':           raw_source_type,
                        'loaded_phrases':        item.get('loaded_phrases', []),
                        'explanation':           item.get('explanation', ''),
                    }

                sources_raw = [
                    src_map.get(i, self._default_source_raw())
                    for i in range(len(sources))
                ]

                return {'claim_bias': claim_bias, 'sources': sources_raw}

            except Exception as e:
                last_error = e
                print(f"  [ClaimAwareBiasAnalyzer] Groq call failed (attempt {attempt + 1}/2): {e}")
                if attempt == 0:
                    time.sleep(2)  # brief pause before retry

        print(f"  [ClaimAwareBiasAnalyzer] Both attempts failed — using defaults. Last error: {last_error}")
        return None

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _default_source_raw() -> Dict:
        return {
            'emotional_tone':        0.0,
            'political_direction':   'neutral',
            'framing_type':          'neutral',
            'gender_bias_signal':    False,
            'ethnic_bias_signal':    False,
            'trauma_trivialization': False,
            'source_type':           'unknown',
            'loaded_phrases':        [],
            'explanation':           'Bias analysis unavailable',
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
            'claim_type':            'general',
            'emotional_tone':        0.0,
            'political_direction':   'neutral',
            'framing_type':          'neutral',
            'gender_bias_signal':    False,
            'ethnic_bias_signal':    False,
            'trauma_trivialization': False,
            'loaded_phrases':        [],
            'explanation':           'Bias analysis unavailable',
        }
        return {
            'claim_bias': claim_bias,
            'sources':    [self._default_source(i) for i in range(len(sources))],
        }
