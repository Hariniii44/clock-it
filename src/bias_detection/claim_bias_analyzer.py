"""
claim_bias_analyzer.py
----------------------
Claim-relative bias analysis using a single Gemini 2.5 Pro call.

Core idea
---------
Bias alignment is measured RELATIVE TO THE CLAIM, not on an absolute
political axis.  A source that mirrors the claim's own framing (same emotional
tone, same political direction, same rhetorical style) is *aligned* — it is
amplifying the claim's bias rather than independently verifying the facts.
Aligned sources receive a lower weight.  Sources that frame the same facts
differently are more independent and receive a higher weight.

Two-stage design
----------------
  Stage 1 — Gemini 2.5 Pro returns RAW DIMENSIONS only (no alignment score),
              using response_mime_type="application/json" to enforce valid JSON
              at the sampling level rather than relying on prompt instructions:
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

              Then gated by a sigmoid bias-signal strength function so that
              genuinely neutral sources receive near-zero penalty, without a
              discontinuous hard threshold.

              Where:
                emotional_proximity = 1 - |claim_tone - source_tone| / 2
                political_match     = 1.0 if directions match AND direction != "neutral"
                                      else 0.0
                framing_match       = 1.0 if framing types match else 0.0

This means the alignment calculation is fully deterministic, reproducible,
and can be described precisely in a dissertation — Gemini only handles the
language interpretation, not the scoring arithmetic.

Fixes over claim_aware_bias_analyzer.py
----------------------------------------
  1. Gemini 2.5 Pro + JSON mode  — more reliable structured output than Groq/Llama
  2. neutral==neutral political match no longer fires — two neutral texts sharing
     neutrality is not bias amplification
  3. Index-default bug fixed      — default was 0 → idx=-1 (silently dropped);
     now defaults to 1 → idx=0
  4. Sigmoid bias gate            — replaces hard 0.3 threshold with a smooth
     function; eliminates the discontinuity that would be indefensible in
     a dissertation
  5. Milestone weights fixed      — gender weight elevated to 0.25 (most relevant
     bias vector for milestone claims per Deepanjalie interview); framing
     exception now applies only when loaded phrases are also present
  6. bias_analysis_failed flag    — added to defaults so evaluation scripts can
     exclude corrupted runs from metric aggregation
  7. Domain length capped         — src['source'] field now capped at 80 chars
  8. Redundant inner json import removed
"""

import json
import math
import os
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

import google.genai as genai
from google.genai import types


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


def _soft_gate(signal_strength: float, threshold: float = 0.3, steepness: float = 10.0) -> float:
    """
    Sigmoid gate centred on `threshold`.

    Returns ~0 when signal_strength is well below the threshold and ~1 when
    well above it.  Replaces the hard if/else threshold so there is no
    discontinuous cliff where a source at 0.29 gets zero penalty and one at
    0.31 gets the full alignment penalty.

    At threshold=0.3, steepness=10:
      signal=0.0  → gate≈0.05
      signal=0.3  → gate=0.50
      signal=0.5  → gate≈0.88
      signal=1.0  → gate≈1.00
    """
    return 1.0 / (1.0 + math.exp(-steepness * (signal_strength - threshold)))


class ClaimBiasAnalyzer:
    """
    Single Gemini 2.5 Pro call for bias dimension extraction; Python computes alignment.

    Drop-in replacement for ClaimAwareBiasAnalyzer with the following fixes:
      - Gemini 2.5 Pro + JSON mode (application/json MIME type)
      - Political match no longer fires for neutral==neutral pairs
      - Index-default bug fixed (was mapping missing index to src_map[-1])
      - Sigmoid soft gate replaces hard 0.3 threshold
      - Milestone claim type now correctly surfaces gender bias
      - bias_analysis_failed flag in defaults for evaluation filtering
    """

    # Sensitivity weights per claim type — how much each bias dimension
    # matters when computing claim-relative alignment.
    #
    # Three classic dimensions + three Sri Lankan-specific dimensions:
    #   gender  — Deepanjalie: gendered framing is endemic in SL political reporting
    #   ethnic  — Deepanjalie: Sinhala/Tamil/Muslim polarisation is a primary bias vector
    #   trauma  — Deepanjalie: trivialization of war-related suffering distorts coverage
    #
    # Milestone note: the old implementation used near-zero weights (0.01–0.03) to
    # suppress all penalties for milestone claims.  This was too aggressive — it also
    # suppressed gender bias detection, which is most relevant for milestone claims
    # (e.g. "first woman PM").  Gender weight is now elevated to 0.25 for milestone,
    # and the framing exception is handled explicitly in _compute_alignment instead.
    _SENSITIVITY: Dict[str, Dict[str, float]] = {
        'political': {
            'emotional': 0.15, 'political': 0.40, 'framing': 0.25,
            'gender': 0.05,    'ethnic': 0.10,    'trauma': 0.05,
        },
        'economic': {
            'emotional': 0.25, 'political': 0.10, 'framing': 0.50,
            'gender': 0.05,    'ethnic': 0.05,    'trauma': 0.05,
        },
        'legal': {
            'emotional': 0.15, 'political': 0.15, 'framing': 0.50,
            'gender': 0.05,    'ethnic': 0.05,    'trauma': 0.10,
        },
        'milestone': {
            # Supportive framing of a genuine achievement is expected accurate
            # reporting.  We handle the framing exception explicitly in
            # _compute_alignment (only count framing alignment when loaded
            # phrases are also present), so full weights are appropriate here.
            # Gender is elevated because gendered framing is the primary bias
            # vector for milestone claims (Deepanjalie interview finding).
            'emotional': 0.15, 'political': 0.15, 'framing': 0.20,
            'gender': 0.25,    'ethnic': 0.15,    'trauma': 0.10,
        },
        'general': {
            'emotional': 0.35, 'political': 0.05, 'framing': 0.35,
            'gender': 0.10,    'ethnic': 0.10,    'trauma': 0.05,
        },
    }

    FALLBACK_MODELS = [
        'gemini-2.5-flash',
        'gemini-2.0-flash',
        'gemini-1.5-flash',
    ]

    def __init__(self, gemini_api_key: str = ''):
        self.api_key      = gemini_api_key or os.getenv('GEMINI_API_KEY', '')
        self.model        = 'gemini-2.5-flash'
        self.bias_profiles = self._load_bias_profiles()
        self._client: Optional[genai.Client] = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            from config import Config
            self._client = genai.Client(
                vertexai=True,
                project=Config.GCP_PROJECT,
                location=Config.GCP_LOCATION,
            )
        return self._client

    @staticmethod
    def _load_bias_profiles() -> Dict:
        try:
            with open('data/bias_profiles.json', 'r', encoding='utf-8') as f:
                return json.load(f)
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
        domain  = self._domain_from_url(url)
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
              "claim_type": str,
              "emotional_tone": float,
              "political_direction": str,
              "framing_type": str,
              "gender_bias_signal": bool,
              "ethnic_bias_signal": bool,
              "trauma_trivialization": bool,
              "loaded_phrases": [str, ...],
              "explanation": str
          },
          "sources": [
              {
                  "index": int,                   # 0-based
                  "source_type": str,             # news_media|official_data|...
                  "emotional_tone": float,
                  "political_direction": str,
                  "framing_type": str,
                  "gender_bias_signal": bool,
                  "ethnic_bias_signal": bool,
                  "trauma_trivialization": bool,
                  "loaded_phrases": [str, ...],
                  "explanation": str,
                  "alignment": float,             # computed in Python, not by Gemini
                  "alignment_breakdown": dict     # for dissertation explainability
              },
              ...
          ],
          "bias_analysis_failed": bool            # True only on API failure
        }
        On failure returns safe defaults so the pipeline continues uninterrupted.
        """
        if not sources:
            return self._default(sources)

        if not self.api_key:
            # Vertex AI authentication is handled by the runtime service account.
            # The API key is not required for the deployed backend.
            pass

        raw_result = self._call_gemini(claim, sources)
        if raw_result is None:
            return self._default(sources)

        claim_bias = raw_result['claim_bias']
        claim_type  = claim_bias.get('claim_type', 'general')
        sensitivity = self._SENSITIVITY.get(claim_type, self._SENSITIVITY['general'])

        sources_out = []
        for i, src_raw in enumerate(raw_result['sources']):
            url = sources[i].get('link', '') or sources[i].get('url', '') if i < len(sources) else ''
            profile_direction = self._profile_political_direction(url)

            # Adjust sensitivity weights based on source type (Mahoshadi interview finding)
            source_type          = src_raw.get('source_type', 'unknown')
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

        print(f"  [ClaimBiasAnalyzer] Done — claim type: {claim_type} | "
              f"framing: {claim_bias['framing_type']} "
              f"({claim_bias['emotional_tone']:+.2f} / {claim_bias['political_direction']})")

        return {'claim_bias': claim_bias, 'sources': sources_out, 'bias_analysis_failed': False}

    # ------------------------------------------------------------------
    # Alignment formula (Python, fully explainable)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_alignment(
        claim_bias: Dict,
        source_bias: Dict,
        sensitivity: Dict = None,
        profile_political_direction: str = 'unknown',
    ) -> Tuple[float, Dict]:
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

        Political match
        ---------------
        Fires only when both claim and source share a NON-neutral political
        direction.  Two neutral texts sharing neutrality is not bias
        amplification and must not contribute to the alignment penalty.

        Milestone framing exception
        ---------------------------
        For milestone claims, supportive framing of a genuine achievement is
        expected accurate reporting, not amplification bias.  Framing match
        only contributes when loaded phrases are also present, indicating the
        source has added rhetorical weight on top of the factual milestone.

        Soft sigmoid gate
        -----------------
        raw_alignment is scaled by a sigmoid function of the overall bias signal
        strength.  This replaces the previous hard threshold (abs(tone) > 0.3)
        which created a discontinuous penalty cliff.  Genuinely neutral sources
        receive a near-zero gate value and thus a near-zero penalty.

        Political match blending
        ------------------------
        When a pre-computed bias profile exists, political match is a 50/50
        blend of the LLM real-time direction and the historical profile
        direction.  This prevents a chronically biased source from "passing"
        on a single neutral snippet.

        Returns (alignment: float, breakdown: dict)
        """
        if sensitivity is None:
            sensitivity = {
                'emotional': 0.30, 'political': 0.25, 'framing': 0.25,
                'gender': 0.07,    'ethnic': 0.08,    'trauma': 0.05,
            }

        claim_tone       = float(claim_bias.get('emotional_tone', 0.0))
        source_tone      = float(source_bias.get('emotional_tone', 0.0))
        claim_political  = claim_bias.get('political_direction', 'neutral')
        source_political = source_bias.get('political_direction', 'neutral')
        claim_framing    = claim_bias.get('framing_type', 'neutral')
        source_framing   = source_bias.get('framing_type', 'neutral')
        claim_type       = claim_bias.get('claim_type', 'general')

        # Emotional amplification: how strongly does the source echo the claim's
        # emotional direction?  Scaled by the claim's own emotional magnitude so
        # that a neutral claim (tone≈0) cannot produce a high proximity score —
        # there is no emotional bias to amplify when the claim itself is neutral.
        directional_proximity = 1.0 - abs(claim_tone - source_tone) / 2.0
        emotional_scale = max(abs(claim_tone), abs(source_tone) * 0.4)
        emotional_proximity = directional_proximity * emotional_scale

        # Political match: only fires when the shared direction is non-neutral.
        # neutral==neutral is not bias amplification — it is just both texts
        # reporting factually, which must not inflate the alignment penalty.
        directions_match   = claim_political == source_political
        non_neutral_match  = directions_match and claim_political != 'neutral'
        llm_political_match = 1.0 if non_neutral_match else 0.0

        # Profile-based political match (historical, 50/50 blend when available)
        if profile_political_direction not in ('unknown', ''):
            profile_non_neutral    = (claim_political == profile_political_direction
                                      and claim_political != 'neutral')
            profile_political_match = 1.0 if profile_non_neutral else 0.0
            political_match = 0.5 * llm_political_match + 0.5 * profile_political_match
        else:
            political_match = llm_political_match

        # Framing match — mirrors the political_match neutral guard.
        # neutral==neutral is factual reporting, not bias amplification.
        # Only fire when the SHARED framing is a non-neutral editorial stance.
        # Milestone exception: supportive framing on a genuine achievement is
        # expected accurate reporting; only count it when loaded phrases are
        # also present (source adds rhetorical weight on top of the fact).
        frames_match = claim_framing == source_framing
        is_neutral_framing_pair = (claim_framing == 'neutral' and source_framing == 'neutral')
        if is_neutral_framing_pair:
            framing_match = 0.0  # two neutral texts sharing neutrality is not amplification
        elif (claim_type == 'milestone'
                and claim_framing == 'supportive'
                and source_framing == 'supportive'):
            framing_match = 1.0 if (frames_match and bool(source_bias.get('loaded_phrases'))) else 0.0
        else:
            framing_match = 1.0 if frames_match else 0.0

        # Sri Lankan-specific bias dimensions — binary signal matching.
        # Alignment increases only when both claim and source share the same flag.
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

        # --- Sigmoid soft gate ---
        # Measures overall bias signal strength across all dimensions, then
        # scales raw_alignment smoothly.  Replaces the previous hard threshold
        # (abs(tone) > 0.3) which created an unjustifiable penalty cliff.
        #
        # Signal strength is the max contribution across detectable signals:
        #   - emotional: continuous tone magnitude
        #   - structural: binary flags (loaded phrases, political lean, etc.)
        has_loaded_phrases = bool(source_bias.get('loaded_phrases'))
        has_political_bias = source_political != 'neutral'
        has_gender_bias    = bool(source_bias.get('gender_bias_signal', False))
        has_ethnic_bias    = bool(source_bias.get('ethnic_bias_signal', False))
        has_trauma_bias    = bool(source_bias.get('trauma_trivialization', False))

        structural_signal = 1.0 if (
            has_loaded_phrases or has_political_bias
            or has_gender_bias or has_ethnic_bias or has_trauma_bias
        ) else 0.0

        # Gate is the max of the smooth emotional gate and the structural gate.
        # Structural signals always fully open the gate; emotional uses sigmoid.
        emotional_gate = _soft_gate(abs(source_tone), threshold=0.3, steepness=10.0)
        bias_gate      = max(emotional_gate, structural_signal)

        alignment = raw_alignment * bias_gate

        breakdown = {
            'claim_type':                claim_type,
            'claim_type_weights':        sensitivity,
            'emotional_proximity':       round(emotional_proximity, 3),
            'political_match_llm':       round(llm_political_match, 3),
            'political_match_profile':   round(
                0.5 * llm_political_match + 0.5 * (
                    1.0 if (claim_political == profile_political_direction
                            and claim_political != 'neutral')
                    else 0.0
                ) if profile_political_direction not in ('unknown', '')
                else llm_political_match,
                3,
            ),
            'political_match_blended':   round(political_match, 3),
            'framing_match':             round(framing_match, 3),
            'gender_bias_match':         round(gender_bias_match, 3),
            'ethnic_bias_match':         round(ethnic_bias_match, 3),
            'trauma_bias_match':         round(trauma_bias_match, 3),
            'profile_direction':         profile_political_direction,
            'bias_gate':                 round(bias_gate, 3),
            'emotional_gate':            round(emotional_gate, 3),
            'structural_signal':         structural_signal,
            'sri_lankan_bias_flags': {
                'gender': has_gender_bias,
                'ethnic': has_ethnic_bias,
                'trauma': has_trauma_bias,
            },
            'raw_alignment':             round(raw_alignment, 3),
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
        - news_media:     Political and framing bias most relevant — ownership and
                          editorial stance shape every story.
        - official_data:  Methodology transparency matters most; political/emotional
                          signals are weak in official documents.
        - expert_opinion: Framing reflects theoretical framework.
        - social_media:   Emotional and ethnic/tribal signals dominate.
        - unknown:        No adjustment — use base weights unchanged.

        Multipliers are applied then renormalised so the weights sum to the
        same total as the original (preserving the overall penalty scale).
        """
        _MULTIPLIERS = {
            'news_media':     {'political': 1.3, 'framing': 1.2, 'emotional': 0.9},
            'official_data':  {'political': 0.5, 'emotional': 0.5, 'framing': 0.7},
            'expert_opinion': {'framing': 1.4,   'political': 0.8, 'emotional': 0.7},
            'social_media':   {'emotional': 1.4,  'ethnic': 1.3,   'political': 1.2},
        }
        multipliers = _MULTIPLIERS.get(source_type)
        if not multipliers:
            return base_sensitivity

        adjusted       = {k: v * multipliers.get(k, 1.0) for k, v in base_sensitivity.items()}
        original_total = sum(base_sensitivity.values()) or 1.0
        adjusted_total = sum(adjusted.values()) or 1.0
        scale          = original_total / adjusted_total
        return {k: round(v * scale, 4) for k, v in adjusted.items()}

    # ------------------------------------------------------------------
    # Gemini call — returns raw dimensions only
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize(text: str, max_len: int) -> str:
        text = ''.join(c for c in text if unicodedata.category(c) not in ('So', 'Cs'))
        replacements = {
            '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
            '\u2013': '-', '\u2014': '-', '\u2026': '...', '\u00a0': ' ',
        }
        for orig, repl in replacements.items():
            text = text.replace(orig, repl)
        # Replace only unescaped double-quotes to keep prompt safe;
        # do not strip all backslashes (would mangle URLs and escape sequences)
        text = text.replace('"', "'")
        return text[:max_len].strip()

    def _call_gemini(self, claim: str, sources: List[Dict]) -> Optional[Dict]:
        """
        Single Gemini 2.5 Pro call with response_mime_type="application/json".

        The MIME type forces valid JSON at the sampling level — not merely
        via prompt instruction — making parse failures much rarer.

        Returns parsed dict with claim_bias + sources[], each containing raw
        dimensions only (no alignment).  Returns None on any failure.
        """
        source_lines = []
        for i, src in enumerate(sources, 1):
            snippet  = self._sanitize(src.get('snippet') or '', 800)
            domain   = (src.get('source', '') or src.get('link', ''))[:80]
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

        client     = self._get_client()
        last_error = None

        for model_name in self.FALLBACK_MODELS:
          for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type='application/json',
                    ),
                )
                raw = response.text.strip()

                # response_mime_type=application/json means no fences,
                # but strip defensively in case
                if raw.startswith('```'):
                    raw = raw.split('```')[1]
                    if raw.startswith('json'):
                        raw = raw[4:]
                raw = raw.strip()

                parsed = json.loads(raw)

                # Normalise claim_bias
                cb             = parsed.get('claim_bias', {})
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

                # Normalise sources, map 1-based index → 0-based
                _valid_source_types = {
                    'news_media', 'official_data', 'expert_opinion',
                    'social_media', 'unknown',
                }
                src_map: Dict[int, Dict] = {}
                for item in parsed.get('sources', []):
                    # FIX: default was 0 → idx=-1 (silently dropped).
                    # Default to 1 so a missing index maps to position 0.
                    idx = int(item.get('index', 1)) - 1
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
                err_str = str(e)
                print(f"  [ClaimBiasAnalyzer] Gemini call failed (attempt {attempt + 1}/3, "
                      f"model={model_name.split('/')[-1]}): {e}")

                is_daily_quota = (
                    ('429' in err_str or 'RESOURCE_EXHAUSTED' in err_str)
                    and any(w in err_str.lower() for w in ('quota', 'daily', 'per day'))
                )
                is_transient = (
                    '503' in err_str or 'UNAVAILABLE' in err_str
                    or 'RemoteProtocolError' in err_str
                    or 'Server disconnected' in err_str
                )
                is_rate_limit = (
                    ('429' in err_str or 'RESOURCE_EXHAUSTED' in err_str)
                    and not is_daily_quota
                )

                if is_daily_quota:
                    print(f"  [ClaimBiasAnalyzer] {model_name.split('/')[-1]} daily quota exhausted, "
                          f"switching model")
                    break  # try next model immediately

                if attempt < 2:
                    if is_rate_limit:
                        delay = 15.0
                    elif is_transient:
                        delay = 5.0 + attempt * 3
                    else:
                        delay = 3.0
                    time.sleep(delay)
                    continue

                # 3 attempts exhausted on this model — try next
                print(f"  [ClaimBiasAnalyzer] {model_name.split('/')[-1]} exhausted, trying next model")
                break

        print(f"  [ClaimBiasAnalyzer] All models failed — using defaults. Last error: {last_error}")
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
            {'claim_type': 'general', 'emotional_tone': 0.0,
             'political_direction': 'neutral', 'framing_type': 'neutral'},
            raw,
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
            'bias_analysis_failed':  True,
        }
        return {
            'claim_bias':           claim_bias,
            'sources':              [self._default_source(i) for i in range(len(sources))],
            'bias_analysis_failed': True,
        }
