from typing import List, Dict
import json

from src.bias_detection.bias_detection import BiasDetector

# ---------------------------------------------------------------------------
# Social-media authority constants
# ---------------------------------------------------------------------------
_SOCIAL_MEDIA_PLATFORMS = frozenset({
    'facebook.com', 'x.com', 'twitter.com', 'youtube.com',
    'instagram.com', 'tiktok.com', 'threads.net'
})

# Maps platform → { handle (lowercase) → parent domain }
# Only handles that belong to recognised Sri Lankan / international news orgs.
_KNOWN_NEWS_SOCIAL_HANDLES: Dict[str, Dict[str, str]] = {
    'facebook.com': {
        'adaderana':        'adaderana.lk',
        'hirunewsenglish':  'hirunews.lk',
        'hirunews':         'hirunews.lk',
        'ceylontoday':      'ceylontoday.lk',
        'dailymirrorlk':    'dailymirror.lk',
        'newsfirstlk':      'newsfirst.lk',
        'themorninglk':     'themorning.lk',
        'economynext':      'economynext.com',
        'sundaytimesssl':   'sundaytimes.lk',
        'newswire.lk':      'newswire.lk',
    },
    'youtube.com': {
        'adaderana':        'adaderana.lk',
        'hirunewsonline':   'hirunews.lk',
    },
    'x.com': {
        'adaderana':        'adaderana.lk',
        'hirunews':         'hirunews.lk',
    },
    'twitter.com': {
        'adaderana':        'adaderana.lk',
        'hirunews':         'hirunews.lk',
    },
}

class EvidenceWeighter:
    def __init__(self):
        """
        Enhanced evidence weighting with bias profiling integration
        """
        # Load pre-computed bias profiles
        self.bias_profiles = self._load_bias_profiles()
        print(f" Evidence weighter loaded with {len(self.bias_profiles)} source profiles")

    def _load_bias_profiles(self) -> dict:
        """Load pre-computed bias profiles from bias detection pipeline"""
        try:
            with open("data/bias_profiles.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(" Bias profiles not found. Using text-based analysis only.")
            return {}

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for bias profile lookup"""
        try:
            if url.startswith('http'):
                url = url.split('://', 1)[1]
            domain = url.split('/')[0]
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return ""

    def calculate_bias_claim_alignment(self, claim: str, bias_analysis: Dict, evidence_url: str = None) -> float:
        """
        Enhanced bias-claim alignment calculation using both source profiles and text analysis
        """
        # Check if we have source profile information
        domain = self._extract_domain(evidence_url) if evidence_url else None
        has_profile = domain in self.bias_profiles if domain else False
        
        if has_profile:
            return self._calculate_alignment_with_profile(claim, bias_analysis, domain)
        else:
            return self._calculate_alignment_text_only(claim, bias_analysis)

    def _calculate_alignment_with_profile(self, claim: str, bias_analysis: Dict, domain: str) -> float:
        """Calculate alignment when source profile is available"""
        # Enhanced political keyword detection
        government_keywords = ["government", "administration", "policy", "minister", 
                            "president", "cabinet", "parliament", "NPP", "opposition", 
                            "election", "political", "party", "MP", "MPs", "representative"]
        
        claim_lower = claim.lower()
        is_political_claim = any(kw in claim_lower for kw in government_keywords)
        
        # Get source bias information
        source_profile = self.bias_profiles[domain]
        source_bias_score = abs(source_profile["bias_score"])  # Absolute bias magnitude (0-50)
        source_confidence = source_profile["confidence"]
        
        # Get text-level bias
        political_bias = bias_analysis.get("political_stance", {}).get("stance_score", 0.0)
        emotional_bias = abs(bias_analysis.get("emotional_tone", {}).get("emotion_score", 0.0))
        framing_bias = abs(bias_analysis.get("framing_bias", {}).get("framing_score", 0.0))
        text_bias = (political_bias + emotional_bias + framing_bias) / 3
        
        if is_political_claim:
            # For political claims, source bias is highly relevant
            source_component = (source_bias_score / 50) * 0.8  # Increase weight for political claims. 80% weight to source
            text_component = text_bias * 0.2           # 20% weight to text
            alignment = source_component + text_component
            alignment *= source_confidence  # Weight by profile confidence
        else:
            # For non-political claims like "female MPs representation", still consider bias but less
            source_component = (source_bias_score / 50) * 0.4  # Moderate weight for factual claims
            text_component = text_bias * 0.6
            alignment = source_component + text_component
            alignment *= (source_confidence * 0.7)  # Moderate confidence impact
        
        return min(alignment, 1.0)

    def _calculate_alignment_text_only(self, claim: str, bias_analysis: Dict) -> float:
        """Original alignment calculation for sources without profiles"""
        government_keywords = ["government", "administration", "policy", "minister", 
                              "president", "cabinet", "parliament"]

        claim_lower = claim.lower()
        is_political_claim = any(kw in claim_lower for kw in government_keywords)

        political_bias = bias_analysis.get("political_stance", {}).get("stance_score", 0.0)
        emotional_bias = abs(bias_analysis.get("emotional_tone", {}).get("emotion_score", 0.0))
        framing_bias = abs(bias_analysis.get("framing_bias", {}).get("framing_score", 0.0))
        
        if is_political_claim:
            alignment = (political_bias + emotional_bias + framing_bias) / 3
        else:
            alignment = (political_bias + emotional_bias + framing_bias) / 6
        
        return min(alignment, 1.0)
    
    def _extract_social_handle(self, url: str) -> str:
        """Extract the page/channel handle from a social media URL."""
        try:
            # Strip protocol + domain, keep path
            path = url.split('://', 1)[-1]
            path = path.split('/', 1)[1] if '/' in path else ''
            handle = path.split('/')[0].split('?')[0].lstrip('@').lower()
            # These are route segments, not handles
            if handle in ('watch', 'posts', 'videos', 'channel', 'c', 'user', 'reel', 'stories', 'p', ''):
                return ''
            return handle
        except Exception:
            return ''

    def _lookup_domain_authority(self, domain: str) -> float:
        """Return the authority score for a bare domain string.

        Tiers are grounded in FactCheck.lk's stated source hierarchy
        (interview with Mahoshadi, FactCheck.lk manager):
          - Primary verification: DCS, Central Bank, IMF, ADB
          - Official government: ministries, presidential office
          - Context-only: Parliament Hansard (not primary verification)
          - Fact-checkers, quality news, international press
        """
        # Primary statistical/financial verification sources
        primary_stats = [
            'statistics.gov.lk',            # Department of Census and Statistics (DCS)
            'cbsl.gov.lk', 'cbsl.lk',       # Central Bank of Sri Lanka
            'imf.org',                       # IMF
            'adb.org',                       # Asian Development Bank
        ]
        if any(d in domain for d in primary_stats):
            return 3.0

        # Official government sources (authoritative but not primary data)
        official_gov = [
            'presidentsoffice.gov.lk', 'president.gov.lk',
            'pmoffice.gov.lk', 'mfa.gov.lk',
            'treasury.gov.lk', 'finance.gov.lk',
            'elections.gov.lk',
        ]
        if any(d in domain for d in official_gov):
            return 2.7

        # Catch-all for any remaining .gov.lk domains
        if '.gov.lk' in domain:
            return 2.5

        # International development/financial institutions
        intl_institutions = ['worldbank.org', 'un.org', 'undp.org', 'who.int', 'unicef.org']
        if any(d in domain for d in intl_institutions):
            return 2.3

        # Parliament — Mahoshadi: used for context only, not primary verification
        if 'parliament.lk' in domain:
            return 2.0

        # Fact-checkers
        fact_checkers = ['factcheck.lk', 'factcrescendo.com', 'boomlive.in', 'factly.in']
        if any(d in domain for d in fact_checkers):
            return 1.8

        # Quality Sri Lankan news
        quality_news = ['dailymirror.lk', 'economynext.com', 'island.lk',
                        'newsfirst.lk', 'adaderana.lk', 'ft.lk']
        if any(d in domain for d in quality_news):
            return 1.3

        # International press
        international = ['reuters.com', 'bbc.com', 'cnn.com', 'aljazeera.com', 'apnews.com']
        if any(d in domain for d in international):
            return 1.2

        return 1.0

    def _get_authority_weight(self, evidence_url: str, dataset_source: str = None, evidence: dict = None) -> float:
        """
        Authority weight calculation.
        Social media URLs are classified as:
          - Known news-org pages  → parent org authority × 0.9
          - Unverified social     → 0.55 (below generic news baseline)
        """
        # Dataset sources — keys match Qdrant collection names.
        # Tiers based on FactCheck.lk's source hierarchy (Mahoshadi interview):
        #   3.0 = primary verification (DCS data, Central Bank reports)
        #   2.8 = official financial/policy announcements
        #   2.7 = legal judgments
        #   2.5 = official administrative/legislative records
        #   2.0 = context-only (Hansard per Mahoshadi), statistics reports
        if dataset_source:
            dataset_authority = {
                # Economic / Financial — primary verification sources
                'central_bank_reports':         3.0,
                'treasury_press_releases':       2.8,
                'fisheries_statistics':          2.5,
                'tourism_reports':               2.5,
                # Legal / Judicial
                'supreme_court':                 2.7,
                'appeal_court':                  2.5,
                'acts':                          2.5,
                'bills':                         2.3,
                # Presidential / Cabinet
                'cabinet_decisions':             2.5,
                'pmd_press_releases':            2.5,
                # Gazettes / Admin
                'extraordinary_gazettes_2020s':  2.3,
                'extraordinary_gazettes_2010s':  2.3,
                # Parliamentary — context only per Mahoshadi
                'hansard_2020s':                 2.0,
                'hansard_2010s':                 2.0,
                'hansard_2000s':                 2.0,
                'hansard':                       2.0,
                # Other
                'police_press_releases':         2.0,
                'education_publications':        1.8,
                'news':                          1.5,
            }
            return dataset_authority.get(dataset_source, 2.0)

        if not evidence_url:
            return 1.0

        domain = self._extract_domain(evidence_url)

        # Social media: determine whether it belongs to a recognised news org
        if domain in _SOCIAL_MEDIA_PLATFORMS:
            handle = self._extract_social_handle(evidence_url)
            parent_domain = _KNOWN_NEWS_SOCIAL_HANDLES.get(domain, {}).get(handle)
            if parent_domain:
                # Known news org's official social page — slight discount vs website
                return self._lookup_domain_authority(parent_domain) * 0.9
            # Unverified social media account
            return 0.55

        return self._lookup_domain_authority(domain)
    
    def _get_temporal_relevance_factor(self, claim: str, evidence: Dict) -> float:
        """
        Penalise sources whose content doesn't mention the specific year(s) in the claim.

        If a claim says "2020" but an article only discusses 2025 figures, that article
        is temporally mismatched and should receive a lower weight regardless of its
        authority or recency.

        Returns:
            1.0  — evidence content mentions at least one of the claim's years (match)
            0.4  — evidence mentions years but none overlap with the claim (mismatch)
            0.85 — evidence has no year mentions at all (mild uncertainty)
        """
        import re

        # Extract years from the claim
        claim_years = set(re.findall(r'\b(?:19|20)\d{2}\b', claim))
        if not claim_years:
            return 1.0  # Claim has no specific year — skip temporal check

        # Get the evidence text to inspect
        snippet = (evidence.get('snippet', '') or evidence.get('passage', '')
                   or evidence.get('text', '') or '')

        # If the snippet mentions any of the claim's years, it's temporally relevant
        for year in claim_years:
            if year in snippet:
                return 1.0

        # Check if the snippet discusses a completely different year
        content_years = set(re.findall(r'\b(?:19|20)\d{2}\b', snippet))
        if content_years:
            # Has years but none overlap with the claim — temporal mismatch
            return 0.4

        # No year mentions in content — mild uncertainty
        return 0.85

    def _get_recency_weight(self, evidence: Dict, claim: str = '') -> float:
        """
        Calculate recency weight.

        For current/recent claims: newer sources score higher (proximity to today).
        For historical claims (claim references a specific past year): recency is
        measured as proximity to the claimed year instead of proximity to today.
        A 2021 article about a 2020 claim is contemporary reporting and should not
        be penalised just because it is now several years old.

        Historical claim detection: if the claim contains a four-digit year that is
        more than one year in the past, the claim is treated as historical and the
        nearest such year is used as the reference point.
        """
        import re
        from datetime import datetime

        # Determine whether this is a historical claim and, if so, which year to
        # use as the reference point.
        current_year = datetime.now().year
        claim_years = [int(y) for y in re.findall(r'\b(?:19|20)\d{2}\b', claim)]
        historical_years = [y for y in claim_years if current_year - y > 1]
        is_historical = bool(historical_years)

        def _parse_date(evidence: Dict):
            """Return a datetime parsed from the evidence date field or URL, or None."""
            date_str = evidence.get('date', '')
            if date_str:
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                    try:
                        return datetime.strptime(date_str[:10], fmt)
                    except Exception:
                        pass
                for fmt in ['%b %d, %Y', '%B %d, %Y', '%d %b %Y', '%d %B %Y']:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except Exception:
                        pass
            # Fall back to year extracted from URL (mid-year estimate)
            url = evidence.get('link', '') or evidence.get('url', '')
            m = re.search(r'/(20\d{2})/', url)
            if m:
                return datetime(int(m.group(1)), 7, 1)
            return None

        try:
            evidence_date = _parse_date(evidence)

            if evidence_date is None:
                return 0.7  # No date available — moderate default

            if is_historical:
                # Proximity to the nearest claimed year.
                #   0–1 years from event  → 1.0  (contemporary reporting)
                #   1–3 years from event  → 0.9  (near-contemporary)
                #   >3 years from event   → 0.75 (retrospective — still useful)
                nearest = min(historical_years, key=lambda y: abs(evidence_date.year - y))
                years_from_event = abs(evidence_date.year - nearest)
                if years_from_event <= 1:
                    return 1.0
                elif years_from_event <= 3:
                    return 0.9
                else:
                    return 0.75
            else:
                # Original logic for current/recent claims.
                days_old = (datetime.now() - evidence_date).days
                if days_old > 365:
                    return 0.3
                elif days_old > 180:
                    return 0.6
                elif days_old > 90:
                    return 0.8
                else:
                    return 1.0

        except Exception:
            return 0.7

    def calculate_evidence_weight(self, verification_confidence: float, bias_alignment: float,
                                 base_credibility: float = 0.7, evidence_url: str = None,
                                 evidence: Dict = None, framing_consistency: str = 'unknown',
                                 claim: str = '', disable_bias: bool = False) -> float:
        """
        Enhanced evidence weighting with source credibility, authority, recency,
        and framing consistency.

        framing_consistency values:
          'consistent'   — article framing matches source's known political lean
                           → amplify bias penalty (source behaving as expected, less independent)
          'inconsistent' — article framing goes against the source's known lean
                           → reduce bias penalty (credibility boost, against own bias)
          'unknown'      — no framing analysis available → no adjustment
        """
        # Adjust base credibility based on source profile if available
        domain = self._extract_domain(evidence_url) if evidence_url else None
        if domain and domain in self.bias_profiles:
            profile = self.bias_profiles[domain]
            profile_credibility = 0.5 + (profile["confidence"] * 0.3)  # Range: 0.5-0.8
            base_credibility = max(base_credibility, profile_credibility)

        # Authority and recency weights
        dataset_source = evidence.get('dataset_source') if evidence else None
        authority_weight = self._get_authority_weight(evidence_url, dataset_source=dataset_source, evidence=evidence)
        recency_weight = self._get_recency_weight(evidence, claim=claim) if evidence else 1.0

        # Bias penalty — adjusted by framing consistency
        # consistent   → 1.3× (expected bias behaviour, down-weight more)
        # inconsistent → 0.6× (against own bias, up-weight)
        # unknown      → 1.0× (no change)
        _consistency_multiplier = {'consistent': 1.3, 'inconsistent': 0.6, 'unknown': 1.0}
        consistency_mult = _consistency_multiplier.get(framing_consistency, 1.0)
        bias_penalty = 0.0 if disable_bias else bias_alignment * 0.5 * consistency_mult

        # Cross-encoder relevance score (set on web sources by RelevanceFilter).
        # Qdrant sources don't have this field — default to 1.0 (no penalty).
        relevance_factor = evidence.get('cross_encoder_relevance', 1.0) if evidence else 1.0

        # Temporal relevance — penalise sources whose content discusses a different
        # year than the one mentioned in the claim (e.g. 2025 article for a 2020 claim).
        temporal_factor = self._get_temporal_relevance_factor(claim, evidence) if (claim and evidence) else 1.0

        weight = base_credibility * verification_confidence * (1 - bias_penalty) * authority_weight * recency_weight * relevance_factor * temporal_factor

        print(f"  Debug: conf={verification_confidence:.3f}, bias_align={bias_alignment:.3f}, "
              f"framing={framing_consistency}, authority={authority_weight:.1f}, "
              f"recency={recency_weight:.1f}, relevance={relevance_factor:.3f}, "
              f"temporal={temporal_factor:.2f}, weight={weight:.3f}")

        return max(weight, 0.1)
    
    def weight_all_evidence(self, claim: str, evidence_list: List[Dict],
                            verification_results: List[Dict], bias_analyses: List[Dict],
                            framing_analyses: Dict = None,
                            precomputed_alignments: List[float] = None,
                            disable_bias: bool = False) -> List[Dict]:
        """
        Weight all evidence pieces with enhanced bias profiling.

        framing_analyses       : optional output of FramingAnalyzer / ClaimAwareBiasAnalyzer.
                                 Per-source framing consistency adjusts the bias penalty.
        precomputed_alignments : optional list of claim-relative alignment scores (0-1),
                                 one per source.  When provided these replace the internal
                                 calculate_bias_claim_alignment() call so the weighter uses
                                 scores from ClaimAwareBiasAnalyzer directly.
        """
        framing_sources = (framing_analyses or {}).get('sources', [])
        # Build index → framing entry map for O(1) lookup
        framing_by_index = {entry.get('index', i): entry for i, entry in enumerate(framing_sources)}

        weighted_evidence = []

        for i, (evidence, verification, bias) in enumerate(
                zip(evidence_list, verification_results, bias_analyses)):
            evidence_url = evidence.get("link", "")

            # Use pre-computed claim-relative alignment when available,
            # otherwise fall back to the profile-based calculation.
            if precomputed_alignments and i < len(precomputed_alignments):
                alignment = precomputed_alignments[i]
            else:
                alignment = self.calculate_bias_claim_alignment(claim, bias, evidence_url)

            # Pull framing consistency for this source (if available)
            framing_entry = framing_by_index.get(i, {})
            framing_consistency = framing_entry.get('consistency_with_profile', 'unknown')

            # Calculate weight — now includes framing consistency adjustment
            weight = self.calculate_evidence_weight(
                verification_confidence=verification["confidence"],
                bias_alignment=alignment,
                evidence_url=evidence_url,
                evidence=evidence,
                framing_consistency=framing_consistency,
                claim=claim,
                disable_bias=disable_bias,
            )

            explanation = self._generate_weight_explanation(
                alignment, weight, framing_entry, evidence_url, evidence, verification, claim
            )

            weighted_evidence.append({
                "evidence": evidence,
                "verification": verification,
                "bias_analysis": bias,
                "bias_alignment": alignment,
                "weight": weight,
                "explanation": explanation,
                "framing_entry": framing_entry,  # carries loaded_phrases, explanation, direction
            })

        return weighted_evidence
        
    def _generate_weight_explanation(
        self,
        alignment: float,
        weight: float,
        framing_entry: Dict,
        evidence_url: str = None,
        evidence: Dict = None,
        verification: Dict = None,
        claim: str = '',
    ) -> str:
        """
        Full weight breakdown explanation using LLaMA bias dimensions.

        Shows every multiplicative component so the user can see exactly
        why a source received its final weight.
        """
        domain = self._extract_domain(evidence_url) if evidence_url else None

        # --- Recompute display components (same logic as calculate_evidence_weight) ---
        base_credibility = 0.7
        if domain and domain in self.bias_profiles:
            profile = self.bias_profiles[domain]
            base_credibility = max(base_credibility, 0.5 + profile["confidence"] * 0.3)

        dataset_source = evidence.get('dataset_source') if evidence else None
        authority_weight  = self._get_authority_weight(evidence_url, dataset_source=dataset_source, evidence=evidence)
        recency_weight    = self._get_recency_weight(evidence) if evidence else 1.0
        temporal_factor   = self._get_temporal_relevance_factor(claim, evidence) if (claim and evidence) else 1.0
        relevance_factor  = evidence.get('cross_encoder_relevance', 1.0) if evidence else 1.0
        nli_confidence    = verification.get('confidence', 0.0) if verification else 0.0

        framing_consistency = framing_entry.get('consistency_with_profile', 'unknown')
        _consistency_multiplier = {'consistent': 1.3, 'inconsistent': 0.6, 'unknown': 1.0}
        consistency_mult = _consistency_multiplier.get(framing_consistency, 1.0)
        bias_penalty = alignment * 0.5 * consistency_mult

        # --- LLaMA bias dimensions ---
        framing_type      = framing_entry.get('framing_type', framing_entry.get('framing_direction', 'unknown'))
        political_dir     = framing_entry.get('political_direction', 'unknown')
        emotional_tone    = framing_entry.get('emotional_tone', 0.0)
        loaded_phrases    = framing_entry.get('loaded_phrases', [])
        llm_explanation   = framing_entry.get('explanation', '')

        # --- Bias profile line (if available) ---
        profile_line = ''
        if domain and domain in self.bias_profiles:
            p = self.bias_profiles[domain]
            profile_line = (
                f"Profile ({domain}): {p.get('interpretation', 'unknown')} "
                f"(score {p['bias_score']:+.1f}, confidence {p['confidence']:.0%})"
            )

        # --- Assemble formula breakdown ---
        parts = [
            f"Weight breakdown:",
            f"  {base_credibility:.2f} base credibility",
            f"× {nli_confidence:.2f} NLI confidence",
            f"× {1 - bias_penalty:.2f} bias factor  (1 − {alignment:.2f} align × 0.5 × {consistency_mult:.1f} framing-mult)",
            f"× {authority_weight:.2f} authority",
            f"× {recency_weight:.2f} recency",
        ]
        if relevance_factor < 1.0:
            parts.append(f"× {relevance_factor:.2f} relevance")
        if temporal_factor < 1.0:
            parts.append(f"× {temporal_factor:.2f} temporal")
        parts.append(f"= {weight:.3f} final weight")

        # --- Bias signal summary (LLaMA) ---
        bias_parts = [
            f"Bias (LLaMA): framing={framing_type}, direction={political_dir}, "
            f"emotional_tone={emotional_tone:+.2f}"
        ]
        if loaded_phrases:
            quoted = ', '.join(f'"{p}"' for p in loaded_phrases[:3])
            bias_parts.append(f"Loaded phrases: {quoted}")
        if llm_explanation:
            bias_parts.append(f"Note: {llm_explanation}")
        if profile_line:
            bias_parts.append(profile_line)

        return '\n'.join(parts + bias_parts)

    def get_source_diversity_score(self, evidence_list: List[Dict]) -> float:
        """
        Calculate diversity score based on bias profile distribution
        """
        domains = []
        for evidence in evidence_list:
            domain = self._extract_domain(evidence.get("url", ""))
            if domain:
                domains.append(domain)
        
        if not domains:
            return 0.0
        
        # Count bias categories represented
        bias_categories = set()
        for domain in domains:
            if domain in self.bias_profiles:
                score = self.bias_profiles[domain]["bias_score"]
                if score < -10:
                    bias_categories.add("opposition")
                elif score > 10:
                    bias_categories.add("pro_government")
                else:
                    bias_categories.add("neutral")
            else:
                bias_categories.add("unknown")
        
        # Diversity score: more bias categories = higher diversity
        max_categories = 4  # opposition, neutral, pro_government, unknown
        diversity_score = len(bias_categories) / max_categories
        
        return diversity_score

class VerdictGenerator:
    def __init__(self):
        """Enhanced verdict generation with bias-aware confidence scoring"""
        pass
    
    def _is_factual_claim(self, weighted_evidence: List[Dict]) -> bool:
        """
        Detect if this is a factual claim that shouldn't be marked as conflicting
        """
        # Create EvidenceWeighter instance to access authority weight method
        weighter = EvidenceWeighter()
        
        # Check for high-authority sources
        authority_weights = []
        for item in weighted_evidence:
            evidence_url = item["evidence"].get("link", "")
            auth_weight = weighter._get_authority_weight(evidence_url, evidence=item["evidence"])
            authority_weights.append(auth_weight)
        
        # If we have official sources (weight > 1.5), treat as factual
        has_official_sources = any(w > 1.5 for w in authority_weights)
        
        # If majority of sources are high-authority, treat as factual
        high_authority_ratio = sum(1 for w in authority_weights if w > 1.2) / len(authority_weights)
        
        return has_official_sources or high_authority_ratio > 0.6

    def _detect_newly_developing(self, weighted_evidence: List[Dict]) -> Dict:
        """
        Detect if this is a newly developing situation where official sources
        haven't yet reported — i.e. all sources are recent web/social media only.

        Signals checked:
          1. No database (government document) sources
          2. No high-authority official sources (authority >= 2.5)
          3. >= 70% of dated sources are within 14 days
        """
        from datetime import datetime, timedelta
        import re

        OFFICIAL_AUTHORITY_THRESHOLD = 2.5
        NEWLY_DEVELOPING_DAYS = 14
        MIN_RECENT_RATIO = 0.70

        now = datetime.now()
        cutoff = now - timedelta(days=NEWLY_DEVELOPING_DAYS)

        weighter = EvidenceWeighter()
        has_official_source = False
        has_database_source = False
        dated_sources = 0
        recent_sources = 0

        for item in weighted_evidence:
            ev = item.get("evidence", {})
            link = ev.get("link", "") or ev.get("url", "")

            # Database source check
            if ev.get("source_type") == "database":
                has_database_source = True

            # Official/high-authority source check
            auth = weighter._get_authority_weight(link, evidence=ev)
            if auth >= OFFICIAL_AUTHORITY_THRESHOLD:
                has_official_source = True

            # Recency check — try date field first, then snippet prefix
            date_str = ev.get("date", "") or ev.get("published_date", "")
            if not date_str:
                snippet = ev.get("snippet", "") or (ev.get("content", "") or "")[:60]
                m = re.match(
                    r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4})',
                    snippet
                )
                if m:
                    date_str = m.group(1)

            if date_str:
                dated_sources += 1
                ev_date = None
                for fmt in ['%Y-%m-%d', '%b %d, %Y', '%B %d, %Y', '%d/%m/%Y']:
                    try:
                        ev_date = datetime.strptime(date_str[:12].strip(), fmt)
                        break
                    except Exception:
                        continue
                if ev_date and ev_date >= cutoff:
                    recent_sources += 1

        no_official = not has_official_source and not has_database_source
        recent_ratio = (recent_sources / dated_sources) if dated_sources > 0 else None
        mostly_recent = (recent_ratio is None) or (recent_ratio >= MIN_RECENT_RATIO)

        is_newly_developing = no_official and mostly_recent

        warning = None
        if is_newly_developing:
            n = len(weighted_evidence)
            days = NEWLY_DEVELOPING_DAYS
            warning = (
                f"NEWLY DEVELOPING SITUATION — all {n} sources are recent web/social media "
                f"reports (within {days} days). No official or government sources have yet "
                f"reported on this. Treat this verdict with caution; it may change as the "
                f"situation develops."
            )

        return {
            "is_newly_developing": is_newly_developing,
            "warning": warning,
            "has_official_source": has_official_source,
            "has_database_source": has_database_source,
            "recent_source_ratio": round(recent_ratio, 2) if recent_ratio is not None else None,
        }

    def generate_verdict(self, weighted_evidence: List[Dict]) -> Dict:
        """
        Enhanced verdict generation considering source diversity, authority, and factual claim detection
        """
        if not weighted_evidence:
            return {
                "verdict": "UNCERTAIN",
                "confidence": 0.0,
                "explanation": "No valid evidence found"
            }
        
        # Aggregate weighted votes
        support_score = 0.0
        refute_score = 0.0
        neutral_score = 0.0
        
        for item in weighted_evidence:
            label = item["verification"]["label"]
            weight = item["weight"]
            
            if label == "SUPPORTED":
                support_score += weight
            elif label == "REFUTED":
                refute_score += weight
            else:
                neutral_score += weight
        
        total = support_score + refute_score + neutral_score

        if total == 0:
            return {
                "verdict": "UNCERTAIN",
                "confidence": 0.0,
                "explanation": "No valid evidence found"
            }

        # NEUTRAL sources are abstentions — they don't address the claim and
        # should not dilute the SUPPORTED/REFUTED signal.  Verdict is decided
        # solely on the balance of active (SUPPORTED vs REFUTED) votes; neutral
        # weight is only used to raise aleatoric uncertainty.
        active_total = support_score + refute_score
        active_count = sum(1 for item in weighted_evidence
                           if item["verification"]["label"] != "NEUTRAL")
        # Require at least 2 active (non-neutral) sources before trusting the
        # active balance. A single weak source should not drive the verdict.
        if active_total > 0 and active_count >= 2:
            support_pct = support_score / active_total
            refute_pct  = refute_score  / active_total
        elif active_total > 0 and active_count == 1:
            # Only one active source — treat as uncertain regardless of direction
            support_pct = 0.0
            refute_pct  = 0.0
        else:
            # No active votes at all → uncertain
            support_pct = 0.0
            refute_pct  = 0.0
        neutral_pct = neutral_score / total
        
        # Calculate source diversity boost
        weighter = EvidenceWeighter()
        diversity_score = weighter.get_source_diversity_score([item["evidence"] for item in weighted_evidence])

        # Enhanced diversity boost for Sri Lankan sources
        sri_lankan_sources = sum(1 for item in weighted_evidence 
                                if weighter._extract_domain(item["evidence"].get("link", "")) in weighter.bias_profiles)
        
        if sri_lankan_sources >= 2:  # Multiple Sri Lankan sources with different biases
            diversity_boost = diversity_score * 0.15  # Increased boost
        else:
            diversity_boost = diversity_score * 0.05  # Standard boost
        
        # --- 6-label verdict system ----------------------------------------------
        # 5 labels on the truth scale + 1 off-scale label for no active evidence.
        # Verdict is driven by support_pct (fraction of active weight that supports
        # the claim). The scale is symmetric and continuous:
        #
        #  support_pct    Verdict          Plain label
        #  ≥ 0.80         VERIFIED         True / Verified
        #  0.60–0.80      MOSTLY_TRUE      Mostly True / Largely Supported
        #  0.35–0.60      PARTIALLY_TRUE   Partial / Mixed Evidence
        #  0.20–0.35      MOSTLY_FALSE     Mostly False / Largely Contradicted
        #  < 0.20         REFUTED          False / Refuted
        #  active_count<2 UNCERTAIN        Unverified / Insufficient Evidence
        #                                  (off-scale — no sources took a position)
        #
        # This replaces the binary SUPPORTED/REFUTED split and correctly handles
        # claims that are partially true (e.g. appointment confirmed, but "first
        # female" qualifier is wrong) without needing factual/opinion branching.
        # -------------------------------------------------------------------------

        # --- Compound claim partial-truth detection ----------------------------------
        # Problem: claims like "X appointed Y as first-ever Z" are compound:
        #   Part A "X appointed Y as Z"            → TRUE  (many NEUTRAL sources confirm)
        #   Part B "Y is the first-ever Z"         → FALSE (REFUTED sources catch it)
        # With the superlative NEUTRAL rule, Part A sources are NEUTRAL (correct —
        # they don't confirm "first"), leaving support_pct = 0 and verdict = REFUTED.
        # But REFUTED implies the whole claim is fabricated; MOSTLY_FALSE is more
        # accurate (underlying event is real, qualifier is wrong).
        #
        # Trigger condition:
        #   • No SUPPORTED sources at all (support_pct == 0)
        #   • At least 2 REFUTED sources (genuine superlative contradiction found)
        #   • ≥3 confident (≥0.70) NEUTRAL sources from authorities whose
        #     combined weight is at least 2× the REFUTED weight
        #     (indicates substantial confirmation of the underlying event)
        # Threshold is 0.70 (not 0.80) because PMD/cabinet sources regularly
        # return 70% confidence on NEUTRAL verdicts for tangentially-related docs.
        # When triggered, verdict becomes MOSTLY_FALSE with capped confidence ≤0.75.
        # ---------------------------------------------------------------------------
        _is_compound_partial = False
        if support_score == 0 and refute_score > 0:
            _hc_neutral_weight = sum(
                item["weight"] for item in weighted_evidence
                if item["verification"]["label"] == "NEUTRAL"
                and item["verification"]["confidence"] >= 0.70
            )
            _hc_neutral_count = sum(
                1 for item in weighted_evidence
                if item["verification"]["label"] == "NEUTRAL"
                and item["verification"]["confidence"] >= 0.70
            )
            _refute_count = sum(
                1 for item in weighted_evidence
                if item["verification"]["label"] == "REFUTED"
            )
            if (_hc_neutral_count >= 3
                    and _refute_count >= 2
                    and _hc_neutral_weight >= refute_score * 2):
                _is_compound_partial = True

        if active_count < 2:
            # Fewer than 2 sources took a position — not enough to place on scale
            verdict    = "UNCERTAIN"
            confidence = 0.0
        elif _is_compound_partial:
            # Compound claim: underlying event confirmed (NEUTRAL) but a qualifier
            # (e.g. "first female", "only ever") is factually wrong (REFUTED)
            verdict    = "MOSTLY_FALSE"
            confidence = min(refute_pct + diversity_boost, 0.75)
        elif support_pct >= 0.80:
            verdict    = "VERIFIED"
            confidence = min(support_pct + diversity_boost, 0.95)
        elif support_pct >= 0.60:
            verdict    = "MOSTLY_TRUE"
            confidence = min(support_pct + diversity_boost, 0.95)
        elif support_pct >= 0.35:
            verdict    = "PARTIALLY_TRUE"
            # Confidence reflects how clearly the mixed nature is established,
            # not which direction it leans
            confidence = min(max(support_pct, refute_pct) + diversity_boost, 0.95)
        elif support_pct >= 0.20:
            verdict    = "MOSTLY_FALSE"
            confidence = min(refute_pct + diversity_boost, 0.95)
        else:
            verdict    = "REFUTED"
            confidence = min(refute_pct + diversity_boost, 0.95)

        is_factual = self._is_factual_claim(weighted_evidence)  # kept for metadata only

        # Compute uncertainty first so bias_induced can reduce confidence
        uncertainty = self._decompose_uncertainty(weighted_evidence)

        # Reduce confidence proportionally to bias-induced uncertainty.
        # High bias alignment across sources means the evidence pool is less
        # independent — the verdict should reflect that.
        # Scale of 0.1 means a maximum ~10% reduction at bias_induced=1.0.
        bias_penalty = uncertainty["bias_induced"] * 0.1
        confidence = confidence - bias_penalty

        confidence = min(confidence, 0.95)  # Cap at 95% — no real-world verdict is certain
        confidence = max(confidence, 0.0)   # Floor at 0

        newly_developing = self._detect_newly_developing(weighted_evidence)

        return {
            "verdict": verdict,
            "confidence": confidence,
            "support_score": support_pct,
            "refute_score": refute_pct,
            "neutral_score": neutral_pct,
            "diversity_score": diversity_score,
            "is_factual_claim": is_factual,
            "uncertainty_decomposition": uncertainty,
            "newly_developing": newly_developing,
        }
    
    def _decompose_uncertainty(self, weighted_evidence: List[Dict]) -> Dict:
        """
        Enhanced uncertainty decomposition including bias-induced uncertainty
        """
        alignments = [item["bias_alignment"] for item in weighted_evidence]
        labels = [item["verification"]["label"] for item in weighted_evidence]
        
        # Epistemic uncertainty: disagreement between source verdicts (weighted).
        # std(confidence) was always ~0 because LLaMA returns uniform 0.9 scores.
        # Instead: measure the fraction of weight on the minority verdict.
        # If all sources agree → epistemic = 0; if evenly split → epistemic = 0.5.
        weights = [item["weight"] for item in weighted_evidence]
        total_w = sum(weights) or 1.0
        support_w = sum(w for item, w in zip(weighted_evidence, weights)
                        if item["verification"]["label"] == "SUPPORTED")
        refute_w  = sum(w for item, w in zip(weighted_evidence, weights)
                        if item["verification"]["label"] == "REFUTED")
        minority_w = min(support_w, refute_w)
        epistemic = float(minority_w / total_w)
        
        # Aleatoric uncertainty: lack of directly relevant evidence.
        # Count sources that actually addressed the claim (SUPPORTED or REFUTED).
        active_sources = sum(1 for item in weighted_evidence
                             if item["verification"]["label"] != "NEUTRAL")
        ideal_sources = 5
        aleatoric = max(0.0, (ideal_sources - active_sources) / ideal_sources)
        
        # Enhanced bias-induced uncertainty
        high_bias_count = sum(1 for a in alignments if a > 0.5)
        high_bias_ratio = high_bias_count / len(alignments) if alignments else 0.0
        label_diversity = len(set(labels)) / 3 if labels else 1.0  # 3 possible labels
        bias_induced = high_bias_ratio * label_diversity
        
        return {
            "epistemic": round(epistemic, 3),
            "aleatoric": round(aleatoric, 3),
            "bias_induced": round(bias_induced, 3),
            "explanation": self._explain_uncertainty(epistemic, aleatoric, bias_induced)
        }
    
    def _explain_uncertainty(self, epistemic, aleatoric, bias_induced) -> str:
        """Enhanced uncertainty explanation"""
        explanations = []
        
        if epistemic > 0.3:
            explanations.append("Model confidence varies across sources")
        if aleatoric > 0.4:
            explanations.append("Limited evidence available")
        if bias_induced > 0.4:
            explanations.append("Sources with different biases disagree")
        
        if not explanations:
            return "Uncertainty is low across all factors"
        
        return "; ".join(explanations)
    







    def weight_all_evidence(self, claim: str, evidence_list: List[Dict], verification_results: List[Dict], bias_analyses: List[Dict]) -> List[Dict]:
        """
        Enhanced weighting with cross-source framing analysis
        """
        # Perform cross-source framing analysis first
        bias_detector = BiasDetector()
        cross_source_analysis = bias_detector.analyze_cross_source_framing(claim, evidence_list, bias_analyses)
        
        weighted_evidence = []
        
        for i, (evidence, verification, bias) in enumerate(zip(evidence_list, verification_results, bias_analyses)):
            evidence_url = evidence.get("link", "")
            source_id = f"source_{i}"
            
            # Enhanced bias analysis including cross-source context
            enhanced_bias = {
                **bias,  # Original individual analysis
                'cross_source_analysis': cross_source_analysis,
                'is_emotional_outlier': self._check_emotional_outlier(source_id, cross_source_analysis),
                'relative_emotional_intensity': self._get_relative_intensity(source_id, cross_source_analysis)
            }
            
            # Calculate alignment with cross-source context
            alignment = self.calculate_enhanced_bias_alignment(claim, enhanced_bias, evidence_url)
            
            # Calculate weight with cross-source penalties
            weight = self.calculate_evidence_weight(
                verification_confidence=verification["confidence"],
                bias_alignment=alignment,
                evidence_url=evidence_url,
                evidence=evidence
            )
            
            # Apply cross-source penalties
            if enhanced_bias['is_emotional_outlier']:
                weight *= 0.8  # 20% penalty for emotional outliers
                print(f"  Cross-source penalty applied to source {i+1}: emotional outlier")
            
            explanation = self._generate_enhanced_explanation(alignment, weight, enhanced_bias, evidence_url)

            weighted_evidence.append({
                "evidence": evidence,
                "verification": verification,
                "bias_analysis": enhanced_bias,
                "bias_alignment": alignment,
                "weight": weight,
                "explanation": explanation
            })

        return weighted_evidence

    def calculate_enhanced_bias_alignment(self, claim: str, enhanced_bias: Dict, evidence_url: str = None) -> float:
        """Enhanced alignment calculation including cross-source context"""
        # Get base alignment
        base_alignment = self.calculate_bias_claim_alignment(claim, enhanced_bias, evidence_url)
        
        # Adjust based on cross-source analysis
        cross_source = enhanced_bias.get('cross_source_analysis', {})
        
        # Penalty for being an emotional outlier
        if enhanced_bias.get('is_emotional_outlier', False):
            outlier_penalty = 0.2
            base_alignment = min(base_alignment + outlier_penalty, 1.0)
        
        # Penalty if source uses systematically different framing
        relative_intensity = enhanced_bias.get('relative_emotional_intensity', 0.0)
        if relative_intensity > 1.5:  # Much higher than average
            intensity_penalty = 0.15
            base_alignment = min(base_alignment + intensity_penalty, 1.0)
        
        return base_alignment

    def _check_emotional_outlier(self, source_id: str, cross_source_analysis: Dict) -> bool:
        """Check if source is flagged as emotional outlier"""
        outliers = cross_source_analysis.get('emotional_outliers', [])
        return any(outlier['source_id'] == source_id for outlier in outliers)

    def _get_relative_intensity(self, source_id: str, cross_source_analysis: Dict) -> float:
        """Get source's emotional intensity relative to baseline"""
        baseline = cross_source_analysis.get('emotional_baseline', 0.0)
        outliers = cross_source_analysis.get('emotional_outliers', [])
        
        for outlier in outliers:
            if outlier['source_id'] == source_id:
                return outlier['score'] / baseline if baseline > 0 else 1.0
        
        return 1.0  # Average intensity if not an outlier

    def _generate_enhanced_explanation(self, alignment: float, weight: float, enhanced_bias: Dict, evidence_url: str = None) -> str:
        """Enhanced explanation with detailed bias reasoning"""
        base_explanation = self._generate_weight_explanation(alignment, weight, {}, evidence_url)
        
        # Add detailed cross-source bias explanation
        cross_source_notes = []
        
        # Check if this source has detailed outlier analysis
        cross_source = enhanced_bias.get('cross_source_analysis', {})
        detailed_outliers = cross_source.get('detailed_outlier_analysis', {})
        
        # Find this source in the outlier analysis
        source_analysis = None
        for source_id, analysis in detailed_outliers.items():
            if enhanced_bias.get('is_emotional_outlier', False):
                source_analysis = analysis
                break
        
        if source_analysis:
            bias_explanation = source_analysis.get('bias_explanation', '')
            if bias_explanation:
                cross_source_notes.append(f"Emotional bias detected: {bias_explanation}")
            
            manipulation_score = source_analysis.get('manipulation_analysis', {}).get('manipulation_score', 0)
            if manipulation_score > 0.5:
                cross_source_notes.append(f"High manipulation score ({manipulation_score:.2f}/1.0)")
        
        relative_intensity = enhanced_bias.get('relative_emotional_intensity', 1.0)
        if relative_intensity > 1.3:
            cross_source_notes.append(f"Uses {relative_intensity:.1f}x more emotional language than other sources")
        elif relative_intensity < 0.7:
            cross_source_notes.append(f"Uses {relative_intensity:.1f}x less emotional language than other sources")
        
        if cross_source_notes:
            return f"{base_explanation} Bias analysis: {' '.join(cross_source_notes)}"
        
        return base_explanation