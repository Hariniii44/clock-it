import numpy as np
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
        """Return the authority score for a bare domain string."""
        official = ['parliament.lk', 'presidentsoffice.gov.lk', 'president.gov.lk',
                    'pmoffice.gov.lk', 'cbsl.lk', 'statistics.gov.lk', 'mfa.gov.lk']
        if any(d in domain for d in official):
            return 2.5

        fact_checkers = ['factcheck.lk', 'factcrescendo.com', 'boomlive.in', 'factly.in']
        if any(d in domain for d in fact_checkers):
            return 1.8

        quality_news = ['dailymirror.lk', 'economynext.com', 'island.lk',
                        'newsfirst.lk', 'adaderana.lk', 'ft.lk']
        if any(d in domain for d in quality_news):
            return 1.3

        international = ['reuters.com', 'bbc.com', 'cnn.com', 'aljazeera.com']
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
        # Dataset sources (highest authority)
        if dataset_source:
            dataset_authority = {
                'hansard': 3.0,
                'pmd': 2.8,
                'cabinet': 2.8,
                'supreme_court': 2.9,
                'central_bank': 2.7,
                'news': 1.5,
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
    
    def _get_recency_weight(self, evidence: Dict) -> float:
        """
        Calculate recency weight - newer sources get higher weight for factual claims
        """
        try:
            # Try to extract date from evidence metadata
            date_str = evidence.get('date', '')
            if date_str:
                # Parse date and calculate recency score
                from datetime import datetime, timedelta
                try:
                    evidence_date = None
                    for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
                        try:
                            evidence_date = datetime.strptime(date_str[:10], fmt)
                            break
                        except Exception:
                            continue
                    if not evidence_date:
                        for fmt in ['%b %d, %Y', '%B %d, %Y', '%d %b %Y', '%d %B %Y']:
                            try:
                                evidence_date = datetime.strptime(date_str, fmt)
                                break
                            except Exception:
                                continue
                    if not evidence_date:
                        raise ValueError(f"Unparseable date: {date_str}")
                    days_old = (datetime.now() - evidence_date).days
                    
                    # Heavy penalty for very old sources on factual claims
                    if days_old > 365:  # Older than 1 year
                        return 0.3
                    elif days_old > 180:  # Older than 6 months
                        return 0.6
                    elif days_old > 90:   # Older than 3 months
                        return 0.8
                    else:  # Recent
                        return 1.0
                except:
                    pass
            
            # Fallback: try to extract year from the URL itself
            # e.g. newsfirst.lk/2022/09/04/... → 2022 → >1yr → 0.3
            import re
            url = evidence.get('link', '') or evidence.get('url', '')
            year_match = re.search(r'/(20\d{2})/', url)
            if year_match:
                from datetime import datetime
                year = int(year_match.group(1))
                days_old = (datetime.now() - datetime(year, 1, 1)).days
                if days_old > 365:
                    return 0.3
                elif days_old > 180:
                    return 0.6

            # If still no date available, assume moderate recency
            return 0.7
            
        except:
            return 0.7

    def calculate_evidence_weight(self, verification_confidence: float, bias_alignment: float,
                                 base_credibility: float = 0.7, evidence_url: str = None,
                                 evidence: Dict = None, framing_consistency: str = 'unknown') -> float:
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
        authority_weight = self._get_authority_weight(evidence_url, evidence=evidence)
        recency_weight = self._get_recency_weight(evidence) if evidence else 1.0

        # Bias penalty — adjusted by framing consistency
        # consistent   → 1.3× (expected bias behaviour, down-weight more)
        # inconsistent → 0.6× (against own bias, up-weight)
        # unknown      → 1.0× (no change)
        _consistency_multiplier = {'consistent': 1.3, 'inconsistent': 0.6, 'unknown': 1.0}
        consistency_mult = _consistency_multiplier.get(framing_consistency, 1.0)
        bias_penalty = bias_alignment * 0.5 * consistency_mult

        weight = base_credibility * verification_confidence * (1 - bias_penalty) * authority_weight * recency_weight

        print(f"  Debug: conf={verification_confidence:.3f}, bias_align={bias_alignment:.3f}, "
              f"framing={framing_consistency}, authority={authority_weight:.1f}, "
              f"recency={recency_weight:.1f}, weight={weight:.3f}")

        return max(weight, 0.1)
    
    def weight_all_evidence(self, claim: str, evidence_list: List[Dict],
                            verification_results: List[Dict], bias_analyses: List[Dict],
                            framing_analyses: Dict = None,
                            precomputed_alignments: List[float] = None) -> List[Dict]:
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
            )

            explanation = self._generate_weight_explanation(alignment, weight, bias, evidence_url)

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
        
    def _generate_weight_explanation(self, alignment: float, weight: float, bias: Dict, evidence_url: str = None) -> str:
        """Enhanced explanation including source profile information"""
        domain = self._extract_domain(evidence_url) if evidence_url else None
        
        # Special handling for parliamentary sources
        if domain == 'parliament.lk':
            return f"Official parliamentary source (parliament.lk) - highest authority. Weight: {weight:.2f}"
        
        # Check if we have source profile
        if domain and domain in self.bias_profiles:
            source_profile = self.bias_profiles[domain]
            bias_interpretation = source_profile["interpretation"]
            bias_score = source_profile["bias_score"]
            confidence = source_profile["confidence"]
            
            if alignment > 0.4:  # Lowered threshold for better sensitivity
                return (
                    f"Source {domain} is {bias_interpretation.lower()} ({bias_score:+.1f}) with "
                    f"bias-claim alignment of {alignment:.2f}. Weight adjusted to {weight:.2f}."
                )
            elif alignment > 0.2:
                return (
                    f"Source {domain} shows {bias_interpretation.lower()} bias with moderate "
                    f"relevance. Weight: {weight:.2f}."
                )
            else:
                return (
                    f"Source {domain} bias ({bias_interpretation.lower()}) has minimal impact "
                    f"on this claim. Weight: {weight:.2f}."
                )
        else:
            # Enhanced fallback for unknown sources
            if domain:
                return f"Source {domain} - no bias profile available. Weight: {weight:.2f}"
            else:
                return f"Unknown source - using text-based analysis only. Weight: {weight:.2f}"

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
        if active_total > 0:
            support_pct = support_score / active_total
            refute_pct  = refute_score  / active_total
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
        
        # Determine verdict
        max_score = max(support_pct, refute_pct, neutral_pct)
        
        # Check if this is a factual claim
        is_factual = self._is_factual_claim(weighted_evidence)
        
        # Enhanced conflict detection considering bias
        bias_alignments = [item["bias_alignment"] for item in weighted_evidence]
        high_bias_sources = sum(1 for alignment in bias_alignments if alignment > 0.5)
        bias_conflict_factor = high_bias_sources / len(weighted_evidence) if weighted_evidence else 0
        
        # Adjust thresholds based on bias conflict and factual nature - less aggressive conflict detection
        if is_factual:
            # For factual claims, require stronger evidence for "conflicting" verdict
            conflict_threshold = 0.25 + (bias_conflict_factor * 0.05)  # Reduced sensitivity
            min_opposing_threshold = 0.35  # Slightly lower threshold
        else:
            # For opinion/subjective claims, use more permissive thresholds
            conflict_threshold = 0.15 + (bias_conflict_factor * 0.05)  # Reduced sensitivity
            min_opposing_threshold = 0.25  # Lower threshold
        
        # Enhanced verdict logic with better factual claim handling
        if is_factual:
            # For factual claims, prioritize official sources and require stronger evidence for conflicts
            # Check if we have strong official support
            official_support_weight = 0
            for item in weighted_evidence:
                if item["verification"]["label"] == "SUPPORTED":
                    domain = weighter._extract_domain(item["evidence"].get("link", ""))
                auth_weight = weighter._get_authority_weight(item["evidence"].get("link", ""), evidence=item["evidence"])
            
            # If official sources support the claim strongly, treat as supported
            if official_support_weight > 0.4 and support_pct > refute_pct:
                verdict = "SUPPORTED" 
                confidence = min(0.8 + (official_support_weight - 0.4) * 0.5, 1.0)
            elif support_pct > 0.45:  # Lowered from 0.6 to 0.45
                verdict = "SUPPORTED"
                confidence = min(support_pct + diversity_boost + 0.1, 1.0)  # Small boost for decisiveness
            elif refute_pct > 0.45:  # Lowered from 0.6 to 0.45
                verdict = "REFUTED"
                confidence = min(refute_pct + diversity_boost + 0.1, 1.0)  # Small boost for decisiveness
            elif support_pct > refute_pct and support_pct > 0.35:  # More decisive for clear majority
                verdict = "SUPPORTED"
                confidence = support_pct + diversity_boost
            elif refute_pct > support_pct and refute_pct > 0.35:  # More decisive for clear majority
                verdict = "REFUTED"
                confidence = refute_pct + diversity_boost
            else:
                verdict = "UNCERTAIN"
                confidence = max_score + diversity_boost
        else:
            # Improved logic for opinion/subjective claims - less conflict-sensitive
            if (abs(support_pct - refute_pct) < 0.15 and 
                min(support_pct, refute_pct) > 0.4 and max_score < 0.6):  # Stricter conflict detection
                verdict = "CONFLICTING"
                confidence = (1 - abs(support_pct - refute_pct)) + diversity_boost
            elif max_score == support_pct and support_pct > 0.35:  # Lowered from 0.4 to 0.35
                verdict = "SUPPORTED"
                confidence = min(support_pct + diversity_boost + 0.05, 1.0)
            elif max_score == refute_pct and refute_pct > 0.35:  # Lowered from 0.4 to 0.35
                verdict = "REFUTED"
                confidence = min(refute_pct + diversity_boost + 0.05, 1.0)
            elif support_pct > refute_pct and support_pct > 0.25:  # Additional fallback for weak support
                verdict = "SUPPORTED"
                confidence = support_pct + diversity_boost
            elif refute_pct > support_pct and refute_pct > 0.25:  # Additional fallback for weak refutation
                verdict = "REFUTED"
                confidence = refute_pct + diversity_boost
            else:
                verdict = "UNCERTAIN"
                confidence = max_score + diversity_boost

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
        confidences = [item["verification"]["confidence"] for item in weighted_evidence]
        alignments = [item["bias_alignment"] for item in weighted_evidence]
        labels = [item["verification"]["label"] for item in weighted_evidence]
        
        # Epistemic uncertainty: variance in model confidence
        epistemic = float(np.std(confidences)) if confidences else 1.0
        
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
        base_explanation = self._generate_weight_explanation(alignment, weight, enhanced_bias, evidence_url)
        
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