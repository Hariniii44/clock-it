import numpy as np
from typing import List, Dict
import json

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
    
    def _get_authority_weight(self, evidence_url: str, dataset_source: str = None) -> float:
        """
        Calculate authority weight based on source type
        """

        # Handle dataset sources (highest authority)
        if dataset_source:
            dataset_authority = {
                'hansard': 3.0,      # Parliamentary debates - highest authority
                'pmd': 2.8,          # Presidential statements
                'cabinet': 2.8,      # Cabinet decisions
                'supreme_court': 2.9, # Legal authority
                'central_bank': 2.7, # Economic authority
                'news': 1.5          # News varies by source
            }
            return dataset_authority.get(dataset_source, 2.0)

        if not evidence_url:
            return 1.0
            
        domain = self._extract_domain(evidence_url)
        
        # Official/Government sources get highest authority
        official_domains = ['parliament.lk', 'presidentsoffice.gov.lk', 'president.gov.lk', 'pmoffice.gov.lk', 
                           'cbsl.lk', 'statistics.gov.lk', 'mfa.gov.lk']
        if any(d in domain for d in official_domains):
            return 2.5  # Even higher weight for official sources on factual claims
            
        # Fact-checkers get high authority
        fact_checker_domains = ['factcheck.lk', 'factcrescendo.com', 'boomlive.in', 'factly.in']
        if any(d in domain for d in fact_checker_domains):
            return 1.8
            
        # Quality news sources
        quality_news = ['dailymirror.lk', 'economynext.com', 'island.lk', 'newsfirst.lk', 'adaderana.lk', 'ft.lk']
        if any(d in domain for d in quality_news):
            return 1.3
            
        # International sources for Sri Lankan news
        international = ['reuters.com', 'bbc.com', 'cnn.com', 'aljazeera.com']
        if any(d in domain for d in international):
            return 1.2
            
        return 1.0  # Standard weight
    
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
                    evidence_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
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
            
            # If no date available, assume moderate recency
            return 0.7
            
        except:
            return 0.7

    def calculate_evidence_weight(self, verification_confidence: float, bias_alignment: float, 
                                 base_credibility: float = 0.7, evidence_url: str = None, evidence: Dict = None) -> float:
        """
        Enhanced evidence weighting with source credibility, authority, and recency
        """
        # Adjust base credibility based on source profile if available
        domain = self._extract_domain(evidence_url) if evidence_url else None
        if domain and domain in self.bias_profiles:
            profile = self.bias_profiles[domain]
            # Higher confidence in profile = higher base credibility
            profile_credibility = 0.5 + (profile["confidence"] * 0.3)  # Range: 0.5-0.8
            base_credibility = max(base_credibility, profile_credibility)

        # Add authority and recency weights for factual claims
        authority_weight = self._get_authority_weight(evidence_url)
        recency_weight = self._get_recency_weight(evidence) if evidence else 1.0
        
        # Original bias penalty calculation
        bias_penalty = bias_alignment * 0.5
        
        # Enhanced weight calculation with authority and recency
        weight = base_credibility * verification_confidence * (1 - bias_penalty) * authority_weight * recency_weight

        print(f"  Debug: conf={verification_confidence:.3f}, bias_align={bias_alignment:.3f}, authority={authority_weight:.1f}, recency={recency_weight:.1f}, weight={weight:.3f}")

        return max(weight, 0.1)  # Ensure minimum weight
    
    def weight_all_evidence(self, claim: str, evidence_list: List[Dict], verification_results: List[Dict], bias_analyses: List[Dict]) -> List[Dict]:
        """
        Weight all evidence pieces with enhanced bias profiling
        """
        weighted_evidence = []
        
        for evidence, verification, bias in zip(evidence_list, verification_results, bias_analyses):
            evidence_url = evidence.get("link", "")
            
            # Calculate alignment with URL context
            alignment = self.calculate_bias_claim_alignment(claim, bias, evidence_url)

            # Calculate weight with URL context, authority, and recency
            weight = self.calculate_evidence_weight(
                verification_confidence=verification["confidence"],
                bias_alignment=alignment,
                evidence_url=evidence_url,
                evidence=evidence
            )

            # Enhanced explanation generation
            explanation = self._generate_weight_explanation(
                alignment, weight, bias, evidence_url
            )

            weighted_evidence.append({
                "evidence": evidence,
                "verification": verification,
                "bias_analysis": bias,
                "bias_alignment": alignment,
                "weight": weight,
                "explanation": explanation
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
            auth_weight = weighter._get_authority_weight(evidence_url)
            authority_weights.append(auth_weight)
        
        # If we have official sources (weight > 1.5), treat as factual
        has_official_sources = any(w > 1.5 for w in authority_weights)
        
        # If majority of sources are high-authority, treat as factual
        high_authority_ratio = sum(1 for w in authority_weights if w > 1.2) / len(authority_weights)
        
        return has_official_sources or high_authority_ratio > 0.6

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
        
        # Normalize scores
        support_pct = support_score / total
        refute_pct = refute_score / total
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
                    auth_weight = weighter._get_authority_weight(item["evidence"].get("link", ""))
                    if auth_weight > 2.0:  # Official sources
                        official_support_weight += item["weight"]
            
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

        confidence = min(confidence, 1.0)  # Cap at 100%
        
        uncertainty = self._decompose_uncertainty(weighted_evidence)
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "support_score": support_pct,
            "refute_score": refute_pct,
            "neutral_score": neutral_pct,
            "diversity_score": diversity_score,
            "is_factual_claim": is_factual,
            "uncertainty_decomposition": uncertainty
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
        
        # Aleatoric uncertainty: lack of evidence
        num_sources = len(weighted_evidence)
        ideal_sources = 5
        aleatoric = max(0.0, (ideal_sources - num_sources) / ideal_sources)
        
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