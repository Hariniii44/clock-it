# import numpy as np
# from typing import List, Dict

# class EvidenceWeighter:
#     def __init__(self):
#         """
#         calculate bias-claim alignmenet 
#         dynamic evidence weighting
#         """
#         pass

#     def calculate_bias_claim_alignment(self, claim: str, bias_analysis: Dict) -> float:
#         """
#         Determine if source bias is contextually relevant
#         Logic:
#         - If claim is about government AND source is pro/anti-government
#           High alignment (bias is relevant)
#         - If claim is neutral fact AND source is biased
#           Low alignment (bias less relevant)
#         """

#         # Simple keyword-based approach for demonstration
#         government_keywords = ["government", "administration", "policy", "minister", "president", "cabinet", "parliament"]

#         claim_lower = claim.lower() 
#         is_political_claim = any(kw in claim_lower for kw in government_keywords)

#         political_bias = bias_analysis["political_stance"]["stance_score"]
#         emotional_bias = abs(bias_analysis["emotional_tone"]["emotion_score"])
#         framing_bias = abs(bias_analysis["framing_bias"]["framing_score"])
        
#         if is_political_claim:
#             alignment = (political_bias + emotional_bias + framing_bias) / 3
#         else:
#             alignment = (political_bias + emotional_bias + framing_bias) / 6  # downweight for non-political claims
        
#         return min(alignment, 1.0)
    
#     def calculate_evidence_weight(self, verification_confidence: float, bias_alignment: float, base_credibility: float = 0.7) -> float:
#         """
#         Dynamic evidence weighting

#         formula:
#         weight = base_credibility * verification_confidence * (1 - bias_penalty)
#         where bias_penalty = bias_alignment * 0.5
#         """

#         bias_penalty = bias_alignment * 0.5
#         weight = base_credibility * verification_confidence * (1 - bias_penalty)

#         print(f"  Debug: conf={verification_confidence:.3f}, bias_align={bias_alignment:.3f}, penalty={bias_penalty:.3f}, weight={weight:.3f}")

#         return max(weight, 0.1)  # Ensure minimum weight
    
#     def weight_all_evidence(self, claim: str, evidence_list: List[Dict], verification_results: List[Dict], bias_analyses: List[Dict]) -> List[Dict]:
#         """
#         Weight all evidence pieces
#         Returns list of evidence with added 'weight' field
#         """

#         weighted_evidence = []
        
#         for evidence, verification, bias in zip(
#             evidence_list, verification_results, bias_analyses
#         ):
#             alignment = self.calculate_bias_claim_alignment(claim, bias) #calculating alignment

#             weight = self.calculate_evidence_weight(
#                 verification_confidence=verification["confidence"],
#                 bias_alignment=alignment
#             )

#             weighted_evidence.append({
#                 "evidence": evidence,
#                 "verification": verification,
#                 "bias_analysis": bias,
#                 "bias_alignment": alignment,
#                 "weight": weight,
#                 "explanation": self._generate_weight_explanation(
#                     alignment, weight, bias
#                 )
#             })

#         return weighted_evidence
        
#     def _generate_weight_explanation(self, alignment: float, weight: float, bias: Dict) -> str:
#         """
#         explain the assigned weight
#         """

#         political = bias["political_stance"]["political_stance"]
#         emotion = bias["emotional_tone"]["emotion"]

#         if alignment > 0.6:
#             return (
#                 f"Source shows {political} bias with {emotion} tone. "
#                 f"Weight reduced to {weight:.2f} due to high bias-claim alignment."
#             )
#         elif alignment > 0.3:
#             return (
#                 f"Source shows moderate {political} bias. "
#                 f"Weight adjusted to {weight:.2f}."
#             )
#         else:
#             return f"Source bias is low or not contextually relevant. Weight: {weight:.2f}"

# class VerdictGenerator:
#     def __init__(self):
#         """FR06: Generate graded verdicts"""
#         pass
    
#     def generate_verdict(
#         self,
#         weighted_evidence: List[Dict]
#     ) -> Dict:
#         """
#         FR06: Generate graded verdict (SUPPORTED/REFUTED/UNCERTAIN/CONFLICTING)
#         FR07: Provide confidence scores with uncertainty decomposition
#         """
#         # Aggregate weighted votes
#         support_score = 0.0
#         refute_score = 0.0
#         neutral_score = 0.0
        
#         for item in weighted_evidence:
#             label = item["verification"]["label"]
#             weight = item["weight"]
            
#             if label == "SUPPORTED":
#                 support_score += weight
#             elif label == "REFUTED":
#                 refute_score += weight
#             else:
#                 neutral_score += weight
        
#         total = support_score + refute_score + neutral_score
        
#         if total == 0:
#             return {
#                 "verdict": "UNCERTAIN",
#                 "confidence": 0.0,
#                 "explanation": "No valid evidence found"
#             }
        
#         # Normalize scores
#         support_pct = support_score / total
#         refute_pct = refute_score / total
#         neutral_pct = neutral_score / total
        
#         # Determine verdict
#         max_score = max(support_pct, refute_pct, neutral_pct)
        
#         # Threshold for conflicting
#         if abs(support_pct - refute_pct) < 0.2 and min(support_pct, refute_pct) > 0.3:
#             verdict = "CONFLICTING"
#             confidence = 1 - abs(support_pct - refute_pct)  # Lower when more conflict
#         elif max_score == support_pct and support_pct > 0.5:
#             verdict = "SUPPORTED"
#             confidence = support_pct
#         elif max_score == refute_pct and refute_pct > 0.5:
#             verdict = "REFUTED"
#             confidence = refute_pct
#         else:
#             verdict = "UNCERTAIN"
#             confidence = max_score

#         uncertainty = self._decompose_uncertainty(weighted_evidence)
        
#         return {
#             "verdict": verdict,
#             "confidence": confidence,
#             "support_score": support_pct,
#             "refute_score": refute_pct,
#             "neutral_score": neutral_pct,
#             "uncertainty_decomposition": uncertainty
#         }
    
#     def _decompose_uncertainty(self, weighted_evidence: List[Dict]) -> Dict:
#         """
#         Decompose uncertainty into components
#         - Epistemic: Model uncertainty (variance in confidence scores)
#         - Aleatoric: Data uncertainty (insufficient evidence)
#         - Bias-induced: Disagreement due to source bias
#         """
#         confidences = [item["verification"]["confidence"] for item in weighted_evidence]
#         alignments = [item["bias_alignment"] for item in weighted_evidence]
#         labels = [item["verification"]["label"] for item in weighted_evidence]
        
#         # Epistemic uncertainty: variance in model confidence
#         epistemic = float(np.std(confidences)) if confidences else 1.0
        
#         # Aleatoric uncertainty: lack of evidence
#         num_sources = len(weighted_evidence)
#         ideal_sources = 5
#         aleatoric = max(0.0, (ideal_sources - num_sources) / ideal_sources)
        
#         # Bias-induced uncertainty: high bias + disagreement
#         high_bias_count = sum(1 for a in alignments if a > 0.5)
#         label_diversity = len(set(labels)) / 3 if labels else 1.0  # 3 possible labels
#         bias_induced = (high_bias_count / len(alignments)) * label_diversity if alignments else 0.0
        

#         return {
#             "epistemic": round(epistemic, 3),
#             "aleatoric": round(aleatoric, 3),
#             "bias_induced": round(bias_induced, 3),
#             "explanation": self._explain_uncertainty(epistemic, aleatoric, bias_induced)
#         }
    
#     def _explain_uncertainty(self, epistemic, aleatoric, bias_induced) -> str:
#         """Generate human-readable uncertainty explanation"""
#         explanations = []
        
#         if epistemic > 0.3:
#             explanations.append("Model confidence varies across sources")
#         if aleatoric > 0.4:
#             explanations.append("Limited evidence available")
#         if bias_induced > 0.4:
#             explanations.append("Sources with different biases disagree")
        
#         if not explanations:
#             return "Uncertainty is low across all factors"
        
#         return "; ".join(explanations)

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
        print(f"✅ Evidence weighter loaded with {len(self.bias_profiles)} source profiles")

    def _load_bias_profiles(self) -> dict:
        """Load pre-computed bias profiles from bias detection pipeline"""
        try:
            with open("data/bias_profiles.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ Bias profiles not found. Using text-based analysis only.")
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
    
    def calculate_evidence_weight(self, verification_confidence: float, bias_alignment: float, 
                                 base_credibility: float = 0.7, evidence_url: str = None) -> float:
        """
        Enhanced evidence weighting with source credibility from bias profiles
        """
        # Adjust base credibility based on source profile if available
        domain = self._extract_domain(evidence_url) if evidence_url else None
        if domain and domain in self.bias_profiles:
            profile = self.bias_profiles[domain]
            # Higher confidence in profile = higher base credibility
            profile_credibility = 0.5 + (profile["confidence"] * 0.3)  # Range: 0.5-0.8
            base_credibility = max(base_credibility, profile_credibility)

        # Original bias penalty calculation
        bias_penalty = bias_alignment * 0.5
        weight = base_credibility * verification_confidence * (1 - bias_penalty)

        print(f"  Debug: conf={verification_confidence:.3f}, bias_align={bias_alignment:.3f}, penalty={bias_penalty:.3f}, weight={weight:.3f}")

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

            # Calculate weight with URL context
            weight = self.calculate_evidence_weight(
                verification_confidence=verification["confidence"],
                bias_alignment=alignment,
                evidence_url=evidence_url
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
            # Fallback for unknown sources
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
    
    def generate_verdict(self, weighted_evidence: List[Dict]) -> Dict:
        """
        Enhanced verdict generation considering source diversity and bias distribution
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
        
        # Enhanced conflict detection considering bias
        bias_alignments = [item["bias_alignment"] for item in weighted_evidence]
        high_bias_sources = sum(1 for alignment in bias_alignments if alignment > 0.5)
        bias_conflict_factor = high_bias_sources / len(weighted_evidence) if weighted_evidence else 0
        
        # Adjust thresholds based on bias conflict
        conflict_threshold = 0.2 + (bias_conflict_factor * 0.1)  # Higher threshold if biased sources
        
        if abs(support_pct - refute_pct) < conflict_threshold and min(support_pct, refute_pct) > 0.3:
            verdict = "CONFLICTING"
            confidence = (1 - abs(support_pct - refute_pct)) + diversity_boost
        elif max_score == support_pct and support_pct > 0.5:
            verdict = "SUPPORTED"
            confidence = support_pct + diversity_boost
        elif max_score == refute_pct and refute_pct > 0.5:
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