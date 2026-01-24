import numpy as np
from typing import List, Dict

class EvidenceWeighter:
    def __init__(self):
        """
        calculate bias-claim alignmenet 
        dynamic evidence weighting
        """
        pass

    def calculate_bias_claim_alignment(self, claim: str, bias_analysis: Dict) -> float:
        """
        Determine if source bias is contextually relevant
        Logic:
        - If claim is about government AND source is pro/anti-government
          High alignment (bias is relevant)
        - If claim is neutral fact AND source is biased
          Low alignment (bias less relevant)
        """

        # Simple keyword-based approach for demonstration
        government_keywords = ["government", "administration", "policy", "minister", "president", "cabinet", "parliament"]

        claim_lower = claim.lower() 
        is_political_claim = any(kw in claim_lower for kw in government_keywords)

        political_bias = bias_analysis["political_stance"]["stance_score"]
        emotional_bias = abs(bias_analysis["emotional_tone"]["emotion_score"])
        framing_bias = abs(bias_analysis["framing_bias"]["framing_score"])
        
        if is_political_claim:
            alignment = (political_bias + emotional_bias + framing_bias) / 3
        else:
            alignment = (political_bias + emotional_bias + framing_bias) / 6  # downweight for non-political claims
        
        return min(alignment, 1.0)
    
    def calculate_evidence_weight(self, verification_confidence: float, bias_alignment: float, base_credibility: float = 0.7) -> float:
        """
        Dynamic evidence weighting

        formula:
        weight = base_credibility * verification_confidence * (1 - bias_penalty)
        where bias_penalty = bias_alignment * 0.5
        """

        bias_penalty = bias_alignment * 0.5
        weight = base_credibility * verification_confidence * (1 - bias_penalty)

        print(f"  Debug: conf={verification_confidence:.3f}, bias_align={bias_alignment:.3f}, penalty={bias_penalty:.3f}, weight={weight:.3f}")

        return max(weight, 0.1)  # Ensure minimum weight
    
    def weight_all_evidence(self, claim: str, evidence_list: List[Dict], verification_results: List[Dict], bias_analyses: List[Dict]) -> List[Dict]:
        """
        Weight all evidence pieces
        Returns list of evidence with added 'weight' field
        """

        weighted_evidence = []
        
        for evidence, verification, bias in zip(
            evidence_list, verification_results, bias_analyses
        ):
            alignment = self.calculate_bias_claim_alignment(claim, bias) #calculating alignment

            weight = self.calculate_evidence_weight(
                verification_confidence=verification["confidence"],
                bias_alignment=alignment
            )

            weighted_evidence.append({
                "evidence": evidence,
                "verification": verification,
                "bias_analysis": bias,
                "bias_alignment": alignment,
                "weight": weight,
                "explanation": self._generate_weight_explanation(
                    alignment, weight, bias
                )
            })

        return weighted_evidence
        
    def _generate_weight_explanation(self, alignment: float, weight: float, bias: Dict) -> str:
        """
        explain the assigned weight
        """

        political = bias["political_stance"]["political_stance"]
        emotion = bias["emotional_tone"]["emotion"]

        if alignment > 0.6:
            return (
                f"Source shows {political} bias with {emotion} tone. "
                f"Weight reduced to {weight:.2f} due to high bias-claim alignment."
            )
        elif alignment > 0.3:
            return (
                f"Source shows moderate {political} bias. "
                f"Weight adjusted to {weight:.2f}."
            )
        else:
            return f"Source bias is low or not contextually relevant. Weight: {weight:.2f}"

class VerdictGenerator:
    def __init__(self):
        """FR06: Generate graded verdicts"""
        pass
    
    def generate_verdict(
        self,
        weighted_evidence: List[Dict]
    ) -> Dict:
        """
        FR06: Generate graded verdict (SUPPORTED/REFUTED/UNCERTAIN/CONFLICTING)
        FR07: Provide confidence scores with uncertainty decomposition
        """
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
        
        # Determine verdict
        max_score = max(support_pct, refute_pct, neutral_pct)
        
        # Threshold for conflicting
        if abs(support_pct - refute_pct) < 0.2 and min(support_pct, refute_pct) > 0.3:
            verdict = "CONFLICTING"
            confidence = 1 - abs(support_pct - refute_pct)  # Lower when more conflict
        elif max_score == support_pct and support_pct > 0.5:
            verdict = "SUPPORTED"
            confidence = support_pct
        elif max_score == refute_pct and refute_pct > 0.5:
            verdict = "REFUTED"
            confidence = refute_pct
        else:
            verdict = "UNCERTAIN"
            confidence = max_score

        uncertainty = self._decompose_uncertainty(weighted_evidence)
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "support_score": support_pct,
            "refute_score": refute_pct,
            "neutral_score": neutral_pct,
            "uncertainty_decomposition": uncertainty
        }
    
    def _decompose_uncertainty(self, weighted_evidence: List[Dict]) -> Dict:
        """
        Decompose uncertainty into components
        - Epistemic: Model uncertainty (variance in confidence scores)
        - Aleatoric: Data uncertainty (insufficient evidence)
        - Bias-induced: Disagreement due to source bias
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
        
        # Bias-induced uncertainty: high bias + disagreement
        high_bias_count = sum(1 for a in alignments if a > 0.5)
        label_diversity = len(set(labels)) / 3 if labels else 1.0  # 3 possible labels
        bias_induced = (high_bias_count / len(alignments)) * label_diversity if alignments else 0.0
        

        return {
            "epistemic": round(epistemic, 3),
            "aleatoric": round(aleatoric, 3),
            "bias_induced": round(bias_induced, 3),
            "explanation": self._explain_uncertainty(epistemic, aleatoric, bias_induced)
        }
    
    def _explain_uncertainty(self, epistemic, aleatoric, bias_induced) -> str:
        """Generate human-readable uncertainty explanation"""
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