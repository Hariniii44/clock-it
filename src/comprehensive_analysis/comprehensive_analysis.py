from typing import Dict, List
from ..verification import ClaimVerifier
from ..bias_detection import BiasDetector

class ComprehensiveAnalyzer:
    def __init__(self):
        self.verifier = ClaimVerifier()
        self.bias_detector = BiasDetector()
    
    def analyze_evidence(self, claim: str, evidence_list: List[Dict]) -> List[Dict]:
        """
        Perform comprehensive analysis on evidence
        Returns enhanced evidence with verification and bias analysis
        """
        analyzed_evidence = []
        
        for evidence in evidence_list:
            # Get verification result
            verification = self.verifier.verify_claim(claim, evidence["snippet"])
            
            # Get bias analysis
            bias_analysis = self.bias_detector.analyze_source(
                evidence["snippet"], 
                evidence["link"]
            )
            
            # Combine all information
            enhanced_evidence = {
                **evidence,  # Original evidence
                "verification": verification,
                "bias_analysis": bias_analysis,
                "overall_credibility": self._calculate_credibility(verification, bias_analysis)
            }
            
            analyzed_evidence.append(enhanced_evidence)
        
        return analyzed_evidence
    
    def _calculate_credibility(self, verification: Dict, bias_analysis: Dict) -> Dict:
        """Calculate overall credibility score based on verification and bias"""
        
        # Base credibility from verification confidence
        verification_score = verification["confidence"]
        
        # Adjust based on political stance (more extreme = less credible)
        stance_penalty = abs(bias_analysis["political_stance"]["stance_score"]) * 0.1
        
        # Adjust based on emotional tone (more extreme = less credible)
        emotion_penalty = abs(bias_analysis["emotional_tone"]["emotion_score"]) * 0.05
        
        # Adjust based on framing (more extreme = less credible)
        framing_penalty = abs(bias_analysis["framing_bias"]["framing_score"]) * 0.05
        
        overall_score = verification_score - stance_penalty - emotion_penalty - framing_penalty
        overall_score = max(0.0, min(1.0, overall_score))  # Clamp between 0 and 1
        
        return {
            "score": overall_score,
            "factors": {
                "verification": verification_score,
                "stance_penalty": stance_penalty,
                "emotion_penalty": emotion_penalty,
                "framing_penalty": framing_penalty
            }
        }