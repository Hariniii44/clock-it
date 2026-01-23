#!/usr/bin/env python3
"""
Claim Verification System - FR06 Implementation
Uses NLI model to verify claims against evidence
"""

from transformers import pipeline
from typing import Dict, List
import logging
import torch

class ClaimVerifier:
    """
    Claim verification using Natural Language Inference (NLI).
    Implements FR06: Generate graded verdicts
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.logger = logging.getLogger(__name__)
        
        # Check if CUDA is available
        device = 0 if torch.cuda.is_available() else -1
        
        print(f"Loading NLI model: {model_name}")
        print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
        
        try:
            self.nli_model = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device
            )
            print("✅ NLI model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading NLI model: {e}")
            raise
    
    def verify_claim(self, claim: str, evidence: str) -> Dict:
        """
        Verify claim against evidence using NLI
        
        Args:
            claim: The claim to verify
            evidence: Evidence text to check against
            
        Returns:
            Dict with label, confidence, and explanation
        """
        try:
            # Prepare input for zero-shot classification
            evidence_truncated = evidence[:400]  # Keep evidence shorter
            claim_truncated = claim[:100]       # Keep claim concise
            
            # Use zero-shot classification with entailment labels
            labels = ["supported", "refuted", "insufficient"]
            hypothesis = f"The evidence supports that {claim_truncated}"
            
            # Get prediction
            result = self.nli_model(evidence_truncated, labels)
            
            # Map to our labels
            label_map = {
                "supported": "SUPPORTED",
                "refuted": "REFUTED", 
                "insufficient": "INSUFFICIENT"
            }
            
            predicted_label = result['labels'][0]
            confidence = result['scores'][0]
            
            label = label_map.get(predicted_label, "INSUFFICIENT")
            
            # Generate explanation based on result
            explanations = {
                "SUPPORTED": f"The evidence supports this claim with {confidence:.1%} confidence.",
                "REFUTED": f"The evidence contradicts this claim with {confidence:.1%} confidence.",
                "INSUFFICIENT": f"The evidence is insufficient to verify this claim ({confidence:.1%} confidence)."
            }
            
            return {
                "label": label,
                "confidence": confidence,
                "explanation": explanations[label],
                "raw_nli_label": predicted_label
            }
            
        except Exception as e:
            self.logger.error(f"Error verifying claim: {e}")
            return {
                "label": "ERROR",
                "confidence": 0.0,
                "explanation": f"Error during verification: {str(e)}",
                "raw_nli_label": "ERROR"
            }
    
    def verify_against_multiple_evidence(self, claim: str, evidence_list: List[Dict]) -> Dict:
        """
        Verify claim against multiple pieces of evidence
        
        Args:
            claim: The claim to verify
            evidence_list: List of evidence dictionaries
            
        Returns:
            Aggregated verification result
        """
        if not evidence_list:
            return {
                "label": "INSUFFICIENT",
                "confidence": 0.0,
                "explanation": "No evidence available for verification",
                "evidence_count": 0,
                "individual_results": []
            }
        
        individual_results = []
        
        # Verify against each piece of evidence
        for i, evidence in enumerate(evidence_list):
            evidence_text = evidence.get('content', evidence.get('snippet', ''))
            
            if not evidence_text.strip():
                continue
                
            result = self.verify_claim(claim, evidence_text)
            result['source'] = evidence.get('source', f'Evidence {i+1}')
            result['similarity'] = evidence.get('similarity', 0.0)
            individual_results.append(result)
        
        if not individual_results:
            return {
                "label": "INSUFFICIENT",
                "confidence": 0.0,
                "explanation": "No valid evidence content available",
                "evidence_count": 0,
                "individual_results": []
            }
        
        # Aggregate results
        supported_count = sum(1 for r in individual_results if r['label'] == 'SUPPORTED')
        refuted_count = sum(1 for r in individual_results if r['label'] == 'REFUTED')
        
        # Weighted confidence based on similarity scores
        total_confidence = 0
        total_weight = 0
        
        for result in individual_results:
            weight = result.get('similarity', 0.5)  # Use similarity as weight
            total_confidence += result['confidence'] * weight
            total_weight += weight
        
        avg_confidence = total_confidence / total_weight if total_weight > 0 else 0
        
        # Determine overall label
        if supported_count > refuted_count:
            overall_label = "SUPPORTED"
            explanation = f"Claim is supported by {supported_count}/{len(individual_results)} sources"
        elif refuted_count > supported_count:
            overall_label = "REFUTED"
            explanation = f"Claim is refuted by {refuted_count}/{len(individual_results)} sources"
        else:
            overall_label = "INSUFFICIENT"
            explanation = f"Evidence is inconclusive ({supported_count} supporting, {refuted_count} refuting)"
        
        return {
            "label": overall_label,
            "confidence": avg_confidence,
            "explanation": explanation,
            "evidence_count": len(individual_results),
            "supported_count": supported_count,
            "refuted_count": refuted_count,
            "individual_results": individual_results
        }
    
    def get_confidence_grade(self, confidence: float) -> str:
        """Convert confidence score to letter grade"""
        if confidence >= 0.9:
            return "A"
        elif confidence >= 0.8:
            return "B"
        elif confidence >= 0.7:
            return "C"
        elif confidence >= 0.6:
            return "D"
        else:
            return "F"