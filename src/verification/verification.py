# src/verification/verification.py
from transformers import pipeline
from typing import Dict
import os

class ClaimVerifier:
    def __init__(self):
        """Initialize NLI model for verification with multiple fallback options"""
        print("Loading NLI model...")
        
        # Get Hugging Face token from environment variable
        hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
        
        # List of models to try in order of preference (your working model first)
        models_to_try = [
            "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", 
            "cross-encoder/nli-deberta-v3-xsmall",         # Extra small backup ~40MB
            "typeform/distilbert-base-uncased-mnli",        # Small backup ~250MB  
            "microsoft/deberta-base-mnli",                  # Medium backup ~400MB
            "microsoft/deberta-large-mnli",                 # Large backup ~1.4GB
            "facebook/bart-large-mnli"                      # Largest backup ~1.6GB
        ]
        
        self.nli_model = None
        self.use_rule_based = False
        device = -1  # Use CPU to avoid CUDA issues
        print(f"Device set to use {'cpu' if device == -1 else 'gpu'}")
        
        for model_name in models_to_try:
            try:
                print(f"Attempting to load: {model_name}")
                self.nli_model = pipeline(
                    "text-classification",
                    model=model_name,
                    device=device,
                    token=hf_token
                )
                print(f"Successfully loaded: {model_name}")
                self.model_name = model_name
                break
            except Exception as e:
                error_str = str(e).lower()
                if any(mem_indicator in error_str for mem_indicator in 
                       ['paging file', 'memory', 'out of memory', 'cuda out of memory']):
                    print(f"MEMORY ERROR: {model_name} too large for available memory")
                else:
                    print(f"Failed to load {model_name}: {str(e)[:100]}...")
                continue
        
        if self.nli_model is None:
            print("WARNING: All ML models failed to load due to memory constraints")
            print("Falling back to rule-based verification")
            self.use_rule_based = True
        else:
            print("NLI model loaded successfully")
    
    def verify_claim(self, claim: str, evidence: str) -> Dict:
        """
        Verify claim against evidence using NLI or rule-based fallback
        Returns: {label: str, confidence: float}
        """
        
        if self.use_rule_based:
            return self._rule_based_verification(claim, evidence)
            
        try:
            nli_input = f"{evidence} </s> {claim}"
            result = self.nli_model(nli_input)
            print(f"      RAW MODEL OUTPUT: {result}")

            # Map NLI labels to fact-checking labels
            label_map = {
                "ENTAILMENT": "SUPPORTED",
                "CONTRADICTION": "REFUTED",
                "NEUTRAL": "NEUTRAL", 
                "entailment": "SUPPORTED",
                "contradiction": "REFUTED",
                "neutral": "NEUTRAL"
            }
            
            return {
                "label": label_map.get(result[0]["label"], "NEUTRAL"),
                "confidence": result[0]["score"]
            }
            
        except Exception as e:
            print(f"      NLI model failed: {str(e)[:50]}..., using rule-based fallback")
            return self._rule_based_verification(claim, evidence)
    
    def _rule_based_verification(self, claim: str, evidence: str) -> Dict:
        """
        Lightweight rule-based verification for memory-constrained systems
        """
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        
        # Extract key terms from claim
        claim_words = set(claim_lower.split())
        evidence_words = set(evidence_lower.split())
        
        # Calculate word overlap
        common_words = claim_words.intersection(evidence_words)
        overlap_ratio = len(common_words) / len(claim_words) if claim_words else 0
        
        # Look for explicit contradiction indicators
        contradiction_words = ['not', 'no', 'never', 'false', 'deny', 'denies', 'refute', 'reject']
        support_words = ['yes', 'true', 'confirm', 'support', 'agree', 'verify']
        
        contradiction_score = sum(1 for word in contradiction_words if word in evidence_lower)
        support_score = sum(1 for word in support_words if word in evidence_lower)
        
        # Simple heuristic classification
        if contradiction_score > support_score and overlap_ratio > 0.3:
            return {"label": "REFUTED", "confidence": 0.6 + min(0.3, contradiction_score * 0.1)}
        elif support_score > contradiction_score and overlap_ratio > 0.4:
            return {"label": "SUPPORTED", "confidence": 0.6 + min(0.3, support_score * 0.1)}
        else:
            return {"label": "NEUTRAL", "confidence": 0.5 + min(0.3, overlap_ratio)}
    