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
        
        # List of models to try in order of preference
        models_to_try = [
            "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",  
            "microsoft/deberta-large-mnli",     
            "microsoft/deberta-base-mnli",      
            "facebook/bart-large-mnli",         
            "typeform/distilbert-base-uncased-mnli"  
        ]
        
        self.nli_model = None
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
                print(f"✅ Successfully loaded: {model_name}")
                self.model_name = model_name
                break
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {e}")
                continue
        
        if self.nli_model is None:
            raise RuntimeError("Failed to load any NLI model. Please check your internet connection and try again.")
        
        print("✅ Fallback NLI model loaded successfully")
    
    def verify_claim(self, claim: str, evidence: str) -> Dict:
        """
        Verify claim against evidence using NLI
        Returns: {label: str, confidence: float}
        """
        # Truncate to model's max length
        # text = f"{claim} [SEP] {evidence}"[:512]
        
        # result = self.nli_model(text)[0]

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
    