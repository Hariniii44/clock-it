"""
Simple debug test for evaluation components
"""
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import Config
    print("✅ Config imported successfully")
    print(f"SERPER_KEY exists: {bool(Config.SERPER_KEY)}")
    print(f"GROQ_API_KEY exists: {bool(Config.GROQ_API_KEY)}")
    
    from evidence_retrieval.evidence_retrieval import AdvancedEvidenceRetriever
    print("✅ AdvancedEvidenceRetriever imported successfully")
    
    from verification.verification import ClaimVerifier
    print("✅ ClaimVerifier imported successfully")
    
    from bias_detection.bias_detection import BiasDetector
    print("✅ BiasDetector imported successfully")
    
    from evidence_weighting.evidence_weighting import EvidenceWeighter
    print("✅ EvidenceWeighter imported successfully")
    
    from synthesis.synthesis_engine import SynthesisEngine
    print("✅ SynthesisEngine imported successfully")
    
    # Test initialization
    print("\nTesting component initialization...")
    
    evidence_retriever = AdvancedEvidenceRetriever(
        serper_key=Config.SERPER_KEY,
        groq_key=Config.GROQ_API_KEY
    )
    print("✅ Evidence retriever initialized")
    
    fact_verifier = ClaimVerifier()
    print("✅ Fact verifier initialized")
    
    bias_detector = BiasDetector()
    print("✅ Bias detector initialized")
    
    evidence_weighter = EvidenceWeighter()
    print("✅ Evidence weighter initialized")
    
    synthesis_engine = SynthesisEngine(Config.GROQ_API_KEY)
    print("✅ Synthesis engine initialized")
    
    print("\n🎉 All components initialized successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()