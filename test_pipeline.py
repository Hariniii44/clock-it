# test_basic_pipeline.py
from typing import Dict, List
from config import Config

if __name__ == "__main__":
    print("="*60)
    print("BIAS-AWARE FACT-CHECKING SYSTEM - TEST")
    print("="*60)
    
    # Step 1: Validate configuration
    print("\n1. Validating configuration...")
    try:
        Config.validate()
        print("   ✅ Configuration valid")
        Config.print_config()
    except ValueError as e:
        print(f"   ❌ Configuration error: {e}")
        exit(1)
    
    # Step 2: Initialize components
    print("2. Initializing components...")
    
    from src.evidence_retrieval import EvidenceRetriever
    from src.verification import ClaimVerifier
    from src.bias_detection import BiasDetector
    
    retriever = EvidenceRetriever(serpapi_key=Config.SERPAPI_KEY)
    print("   ✅ Evidence Retriever initialized")
    
    verifier = ClaimVerifier()
    print("   ✅ Claim Verifier initialized")

    bias_detector = BiasDetector()
    print("   ✅ Bias Detector initialized")
    
    # Step 3: Test with sample claim
    print("\n3. Testing with sample claim...")
    
    # FR01: Accept textual claims
    claim = "GDP growth in Sri Lanka reached 5% in 2024"
    print(f"\n   CLAIM: {claim}")
    
    # FR02: Retrieve evidence
    print(f"\n   Retrieving {Config.NUM_EVIDENCE_SOURCES} evidence sources...")
    evidence = retriever.retrieve_evidence(
        claim, 
        num_results=Config.NUM_EVIDENCE_SOURCES
    )
    
    if not evidence:
        print("   ❌ No evidence found")
        exit(1)
    
    print(f"   ✅ Found {len(evidence)} evidence sources\n")
    
    # Step 4: Verify and analyse bias for each piece of evidence
    print("4. Verifying claim and analysing bias for each piece of evidence...\n")
    
    for i, e in enumerate(evidence, 1):
        print(f"   Source {i}: {e['source']}")
        print(f"   Title: {e['title'][:60]}...")
        print(f"   URL: {e['link']}")
        
        #verification
        verification_result = verifier.verify_claim(claim, e["snippet"])
        print(f"   Verdict: {verification_result['label']}")
        print(f"   Confidence: {verification_result['confidence']:.2%}")

        # bias analysis
        bias_analysis = bias_detector.analyze_source(e["snippet"], e["link"])

        print(f"   Political Stance: {bias_analysis['political_stance']['political_stance']} "
              f"(score: {bias_analysis['political_stance']['stance_score']})")
        print(f"   Emotional Tone: {bias_analysis['emotional_tone']['emotion']} "
              f"(score: {bias_analysis['emotional_tone']['emotion_score']:.2f})")
        print(f"   Framing: {bias_analysis['framing_bias']['framing_type']} "
              f"(score: {bias_analysis['framing_bias']['framing_score']:.2f})")
        print("-" * 40)
        print()
    
    print("="*60)
    print("TEST COMPLETE")
    print("="*60)
    