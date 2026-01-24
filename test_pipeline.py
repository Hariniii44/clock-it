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
    
    from src.evidence_retrieval import AdvancedEvidenceRetriever
    from src.verification import ClaimVerifier
    from src.bias_detection import BiasDetector
    from src.evidence_weighting import EvidenceWeighter, VerdictGenerator
    
    retriever = AdvancedEvidenceRetriever(
        serpapi_key=Config.SERPAPI_KEY, 
        gemini_key=Config.GEMINI_API_KEY, 
        groq_key=Config.GROQ_API_KEY
    )
    print("   ✅ Advanced Evidence Retriever initialized")
    
    verifier = ClaimVerifier()
    bias_detector = BiasDetector()
    weighter = EvidenceWeighter()
    verdict_generator = VerdictGenerator()
    
    # Step 3: Test with sample claim
    print("\n3. Testing with sample claim...")
    
    claim = "Parliament of Sri Lanka has one of the lowest representations of female MPs in all of South Asia."
    print(f"\n   CLAIM: {claim}")
    
    # Retrieve evidence
    print(f"\n   Retrieving {Config.NUM_EVIDENCE_SOURCES} evidence sources...")
    evidence = retriever.retrieve_diverse_evidence(
        claim, 
        num_results=Config.NUM_EVIDENCE_SOURCES,
        include_fact_checkers=True
    )

    # Show retrieval analysis
    print(f"   ✅ Found {len(evidence)} evidence sources")
    print("   📊 Source diversity:")
    alignments = {}
    for e in evidence:
        alignment = e.get('source_alignment', 'unknown')
        alignments[alignment] = alignments.get(alignment, 0) + 1
    for alignment, count in alignments.items():
        print(f"      {alignment}: {count}")
    
    if not evidence:
        print("   ❌ No evidence found")
        exit(1)

    verification_results = []
    bias_analyses = []
    
    # Step 4: Verify and analyse bias for each piece of evidence
    print("4. Verifying claim and analysing bias for each piece of evidence...\n")
    
    for i, e in enumerate(evidence, 1):
        print(f"   Source {i}: {e['source']}")
        print(f"   Title: {e['title'][:60]}...")
        print(f"   URL: {e['link']}")
        
        # Verification
        verification_result = verifier.verify_claim(claim, e["snippet"])
        verification_results.append(verification_result)
        print(f"   Verdict: {verification_result['label']}")
        print(f"   Confidence: {verification_result['confidence']:.2%}")

        # Bias analysis
        bias_analysis = bias_detector.analyze_source(e["snippet"], e["link"])
        bias_analyses.append(bias_analysis)

        print(f"   Political Stance: {bias_analysis['political_stance']['political_stance']} "
              f"(score: {bias_analysis['political_stance']['stance_score']})")
        print(f"   Emotional Tone: {bias_analysis['emotional_tone']['emotion']} "
              f"(score: {bias_analysis['emotional_tone']['emotion_score']:.2f})")
        print(f"   Framing: {bias_analysis['framing_bias']['framing_type']} "
              f"(score: {bias_analysis['framing_bias']['framing_score']:.2f})")
        print("-" * 40)

    # Step 5: Weight evidence and generate final verdict
    print("5. Weighting evidence and generating final verdict...\n")

    weighted_evidence = weighter.weight_all_evidence(
        claim, evidence, verification_results, bias_analyses
    )

    for i, item in enumerate(weighted_evidence, 1):
        print(f"   Source {i} Weight: {item['weight']:.2f}")
        print(f"   Bias Alignment: {item['bias_alignment']:.2f}")
        print(f"   Explanation: {item['explanation']}")

    # Generate final verdict
    final_verdict = verdict_generator.generate_verdict(weighted_evidence)

    print("="*60)
    print("FINAL VERDICT")
    print("="*60)
    print(f"Verdict: {final_verdict['verdict']}")
    print(f"Confidence: {final_verdict['confidence']:.2%}")
    print(f"Support Score: {final_verdict['support_score']:.2%}")
    print(f"Refute Score: {final_verdict['refute_score']:.2%}")
    print(f"Neutral Score: {final_verdict['neutral_score']:.2%}")
    print()
    print("Uncertainty Breakdown:")
    unc = final_verdict['uncertainty_decomposition']
    print(f"  Epistemic (model variance): {unc['epistemic']}")
    print(f"  Aleatoric (lack of evidence): {unc['aleatoric']}")
    print(f"  Bias-induced (source disagreement): {unc['bias_induced']}")
    print(f"  Explanation: {unc['explanation']}")
    
    print("="*60)
    print("TEST COMPLETE")
    print("="*60)