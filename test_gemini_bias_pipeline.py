import sys
sys.path.append('src')
from config import Config
from verification.gemini_verification import GeminiClaimVerifier

def main():
    """Test Gemini verification with bias-aware pipeline"""
    
    print("="*60)
    print("GEMINI + BIAS-AWARE FACT-CHECKING SYSTEM - TEST")
    print("="*60)
    
    # Step 1: Validate configuration
    print("\n1. Validating configuration...")
    try:
        Config.validate()
        print("Configuration valid")
    except ValueError as e:
        print(f"Configuration error: {e}")
        exit(1)
    
    # Step 2: Initialize components
    print("2. Initializing components...")
    
    from src.evidence_retrieval import AdvancedEvidenceRetriever
    from src.verification import ClaimVerifier
    from src.bias_detection import BiasDetector
    from src.evidence_weighting import EvidenceWeighter, VerdictGenerator
    from src.synthesis import SynthesisEngine
    
    retriever = AdvancedEvidenceRetriever(
        serper_key=Config.SERPER_KEY, 
        groq_key=Config.GROQ_API_KEY
    )
    print("Evidence Retriever initialized")
    
    verifier = ClaimVerifier()
    bias_detector = BiasDetector()
    weighter = EvidenceWeighter()
    verdict_generator = VerdictGenerator()
    synthesizer = SynthesisEngine(Config.GROQ_API_KEY)
    
    # Initialize Gemini verifier
    try:
        gemini_verifier = GeminiClaimVerifier()
        print("Gemini Verifier initialized")
    except Exception as e:
        print(f"Gemini initialization failed: {e}")
        print("Continuing with standard verification only...")
        gemini_verifier = None
    
    print("All components initialized")

    # Step 3: Get claim from user
    print("\n3. Testing with user input claim...")

    print("\n" + "="*60)
    print("Enter claim to fact-check")
    print("="*60)
    print("Examples:")
    print(" - President said giving rice to animals causes shortage")
    print(" - Parliament has low female representation in South Asia")
    print(" - IMF bailout improved economic stability")
    print("\nEnter your claim below:")
    claim = input(">> ").strip()

    if not claim:
        print("No claim entered.")
        exit(1)

    print(f"\nAnalyzing claim: {claim}")
    
    # Step 4: Evidence Retrieval (same as original)
    print("\n" + "="*50)
    print("EVIDENCE RETRIEVAL STRATEGY")
    print("="*50)
    
    print("Phase 1: Searching Local Knowledge Base...")
    
    try:
        parliamentary_evidence = retriever.retrieve_with_hansard_fallback(claim, num_results=8)
        print(f"   Parliamentary/Database search: {len(parliamentary_evidence)} sources found")
        
        if parliamentary_evidence:
            print("   Database sources found:")
            for i, e in enumerate(parliamentary_evidence[:3], 1):
                source_type = e.get('dataset_source', 'unknown')
                authority = e.get('source_alignment', 'unknown')
                score = e.get('relevance_score', 0)
                print(f"      {i}. {source_type.upper()} - {authority} (score: {score:.3f})")
        
        high_quality_db_results = len([e for e in parliamentary_evidence if e.get('relevance_score', 0) > 0.3])
        
        if high_quality_db_results >= 5:
            print(f"   Found {high_quality_db_results} high-quality database sources")
            print("   Skipping web search (sufficient local evidence)")
            evidence = parliamentary_evidence
        else:
            print(f"   Only {high_quality_db_results} high-quality database sources")
            print("   Phase 2: Web Search Fallback...")
            
            web_evidence = retriever.retrieve_hybrid_serper_decomposition(
                claim, 
                num_results=10,
                results_per_query=8
            )
            
            print(f"   Web search: {len(web_evidence)} additional sources")
            
            evidence = parliamentary_evidence + web_evidence
            evidence = retriever._advanced_deduplication(evidence)[:15]
            
            print(f"   Combined evidence: {len(evidence)} total sources")
    
    except Exception as e:
        print(f"   Database search failed: {e}")
        print("   Falling back to web-only search...")
        
        evidence = retriever.retrieve_hybrid_serper_decomposition(
            claim, 
            num_results=15,
            results_per_query=10
        )

    print(f"\nEVIDENCE COLLECTION SUMMARY:")
    print(f"   Total Sources: {len(evidence)}")

    if not evidence:
        print("   No evidence found")
        exit(1)

    # Step 5: Standard Verification and Bias Analysis (same as original)
    verification_results = []
    bias_analyses = []
    
    print("\n" + "="*60)
    print("EVIDENCE VERIFICATION & BIAS ANALYSIS")
    print("="*60)
    
    for i, e in enumerate(evidence, 1):
        print(f"\nSource {i}: {e['source']}")
        print(f"   Title: {e['title'][:60]}...")
        print(f"   URL: {e['link']}")
        
        if 'relevance_score' in e:
            print(f"   Relevance Score: {e['relevance_score']:.3f}")
        
        # Standard verification
        verification_result = verifier.verify_claim(claim, e["snippet"])
        verification_results.append(verification_result)
        
        print(f"   Standard Verdict: {verification_result['label']}")
        print(f"   Standard Confidence: {verification_result['confidence']:.2%}")

        # Bias analysis
        bias_analysis = bias_detector.analyze_source(e["snippet"], e["link"])
        bias_analyses.append(bias_analysis)

        print(f"   BIAS ANALYSIS:")
        if "source_profile" in bias_analysis and bias_analysis["source_profile"]["has_profile"]:
            profile = bias_analysis["source_profile"]
            print(f"      Source Profile: {profile['source']}")
            print(f"      Bias Profile: {profile['bias_interpretation']} ({profile['bias_score']:+.1f})")
            print(f"      Profile Confidence: {profile['confidence']:.0%}")
        else:
            print(f"      No bias profile - using real-time analysis")

        print(f"      Emotional Tone: {bias_analysis['emotional_tone']['emotion']} "
              f"({bias_analysis['emotional_tone']['emotion_score']:.2f})")
        print(f"      Framing: {bias_analysis['framing_bias']['framing_type']} "
              f"({bias_analysis['framing_bias']['framing_score']:.2f})")

    # Step 6: GEMINI VERIFICATION (NEW)
    print("\n" + "="*60)
    print("GEMINI AI VERIFICATION")
    print("="*60)
    
    gemini_result = None
    if gemini_verifier:
        try:
            print("Running Gemini analysis...")
            
            # Convert evidence to format expected by Gemini
            gemini_evidence = []
            for e in evidence:
                gemini_evidence.append({
                    'dataset': e.get('dataset_source', 'web'),
                    'authority': e.get('authority', 0.7),
                    'title': e.get('title', ''),
                    'passage': e.get('snippet', ''),
                    'text': e.get('snippet', ''),
                    'date': 'Unknown date',
                    'source': e.get('source', ''),
                    'score': e.get('relevance_score', 0)
                })
            
            gemini_result = gemini_verifier.verify_claim(claim, gemini_evidence)
            
            print("GEMINI ANALYSIS RESULTS:")
            print("-" * 40)
            print(gemini_verifier.get_verification_summary(gemini_result))
            
        except Exception as e:
            print(f"Gemini verification failed: {e}")
            print("Continuing with standard pipeline only...")

    # Step 7: Standard Evidence Weighting and Verdict (same as original)
    print("\n" + "="*60)
    print("STANDARD EVIDENCE WEIGHTING & VERDICT")
    print("="*60)

    weighted_evidence = weighter.weight_all_evidence(
        claim, evidence, verification_results, bias_analyses
    )

    print("EVIDENCE WEIGHTING RESULTS:")
    for i, item in enumerate(weighted_evidence, 1):
        evidence_url = item["evidence"].get("link", "")
        domain = weighter._extract_domain(evidence_url) if evidence_url else "Unknown"
        
        print(f"\n   Source {i} ({domain}):")
        print(f"      Final Weight: {item['weight']:.3f}")
        print(f"      Bias Alignment: {item['bias_alignment']:.3f}")
        print(f"      Explanation: {item['explanation']}")

    # Generate standard verdict
    final_verdict = verdict_generator.generate_verdict(weighted_evidence)

    # Generate standard AI synthesis
    ai_synthesis = synthesizer.generate_synthesis(
        claim, evidence, verification_results, bias_analyses, final_verdict
    )

    # Step 8: COMPARISON RESULTS
    print("\n" + "="*60)
    print("VERIFICATION COMPARISON")
    print("="*60)
    
    print("STANDARD PIPELINE SYNTHESIS:")
    print("-" * 40)
    print(f"{ai_synthesis}")
    print()
    
    standard_verdict = f"{final_verdict['verdict']} ({final_verdict['confidence']:.0%} confidence)"
    print(f"STANDARD VERDICT: {standard_verdict}")
    
    if gemini_result:
        gemini_verdict = f"{gemini_result.verdict} ({gemini_result.confidence:.0%} confidence)"
        print(f"GEMINI VERDICT: {gemini_verdict}")
        
        print("\nCOMPARISON ANALYSIS:")
        print("-" * 30)
        
        # Compare verdicts
        standard_label = final_verdict['verdict']
        gemini_label = gemini_result.verdict
        
        if standard_label.upper() == gemini_label.upper():
            print("VERDICT AGREEMENT: Both systems agree")
        else:
            print(f"VERDICT DISAGREEMENT: Standard={standard_label}, Gemini={gemini_label}")
        
        # Compare confidence levels
        standard_conf = final_verdict['confidence']
        gemini_conf = gemini_result.confidence
        
        conf_diff = abs(standard_conf - gemini_conf)
        print(f"CONFIDENCE DIFFERENCE: {conf_diff:.1%}")
        
        if conf_diff < 0.1:
            print("   Agreement: Similar confidence levels")
        elif conf_diff < 0.3:
            print("   Moderate difference in confidence")
        else:
            print("   Significant confidence disagreement")
        
        # Show reasoning comparison
        print("\nREASONING COMPARISON:")
        print("Standard reasoning: Based on NLI model + bias weighting")
        print("Gemini reasoning:")
        for i, step in enumerate(gemini_result.reasoning_steps, 1):
            print(f"   {i}. {step}")
    
    print("\n" + "="*60)
    print("DETAILED BREAKDOWN")
    print("="*60)
    
    print("Standard Pipeline Results:")
    print(f"   Support Score: {final_verdict['support_score']:.1%}")
    print(f"   Refute Score: {final_verdict['refute_score']:.1%}")
    print(f"   Neutral Score: {final_verdict['neutral_score']:.1%}")
    
    if gemini_result:
        print(f"\nGemini Analysis:")
        print(f"   Evidence Assessment: {gemini_result.evidence_assessment}")
        print(f"   Limitations: {gemini_result.limitations}")

    print("\n" + "="*60)
    print("SYSTEM PERFORMANCE SUMMARY")
    print("="*60)

    database_sources = len([e for e in evidence if e.get('dataset_source')])
    web_sources = len(evidence) - database_sources
    
    print(f"Evidence Sources: {len(evidence)} total")
    print(f"   Database Sources: {database_sources}")
    print(f"   Web Sources: {web_sources}")
    
    sources_with_profiles = sum(1 for item in weighted_evidence 
                               if weighter._extract_domain(item["evidence"].get("link", "")) in weighter.bias_profiles)
    
    print(f"Bias Profiling: {sources_with_profiles}/{len(evidence)} sources profiled")
    
    if gemini_result:
        print(f"Gemini Integration: SUCCESS")
        print(f"Dual Verification: Standard + AI reasoning available")
    else:
        print(f"Gemini Integration: FAILED")
        print(f"Single Verification: Standard pipeline only")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()