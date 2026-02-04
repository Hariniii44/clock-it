from config import Config

if __name__ == "__main__":
    print("="*60)
    print("BIAS-AWARE FACT-CHECKING SYSTEM - TEST")
    print("="*60)
    
    # Step 1: Validate configuration
    print("\n1. Validating configuration...")
    try:
        Config.validate()
        print("Configuration valid")
        # Config.print_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        exit(1)
    
    # Step 2: Initialize components
    print("2. Initializing components...")
    
    from src.evidence_retrieval import AdvancedEvidenceRetriever
    from src.verification import ClaimVerifier
    from src.bias_detection import BiasDetector
    from src.evidence_weighting import EvidenceWeighter, VerdictGenerator
    
    retriever = AdvancedEvidenceRetriever(
        serper_key=Config.SERPER_KEY, 
        groq_key=Config.GROQ_API_KEY
    )
    print("Evidence Retriever initialized")
    
    verifier = ClaimVerifier()
    bias_detector = BiasDetector()
    weighter = EvidenceWeighter()
    verdict_generator = VerdictGenerator()
    
    # Step 3: Test with user input claim
    print("\n3. Testing with user input claim...")

    print("\n" + "="*60)
    print("Enter claim to fact-check")
    print("="*60)
    print("Examples:")
    print(" - Parliament of Sri Lanka has one of the lowest representations of female MPs in all of South Asia.")
    print(" - Sri Lanka's IMF bailout program has improved the country's economic stability.")
    print("\nEnter your claim below:")
    claim = input(">> ").strip()

    if not claim:
        print("No claim entered.")
        exit(1)

    print(f"\n   Analyzing claim: {claim}")
    
    evidence = retriever.retrieve_hybrid_serper_decomposition(
        claim, 
        num_results=20,
        results_per_query=10
    )

    # Show retrieval analysis
    print(f"   Found {len(evidence)} evidence sources")
    print("   Source diversity:")
    alignments = {}
    for e in evidence:
        alignment = e.get('source_alignment', 'unknown')
        alignments[alignment] = alignments.get(alignment, 0) + 1
    for alignment, count in alignments.items():
        print(f"      {alignment}: {count}")

    print("Relevance Distribution:")
    relevance_scores = [e.get('relevance_score', 0) for e in evidence]
    if relevance_scores:
        print(f"      Highest: {max(relevance_scores):.3f}")
        print(f"      Lowest: {min(relevance_scores):.3f}")
        print(f"      Average: {sum(relevance_scores)/len(relevance_scores):.3f}")

    print("Quick bias profile check:")
    sri_lankan_preview = 0
    for e in evidence:
        domain = None
        try:
            if e.get("link", "").startswith('http'):
                url = e["link"].split('://', 1)[1]
            else:
                url = e["link"]
            domain = url.split('/')[0]
            if domain.startswith('www.'):
                domain = domain[4:]
        except:
            domain = "unknown"
        
        # Quick check against known Sri Lankan domains
        sri_lankan_domains = ["adaderana.lk", "dailymirror.lk", "economynext.com", 
                            "ft.lk", "island.lk", "newsfirst.lk"]
        if domain in sri_lankan_domains:
            sri_lankan_preview += 1
            print(f"      🇱🇰 {domain}")

    if sri_lankan_preview == 0:
        print("       No Sri Lankan sources detected - bias profiling will be limited")
    else:
        print(f"       {sri_lankan_preview} Sri Lankan sources detected")
    
    if not evidence:
        print("   No evidence found")
        exit(1)

    verification_results = []
    bias_analyses = []
    
    # Step 4: Verify and analyse bias for each piece of evidence
    print("4. Verifying claim and analysing bias for each piece of evidence...\n")
    
    for i, e in enumerate(evidence, 1):
        print(f"   Source {i}: {e['source']}")
        print(f"   Title: {e['title'][:60]}...")

        if 'relevance_score' in e:
            print(f"   Relevance Score: {e['relevance_score']:.3f}")

        print(f"   URL: {e['link']}")
        
        # Verification
        verification_result = verifier.verify_claim(claim, e["snippet"])
        verification_results.append(verification_result)
        print(f"      RAW MODEL OUTPUT: {verification_result.get('raw_output', 'N/A')}")
        print(f"   Verdict: {verification_result['label']}")
        print(f"   Confidence: {verification_result['confidence']:.2%}")

        # Enhanced bias analysis with URL context
        bias_analysis = bias_detector.analyze_source(e["snippet"], e["link"])
        bias_analyses.append(bias_analysis)

        # Show enhanced bias information
        print(f"BIAS ANALYSIS:")
        
        # Check if source profile is available
        if "source_profile" in bias_analysis and bias_analysis["source_profile"]["has_profile"]:
            profile = bias_analysis["source_profile"]
            print(f"       Source Profile Available: {profile['source']}")
            print(f"       Bias Profile: {profile['bias_interpretation']} ({profile['bias_score']:+.1f})")
            print(f"       Profile Confidence: {profile['confidence']:.0%} ({profile['articles_analyzed']} articles)")
            
            if "combined_score" in bias_analysis:
                combined = bias_analysis["combined_score"]
                print(f"       Analysis Method: {combined['method']}")
                print(f"       Overall Bias Score: {combined['overall_bias']:.2f}")
                if combined["method"] == "combined":
                    print(f"       Source Bias Component: {combined.get('source_bias_score', 'N/A')}")
        else:
            print(f"       No Pre-computed Profile Available")
            print(f"       Using Real-time Analysis Only")

        # Real-time analysis results
        print(f"       Real-time Analysis:")
        print(f"         Emotional Tone: {bias_analysis['emotional_tone']['emotion']} "
              f"(score: {bias_analysis['emotional_tone']['emotion_score']:.2f})")
        print(f"         Framing: {bias_analysis['framing_bias']['framing_type']} "
              f"(score: {bias_analysis['framing_bias']['framing_score']:.2f})")
        print("-" * 40)

    # Step 5: Weight evidence and generate final verdict
    print("5. Weighting evidence and generating final verdict...\n")

    weighted_evidence = weighter.weight_all_evidence(
        claim, evidence, verification_results, bias_analyses
    )

    # Enhanced weighting display
    print("    EVIDENCE WEIGHTING RESULTS:")
    for i, item in enumerate(weighted_evidence, 1):
        evidence_url = item["evidence"].get("link", "")
        domain = weighter._extract_domain(evidence_url) if evidence_url else "Unknown"
        
        print(f"   Source {i} ({domain}):")
        print(f"      Weight: {item['weight']:.2f}")
        print(f"      Bias Alignment: {item['bias_alignment']:.2f}")
        print(f"      Explanation: {item['explanation']}")
        print()

    # Generate final verdict with enhanced information
    final_verdict = verdict_generator.generate_verdict(weighted_evidence)

    # Generate AI synthesis
    from src.synthesis import SynthesisEngine
    synthesizer = SynthesisEngine(Config.GROQ_API_KEY)
    ai_synthesis = synthesizer.generate_synthesis(
        claim, evidence, verification_results, bias_analyses, final_verdict
    )

    # Display synthesis first (Option C format)
    print("="*60)
    print("VERDICTS EXPLANATION")
    print("="*60)
    print(f" {ai_synthesis}")
    print()
    
    print(f"FACT-CHECK VERDICT: {final_verdict['verdict']} ({final_verdict['confidence']:.0%} confidence)")
    
    # Show factual claim detection
    if 'is_factual_claim' in final_verdict:
        claim_type = "Factual Claim" if final_verdict['is_factual_claim'] else "Opinion/Subjective"
        print(f"Claim Type: {claim_type}")
    
    print()
    
    print("▼ Detailed Analysis Available:")
    print("    Evidence Breakdown")
    print("    Bias Analysis") 
    print("    Uncertainty Analysis")
    print("    System Performance")
    print()

    # Show detailed analysis
    print("="*60)
    print("DETAILED ANALYSIS")
    print("="*60)

    # Show enhanced evidence breakdown
    print("\nEVIDENCE BREAKDOWN:")
    print("-" * 40)
    
    for i, item in enumerate(weighted_evidence, 1):
        evidence_url = item["evidence"].get("link", "")
        domain = weighter._extract_domain(evidence_url) if evidence_url else "Unknown"
        authority_weight = weighter._get_authority_weight(evidence_url)
        
        print(f"\n{i}. {item['evidence'].get('source', 'Unknown Source')}")
        print(f"   Domain: {domain}")
        print(f"   Authority: {authority_weight:.1f}x")
        print(f"   Verification: {item['verification']['label']} ({item['verification']['confidence']:.0%})")
        print(f"   Final Weight: {item['weight']:.3f}")

    # Continue with existing detailed output...
    print(f"Support Score: {final_verdict['support_score']:.2%}")
    print(f"Refute Score: {final_verdict['refute_score']:.2%}")
    print(f"Neutral Score: {final_verdict['neutral_score']:.2%}")


    print("="*60)
    print("FINAL VERDICT")
    print("="*60)
    print(f"Verdict: {final_verdict['verdict']}")
    print(f"Confidence: {final_verdict['confidence']:.2%}")
    print(f"Support Score: {final_verdict['support_score']:.2%}")
    print(f"Refute Score: {final_verdict['refute_score']:.2%}")
    print(f"Neutral Score: {final_verdict['neutral_score']:.2%}")
    
    # Enhanced verdict information
    if 'diversity_score' in final_verdict:
        print(f"Source Diversity Score: {final_verdict['diversity_score']:.2%}")
    
    print()
    print("Uncertainty Breakdown:")
    unc = final_verdict['uncertainty_decomposition']
    print(f"  Epistemic (model variance): {unc['epistemic']}")
    print(f"  Aleatoric (lack of evidence): {unc['aleatoric']}")
    print(f"  Bias-induced (source disagreement): {unc['bias_induced']}")
    print(f"  Explanation: {unc['explanation']}")
    
    # Summary of bias profiling integration
    print("\n" + "="*60)
    print("BIAS PROFILING INTEGRATION SUMMARY")
    print("="*60)

    # Count sources with profiles
    sources_with_profiles = 0
    sri_lankan_sources = 0
    bias_distribution = {"opposition": 0, "neutral": 0, "pro_government": 0}
    total_bias_impact = 0

    for item in weighted_evidence:
        url = item["evidence"].get("link", "")
        domain = weighter._extract_domain(url)
        
        # Add to total bias impact for all sources
        total_bias_impact += item["bias_alignment"]
        
        if domain in weighter.bias_profiles:
            sources_with_profiles += 1
            sri_lankan_sources += 1

            # Categorize bias
            score = weighter.bias_profiles[domain]["bias_score"]
            if score < -10:
                bias_distribution["opposition"] += 1
            elif score > 10:
                bias_distribution["pro_government"] += 1
            else:
                bias_distribution["neutral"] += 1

    # Display results
    print(f"Sources Analyzed: {len(weighted_evidence)}")
    print(f"Sri Lankan Sources with Bias Profiles: {sources_with_profiles}/{len(weighted_evidence)}")
    print(f"International/Unknown Sources: {len(weighted_evidence) - sri_lankan_sources}")

    if sources_with_profiles > 0:
        print(f"\n Bias Distribution of Profiled Sources:")
        for category, count in bias_distribution.items():
            if count > 0:
                emoji = {"opposition": "🔴", "neutral": "🟡", "pro_government": "🟢"}[category]
                print(f"   {emoji} {category.replace('_', '-').title()}: {count}")
        
        # Calculate and show bias impact
        avg_bias_impact = total_bias_impact / len(weighted_evidence) if weighted_evidence else 0
        print(f"\n Average Bias Impact: {avg_bias_impact:.3f}")
        
        if avg_bias_impact > 0.3:
            print("    High bias impact detected - weights significantly adjusted")
        elif avg_bias_impact > 0.1:
            print("    Moderate bias impact - weights moderately adjusted") 
        else:
            print("    Low bias impact - minimal weight adjustments")

    # Integration status with more granular feedback
    if sources_with_profiles >= 3:
        status = "FULLY ACTIVE"
    elif sources_with_profiles >= 1:
        status = "ACTIVE"
    else:
        status = "LIMITED"

    print(f"\n Bias-aware weighting: {status}")

    # Performance summary
    print(f"\nSYSTEM PERFORMANCE:")
    print(f"   Evidence Retrieval:  {len(evidence)} sources found")
    status_parts = status.split()
    status_word = status_parts[-1] if status_parts else "UNKNOWN"
    print(f"   Bias Profiling: {status_word} ({sources_with_profiles} profiled sources)")
    print(f"   Verification:  All sources processed")
    print(f"   Final Verdict:  {final_verdict['verdict']} ({final_verdict['confidence']:.0%})")

    print("="*60)
    print("TEST COMPLETE")
    print("="*60)