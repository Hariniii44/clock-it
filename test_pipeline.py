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
    
    print("All components initialized")

    # Step 3: Test with user input claim
    print("\n3. Testing with user input claim...")

    print("\n" + "="*60)
    print("Enter claim to fact-check")
    print("="*60)
    print("Examples:")
    print(" - President said giving rice to animals causes shortage")
    print(" - Parliament of Sri Lanka has one of the lowest representations of female MPs in South Asia")
    print(" - Sri Lanka's IMF bailout program has improved the country's economic stability")
    print("\nEnter your claim below:")
    claim = input(">> ").strip()

    if not claim:
        print("No claim entered.")
        exit(1)

    print(f"\nAnalyzing claim: {claim}")
    
    # UPDATED EVIDENCE RETRIEVAL STRATEGY
    print("\n" + "="*50)
    print("EVIDENCE RETRIEVAL STRATEGY")
    print("="*50)
    
    # Step 4: Hybrid Evidence Retrieval (Datasets FIRST, then Serper)
    print("Phase 1: Searching Local Knowledge Base (Nuwan's Datasets)...")
    
    try:
        # Try DatasetManager first for Sri Lankan governmental/news content
        parliamentary_evidence = retriever.retrieve_with_hansard_fallback(claim, num_results=8)
        
        print(f"   Parliamentary/Database search: {len(parliamentary_evidence)} sources found")
        
        # Display database results quality
        if parliamentary_evidence:
            print("   Database sources found:")
            for i, e in enumerate(parliamentary_evidence[:3], 1):
                source_type = e.get('dataset_source', 'unknown')
                authority = e.get('source_alignment', 'unknown')
                score = e.get('relevance_score', 0)
                print(f"      {i}. {source_type.upper()} - {authority} (score: {score:.3f})")
        
        # Add after retrieving parliamentary evidence:
        if parliamentary_evidence:
            print("\nSAMPLE PARLIAMENTARY CONTENT:")
            for i, e in enumerate(parliamentary_evidence[:2], 1):
                print(f"\nSample {i} (score: {e.get('relevance_score', 0):.3f}):")
                snippet = e.get('snippet', '')
                
                # Clean and show meaningful content
                if snippet:
                    # Remove metadata headers and formatting
                    import re
                    cleaned_snippet = re.sub(r'====.*?====', '', snippet)
                    cleaned_snippet = re.sub(r'Volume \d+.*?No\. \d+', '', cleaned_snippet)
                    cleaned_snippet = re.sub(r'\d{4} .*?day,', '', cleaned_snippet)
                    cleaned_snippet = ' '.join(cleaned_snippet.split())
                    
                    if len(cleaned_snippet) > 250:
                        content_display = cleaned_snippet[:250] + "..."
                    else:
                        content_display = cleaned_snippet
                    
                    if content_display.strip() and len(content_display.strip()) > 20:
                        print(f"   Content: \"{content_display}\"")
                    else:
                        print(f"   Content: [Parliamentary document - {len(snippet)} characters]")
                else:
                    print(f"   Content: [No content available]")
        
        # Determine if we need Serper fallback
        high_quality_db_results = len([e for e in parliamentary_evidence if e.get('relevance_score', 0) > 0.3])
        
        if high_quality_db_results >= 5:
            print(f"   Found {high_quality_db_results} high-quality database sources")
            print("   Skipping web search (sufficient local evidence)")
            evidence = parliamentary_evidence
        else:
            print(f"   Only {high_quality_db_results} high-quality database sources")
            print("   Phase 2: Web Search Fallback...")
            
            # Use Serper for additional evidence
            web_evidence = retriever.retrieve_hybrid_serper_decomposition(
                claim, 
                num_results=10,
                results_per_query=8
            )
            
            print(f"   Web search: {len(web_evidence)} additional sources")
            
            # Combine evidence with priority to database sources
            evidence = parliamentary_evidence + web_evidence
            
            # Remove duplicates and limit total
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

    # Enhanced evidence analysis
    print(f"\nEVIDENCE COLLECTION SUMMARY:")
    print(f"   Total Sources: {len(evidence)}")
    
    # Categorize evidence sources
    source_categories = {
        'parliamentary': 0,
        'government': 0,
        'sri_lankan_news': 0,
        'international': 0,
        'unknown': 0
    }
    
    print("   Source Breakdown:")
    for e in evidence:
        source_alignment = e.get('source_alignment', 'unknown')
        dataset_source = e.get('dataset_source', None)
        
        if dataset_source == 'hansard':
            source_categories['parliamentary'] += 1
            print(f"      Parliamentary: {e.get('source', 'Unknown')}")
        elif source_alignment == 'official_primary':
            source_categories['government'] += 1
        elif source_alignment == 'sri_lankan_news':
            source_categories['sri_lankan_news'] += 1
        elif source_alignment == 'international':
            source_categories['international'] += 1
        else:
            source_categories['unknown'] += 1
    
    # Display summary
    for category, count in source_categories.items():
        if count > 0:
            emoji = {
                'parliamentary': '',
                'government': '',
                'sri_lankan_news': '',
                'international': '',
                'unknown': ''
            }[category]
            print(f"      {emoji} {category.replace('_', ' ').title()}: {count}")

    # Show relevance distribution
    print("   Relevance Distribution:")
    relevance_scores = [e.get('relevance_score', 0) for e in evidence if 'relevance_score' in e]
    if relevance_scores:
        print(f"      Highest: {max(relevance_scores):.3f}")
        print(f"      Average: {sum(relevance_scores)/len(relevance_scores):.3f}")
        print(f"      Lowest: {min(relevance_scores):.3f}")

    # Enhanced bias profile preview
    print("   Bias Profile Preview:")
    sri_lankan_sources = 0
    profiled_sources = 0
    
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
        
        # Check against known Sri Lankan domains
        sri_lankan_domains = ["adaderana.lk", "dailymirror.lk", "economynext.com", 
                            "ft.lk", "island.lk", "newsfirst.lk", "themorning.lk"]
        if domain in sri_lankan_domains:
            sri_lankan_sources += 1
            profiled_sources += 1
            print(f"      {domain} (bias profile available)")
        elif e.get('dataset_source') == 'hansard':
            print(f"      parliament.lk (official source)")

    if profiled_sources == 0:
        print("      No Sri Lankan sources with bias profiles detected")
        print("      Bias-aware weighting will be limited")
    else:
        print(f"      {profiled_sources} sources with bias profiles")
    
    if not evidence:
        print("   No evidence found")
        exit(1)

    # Continue with existing verification and bias analysis...
    verification_results = []
    bias_analyses = []
    
    # Step 5: Verify and analyse bias for each piece of evidence
    print("\n" + "="*60)
    print("EVIDENCE VERIFICATION & BIAS ANALYSIS")
    print("="*60)
    
    for i, e in enumerate(evidence, 1):
        print(f"\nSource {i}: {e['source']}")
        print(f"   Title: {e['title'][:60]}...")

        # Show source type and authority
        if e.get('dataset_source'):
            print(f"   Source Type: {e['dataset_source'].title()} Dataset")
            print(f"   Authority: {e.get('source_alignment', 'Unknown')}")
        
        if 'relevance_score' in e:
            print(f"   Relevance Score: {e['relevance_score']:.3f}")

        print(f"   URL: {e['link']}")
        
        # Show actual content snippet for parliamentary sources
        if e.get('dataset_source') == 'hansard':
            content_snippet = e.get('snippet', '')[:300] + "..." if len(e.get('snippet', '')) > 300 else e.get('snippet', '')
            if content_snippet.strip():
                print(f"   Content: \"{content_snippet}\"")
        
        # Verification
        verification_result = verifier.verify_claim(claim, e["snippet"])
        verification_results.append(verification_result)
        
        print(f"   Verdict: {verification_result['label']}")
        print(f"   Confidence: {verification_result['confidence']:.2%}")

        # Enhanced bias analysis with URL context
        bias_analysis = bias_detector.analyze_source(e["snippet"], e["link"])
        bias_analyses.append(bias_analysis)

        # Show enhanced bias information
        print(f"   BIAS ANALYSIS:")
        
        # Check if source profile is available
        if "source_profile" in bias_analysis and bias_analysis["source_profile"]["has_profile"]:
            profile = bias_analysis["source_profile"]
            print(f"      Source Profile: {profile['source']}")
            print(f"      Bias Profile: {profile['bias_interpretation']} ({profile['bias_score']:+.1f})")
            print(f"      Profile Confidence: {profile['confidence']:.0%} ({profile['articles_analyzed']} articles)")
        else:
            print(f"      No bias profile available - using real-time analysis")

        # Real-time analysis results
        print(f"      Emotional Tone: {bias_analysis['emotional_tone']['emotion']} "
              f"({bias_analysis['emotional_tone']['emotion_score']:.2f})")
        print(f"      Framing: {bias_analysis['framing_bias']['framing_type']} "
              f"({bias_analysis['framing_bias']['framing_score']:.2f})")

    # Continue with existing weighting and verdict generation...
    print("\n" + "="*60)
    print("EVIDENCE WEIGHTING & VERDICT GENERATION")
    print("="*60)

    weighted_evidence = weighter.weight_all_evidence(
        claim, evidence, verification_results, bias_analyses
    )

    # Enhanced weighting display
    print("EVIDENCE WEIGHTING RESULTS:")
    for i, item in enumerate(weighted_evidence, 1):
        evidence_url = item["evidence"].get("link", "")
        domain = weighter._extract_domain(evidence_url) if evidence_url else "Unknown"
        
        print(f"\n   Source {i} ({domain}):")
        print(f"      Final Weight: {item['weight']:.3f}")
        print(f"      Bias Alignment: {item['bias_alignment']:.3f}")
        print(f"      Explanation: {item['explanation']}")

    # Generate final verdict with enhanced information
    final_verdict = verdict_generator.generate_verdict(weighted_evidence)

    # Generate AI synthesis
    from src.synthesis import SynthesisEngine
    synthesizer = SynthesisEngine(Config.GROQ_API_KEY)
    ai_synthesis = synthesizer.generate_synthesis(
        claim, evidence, verification_results, bias_analyses, final_verdict
    )

    # Display final results (keep existing format)
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    print(f"{ai_synthesis}")
    print()
    
    print(f"FACT-CHECK VERDICT: {final_verdict['verdict']} ({final_verdict['confidence']:.0%} confidence)")
    
    # Show factual claim detection
    if 'is_factual_claim' in final_verdict:
        claim_type = "Factual Claim" if final_verdict['is_factual_claim'] else "Opinion/Subjective"
        print(f"Claim Type: {claim_type}")
    
    print()
    print("Detailed Breakdown:")
    print(f"   Support Score: {final_verdict['support_score']:.1%}")
    print(f"   Refute Score: {final_verdict['refute_score']:.1%}")
    print(f"   Neutral Score: {final_verdict['neutral_score']:.1%}")
    
    # Enhanced verdict information
    if 'diversity_score' in final_verdict:
        print(f"   Source Diversity Score: {final_verdict['diversity_score']:.1%}")
    
    print()
    print("Uncertainty Analysis:")
    unc = final_verdict['uncertainty_decomposition']
    print(f"   Epistemic (model variance): {unc['epistemic']}")
    print(f"   Aleatoric (lack of evidence): {unc['aleatoric']}")
    print(f"   Bias-induced (source disagreement): {unc['bias_induced']}")
    print(f"   Explanation: {unc['explanation']}")
    
    # ENHANCED: Summary of hybrid retrieval effectiveness
    print("\n" + "="*60)
    print("HYBRID RETRIEVAL SYSTEM PERFORMANCE")
    print("="*60)

    database_sources = len([e for e in evidence if e.get('dataset_source')])
    web_sources = len(evidence) - database_sources
    
    print(f"Database Sources: {database_sources}")
    print(f"Web Sources: {web_sources}")
    
    if database_sources > 0:
        parliamentary_sources = len([e for e in evidence if e.get('dataset_source') == 'hansard'])
        if parliamentary_sources > 0:
            print(f"   Parliamentary: {parliamentary_sources} hansard chunks")
        
        avg_db_relevance = sum(e.get('relevance_score', 0) for e in evidence if e.get('dataset_source')) / database_sources
        print(f"   Avg Database Relevance: {avg_db_relevance:.3f}")
    
    if web_sources > 0:
        avg_web_relevance = sum(e.get('relevance_score', 0) for e in evidence if not e.get('dataset_source')) / web_sources
        print(f"   Avg Web Relevance: {avg_web_relevance:.3f}")
    
    # Strategy effectiveness
    if database_sources >= 5:
        strategy_status = "DATABASE-PRIMARY (Optimal)"
    elif database_sources >= 2:
        strategy_status = "HYBRID (Good)"
    else:
        strategy_status = "WEB-PRIMARY (Fallback)"
    
    print(f"\nRetrieval Strategy: {strategy_status}")

    # Continue with existing bias profiling summary...
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
        print(f"\nBias Distribution of Profiled Sources:")
        for category, count in bias_distribution.items():
            if count > 0:
                emoji = {"opposition": "", "neutral": "", "pro_government": ""}[category]
                print(f"   {emoji} {category.replace('_', '-').title()}: {count}")
        
        # Calculate and show bias impact
        avg_bias_impact = total_bias_impact / len(weighted_evidence) if weighted_evidence else 0
        print(f"\nAverage Bias Impact: {avg_bias_impact:.3f}")
        
        if avg_bias_impact > 0.3:
            print("    High bias impact detected - weights significantly adjusted")
        elif avg_bias_impact > 0.1:
            print("    Moderate bias impact - weights moderately adjusted") 
        else:
            print("    Low bias impact - minimal weight adjustments")

    # Integration status
    if sources_with_profiles >= 3:
        status = "FULLY ACTIVE"
    elif sources_with_profiles >= 1:
        status = "ACTIVE"
    else:
        status = "LIMITED"

    print(f"\nBias-aware weighting: {status}")

    # Final system performance summary
    print(f"\nSYSTEM PERFORMANCE SUMMARY:")
    print(f"   Evidence Retrieval: {len(evidence)} sources ({database_sources} local, {web_sources} web)")
    print(f"   Bias Profiling: {status} ({sources_with_profiles} profiled sources)")
    print(f"   Verification: All sources processed")
    print(f"   Final Verdict: {final_verdict['verdict']} ({final_verdict['confidence']:.0%} confidence)")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)