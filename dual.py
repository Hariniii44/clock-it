"""Dual Evidence Retrieval with Gemini Synthesis"""
import sys
sys.path.append('src')
from config import Config
from verification.gemini_verification import GeminiClaimVerifier

def main():
    """Dual evidence approach: Database + Web + Gemini synthesis"""
    
    print("="*60)
    print("DUAL EVIDENCE + GEMINI SYNTHESIS - TEST")
    print("="*60)
    
    # Components
    from src.evidence_retrieval import AdvancedEvidenceRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.verification import ClaimVerifier
    from src.bias_detection import BiasDetector
    from src.evidence_weighting import EvidenceWeighter, VerdictGenerator
    
    hybrid_retriever = HybridRetriever()
    web_retriever = AdvancedEvidenceRetriever(
        serper_key=Config.SERPER_KEY, 
        groq_key=Config.GROQ_API_KEY
    )
    
    verifier = ClaimVerifier()
    bias_detector = BiasDetector()
    weighter = EvidenceWeighter()
    verdict_generator = VerdictGenerator()
    
    try:
        gemini_verifier = GeminiClaimVerifier()
        print("Gemini Verifier initialized")
    except Exception as e:
        print(f"Gemini initialization failed: {e}")
        exit(1)
    
    # Get claim
    print("\n" + "="*60)
    print("Enter claim to fact-check:")
    claim = input(">> ").strip()
    
    if not claim:
        exit(1)
    
    print(f"\nAnalyzing claim: {claim}")
    
    # PARALLEL RETRIEVAL
    print("\n" + "="*60)
    print("DUAL EVIDENCE RETRIEVAL")
    print("="*60)
    
    # 1. DATABASE EVIDENCE
    print("\n1. DATABASE EVIDENCE (Government Documents):")
    print("-" * 40)
    
    try:
        database_evidence = hybrid_retriever.hybrid_search(
            query=claim,
            claim_types=None,
            total_results=10,
            use_query_expansion=True
        )
        
        # Convert format
        db_formatted = []
        for result in database_evidence:
            formatted = {
                'source': result['source'],
                'title': result['title'],
                'snippet': result.get('passage', result['text']),
                'link': result['url'],
                'dataset_source': result['dataset'],
                'authority': result['authority'],
                'relevance_score': result['similarity_score'],
                'evidence_type': 'database'
            }
            db_formatted.append(formatted)
        
        print(f"Database results: {len(db_formatted)}")
        
        # Show top 3 database results with dates/relevance
        print("Top database results:")
        for i, e in enumerate(db_formatted[:3], 1):
            print(f"  {i}. [{e['dataset_source']}] {e['title'][:80]}...")
            print(f"     Score: {e['relevance_score']:.3f}")
        
    except Exception as e:
        print(f"Database search failed: {e}")
        db_formatted = []
    
    # 2. WEB EVIDENCE  
    print(f"\n2. WEB EVIDENCE (Current News/Analysis):")
    print("-" * 40)
    
    try:
        web_evidence = web_retriever.retrieve_hybrid_serper_decomposition(
            claim, 
            num_results=10,
            results_per_query=8
        )
        
        # Add evidence type
        for e in web_evidence:
            e['evidence_type'] = 'web'
        
        print(f"Web results: {len(web_evidence)}")
        
        # Show top 3 web results
        print("Top web results:")
        for i, e in enumerate(web_evidence[:3], 1):
            print(f"  {i}. [{e.get('source', 'Web')}] {e['title'][:80]}...")
        
    except Exception as e:
        print(f"Web search failed: {e}")
        web_evidence = []
    
    # EVIDENCE ANALYSIS
    print(f"\n" + "="*60)
    print("EVIDENCE ANALYSIS")
    print("="*60)
    
    print(f"Database Evidence: {len(db_formatted)} sources")
    print(f"Web Evidence: {len(web_evidence)} sources")
    print(f"Total Evidence: {len(db_formatted) + len(web_evidence)} sources")
    
    # Quick verification for both sets
    all_evidence = db_formatted + web_evidence
    verification_results = []
    bias_analyses = []
    
    print(f"\nRunning verification on all sources...")
    
    for i, e in enumerate(all_evidence, 1):
        # Standard verification
        verification_result = verifier.verify_claim(claim, e["snippet"])
        verification_results.append(verification_result)
        
        # Bias analysis
        bias_analysis = bias_detector.analyze_source(e["snippet"], e["link"])
        bias_analyses.append(bias_analysis)
        
        print(f"Source {i} [{e['evidence_type']}]: {verification_result['label']} ({verification_result['confidence']:.1%})")
    

    # Add this after the individual bias analysis (around line 120 in your current code):





    # CROSS-SOURCE FRAMING ANALYSIS
    print(f"\n" + "="*60)
    print("CROSS-SOURCE FRAMING ANALYSIS")
    print("="*60)

    try:
        cross_source_framing = bias_detector.analyze_cross_source_framing(claim, all_evidence, bias_analyses)
        
        print(f"Cross-source analysis completed:")
        print(f"  Emotional baseline: {cross_source_framing['emotional_baseline']:.3f}")
        print(f"  Emotional outliers: {len(cross_source_framing['emotional_outliers'])}")
        print(f"  Entities with different framing: {len(cross_source_framing['semantic_differences'])}")
        print(f"  Framing consistency: {cross_source_framing['cross_source_summary']['framing_consistency']}")
        
        # Show emotional outliers
        if cross_source_framing['emotional_outliers']:
            print(f"\n  Emotional outliers detected:")
            for outlier in cross_source_framing['emotional_outliers']:
                source_idx = int(outlier['source_id'].split('_')[1])
                source_name = all_evidence[source_idx].get('source', 'Unknown')
                print(f"    Source {source_idx+1} ({source_name}): {outlier['type']} (score: {outlier['score']:.3f})")
        
        # Show systematic patterns
        patterns = cross_source_framing['systematic_patterns']
        if patterns.get('database_vs_web'):
            db_avg = patterns['database_vs_web'].get('database_emotional_avg', 0)
            web_avg = patterns['database_vs_web'].get('web_emotional_avg', 0)
            if db_avg > 0 or web_avg > 0:
                print(f"\n  Database vs Web emotional intensity:")
                print(f"    Database sources average: {db_avg:.3f}")
                print(f"    Web sources average: {web_avg:.3f}")
                
                if abs(db_avg - web_avg) > 0.2:
                    if db_avg > web_avg:
                        print(f"    Database sources use more emotional language")
                    else:
                        print(f"    Web sources use more emotional language")
        
    except Exception as e:
        print(f"Cross-source framing analysis failed: {e}")
        cross_source_framing = None





    # GEMINI DUAL ANALYSIS
    print(f"\n" + "="*60)
    print("GEMINI DUAL EVIDENCE ANALYSIS")  
    print("="*60)
    
    try:
        # Format for Gemini
        gemini_database_evidence = []
        for e in db_formatted:
            gemini_database_evidence.append({
                'dataset': e.get('dataset_source', 'database'),
                'authority': e.get('authority', 1.0),
                'title': e.get('title', ''),
                'text': e.get('snippet', ''),
                'source': e.get('source', ''),
                'score': e.get('relevance_score', 0),
                'evidence_type': 'government_database'
            })
        
        gemini_web_evidence = []
        for e in web_evidence:
            gemini_web_evidence.append({
                'dataset': 'web_search',
                'authority': e.get('authority', 0.7),
                'title': e.get('title', ''),
                'text': e.get('snippet', ''),
                'source': e.get('source', ''),
                'score': e.get('relevance_score', 0.8),
                'evidence_type': 'web_search'
            })
        
        # Get Gemini analysis for both evidence sets
        print("Analyzing database evidence with Gemini...")
        database_result = gemini_verifier.verify_claim(claim, gemini_database_evidence)
        
        print("Analyzing web evidence with Gemini...")
        web_result = gemini_verifier.verify_claim(claim, gemini_web_evidence)
        
        print("Analyzing combined evidence with Gemini...")
        combined_result = gemini_verifier.verify_claim(claim, gemini_database_evidence + gemini_web_evidence)
        
        # RESULTS COMPARISON
        print(f"\n" + "="*60)
        print("GEMINI ANALYSIS RESULTS")
        print("="*60)
        
        print("DATABASE EVIDENCE ANALYSIS:")
        print("-" * 30)
        print(f"Verdict: {database_result.verdict} ({database_result.confidence:.0%})")
        print(f"Assessment: {database_result.evidence_assessment}")
        print(f"Reasoning:")
        for i, step in enumerate(database_result.reasoning_steps, 1):
            print(f"  {i}. {step}")
        
        print(f"\nWEB EVIDENCE ANALYSIS:")
        print("-" * 30)
        print(f"Verdict: {web_result.verdict} ({web_result.confidence:.0%})")
        print(f"Assessment: {web_result.evidence_assessment}")
        print(f"Reasoning:")
        for i, step in enumerate(web_result.reasoning_steps, 1):
            print(f"  {i}. {step}")
        
        print(f"\nCOMBINED EVIDENCE ANALYSIS:")
        print("-" * 30)
        print(f"Verdict: {combined_result.verdict} ({combined_result.confidence:.0%})")
        print(f"Assessment: {combined_result.evidence_assessment}")
        print(f"Reasoning:")
        for i, step in enumerate(combined_result.reasoning_steps, 1):
            print(f"  {i}. {step}")
        
        # COMPARATIVE ANALYSIS
        print(f"\n" + "="*60)
        print("EVIDENCE SOURCE COMPARISON")
        print("="*60)
        
        print("DATABASE vs WEB EVIDENCE QUALITY:")
        print(f"Database sources: {len(db_formatted)} (Authority: Government records)")
        print(f"Web sources: {len(web_evidence)} (Authority: Current reporting)")
        
        print(f"\nVERDICT COMPARISON:")
        print(f"Database-only verdict: {database_result.verdict} ({database_result.confidence:.0%})")
        print(f"Web-only verdict: {web_result.verdict} ({web_result.confidence:.0%})")
        print(f"Combined verdict: {combined_result.verdict} ({combined_result.confidence:.0%})")
        
        # Agreement analysis
        if database_result.verdict == web_result.verdict:
            print(f"VERDICT AGREEMENT: Both evidence types support same conclusion")
        else:
            print(f"VERDICT DISAGREEMENT: Database={database_result.verdict}, Web={web_result.verdict}")
        
        # Confidence analysis
        db_conf = database_result.confidence
        web_conf = web_result.confidence
        combined_conf = combined_result.confidence
        
        print(f"\nCONFIDENCE ANALYSIS:")
        print(f"Database confidence: {db_conf:.0%}")
        print(f"Web confidence: {web_conf:.0%}")
        print(f"Combined confidence: {combined_conf:.0%}")
        
        if combined_conf > max(db_conf, web_conf):
            print("Combined evidence increased confidence")
        elif combined_conf < min(db_conf, web_conf):
            print("Conflicting evidence decreased confidence")
        else:
            print("Combined evidence maintained confidence level")
        
    except Exception as e:
        print(f"Gemini analysis failed: {e}")
    
    # GEMINI BIAS EXPLANATION
    print(f"\n" + "="*60)
    print("GEMINI AI BIAS ANALYSIS & EXPLANATIONS")
    print("="*60)
    
    try:
        print("Analyzing bias patterns with Gemini AI...")
        bias_analysis_result = gemini_verifier.explain_bias_patterns(claim, all_evidence, bias_analyses)
        
        print(f"\n" + "="*60)
        print("BIAS ANALYSIS RESULTS")
        print("="*60)
        
        # Display Gemini's bias summary
        print(gemini_verifier.get_bias_summary(bias_analysis_result))
        
        print(f"\n" + "-" * 40)
        print("DETAILED SOURCE-BY-SOURCE BIAS ANALYSIS:")
        print("-" * 40)
        
        # Show detailed per-source analysis
        for source_id, analysis in bias_analysis_result.detailed_analysis.items():
            if 'error' in analysis:
                continue
                
            print(f"\n{source_id.upper()}: {analysis.get('source_name', 'Unknown Source')}")
            print(f"  Source Type: {analysis.get('evidence_type', 'unknown')}")
            print(f"  Bias Score: {analysis.get('bias_score', 0):.2f}/1.0")
            
            # Emotional techniques
            emotional_techniques = analysis.get('emotional_techniques', [])
            if emotional_techniques:
                print(f"  Emotional Techniques: {', '.join(emotional_techniques)}")
            
            # Framing techniques
            framing_techniques = analysis.get('framing_techniques', [])
            if framing_techniques:
                print(f"  Framing Techniques: {', '.join(framing_techniques)}")
            
            # Specific bias examples with Gemini's explanations
            bias_examples = analysis.get('bias_examples', [])
            if bias_examples:
                print(f"  Specific Bias Examples:")
                for example in bias_examples[:2]:  # Show top 2 examples
                    quote = example.get('quote', '')
                    technique = example.get('technique', '')
                    explanation = example.get('explanation', '')
                    better_alternative = example.get('better_alternative', '')
                    
                    print(f"    Quote: \"{quote}\"")
                    print(f"    Technique: {technique}")
                    print(f"    Why biased: {explanation}")
                    if better_alternative:
                        print(f"    Better phrasing: \"{better_alternative}\"")
                    print()
            
            # Appropriateness assessment
            appropriateness = analysis.get('appropriateness_assessment', '')
            if appropriateness:
                print(f"  Context Assessment: {appropriateness}")
        
        # Show comparative examples
        if bias_analysis_result.specific_examples:
            print(f"\n" + "-" * 40)
            print("COMPARATIVE FRAMING EXAMPLES:")
            print("-" * 40)
            
            for example in bias_analysis_result.specific_examples:
                comparison_type = example.get('comparison_type', 'Comparison')
                source_a = example.get('source_a', 'Source A')
                source_b = example.get('source_b', 'Source B')
                bias_difference = example.get('bias_difference', 'No explanation')
                
                print(f"\n{comparison_type}:")
                print(f"  {source_a}")
                print(f"  {source_b}")
                print(f"  Analysis: {bias_difference}")
        
        print(f"\n" + "-" * 40)
        print("USER RECOMMENDATIONS:")
        print("-" * 40)
        
        for i, rec in enumerate(bias_analysis_result.recommendations, 1):
            print(f"{i}. {rec}")
        
    except Exception as e:
        print(f"Gemini bias analysis failed: {e}")
        print("Falling back to basic bias detection...")
        
        # Fallback to our existing cross-source analysis
        try:
            cross_source_framing = bias_detector.analyze_cross_source_framing(claim, all_evidence, bias_analyses)
            
            print(f"Cross-source analysis completed:")
            print(f"  Emotional baseline: {cross_source_framing['emotional_baseline']:.3f}")
            print(f"  Emotional outliers: {len(cross_source_framing['emotional_outliers'])}")
            print(f"  Entities with different framing: {len(cross_source_framing['semantic_differences'])}")
            print(f"  Framing consistency: {cross_source_framing['cross_source_summary']['framing_consistency']}")
            
        except Exception as fallback_e:
            print(f"Fallback bias analysis also failed: {fallback_e}")
    
    # TRADITIONAL ML-BASED BIAS ANALYSIS (For Comparison)
    print(f"\n" + "="*60)
    print("TRADITIONAL ML-BASED BIAS ANALYSIS (For Comparison)")
    print("="*60)

    try:
        # Run enhanced cross-source analysis with detailed evidence
        enhanced_cross_source = bias_detector.analyze_cross_source_framing_with_evidence(
            claim, all_evidence, bias_analyses
        )
        
        detailed_outliers = enhanced_cross_source.get('detailed_outlier_analysis', {})
        
        if detailed_outliers:
            print(f"\nDETAILED EMOTIONAL OUTLIER ANALYSIS:")
            print("-" * 40)
            
            for source_id, analysis in detailed_outliers.items():
                source_idx = int(source_id.split('_')[1]) + 1
                source_name = analysis['source_name']
                outlier_type = analysis['outlier_info']['type']
                emotional_score = analysis['outlier_info']['score']
                
                print(f"\nSource {source_idx}: {source_name}")
                print(f"  Type: {outlier_type} (emotional score: {emotional_score:.3f})")
                print(f"  Evidence type: {analysis['evidence_type']}")
                
                # Show bias explanation
                bias_explanation = analysis['bias_explanation']
                print(f"  Bias Analysis: {bias_explanation}")
                
                # Show specific examples
                emotional_evidence = analysis['emotional_evidence']
                emotional_words = emotional_evidence.get('emotional_words', [])
                
                if emotional_words:
                    print(f"  Emotional Language Examples:")
                    for word_info in emotional_words[:3]:  # Show top 3 examples
                        if 'sentence' in word_info:
                            sentence = word_info['sentence']
                            intensity = word_info.get('intensity', 0)
                            context = sentence[:100] + "..." if len(sentence) > 100 else sentence
                            print(f"    - Sentence (intensity: {intensity:.1f}): \"{context}\"")
                        elif 'word' in word_info:  # Fallback for old format
                            word = word_info['word']
                            context = word_info.get('context', '')[:100] + "..." if len(word_info.get('context', '')) > 100 else word_info.get('context', '')
                            intensity = word_info.get('intensity', 0)
                            print(f"    - '{word}' (intensity: {intensity:.1f}): \"{context}\"")
                
                loaded_phrases = emotional_evidence.get('loaded_phrases', [])
                if loaded_phrases:
                    print(f"  Loaded Phrases:")
                    for phrase_info in loaded_phrases[:2]:  # Show top 2 examples
                        phrase = phrase_info['phrase']
                        context = phrase_info['context'][:100] + "..." if len(phrase_info['context']) > 100 else phrase_info['context']
                        print(f"    - '{phrase}': \"{context}\"")
                
                # Show manipulation techniques
                manipulation = analysis['manipulation_analysis']
                techniques = manipulation.get('techniques_detected', {})
                active_techniques = [k for k, v in techniques.items() if v]
                
                if active_techniques:
                    print(f"  Manipulation Techniques: {', '.join(active_techniques)}")
                    manipulation_score = manipulation.get('manipulation_score', 0)
                    print(f"  Manipulation Score: {manipulation_score:.2f}/1.0")
        
        # Cross-source comparison insights
        print(f"\n" + "-" * 40)
        print("CROSS-SOURCE BIAS PATTERNS:")
        print("-" * 40)
        
        baseline = enhanced_cross_source.get('emotional_baseline', 0)
        print(f"Emotional baseline across all sources: {baseline:.3f}")
        
        # Compare database vs web framing
        patterns = enhanced_cross_source.get('systematic_patterns', {})
        db_vs_web = patterns.get('database_vs_web', {})
        
        if db_vs_web:
            db_avg = db_vs_web.get('database_emotional_avg', 0)
            web_avg = db_vs_web.get('web_emotional_avg', 0)
            
            print(f"\nFraming Pattern Analysis:")
            print(f"  Database sources (Government): {db_avg:.3f} average emotional intensity")
            print(f"  Web sources (Independent): {web_avg:.3f} average emotional intensity")
            
            if abs(db_avg - web_avg) > 0.2:
                if db_avg > web_avg:
                    print(f"  Pattern: Government sources use more emotional language")
                    print(f"  Implication: Official sources may be using emotional appeals to shape public opinion")
                else:
                    print(f"  Pattern: Independent sources use more emotional language")
                    print(f"  Implication: Independent media may be sensationalizing for engagement/clicks")
            else:
                print(f"  Pattern: Similar emotional intensity between source types")
                print(f"  Implication: Consistent framing approach across different source types")
        
        # Semantic differences analysis
        semantic_diffs = enhanced_cross_source.get('semantic_differences', {})
        if semantic_diffs:
            print(f"\nSemantic Framing Differences:")
            for entity, comparisons in list(semantic_diffs.items())[:3]:  # Show top 3 entities
                print(f"  Entity: '{entity}'")
                for source_id, comparison in comparisons.items():
                    source_idx = int(source_id.split('_')[1]) + 1
                    intensity = comparison.get('intensity_level', 'unknown')
                    descriptions = comparison.get('descriptions', [])
                    if descriptions:
                        example = descriptions[0][:80] + "..." if len(descriptions[0]) > 80 else descriptions[0]
                        print(f"    Source {source_idx}: {intensity} intensity - \"{example}\"")
        
        # Overall framing assessment
        total_outliers = len(detailed_outliers)
        total_sources = len(all_evidence)
        outlier_percentage = (total_outliers / total_sources) * 100 if total_sources > 0 else 0
        
        print(f"\n" + "-" * 40)
        print("OVERALL FRAMING ASSESSMENT:")
        print("-" * 40)
        print(f"Total sources analyzed: {total_sources}")
        print(f"Emotional outliers detected: {total_outliers} ({outlier_percentage:.1f}%)")
        
        if outlier_percentage > 40:
            print(f"Assessment: HIGH bias concern - Many sources using emotional manipulation")
        elif outlier_percentage > 20:
            print(f"Assessment: MODERATE bias concern - Some sources showing emotional bias")
        else:
            print(f"Assessment: LOW bias concern - Most sources using appropriate language")
        
        consistency = enhanced_cross_source.get('cross_source_summary', {}).get('framing_consistency', 'unknown')
        print(f"Framing consistency: {consistency}")
        
        if consistency == 'low':
            print(f"Recommendation: Exercise caution - significant disagreement in how sources frame the claim")
        else:
            print(f"Recommendation: Sources generally consistent in framing approach")

    except Exception as e:
        print(f"Detailed bias analysis failed: {e}")


    
    # STANDARD PIPELINE COMPARISON
    print(f"\n" + "="*60)
    print("STANDARD PIPELINE COMPARISON")
    print("="*60)
    
    # Run standard weighting
    weighted_evidence = weighter.weight_all_evidence(
        claim, all_evidence, verification_results, bias_analyses
    )
    
    standard_verdict = verdict_generator.generate_verdict(weighted_evidence)
    
    print("STANDARD ALGORITHM RESULTS:")
    print(f"Verdict: {standard_verdict['verdict']} ({standard_verdict['confidence']:.0%})")
    print(f"Support: {standard_verdict['support_score']:.1%} | Refute: {standard_verdict['refute_score']:.1%} | Neutral: {standard_verdict['neutral_score']:.1%}")
    
    if 'combined_result' in locals():
        print(f"\nSTANDARD vs GEMINI:")
        print(f"Standard: {standard_verdict['verdict']} ({standard_verdict['confidence']:.0%})")
        print(f"Gemini: {combined_result.verdict} ({combined_result.confidence:.0%})")
        
        if standard_verdict['verdict'].upper() == combined_result.verdict.upper():
            print("Algorithm and AI agree on verdict")
        else:
            print("Algorithm and AI disagree on verdict")
    
    print(f"\n" + "="*60)
    print("SYSTEM SUMMARY")
    print("="*60)
    
    print(f"Dual Retrieval Architecture:")
    print(f"  Database sources: {len(db_formatted)} (Government documents)")
    print(f"  Web sources: {len(web_evidence)} (Current reporting)")
    print(f"  Total evidence: {len(all_evidence)}")
    
    print(f"\nAnalysis Methods:")
    print(f"  Standard NLI: Verification + Bias + Weighting")
    print(f"  Gemini AI: Contextual reasoning + Evidence synthesis")
    print(f"  Comparison: Algorithm vs AI verdict agreement")
    
    print(f"\nEvidence Quality Assessment:")
    if len(db_formatted) > 0 and len(web_evidence) > 0:
        print(f"Dual evidence sources available")
        print(f"Government + Web perspectives captured")
        print(f"Comprehensive fact-checking possible")
    elif len(db_formatted) > 0:
        print(f"Only government database sources found")
        print(f"May lack current/independent perspectives")
    elif len(web_evidence) > 0:
        print(f"Only web sources found")  
        print(f"May lack official government records")
    else:
        print(f"No evidence found in either source")
    
    print(f"\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()