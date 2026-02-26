#!/usr/bin/env python3
"""
Enhanced Test Pipeline for Bias-Aware Fact-Checking System
Now with Dataset-First Evidence Retrieval using PMD datasets!
"""

from config import Config

if __name__ == "__main__":
    print("="*70)
    print("🚀 ENHANCED BIAS-AWARE FACT-CHECKING SYSTEM")
    print("="*70)
    print("🗂️  NEW: PMD Presidential datasets for 'President said X' claims")
    print("🔍 NEW: Cabinet & Supreme Court datasets for government claims")
    print("🧠 NEW: Semantic search with claim routing")
    print("🌐 IMPROVED: Serper API as fallback only")
    
    # Step 1: Validate configuration
    print("\\n1. Validating configuration...")
    try:
        Config.validate()
        print("   ✅ Configuration valid")
    except ValueError as e:
        print(f"   ❌ Configuration error: {e}")
        exit(1)
    
    # Step 2: Initialize components
    print("\\n2. Initializing enhanced components...")
    
    from src.evidence_retrieval import AdvancedEvidenceRetriever
    from src.verification import ClaimVerifier
    from src.bias_detection import BiasDetector
    from src.evidence_weighting import EvidenceWeighter, VerdictGenerator
    
    retriever = AdvancedEvidenceRetriever(
        serper_key=Config.SERPER_KEY, 
        groq_key=Config.GROQ_API_KEY
    )
    print("   ✅ Enhanced Evidence Retriever initialized")
    
    verifier = ClaimVerifier()
    bias_detector = BiasDetector()
    weighter = EvidenceWeighter()
    verdict_generator = VerdictGenerator()
    
    print("   ✅ All components initialized successfully")

    # Step 3: Test with user input claim
    print("\\n3. Enter claim for enhanced fact-checking...")

    print("\\n" + "="*70)
    print("🎯 ENTER CLAIM TO FACT-CHECK")
    print("="*70)
    print("✨ Try these examples to see the enhancement:")
    print("   • President said giving rice to animals causes shortage")
    print("   • Cabinet approved new economic recovery plan")
    print("   • Supreme Court ruled on constitutional amendment")
    print("   • Parliament debated education reform bill")
    print("\\nEnter your claim below:")
    
    claim = input(">> ").strip()

    if not claim:
        print("❌ No claim entered.")
        exit(1)

    print(f"\\n🔎 Analyzing claim: '{claim}'")
    
    # STEP 4: Enhanced Dataset-First Evidence Retrieval
    print("\\n" + "="*70)
    print("🚀 ENHANCED EVIDENCE RETRIEVAL (Dataset-First)")
    print("="*70)
    print("🗂️  Priority 1: PMD, Cabinet, Parliament & Court databases")
    print("🌐 Priority 2: Serper API fallback when needed")
    
    try:
        # Use the NEW enhanced dataset-first method
        print("\\n📊 Initiating enhanced search with claim routing...")
        
        evidence = retriever.retrieve_evidence_dataset_first(
            claim, 
            num_results=15  # Intelligently balances database vs web sources
        )
        
        print(f"\\n✅ EVIDENCE COLLECTION COMPLETE")
        print(f"   📋 Total Sources Retrieved: {len(evidence)}")
        
        # Analyze source distribution
        database_sources = sum(1 for e in evidence if e.get('search_method') != 'web_fallback')
        web_sources = len(evidence) - database_sources
        
        print(f"\\n📊 SOURCE DISTRIBUTION:")
        print(f"   🗂️  Authoritative Database Sources: {database_sources}")
        print(f"   🌐 Web Fallback Sources: {web_sources}")
        
        # Show detailed source breakdown by dataset
        if database_sources > 0:
            print(f"\\n🏛️  DATABASE SOURCES BREAKDOWN:")
            dataset_counts = {}
            for e in evidence:
                if e.get('search_method') != 'web_fallback':
                    dataset = e.get('dataset_source', 'unknown')
                    dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
            
            for dataset, count in sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True):
                dataset_display = dataset.replace('_', ' ').title()
                print(f"      📄 {dataset_display}: {count} sources")
        
        # Display top evidence with quality metrics
        if evidence:
            print(f"\\n🔍 TOP EVIDENCE QUALITY ANALYSIS:")
            print(f"{'#':<3} {'Source Type':<25} {'Authority':<10} {'Score':<8} {'Method'}")
            print("-" * 70)
            
            for i, e in enumerate(evidence[:5], 1):
                source_type = e.get('dataset_source', e.get('source', 'web'))
                search_method = e.get('search_method', 'keyword')
                authority = e.get('authority_weight', 0)
                score = e.get('relevance_score', e.get('final_score', 0))
                
                if source_type != 'web' and e.get('search_method') != 'web_fallback':
                    source_display = f"{source_type.replace('_', ' ').title()}"[:24]
                else:
                    source_display = "Web Search"
                
                print(f"{i:<3} {source_display:<25} {authority:<10.2f} {score:<8.3f} {search_method}")
            
            # Show content previews
            print(f"\\n📄 CONTENT PREVIEWS:")
            for i, e in enumerate(evidence[:3], 1):
                title = e.get('title', 'No title')[:80]
                snippet = e.get('snippet', '')
                
                print(f"\\n{i}. {title}...")
                if snippet:
                    # Clean snippet for better display
                    clean_snippet = snippet.replace('\\n', ' ').strip()
                    if len(clean_snippet) > 200:
                        clean_snippet = clean_snippet[:200] + "..."
                    print(f"   📝 {clean_snippet}")
                    
                source_info = e.get('dataset_source', 'web').replace('_', ' ').title()
                print(f"   📊 Source: {source_info} | Authority: {e.get('authority_weight', 0):.2f}")
                
    except Exception as e:
        print(f"💥 Enhanced evidence retrieval failed: {e}")
        print("🔄 Falling back to legacy search method...")
        import traceback
        traceback.print_exc()
        
        # Fallback to previous method
        try:
            evidence = retriever.retrieve_hybrid_serper_decomposition(
                claim, 
                num_results=15,
                results_per_query=8
            )
            print(f"   ✅ Legacy search completed: {len(evidence)} sources")
        except Exception as fallback_error:
            print(f"   💥 Legacy search also failed: {fallback_error}")
            evidence = []

    if not evidence:
        print("\\n❌ No evidence found. Cannot proceed with fact-checking.")
        exit(1)
    
    # STEP 5: Evidence Analysis and Verification
    print(f"\\n" + "="*70)
    print("🔬 EVIDENCE ANALYSIS & VERIFICATION")
    print("="*70)

    # Verify each piece of evidence
    print("\\n🔍 Verifying evidence against claim...")
    verification_results = []
    
    for i, evidence_piece in enumerate(evidence):
        print(f"\\n   Analyzing evidence {i+1}/{len(evidence)}...")
        
        try:
            verification = verifier.verify_evidence_against_claim(
                claim, 
                evidence_piece.get('snippet', evidence_piece.get('title', ''))
            )
            
            verification_results.append({
                'evidence': evidence_piece,
                'verification': verification
            })
            
            # Show verification result with enhanced info
            verdict = verification.get('verdict', 'UNKNOWN')
            confidence = verification.get('confidence', 0)
            source_type = evidence_piece.get('dataset_source', 'web').replace('_', ' ').title()
            
            if verdict == 'SUPPORTS':
                status = "✅ SUPPORTS"
            elif verdict == 'REFUTES':
                status = "❌ REFUTES"
            else:
                status = "❓ NEUTRAL"
            
            print(f"      {status} | Confidence: {confidence:.2f} | Source: {source_type}")
            
        except Exception as e:
            print(f"      ⚠️  Verification failed: {str(e)[:50]}...")
            verification_results.append({
                'evidence': evidence_piece,
                'verification': {'verdict': 'UNKNOWN', 'confidence': 0.0}
            })

    # STEP 6: Bias Detection
    print(f"\\n🎭 BIAS DETECTION ANALYSIS:")
    bias_results = []
    
    print("   Analyzing source bias and reliability...")
    for i, result in enumerate(verification_results):
        evidence_piece = result['evidence']
        try:
            bias_analysis = bias_detector.analyze_bias(
                text=evidence_piece.get('snippet', ''),
                source_url=evidence_piece.get('link', ''),
                title=evidence_piece.get('title', '')
            )
            bias_results.append(bias_analysis)
            
            bias_score = bias_analysis.get('overall_bias_score', 0)
            bias_direction = bias_analysis.get('bias_direction', 'neutral')
            reliability = bias_analysis.get('source_reliability', 'unknown')
            
            print(f"      Source {i+1}: {bias_direction.upper()} bias ({bias_score:.2f}) | Reliability: {reliability.upper()}")
            
        except Exception as e:
            print(f"      Source {i+1}: Bias analysis failed - {str(e)[:40]}...")
            bias_results.append({'overall_bias_score': 0, 'bias_direction': 'neutral', 'source_reliability': 'unknown'})

    # STEP 7: Evidence Weighting & Final Verdict
    print(f"\\n⚖️  EVIDENCE WEIGHTING & FINAL VERDICT:")
    
    try:
        # Calculate weighted scores
        weighted_evidence = weighter.weight_evidence_by_authority_and_bias(
            evidence, 
            bias_results, 
            verification_results
        )
        
        # Generate final verdict
        final_verdict = verdict_generator.generate_verdict(
            claim,
            weighted_evidence,
            verification_results,
            bias_results
        )
        
        # Display comprehensive results
        verdict = final_verdict.get('verdict', 'UNCERTAIN')
        confidence = final_verdict.get('confidence', 0)
        explanation = final_verdict.get('explanation', 'No explanation provided')
        
        print(f"\\n" + "="*70)
        print(f"🏆 FINAL FACT-CHECK RESULT")
        print(f"="*70)
        
        # Verdict with emoji
        if verdict == 'SUPPORTS':
            verdict_display = "✅ SUPPORTS"
            color = "\\033[92m"  # Green
        elif verdict == 'REFUTES':
            verdict_display = "❌ REFUTES"  
            color = "\\033[91m"  # Red
        else:
            verdict_display = "❓ UNCERTAIN"
            color = "\\033[93m"  # Yellow
        
        reset_color = "\\033[0m"
        
        print(f"\\n🎯 VERDICT: {color}{verdict_display}{reset_color}")
        print(f"📊 CONFIDENCE: {confidence:.1%}")
        print(f"\\n💭 EXPLANATION:")
        print(f"   {explanation}")
        
        # Evidence summary
        supports_count = sum(1 for r in verification_results if r['verification'].get('verdict') == 'SUPPORTS')
        refutes_count = sum(1 for r in verification_results if r['verification'].get('verdict') == 'REFUTES')
        neutral_count = len(verification_results) - supports_count - refutes_count
        
        print(f"\\n📈 EVIDENCE SUMMARY:")
        print(f"   ✅ Supporting: {supports_count} sources")
        print(f"   ❌ Refuting: {refutes_count} sources") 
        print(f"   ❓ Neutral/Uncertain: {neutral_count} sources")
        
        # Source quality summary
        database_quality = sum(1 for e in evidence if e.get('search_method') != 'web_fallback')
        print(f"\\n🏛️  SOURCE QUALITY:")
        print(f"   🗂️  Authoritative Database Sources: {database_quality}/{len(evidence)}")
        print(f"   🌐 Web Fallback Sources: {len(evidence) - database_quality}/{len(evidence)}")
        
        if database_quality > 0:
            print(f"\\n🎉 SUCCESS: Enhanced system used authoritative Sri Lankan government databases!")
            print(f"   📄 This includes PMD, Cabinet, Parliament, and Court records")
            print(f"   🎯 Much more reliable than generic web search results")
        else:
            print(f"\\n⚠️  NOTE: Only web sources available for this claim")
            print(f"   💡 Consider checking if claim relates to Sri Lankan government matters")
        
    except Exception as e:
        print(f"\\n💥 Final verdict generation failed: {e}")
        import traceback
        traceback.print_exc()
        
        print(f"\\n📋 RAW VERIFICATION SUMMARY:")
        for i, result in enumerate(verification_results, 1):
            verdict = result['verification'].get('verdict', 'UNKNOWN')
            conf = result['verification'].get('confidence', 0)
            print(f"   {i}. {verdict} (confidence: {conf:.2f})")

    print(f"\\n" + "="*70)
    print(f"🏁 FACT-CHECK COMPLETE!")
    print(f"="*70)
    print(f"\\n🚀 System Enhancement Notes:")
    print(f"   • Used {database_quality if 'database_quality' in locals() else 0} authoritative database sources")
    print(f"   • PMD datasets now handle 'President said X' claims effectively")
    print(f"   • Claim routing directs searches to appropriate government datasets")
    print(f"   • Much more accurate than previous web-only approach")
    print(f"\\n💡 For best results, try claims about:")
    print(f"   - Presidential statements (uses PMD datasets)")
    print(f"   - Cabinet decisions (uses Cabinet datasets)")
    print(f"   - Parliamentary debates (uses Hansard datasets)")  
    print(f"   - Court rulings (uses Supreme & Appeal Court datasets)")