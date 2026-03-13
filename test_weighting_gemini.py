import sys
sys.path.append('src')
from config import Config
from verification.gemini_verification import GeminiClaimVerifier

def main():
    """Test Pipeline: Bias-aware evidence weighting + Gemini explanation"""
    
    print("="*60)
    print("BIAS-AWARE PIPELINE + GEMINI EXPLAINER - TEST")
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
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.verification import ClaimVerifier
    from src.bias_detection import BiasDetector
    from src.evidence_weighting import EvidenceWeighter, VerdictGenerator
    from src.synthesis import SynthesisEngine
    
    # Initialize both retrievers
    hybrid_retriever = HybridRetriever()  # 25k+ documents
    web_retriever = AdvancedEvidenceRetriever(
        serper_key=Config.SERPER_KEY, 
        groq_key=Config.GROQ_API_KEY
    )
    print("Evidence Retrievers initialized")
    
    verifier = ClaimVerifier()
    bias_detector = BiasDetector()
    weighter = EvidenceWeighter()
    verdict_generator = VerdictGenerator()
    synthesizer = SynthesisEngine(Config.GROQ_API_KEY)
    
    # Initialize Gemini explainer
    try:
        gemini_explainer = BiasAnalysisExplainer()
        print("Gemini Explainer initialized")
    except Exception as e:
        print(f"Gemini explainer initialization failed: {e}")
        exit(1)
    
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
    
    # Step 4: Evidence Retrieval
    print("\n" + "="*50)
    print("EVIDENCE RETRIEVAL FROM 25K+ DOCUMENTS")
    print("="*50)
    
    try:
        # Search 25k+ document collection first
        print("Searching 25k+ document collection...")
        dataset_evidence = hybrid_retriever.hybrid_search(
            query=claim,
            claim_types=None,  # Auto-detect
            total_results=8,
            use_query_expansion=True  # Enable query variations
        )
        
        # Convert to compatible format
        parliamentary_evidence = []
        for result in dataset_evidence:
            formatted_result = {
                'source': result['source'],
                'title': result['title'],
                'snippet': result.get('passage', result['text']),  #Use passage if available
                'link': result['url'],
                'dataset_source': result['dataset'],
                'authority': result['authority'],
                'relevance_score': result['similarity_score'],
                'passage_extracted': result.get('passage_extracted', False)  # Track if extracted
            }
            parliamentary_evidence.append(formatted_result)

        # Count high-quality database results
        # Count results by authority and relevance
        official_sources = [e for e in parliamentary_evidence 
                        if e.get('authority', 0) >= 0.95 and e.get('relevance_score', 0) > 0.5]

        news_sources = [e for e in parliamentary_evidence 
                    if e.get('authority', 0) >= 0.6 and e.get('relevance_score', 0) > 0.4]

        # Use database-only if we have enough quality sources
        if len(official_sources) >= 3 or len(news_sources) >= 5:
            print(f"Using database results: {len(official_sources)} official + {len(news_sources)} news sources")
            evidence = parliamentary_evidence
        else:
            print(f"Insufficient database results, adding web search...")
            print("Performing supplemental web search...")
            
            web_evidence = web_retriever.retrieve_hybrid_serper_decomposition(
                claim, 
                num_results=10,
                results_per_query=8
            )
            
            print(f"Web search: {len(web_evidence)} additional sources")
            evidence = parliamentary_evidence + web_evidence
            evidence = web_retriever._advanced_deduplication(evidence)[:15]
    
    except Exception as e:
        print(f"Database search failed: {e}")
        print("Falling back to web-only search...")
        
        evidence = web_retriever.retrieve_hybrid_serper_decomposition(
            claim, 
            num_results=15,
            results_per_query=10
        )

    print(f"Total Sources Retrieved: {len(evidence)}")

    if not evidence:
        print("No evidence found")
        exit(1)

    # Step 5: Standard Verification and Bias Analysis
    verification_results = []
    bias_analyses = []
    
    print("\n" + "="*60)
    print("VERIFICATION & BIAS ANALYSIS")
    print("="*60)
    
    for i, e in enumerate(evidence, 1):
        print(f"\nSource {i}: {e['source']}")
        print(f"   Title: {e['title'][:60]}...")
        
        # Add dataset info if available
        if e.get('dataset_source'):
            print(f"   Dataset: {e['dataset_source']}")
        
        # Standard verification
        verification_result = verifier.verify_claim(claim, e["snippet"])
        verification_results.append(verification_result)
        
        print(f"   Verdict: {verification_result['label']} ({verification_result['confidence']:.1%})")

        # Bias analysis
        bias_analysis = bias_detector.analyze_source(e["snippet"], e["link"])
        bias_analyses.append(bias_analysis)

        if "source_profile" in bias_analysis and bias_analysis["source_profile"]["has_profile"]:
            profile = bias_analysis["source_profile"]
            print(f"   Bias Profile: {profile['bias_interpretation']} ({profile['bias_score']:+.1f})")
        else:
            print(f"   Bias Profile: No profile available")

    # Step 6: Evidence Weighting (Original Algorithm)
    print("\n" + "="*60)
    print("EVIDENCE WEIGHTING")
    print("="*60)

    weighted_evidence = weighter.weight_all_evidence(
        claim, evidence, verification_results, bias_analyses
    )

    print("Weighting Results:")
    for i, item in enumerate(weighted_evidence, 1):
        evidence_url = item["evidence"].get("link", "")
        domain = weighter._extract_domain(evidence_url) if evidence_url else "Unknown"
        dataset_info = f" [{item['evidence'].get('dataset_source', 'web')}]"
        
        print(f"   Source {i} ({domain}{dataset_info}): Weight = {item['weight']:.3f}")

    # Step 7: Generate Final Verdict (Original Algorithm)
    final_verdict = verdict_generator.generate_verdict(weighted_evidence)

    print(f"\nFINAL VERDICT: {final_verdict['verdict']} ({final_verdict['confidence']:.0%} confidence)")
    print(f"Support: {final_verdict['support_score']:.1%} | Refute: {final_verdict['refute_score']:.1%} | Neutral: {final_verdict['neutral_score']:.1%}")

    # Step 8: GEMINI EXPLANATION (NEW)
    print("\n" + "="*60)
    print("GEMINI AI EXPLANATION OF BIAS ANALYSIS")
    print("="*60)
    
    try:
        print("Generating comprehensive bias explanation...")
        
        explanation = gemini_explainer.explain_bias_analysis(
            claim=claim,
            evidence_list=evidence,
            verification_results=verification_results,
            bias_analyses=bias_analyses,
            weighted_evidence=weighted_evidence,
            final_verdict=final_verdict
        )
        
        print("GEMINI BIAS EXPLANATION:")
        print("-" * 40)
        print(explanation)
        
    except Exception as e:
        print(f"Gemini explanation failed: {e}")

    # Step 9: Standard AI Synthesis for Comparison
    ai_synthesis = synthesizer.generate_synthesis(
        claim, evidence, verification_results, bias_analyses, final_verdict
    )

    print("\n" + "="*60)
    print("COMPARISON: STANDARD vs GEMINI EXPLANATION")
    print("="*60)
    
    print("STANDARD SYNTHESIS:")
    print("-" * 20)
    print(f"{ai_synthesis}")
    
    print(f"\nSTANDARD APPROACH: Algorithmic weighting + template-based explanation")
    print(f"GEMINI APPROACH: AI-powered contextual explanation of bias factors")

    print("\n" + "="*60)
    print("SYSTEM SUMMARY")
    print("="*60)

    database_sources = len([e for e in evidence if e.get('dataset_source')])
    web_sources = len(evidence) - database_sources
    sources_with_profiles = sum(1 for item in weighted_evidence 
                               if weighter._extract_domain(item["evidence"].get("link", "")) in weighter.bias_profiles)
    
    print(f"Evidence Sources: {len(evidence)} total ({database_sources} database, {web_sources} web)")
    print(f"Bias Profiles Available: {sources_with_profiles}/{len(evidence)} sources")
    print(f"Final Verdict: {final_verdict['verdict']} ({final_verdict['confidence']:.0%})")
    
    # Show dataset breakdown if any database sources found
    if database_sources > 0:
        dataset_breakdown = {}
        for e in evidence:
            if e.get('dataset_source'):
                dataset_breakdown[e['dataset_source']] = dataset_breakdown.get(e['dataset_source'], 0) + 1
        
        print("Dataset Source Breakdown:")
        for dataset, count in dataset_breakdown.items():
            print(f"   {dataset}: {count} sources")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


class BiasAnalysisExplainer:
    """Gemini-powered explainer for bias analysis and evidence weighting"""
    
    def __init__(self):
        try:
            self.gemini_verifier = GeminiClaimVerifier()
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini: {e}")
    
    def explain_bias_analysis(self, claim, evidence_list, verification_results, 
                            bias_analyses, weighted_evidence, final_verdict):
        """
        Generate comprehensive explanation of bias analysis and evidence weighting
        """
        
        # Build comprehensive explanation prompt
        explanation_prompt = self._build_explanation_prompt(
            claim, evidence_list, verification_results, bias_analyses, 
            weighted_evidence, final_verdict
        )
        
        try:
            # Use Gemini to generate explanation
            response = self.gemini_verifier.client.models.generate_content(
                model=self.gemini_verifier.model_name,
                contents=explanation_prompt
            )
            
            return response.text
            
        except Exception as e:
            return f"Error generating explanation: {e}"
    
    def _build_explanation_prompt(self, claim, evidence_list, verification_results,
                                bias_analyses, weighted_evidence, final_verdict):
        """Build detailed prompt for bias explanation"""
        
        # Organize information for explanation
        source_details = []
        
        for i, (evidence, verification, bias, weighted) in enumerate(
            zip(evidence_list, verification_results, bias_analyses, weighted_evidence), 1
        ):
            
            source_info = {
                'number': i,
                'domain': evidence.get('source', 'unknown'),
                'title': evidence.get('title', 'No title')[:100],
                'verification': verification['label'],
                'confidence': verification['confidence'],
                'final_weight': weighted['weight'],
                'bias_alignment': weighted['bias_alignment'],
                'explanation': weighted['explanation'],
                'dataset_source': evidence.get('dataset_source', 'web')
            }
            
            # Add bias profile information
            if bias['source_profile']['has_profile']:
                profile = bias['source_profile']
                source_info.update({
                    'has_bias_profile': True,
                    'bias_score': profile['bias_score'],
                    'bias_interpretation': profile['bias_interpretation'],
                    'profile_confidence': profile['confidence']
                })
            else:
                source_info['has_bias_profile'] = False
            
            # Add emotional analysis
            source_info.update({
                'emotion': bias['emotional_tone']['emotion'],
                'emotion_score': bias['emotional_tone']['emotion_score'],
                'framing_type': bias['framing_bias']['framing_type'],
                'framing_score': bias['framing_bias']['framing_score']
            })
            
            source_details.append(source_info)
        
        prompt = f"""
You are an expert in media literacy and bias analysis. Your task is to explain to a general audience how bias detection and evidence weighting works in fact-checking.

CLAIM BEING FACT-CHECKED:
"{claim}"

FINAL SYSTEM VERDICT:
{final_verdict['verdict']} ({final_verdict['confidence']:.0%} confidence)
- Support Score: {final_verdict['support_score']:.1%}
- Refute Score: {final_verdict['refute_score']:.1%}  
- Neutral Score: {final_verdict['neutral_score']:.1%}

DETAILED SOURCE ANALYSIS:
{self._format_source_details(source_details)}

EXPLANATION TASK:
Provide a comprehensive, accessible explanation that covers:

1. BIAS DETECTION OVERVIEW
   - Explain what bias detection means in fact-checking
   - Why it matters for evaluating sources
   - How different types of bias affect credibility

2. SOURCE-BY-SOURCE ANALYSIS
   - For each source, explain:
     * What bias indicators were found (or not found)
     * How the bias profile affects the source's credibility on this claim
     * Why the source received its specific weight
     * Any red flags or strengths in the source
     * Whether it's from a government dataset or web source

3. WEIGHTING METHODOLOGY EXPLANATION  
   - Explain how verification confidence combines with bias assessment
   - Why sources with known bias patterns get adjusted weights
   - How authority levels (official vs news vs web) factor in
   - Why bias alignment matters (pro-government sources on government claims)
   - How government dataset sources vs web sources are weighted

4. OVERALL ASSESSMENT
   - Synthesize the bias landscape across all sources
   - Explain how bias patterns influenced the final verdict
   - Identify any concerning bias trends (e.g., only pro-government sources support)
   - Assess the mix of government dataset vs web sources
   - Suggest what additional sources might improve the analysis

5. USER GUIDANCE
   - Help readers understand what the bias analysis reveals about information quality
   - Provide practical tips for evaluating these types of sources independently
   - Explain limitations and uncertainties in the bias assessment
   - Discuss the value of government dataset sources vs web sources

Write in clear, accessible language suitable for an educated general audience. Use specific examples from the sources analyzed. Focus on transparency about how bias detection influences fact-checking conclusions.
"""
        
        return prompt
    
    def _format_source_details(self, source_details):
        """Format source details for the prompt"""
        
        formatted = []
        
        for source in source_details:
            dataset_info = f" (From {source['dataset_source']} dataset)" if source['dataset_source'] != 'web' else " (Web source)"
            
            section = f"""
SOURCE {source['number']}: {source['domain']}{dataset_info}
Title: {source['title']}
Verification Result: {source['verification']} ({source['confidence']:.1%} confidence)
Final Weight: {source['final_weight']:.3f}
Bias Alignment Score: {source['bias_alignment']:.3f}
System Explanation: {source['explanation']}

Bias Analysis:
"""
            
            if source['has_bias_profile']:
                section += f"""- Known Bias Profile: {source['bias_interpretation']} (score: {source['bias_score']:+.1f})
- Profile Confidence: {source['profile_confidence']:.0%}
"""
            else:
                section += "- No historical bias profile available\n"
            
            section += f"""- Emotional Tone: {source['emotion']} ({source['emotion_score']:.2f})
- Framing Style: {source['framing_type']} ({source['framing_score']:.2f})
"""
            
            formatted.append(section)
        
        return "\n".join(formatted)


if __name__ == "__main__":
    main()