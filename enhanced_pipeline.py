"""Enhanced hybrid pipeline integrating datasets with existing bias-aware system"""
import sys
sys.path.append('src')
sys.path.append('.')

from src.retrieval.hybrid_retriever import HybridRetriever
from src.evidence_retrieval import AdvancedEvidenceRetriever
from src.verification import ClaimVerifier
from src.bias_detection import BiasDetector
from src.evidence_weighting import EvidenceWeighter, VerdictGenerator
from src.synthesis import SynthesisEngine
from config import Config
from datetime import datetime
from typing import Dict, List, Any

class EnhancedHybridPipeline:
    """Enhanced pipeline combining datasets + existing bias-aware system"""
    
    def __init__(self):
        """Initialize enhanced hybrid pipeline"""
        print("Initializing Enhanced Hybrid Fact-Checking Pipeline...")
        
        # Initialize hybrid retriever for government datasets
        print("   Loading government dataset retriever...")
        self.hybrid_retriever = HybridRetriever()
        
        # Initialize existing components
        print("   Loading web evidence retriever...")
        self.web_retriever = AdvancedEvidenceRetriever(
            serper_key=Config.SERPER_KEY,
            groq_key=Config.GROQ_API_KEY
        )
        
        print("   Loading verification & bias detection...")
        self.verifier = ClaimVerifier()
        self.bias_detector = BiasDetector()
        self.weighter = EvidenceWeighter()
        self.verdict_generator = VerdictGenerator()
        self.synthesizer = SynthesisEngine(Config.GROQ_API_KEY)
        
        print("Enhanced Hybrid Pipeline Ready!")
        
        # Get system capabilities
        self.capabilities = self.hybrid_retriever.get_search_summary()
        print(f"   Local datasets: {self.capabilities['total_local_documents']:,} documents")
        print(f"   Available datasets: {len(self.capabilities['loaded_datasets'])}")
    

    def enhanced_fact_check(self, claim: str, max_sources: int = 15) -> Dict[str, Any]:
        """
        Enhanced fact-checking combining:
        1. Government dataset search (high authority)
        2. Web search (comprehensive coverage) 
        3. Existing bias-aware analysis
        """
        
        print(f"\nENHANCED HYBRID FACT-CHECK")
        print("=" * 80)
        print(f"Claim: '{claim}'")
        print("=" * 80)
        
        # PHASE 1: HYBRID EVIDENCE RETRIEVAL
        print("\nPHASE 1: HYBRID EVIDENCE RETRIEVAL")
        print("=" * 50)
        
        # Step 1A: Search Government Datasets First 
        print("Step 1A: Searching Government Datasets...")
        
        dataset_evidence = self._search_government_datasets(claim, max_sources // 2)
        
        print(f"   Found {len(dataset_evidence)} government sources")
        self._display_dataset_breakdown(dataset_evidence)
        
        # FIXED Step 1B: Proper Web Search (not dataset search)
        print("\nStep 1B: Searching Web Sources via Serper...")
        
        web_evidence = []
        try:
            # Use ACTUAL web search - not dataset search
            web_evidence = self.web_retriever.retrieve_hybrid_serper_decomposition(
                claim,
                num_results=max_sources // 2,
                results_per_query=8
            )
            print(f"   Found {len(web_evidence)} web sources")
            
        except Exception as e:
            print(f"   Web search failed: {e}")
            # Try simple serper as fallback
            try:
                web_evidence = self.web_retriever._simple_serper_query(claim, max_sources // 2)
                print(f"   Fallback web search: {len(web_evidence)} sources")
            except Exception as e2:
                print(f"   Fallback also failed: {e2}")
                web_evidence = []
        
        # Step 1C: Combine and Prioritize Evidence
        print("\nStep 1C: Combining Evidence Sources...")
        
        combined_evidence = self._combine_evidence_sources(
            dataset_evidence, 
            web_evidence, 
            max_sources
        )

        print(f"   Total evidence: {len(combined_evidence)} sources")
        print(f"   Government sources: {len(dataset_evidence)}")
        print(f"   Web sources: {len(web_evidence)}")
        
        # PHASE 2: ENHANCED VERIFICATION & BIAS ANALYSIS
        print(f"\nPHASE 2: VERIFICATION & BIAS ANALYSIS")
        print("=" * 50)
        
        verification_results = []
        bias_analyses = []
        
        for i, evidence_item in enumerate(combined_evidence, 1):
            print(f"\nSource {i}: {evidence_item.get('source', 'Unknown')}")
            
            # Show source type and authority
            source_type = "Government Dataset" if evidence_item.get('dataset') else "Web Source"
            authority = evidence_item.get('authority', 0.5)
            
            print(f"   Type: {source_type} (Authority: {authority:.2f})")
            
            if evidence_item.get('dataset'):
                print(f"   Dataset: {evidence_item['dataset']}")
                print(f"   Similarity: {evidence_item.get('similarity_score', 0):.3f}")
            
            if evidence_item.get('relevance_score'):
                print(f"   Relevance: {evidence_item['relevance_score']:.3f}")
            
            print(f"   Title: {evidence_item.get('title', 'No title')[:60]}...")
            print(f"   URL: {evidence_item.get('link', evidence_item.get('url', 'No URL'))}")
            
            # Verification using your existing system
            snippet = evidence_item.get('snippet', evidence_item.get('text', ''))
            verification_result = self.verifier.verify_claim(claim, snippet)
            verification_results.append(verification_result)
            
            print(f"   Verdict: {verification_result['label']}")
            print(f"   Confidence: {verification_result['confidence']:.2%}")
            
            # Bias analysis using your existing system
            url = evidence_item.get('link', evidence_item.get('url', ''))
            bias_analysis = self.bias_detector.analyze_source(snippet, url)
            bias_analyses.append(bias_analysis)
            
            # Show bias information
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
        
        # PHASE 3: ENHANCED EVIDENCE WEIGHTING
        print(f"\nPHASE 3: ENHANCED EVIDENCE WEIGHTING")
        print("=" * 50)
        
        # Use your existing evidence weighting with enhanced authority
        weighted_evidence = self.weighter.weight_all_evidence(
            claim, combined_evidence, verification_results, bias_analyses
        )
        
        # Display weighting results with dataset context
        print("ENHANCED WEIGHTING RESULTS:")
        
        for i, item in enumerate(weighted_evidence, 1):
            evidence_url = item["evidence"].get("link", item["evidence"].get("url", ""))
            domain = self.weighter._extract_domain(evidence_url) if evidence_url else "Unknown"
            
            # Enhanced display with dataset info
            source_type = ""
            if item["evidence"].get('dataset'):
                source_type = f"[{item['evidence']['dataset']}] "
            elif item["evidence"].get('dataset_source'):
                source_type = f"[{item['evidence']['dataset_source']}] "
            
            print(f"\n   Source {i} {source_type}({domain}):")
            print(f"      Final Weight: {item['weight']:.3f}")
            print(f"      Bias Alignment: {item['bias_alignment']:.3f}")
            
            # Show authority boost for government sources
            if item["evidence"].get('authority', 0) >= 1.0:
                print(f"      Authority Boost: HIGH (Government Source)")
            
            print(f"      Explanation: {item['explanation']}")
        
        # PHASE 4: ENHANCED VERDICT GENERATION
        print(f"\n🏁 PHASE 4: ENHANCED VERDICT GENERATION")
        print("=" * 50)
        
        # Generate verdict with enhanced context
        final_verdict = self.verdict_generator.generate_verdict(weighted_evidence)
        
        # Add dataset-specific context
        dataset_source_count = len([e for e in combined_evidence if e.get('dataset')])
        web_source_count = len(combined_evidence) - dataset_source_count
        
        enhanced_verdict = {
            **final_verdict,
            'dataset_sources': dataset_source_count,
            'web_sources': web_source_count,
            'hybrid_strategy': self._determine_hybrid_strategy(dataset_source_count, web_source_count)
        }
        
        # PHASE 5: AI SYNTHESIS WITH DATASET CONTEXT
        print(f"\n🤖 PHASE 5: AI SYNTHESIS")
        print("=" * 50)
        
        # Enhanced synthesis including dataset context
        ai_synthesis = self._generate_enhanced_synthesis(
            claim, combined_evidence, verification_results, 
            bias_analyses, enhanced_verdict
        )
        
        return {
            'claim': claim,
            'timestamp': datetime.now().isoformat(),
            'evidence': {
                'dataset_sources': dataset_evidence,
                'web_sources': web_evidence, 
                'combined': combined_evidence,
                'total_count': len(combined_evidence)
            },
            'analysis': {
                'verification_results': verification_results,
                'bias_analyses': bias_analyses,
                'weighted_evidence': weighted_evidence
            },
            'verdict': enhanced_verdict,
            'synthesis': ai_synthesis,
            'system_performance': {
                'hybrid_strategy': enhanced_verdict['hybrid_strategy'],
                'dataset_coverage': len(self.capabilities['loaded_datasets']),
                'total_local_documents': self.capabilities['total_local_documents']
            }
        }

    def _search_government_datasets(self, claim: str, max_results: int) -> List[Dict]:
        """Search government datasets using hybrid retriever"""
        
        try:
            # Use hybrid retriever to search local datasets
            results = self.hybrid_retriever.search_datasets(
                query=claim,
                claim_types=None,  # Search all available datasets
                top_k=max_results
            )
            
            # Convert to format compatible with existing system
            formatted_results = []
            for result in results:
                formatted_result = {
                    'source': result['source'],
                    'title': result['title'],
                    'snippet': result['text'],  # Map 'text' to 'snippet'
                    'text': result['text'],     # Keep both for compatibility
                    'link': result['url'],
                    'url': result['url'],       # Keep both for compatibility
                    'authority': result['authority'],
                    'similarity_score': result['similarity_score'],
                    'dataset': result['dataset'],
                    'retrieval_method': 'local_dataset',
                    'dataset_source': result['dataset'],  # For compatibility with existing code
                    'source_alignment': 'official_primary'  # Government sources are primary
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            print(f"   Error searching datasets: {e}")
            return []

    
    def _display_dataset_breakdown(self, dataset_evidence: List[Dict]):
        """Display breakdown of dataset sources"""
        
        if not dataset_evidence:
            print("   No government sources found")
            return
        
        # Count by dataset
        dataset_counts = {}
        for evidence in dataset_evidence:
            dataset = evidence.get('dataset', 'unknown')
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        print("   Government sources by dataset:")
        for dataset, count in sorted(dataset_counts.items()):
            print(f"     - {dataset}: {count} sources")
        
        # Show authority distribution
        high_authority = len([e for e in dataset_evidence if e.get('authority', 0) >= 1.0])
        print(f"   High authority sources: {high_authority}/{len(dataset_evidence)}")
        
        # Show relevance distribution
        relevance_scores = [e.get('similarity_score', 0) for e in dataset_evidence]
        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            print(f"   Average relevance: {avg_relevance:.3f}")

    def _combine_evidence_sources(self, dataset_evidence: List[Dict], 
                            web_evidence: List[Dict], 
                            max_total: int) -> List[Dict]:
        """Combine dataset and web evidence with proper prioritization - FIXED"""
        
        # Priority order: Dataset sources first (higher authority), then web
        combined = []
        
        # Add all dataset sources (they have priority)
        combined.extend(dataset_evidence)
        print(f"   Added {len(dataset_evidence)} government dataset sources")
        
        # Add web sources up to total limit
        remaining_slots = max_total - len(dataset_evidence)
        if remaining_slots > 0:
            combined.extend(web_evidence[:remaining_slots])
            print(f"   Added {min(len(web_evidence), remaining_slots)} web sources")
        
        print(f"   Pre-deduplication total: {len(combined)} sources")
        
        # FIXED: Use the improved deduplication
        if hasattr(self.web_retriever, '_advanced_deduplication'):
            deduplicated = self.web_retriever._advanced_deduplication(combined)
        else:
            # Fallback deduplication if method not available
            deduplicated = self._simple_deduplication(combined)
        
        print(f"   Post-deduplication: {len(deduplicated)} sources")
        
        return deduplicated[:max_total]

    def _simple_deduplication(self, evidence_list: List[Dict]) -> List[Dict]:
        """Simple fallback deduplication by URL only"""
        seen_urls = set()
        unique_evidence = []
        
        for evidence in evidence_list:
            url = evidence.get('link', evidence.get('url', ''))
            if url not in seen_urls:
                unique_evidence.append(evidence)
                seen_urls.add(url)
        
        return unique_evidence
    
    def _determine_hybrid_strategy(self, dataset_count: int, web_count: int) -> str:
        """Determine which hybrid strategy was used"""
        
        if dataset_count >= 5:
            return "DATASET-PRIMARY"
        elif dataset_count >= 2:
            return "HYBRID-BALANCED"
        elif dataset_count >= 1:
            return "WEB-PRIMARY-WITH-DATASETS"
        else:
            return "WEB-ONLY-FALLBACK"
    
    def _generate_enhanced_synthesis(self, claim: str, evidence: List[Dict],
                                   verification_results: List[Dict],
                                   bias_analyses: List[Dict],
                                   verdict: Dict) -> str:
        """Generate AI synthesis with dataset context"""
        
        try:
            # Use existing synthesis but add dataset context
            base_synthesis = self.synthesizer.generate_synthesis(
                claim, evidence, verification_results, bias_analyses, verdict
            )
            
            # Add dataset-specific context
            dataset_count = verdict.get('dataset_sources', 0)
            web_count = verdict.get('web_sources', 0)
            strategy = verdict.get('hybrid_strategy', 'UNKNOWN')
            
            enhanced_context = f"\n\nHYBRID RETRIEVAL ANALYSIS:\n"
            enhanced_context += f"Strategy Used: {strategy}\n"
            enhanced_context += f"Government Dataset Sources: {dataset_count}\n"
            enhanced_context += f"Web Sources: {web_count}\n"
            
            if dataset_count >= 3:
                enhanced_context += "This claim was verified primarily using official government sources, "
                enhanced_context += "providing high confidence in the analysis."
            elif dataset_count >= 1:
                enhanced_context += "This claim was analyzed using a combination of government sources "
                enhanced_context += "and web sources for comprehensive coverage."
            else:
                enhanced_context += "No official government sources were found for this claim. "
                enhanced_context += "Analysis is based on web sources only."
            
            return base_synthesis + enhanced_context
            
        except Exception as e:
            return f"Enhanced synthesis generation failed: {str(e)}"

def main():
    """Test enhanced hybrid pipeline"""
    
    try:
        # Initialize pipeline
        pipeline = EnhancedHybridPipeline()
        
        # Test claims
        test_claims = [
            # "The Sri Lankan government announced a new education policy framework",
            # "Cabinet decided to increase the education budget",
            # "President made statements about economic development in Sri Lanka"
            "The use of rice for animal feed is a eason for the heavy consumption of rice in sri lanka"
        ]
        
        print("\n🧪 TESTING ENHANCED HYBRID PIPELINE")
        print("=" * 80)
        
        for i, claim in enumerate(test_claims, 1):
            print(f"\n{'='*20} TEST {i}/{len(test_claims)} {'='*20}")
            
            try:
                result = pipeline.enhanced_fact_check(claim, max_sources=10)
                
                # Display results like your existing system
                print(f"\nFINAL RESULTS")
                print("=" * 60)
                print(f"AI SYNTHESIS:")
                print(result['synthesis'])
                print()
                
                verdict = result['verdict']
                print(f"FACT-CHECK VERDICT: {verdict['verdict']} ({verdict['confidence']:.0%} confidence)")
                print()
                print("Detailed Breakdown:")
                print(f"   Support Score: {verdict['support_score']:.1%}")
                print(f"   Refute Score: {verdict['refute_score']:.1%}")
                print(f"   Neutral Score: {verdict['neutral_score']:.1%}")
                
                print()
                print("Enhanced System Performance:")
                perf = result['system_performance']
                print(f"   Hybrid Strategy: {perf['hybrid_strategy']}")
                print(f"   Dataset Coverage: {perf['dataset_coverage']} datasets loaded")
                print(f"   Local Documents: {perf['total_local_documents']:,}")
                
                evidence_summary = result['evidence']
                print(f"   Dataset Sources: {len(evidence_summary['dataset_sources'])}")
                print(f"   Web Sources: {len(evidence_summary['web_sources'])}")
                
            except Exception as e:
                print(f"Test {i} failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🎉 Enhanced hybrid pipeline testing complete!")
        
    except Exception as e:
        print(f"Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()