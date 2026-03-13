"""
Baseline Comparison for Evidence Weighting Evaluation
===================================================

This module provides baseline comparison functionality to demonstrate
the improvement achieved by bias-aware weighting over traditional approaches.

For evaluation with FEVER dataset or other ground truth data.
"""
import sys
sys.path.append('src')
from config import Config
from typing import List, Dict, Tuple

class BaselineComparator:
    """Compare bias-aware weighting against baseline approaches"""
    
    def __init__(self):
        # Initialize components
        from src.evidence_retrieval import AdvancedEvidenceRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.verification import ClaimVerifier
        from src.bias_detection import BiasDetector
        from src.evidence_weighting import EvidenceWeighter, VerdictGenerator
        
        self.hybrid_retriever = HybridRetriever()
        self.web_retriever = AdvancedEvidenceRetriever(
            serper_key=Config.SERPER_KEY, 
            groq_key=Config.GROQ_API_KEY
        )
        
        self.verifier = ClaimVerifier()
        self.bias_detector = BiasDetector()
        self.weighter = EvidenceWeighter()
        self.verdict_generator = VerdictGenerator()
    
    def compare_weighting_approaches(self, claim: str) -> Dict:
        """
        Compare different weighting approaches on the same evidence
        
        Returns:
            Dictionary with results from each approach
        """
        
        print(f"Comparing weighting approaches for: {claim}")
        
        # Step 1: Get evidence (same for all approaches)
        all_evidence, verification_results, bias_analyses = self._get_evidence_and_analysis(claim)
        
        if not all_evidence:
            return {"error": "No evidence found"}
        
        # Step 2: Apply different weighting approaches
        results = {}
        
        # Approach 1: Uniform weighting (baseline)
        print("\nApproach 1: Uniform Weighting (Baseline)")
        uniform_verdict = self._uniform_weighting_approach(
            claim, all_evidence, verification_results
        )
        results['uniform'] = uniform_verdict
        
        # Approach 2: Simple authority weighting
        print("Approach 2: Authority-Only Weighting")
        authority_verdict = self._authority_weighting_approach(
            claim, all_evidence, verification_results, bias_analyses
        )
        results['authority'] = authority_verdict
        
        # Approach 3: Your bias-aware weighting (research contribution)
        print("Approach 3: BIAS-AWARE WEIGHTING (Your Algorithm)")
        bias_aware_verdict = self._bias_aware_weighting_approach(
            claim, all_evidence, verification_results, bias_analyses
        )
        results['bias_aware'] = bias_aware_verdict
        
        # Step 3: Compare results
        comparison = self._analyze_comparison(results)
        results['comparison'] = comparison
        
        return results
    
    def _get_evidence_and_analysis(self, claim: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Get evidence and run verification/bias analysis"""
        
        # Retrieve evidence
        try:
            database_evidence = self.hybrid_retriever.hybrid_search(
                query=claim, total_results=10
            )
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
        except:
            db_formatted = []
        
        try:
            web_evidence = self.web_retriever.retrieve_hybrid_serper_decomposition(
                claim, num_results=10
            )
            for e in web_evidence:
                e['evidence_type'] = 'web'
        except:
            web_evidence = []
        
        all_evidence = db_formatted + web_evidence
        
        if not all_evidence:
            return [], [], []
        
        # Run verification and bias analysis
        verification_results = []
        bias_analyses = []
        
        for evidence in all_evidence:
            verification_result = self.verifier.verify_claim(claim, evidence["snippet"])
            verification_results.append(verification_result)
            
            bias_analysis = self.bias_detector.analyze_source(evidence["snippet"], evidence["link"])
            bias_analyses.append(bias_analysis)
        
        return all_evidence, verification_results, bias_analyses
    
    def _uniform_weighting_approach(self, claim: str, all_evidence: List[Dict], 
                                   verification_results: List[Dict]) -> Dict:
        """Baseline: All sources get equal weight"""
        
        uniform_weighted = []
        for evidence, verification in zip(all_evidence, verification_results):
            weighted_item = {
                'evidence': evidence,
                'verification': verification,
                'weight': 1.0,  # Equal weight for all
                'bias_alignment': 0.0,  # No bias consideration
                'explanation': "Equal weight - no bias adjustment"
            }
            uniform_weighted.append(weighted_item)
        
        verdict = self.verdict_generator.generate_verdict(uniform_weighted)
        verdict['approach'] = 'uniform'
        verdict['description'] = "Equal weight for all sources (baseline)"
        
        print(f"  Result: {verdict['verdict']} ({verdict['confidence']:.1%})")
        
        return verdict
    
    def _authority_weighting_approach(self, claim: str, all_evidence: List[Dict], 
                                     verification_results: List[Dict], 
                                     bias_analyses: List[Dict]) -> Dict:
        """Authority-only weighting (no bias consideration)"""
        
        authority_weighted = []
        for evidence, verification, bias in zip(all_evidence, verification_results, bias_analyses):
            # Simple authority weight (government = 1.0, web = 0.7)
            base_authority = 1.0 if evidence['evidence_type'] == 'database' else 0.7
            weight = base_authority * verification['confidence']
            
            weighted_item = {
                'evidence': evidence,
                'verification': verification,
                'weight': weight,
                'bias_alignment': 0.0,  # No bias consideration
                'explanation': f"Authority weight: {base_authority} × confidence: {verification['confidence']:.3f}"
            }
            authority_weighted.append(weighted_item)
        
        verdict = self.verdict_generator.generate_verdict(authority_weighted)
        verdict['approach'] = 'authority'
        verdict['description'] = "Authority-based weighting only"
        
        print(f"  Result: {verdict['verdict']} ({verdict['confidence']:.1%})")
        
        return verdict
    
    def _bias_aware_weighting_approach(self, claim: str, all_evidence: List[Dict], 
                                      verification_results: List[Dict], 
                                      bias_analyses: List[Dict]) -> Dict:
        """Your bias-aware weighting algorithm"""
        
        # Use your actual algorithm
        weighted_evidence = self.weighter.weight_all_evidence(
            claim, all_evidence, verification_results, bias_analyses
        )
        
        verdict = self.verdict_generator.generate_verdict(weighted_evidence)
        verdict['approach'] = 'bias_aware'
        verdict['description'] = "Bias-aware dynamic weighting (research contribution)"
        
        print(f"  Result: {verdict['verdict']} ({verdict['confidence']:.1%})")
        
        return verdict
    
    def _analyze_comparison(self, results: Dict) -> Dict:
        """Analyze differences between approaches"""
        
        comparison = {
            'verdicts': {},
            'confidences': {},
            'agreement': {},
            'insights': []
        }
        
        # Extract verdicts and confidences
        for approach, result in results.items():
            if approach != 'comparison' and 'verdict' in result:
                comparison['verdicts'][approach] = result['verdict']
                comparison['confidences'][approach] = result['confidence']
        
        # Check agreement
        verdicts = list(comparison['verdicts'].values())
        all_same = len(set(verdicts)) == 1
        comparison['agreement']['all_agree'] = all_same
        
        # Generate insights
        if 'uniform' in results and 'bias_aware' in results:
            uniform_conf = results['uniform']['confidence']
            bias_aware_conf = results['bias_aware']['confidence']
            
            conf_diff = bias_aware_conf - uniform_conf
            if abs(conf_diff) > 0.05:  # 5% difference threshold
                if conf_diff > 0:
                    comparison['insights'].append(
                        f"Bias-aware weighting increased confidence by {conf_diff:.1%}"
                    )
                else:
                    comparison['insights'].append(
                        f"Bias-aware weighting decreased confidence by {abs(conf_diff):.1%} (detected conflicting bias)"
                    )
            
            # Check verdict changes
            if results['uniform']['verdict'] != results['bias_aware']['verdict']:
                comparison['insights'].append(
                    f"Verdict changed from {results['uniform']['verdict']} to {results['bias_aware']['verdict']} with bias awareness"
                )
        
        return comparison
    
    def print_detailed_comparison(self, results: Dict):
        """Print detailed comparison results"""
        
        print("\n" + "="*70)
        print("DETAILED COMPARISON RESULTS")
        print("="*70)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        # Show results by approach
        for approach in ['uniform', 'authority', 'bias_aware']:
            if approach in results:
                result = results[approach]
                print(f"\n{approach.upper().replace('_', '-')} APPROACH:")
                print(f"  Verdict: {result['verdict']}")
                print(f"  Confidence: {result['confidence']:.1%}")
                print(f"  Description: {result['description']}")
        
        # Show comparison insights
        if 'comparison' in results:
            comp = results['comparison']
            print(f"\nCOMPARISON INSIGHTS:")
            
            if comp['agreement']['all_agree']:
                print("  All approaches agree on verdict")
            else:
                print("  Approaches disagree on verdict:")
                for approach, verdict in comp['verdicts'].items():
                    print(f"    {approach}: {verdict}")
            
            for insight in comp['insights']:
                print(f"  Key insight: {insight}")
        
        print(f"\nResearch Value:")
        print(f"   This comparison demonstrates the impact of bias-aware weighting")
        print(f"   on fact-checking accuracy in politically polarized information environments.")

def demo_comparison():
    """Run a demonstration of baseline comparison"""
    
    comparator = BaselineComparator()
    
    # Example claims for testing
    test_claims = [
        "Sri Lanka's economy has fully recovered from the 2022 crisis",
        "The government's new policy will reduce inflation significantly",
        "Opposition parties are blocking important economic reforms"
    ]
    
    print("="*70)
    print("BASELINE COMPARISON DEMONSTRATION")
    print("="*70)
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\nTEST CASE {i}: {claim}")
        print("-" * 50)
        
        results = comparator.compare_weighting_approaches(claim)
        comparator.print_detailed_comparison(results)
        
        if i < len(test_claims):
            print(f"\n{'='*70}")

if __name__ == "__main__":
    demo_comparison()