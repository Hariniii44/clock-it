"""
Evaluation script using REAL fact-checks from factcheck.lk
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import exactly the same components as test_pipeline.py
from src.evidence_retrieval.evidence_retrieval import AdvancedEvidenceRetriever
from src.verification.verification import ClaimVerifier
from src.bias_detection.bias_detection import BiasDetector  
from src.evidence_weighting.evidence_weighting import EvidenceWeighter, VerdictGenerator
from config import Config

def evaluate_factcheck_lk_claims():
    """
    Test our system against real fact-checks from factcheck.lk
    """
    
    # Real fact-checks from factcheck.lk (February 2026)
    factcheck_lk_claims = [
        {
            "claim": "As of March 31, our country's total debt has increased up to almost USD 106 billion",
            "speaker": "S.M. Marikkar",
            "factcheck_lk_verdict": "PARTLY_TRUE",
            "source": "Parliament | August 19, 2025",
            "category": "DEBT"
        },
        {
            "claim": "An amount that is higher than anything ever allocated in [the country's] history has been allocated for the president's expenditure head through this [2026] Appropriation Bill; an amount of 11 point something billion has been allocated",
            "speaker": "Tissa Attanayake", 
            "factcheck_lk_verdict": "FALSE",
            "source": "Esana News | October 7, 2025",
            "category": "ECONOMY"
        },
        {
            "claim": "According to international concepts, generally 3% of a country's land area should be allocated for industries, whereas in Sri Lanka this figure stands at only 0.01%",
            "speaker": "Thilaka Jayasundara",
            "factcheck_lk_verdict": "PARTLY_TRUE", 
            "source": "Parliament Website | September 2, 2025",
            "category": "ECONOMY"
        },
        {
            "claim": "Since the number of people suffering from extreme poverty in this country has now exceeded the 25% limit, the country has fallen to the very bottom in terms of poverty",
            "speaker": "Dilith Jayaweera",
            "factcheck_lk_verdict": "TRUE",
            "source": "Aruna | October 20, 2025", 
            "category": "SOCIOECONOMIC"
        },
        {
            "claim": "We have approximately LKR 30 billion out of the allocations approved by the previous budget [That money] can be spent without the parliament's approval. All those allocations have been set aside for this [disaster-related work]",
            "speaker": "Anura Kumara Dissanayake",
            "factcheck_lk_verdict": "TRUE",
            "source": "Art TV | November 30, 2025",
            "category": "ECONOMY"
        },
        {
            "claim": "The officials have fulfilled their duties from their end [issued timely warnings]…Especially, it can be seen from the statements made to the media by the officials of the Department of Meteorology that they have appeared in the media from November 11 – 23 and forecasted that a situation such as this could arise in the country in the upcoming days",
            "speaker": "Mujibur Rahman",
            "factcheck_lk_verdict": "FALSE",
            "source": "Daily Mirror Facebook page | December 4, 2025",
            "category": "ENVIRONMENT"
        }
    ]

    def normalize_verdict(verdict):
        """Normalize our system's verdicts to match factcheck.lk format"""
        mapping = {
            'SUPPORTED': 'TRUE',
            'REFUTED': 'FALSE', 
            'UNCERTAIN': 'INSUFFICIENT_INFO',
            'CONFLICTING': 'PARTLY_TRUE',
            'TRUE': 'TRUE',
            'FALSE': 'FALSE',
            'PARTLY_TRUE': 'PARTLY_TRUE',
            'PARTIALLY_TRUE': 'PARTLY_TRUE'
        }
        return mapping.get(verdict.upper(), verdict.upper())

    def evaluate_single_claim(claim_data):
        """Evaluate a single claim using our pipeline"""
        claim = claim_data['claim']
        
        print(f"🔍 CLAIM: {claim[:100]}...")
        print(f"👤 SPEAKER: {claim_data['speaker']}")
        print(f"📊 FACTCHECK.LK VERDICT: {claim_data['factcheck_lk_verdict']}")
        print("-" * 80)
        
        try:
            # Initialize components
            retriever = AdvancedEvidenceRetriever(Config.SERPER_KEY, groq_key=Config.GROQ_API_KEY)
            verifier = ClaimVerifier()
            bias_detector = BiasDetector()
            weighter = EvidenceWeighter()
            verdict_generator = VerdictGenerator()
            
            # Run through pipeline
            print("Retrieving evidence...")
            evidence = retriever.retrieve_hybrid_serper_decomposition(claim)
            
            if not evidence:
                return {
                    'claim': claim[:100] + "...",
                    'speaker': claim_data['speaker'],
                    'factcheck_lk': claim_data['factcheck_lk_verdict'],
                    'our_system': 'INSUFFICIENT_INFO',
                    'confidence': 0.0,
                    'correct': False,
                    'evidence_count': 0
                }
            
            print(f"Found {len(evidence)} evidence sources")
            
            # Verification
            verification_results = []
            for e in evidence:
                verification = verifier.verify_claim(claim, e["snippet"])
                verification_results.append(verification)
            
            # Bias detection
            bias_analyses = []
            for e in evidence:
                bias = bias_detector.analyze_source(e["snippet"], e["link"])
                bias_analyses.append(bias)
            
            # Weighting and verdict
            weighted_evidence = weighter.weight_all_evidence(
                claim, evidence, verification_results, bias_analyses
            )
            
            final_verdict = verdict_generator.generate_verdict(weighted_evidence)
            
            # Compare results
            our_verdict = normalize_verdict(final_verdict['verdict'])
            factcheck_verdict = claim_data['factcheck_lk_verdict']
            is_correct = our_verdict == factcheck_verdict
            
            print(f"OUR VERDICT: {our_verdict} ({final_verdict['confidence']:.1%} confidence)")
            print(f"FACTCHECK.LK: {factcheck_verdict}")
            print(f"MATCH: {'YES' if is_correct else 'NO'}")
            
            return {
                'claim': claim[:100] + "...",
                'speaker': claim_data['speaker'],
                'factcheck_lk': factcheck_verdict,
                'our_system': our_verdict,
                'confidence': final_verdict['confidence'],
                'correct': is_correct,
                'evidence_count': len(evidence),
                'category': claim_data['category']
            }
            
        except Exception as e:
            print(f"ERROR: {e}")
            return {
                'claim': claim[:100] + "...",
                'speaker': claim_data['speaker'], 
                'factcheck_lk': claim_data['factcheck_lk_verdict'],
                'our_system': 'ERROR',
                'confidence': 0.0,
                'correct': False,
                'evidence_count': 0,
                'error': str(e)
            }

    print("🚀 EVALUATING AGAINST FACTCHECK.LK VERDICTS")
    print(f"📊 Testing {len(factcheck_lk_claims)} real fact-checks")
    print("=" * 80)
    
    results = []
    for i, claim_data in enumerate(factcheck_lk_claims, 1):
        print(f"\n📋 TEST {i}/{len(factcheck_lk_claims)}")
        result = evaluate_single_claim(claim_data)
        results.append(result)
        print("\n" + "="*80)
    
    # Calculate metrics
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        accuracy = sum(1 for r in valid_results if r['correct']) / len(valid_results)
        avg_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results)
        
        print(f"\n🎯 FACTCHECK.LK VALIDATION RESULTS:")
        print(f"   Total Claims: {len(results)}")
        print(f"   Valid Results: {len(valid_results)}")
        print(f"   Accuracy vs FactCheck.lk: {accuracy:.1%}")
        print(f"   Average Confidence: {avg_confidence:.1%}")
        
        print(f"\n📊 DETAILED COMPARISON:")
        for i, result in enumerate(results, 1):
            status = "✅" if result['correct'] else "❌"
            print(f"   {i}. {status} Our: {result['our_system']:12s} | FactCheck.lk: {result['factcheck_lk']:12s} | {result['confidence']:.0%} | {result['speaker']}")
        
        # Category analysis
        categories = {}
        for result in valid_results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = {'correct': 0, 'total': 0}
            categories[cat]['total'] += 1
            if result['correct']:
                categories[cat]['correct'] += 1
        
        print(f"\n📈 PERFORMANCE BY CATEGORY:")
        for category, stats in categories.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"   {category}: {acc:.1%} ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    evaluate_factcheck_lk_claims()