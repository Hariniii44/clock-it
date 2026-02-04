"""
Evaluation script that follows the EXACT same pipeline as test_pipeline.py
"""
import json
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import exactly the same components as test_pipeline.py
from src.evidence_retrieval.evidence_retrieval import AdvancedEvidenceRetriever
from src.verification.verification import ClaimVerifier
from src.bias_detection.bias_detection import BiasDetector  
from src.evidence_weighting.evidence_weighting import EvidenceWeighter, VerdictGenerator
from src.synthesis.synthesis_engine import SynthesisEngine
from config import Config

def evaluate_single_claim_exact_pipeline(claim, expected_verdict):
    """
    Run a single claim through the EXACT same pipeline as test_pipeline.py
    """
    print(f"🔍 EVALUATING: {claim}")
    print(f"📋 Expected: {expected_verdict}")
    print("=" * 80)

    # Initialize components exactly like test_pipeline.py
    retriever = AdvancedEvidenceRetriever(Config.SERPER_KEY, groq_key=Config.GROQ_API_KEY)
    verifier = ClaimVerifier()
    bias_detector = BiasDetector()
    weighter = EvidenceWeighter()
    verdict_generator = VerdictGenerator()

    try:
        # STEP 1: Evidence Retrieval (exactly like test_pipeline.py)
        print("1. Retrieving evidence...\n")
        evidence = retriever.retrieve_hybrid_serper_decomposition(claim)
        
        if not evidence:
            print("❌ No evidence found")
            return {
                'claim': claim,
                'expected': expected_verdict,
                'actual': 'INSUFFICIENT_INFO', 
                'confidence': 0.0,
                'correct': False,
                'error': 'No evidence found'
            }

        # STEP 2: Verification (exactly like test_pipeline.py)
        print("2. Verifying evidence against claim...\n")
        verification_results = []
        for i, e in enumerate(evidence, 1):
            verification_result = verifier.verify_claim(claim, e["snippet"])
            verification_results.append(verification_result)
        
        # STEP 3: Bias Detection (exactly like test_pipeline.py)  
        print("3. Analyzing bias...\n")
        bias_analyses = []
        for i, e in enumerate(evidence, 1):
            bias_analysis = bias_detector.analyze_source(e["snippet"], e["link"])
            bias_analyses.append(bias_analysis)

        # STEP 4: Evidence Weighting (exactly like test_pipeline.py)
        print("4. Weighting evidence...\n")
        weighted_evidence = weighter.weight_all_evidence(
            claim, evidence, verification_results, bias_analyses
        )

        # STEP 5: Generate Final Verdict (exactly like test_pipeline.py)
        print("5. Generating final verdict...\n")
        final_verdict = verdict_generator.generate_verdict(weighted_evidence)

        # Normalize verdicts for comparison
        def normalize_verdict(verdict):
            mapping = {
                'SUPPORTED': 'TRUE',
                'REFUTED': 'FALSE', 
                'UNCERTAIN': 'INSUFFICIENT_INFO',
                'CONFLICTING': 'CONTROVERSIAL',
                'TRUE': 'TRUE',
                'FALSE': 'FALSE',
                'PARTIALLY_TRUE': 'PARTIALLY_TRUE'
            }
            return mapping.get(verdict.upper(), verdict.upper())

        actual_verdict = normalize_verdict(final_verdict['verdict'])
        expected_normalized = normalize_verdict(expected_verdict)
        is_correct = actual_verdict == expected_normalized

        result = {
            'claim': claim,
            'expected': expected_normalized,
            'actual': actual_verdict,
            'confidence': final_verdict['confidence'],
            'correct': is_correct,
            'error': None,
            'evidence_count': len(evidence),
            'support_score': final_verdict.get('support_score', 0),
            'refute_score': final_verdict.get('refute_score', 0),
            'is_factual_claim': final_verdict.get('is_factual_claim', False)
        }

        # Print results
        print("🏆 FINAL RESULTS:")
        print(f"   Expected: {expected_normalized}")
        print(f"   Actual: {actual_verdict}")
        print(f"   Confidence: {final_verdict['confidence']:.1%}")
        print(f"   Correct: {'✅' if is_correct else '❌'}")
        print(f"   Evidence Sources: {len(evidence)}")
        print(f"   Support Score: {final_verdict.get('support_score', 0):.1%}")
        print(f"   Refute Score: {final_verdict.get('refute_score', 0):.1%}")
        if 'is_factual_claim' in final_verdict:
            claim_type = "Factual" if final_verdict['is_factual_claim'] else "Opinion"
            print(f"   Claim Type: {claim_type}")

        return result

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'claim': claim,
            'expected': expected_verdict,
            'actual': 'ERROR',
            'confidence': 0.0,
            'correct': False,
            'error': str(e)
        }

def run_evaluation_test():
    """Run evaluation on a comprehensive set of claims"""
    
    # Expanded test claims covering different categories and difficulties
    test_claims = [
        # FACTUAL CLAIMS - Government & Politics
        {
            "claim": "Anura Kumara Dissanayake is the current President of Sri Lanka",
            "expected": "TRUE"
        },
        {
            "claim": "The Parliament of Sri Lanka has 225 members", 
            "expected": "TRUE"
        },
        {
            "claim": "Ranil Wickremesinghe served as Prime Minister of Sri Lanka multiple times",
            "expected": "TRUE"
        },
        {
            "claim": "Gotabaya Rajapaksa resigned as President in 2022",
            "expected": "TRUE"
        },
        
        # FACTUAL CLAIMS - Geography
        {
            "claim": "Sri Lanka is an island nation in the Indian Ocean",
            "expected": "TRUE"
        },
        {
            "claim": "Sri Lanka shares a land border with India",
            "expected": "FALSE"
        },
        {
            "claim": "Sri Lanka has 9 provinces",
            "expected": "TRUE"
        },
        {
            "claim": "Colombo is the capital city of Sri Lanka",
            "expected": "PARTIALLY_TRUE"  # Commercial vs Administrative capital
        },
        
        # FACTUAL CLAIMS - History & Culture
        {
            "claim": "Sri Lanka became independent from Britain in 1948",
            "expected": "TRUE"
        },
        {
            "claim": "Ceylon was the former name of Sri Lanka",
            "expected": "TRUE"
        },
        {
            "claim": "The 2004 Indian Ocean tsunami affected Sri Lanka",
            "expected": "TRUE"
        },
        {
            "claim": "Buddhism is the majority religion in Sri Lanka",
            "expected": "TRUE"
        },
        
        # FACTUAL CLAIMS - Economy
        {
            "claim": "The Sri Lankan Rupee is the official currency of Sri Lanka",
            "expected": "TRUE"
        },
        {
            "claim": "Tea is one of Sri Lanka's major export commodities",
            "expected": "TRUE"
        },
        {
            "claim": "Sri Lanka defaulted on its foreign debt in 2022",
            "expected": "TRUE"
        },
        
        # FALSE CLAIMS
        {
            "claim": "The population of Sri Lanka is over 50 million",
            "expected": "FALSE"  # Actually ~22 million
        },
        {
            "claim": "Tamil is not an official language of Sri Lanka",
            "expected": "FALSE"  # Both Sinhala and Tamil are official
        },
        {
            "claim": "Sinhala and Tamil are the only languages spoken in Sri Lanka",
            "expected": "FALSE"  # English is also widely spoken
        },
        
        # CONTROVERSIAL/POLICY CLAIMS
        {
            "claim": "The current government's economic reforms are successfully stabilizing the economy",
            "expected": "CONTROVERSIAL"
        },
        {
            "claim": "The 13th Amendment should be fully implemented in Sri Lanka",
            "expected": "CONTROVERSIAL"
        },
        {
            "claim": "Sri Lanka's education system is among the best in South Asia",
            "expected": "CONTROVERSIAL"
        }
    ]

    print("🚀 STARTING COMPREHENSIVE EVALUATION WITH EXACT PIPELINE")
    print(f"📊 Testing {len(test_claims)} claims across multiple categories")
    print("=" * 80)
    
    results = []
    for i, test in enumerate(test_claims, 1):
        print(f"\n📋 TEST {i}/{len(test_claims)}")
        result = evaluate_single_claim_exact_pipeline(test['claim'], test['expected'])
        results.append(result)
        print("\n" + "="*80)
    
    # Calculate comprehensive metrics
    valid_results = [r for r in results if r['error'] is None]
    if valid_results:
        accuracy = sum(1 for r in valid_results if r['correct']) / len(valid_results)
        avg_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results)
        
        # Category breakdown
        factual_results = [r for r in valid_results if r['expected'] in ['TRUE', 'FALSE', 'PARTIALLY_TRUE']]
        controversial_results = [r for r in valid_results if r['expected'] == 'CONTROVERSIAL']
        
        factual_accuracy = sum(1 for r in factual_results if r['correct']) / len(factual_results) if factual_results else 0
        controversial_accuracy = sum(1 for r in controversial_results if r['correct']) / len(controversial_results) if controversial_results else 0
        
        print(f"\n🎯 COMPREHENSIVE EVALUATION RESULTS:")
        print(f"   Total Claims: {len(results)}")
        print(f"   Valid Results: {len(valid_results)}")
        print(f"   Overall Accuracy: {accuracy:.1%}")
        print(f"   Average Confidence: {avg_confidence:.1%}")
        print(f"   Factual Claims Accuracy: {factual_accuracy:.1%} ({len(factual_results)} claims)")
        print(f"   Controversial Claims Accuracy: {controversial_accuracy:.1%} ({len(controversial_results)} claims)")
        
        print(f"\n📊 DETAILED RESULTS:")
        for i, result in enumerate(results, 1):
            status = "✅" if result['correct'] else "❌"
            claim_short = result['claim'][:60] + "..." if len(result['claim']) > 60 else result['claim']
            print(f"   {i:2d}. {status} {result['actual']:15s} (expected {result['expected']:15s}) - {result['confidence']:.0%} | {claim_short}")
        
        # Error analysis
        incorrect_results = [r for r in valid_results if not r['correct']]
        if incorrect_results:
            print(f"\n❌ INCORRECT PREDICTIONS ANALYSIS:")
            for result in incorrect_results:
                print(f"   • Expected {result['expected']}, got {result['actual']} ({result['confidence']:.0%})")
                print(f"     Claim: {result['claim']}")
        
        print(f"\n📈 SYSTEM PERFORMANCE SUMMARY:")
        print(f"   • Evidence Retrieval: {sum(r['evidence_count'] for r in valid_results) / len(valid_results):.1f} sources/claim average")
        print(f"   • High Confidence Predictions (>80%): {sum(1 for r in valid_results if r['confidence'] > 0.8) / len(valid_results):.1%}")
        print(f"   • Low Confidence Predictions (<50%): {sum(1 for r in valid_results if r['confidence'] < 0.5) / len(valid_results):.1%}")

if __name__ == "__main__":
    run_evaluation_test()