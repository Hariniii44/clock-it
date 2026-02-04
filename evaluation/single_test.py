"""
Simple test with just one claim to debug the evaluation process
"""
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.evaluate_ground_truth import FactCheckingEvaluator

def run_single_test():
    """Test with just one simple claim"""
    
    # Simple test claim
    test_claim = {
        "id": 1,
        "claim": "Anura Kumara Dissanayake is the current President of Sri Lanka",
        "true_label": "TRUE",
        "category": "factual"
    }
    
    print(f"Testing claim: {test_claim['claim']}")
    print(f"Expected result: {test_claim['true_label']}")
    
    try:
        # Initialize evaluator
        print("\nInitializing evaluator...")
        evaluator = FactCheckingEvaluator()
        
        print("\nRunning fact check...")
        result = evaluator.fact_check_claim(test_claim['claim'])
        
        print(f"\nResults:")
        print(f"  Verdict: {result['verdict']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Evidence Count: {result['evidence_count']}")
        if result['error']:
            print(f"  Error: {result['error']}")
        
        # Check correctness
        normalized_true = evaluator.normalize_verdict(test_claim['true_label'])
        correct = normalized_true == result['verdict']
        print(f"  Correct: {correct} (Expected: {normalized_true})")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_single_test()