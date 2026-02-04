"""
Quick test script to run evaluation on a small subset
"""
import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.evaluate_ground_truth import FactCheckingEvaluator

def run_quick_test():
    """Run evaluation on first 5 claims for quick testing"""
    
    # Load dataset
    with open('ground_truth_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Take first 5 claims for quick test
    test_dataset = dataset[:5]
    
    print(f"Running quick test with {len(test_dataset)} claims...")
    print("Claims to test:")
    for claim in test_dataset:
        print(f"  {claim['id']}: {claim['claim']} (Expected: {claim['true_label']})")
    
    # Initialize evaluator
    evaluator = FactCheckingEvaluator()
    
    # Test each claim
    results = []
    for item in test_dataset:
        print(f"\nTesting: {item['claim']}")
        result = evaluator.fact_check_claim(item['claim'])
        
        evaluation_result = {
            'id': item['id'],
            'claim': item['claim'],
            'true_label': evaluator.normalize_verdict(item['true_label']),
            'predicted_label': result['verdict'],
            'confidence': result['confidence'],
            'correct': evaluator.normalize_verdict(item['true_label']) == result['verdict'],
            'error': result['error']
        }
        results.append(evaluation_result)
        
        print(f"  Expected: {evaluation_result['true_label']}")
        print(f"  Predicted: {evaluation_result['predicted_label']}")
        print(f"  Confidence: {evaluation_result['confidence']:.3f}")
        print(f"  Correct: {evaluation_result['correct']}")
        if evaluation_result['error']:
            print(f"  Error: {evaluation_result['error']}")
    
    # Calculate quick metrics
    correct_count = sum(1 for r in results if r['correct'] and not r['error'])
    total_valid = len([r for r in results if not r['error']])
    accuracy = correct_count / total_valid if total_valid > 0 else 0
    
    print(f"\n{'='*50}")
    print("QUICK TEST RESULTS")
    print(f"{'='*50}")
    print(f"Total claims tested: {len(results)}")
    print(f"Valid predictions: {total_valid}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.3f}")
    
    return results

if __name__ == "__main__":
    run_quick_test()