import json
import sys
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import pandas as pd
from datetime import datetime
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evidence_retrieval.evidence_retrieval import AdvancedEvidenceRetriever
from verification.verification import ClaimVerifier  
from bias_detection.bias_detection import BiasDetector
from evidence_weighting.evidence_weighting import EvidenceWeighter
from evidence_weighting.evidence_weighting import VerdictGenerator
from synthesis.synthesis_engine import SynthesisEngine
from config import Config

class FactCheckingEvaluator:
    def __init__(self):
        """Initialize the evaluator with all pipeline components"""
        print("Initializing fact-checking pipeline components...")
        self.evidence_retriever = AdvancedEvidenceRetriever(
            serper_key=Config.SERPER_KEY,
            groq_key=Config.GROQ_API_KEY
        )
        self.fact_verifier = ClaimVerifier()
        self.bias_detector = BiasDetector()
        self.evidence_weighter = EvidenceWeighter()
        self.verdict_generator = VerdictGenerator()
        self.synthesis_engine = SynthesisEngine(Config.GROQ_API_KEY)
        
    def load_dataset(self, dataset_path):
        """Load the ground truth dataset"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def normalize_verdict(self, verdict):
        """Normalize verdict strings for comparison"""
        verdict_mapping = {
            'SUPPORTED': 'TRUE',
            'REFUTED': 'FALSE', 
            'NOT ENOUGH INFO': 'INSUFFICIENT_INFO',
            'PARTIALLY SUPPORTED': 'PARTIALLY_TRUE',
            'TRUE': 'TRUE',
            'FALSE': 'FALSE',
            'PARTIALLY_TRUE': 'PARTIALLY_TRUE',
            'CONTROVERSIAL': 'CONTROVERSIAL',
            'INSUFFICIENT_INFO': 'INSUFFICIENT_INFO'
        }
        return verdict_mapping.get(verdict.upper(), verdict.upper())
    
    def fact_check_claim(self, claim):
        """Run a single claim through the complete fact-checking pipeline"""
        try:
            # Step 1: Evidence Retrieval
            evidence_results = self.evidence_retriever.retrieve_hybrid_serper_decomposition(claim)
            if not evidence_results or len(evidence_results) == 0:
                return {
                    'verdict': 'INSUFFICIENT_INFO',
                    'confidence': 0.0,
                    'error': None,
                    'evidence_count': 0
                }
            
            # Step 2: Verification
            verification_results = []
            for evidence in evidence_results:
                verification = self.fact_verifier.verify_claim(claim, evidence['snippet'])
                verification_results.append({
                    'source': evidence['source'],
                    'url': evidence['link'],
                    'content': evidence['snippet'],
                    'title': evidence.get('title', ''),
                    'verification': verification
                })
            
            # Step 3: Bias Detection
            bias_results = []
            for result in verification_results:
                bias_info = self.bias_detector.analyze_source(
                    result['content'], 
                    result['url']
                )
                result['bias'] = bias_info
                bias_results.append(result)
            
            # Step 4: Evidence Weighting and Verdict Generation
            weighted_evidence = self.evidence_weighter.weight_all_evidence(
                claim, evidence_results, verification_results, bias_results
            )
            
            final_verdict = self.verdict_generator.generate_verdict(weighted_evidence)
            
            return {
                'verdict': self.normalize_verdict(final_verdict['verdict']),
                'confidence': final_verdict['confidence'],
                'error': None,
                'evidence_count': len(evidence_results)
            }
            
        except Exception as e:
            return {
                'verdict': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'evidence_count': 0
            }
    
    def evaluate_dataset(self, dataset_path, output_path=None):
        """Evaluate the entire dataset and calculate metrics"""
        dataset = self.load_dataset(dataset_path)
        results = []
        
        print(f"Evaluating {len(dataset)} claims...")
        start_time = time.time()
        
        for i, item in enumerate(dataset):
            print(f"Processing claim {i+1}/{len(dataset)}: {item['claim'][:60]}...")
            
            # Run fact-checking
            result = self.fact_check_claim(item['claim'])
            
            # Store results
            evaluation_result = {
                'id': item['id'],
                'claim': item['claim'],
                'true_label': self.normalize_verdict(item['true_label']),
                'predicted_label': result['verdict'],
                'confidence': result['confidence'],
                'category': item['category'],
                'subcategory': item['subcategory'],
                'difficulty': item['difficulty'],
                'evidence_count': result['evidence_count'],
                'error': result['error'],
                'correct': self.normalize_verdict(item['true_label']) == result['verdict']
            }
            results.append(evaluation_result)
            
            # Print progress
            if evaluation_result['error']:
                print(f"  ERROR: {evaluation_result['error']}")
            else:
                print(f"  True: {evaluation_result['true_label']}, "
                      f"Predicted: {evaluation_result['predicted_label']}, "
                      f"Confidence: {evaluation_result['confidence']:.2f}, "
                      f"Correct: {evaluation_result['correct']}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Calculate metrics
        metrics = self.calculate_metrics(results)
        metrics['total_processing_time'] = processing_time
        metrics['avg_time_per_claim'] = processing_time / len(dataset)
        
        # Save results if output path provided
        if output_path:
            self.save_results(results, metrics, output_path)
        
        return results, metrics
    
    def calculate_metrics(self, results):
        """Calculate evaluation metrics"""
        # Filter out error cases for metric calculation
        valid_results = [r for r in results if r['error'] is None and r['predicted_label'] != 'ERROR']
        
        if not valid_results:
            return {'error': 'No valid predictions to evaluate'}
        
        true_labels = [r['true_label'] for r in valid_results]
        predicted_labels = [r['predicted_label'] for r in valid_results]
        
        # Overall accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Per-class metrics
        labels = list(set(true_labels + predicted_labels))
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, labels=labels, average=None, zero_division=0
        )
        
        # Macro and micro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='macro', zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='micro', zero_division=0
        )
        
        # Category-wise performance
        category_performance = {}
        for category in set(r['category'] for r in valid_results):
            category_results = [r for r in valid_results if r['category'] == category]
            category_accuracy = sum(r['correct'] for r in category_results) / len(category_results)
            category_performance[category] = {
                'accuracy': category_accuracy,
                'count': len(category_results)
            }
        
        # Confidence analysis
        avg_confidence_correct = sum(r['confidence'] for r in valid_results if r['correct']) / max(1, sum(r['correct'] for r in valid_results))
        avg_confidence_incorrect = sum(r['confidence'] for r in valid_results if not r['correct']) / max(1, sum(not r['correct'] for r in valid_results))
        
        return {
            'total_claims': len(results),
            'valid_predictions': len(valid_results),
            'errors': len(results) - len(valid_results),
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro, 
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'per_class_metrics': {
                labels[i]: {
                    'precision': precision[i],
                    'recall': recall[i], 
                    'f1': f1[i],
                    'support': support[i]
                } for i in range(len(labels))
            },
            'category_performance': category_performance,
            'avg_confidence_correct': avg_confidence_correct,
            'avg_confidence_incorrect': avg_confidence_incorrect
        }
    
    def save_results(self, results, metrics, output_path):
        """Save evaluation results and metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"{output_path}/evaluation_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'results': results,
                'metrics': metrics
            }, f, indent=2, ensure_ascii=False)
        
        # Save metrics summary
        metrics_file = f"{output_path}/evaluation_metrics_{timestamp}.json" 
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # Save CSV for analysis
        csv_file = f"{output_path}/evaluation_results_{timestamp}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"Results saved to:")
        print(f"  - {results_file}")
        print(f"  - {metrics_file}")
        print(f"  - {csv_file}")
    
    def print_metrics_summary(self, metrics):
        """Print a formatted summary of metrics"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Claims: {metrics['total_claims']}")
        print(f"Valid Predictions: {metrics['valid_predictions']}")
        print(f"Errors: {metrics['errors']}")
        print(f"Processing Time: {metrics.get('total_processing_time', 0):.2f}s")
        print(f"Avg Time per Claim: {metrics.get('avg_time_per_claim', 0):.2f}s")
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Macro F1: {metrics['f1_macro']:.3f}")
        print(f"Macro Precision: {metrics['precision_macro']:.3f}")
        print(f"Macro Recall: {metrics['recall_macro']:.3f}")
        
        print(f"\nPER-CLASS PERFORMANCE:")
        for label, class_metrics in metrics['per_class_metrics'].items():
            print(f"{label:15s}: P={class_metrics['precision']:.3f}, "
                  f"R={class_metrics['recall']:.3f}, "
                  f"F1={class_metrics['f1']:.3f}, "
                  f"Support={class_metrics['support']}")
        
        print(f"\nCATEGORY PERFORMANCE:")
        for category, perf in metrics['category_performance'].items():
            print(f"{category:15s}: Accuracy={perf['accuracy']:.3f} ({perf['count']} claims)")
        
        print(f"\nCONFIDENCE ANALYSIS:")
        print(f"Avg Confidence (Correct): {metrics['avg_confidence_correct']:.3f}")
        print(f"Avg Confidence (Incorrect): {metrics['avg_confidence_incorrect']:.3f}")

def main():
    """Main evaluation function"""
    evaluator = FactCheckingEvaluator()
    
    # Paths
    dataset_path = "ground_truth_dataset.json"
    output_dir = "results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    print("Starting fact-checking evaluation...")
    results, metrics = evaluator.evaluate_dataset(dataset_path, output_dir)
    
    # Print summary
    evaluator.print_metrics_summary(metrics)

if __name__ == "__main__":
    main()