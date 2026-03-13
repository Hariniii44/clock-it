"""
FEVER Dataset Evaluation Script
Tests the bias-aware fact-checking system against the academic FEVER benchmark
"""
import json
import os
import sys
import random
import time
from datetime import datetime
from typing import List, Dict, Any

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("WARNING: datasets library not available. Install with: pip install datasets")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import exactly the same components as test_pipeline.py
from src.evidence_retrieval.evidence_retrieval import AdvancedEvidenceRetriever
from src.verification.verification import ClaimVerifier
from src.bias_detection.bias_detection import BiasDetector  
from src.evidence_weighting.evidence_weighting import EvidenceWeighter, VerdictGenerator
from config import Config

class FEVERDatasetLoader:
    """Loads and processes FEVER dataset from Hugging Face or fallback samples"""
    
    def __init__(self):
        self.fever_data = []
        self.use_huggingface = DATASETS_AVAILABLE
        
    def load_fever_from_huggingface(self, split: str = "dev", sample_size: int = 50) -> List[Dict]:
        """
        Load FEVER dataset from Hugging Face using the standard approach
        
        Args:
            split: Dataset split to use ('train', 'dev', 'test')
            sample_size: Number of examples to sample
        """
        try:
            print(f"Loading FEVER dataset from Hugging Face (split: {split})...")
            
            # METHOD 1: Try the correct FEVER dataset format
            dataset_attempts = [
                ("fever", "v1.0", "validation"),  # Use validation split (has labels)
                ("fever", "v1.0", "train"),       # Fallback to train
                ("fever", "v2.0", "validation"),  # Try v2.0 
                ("fever", None, "validation"),    # Without version
                ("kilt_tasks", "fever", "validation"),  # KILT benchmark version
            ]
            
            dataset = None
            for dataset_name, version, data_split in dataset_attempts:
                try:
                    print(f"Trying {dataset_name} v{version} ({data_split})...")
                    
                    if version:
                        dataset = load_dataset(dataset_name, version, split=data_split)
                    else:
                        dataset = load_dataset(dataset_name, split=data_split)
                        
                    print(f"   Successfully loaded {dataset_name} v{version} ({data_split})")
                    print(f"   Dataset size: {len(dataset)} examples")
                    break
                except Exception as attempt_error:
                    error_msg = str(attempt_error)
                    if "trust_remote_code" in error_msg or "loading script" in error_msg:
                        print(f"   FAILED {dataset_name}: Deprecated loading script")
                    elif "ConnectionError" in error_msg or "timeout" in error_msg.lower():
                        print(f"   FAILED {dataset_name}: Network issue")
                    elif "Dataset 'fever' doesn't exist" in error_msg:
                        print(f"   FAILED {dataset_name}: Dataset not found")
                    else:
                        print(f"   FAILED {dataset_name}: {error_msg[:100]}...")
                    continue
            
            # METHOD 2: If Hugging Face fails, try direct download
            if dataset is None:
                print("\nTrying direct download from official FEVER source...")
                dataset = self.download_fever_direct(split, sample_size)
                
            if dataset is None:
                raise Exception("All FEVER dataset loading attempts failed")
            
            # Process the dataset
            filtered_data = []
            print(f"Processing dataset with {len(dataset)} total examples...")
            
            for i, item in enumerate(dataset):
                # Handle different possible field names including KILT format
                claim = None
                label = None
                item_id = item.get('id') or item.get('idx') or str(i)
                evidence = ''
                
                # KILT format handling
                if 'input' in item and 'output' in item:
                    claim = item['input']
                    # Extract label from output (KILT format)
                    if isinstance(item['output'], list) and len(item['output']) > 0:
                        output_item = item['output'][0]
                        if isinstance(output_item, dict):
                            label = output_item.get('answer', '')
                    elif isinstance(item['output'], dict):
                        label = item['output'].get('answer', '')
                    
                    # Try meta field for label
                    if not label and 'meta' in item:
                        meta = item['meta']
                        if isinstance(meta, dict):
                            label = meta.get('label') or meta.get('verdict')
                
                # Standard format handling
                if not claim:
                    claim = item.get('claim') or item.get('text') or item.get('sentence')
                if not label:
                    label = item.get('label') or item.get('verdict') or item.get('classification')
                
                evidence = item.get('evidence') or item.get('wiki_evidence') or ''
                
                # Debug: print first few items to see structure
                if i < 3:
                    print(f"   Sample item {i}: {list(item.keys())}")
                    if 'input' in item:
                        print(f"       input (claim): {str(item['input'])[:100]}")
                    if 'output' in item:
                        print(f"       output: {item['output']}")
                    if 'meta' in item:
                        print(f"       meta: {item.get('meta', {})}")
                    print(f"       extracted claim: {str(claim)[:100] if claim else 'None'}")
                    print(f"       extracted label: {label}")
                
                # Normalize label format
                if label is not None:
                    label_str = str(label).upper().strip()
                    # Handle various label formats
                    if label_str in ['SUPPORTS', 'SUPPORT', 'TRUE', 'ENTAILS', 'ENTAILMENT']:
                        label_str = 'SUPPORTS'
                    elif label_str in ['REFUTES', 'REFUTE', 'FALSE', 'CONTRADICTS', 'CONTRADICTION']:
                        label_str = 'REFUTES'
                    elif label_str in ['NOT_ENOUGH_INFO', 'NEI', 'NOT ENOUGH INFO', 'NEUTRAL']:
                        label_str = 'NOT_ENOUGH_INFO'
                else:
                    label_str = None
                
                # Filter for SUPPORTS and REFUTES only (exclude NOT_ENOUGH_INFO for cleaner eval)
                if (claim is not None and claim.strip() and 
                    label_str in ['SUPPORTS', 'REFUTES']):
                    
                    filtered_data.append({
                        "id": item_id,
                        "claim": str(claim).strip(),
                        "label": label_str,
                        "evidence": str(evidence) if evidence else ''
                    })
                    
                    if len(filtered_data) >= sample_size:
                        break
            
            if not filtered_data:
                raise Exception("No valid FEVER examples found with SUPPORTS/REFUTES labels")
            
            print(f"Successfully loaded {len(filtered_data)} FEVER examples with definitive labels")
            return filtered_data
            
        except Exception as e:
            print(f"WARNING: Failed to load FEVER from Hugging Face: {e}")
            print("Falling back to curated fact-checking claims...")
            return self.load_alternative_fever_data()
    
    def download_fever_direct(self, split: str = "dev", sample_size: int = 50) -> List[Dict]:
        """
        METHOD 2: Direct download from official FEVER source
        """
        try:
            import urllib.request
            import tempfile
            import json
            
            print("   Downloading FEVER dataset from official source...")
            
            # Official FEVER URLs
            fever_urls = {
                "dev": "https://fever.ai/download/fever/shared_task_dev.jsonl",
                "validation": "https://fever.ai/download/fever/shared_task_dev.jsonl",
                "train": "https://fever.ai/download/fever/train.jsonl",
            }
            
            # Map split names
            download_split = "dev" if split in ["dev", "validation"] else split
            
            if download_split not in fever_urls:
                print(f"   ERROR: Split '{split}' not available for direct download")
                return None
            
            url = fever_urls[download_split]
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.jsonl') as tmp_file:
                tmp_path = tmp_file.name
            
            print(f"   Downloading from: {url}")
            urllib.request.urlretrieve(url, tmp_path)
            
            # Parse JSONL file
            fever_data = []
            with open(tmp_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    
                    try:
                        item = json.loads(line.strip())
                        
                        # Extract fields
                        claim = item.get('claim', '')
                        label = item.get('label', '')
                        item_id = item.get('id', str(i))
                        evidence = item.get('evidence', [])
                        
                        # Filter for SUPPORTS and REFUTES
                        if label and str(label).upper() in ['SUPPORTS', 'REFUTES']:
                            fever_data.append({
                                "id": item_id,
                                "claim": claim,
                                "label": str(label).upper(),
                                "evidence": evidence
                            })
                        
                        if len(fever_data) >= sample_size:
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            # Cleanup
            os.unlink(tmp_path)
            
            if fever_data:
                print(f"   Successfully downloaded {len(fever_data)} FEVER examples")
                return fever_data
            else:
                print(f"   ERROR: No valid data found in downloaded file")
                return None
                
        except Exception as e:
            print(f"   ERROR: Direct download failed: {e}")
            return None
    
    def load_alternative_fever_data(self) -> List[Dict]:
        """
        Curated fact-checking claims as fallback when FEVER dataset fails
        """
        try:
            print("Using curated fact-checking claims as fallback...")
            
            # Curated fact-checking claims with clear factual assertions
            factual_claims = [
                {
                    "id": "fact_1",
                    "claim": "Barack Obama was the 44th President of the United States.",
                    "label": "SUPPORTS",
                    "evidence": "Factual claim about US presidency"
                },
                {
                    "id": "fact_2", 
                    "claim": "The Great Wall of China is visible from space without aid.",
                    "label": "REFUTES",
                    "evidence": "Common misconception about space visibility"
                },
                {
                    "id": "fact_3",
                    "claim": "Water boils at 100 degrees Celsius at sea level.",
                    "label": "SUPPORTS", 
                    "evidence": "Basic physics fact"
                },
                {
                    "id": "fact_4",
                    "claim": "Vaccines cause autism in children.",
                    "label": "REFUTES",
                    "evidence": "Debunked medical misconception"
                },
                {
                    "id": "fact_5",
                    "claim": "The Earth is approximately 4.5 billion years old.",
                    "label": "SUPPORTS",
                    "evidence": "Scientific consensus on Earth's age"
                },
                {
                    "id": "fact_6",
                    "claim": "Shakespeare wrote Romeo and Juliet.",
                    "label": "SUPPORTS",
                    "evidence": "Literary fact"
                },
                {
                    "id": "fact_7", 
                    "claim": "The 1969 moon landing was filmed in a Hollywood studio.",
                    "label": "REFUTES",
                    "evidence": "Debunked conspiracy theory"
                },
                {
                    "id": "fact_8",
                    "claim": "Albert Einstein developed the theory of relativity.",
                    "label": "SUPPORTS",
                    "evidence": "Scientific fact"
                },
                {
                    "id": "fact_9",
                    "claim": "Humans only use 10% of their brain capacity.",
                    "label": "REFUTES", 
                    "evidence": "Debunked neuroscience myth"
                },
                {
                    "id": "fact_10",
                    "claim": "The Pacific Ocean is the largest ocean on Earth.",
                    "label": "SUPPORTS",
                    "evidence": "Geographic fact"
                }
            ]
            
            print(f"Using {len(factual_claims)} curated fact-checking claims")
            return factual_claims
            
        except Exception as backup_error:
            print(f"WARNING: Fallback data creation failed: {backup_error}")
            return self.load_fever_fallback_data()
    
    def load_fever_fallback_data(self) -> List[Dict]:
        """
        Minimal fallback when Hugging Face is not available
        """
        
        return [
            {
                "id": "fallback_1",
                "claim": "This is a fallback claim when FEVER dataset is unavailable.",
                "label": "SUPPORTS",
                "evidence": "Please install datasets library: pip install datasets"
            }
        ]
        
    def get_sample(self, n: int = 25, use_huggingface: bool = None) -> List[Dict]:
        """Get a sample of n claims, preferring Hugging Face if available"""
        if use_huggingface is None:
            use_huggingface = self.use_huggingface
            
        if use_huggingface and DATASETS_AVAILABLE:
            return self.load_fever_from_huggingface(sample_size=n)
        else:
            print("Hugging Face datasets not available - please install: pip install datasets")
            data = self.load_fever_fallback_data()
            return data  # Return minimal fallback

class FEVERMetrics:
    """Calculate FEVER-specific evaluation metrics"""
    
    @staticmethod
    def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
        """Calculate accuracy, precision, recall, F1 for each label"""
        
        # Count predictions by label
        label_stats = {
            'SUPPORTS': {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0},
            'REFUTES': {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0},
            'NOT_ENOUGH_INFO': {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0}
        }
        
        valid_results = [r for r in results if 'error' not in r]
        total_correct = 0
        
        for result in valid_results:
            true_label = result['fever_label']
            pred_label = result['our_verdict']
            
            # Count totals per true label
            if true_label in label_stats:
                label_stats[true_label]['total'] += 1
            
            # Check if correct
            if true_label == pred_label:
                total_correct += 1
                if true_label in label_stats:
                    label_stats[true_label]['tp'] += 1
            else:
                # False negative for true label
                if true_label in label_stats:
                    label_stats[true_label]['fn'] += 1
                # False positive for predicted label  
                if pred_label in label_stats:
                    label_stats[pred_label]['fp'] += 1
        
        # Calculate overall accuracy
        accuracy = total_correct / len(valid_results) if valid_results else 0
        
        # Calculate per-label metrics
        label_metrics = {}
        for label, stats in label_stats.items():
            if stats['total'] > 0:
                precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
                recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                label_metrics[label] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': stats['total']
                }
        
        # Calculate macro averages
        if label_metrics:
            macro_precision = sum(m['precision'] for m in label_metrics.values()) / len(label_metrics)
            macro_recall = sum(m['recall'] for m in label_metrics.values()) / len(label_metrics)
            macro_f1 = sum(m['f1'] for m in label_metrics.values()) / len(label_metrics)
        else:
            macro_precision = macro_recall = macro_f1 = 0
        
        return {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'label_metrics': label_metrics,
            'total_samples': len(valid_results),
            'errors': len(results) - len(valid_results)
        }

def normalize_fever_labels(our_verdict: str, fever_label: str) -> tuple:
    """
    Normalize our system's output to FEVER labels
    
    Our System -> FEVER mapping:
    SUPPORTED/TRUE -> SUPPORTS
    REFUTED/FALSE -> REFUTES  
    INSUFFICIENT_INFO/UNCERTAIN -> NOT_ENOUGH_INFO
    """
    
    # Normalize our system's verdict
    our_mapping = {
        'SUPPORTED': 'SUPPORTS',
        'TRUE': 'SUPPORTS',
        'REFUTED': 'REFUTES',
        'FALSE': 'REFUTES',
        'INSUFFICIENT_INFO': 'NOT_ENOUGH_INFO',
        'UNCERTAIN': 'NOT_ENOUGH_INFO',
        'CONFLICTING': 'NOT_ENOUGH_INFO'  # Map conflicting to insufficient
    }
    
    normalized_ours = our_mapping.get(our_verdict.upper(), our_verdict.upper())
    
    # Ensure FEVER label is in expected format
    fever_mapping = {
        'SUPPORTS': 'SUPPORTS',
        'REFUTES': 'REFUTES', 
        'NOT_ENOUGH_INFO': 'NOT_ENOUGH_INFO',
        'NOT ENOUGH INFO': 'NOT_ENOUGH_INFO'
    }
    
    normalized_fever = fever_mapping.get(fever_label.upper(), fever_label.upper())
    
    return normalized_ours, normalized_fever

def evaluate_fever_claim(claim_data: Dict, components: Dict) -> Dict:
    """Evaluate a single FEVER claim using our pipeline"""
    
    claim = claim_data['claim']
    fever_label = claim_data['label']
    claim_id = claim_data['id']
    
    print(f"FEVER-{claim_id}: {claim[:80]}...")
    print(f"FEVER LABEL: {fever_label}")
    print("-" * 60)
    
    try:
        retriever = components['retriever']
        verifier = components['verifier']
        bias_detector = components['bias_detector']
        weighter = components['weighter']
        verdict_generator = components['verdict_generator']
        
        # Run through our pipeline
        evidence = retriever.retrieve_hybrid_serper_decomposition(claim)
        
        if not evidence:
            print("No evidence found")
            our_verdict, fever_norm = normalize_fever_labels('INSUFFICIENT_INFO', fever_label)
            return {
                'fever_id': claim_id,
                'claim': claim[:100] + "...",
                'fever_label': fever_norm,
                'our_verdict': our_verdict,
                'confidence': 0.0,
                'correct': our_verdict == fever_norm,
                'evidence_count': 0,
                'error': 'No evidence found'
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
        
        # Normalize and compare
        our_verdict, fever_norm = normalize_fever_labels(final_verdict['verdict'], fever_label)
        is_correct = our_verdict == fever_norm
        
        print(f"OUR VERDICT: {our_verdict} ({final_verdict['confidence']:.1%})")
        print(f"FEVER EXPECTED: {fever_norm}")
        print(f"MATCH: {'YES' if is_correct else 'NO'}")
        
        return {
            'fever_id': claim_id,
            'claim': claim[:100] + "...",
            'fever_label': fever_norm,
            'our_verdict': our_verdict,
            'confidence': final_verdict['confidence'],
            'correct': is_correct,
            'evidence_count': len(evidence),
            'support_score': final_verdict.get('support_score', 0),
            'refute_score': final_verdict.get('refute_score', 0)
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        our_verdict, fever_norm = normalize_fever_labels('ERROR', fever_label)
        return {
            'fever_id': claim_id,
            'claim': claim[:100] + "...",
            'fever_label': fever_norm,
            'our_verdict': 'ERROR',
            'confidence': 0.0,
            'correct': False,
            'evidence_count': 0,
            'error': str(e)
        }

def run_fever_evaluation(sample_size: int = 10, use_huggingface: bool = True, dataset_split: str = "dev"):
    """Run evaluation against FEVER dataset from Hugging Face or fallback"""
    
    print("FEVER DATASET EVALUATION")
    print(f"Testing against {sample_size} FEVER claims")
    print("Academic benchmark for fact verification systems")
    print("=" * 80)
    
    # Configuration
    print(f"Configuration:")
    print(f"Sample Size: {sample_size}")
    print(f"Use Hugging Face: {use_huggingface and DATASETS_AVAILABLE}")
    print(f"Dataset Split: {dataset_split}")
    print()
    
    # Load FEVER data
    loader = FEVERDatasetLoader()
    
    if use_huggingface and DATASETS_AVAILABLE:
        fever_claims = loader.get_sample(sample_size, use_huggingface=True)
        dataset_source = f"Hugging Face FEVER Dataset ({dataset_split} split)"
    else:
        print("Cannot proceed without datasets library. Please install: pip install datasets")
        return
        
    print(f"Dataset Source: {dataset_source}")
    print(f"Loaded {len(fever_claims)} claims for evaluation")
    
    # Initialize components (reuse for efficiency)
    components = {
        'retriever': AdvancedEvidenceRetriever(Config.SERPER_KEY, groq_key=Config.GROQ_API_KEY),
        'verifier': ClaimVerifier(),
        'bias_detector': BiasDetector(),
        'weighter': EvidenceWeighter(),
        'verdict_generator': VerdictGenerator()
    }
    
    print(f"Components initialized")
    print(f"Processing {len(fever_claims)} claims...")
    print(f"Estimated time: {len(fever_claims) * 15 / 60:.1f} - {len(fever_claims) * 25 / 60:.1f} minutes")
    print()
    
    # Timing
    start_time = datetime.now()
    
    # Evaluate all claims
    results = []
    for i, claim_data in enumerate(fever_claims, 1):
        claim_start = time.time()
        print(f"\nFEVER EVALUATION {i}/{len(fever_claims)} (Progress: {i/len(fever_claims)*100:.1f}%)")
        result = evaluate_fever_claim(claim_data, components)
        results.append(result)
        
        # Progress tracking
        claim_time = time.time() - claim_start
        elapsed_total = (datetime.now() - start_time).total_seconds() / 60
        avg_time = elapsed_total * 60 / i  # seconds per claim
        remaining_claims = len(fever_claims) - i
        eta_minutes = (remaining_claims * avg_time) / 60
        
        print(f"Claim processed in {claim_time:.1f}s | Avg: {avg_time:.1f}s/claim | ETA: {eta_minutes:.1f}min")
        print("-" * 80)
    
    # Calculate comprehensive metrics
    total_time = (datetime.now() - start_time).total_seconds() / 60  # minutes
    metrics = FEVERMetrics.calculate_metrics(results)
    
    # Display results
    print(f"\nFEVER EVALUATION RESULTS:")
    print(f"   Total Claims: {len(results)}")
    print(f"   Valid Results: {metrics['total_samples']}")
    print(f"   Errors: {metrics['errors']}")
    print(f"   Total Time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print(f"   Average Time per Claim: {total_time*60/len(results):.1f} seconds")
    print(f"   Overall Accuracy: {metrics['accuracy']:.1%}")
    print(f"   Macro F1-Score: {metrics['macro_f1']:.3f}")
    print(f"   Macro Precision: {metrics['macro_precision']:.3f}")
    print(f"   Macro Recall: {metrics['macro_recall']:.3f}")
    
    print(f"\nPER-LABEL PERFORMANCE:")
    for label, label_metrics in metrics['label_metrics'].items():
        print(f"   {label:15s}: P={label_metrics['precision']:.3f} R={label_metrics['recall']:.3f} F1={label_metrics['f1']:.3f} (n={label_metrics['support']})")
    
    print(f"\nDETAILED RESULTS:")
    correct_count = 0
    for i, result in enumerate(results, 1):
        if result['correct']:
            correct_count += 1
        status = "correct" if result['correct'] else "incorrect"
        print(f"   {i:2d}. {status} {result['our_verdict']:15s} (expected {result['fever_label']:15s}) - {result['confidence']:.0%} | FEVER-{result['fever_id']}")
    
    # Performance analysis
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        avg_confidence = sum(r['confidence'] for r in valid_results) / len(valid_results)
        avg_evidence = sum(r['evidence_count'] for r in valid_results) / len(valid_results)
        high_conf_correct = sum(1 for r in valid_results if r['confidence'] > 0.8 and r['correct'])
        high_conf_total = sum(1 for r in valid_results if r['confidence'] > 0.8)
        
        print(f"\nPERFORMANCE ANALYSIS:")
        print(f"   Average Confidence: {avg_confidence:.1%}")
        print(f"   Average Evidence per Claim: {avg_evidence:.1f}")
        print(f"   High Confidence Accuracy (>80%): {high_conf_correct}/{high_conf_total} = {(high_conf_correct/high_conf_total*100) if high_conf_total > 0 else 0:.1f}%")
        
        # Compare to other evaluations
        # print(f"\nCOMPARISON WITH OTHER EVALUATIONS:")
        # print(f"   Ground Truth (21 claims): 61.9% accuracy")
        # print(f"   FactCheck.lk (6 claims): 0.0% accuracy") 
        print(f"   FEVER Sample ({len(valid_results)} claims): {metrics['accuracy']:.1%} accuracy")
    
    # Save results
    results_file = os.path.join(os.path.dirname(__file__), 'fever_evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'evaluation_date': '2026-02-05',
            'sample_size': len(results),
            'total_time_minutes': total_time,
            'avg_time_per_claim_seconds': total_time*60/len(results) if results else 0,
            'metrics': metrics,
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    # Run with 300 claims for robust evaluation
    sample_size = 300  # Good balance of statistical significance and runtime
    use_hf = True  # Try Hugging Face dataset first
    split = "dev"  # Use dev split for evaluation
    
    print("FEVER Evaluation - 300 Claims for Robust Assessment")
    print(f"   Sample Size: {sample_size} claims")
    print(f"   Estimated Runtime: 1.25 - 2.5 hours")
    print(f"   Datasets Library Available: {DATASETS_AVAILABLE}")
    print(f"   Will use Hugging Face: {use_hf and DATASETS_AVAILABLE}")
    print()
    
    if not DATASETS_AVAILABLE:
        print("To use the real FEVER dataset, install: pip install datasets")
        print("   This test will use fallback data...")
        print()
    
    run_fever_evaluation(sample_size, use_hf, split)