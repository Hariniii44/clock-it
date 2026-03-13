"""Build priority indices for legal and parliamentary content"""
import sys
sys.path.append('src')
sys.path.append('.')
from src.retrieval.faiss_indexer import FAISSIndexBuilder
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('priority_index_building.log'),
        logging.StreamHandler()
    ]
)

def build_priority_indices_only():
    """Build only the highest priority datasets"""
    
    priority_datasets = [
        ('acts', 23500, 'Legal statutes - critical for legal claims'),
        ('supreme_court', 24800, 'Supreme Court judgments - legal precedents'),
        ('bills', 51000, 'Parliamentary bills - legislative content'),
        ('hansard_2020s', 53000, 'Recent parliamentary debates'),
    ]
    
    # Check existing
    from config import INDICES_DIR
    existing = {f.stem for f in     priority_datasets = [
        ('acts', 23500, 'Legal statutes - critical for legal claims'),
        ('supreme_court', 24800, 'Supreme Court judgments - legal precedents'),
        # ('bills', 51000, 'Parliamentary bills - legislative content'),
        # ('hansard_2020s', 53000, 'Recent parliamentary debates'),
    ].glob("*.index")}
    
    to_build = [(name, size, desc) for name, size, desc in priority_datasets 
                if name not in existing]
    
    if not to_build:
        print("All priority indices already exist!")
        return
    
    total_docs = sum(size for _, size, _ in to_build)
    estimated_time = total_docs / 200  # minutes
    
    print(f"Priority datasets to build: {len(to_build)}")
    print(f"Total documents: {total_docs:,}")
    print(f"Estimated time: {estimated_time/60:.1f} hours")
    
    print("\nDatasets to build:")
    for i, (name, size, desc) in enumerate(to_build, 1):
        est_minutes = size / 200
        print(f"  {i}. {name}: {size:,} docs (~{est_minutes/60:.1f}h) - {desc}")
    
    print("\nThese datasets are critical for legal claim fact-checking.")
    print("After building, your budget approval claim should find actual legal documents.")
    
    response = input("\nProceed with building priority indices? (y/N): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Initialize builder
    builder = FAISSIndexBuilder()
    start_time = time.time()
    results = {'successful': [], 'failed': []}
    
    # Build each dataset
    for i, (dataset_name, expected_size, description) in enumerate(to_build, 1):
        dataset_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Building {i}/{len(to_build)}: {dataset_name}")
        print(f"Expected: {expected_size:,} documents")
        print(f"Description: {description}")
        print(f"{'='*60}")
        
        try:
            from config import DATASETS_DIR
            dataset_path = DATASETS_DIR / dataset_name
            
            if not dataset_path.exists():
                print(f"ERROR: Dataset path not found: {dataset_path}")
                results['failed'].append((dataset_name, "Path not found"))
                continue
            
            # Build the index
            result = builder.build_index_for_dataset(dataset_path, dataset_name)
            
            dataset_elapsed = time.time() - dataset_start
            
            if result:
                actual_docs = result.get('total_documents', 0)
                print(f"SUCCESS: {dataset_name}")
                print(f"  Documents indexed: {actual_docs:,}")
                print(f"  Time taken: {dataset_elapsed/60:.1f} minutes")
                print(f"  Rate: {actual_docs/(dataset_elapsed/60):.0f} docs/minute")
                
                results['successful'].append((dataset_name, actual_docs, dataset_elapsed))
                logging.info(f"Completed {dataset_name}: {actual_docs:,} docs in {dataset_elapsed/60:.1f} minutes")
                
            else:
                print(f"FAILED: {dataset_name} - No result returned")
                results['failed'].append((dataset_name, "No result returned"))
                logging.error(f"Failed {dataset_name}: No result returned")
                
        except Exception as e:
            dataset_elapsed = time.time() - dataset_start
            print(f"ERROR: {dataset_name} failed after {dataset_elapsed/60:.1f} minutes")
            print(f"Error: {str(e)}")
            
            results['failed'].append((dataset_name, str(e)))
            logging.error(f"Failed {dataset_name}: {str(e)}")
        
        # Progress summary after each dataset
        total_elapsed = time.time() - start_time
        completed = len(results['successful'])
        remaining = len(to_build) - i
        
        print(f"\nProgress: {i}/{len(to_build)} datasets processed")
        print(f"Successful: {completed}, Failed: {len(results['failed'])}")
        print(f"Total time so far: {total_elapsed/3600:.1f} hours")
        
        if i > 0 and remaining > 0:
            avg_time_per_dataset = total_elapsed / i
            estimated_remaining = avg_time_per_dataset * remaining
            print(f"Estimated remaining: {estimated_remaining/3600:.1f} hours")
    
    # Final summary
    total_time = time.time() - start_time
    total_successful_docs = sum(docs for _, docs, _ in results['successful'])
    
    print(f"\n{'='*60}")
    print("PRIORITY BUILD SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Successful datasets: {len(results['successful'])}/{len(to_build)}")
    print(f"Total documents indexed: {total_successful_docs:,}")
    
    if results['successful']:
        print(f"\nSuccessful builds:")
        for name, docs, elapsed in results['successful']:
            rate = docs / (elapsed / 60) if elapsed > 0 else 0
            print(f"  {name}: {docs:,} docs ({elapsed/60:.1f} min, {rate:.0f} docs/min)")
    
    if results['failed']:
        print(f"\nFailed builds:")
        for name, error in results['failed']:
            print(f"  {name}: {error}")
    
    # Next steps
    if len(results['successful']) >= 2:
        print(f"\nNEXT STEPS:")
        print("1. Test your budget approval claim now:")
        print("   python test_gemini_bias_pipeline.py")
        print("2. You should now get legal documents instead of just procedural docs")
        print("3. The system should find Appropriation Acts and legal statutes")
        
        if 'acts' in [name for name, _, _ in results['successful']]:
            print("4. Acts dataset built - legal statutes now available!")
        if 'supreme_court' in [name for name, _, _ in results['successful']]:
            print("5. Supreme Court dataset built - legal precedents available!")
    
    print(f"\nLog file: priority_index_building.log")
    return results

if __name__ == "__main__":
    build_priority_indices_only()