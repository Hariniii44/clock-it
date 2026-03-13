"""Build all remaining indices comprehensively"""
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
        logging.FileHandler('index_building.log'),
        logging.StreamHandler()
    ]
)

def build_all_remaining_indices():
    """Build all missing indices in order of priority"""
    
    # Check what already exists
    from config import INDICES_DIR
    existing_indices = set()
    for index_file in INDICES_DIR.glob("*.index"):
        existing_indices.add(index_file.stem)
    
    print(f"Existing indices: {len(existing_indices)}")
    print(f"Existing: {sorted(existing_indices)}")
    
    # Define all datasets in priority order
    all_datasets = [
        # Legal documents - highest priority
        ('acts', 23500, 'Legal statutes and acts'),
        ('supreme_court', 24800, 'Supreme Court judgments'),
        ('bills', 51000, 'Parliamentary bills'),
        ('appeal_court', 73800, 'Appeals court decisions'),
        
        # Recent parliamentary content
        ('hansard_2020s', 53000, 'Recent parliamentary debates'),
        
        # Large comprehensive datasets
        ('news', 164000, 'News coverage'),
        ('extraordinary_gazettes', 312000, 'Official notifications'),
        ('hansard', 354000, 'Complete parliamentary record'),
    ]
    
    # Filter out existing indices
    datasets_to_build = [
        (name, size, desc) for name, size, desc in all_datasets 
        if name not in existing_indices
    ]
    
    if not datasets_to_build:
        print("All indices already exist!")
        return
    
    total_docs = sum(size for _, size, _ in datasets_to_build)
    estimated_hours = total_docs / 12000  # Conservative estimate
    
    print(f"\nDatasets to build: {len(datasets_to_build)}")
    print(f"Total documents: {total_docs:,}")
    print(f"Estimated time: {estimated_hours:.1f} hours")
    print("\nBuild order:")
    
    for i, (name, size, desc) in enumerate(datasets_to_build, 1):
        est_time = size / 200  # docs per minute
        print(f"  {i}. {name}: {size:,} docs ({est_time:.0f} minutes) - {desc}")
    
    print("\nThis is a long-running process. Consider:")
    print("1. Run overnight or during weekend")
    print("2. Ensure stable power/internet connection") 
    print("3. Monitor logs in 'index_building.log'")
    
    response = input("\nProceed with building all indices? (y/N): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Initialize builder
    builder = FAISSIndexBuilder()
    start_time = time.time()
    results = {'successful': [], 'failed': []}
    
    # Build each dataset
    for i, (dataset_name, expected_size, description) in enumerate(datasets_to_build, 1):
        dataset_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"Building {i}/{len(datasets_to_build)}: {dataset_name}")
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
        
        # Progress summary
        total_elapsed = time.time() - start_time
        completed = len(results['successful'])
        remaining = len(datasets_to_build) - i
        
        if completed > 0:
            avg_time_per_dataset = total_elapsed / i
            estimated_remaining = avg_time_per_dataset * remaining
            
            print(f"\nProgress: {i}/{len(datasets_to_build)} datasets processed")
            print(f"Successful: {completed}, Failed: {len(results['failed'])}")
            print(f"Total time so far: {total_elapsed/3600:.1f} hours")
            print(f"Estimated remaining: {estimated_remaining/3600:.1f} hours")
    
    # Final summary
    total_time = time.time() - start_time
    total_successful_docs = sum(docs for _, docs, _ in results['successful'])
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Successful datasets: {len(results['successful'])}/{len(datasets_to_build)}")
    print(f"Total documents indexed: {total_successful_docs:,}")
    
    if results['successful']:
        print(f"\nSuccessful builds:")
        for name, docs, elapsed in results['successful']:
            print(f"  {name}: {docs:,} docs ({elapsed/60:.1f} minutes)")
    
    if results['failed']:
        print(f"\nFailed builds:")
        for name, error in results['failed']:
            print(f"  {name}: {error}")
    
    # Performance stats
    if total_successful_docs > 0:
        overall_rate = total_successful_docs / (total_time / 60)
        print(f"\nOverall processing rate: {overall_rate:.0f} documents/minute")
    
    print(f"\nLog file: index_building.log")
    
    return results

if __name__ == "__main__":
    build_all_remaining_indices()