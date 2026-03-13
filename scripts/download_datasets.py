"""
Download Nuwan's datasets from Hugging Face
Run this script to download all Sri Lankan government datasets
"""
import sys
sys.path.append('src')
sys.path.append('.')

from src.retrieval.dataset_loader import DatasetLoader
from config import DATASETS_CONFIG, print_config
import argparse

def download_priority_datasets():
    """Download high-priority datasets first (for testing)"""
    priority_datasets = [
        'pmd_press_releases',  # Presidential statements
        'cabinet_decisions',   # Government decisions  
        'hansard'             # Parliamentary debates
    ]
    
    loader = DatasetLoader()
    
    print("🚀 DOWNLOADING PRIORITY DATASETS")
    print("="*60)
    print("This will download the most important datasets first")
    print(f"Priority datasets: {priority_datasets}")
    
    results = {'successful': [], 'failed': [], 'already_existed': [], 'total_documents': 0}
    
    for i, dataset_name in enumerate(priority_datasets, 1):
        if dataset_name not in DATASETS_CONFIG:
            print(f"⚠️ Skipping {dataset_name} - not in config")
            continue
            
        print(f"\n📋 PRIORITY {i}/{len(priority_datasets)}: {dataset_name}")
        
        try:
            success, info = loader.download_dataset(dataset_name, force_reload=False)
            
            if success:
                if info['status'] == 'already_exists':
                    results['already_existed'].append(info)
                else:
                    results['successful'].append(info)
                results['total_documents'] += info.get('documents', 0)
            else:
                results['failed'].append(info)
                
        except Exception as e:
            print(f"❌ Error with {dataset_name}: {e}")
            results['failed'].append({
                'name': dataset_name,
                'error': str(e),
                'status': 'failed'
            })
    
    # Print summary
    loader._print_download_summary(results)
    
    return results

def download_all_datasets():
    """Download all configured datasets"""
    loader = DatasetLoader()
    
    print("🚀 DOWNLOADING ALL DATASETS")
    print("="*60)
    
    results = loader.download_all_datasets(force_reload=False, skip_on_error=True)
    
    return results

def main():
    """Main download script"""
    parser = argparse.ArgumentParser(description='Download Sri Lankan government datasets')
    parser.add_argument('--mode', choices=['priority', 'all', 'status'], default='priority',
                        help='Download mode: priority (key datasets), all (everything), status (check only)')
    parser.add_argument('--force', action='store_true', 
                        help='Force re-download existing datasets')
    
    args = parser.parse_args()
    
    # Show configuration
    print_config()
    
    # Execute based on mode
    if args.mode == 'status':
        loader = DatasetLoader()
        status = loader.get_dataset_status()
        
        print("\n📋 DATASET STATUS:")
        print("="*40)
        for name, info in status.items():
            if info['exists']:
                print(f"✅ {name}: {info['documents']:,} documents")
            else:
                print(f"❌ {name}: {info.get('error', 'Not found')}")
        
        # Summary
        existing = sum(1 for info in status.values() if info['exists'])
        total = len(status)
        print(f"\n📊 {existing}/{total} datasets available locally")
        
    elif args.mode == 'priority':
        if args.force:
            print("⚠️ Force mode not supported for priority downloads")
        
        results = download_priority_datasets()
        
        # Next steps
        successful = len(results['successful'])
        existed = len(results['already_existed'])
        
        if successful + existed >= 2:  # At least 2 datasets
            print(f"\n🎉 SUCCESS! You have enough datasets to proceed.")
            print(f"📋 Next step: python scripts/2_build_indices.py")
        else:
            print(f"\n⚠️ Only {successful + existed} datasets available.")
            print(f"📋 Consider running: python scripts/1_download_datasets.py --mode all")
        
    elif args.mode == 'all':
        loader = DatasetLoader()
        results = loader.download_all_datasets(force_reload=args.force, skip_on_error=True)
        
        # Next steps
        successful = len(results['successful'])
        existed = len(results['already_existed'])
        total_ready = successful + existed
        
        if total_ready >= 5:
            print(f"\n🎉 EXCELLENT! {total_ready} datasets ready.")
            print(f"📋 Next step: python scripts/2_build_indices.py")
        elif total_ready >= 3:
            print(f"\n✅ GOOD! {total_ready} datasets ready - enough to proceed.")
            print(f"📋 Next step: python scripts/2_build_indices.py")
        else:
            print(f"\n⚠️ Only {total_ready} datasets ready. Check errors above.")

if __name__ == "__main__":
    main()