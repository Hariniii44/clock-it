"""Build FAISS indices for all datasets"""
import sys
sys.path.append('src')
sys.path.append('.')

from src.retrieval.faiss_indexer import FAISSIndexBuilder
from config import DATASETS_DIR, INDICES_DIR

def main():
    """Build all indices"""
    print("🏗️  FAISS INDEX BUILDER")
    print("="*40)
    
    # Check if datasets exist
    available_datasets = []
    for dataset_dir in DATASETS_DIR.iterdir():
        if dataset_dir.is_dir():
            available_datasets.append(dataset_dir.name)
    
    if not available_datasets:
        print("❌ No datasets found!")
        print("📋 First run: python scripts/download_datasets.py --mode priority")
        return
    
    print(f"📊 Found {len(available_datasets)} datasets:")
    for name in available_datasets:
        print(f"   - {name}")
    
    print(f"📁 Indices will be saved to: {INDICES_DIR}")
    
    # Build indices
    builder = FAISSIndexBuilder()
    results = builder.build_all_indices()
    
    # Next steps
    successful = len(results['successful'])
    if successful > 0:
        print(f"\n🎉 SUCCESS! Built {successful} search indices.")
        print(f"📋 Next step: python scripts/test_retrieval.py")
    else:
        print(f"\n❌ No indices built successfully.")
        print(f"📋 Check errors above and retry.")

if __name__ == "__main__":
    main()