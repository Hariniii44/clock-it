"""Test FAISS index building"""
import sys
sys.path.append('src')
sys.path.append('.')

def test_index_builder():
    """Test index building on one dataset"""
    try:
        from src.retrieval.faiss_indexer import FAISSIndexBuilder
        from config import DATASETS_DIR
        
        # Check available datasets
        available_datasets = []
        for dataset_dir in DATASETS_DIR.iterdir():
            if dataset_dir.is_dir():
                available_datasets.append(dataset_dir.name)
        
        if not available_datasets:
            print("❌ No datasets found!")
            return False
        
        print(f"✅ Found datasets: {available_datasets}")
        
        # Test with first available (likely cabinet_decisions)
        test_dataset = available_datasets[0]
        print(f"\n🧪 Testing index building with: {test_dataset}")
        
        builder = FAISSIndexBuilder()
        dataset_path = DATASETS_DIR / test_dataset
        
        result = builder.build_index_for_dataset(dataset_path, test_dataset)
        
        if result:
            print(f"\n✅ Index building test successful!")
            print(f"📊 Indexed {result['total_documents']:,} documents")
            print(f"📁 Index saved to: {result['index_path']}")
            return True
        else:
            print(f"\n❌ Index building failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_index_builder()
    
    if success:
        print(f"\n🎉 Ready to build all indices!")
        print(f"📋 Run: python scripts/build_indices.py")
    else:
        print(f"\n❌ Fix issues before proceeding.")