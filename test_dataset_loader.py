"""Test dataset loader functionality"""
import sys
sys.path.append('src')
sys.path.append('.')

def test_loader_setup():
    """Test that loader can be imported and initialized"""
    try:
        from src.retrieval.dataset_loader import DatasetLoader
        from config import DATASETS_CONFIG
        
        print("✅ Dataset loader imported successfully")
        
        loader = DatasetLoader()
        print(f"✅ Dataset loader initialized")
        print(f"📊 Configured datasets: {len(DATASETS_CONFIG)}")
        
        # Test status check
        status = loader.get_dataset_status()
        print(f"✅ Status check works")
        
        existing_count = sum(1 for info in status.values() if info['exists'])
        print(f"📋 Currently available: {existing_count}/{len(status)} datasets")
        
        return True
        
    except Exception as e:
        print(f"❌ Loader test failed: {e}")
        return False

def test_single_download():
    """Test downloading a single small dataset"""
    try:
        from src.retrieval.dataset_loader import DatasetLoader
        
        loader = DatasetLoader()
        
        # Try to download cabinet decisions (usually smaller)
        print("\n🧪 Testing single dataset download...")
        print("Will download: cabinet_decisions (if not exists)")
        
        success, info = loader.download_dataset('cabinet_decisions', force_reload=False)
        
        if success:
            print(f"✅ Download test passed!")
            print(f"📊 Documents: {info.get('documents', 'Unknown')}")
            return True
        else:
            print(f"❌ Download failed: {info}")
            return False
            
    except Exception as e:
        print(f"❌ Download test failed: {e}")
        return False

def _load_and_validate_dataset(self, save_path):
    """Load dataset and validate it properly"""
    try:
        dataset = load_from_disk(str(save_path))
        
        # Handle different dataset structures
        if hasattr(dataset, 'keys'):
            # Multi-split dataset (has 'train', 'test', etc.)
            if 'train' in dataset:
                data = dataset['train']
            elif len(dataset.keys()) == 1:
                # Single split with custom name
                split_name = list(dataset.keys())[0]
                data = dataset[split_name]
            else:
                # Multiple splits - combine or use largest
                largest_split = max(dataset.keys(), key=lambda k: len(dataset[k]))
                data = dataset[largest_split]
                print(f"   📊 Using '{largest_split}' split (largest)")
        else:
            # Single dataset without splits
            data = dataset
        
        doc_count = len(data)
        
        # Validate we have reasonable data
        if doc_count == 0:
            return None, "Empty dataset"
        
        if doc_count == 1:
            # Check if it's actually loaded properly
            sample = data[0]
            if not sample or len(str(sample)) < 100:
                return None, "Dataset appears corrupted (single tiny record)"
        
        return data, doc_count
        
    except Exception as e:
        return None, f"Load error: {str(e)}"

def main():
    """Run loader tests"""
    print("🧪 TESTING DATASET LOADER")
    print("="*40)
    
    # Test 1: Basic setup
    test1 = test_loader_setup()
    
    if test1:
        print("\n" + "="*40)
        answer = input("Test single dataset download? (y/N): ").strip().lower()
        
        if answer == 'y':
            # Test 2: Single download
            test2 = test_single_download()
            
            if test2:
                print(f"\n🎉 All tests passed!")
                print(f"📋 Ready to run: python scripts/1_download_datasets.py")
            else:
                print(f"\n⚠️ Download test failed, but basic setup works.")
        else:
            print(f"\n✅ Basic setup test passed!")
            print(f"📋 Ready to run: python scripts/1_download_datasets.py")
    else:
        print(f"\n❌ Setup test failed. Check configuration.")

if __name__ == "__main__":
    main()