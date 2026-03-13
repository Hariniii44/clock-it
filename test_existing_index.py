# Add this to test_index_builder.py
def check_existing_indices():
    """Check what indices already exist"""
    from config import INDICES_DIR
    
    print("🔍 EXISTING INDICES:")
    existing = []
    for index_file in INDICES_DIR.glob("*.index"):
        dataset_name = index_file.stem
        metadata_file = INDICES_DIR / f"{dataset_name}.metadata.pkl"
        
        if metadata_file.exists():
            try:
                import pickle
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                doc_count = metadata.get('total_documents', 0)
                print(f"   ✅ {dataset_name}: {doc_count:,} documents")
                existing.append(dataset_name)
            except:
                print(f"   ⚠️ {dataset_name}: Corrupted metadata")
        else:
            print(f"   ❌ {dataset_name}: Missing metadata")
    
    print(f"\nTotal indices: {len(existing)}")
    return existing

if __name__ == "__main__":
    check_existing_indices()