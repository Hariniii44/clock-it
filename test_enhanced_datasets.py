#!/usr/bin/env python3
"""
Test script for the enhanced dataset manager with PMD datasets
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import after path setup
try:
    from src.database_retrieval.dataset_manager import DatasetManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import...")
    try:
        # Alternative import path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dataset_manager", 
            "src/database_retrieval/dataset_manager.py"
        )
        dataset_manager = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset_manager)
        DatasetManager = dataset_manager.DatasetManager
        print("Successfully imported DatasetManager")
    except Exception as e2:
        print(f"Failed to import DatasetManager: {e2}")
        sys.exit(1)

def test_enhanced_dataset_manager():
    """Test the enhanced dataset manager with PMD datasets"""
    print("Testing Enhanced Dataset Manager with PMD Integration")
    print("="*70)
    
    try:
        # Initialize manager
        manager = DatasetManager()
        print("Dataset manager initialized")
        
        # Test 1: Claim routing
        print("\nSTEP 1: Testing Claim Routing Logic")
        test_claim = "President said giving rice to animals causes shortage"
        
        claim_types = manager.classify_claim_type(test_claim)
        routed_datasets = manager.route_datasets_for_claim(test_claim)
        
        print(f"   Claim: '{test_claim}'")
        print(f"   Detected types: {claim_types}")
        print(f"   Routed to datasets: {routed_datasets[:5]}")
        
        # Test 2: Try to load PMD datasets
        print("\nSTEP 2: Loading PMD Datasets")
        
        priority_datasets = ['pmd_press_releases', 'pmd_press_release']
        loaded_count = 0
        
        for dataset_key in priority_datasets:
            try:
                print(f"   Loading {dataset_key}...")
                dataset = manager.download_dataset(dataset_key)
                if len(dataset) > 0:
                    loaded_count += 1
                    print(f"   {dataset_key}: {len(dataset)} documents loaded")
                    # Show sample
                    sample_title = dataset['title'].iloc[0][:80]
                    print(f"      Sample: {sample_title}...")
                else:
                    print(f"   {dataset_key}: No documents found")
                    
            except Exception as e:
                print(f"   Failed to load {dataset_key}: {str(e)[:100]}...")
        
        # Test 3: Search functionality
        if loaded_count > 0:
            print(f"\nSTEP 3: Testing Search with Claim Routing")
            
            # Test routed search
            results = manager.search_datasets(
                "rice animals shortage", 
                max_results=5, 
                claim=test_claim
            )
            
            print(f"   Found {len(results)} results with claim routing:")
            for i, result in enumerate(results[:3], 1):
                dataset_source = result['dataset_source'].replace('_', ' ').title()
                print(f"   {i}. [{dataset_source}] {result['title'][:60]}...")
                print(f"      Authority: {result['authority_weight']:.2f} | Score: {result.get('final_score', 0):.3f}")
            
            # Test semantic search setup
            print(f"\nSTEP 4: Testing Semantic Search (Optional)")
            try:
                if manager.initialize_semantic_search():
                    print("   Semantic search model loaded")
                    
                    # Try to build embeddings for one dataset
                    for dataset_key in manager.get_all_datasets().keys():
                        if manager.build_embeddings_for_dataset(dataset_key):
                            print(f"   Built embeddings for {dataset_key}")
                            
                            # Test semantic search
                            semantic_results = manager.semantic_search_datasets(
                                "government agricultural policy",
                                max_results=2,
                                claim=test_claim
                            )
                            
                            if semantic_results:
                                print(f"   Semantic search found {len(semantic_results)} results")
                                for result in semantic_results:
                                    sim_score = result['similarity_score']
                                    print(f"      {result['title'][:50]}... (similarity: {sim_score:.3f})")
                            break
                else:
                    print("   WARNING: Semantic search model not available")
                    
            except Exception as e:
                print(f"   Semantic search failed: {e}")
        
        else:
            print("\nERROR: No datasets loaded successfully")
        
        # Summary
        print(f"\nSUMMARY")
        print(f"   Datasets loaded: {loaded_count}")
        total_docs = sum(len(df) for df in manager.get_all_datasets().values())
        print(f"   Total documents: {total_docs}")
        print(f"   Available datasets: {list(manager.get_all_datasets().keys())}")
        
        if loaded_count > 0:
            print("\nSUCCESS: Enhanced dataset manager is working!")
            print("   PMD datasets can now handle 'President said X' claims")
            print("   Claim routing directs searches to appropriate datasets")
            print("   Semantic search available (if sentence-transformers installed)")
        else:
            print("\nWARNING: No PMD datasets loaded")
            print("   Check internet connection and dataset availability")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Test failed with: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_dataset_manager()
    
    if success:
        print(f"\nNEXT STEPS:")
        print(f"   1. Update test_pipeline.py to use retrieve_evidence_dataset_first()")
        print(f"   2. Test with: 'President said giving rice to animals causes shortage'")
        print(f"   3. Install sentence-transformers: pip install sentence-transformers")
        print(f"   4. Compare results with Google AI Overview!")
    else:
        print(f"\nTROUBLESHOOTING:")
        print(f"   1. Check internet connection")
        print(f"   2. Verify HuggingFace datasets are accessible")
        print(f"   3. Check if PMD datasets exist at nuuuwan/lk-pmd-*")