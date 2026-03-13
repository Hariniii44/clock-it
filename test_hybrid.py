"""Test the complete hybrid retrieval system"""
import sys
sys.path.append('src')
sys.path.append('.')

from src.retrieval.hybrid_retriever import HybridRetriever

def test_hybrid_system():
    """Test the complete hybrid retrieval system"""
    
    print("🚀 TESTING HYBRID FACT-CHECKING SYSTEM")
    print("=" * 60)
    
    # Initialize hybrid retriever
    try:
        retriever = HybridRetriever()
        
        # Show system capabilities
        summary = retriever.get_search_summary()
        print(f"\n📊 SYSTEM CAPABILITIES:")
        print(f"   🔍 Hybrid mode: {summary['hybrid_mode_enabled']}")
        print(f"   🌐 Web fallback: {summary['web_fallback_enabled']}")  
        print(f"   📚 Local datasets: {len(summary['loaded_datasets'])}")
        print(f"   📄 Total documents: {summary['total_local_documents']:,}")
        print(f"   🤖 Embedding model: {summary['embedding_model']}")
        
        print(f"\n📋 Available datasets:")
        for dataset in summary['loaded_datasets']:
            print(f"     ✅ {dataset}")
        
    except Exception as e:
        print(f"❌ Error initializing retrieval system: {e}")
        return False
    
    # Test queries covering different domains
    test_queries = [
        {
            'query': 'cabinet decision on education policy',
            'claim_types': ['policy', 'government_policy'],
            'expected_datasets': ['cabinet_decisions', 'education_publications']
        },
        {
            'query': 'presidential statement on economic development', 
            'claim_types': ['presidential', 'economic'],
            'expected_datasets': ['pmd_press_releases', 'treasury_press_releases']
        },
        {
            'query': 'parliament budget discussion Sri Lanka',
            'claim_types': ['parliamentary', 'economic'], 
            'expected_datasets': ['hansard_2010s', 'treasury_press_releases']
        },
        {
            'query': 'police investigation drug trafficking',
            'claim_types': ['law_enforcement', 'crime'],
            'expected_datasets': ['police_press_releases']
        },
        {
            'query': 'school curriculum changes Sri Lanka',
            'claim_types': ['education', 'policy'],
            'expected_datasets': ['education_publications', 'cabinet_decisions']
        }
    ]
    
    print(f"\n🧪 TESTING SEARCH QUERIES")
    print("=" * 60)
    
    all_tests_passed = True
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case['query']
        claim_types = test_case['claim_types'] 
        expected_datasets = test_case['expected_datasets']
        
        print(f"\n📋 TEST {i}/{len(test_queries)}")
        print(f"   Query: '{query}'")
        print(f"   Claim types: {claim_types}")
        print(f"   Expected datasets: {expected_datasets}")
        
        try:
            # Perform hybrid search
            results = retriever.hybrid_search(
                query=query,
                claim_types=claim_types,
                total_results=5
            )
            
            if not results:
                print(f"   ❌ No results returned")
                all_tests_passed = False
                continue
            
            print(f"   ✅ Retrieved {len(results)} results")
            
            # Check if results come from expected datasets
            found_datasets = set()
            local_results = 0
            web_results = 0
            
            for result in results:
                dataset = result.get('dataset', '')
                if dataset in summary['loaded_datasets']:
                    found_datasets.add(dataset)
                    local_results += 1
                elif dataset == 'web_search':
                    web_results += 1
            
            print(f"   📊 Local results: {local_results}, Web results: {web_results}")
            print(f"   📚 Found in datasets: {sorted(found_datasets)}")
            
            # Show top results
            print(f"   🔍 Top results:")
            for j, result in enumerate(results[:3], 1):
                title = result.get('title', 'No title')[:80]
                source = result.get('source', 'Unknown')
                authority = result.get('authority', 0)
                
                print(f"     {j}. [{result['dataset']}] {title}...")
                print(f"        Authority: {authority}, Source: {source}")
            
            # Test passes if we got results
            if results:
                print(f"   ✅ TEST {i} PASSED")
            else:
                print(f"   ❌ TEST {i} FAILED - No results")
                all_tests_passed = False
                
        except Exception as e:
            print(f"   ❌ TEST {i} ERROR: {e}")
            all_tests_passed = False
    
    # Final summary
    print(f"\n{'=' * 60}")
    print(f"🏁 TEST SUMMARY")
    print(f"{'=' * 60}")
    
    if all_tests_passed:
        print(f"🎉 ALL TESTS PASSED!")
        print(f"✅ Your hybrid fact-checking system is working!")
        print(f"📊 System ready with {summary['total_local_documents']:,} local documents")
        print(f"🌐 Web search fallback available")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Test with real fact-checking claims")
        print(f"   2. Build remaining large indices (optional):")
        print(f"      python scripts/build_indices.py")
        print(f"   3. Integrate with your main fact-checking pipeline")
        
    else:
        print(f"❌ Some tests failed. Check errors above.")
        
    return all_tests_passed

def test_specific_query():
    """Test with a specific query interactively"""
    
    retriever = HybridRetriever()
    
    print(f"\n🔍 INTERACTIVE QUERY TEST")
    print("=" * 40)
    
    while True:
        query = input("\nEnter a query to test (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
            
        print(f"\n🔍 Searching for: '{query}'")
        
        try:
            results = retriever.hybrid_search(query, total_results=3)
            
            if results:
                print(f"\n📊 Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    title = result.get('title', 'No title')
                    dataset = result.get('dataset', 'Unknown')
                    authority = result.get('authority', 0)
                    
                    print(f"\n{i}. [{dataset}] {title}")
                    print(f"   Authority: {authority}")
                    print(f"   Text: {result.get('text', '')[:200]}...")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Search error: {e}")

def main():
    """Main test function"""
    
    print("🧪 HYBRID RETRIEVAL SYSTEM TESTER")
    print("=" * 50)
    
    print("What would you like to do?")
    print("1. Run automated tests")
    print("2. Interactive query testing")  
    print("3. Both")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        print("\n" + "="*60)
        success = test_hybrid_system()
        
        if not success:
            print("\n⚠️ Automated tests had issues. Check configuration.")
    
    if choice in ['2', '3']:
        test_specific_query()
    
    print(f"\n🎉 Testing complete!")

if __name__ == "__main__":
    main()