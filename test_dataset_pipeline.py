"""Test enhanced configuration setup"""
import sys
sys.path.append('.')

def test_existing_config():
    """Test existing configuration still works"""
    try:
        from config import Config
        
        print("🔍 Testing existing configuration...")
        
        # Test existing functionality
        serper_key = Config.SERPER_KEY
        groq_key = Config.GROQ_API_KEY
        num_sources = Config.NUM_EVIDENCE_SOURCES
        
        print(f"  ✅ Serper Key: {'Set' if serper_key else 'Not Set'}")
        print(f"  ✅ Groq Key: {'Set' if groq_key else 'Not Set'}")
        print(f"  ✅ Evidence Sources: {num_sources}")
        
        # Test validation (if you want)
        try:
            Config.validate()
            print("  ✅ Validation: Passed")
        except ValueError as e:
            print(f"  ⚠️ Validation: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Existing config error: {e}")
        return False

def test_new_config():
    """Test new dataset configuration"""
    try:
        from config import (
            PROJECT_ROOT, DATA_DIR, INDICES_DIR, DATASETS_DIR,
            DATASETS_CONFIG, EMBEDDING_MODEL, USE_HYBRID_RETRIEVAL,
            get_datasets_for_claim_type, print_config
        )
        
        print("\n🔍 Testing new dataset configuration...")
        
        print(f"  ✅ Project root: {PROJECT_ROOT}")
        print(f"  ✅ Data directory: {DATA_DIR}")
        print(f"  ✅ Indices directory: {INDICES_DIR}")
        print(f"  ✅ Datasets directory: {DATASETS_DIR}")
        print(f"  ✅ Embedding model: {EMBEDDING_MODEL}")
        print(f"  ✅ Hybrid retrieval: {USE_HYBRID_RETRIEVAL}")
        print(f"  ✅ Available datasets: {len(DATASETS_CONFIG)}")
        
        # Test dataset routing
        print("\n🔍 Testing dataset routing...")
        presidential_datasets = get_datasets_for_claim_type(['presidential'])
        economic_datasets = get_datasets_for_claim_type(['economic'])
        
        print(f"  ✅ Presidential claims route to: {presidential_datasets}")
        print(f"  ✅ Economic claims route to: {economic_datasets}")
        
        return True
        
    except Exception as e:
        print(f"❌ New config error: {e}")
        return False

def test_compatibility():
    """Test backward compatibility functions"""
    try:
        from config import get_serper_key, get_groq_key, get_num_evidence_sources
        
        print("\n🔍 Testing backward compatibility...")
        
        serper = get_serper_key()
        groq = get_groq_key()
        num_sources = get_num_evidence_sources()
        
        print(f"  ✅ get_serper_key(): {'Works' if serper is not None else 'Returns None'}")
        print(f"  ✅ get_groq_key(): {'Works' if groq is not None else 'Returns None'}")
        print(f"  ✅ get_num_evidence_sources(): {num_sources}")
        
        return True
        
    except Exception as e:
        print(f"❌ Compatibility error: {e}")
        return False

def main():
    """Run all configuration tests"""
    print("🚀 TESTING ENHANCED CONFIGURATION")
    print("="*60)
    
    # Test existing functionality
    test1_passed = test_existing_config()
    
    # Test new functionality  
    test2_passed = test_new_config()
    
    # Test compatibility
    test3_passed = test_compatibility()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Existing Config: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"New Dataset Config: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Backward Compatibility: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 ALL TESTS PASSED! Configuration is ready.")
        
        # Show full config
        print("\n" + "="*60)
        print("FULL CONFIGURATION OVERVIEW")
        print("="*60)
        from config import print_config
        print_config()
        
        return True
    else:
        print("\n❌ Some tests failed. Please fix errors before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Ready for Step 2: Dataset Download!")
    else:
        print("\n❌ Please fix configuration issues first.")