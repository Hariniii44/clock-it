"""Build indices for priority datasets first"""
import sys
sys.path.append('src')
sys.path.append('.')

from src.retrieval.faiss_indexer import FAISSIndexBuilder

def build_priority_indices():
    """Build indices for smaller datasets first"""
    
    # Start with manageable datasets
    priority_datasets = [
        'cabinet_decisions',           # 10,518
        'pmd_press_releases',         # 3,382  
        'treasury_press_releases',    # 210
        'police_press_releases',      # 2,190
        'education_publications',     # 8,311
        'hansard_2010s',              # 1,087  (smallest hansard)
    ]
    
    print(f"🏗️  BUILDING PRIORITY INDICES")
    print(f"=" * 60)
    
    builder = FAISSIndexBuilder()
    results = builder.build_all_indices(dataset_names=priority_datasets)
    
    return results

if __name__ == "__main__":
    results = build_priority_indices()
    
    if len(results['successful']) >= 4:
        print(f"\n🎉 Priority indices built successfully!")
        print(f"📋 Ready to test retrieval system.")
        print(f"\n💡 To build remaining large indices later:")
        print(f"   python scripts/build_indices.py")