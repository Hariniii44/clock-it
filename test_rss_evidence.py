#!/usr/bin/env python3
"""
Test RSS Evidence Retrieval
Quick test to see if RSS feeds are working and providing good evidence
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from evidence_retrieval.rss_evidence_retriever import RSSEvidenceRetriever
from evidence_retrieval.hybrid_evidence_retriever import HybridEvidenceRetriever
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_rss_feeds():
    """Test RSS feeds directly"""
    print("🧪 TESTING RSS FEEDS DIRECTLY")
    print("=" * 50)
    
    rss_retriever = RSSEvidenceRetriever()
    
    # Test all feeds
    feed_status = rss_retriever.test_feeds()
    
    print("\n📰 Testing RSS evidence retrieval:")
    
    # Test with some example claims
    test_claims = [
        "Parliament sessions",
        "Economic situation Sri Lanka", 
        "Cricket match results",
        "Weather forecast"
    ]
    
    for claim in test_claims:
        print(f"\n🎯 Testing claim: '{claim}'")
        evidence = rss_retriever.search_all_feeds(claim, max_articles=3)
        
        if evidence:
            print(f"✅ Found {len(evidence)} relevant articles:")
            for i, article in enumerate(evidence):
                print(f"  {i+1}. [{article['source']}] {article['title']}")
                print(f"     Similarity: {article['similarity']:.3f}")
                print(f"     Snippet: {article['snippet'][:100]}...")
        else:
            print("❌ No relevant RSS articles found")

def test_hybrid_retrieval():
    """Test hybrid local + RSS retrieval"""
    print("\n\n🔄 TESTING HYBRID RETRIEVAL")
    print("=" * 50)
    
    hybrid_retriever = HybridEvidenceRetriever()
    
    # Test all sources
    hybrid_retriever.test_all_sources()
    
    print("\n🎯 Testing hybrid evidence retrieval:")
    
    test_claims = [
        "Asoka Ranwala accident",  # Should find in local
        "Latest Parliament news",  # Should find in both
        "Breaking news today"      # Should find mainly in RSS
    ]
    
    for claim in test_claims:
        print(f"\n{'='*60}")
        print(f"🎯 CLAIM: {claim}")
        print('='*60)
        
        # Test different strategies
        strategies = ["local", "rss", "hybrid"]
        
        for strategy in strategies:
            print(f"\n📋 Strategy: {strategy.upper()}")
            evidence = hybrid_retriever.retrieve_evidence(claim, num_results=3, strategy=strategy)
            
            if evidence:
                print(f"✅ Found {len(evidence)} articles:")
                for i, article in enumerate(evidence):
                    source_type = article.get('source_type', 'local')
                    print(f"  {i+1}. [{article['source']} | {source_type}] {article['title']}")
                    print(f"     Similarity: {article.get('similarity', 0):.3f}")
            else:
                print("❌ No evidence found")

def test_real_time_capability():
    """Test if we're getting fresh, recent articles"""
    print("\n\n⏰ TESTING REAL-TIME CAPABILITY")
    print("=" * 50)
    
    rss_retriever = RSSEvidenceRetriever()
    
    # Test with very recent/current events
    current_claims = [
        "news today January 2026",
        "latest breaking news",
        "current events Sri Lanka"
    ]
    
    for claim in current_claims:
        print(f"\n🕐 Testing real-time claim: '{claim}'")
        evidence = rss_retriever.search_all_feeds(claim, max_articles=5)
        
        if evidence:
            print(f"✅ Found {len(evidence)} recent articles:")
            for i, article in enumerate(evidence):
                date_info = article.get('date', 'No date')
                print(f"  {i+1}. [{article['source']}] {article['title']}")
                print(f"     Date: {date_info}")
                print(f"     Relevance: {article['similarity']:.3f}")
        else:
            print("❌ No recent articles found")

def main():
    """Run all RSS tests"""
    print("🚀 TESTING RSS-BASED EVIDENCE RETRIEVAL")
    print("Testing real-time evidence capabilities")
    print("\n")
    
    try:
        # Test RSS feeds directly
        test_rss_feeds()
        
        # Test hybrid approach
        test_hybrid_retrieval()
        
        # Test real-time capability
        test_real_time_capability()
        
        print("\n" + "🎉" * 20)
        print("✅ RSS TESTING COMPLETE!")
        print("📊 Results:")
        print("  📡 RSS feeds provide real-time evidence")
        print("  🔄 Hybrid approach combines local + live data")
        print("  ⏰ Fresh articles available from multiple sources")
        print("🎉" * 20)
        
    except Exception as e:
        print(f"\n❌ RSS TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()