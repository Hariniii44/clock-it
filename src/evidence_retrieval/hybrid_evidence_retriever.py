#!/usr/bin/env python3
"""
Hybrid Evidence Retrieval System
Combines local scraped articles with real-time RSS feeds
"""

import sys
import os
from typing import List, Dict, Optional

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, src_path)

from evidence_retrieval.local_evidence_retriever import LocalEvidenceRetriever
from evidence_retrieval.rss_evidence_retriever import RSSEvidenceRetriever
import logging

class HybridEvidenceRetriever:
    """
    Hybrid evidence retrieval system combining local and RSS sources
    """
    
    def __init__(self, scraped_data_dir: str = "data/scraped_articles"):
        self.logger = logging.getLogger(__name__)
        
        # Initialize both retrievers
        print("🏠 Initializing local evidence retriever...")
        self.local_retriever = LocalEvidenceRetriever(scraped_data_dir)
        
        print("📡 Initializing RSS evidence retriever...")
        self.rss_retriever = RSSEvidenceRetriever()
        
        print(f"✅ Hybrid retriever ready!")
        print(f"   📚 Local articles: {len(self.local_retriever.articles)}")
        print(f"   📰 RSS sources: {len(self.rss_retriever.get_available_sources())}")
    
    def retrieve_evidence(self, claim: str, num_results: int = 5, strategy: str = "hybrid") -> List[Dict]:
        """
        Retrieve evidence using hybrid approach
        
        Args:
            claim: The claim to find evidence for
            num_results: Total number of evidence pieces to return
            strategy: "local", "rss", or "hybrid"
            
        Returns:
            List of evidence dictionaries with similarity scores
        """
        evidence = []
        
        if strategy == "local":
            # Local only
            evidence = self.local_retriever.retrieve_evidence(claim, num_results)
            
        elif strategy == "rss":
            # RSS only
            evidence = self.rss_retriever.search_all_feeds(claim, num_results)
            
        else:  # hybrid
            # Hybrid approach: Try local first, supplement with RSS
            
            # 1. Get local evidence first (faster)
            local_evidence = self.local_retriever.retrieve_evidence(claim, num_results)
            print(f"📚 Local evidence: {len(local_evidence)} articles")
            evidence.extend(local_evidence)
            
            # 2. If we need more evidence, get from RSS
            remaining_needed = num_results - len(evidence)
            if remaining_needed > 0:
                print(f"📡 Getting {remaining_needed} more from RSS...")
                rss_evidence = self.rss_retriever.search_all_feeds(claim, remaining_needed * 2)
                
                # Filter out duplicates (by URL or title similarity)
                unique_rss = self._filter_duplicates(evidence, rss_evidence)
                evidence.extend(unique_rss[:remaining_needed])
                
                print(f"📰 RSS evidence added: {len(unique_rss[:remaining_needed])} articles")
            
            # 3. Sort by relevance (similarity score)
            evidence = sorted(evidence, key=lambda x: x.get('similarity', 0), reverse=True)
            evidence = evidence[:num_results]
        
        print(f"🎯 Total evidence returned: {len(evidence)}")
        return evidence
    
    def _filter_duplicates(self, existing_evidence: List[Dict], new_evidence: List[Dict]) -> List[Dict]:
        """
        Filter out duplicate articles from new evidence
        """
        if not existing_evidence:
            return new_evidence
        
        # Get existing URLs and titles
        existing_urls = {ev.get('link', '') for ev in existing_evidence}
        existing_titles = {ev.get('title', '').lower() for ev in existing_evidence}
        
        unique_evidence = []
        for article in new_evidence:
            url = article.get('link', '')
            title = article.get('title', '').lower()
            
            # Skip if URL or title already exists
            if url in existing_urls or title in existing_titles:
                continue
                
            unique_evidence.append(article)
            existing_urls.add(url)
            existing_titles.add(title)
        
        return unique_evidence
    
    def get_source_stats(self) -> Dict:
        """Get comprehensive source statistics"""
        stats = {
            "local": self.local_retriever.get_source_stats(),
            "rss_sources": self.rss_retriever.get_available_sources(),
            "total_local_articles": len(self.local_retriever.articles)
        }
        return stats
    
    def test_all_sources(self) -> Dict:
        """Test both local and RSS sources"""
        print("🧪 Testing all evidence sources...\n")
        
        # Test local sources
        print("📚 LOCAL SOURCES:")
        local_stats = self.local_retriever.get_source_stats()
        for source, data in local_stats.items():
            print(f"  ✅ {source}: {data['article_count']} articles")
        
        # Test RSS sources
        print("\n📡 RSS SOURCES:")
        rss_status = self.rss_retriever.test_feeds()
        
        working_rss = sum(1 for status in rss_status.values() if status)
        total_rss = len(rss_status)
        
        print(f"\n📊 SUMMARY:")
        print(f"  📚 Local: {len(local_stats)} sources, {len(self.local_retriever.articles)} total articles")
        print(f"  📡 RSS: {working_rss}/{total_rss} feeds working")
        
        return {
            "local": local_stats,
            "rss": rss_status,
            "summary": {
                "local_sources": len(local_stats),
                "local_articles": len(self.local_retriever.articles),
                "working_rss": working_rss,
                "total_rss": total_rss
            }
        }
    
    def search_by_source_type(self, claim: str, source_type: str, num_results: int = 5) -> List[Dict]:
        """
        Search specific source types
        
        Args:
            source_type: "local", "rss", or specific source names
        """
        if source_type == "local":
            return self.local_retriever.retrieve_evidence(claim, num_results)
        elif source_type == "rss":
            return self.rss_retriever.search_all_feeds(claim, num_results)
        else:
            # Try to search specific sources
            local_sources = list(self.local_retriever.get_source_stats().keys())
            rss_sources = self.rss_retriever.get_available_sources()
            
            evidence = []
            
            if source_type in local_sources:
                evidence = self.local_retriever.retrieve_evidence(claim, num_results, [source_type])
            elif source_type in rss_sources:
                evidence = self.rss_retriever.search_specific_feeds(claim, [source_type], num_results)
            
            return evidence