#!/usr/bin/env python3
"""
RSS-based Evidence Retrieval System
Real-time evidence from news RSS feeds without APIs
"""

import feedparser
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import re
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

class RSSEvidenceRetriever:
    """
    Real-time evidence retrieval using RSS feeds from news sources
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize similarity model for relevance scoring
        print("Loading similarity model for RSS retrieval...")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sri Lankan news RSS feeds
        self.rss_feeds = {
            "adaderana": "http://www.adaderana.lk/rss.php",
            "dailynews": "https://www.dailynews.lk/rss.xml",
            "island": "http://www.island.lk/rss.xml", 
            "hirunews": "http://www.hirunews.lk/rss/english.xml",
            "newswire": "http://www.newswire.lk/feed/",
            "colombopage": "http://www.colombopage.com/news/rss.php",
            "lankaweb": "http://www.lankaweb.com/news/feed/",
            "ft": "http://www.ft.lk/rss"  # Financial Times Sri Lanka
        }
        
        self.cache = {}  # Simple cache to avoid repeated RSS calls
        self.cache_duration = timedelta(minutes=15)  # Cache for 15 minutes
        
    def fetch_rss_feed(self, feed_url: str, source_name: str) -> List[Dict]:
        """
        Fetch and parse RSS feed from a news source
        """
        try:
            # Check cache first
            cache_key = f"{source_name}_{feed_url}"
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if datetime.now() - cached_time < self.cache_duration:
                    return cached_data
            
            print(f"Fetching RSS from {source_name}...")
            
            # Add headers to avoid blocks
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Set timeout to avoid hanging
            feed = feedparser.parse(feed_url, request_headers=headers)
            
            if hasattr(feed, 'status') and feed.status != 200:
                self.logger.warning(f"RSS feed {source_name} returned status {feed.status}")
                return []
            
            articles = []
            for entry in feed.entries[:20]:  # Limit to 20 most recent
                # Extract article data
                article = {
                    'title': entry.get('title', '').strip(),
                    'snippet': entry.get('summary', entry.get('description', '')).strip(),
                    'content': entry.get('content', entry.get('summary', entry.get('description', ''))),
                    'link': entry.get('link', ''),
                    'source': source_name,
                    'date': self._parse_date(entry.get('published', entry.get('updated', ''))),
                    'source_type': 'rss'
                }
                
                # Clean HTML from snippet and content
                if isinstance(article['content'], list):
                    article['content'] = article['content'][0].get('value', '') if article['content'] else ''
                
                article['snippet'] = self._clean_html(article['snippet'])
                article['content'] = self._clean_html(str(article['content']))
                
                # Skip if no meaningful content
                if len(article['title']) < 10 or len(article['snippet']) < 20:
                    continue
                    
                articles.append(article)
            
            # Cache the results
            self.cache[cache_key] = (datetime.now(), articles)
            
            print(f"✅ Found {len(articles)} articles from {source_name}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching RSS from {source_name}: {e}")
            return []
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags and clean text"""
        if not text:
            return ""
            
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common RSS artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove [CDATA] etc
        text = re.sub(r'&\w+;', ' ', text)   # Remove HTML entities
        
        return text
    
    def _parse_date(self, date_str: str) -> str:
        """Parse RSS date string to readable format"""
        if not date_str:
            return ""
        try:
            # feedparser usually handles this, but just in case
            return date_str
        except:
            return ""
    
    def search_all_feeds(self, claim: str, max_articles: int = 10) -> List[Dict]:
        """
        Search all RSS feeds for articles related to claim
        """
        all_articles = []
        
        # Fetch from all RSS feeds
        for source_name, feed_url in self.rss_feeds.items():
            articles = self.fetch_rss_feed(feed_url, source_name)
            all_articles.extend(articles)
        
        if not all_articles:
            print("⚠️ No RSS articles found")
            return []
        
        print(f"📰 Total RSS articles fetched: {len(all_articles)}")
        
        # Filter articles relevant to claim using semantic similarity
        relevant_articles = self._find_relevant_articles(claim, all_articles, max_articles)
        
        return relevant_articles
    
    def _find_relevant_articles(self, claim: str, articles: List[Dict], max_results: int) -> List[Dict]:
        """
        Find articles most relevant to the claim using semantic similarity
        """
        if not articles:
            return []
        
        print(f"🔍 Computing relevance scores for {len(articles)} RSS articles...")
        
        # Get claim embedding
        claim_embedding = self.similarity_model.encode(claim)
        
        # Prepare article texts for comparison
        article_texts = []
        for article in articles:
            # Combine title and snippet for better matching
            text = f"{article['title']} {article['snippet']}"
            article_texts.append(text)
        
        # Get article embeddings
        article_embeddings = self.similarity_model.encode(article_texts)
        
        # Calculate similarities
        similarities = np.dot(article_embeddings, claim_embedding)
        
        # Get top matches
        top_indices = similarities.argsort()[-max_results:][::-1]
        
        relevant_articles = []
        for idx in top_indices:
            article = articles[idx].copy()
            article['similarity'] = float(similarities[idx])
            
            # Only include if similarity is above threshold
            if article['similarity'] > 0.2:  # Relevance threshold
                relevant_articles.append(article)
        
        print(f"✅ Found {len(relevant_articles)} relevant RSS articles")
        return relevant_articles
    
    def search_specific_feeds(self, claim: str, source_names: List[str], max_articles: int = 5) -> List[Dict]:
        """
        Search specific RSS feeds only
        """
        all_articles = []
        
        for source_name in source_names:
            if source_name in self.rss_feeds:
                feed_url = self.rss_feeds[source_name]
                articles = self.fetch_rss_feed(feed_url, source_name)
                all_articles.extend(articles)
            else:
                self.logger.warning(f"RSS feed not found for source: {source_name}")
        
        if not all_articles:
            return []
        
        return self._find_relevant_articles(claim, all_articles, max_articles)
    
    def get_available_sources(self) -> List[str]:
        """Get list of available RSS sources"""
        return list(self.rss_feeds.keys())
    
    def test_feeds(self) -> Dict[str, bool]:
        """Test all RSS feeds to see which ones are working"""
        print("🧪 Testing RSS feeds...")
        
        feed_status = {}
        for source_name, feed_url in self.rss_feeds.items():
            try:
                articles = self.fetch_rss_feed(feed_url, source_name)
                feed_status[source_name] = len(articles) > 0
                status = "✅" if feed_status[source_name] else "⚠️"
                print(f"  {status} {source_name}: {len(articles)} articles")
            except Exception as e:
                feed_status[source_name] = False
                print(f"  ❌ {source_name}: Error - {e}")
        
        return feed_status