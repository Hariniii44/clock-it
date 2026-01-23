#!/usr/bin/env python3
"""
Local Evidence Retrieval System - FR02 Implementation
Uses scraped articles from local CSV files instead of external APIs
"""

import os
import pandas as pd
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

class LocalEvidenceRetriever:
    """
    Evidence retrieval using locally scraped news articles.
    Implements FR02: Retrieve evidence from news sources
    """
    
    def __init__(self, scraped_data_dir: str = "data/scraped_articles"):
        self.data_dir = scraped_data_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize similarity model for semantic search
        print("Loading similarity model...")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load all scraped articles
        self.articles = self._load_all_articles()
        print(f"Loaded {len(self.articles)} articles from {self.data_dir}")
    
    def _load_all_articles(self) -> pd.DataFrame:
        """Load all scraped articles from CSV files"""
        all_articles = []
        
        if not os.path.exists(self.data_dir):
            self.logger.warning(f"Scraped data directory not found: {self.data_dir}")
            return pd.DataFrame()
        
        for csv_file in os.listdir(self.data_dir):
            if csv_file.endswith('_articles.csv'):
                filepath = os.path.join(self.data_dir, csv_file)
                try:
                    df = pd.read_csv(filepath)
                    
                    # Add source column if missing
                    if 'source' not in df.columns:
                        source_name = csv_file.replace('_articles.csv', '')
                        df['source'] = source_name
                    
                    all_articles.append(df)
                    self.logger.info(f"Loaded {len(df)} articles from {csv_file}")
                    
                except Exception as e:
                    self.logger.error(f"Error loading {csv_file}: {e}")
        
        if all_articles:
            combined = pd.concat(all_articles, ignore_index=True)
            # Remove duplicates based on URL
            combined = combined.drop_duplicates(subset=['url'])
            
            # Clean data - remove articles with missing content
            combined = combined.dropna(subset=['title', 'content'])
            combined = combined[combined['content'].str.len() > 50]  # Min content length
            
            return combined
        
        return pd.DataFrame()
    
    def retrieve_evidence(self, claim: str, num_results: int = 5, source_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        FR02: Retrieve relevant evidence from locally scraped articles
        
        Args:
            claim: The claim to find evidence for
            num_results: Number of evidence pieces to return
            source_filter: Optional list of sources to filter by
            
        Returns:
            List of evidence dictionaries with similarity scores
        """
        if self.articles.empty:
            self.logger.warning("No scraped articles available!")
            return []
        
        # Filter by source if specified
        articles_to_search = self.articles
        if source_filter:
            articles_to_search = self.articles[self.articles['source'].isin(source_filter)]
            
        if articles_to_search.empty:
            self.logger.warning(f"No articles found for sources: {source_filter}")
            return []
        
        # Get claim embedding
        claim_embedding = self.similarity_model.encode(claim)
        
        # Prepare article texts for comparison
        article_texts = []
        for _, row in articles_to_search.iterrows():
            # Combine title and content for better matching
            text = f"{row['title']} {row['content'][:500]}"  # First 500 chars of content
            article_texts.append(text)
        
        # Get article embeddings
        print(f"Computing similarities for {len(article_texts)} articles...")
        article_embeddings = self.similarity_model.encode(article_texts)
        
        # Calculate similarities
        similarities = np.dot(article_embeddings, claim_embedding)
        
        # Get top matches
        top_indices = similarities.argsort()[-num_results:][::-1]
        
        evidence = []
        for idx in top_indices:
            row = articles_to_search.iloc[idx]
            
            # Create snippet (first 200 chars of content)
            content = str(row['content'])
            snippet = content[:200] + "..." if len(content) > 200 else content
            
            evidence.append({
                "title": row['title'],
                "snippet": snippet,
                "content": content,  # Full content for verification
                "source": row['source'],
                "link": row['url'],
                "date": row.get('date', ''),
                "similarity": float(similarities[idx]),
                "word_count": row.get('word_count', 0)
            })
        
        return evidence
    
    def get_source_stats(self) -> Dict:
        """Get statistics about available sources"""
        if self.articles.empty:
            return {}
            
        stats = {}
        for source in self.articles['source'].unique():
            source_articles = self.articles[self.articles['source'] == source]
            stats[source] = {
                'article_count': len(source_articles),
                'avg_word_count': source_articles['word_count'].mean() if 'word_count' in source_articles else 0
            }
        
        return stats
    
    def search_by_keyword(self, keywords: List[str], num_results: int = 10) -> List[Dict]:
        """Simple keyword-based search for testing"""
        if self.articles.empty:
            return []
            
        # Create search pattern
        pattern = '|'.join(keywords)
        
        # Search in title and content
        matches = self.articles[
            self.articles['title'].str.contains(pattern, case=False, na=False) |
            self.articles['content'].str.contains(pattern, case=False, na=False)
        ]
        
        results = []
        for _, row in matches.head(num_results).iterrows():
            results.append({
                "title": row['title'],
                "snippet": str(row['content'])[:200] + "...",
                "source": row['source'],
                "link": row['url'],
                "date": row.get('date', '')
            })
            
        return results
