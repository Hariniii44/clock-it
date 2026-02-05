import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random
from typing import List, Dict, Tuple
from collections import defaultdict

class ArticleGrouper:
    def __init__(self, similarity_threshold: float = 0.3, max_similar_articles: int = 15):
        self.similarity_threshold = similarity_threshold
        self.max_similar_articles = max_similar_articles
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )
        
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if pd.isna(text) or not text:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove common prefixes/suffixes from headlines
        prefixes_to_remove = [
            'breaking:', 'urgent:', 'latest:', 'update:', 
            'sri lanka:', 'lk:', 'colombo:'
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        return text
    
    def group_related_articles(self, df: pd.DataFrame) -> List[List[Dict]]:
        """
        Implementation of Algorithm 1 from the paper
        Groups articles discussing the same events from different sources
        """
        print(f"Grouping {len(df)} political articles into related events...")
        
        # Prepare data
        articles = []
        for idx, row in df.iterrows():
            # Use description as headline (since we don't have separate headline)
            headline = self.preprocess_text(row['description'])
            if headline:  # Only include articles with valid headlines
                articles.append({
                    'id': idx,
                    'headline': headline,
                    'source': row['source'],
                    'description': row['description'],
                    'date': row['date_str'],
                    'url': row['url_metadata']
                })
        
        print(f"Processing {len(articles)} articles with valid headlines")
        
        # Create TF-IDF vectors for all headlines
        headlines = [article['headline'] for article in articles]
        tfidf_matrix = self.vectorizer.fit_transform(headlines)
        
        # Initialize groups and remaining articles
        article_groups = []
        remaining_articles = set(range(len(articles)))
        processed_sources = set()
        
        group_count = 0
        
        # Main grouping algorithm (following Algorithm 1)
        while remaining_articles:
            # Randomly pick a source first to avoid bias (as per paper)
            available_sources = {articles[i]['source'] for i in remaining_articles}
            if not available_sources:
                break
                
            chosen_source = random.choice(list(available_sources))
            
            # Pick a random article from chosen source
            source_articles = [i for i in remaining_articles 
                             if articles[i]['source'] == chosen_source]
            if not source_articles:
                continue
                
            seed_idx = random.choice(source_articles)
            seed_article = articles[seed_idx]
            
            # Find similar articles
            similar_indices = self._find_similar_articles(
                seed_idx, tfidf_matrix, remaining_articles
            )
            
            # Create group with one article per source
            group = []
            seen_sources = set()
            
            # Add seed article
            group.append(seed_article)
            seen_sources.add(seed_article['source'])
            remaining_articles.remove(seed_idx)
            
            # Add similar articles (max one per source)
            for sim_idx in similar_indices:
                if sim_idx in remaining_articles:
                    sim_article = articles[sim_idx]
                    if sim_article['source'] not in seen_sources:
                        group.append(sim_article)
                        seen_sources.add(sim_article['source'])
                        remaining_articles.remove(sim_idx)
            
            # Only keep groups with articles from multiple sources
            if len(group) > 1:
                article_groups.append(group)
                group_count += 1
                
                if group_count % 100 == 0:
                    print(f"Created {group_count} groups, {len(remaining_articles)} articles remaining")
        
        print(f"Created {len(article_groups)} article groups")
        self._print_group_statistics(article_groups)
        
        return article_groups
    
    def _find_similar_articles(self, seed_idx: int, tfidf_matrix, 
                              remaining_articles: set) -> List[int]:
        """Find articles similar to seed article using cosine similarity"""
        
        # Calculate similarity with seed article
        seed_vector = tfidf_matrix[seed_idx]
        similarities = cosine_similarity(seed_vector, tfidf_matrix).flatten()
        
        # Get indices of similar articles
        similar_pairs = []
        for idx in remaining_articles:
            if idx != seed_idx and similarities[idx] >= self.similarity_threshold:
                similar_pairs.append((idx, similarities[idx]))
        
        # Sort by similarity and return top similar articles
        similar_pairs.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in similar_pairs[:self.max_similar_articles]]
    
    def _print_group_statistics(self, article_groups: List[List[Dict]]):
        """Print statistics about the article groups"""
        if not article_groups:
            return
            
        group_sizes = [len(group) for group in article_groups]
        source_coverage = defaultdict(int)
        
        for group in article_groups:
            sources_in_group = {article['source'] for article in group}
            for source in sources_in_group:
                source_coverage[source] += 1
        
        print(f"\nGroup Statistics:")
        print(f"   Total groups: {len(article_groups)}")
        print(f"   Average group size: {np.mean(group_sizes):.1f} sources")
        print(f"   Median group size: {np.median(group_sizes):.0f} sources")
        print(f"   Largest group: {max(group_sizes)} sources")
        print(f"   Smallest group: {min(group_sizes)} sources")
        
        print(f"\nSource participation in groups:")
        for source, count in sorted(source_coverage.items()):
            percentage = count / len(article_groups) * 100
            print(f"   {source}: {count} groups ({percentage:.1f}%)")
    
    def save_groups(self, article_groups: List[List[Dict]], filepath: str):
        """Save article groups to JSON file"""
        # Convert to serializable format
        groups_data = {
            'total_groups': len(article_groups),
            'groups': []
        }
        
        for i, group in enumerate(article_groups):
            group_data = {
                'group_id': i,
                'size': len(group),
                'sources': [article['source'] for article in group],
                'articles': group
            }
            groups_data['groups'].append(group_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(groups_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(article_groups)} article groups to {filepath}")
    
    def load_groups(self, filepath: str) -> List[List[Dict]]:
        """Load article groups from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            article_groups = [group['articles'] for group in data['groups']]
            print(f"Loaded {len(article_groups)} article groups from {filepath}")
            return article_groups
            
        except FileNotFoundError:
            print(f"No saved groups found at {filepath}")
            return []

if __name__ == "__main__":
    # Load political articles from Step 2
    try:
        df = pd.read_csv("data/political_articles.csv")
        print(f"Loaded {len(df)} political articles from Step 2")
        
        # Initialize grouper
        grouper = ArticleGrouper(similarity_threshold=0.25)  # Lower threshold for more groups
        
        # Try to load existing groups first
        groups_path = "data/article_groups.json"
        article_groups = grouper.load_groups(groups_path)
        
        if not article_groups:
            # Create new groups
            print(f"\nCreating article groups...")
            article_groups = grouper.group_related_articles(df)
            
            # Save groups
            if article_groups:
                grouper.save_groups(article_groups, groups_path)
        
        if article_groups:
            print(f"\nReady for Step 4 with {len(article_groups)} article groups")
            print("\nSTEP 3 COMPLETE: Article groups created successfully!")
            
            # Show a sample group
            if len(article_groups) > 0:
                sample_group = article_groups[0]
                print(f"\nSample group (Group 1):")
                for article in sample_group:
                    headline = article['headline'][:80] + "..." if len(article['headline']) > 80 else article['headline']
                    print(f"   {article['source']}: {headline}")
        else:
            print("\nNo article groups created!")
            
    except FileNotFoundError:
        print("Please run Step 2 first to create political_articles.csv")