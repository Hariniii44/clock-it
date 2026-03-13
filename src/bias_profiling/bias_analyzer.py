import pandas as pd
import numpy as np
import json
from transformers import pipeline
from typing import Dict, List, Tuple
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class BiasAnalyzer:
    def __init__(self):
        print("Loading pre-trained ABSA model...")
        # Using a pre-trained ABSA model instead of training our own
        self.absa_model = pipeline(
            "text-classification",
            model="yangheng/deberta-v3-base-absa-v1.1",
            device=-1  # Use CPU (set to 0 for GPU)
        )
        print("ABSA model loaded successfully")
        
        # Load politician lists
        self.load_politicians()
        
    def load_politicians(self):
        """Load politician lists and party affiliations"""
        try:
            with open("data/sri_lankan_politicians.json", 'r', encoding='utf-8') as f:
                pol_data = json.load(f)
            
            self.politicians = pol_data['politicians']
            self.government_politicians = set(pol_data['government'])
            self.opposition_politicians = set(pol_data['opposition'])
            
            print(f"Loaded {len(self.politicians)} politicians")
            print(f"   Government: {len(self.government_politicians)} politicians")
            print(f"   Opposition: {len(self.opposition_politicians)} politicians")
            
        except FileNotFoundError:
            print("Please run Step 2 first to create politician database")
            raise
    
    def extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if pd.isna(text) or not text:
            return []
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', str(text))
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return sentences
    
    def find_politicians_in_text(self, text: str) -> List[str]:
        """Find politician names mentioned in text"""
        if pd.isna(text) or not text:
            return []
        
        text_lower = str(text).lower()
        found_politicians = []
        
        for politician in self.politicians.keys():
            if politician in text_lower:
                found_politicians.append(politician)
        
        return found_politicians
    
    def get_aspect_sentiment(self, sentence: str, aspect: str) -> Dict[str, float]:
        """Get sentiment towards specific aspect using ABSA model"""
        try:
            # Format input for ABSA model
            input_text = f"{sentence} [SEP] {aspect}"
            
            # Get prediction
            result = self.absa_model(input_text)
            
            # Parse result (model returns positive/negative/neutral)
            if isinstance(result, list) and len(result) > 0:
                pred = result[0]
                label = pred['label'].lower()
                score = pred['score']
                
                # Convert to our scoring system
                if 'positive' in label:
                    return {'positive': score, 'negative': 0, 'neutral': 1-score}
                elif 'negative' in label:
                    return {'positive': 0, 'negative': score, 'neutral': 1-score}
                else:  # neutral
                    return {'positive': 0, 'negative': 0, 'neutral': score}
            
            # Fallback
            return {'positive': 0, 'negative': 0, 'neutral': 1.0}
            
        except Exception as e:
            # If ABSA fails, return neutral
            return {'positive': 0, 'negative': 0, 'neutral': 1.0}
    
    def analyze_article_sentiment(self, article: Dict) -> float:
        """
        Analyze sentiment of an article towards politicians
        Following Algorithm 2 from the paper
        """
        description = article.get('description', '')
        sentences = self.extract_sentences(description)
        
        if not sentences:
            return 0.0
        
        sentence_scores = []
        
        for sentence in sentences:
            # Find politicians mentioned in this sentence
            politicians_in_sentence = self.find_politicians_in_text(sentence)
            
            if not politicians_in_sentence:
                continue
            
            # Calculate sentiment for each politician in sentence
            for politician in politicians_in_sentence:
                sentiment = self.get_aspect_sentiment(sentence, politician)
                
                # Calculate bias score: (positive - negative)
                bias_score = sentiment['positive'] - sentiment['negative']
                
                # Apply party multiplier as per Algorithm 2
                if politician in self.government_politicians:
                    party_multiplier = 1.0  # Government = Democratic equivalent
                elif politician in self.opposition_politicians:
                    party_multiplier = -1.0  # Opposition = Republican equivalent
                else:
                    party_multiplier = 0.0  # Unknown
                
                # Final sentence score
                sentence_score = bias_score * party_multiplier
                sentence_scores.append(sentence_score)
        
        # Return average sentiment across sentences
        return np.mean(sentence_scores) if sentence_scores else 0.0
    
    def calculate_pairwise_bias(self, article_groups: List[List[Dict]]) -> np.ndarray:
        """
        Calculate pairwise bias matrix between sources
        Following Algorithm 2 from the paper
        """
        print(f"Analyzing bias across {len(article_groups)} article groups...")
        
        # Get all unique sources
        all_sources = set()
        for group in article_groups:
            for article in group:
                all_sources.add(article['source'])
        
        sources = sorted(list(all_sources))
        n_sources = len(sources)
        source_to_idx = {source: idx for idx, source in enumerate(sources)}
        
        print(f"Analyzing {n_sources} sources: {sources}")
        
        # Initialize bias collection matrix
        bias_differences = defaultdict(list)
        
        processed_groups = 0
        
        for group_idx, group in enumerate(article_groups):
            if len(group) < 2:  # Skip single-source groups
                continue
            
            # Calculate sentiment scores for all articles in group
            article_scores = {}
            for article in group:
                score = self.analyze_article_sentiment(article)
                article_scores[article['source']] = score
            
            # Calculate pairwise differences within this group
            for i, article1 in enumerate(group):
                for j, article2 in enumerate(group):
                    if i != j:  # Don't compare article with itself
                        source1 = article1['source']
                        source2 = article2['source']
                        
                        score_diff = article_scores[source1] - article_scores[source2]
                        bias_differences[(source1, source2)].append(score_diff)
            
            processed_groups += 1
            if processed_groups % 200 == 0:
                print(f"Processed {processed_groups}/{len(article_groups)} groups")
        
        # Create final bias matrix by averaging differences
        bias_matrix = np.zeros((n_sources, n_sources))
        
        for (source1, source2), differences in bias_differences.items():
            if differences:  # Only if we have data
                idx1 = source_to_idx[source1]
                idx2 = source_to_idx[source2]
                avg_diff = np.mean(differences)
                bias_matrix[idx1][idx2] = avg_diff
        
        print(f"Calculated bias matrix for {n_sources} sources")
        print(f"Processed {processed_groups} valid groups")
        
        return bias_matrix, sources
    
    def save_bias_matrix(self, bias_matrix: np.ndarray, sources: List[str], filepath: str):
        """Save bias matrix and sources"""
        data = {
            'sources': sources,
            'bias_matrix': bias_matrix.tolist(),
            'matrix_shape': bias_matrix.shape
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved bias matrix to {filepath}")
    
    def load_bias_matrix(self, filepath: str) -> Tuple[np.ndarray, List[str]]:
        """Load bias matrix from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            bias_matrix = np.array(data['bias_matrix'])
            sources = data['sources']
            
            print(f"Loaded bias matrix for {len(sources)} sources")
            return bias_matrix, sources
            
        except FileNotFoundError:
            print(f"No saved bias matrix found at {filepath}")
            return None, None
    
    def print_bias_matrix(self, bias_matrix: np.ndarray, sources: List[str]):
        """Print bias matrix in readable format"""
        print(f"\nBias Matrix ({len(sources)}x{len(sources)}):")
        print("Rows = Source A, Columns = Source B")
        print("Positive values = Source A more pro-government than Source B")
        print("Negative values = Source A more opposition-leaning than Source B\n")
        
        # Print header
        print(f"{'Source':<20}", end="")
        for source in sources:
            print(f"{source[:12]:<12}", end="")
        print()
        
        # Print matrix
        for i, source_row in enumerate(sources):
            print(f"{source_row:<20}", end="")
            for j in range(len(sources)):
                value = bias_matrix[i][j]
                print(f"{value:>11.3f}", end=" ")
            print()

if __name__ == "__main__":
    # Load article groups from Step 3
    try:
        with open("data/article_groups.json", 'r', encoding='utf-8') as f:
            groups_data = json.load(f)
        
        article_groups = [group['articles'] for group in groups_data['groups']]
        print(f"Loaded {len(article_groups)} article groups from Step 3")
        
        # Initialize bias analyzer
        analyzer = BiasAnalyzer()
        
        # Try to load existing bias matrix
        matrix_path = "data/bias_matrix.json"
        bias_matrix, sources = analyzer.load_bias_matrix(matrix_path)
        
        if bias_matrix is None:
            # Calculate new bias matrix
            print(f"\nCalculating bias matrix...")
            bias_matrix, sources = analyzer.calculate_pairwise_bias(article_groups)
            
            # Save bias matrix
            if bias_matrix is not None:
                analyzer.save_bias_matrix(bias_matrix, sources, matrix_path)
        
        if bias_matrix is not None:
            # Display results
            analyzer.print_bias_matrix(bias_matrix, sources)
            
            print(f"\nReady for Step 5 (Graph-based scoring)")
            print("\nSTEP 4 COMPLETE: Bias matrix calculated successfully!")
        else:
            print("\nFailed to calculate bias matrix!")
            
    except FileNotFoundError:
        print("Please run Step 3 first to create article groups")
    except Exception as e:
        print(f"Error in ABSA analysis: {e}")
        import traceback
        traceback.print_exc()