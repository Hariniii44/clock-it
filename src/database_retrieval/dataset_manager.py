from datasets import load_dataset
import pandas as pd
import os
import json
from typing import Dict, List, Optional
from tqdm import tqdm
import logging

class DatasetManager:
    """Manages downloading and preparing Nuwan's Sri Lankan datasets"""
    
    def __init__(self, data_dir: str = "data/databases"):
        self.data_dir = data_dir
        self.datasets = {}
        self.setup_logging()
        
        # Updated dataset configurations - using CHUNKS datasets for actual content
        self.dataset_configs = {
            'hansard': {
                'hf_name': 'nuuuwan/lk-hansard-2020s-chunks',  # Changed to chunks
                'description': 'Parliamentary debates and speeches (2020s) - Full Text',
                'authority_weight': 1.0,
                'priority': 1,
                'has_full_text': True
            },
            'hansard_2010s': {
                'hf_name': 'lk-hansard-2010s-chunks',
                'description': 'Parliamentary debates and speeches (2010s) - Full Text',
                'authority_weight': 0.9,
                'priority': 1,
                'has_full_text': True
            },
            'hansard_2000s': {
                'hf_name': 'lk-hansard-2000s-chunks',
                'description': 'Parliamentary debates and speeches (2000s) - Full Text',
                'authority_weight': 0.8,
                'priority': 1,
                'has_full_text': True
            },
            'police': {
                'hf_name': 'nuuuwan/lk-police-press-releases-chunks',  # Changed to chunks
                'description': 'Police press releases - Full Text',
                'authority_weight': 0.9,
                'priority': 2,
                'has_full_text': True
            },
            'treasury': {
                'hf_name': 'nuuuwan/lk-treasury-press-releases-chunks',  # Changed to chunks
                'description': 'Treasury statements - Full Text',
                'authority_weight': 0.95,
                'priority': 2,
                'has_full_text': True
            },
            'news': {
                'hf_name': 'nuuuwan/lk-news-chunks',  # Changed to chunks
                'description': 'Sri Lankan news articles - Full Text',
                'authority_weight': 0.7,
                'priority': 3,
                'has_full_text': True
            },
            'fisheries_weekly': {
                'hf_name': 'nuuuwan/lk-fisheries-weekly-fish-prices-reports-chunks',
                'description': 'Weekly fisheries reports - Full Text',
                'authority_weight': 0.8,
                'priority': 4,
                'has_full_text': True
            },
            'tourism_weekly': {
                'hf_name': 'nuuuwan/lk-tourism-weekly-reports-chunks',
                'description': 'Tourism weekly reports - Full Text',
                'authority_weight': 0.8,
                'priority': 4,
                'has_full_text': True
            }
        }
        
        os.makedirs(data_dir, exist_ok=True)
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def explore_dataset_structure(self, dataset_key: str):
        """Explore the structure of a dataset before processing"""
        if dataset_key not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        config = self.dataset_configs[dataset_key]
        self.logger.info(f"Exploring structure of {dataset_key}...")
        
        try:
            # Load just a few samples to understand structure
            dataset = load_dataset(config['hf_name'], split='train', streaming=True)
            
            # Get first 3 examples
            examples = list(dataset.take(3))
            
            if examples:
                print(f"\n📊 Dataset Structure for {dataset_key} ({config['hf_name']}):")
                print(f"Available columns: {list(examples[0].keys())}")
                print(f"\n🔍 Sample data:")
                
                for i, example in enumerate(examples, 1):
                    print(f"\n--- Example {i} ---")
                    for key, value in example.items():
                        if isinstance(value, str):
                            display_value = value[:200] + "..." if len(value) > 200 else value
                        else:
                            display_value = str(value)
                        print(f"{key}: {display_value}")
                        
            else:
                print(f"No examples found in {dataset_key}")
                
        except Exception as e:
            self.logger.error(f"Error exploring {dataset_key}: {str(e)}")
    
    def download_dataset(self, dataset_key: str, max_samples: Optional[int] = None) -> pd.DataFrame:
        """Download and prepare a specific dataset"""
        
        if dataset_key not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        config = self.dataset_configs[dataset_key]
        cache_file = os.path.join(self.data_dir, f"{dataset_key}_processed.csv")
        
        # Check if already cached
        if os.path.exists(cache_file):
            self.logger.info(f"Loading cached {dataset_key} dataset...")
            df = pd.read_csv(cache_file)
            self.datasets[dataset_key] = df
            return df
        
        self.logger.info(f"Downloading {dataset_key} dataset from HuggingFace...")
        
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset(config['hf_name'], split='train')
            
            # Convert to DataFrame
            df = dataset.to_pandas()
            
            self.logger.info(f"Raw dataset loaded: {len(df)} documents")
            self.logger.info(f"Available columns: {list(df.columns)}")
            
            # Limit samples if specified (for testing)
            if max_samples and len(df) > max_samples:
                df = df.head(max_samples)
                self.logger.info(f"Limited to {max_samples} samples for testing")
            
            # Standardize columns for chunks datasets
            df = self._standardize_chunks_columns(df, dataset_key)
            
            # Filter for English content (more flexible approach for chunks)
            df = self._filter_english_content_chunks(df)
            
            # Add metadata
            df['dataset_source'] = dataset_key
            df['authority_weight'] = config['authority_weight']
            df['priority'] = config['priority']
            
            self.logger.info(f"Final dataset size after processing: {len(df)} documents")
            
            # Cache the processed dataset
            if len(df) > 0:
                df.to_csv(cache_file, index=False)
                self.logger.info(f"Cached {len(df)} documents from {dataset_key}")
            else:
                self.logger.warning(f"No documents remaining after processing {dataset_key}")
            
            self.datasets[dataset_key] = df
            return df
            
        except Exception as e:
            self.logger.error(f"Error downloading {dataset_key}: {str(e)}")
            raise
    
    def _filter_english_content_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for English content in chunks datasets"""
        original_count = len(df)
        
        # Strategy 1: Check for 'lang' or 'language' column
        if 'lang' in df.columns:
            # For multilingual content like hansard (si-ta-en), keep them
            english_patterns = ['en', 'english', 'si-ta-en', 'ta-en', 'en-si']
            english_mask = df['lang'].str.contains('|'.join(english_patterns), na=False, case=False)
            df = df[english_mask]
            self.logger.info(f"Filtered by 'lang' column: {len(df)} documents (from {original_count})")
            
        # Strategy 2: Use text analysis for better English detection
        if 'text' in df.columns and len(df) > 0:
            def is_likely_english_chunk(text):
                if not isinstance(text, str) or len(text.strip()) < 20:
                    return False
                
                text_lower = text.lower()
                
                # Common English words that are likely in parliamentary debates
                english_indicators = [
                    'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'that', 'was', 'for', 'on', 'are', 'as', 'with',
                    'president', 'minister', 'government', 'parliament', 'committee', 'members', 'sri lanka',
                    'honorable', 'hon', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those'
                ]
                
                # Count English indicators
                indicator_count = sum(1 for word in english_indicators if word in text_lower)
                
                # If we find several English indicators, likely English content
                return indicator_count >= 3
            
            # Apply English detection
            english_mask = df['text'].apply(is_likely_english_chunk)
            english_count = english_mask.sum()
            
            self.logger.info(f"Text analysis found {english_count} likely English chunks")
            
            if english_count > 0:
                df = df[english_mask]
            else:
                # If no English detected, keep a sample for manual inspection
                self.logger.warning(f"No English content detected by text analysis, keeping first 50 for inspection")
                df = df.head(50)
        
        return df
    
    def _standardize_chunks_columns(self, df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
        """Standardize column names for chunks datasets"""
        
        self.logger.info(f"Standardizing chunks columns for {dataset_key}")
        self.logger.info(f"Original columns: {list(df.columns)}")
        
        # For chunks datasets, look for different column patterns
        
        # Try to find text content column (chunks usually have 'text' or 'content')
        text_candidates = ['text', 'content', 'body', 'chunk_text', 'full_text']
        text_col = None
        for candidate in text_candidates:
            if candidate in df.columns:
                text_col = candidate
                break
        
        # Try to find title/heading column
        title_candidates = ['title', 'heading', 'subject', 'chunk_title']
        title_col = None
        for candidate in title_candidates:
            if candidate in df.columns:
                title_col = candidate
                break
        
        # Try to find date column
        date_candidates = ['date', 'date_str', 'time_published', 'published_date', 'created_date']
        date_col = None
        for candidate in date_candidates:
            if candidate in df.columns:
                date_col = candidate
                break
        
        # Try to find URL or ID column
        url_candidates = ['url', 'url_canonical', 'link', 'source_url', 'doc_id']
        url_col = None
        for candidate in url_candidates:
            if candidate in df.columns:
                url_col = candidate
                break
        
        # Create standardized columns
        df['text'] = df[text_col].fillna('').astype(str) if text_col else ''
        df['title'] = df[title_col].fillna('').astype(str) if title_col else df['text'].str[:100]  # Use first 100 chars as title
        df['date'] = df[date_col].fillna('').astype(str) if date_col else ''
        df['url'] = df[url_col].fillna('').astype(str) if url_col else ''
        
        self.logger.info(f"Mapped columns - text: {text_col}, title: {title_col}, date: {date_col}, url: {url_col}")
        
        # Combine title and text for better search (but text is the main content now)
        df['combined_text'] = df['text'].str.strip()  # For chunks, text is the main content
        
        # Filter out very short chunks (but be more lenient than before)
        min_length = 100  # Chunks should have substantial content
        original_count = len(df)
        df = df[df['combined_text'].str.len() >= min_length]
        filtered_count = len(df)
        
        self.logger.info(f"Filtered short chunks: {original_count} -> {filtered_count} (min_length={min_length})")
        
        return df
    
    def download_all_datasets(self, max_samples_per_dataset: Optional[int] = None):
        """Download all configured datasets"""
        
        self.logger.info("Starting download of all datasets...")
        
        for dataset_key in self.dataset_configs.keys():
            try:
                self.logger.info(f"Processing {dataset_key}...")
                df = self.download_dataset(dataset_key, max_samples_per_dataset)
                self.logger.info(f"Successfully loaded {len(df)} documents from {dataset_key}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {dataset_key}: {str(e)}")
                continue
        
        # Save summary
        self._save_dataset_summary()
        self.logger.info("Dataset download complete!")
    
    def _save_dataset_summary(self):
        """Save summary of downloaded datasets"""
        summary = {}
        total_docs = 0
        
        for key, df in self.datasets.items():
            if len(df) > 0:
                summary[key] = {
                    'document_count': len(df),
                    'authority_weight': self.dataset_configs[key]['authority_weight'],
                    'description': self.dataset_configs[key]['description'],
                    'sample_titles': df['title'].head(3).tolist(),
                    'avg_text_length': df['text'].str.len().mean(),
                    'sample_text_snippet': df['text'].iloc[0][:200] + "..." if len(df) > 0 else ""
                }
                total_docs += len(df)
            else:
                summary[key] = {
                    'document_count': 0,
                    'error': 'No documents after filtering'
                }
        
        summary['total_documents'] = total_docs
        summary['datasets_loaded'] = [k for k, v in summary.items() if isinstance(v, dict) and v.get('document_count', 0) > 0]
        
        summary_file = os.path.join(self.data_dir, 'dataset_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Dataset summary saved to {summary_file}")
        self.logger.info(f"Total documents loaded: {total_docs}")
    
    def get_dataset(self, dataset_key: str) -> Optional[pd.DataFrame]:
        """Get a specific dataset"""
        return self.datasets.get(dataset_key)
    
    def get_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Get all loaded datasets"""
        return self.datasets
    
    def get_combined_dataset(self) -> pd.DataFrame:
        """Combine all datasets into single DataFrame"""
        if not self.datasets:
            raise ValueError("No datasets loaded. Call download_all_datasets() first.")
        
        combined_dfs = []
        for key, df in self.datasets.items():
            if len(df) > 0:
                df_copy = df.copy()
                df_copy['dataset_key'] = key
                combined_dfs.append(df_copy)
        
        if not combined_dfs:
            raise ValueError("No valid datasets with content found.")
        
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        self.logger.info(f"Combined dataset contains {len(combined_df)} total documents")
        
        return combined_df
    
    def search_datasets(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search across all loaded datasets for relevant content
        """
        if not self.datasets:
            self.logger.warning("No datasets loaded for search")
            return []
        
        query_lower = query.lower()
        all_results = []
        
        for dataset_name, df in self.datasets.items():
            if len(df) == 0:
                continue
                
            # Search in text and title
            text_matches = df['text'].str.lower().str.contains(query_lower, na=False)
            title_matches = df['title'].str.lower().str.contains(query_lower, na=False)
            
            matches = text_matches | title_matches
            matching_rows = df[matches]
            
            for _, row in matching_rows.iterrows():
                result = {
                    'title': row['title'][:200],
                    'snippet': row['text'][:400] + "...",
                    'source': f"{dataset_name.title()} Database",
                    'link': row.get('url', f"local:{dataset_name}"),
                    'date': row.get('date', ''),
                    'dataset_source': dataset_name,
                    'authority_weight': row.get('authority_weight', 0.7),
                    'relevance_score': self._calculate_search_relevance(query_lower, row['text'].lower(), row['title'].lower())
                }
                all_results.append(result)
        
        # Sort by relevance and authority
        all_results.sort(key=lambda x: x['relevance_score'] * (1 + x['authority_weight']), reverse=True)
        
        return all_results[:max_results]
    
    def _calculate_search_relevance(self, query: str, text: str, title: str) -> float:
        """Calculate relevance score for search results"""
        score = 0.0
        
        # Title matches get higher weight
        if query in title:
            score += 0.5
        
        # Text matches
        text_count = text.count(query)
        score += min(text_count * 0.1, 0.3)  # Cap contribution
        
        # Bonus for exact phrase matches
        if len(query) > 10 and query in text:
            score += 0.3
        
        return min(score, 1.0)


if __name__ == "__main__":
    # Test the dataset manager with chunks
    print("Testing Dataset Manager with Chunks Datasets...")
    
    manager = DatasetManager()
    
    # First, explore hansard chunks structure
    print("\n1. Exploring hansard CHUNKS dataset structure...")
    try:
        manager.explore_dataset_structure('hansard')
    except Exception as e:
        print(f"Error exploring structure: {e}")
    
    print("\n" + "="*60)
    print("2. Exploring news CHUNKS dataset structure...")
    try:
        manager.explore_dataset_structure('news')
    except Exception as e:
        print(f"Error exploring structure: {e}")
    
    # Then try to download with small samples
    print("\n" + "="*60)
    print("3. Downloading small samples for testing...")
    try:
        # Start with hansard chunks (most likely to have English content)
        manager.download_dataset('hansard', max_samples=100)
        
        print("\n4. Dataset Summary:")
        for key, df in manager.get_all_datasets().items():
            print(f"   {key}: {len(df)} documents")
            if len(df) > 0:
                print(f"      Sample title: {df['title'].iloc[0][:100]}...")
                print(f"      Average text length: {df['text'].str.len().mean():.0f} characters")
                print(f"      Sample text snippet: {df['text'].iloc[0][:150]}...")
        
        if any(len(df) > 0 for df in manager.get_all_datasets().values()):
            print("\n5. Testing search functionality...")
            results = manager.search_datasets("president", max_results=3)
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['title'][:80]}... (score: {result['relevance_score']:.3f})")
        
        print("\nDataset Manager test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()