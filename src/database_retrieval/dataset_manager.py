from datasets import load_dataset
import pandas as pd
import os
import json
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import logging
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Semantic search disabled.")

class DatasetManager:
    """Manages downloading and preparing Nuwan's Sri Lankan datasets"""
    
    def __init__(self, data_dir: str = "data/databases"):
        self.data_dir = data_dir
        self.datasets = {}
        self.embeddings = {}  # Store embeddings for semantic search
        self.embedding_model = None
        self.setup_logging()
        
        # Updated dataset configurations - using CHUNKS datasets for actual content
        self.dataset_configs = {
            # PRESIDENTIAL DATASETS (Highest Priority for Presidential Claims)
            'pmd_press_releases': {
                'hf_name': 'nuuuwan/lk-pmd-press-releases-chunks',
                'description': 'Presidential press releases and statements - Full Text',
                'authority_weight': 1.0,
                'priority': 1,
                'has_full_text': True,
                'claim_types': ['presidential', 'president_said', 'government_policy']
            },
            'pmd_press_release': {
                'hf_name': 'nuuuwan/lk-pmd-press-release-chunks',
                'description': 'PMD press release documents - Full Text',
                'authority_weight': 1.0,
                'priority': 1,
                'has_full_text': True,
                'claim_types': ['presidential', 'president_said', 'government_policy']
            },
            
            # CABINET & GOVERNMENT DATASETS
            'cabinet_decisions': {
                'hf_name': 'nuuuwan/lk-cabinet-decisions-chunks',
                'description': 'Cabinet decisions and government policies - Full Text',
                'authority_weight': 0.95,
                'priority': 1,
                'has_full_text': True,
                'claim_types': ['government_policy', 'cabinet_decision', 'ministerial']
            },
            
            # PARLIAMENTARY DATASETS
            'hansard': {
                'hf_name': 'nuuuwan/lk-hansard-2020s-chunks',  # Changed to chunks
                'description': 'Parliamentary debates and speeches (2020s) - Full Text',
                'authority_weight': 0.9,
                'priority': 1,
                'has_full_text': True,
                'claim_types': ['parliamentary', 'legislative', 'political']
            },
            'hansard_2010s': {
                'hf_name': 'nuuuwan/lk-hansard-2010s-chunks',
                'description': 'Parliamentary debates and speeches (2010s) - Full Text',
                'authority_weight': 0.85,
                'priority': 2,
                'has_full_text': True,
                'claim_types': ['parliamentary', 'legislative', 'political']
            },
            'hansard_2000s': {
                'hf_name': 'nuuuwan/lk-hansard-2000s-chunks',
                'description': 'Parliamentary debates and speeches (2000s) - Full Text',
                'authority_weight': 0.8,
                'priority': 3,
                'has_full_text': True,
                'claim_types': ['parliamentary', 'legislative', 'political']
            },
            
            # JUDICIAL DATASETS
            'supreme_court': {
                'hf_name': 'nuuuwan/lk-supreme-court-judgements-chunks',
                'description': 'Supreme Court judgments and legal decisions - Full Text',
                'authority_weight': 1.0,
                'priority': 1,
                'has_full_text': True,
                'claim_types': ['legal', 'judicial', 'constitutional']
            },
            'appeal_court': {
                'hf_name': 'nuuuwan/lk-appeal-court-judgements-chunks',
                'description': 'Appeal Court judgments and legal decisions - Full Text',
                'authority_weight': 0.9,
                'priority': 2,
                'has_full_text': True,
                'claim_types': ['legal', 'judicial']
            },
            # MINISTRY & DEPARTMENT DATASETS
            'police': {
                'hf_name': 'nuuuwan/lk-police-press-releases-chunks',
                'description': 'Police press releases and statements - Full Text',
                'authority_weight': 0.85,
                'priority': 2,
                'has_full_text': True,
                'claim_types': ['law_enforcement', 'security', 'crime']
            },
            'treasury': {
                'hf_name': 'nuuuwan/lk-treasury-press-releases-chunks',
                'description': 'Treasury economic statements and reports - Full Text',
                'authority_weight': 0.9,
                'priority': 2,
                'has_full_text': True,
                'claim_types': ['economic', 'financial', 'budget', 'fiscal']
            },
            # NEWS & MEDIA DATASETS
            'news': {
                'hf_name': 'nuuuwan/lk-news-chunks',
                'description': 'Sri Lankan news articles - Full Text',
                'authority_weight': 0.6,
                'priority': 4,
                'has_full_text': True,
                'claim_types': ['general', 'media_reports']
            },
            
            # SECTOR-SPECIFIC DATASETS
            'fisheries_weekly': {
                'hf_name': 'nuuuwan/lk-fisheries-weekly-fish-prices-reports-chunks',
                'description': 'Weekly fisheries reports and fish prices - Full Text',
                'authority_weight': 0.8,
                'priority': 3,
                'has_full_text': True,
                'claim_types': ['fisheries', 'agriculture', 'sector_specific']
            },
            'tourism_weekly': {
                'hf_name': 'nuuuwan/lk-tourism-weekly-reports-chunks',
                'description': 'Tourism industry weekly reports - Full Text',
                'authority_weight': 0.8,
                'priority': 3,
                'has_full_text': True,
                'claim_types': ['tourism', 'sector_specific']
            }
        }
        
        os.makedirs(data_dir, exist_ok=True)
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def classify_claim_type(self, claim: str) -> List[str]:
        """Classify claim to determine which datasets to prioritize"""
        claim_lower = claim.lower()
        claim_types = []
        
        # Presidential claims (HIGHEST PRIORITY)
        if any(term in claim_lower for term in ['president said', 'president announced', 'president declared', 
                                               'presidential statement', 'president\'s office', 'pmd']):
            claim_types.append('presidential')
        
        # Government policy claims
        if any(term in claim_lower for term in ['government policy', 'cabinet decision', 'cabinet approved',
                                               'minister announced', 'ministry statement', 'government declared']):
            claim_types.append('government_policy')
        
        # Parliamentary claims
        if any(term in claim_lower for term in ['parliament', 'hansard', 'mp said', 'parliamentary debate',
                                               'speaker announced', 'parliamentary committee']):
            claim_types.append('parliamentary')
        
        # Legal/Judicial claims
        if any(term in claim_lower for term in ['supreme court', 'appeal court', 'court ruled', 'judgment',
                                               'legal decision', 'constitutional', 'court order']):
            claim_types.append('legal')
        
        # Economic/Financial claims
        if any(term in claim_lower for term in ['budget', 'treasury', 'economic policy', 'fiscal', 'tax',
                                               'inflation', 'currency', 'central bank', 'financial']):
            claim_types.append('economic')
        
        # Security/Law enforcement claims
        if any(term in claim_lower for term in ['police', 'security', 'crime', 'arrest', 'investigation',
                                               'law enforcement', 'criminal']):
            claim_types.append('law_enforcement')
        
        # Sector-specific claims
        if any(term in claim_lower for term in ['fisheries', 'fishing', 'fish prices', 'marine']):
            claim_types.append('fisheries')
        if any(term in claim_lower for term in ['tourism', 'tourist', 'hotel', 'travel']):
            claim_types.append('tourism')
        
        # If no specific type identified, mark as general
        if not claim_types:
            claim_types.append('general')
        
        self.logger.info(f"Claim classified as: {claim_types}")
        return claim_types
    
    def route_datasets_for_claim(self, claim: str) -> List[str]:
        """Route claim to most appropriate datasets in priority order"""
        claim_types = self.classify_claim_type(claim)
        
        # Build priority-ordered dataset list
        routed_datasets = []
        
        # Priority 1: Exact match datasets (highest relevance)
        for dataset_key, config in self.dataset_configs.items():
            dataset_claim_types = config.get('claim_types', [])
            if any(ct in dataset_claim_types for ct in claim_types):
                if config['priority'] == 1:
                    routed_datasets.append((dataset_key, 1.0))  # Full weight
        
        # Priority 2: Secondary relevant datasets
        for dataset_key, config in self.dataset_configs.items():
            dataset_claim_types = config.get('claim_types', [])
            if any(ct in dataset_claim_types for ct in claim_types):
                if config['priority'] == 2:
                    routed_datasets.append((dataset_key, 0.8))  # Reduced weight
        
        # Priority 3: General datasets (if no specific matches or as backup)
        if not routed_datasets or 'general' in claim_types:
            for dataset_key, config in self.dataset_configs.items():
                if dataset_key not in [d[0] for d in routed_datasets]:
                    if config['priority'] >= 3:
                        routed_datasets.append((dataset_key, 0.6))  # Lower weight
        
        # Sort by priority and authority weight
        routed_datasets.sort(key=lambda x: (
            self.dataset_configs[x[0]]['priority'], 
            -self.dataset_configs[x[0]]['authority_weight']
        ))
        
        dataset_names = [d[0] for d in routed_datasets]
        self.logger.info(f"Routing claim to datasets: {dataset_names[:5]}...")  # Show top 5
        
        return dataset_names
    
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
                print(f"\nDataset Structure for {dataset_key} ({config['hf_name']}):")
                print(f"Available columns: {list(examples[0].keys())}")
                print(f"\nSample data:")
                
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
        """COMPREHENSIVE English content extraction - don't waste valuable evidence!"""
        original_count = len(df)
        
        print(f"Processing {original_count} documents for English content...")
        
        # STRATEGY 1: Language Column Analysis (More Inclusive)
        if 'lang' in df.columns:
            lang_distribution = df['lang'].value_counts()
            print(f"   Language distribution: {dict(lang_distribution.head(10))}")
            
            # INCLUSIVE English patterns - keep multilingual content too
            inclusive_patterns = [
                'en',           # Pure English
                'english', 
                'si-ta-en',     # Multilingual with English
                'ta-en',        # Tamil-English 
                'en-si',        # English-Sinhala
                'en-ta',        # English-Tamil
                'mix',          # Mixed language
                'multi'         # Multilingual
            ]
            
            # Create inclusive mask
            lang_mask = df['lang'].str.contains('|'.join(inclusive_patterns), na=False, case=False)
            english_by_lang = df[lang_mask]
            
            if len(english_by_lang) > 0:
                print(f"Language filtering kept: {len(english_by_lang)}/{original_count} documents")
            else:
                print(f"Language filtering found no matches, proceeding with content analysis...")
                english_by_lang = df  # Keep all for content analysis
        else:
            print(f"No language column - analyzing all content")
            english_by_lang = df
        
        # STRATEGY 2: SMART Content Analysis (Most Important)
        if 'chunk_text' in df.columns or 'text' in df.columns:
            
            # Determine text column
            text_col = 'chunk_text' if 'chunk_text' in df.columns else 'text'
            print(f"Analyzing content in '{text_col}' column...")
            
            def smart_english_detection(text):
                """SMART English detection - don't be overly picky!"""
                if not isinstance(text, str):
                    return False
                
                text_clean = text.strip()
                if len(text_clean) < 30:  # Too short to analyze properly
                    return False
                
                text_lower = text_clean.lower()
                
                # COUNT 1: English function words (very reliable indicators)
                function_words = [
                    'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'that', 'was', 'for', 'are', 'as', 'with',
                    'his', 'they', 'be', 'at', 'one', 'have', 'this', 'from', 'by', 'not', 'but', 'what', 'all',
                    'will', 'would', 'could', 'should', 'can', 'may', 'must'
                ]
                
                function_word_count = sum(1 for word in function_words if f' {word} ' in f' {text_lower} ')
                
                # COUNT 2: Sri Lankan context words (highly relevant)
                sri_lankan_indicators = [
                    'sri lanka', 'president', 'minister', 'government', 'parliament', 'cabinet', 'ministry',
                    'honorable', 'hon', 'mp', 'committee', 'member', 'speaker', 'prime minister',
                    'colombo', 'rupees', 'lkr', 'central bank', 'cbsl'
                ]
                
                context_count = sum(1 for phrase in sri_lankan_indicators if phrase in text_lower)
                
                # COUNT 3: Latin script ratio
                latin_chars = sum(1 for c in text_clean if c.isalpha() and ord(c) < 256)
                total_alpha_chars = sum(1 for c in text_clean if c.isalpha())
                latin_ratio = latin_chars / max(total_alpha_chars, 1)
                
                # DECISION LOGIC (Be more inclusive!)
                is_english = (
                    function_word_count >= 4 or                                    # Clear English markers
                    (function_word_count >= 2 and context_count >= 1) or          # Some English + Sri Lankan context
                    (context_count >= 2) or                                       # Strong Sri Lankan context
                    (latin_ratio > 0.85 and len(text_clean) > 100) or            # Mostly Latin script
                    (function_word_count >= 1 and latin_ratio > 0.9)             # Some English + mostly Latin
                )
                
                return is_english
            
            # Apply smart detection
            print(f"Running smart English detection on {len(df)} documents...")
            english_mask = df[text_col].apply(smart_english_detection)
            english_count = english_mask.sum()
            
            print(f"Smart detection found {english_count} English documents ({english_count/len(df)*100:.1f}%)")
            
            if english_count > 0:
                df = df[english_mask]
            else:
                # FALLBACK: If still nothing, take documents with highest Latin ratio
                print(f"FALLBACK: Taking documents with highest English character ratio...")
                
                def latin_ratio_fallback(text):
                    if not isinstance(text, str) or len(text) < 20:
                        return 0
                    latin_chars = sum(1 for c in text if c.isalpha() and ord(c) < 256)
                    total_chars = sum(1 for c in text if c.isalpha())
                    return latin_chars / max(total_chars, 1)
                
                df['latin_score'] = df[text_col].apply(latin_ratio_fallback)
                # Take top 70% by Latin ratio (much more inclusive)
                cutoff_count = max(50, int(len(df) * 0.7))  
                df = df.nlargest(cutoff_count, 'latin_score')
                df = df.drop('latin_score', axis=1)
                
                print(f"Fallback kept {len(df)} documents")
        
        else:
            print(f"No text content column found for analysis!")
        
        final_count = len(df)
        retention_rate = (final_count / original_count * 100) if original_count > 0 else 0
        
        print(f"FINAL: {original_count} → {final_count} documents ({retention_rate:.1f}% retention)")
        
        # Quality check
        if retention_rate < 20:
            print(f"WARNING: Low retention rate ({retention_rate:.1f}%). Dataset may need manual inspection.")
        elif retention_rate > 60:
            print(f"GOOD: High retention rate ({retention_rate:.1f}%). Quality filtering successful.")
        
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
    
    def search_datasets(self, query: str, max_results: int = 10, claim: Optional[str] = None) -> List[Dict]:
        """
        Enhanced search across datasets with claim routing
        """
        if not self.datasets:
            self.logger.warning("No datasets loaded for search")
            return []
        
        # If claim provided, use intelligent routing
        if claim:
            routed_datasets = self.route_datasets_for_claim(claim)
            search_datasets = [(name, 1.0) for name in routed_datasets if name in self.datasets]
        else:
            # Search all datasets with equal weight
            search_datasets = [(name, 1.0) for name in self.datasets.keys()]
        
        query_lower = query.lower()
        all_results = []
        
        for dataset_name, routing_weight in search_datasets:
            df = self.datasets[dataset_name]
            if len(df) == 0:
                continue
                
            # Enhanced search in text and title
            text_matches = df['text'].str.lower().str.contains(query_lower, na=False, regex=False)
            title_matches = df['title'].str.lower().str.contains(query_lower, na=False, regex=False)
            
            matches = text_matches | title_matches
            matching_rows = df[matches]
            
            for _, row in matching_rows.iterrows():
                relevance_score = self._calculate_search_relevance(query_lower, row['text'].lower(), row['title'].lower())
                authority_weight = row.get('authority_weight', 0.7)
                
                # Apply routing weight boost
                final_score = relevance_score * (1 + authority_weight) * routing_weight
                
                result = {
                    'title': row['title'][:200],
                    'snippet': row['text'][:500] + "...",
                    'source': f"{dataset_name.replace('_', ' ').title()} Database",
                    'link': row.get('url', f"local:{dataset_name}"),
                    'date': row.get('date', ''),
                    'dataset_source': dataset_name,
                    'authority_weight': authority_weight,
                    'routing_weight': routing_weight,
                    'relevance_score': relevance_score,
                    'final_score': final_score
                }
                all_results.append(result)
        
        # Sort by final score (relevance * authority * routing)
        all_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        self.logger.info(f"Found {len(all_results)} results across {'routed' if claim else 'all'} datasets")
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
    
    def initialize_semantic_search(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize semantic search with sentence embeddings"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("Sentence transformers not available. Install with: pip install sentence-transformers")
            return False
        
        try:
            self.logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            return True
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            return False
    
    def build_embeddings_for_dataset(self, dataset_key: str, force_rebuild: bool = False):
        """Build embeddings for semantic search on a specific dataset"""
        if not self.embedding_model:
            self.logger.warning("Embedding model not initialized. Call initialize_semantic_search() first")
            return False
        
        if dataset_key not in self.datasets:
            self.logger.warning(f"Dataset {dataset_key} not loaded")
            return False
        
        embedding_file = os.path.join(self.data_dir, f"{dataset_key}_embeddings.npy")
        
        # Load cached embeddings if available
        if os.path.exists(embedding_file) and not force_rebuild:
            self.logger.info(f"Loading cached embeddings for {dataset_key}")
            self.embeddings[dataset_key] = np.load(embedding_file)
            return True
        
        # Build embeddings
        df = self.datasets[dataset_key]
        if len(df) == 0:
            self.logger.warning(f"No data in dataset {dataset_key}")
            return False
        
        self.logger.info(f"Building embeddings for {len(df)} documents in {dataset_key}...")
        
        # Combine title and text for better semantic representation
        texts_to_embed = []
        for _, row in df.iterrows():
            combined_text = f"{row['title']} {row['text'][:1000]}"  # Limit to 1000 chars for efficiency
            texts_to_embed.append(combined_text.strip())
        
        try:
            # Encode in batches to avoid memory issues
            batch_size = 32
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts_to_embed), batch_size), desc=f"Encoding {dataset_key}"):
                batch = texts_to_embed[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
            
            # Cache embeddings
            np.save(embedding_file, embeddings)
            self.embeddings[dataset_key] = embeddings
            
            self.logger.info(f"Built and cached embeddings for {dataset_key}: {embeddings.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error building embeddings for {dataset_key}: {e}")
            return False
    
    def build_all_embeddings(self, force_rebuild: bool = False):
        """Build embeddings for all loaded datasets"""
        if not self.initialize_semantic_search():
            self.logger.error("Failed to initialize semantic search")
            return False
        
        success_count = 0
        for dataset_key in self.datasets.keys():
            if self.build_embeddings_for_dataset(dataset_key, force_rebuild):
                success_count += 1
        
        self.logger.info(f"Successfully built embeddings for {success_count}/{len(self.datasets)} datasets")
        return success_count > 0
    
    def semantic_search_datasets(self, query: str, max_results: int = 10, claim: Optional[str] = None, 
                                similarity_threshold: float = 0.3) -> List[Dict]:
        """
        Perform semantic similarity search across datasets using embeddings
        """
        if not self.embedding_model:
            self.logger.warning("Semantic search not available. Falling back to keyword search")
            return self.search_datasets(query, max_results, claim)
        
        # Encode query
        try:
            query_embedding = self.embedding_model.encode([query])
        except Exception as e:
            self.logger.error(f"Error encoding query: {e}")
            return self.search_datasets(query, max_results, claim)  # Fallback
        
        # Determine datasets to search
        if claim:
            routed_datasets = self.route_datasets_for_claim(claim)
            search_datasets = [(name, 1.0) for name in routed_datasets if name in self.datasets and name in self.embeddings]
        else:
            search_datasets = [(name, 1.0) for name in self.datasets.keys() if name in self.embeddings]
        
        all_results = []
        
        for dataset_name, routing_weight in search_datasets:
            if dataset_name not in self.embeddings:
                self.logger.warning(f"No embeddings for {dataset_name}, skipping semantic search")
                continue
            
            df = self.datasets[dataset_name]
            embeddings = self.embeddings[dataset_name]
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
            # Find top matches above threshold
            top_indices = np.where(similarities >= similarity_threshold)[0]
            top_similarities = similarities[top_indices]
            
            # Sort by similarity
            sorted_indices = top_indices[np.argsort(top_similarities)[::-1]]
            
            for idx in sorted_indices[:max_results]:  # Limit per dataset
                row = df.iloc[idx]
                similarity_score = similarities[idx]
                authority_weight = row.get('authority_weight', 0.7)
                
                # Calculate final score combining similarity, authority, and routing
                final_score = similarity_score * (1 + authority_weight) * routing_weight
                
                result = {
                    'title': row['title'][:200],
                    'snippet': row['text'][:500] + "...",
                    'source': f"{dataset_name.replace('_', ' ').title()} Database (Semantic)",
                    'link': row.get('url', f"local:{dataset_name}"),
                    'date': row.get('date', ''),
                    'dataset_source': dataset_name,
                    'authority_weight': authority_weight,
                    'routing_weight': routing_weight,
                    'similarity_score': float(similarity_score),
                    'final_score': float(final_score),
                    'search_type': 'semantic'
                }
                all_results.append(result)
        
        # Sort by final score
        all_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        self.logger.info(f"Semantic search found {len(all_results)} results")
        return all_results[:max_results]
    
    def hybrid_search_datasets(self, query: str, max_results: int = 10, claim: Optional[str] = None) -> List[Dict]:
        """
        Perform hybrid search combining keyword and semantic search
        """
        results = []
        
        # Try semantic search first (more accurate)
        if self.embedding_model and any(name in self.embeddings for name in self.datasets.keys()):
            semantic_results = self.semantic_search_datasets(query, max_results//2, claim)
            for result in semantic_results:
                result['search_method'] = 'semantic'
            results.extend(semantic_results)
        
        # Add keyword search results
        keyword_results = self.search_datasets(query, max_results//2, claim)
        for result in keyword_results:
            result['search_method'] = 'keyword'
            # Avoid duplicates by checking if similar result exists
            is_duplicate = any(
                result['dataset_source'] == existing['dataset_source'] and 
                result['title'][:50] == existing['title'][:50]
                for existing in results
            )
            if not is_duplicate:
                results.append(result)
        
        # Re-sort combined results
        results.sort(key=lambda x: x.get('final_score', x.get('relevance_score', 0)), reverse=True)
        
        self.logger.info(f"Hybrid search returned {len(results[:max_results])} unique results")
        return results[:max_results]

    def download_full_dataset(self, dataset_key: str, force_reload: bool = False) -> pd.DataFrame:
        """Download FULL dataset without sampling limitations - for production use"""
        
        if dataset_key not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        config = self.dataset_configs[dataset_key]
        full_cache_file = os.path.join(self.data_dir, f"{dataset_key}_FULL.csv")
        
        # Check if full dataset already cached
        if os.path.exists(full_cache_file) and not force_reload:
            self.logger.info(f"Loading FULL cached {dataset_key} dataset...")
            df = pd.read_csv(full_cache_file)
            self.datasets[f"{dataset_key}_full"] = df
            return df
        
        self.logger.info(f"Downloading FULL {dataset_key} dataset (no sampling)...")
        
        try:
            # Load complete dataset
            dataset = load_dataset(config['hf_name'], split='train')
            df = dataset.to_pandas()
            
            self.logger.info(f"Full dataset loaded: {len(df)} documents")
            
            # Apply processing but NO sampling
            df = self._standardize_chunks_columns(df, dataset_key)
            df = self._filter_english_content_chunks(df)  # Now much more inclusive
            
            # Add metadata
            df['dataset_source'] = dataset_key
            df['authority_weight'] = config['authority_weight']
            df['priority'] = config['priority']
            
            self.logger.info(f"Full processed dataset: {len(df)} documents")
            
            # Cache full dataset
            if len(df) > 0:
                df.to_csv(full_cache_file, index=False)
                self.logger.info(f"Cached FULL {len(df)} documents from {dataset_key}")
            
            self.datasets[f"{dataset_key}_full"] = df
            return df
            
        except Exception as e:
            self.logger.error(f"Error downloading full {dataset_key}: {str(e)}")
            raise


if __name__ == "__main__":
    # Enhanced test for the improved dataset manager
    print("Testing Enhanced Dataset Manager with PMD and Semantic Search...")
    
    try:
        manager = DatasetManager()
        print("✅ DatasetManager initialized successfully")
        
        # Test 1: Explore PMD dataset structure (most important for Presidential claims)
        print("\\n1. Exploring PMD Presidential Press Releases dataset...")
        try:
            manager.explore_dataset_structure('pmd_press_releases')
        except Exception as e:
            print(f"Error exploring PMD structure: {e}")
            # Try alternative PMD dataset
            try:
                manager.explore_dataset_structure('pmd_press_release')
            except Exception as e2:
                print(f"Error exploring alternative PMD structure: {e2}")
        
        print("\n" + "="*80)
        print("2. Testing Claim Routing Logic...")
    
        # Test claims routing
        test_claims = [
            "President said giving rice to animals causes shortage",
            "Cabinet approved new economic policy",
            "Supreme Court ruled on constitutional matter",
            "Parliament debated new legislation"
        ]
        
        for claim in test_claims:
            claim_types = manager.classify_claim_type(claim)
            routed_datasets = manager.route_datasets_for_claim(claim)
            print(f"   Claim: '{claim[:50]}...'")
            print(f"   Types: {claim_types}")
            print(f"   Routed to: {routed_datasets[:3]}...")  # Show top 3
            print()
        
        print("\n" + "="*80)
        print("3. Downloading Priority Datasets (starting with PMD)...")
    
        # Download datasets in priority order
        priority_datasets = ['pmd_press_releases', 'pmd_press_release', 'cabinet_decisions', 'hansard']
        
        for dataset_key in priority_datasets:
            try:
                print(f"\nDownloading {dataset_key}...")
                manager.download_dataset(dataset_key, max_samples=50)  # Small sample for testing
                
            except Exception as e:
                print(f"Failed to load {dataset_key}: {str(e)}")
                continue
        
        print("\n" + "="*80)
        print("4. Testing Search Capabilities...")
    
        if any(len(df) > 0 for df in manager.get_all_datasets().values()):
            # Test 1: Presidential claim routing
            print("\n4.1 Testing Presidential Claim Search:")
            presidential_claim = "President said giving rice to animals causes shortage"
            results = manager.search_datasets("rice animals shortage", max_results=3, claim=presidential_claim)
            
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['title'][:80]}...")
                print(f"      Source: {result['source']} (Authority: {result['authority_weight']:.2f})")
                print(f"      Score: {result.get('final_score', result.get('relevance_score', 0)):.3f}")
                print()
            
            # Test 2: Initialize semantic search if possible
            print("\n4.2 Testing Semantic Search Setup:")
            try:
                if manager.initialize_semantic_search():
                    print("   ✅ Semantic search model loaded successfully")
                    
                    # Build embeddings for available datasets
                    embedded_count = 0
                    for dataset_key in manager.get_all_datasets().keys():
                        if manager.build_embeddings_for_dataset(dataset_key):
                            embedded_count += 1
                    
                    print(f"   ✅ Built embeddings for {embedded_count} datasets")
                    
                    if embedded_count > 0:
                        # Test semantic search
                        print("\n   Testing semantic search...")
                        semantic_results = manager.semantic_search_datasets(
                            "government policy on agriculture", 
                            max_results=3, 
                            claim=presidential_claim
                        )
                        
                        for i, result in enumerate(semantic_results, 1):
                            print(f"      {i}. {result['title'][:60]}... (sim: {result['similarity_score']:.3f})")
                    
            except Exception as e:
                print(f"   ❌ Semantic search setup failed: {e}")
                print("   💡 Install sentence-transformers: pip install sentence-transformers")
            
            print("\n4.3 Dataset Summary:")
            for key, df in manager.get_all_datasets().items():
                print(f"   📊 {key}: {len(df)} documents")
                if len(df) > 0:
                    print(f"      Authority weight: {manager.dataset_configs[key]['authority_weight']}")
                    print(f"      Claim types: {manager.dataset_configs[key].get('claim_types', ['general'])}")
                    print(f"      Sample: {df['title'].iloc[0][:60]}...")
                    print()
            
            print("🎯 Enhanced Dataset Manager test completed!")
            print("\n📋 Next Steps:")
            print("   1. Install sentence-transformers for semantic search")
            print("   2. Update evidence_retrieval.py to use hybrid_search_datasets()")
            print("   3. Test with real 'President said' claims")
            
        else:
            print("❌ No datasets loaded successfully. Check network connection and dataset names.")
    
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()