"""Hybrid retrieval system combining local datasets and web search"""
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
sys.path.append('.')
from config import (
    INDICES_DIR, EMBEDDING_MODEL, TOP_K_PER_DATASET, 
    RELEVANCE_THRESHOLD, get_datasets_for_claim_type, 
    DATASETS_CONFIG, USE_HYBRID_RETRIEVAL, HYBRID_FALLBACK_TO_SERPER
)
from typing import List, Dict, Any

"""Hybrid retrieval system combining local datasets and web search"""
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import re
from typing import List, Dict, Any
import sys
sys.path.append('.')
from config import (
    INDICES_DIR, EMBEDDING_MODEL, TOP_K_PER_DATASET, 
    RELEVANCE_THRESHOLD, get_datasets_for_claim_type, 
    DATASETS_CONFIG, USE_HYBRID_RETRIEVAL, HYBRID_FALLBACK_TO_SERPER
)

# Passage Extractor Class

class PassageExtractor:
    """Extract relevant passages from long documents for better NLI performance"""
    
    def __init__(self, max_passage_length: int = 1000):
        """
        Initialize passage extractor
        
        Args:
            max_passage_length: Maximum length of extracted passage in characters
        """
        self.max_passage_length = max_passage_length
        
        # Common stopwords to ignore
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'it', 'its', 'they', 'their', 'them'
        }
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query
        
        Args:
            query: Search query or claim
            
        Returns:
            List of important keywords
        """
        # Tokenize
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter stopwords and short words
        keywords = [
            word for word in words 
            if word not in self.stopwords and len(word) > 2
        ]
        
        return keywords
    
    def score_sentence(self, sentence: str, keywords: List[str]) -> float:
        """
        Score a sentence based on keyword matches
        
        Args:
            sentence: Sentence to score
            keywords: List of keywords to match
            
        Returns:
            Score (number of keyword matches)
        """
        sentence_lower = sentence.lower()
        score = sum(1 for keyword in keywords if keyword in sentence_lower)
        return score
    
    def extract_passage(self, text: str, query: str) -> str:
        """
        Extract most relevant passage from text
        
        Args:
            text: Full document text
            query: Query or claim
            
        Returns:
            Extracted relevant passage
        """
        # If text is already short enough, return as-is
        if len(text) <= self.max_passage_length:
            return text
        
        # Extract keywords from query
        keywords = self.extract_keywords(query)
        
        if not keywords:
            # No keywords - return beginning of text
            return text[:self.max_passage_length]
        
        # Split into sentences (simple approach)
        # Split on '. ' but keep the period
        sentences = []
        for part in text.split('. '):
            if part.strip():
                sentences.append(part.strip() + '.')
        
        # If only one sentence, return it (truncated if needed)
        if len(sentences) <= 1:
            return text[:self.max_passage_length]
        
        # Score each sentence
        scored_sentences = [
            (self.score_sentence(sent, keywords), sent) 
            for sent in sentences
        ]
        
        # Sort by score (highest first)
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        # Build passage from highest-scored sentences
        passage_sentences = []
        current_length = 0
        
        for score, sentence in scored_sentences:
            # Stop if no keywords match
            if score == 0:
                break
            
            # Stop if adding this sentence exceeds max length
            if current_length + len(sentence) > self.max_passage_length:
                # Check if we have at least one sentence
                if passage_sentences:
                    break
                else:
                    # Include at least partial first sentence
                    remaining = self.max_passage_length - current_length
                    passage_sentences.append(sentence[:remaining])
                    break
            
            passage_sentences.append(sentence)
            current_length += len(sentence)
        
        # If no high-scoring sentences found, use beginning
        if not passage_sentences:
            return text[:self.max_passage_length]
        
        # Join sentences
        passage = ' '.join(passage_sentences)
        
        return passage
    
    def extract_passages_from_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Extract passages for all results
        
        Args:
            results: List of search results
            query: Search query
            
        Returns:
            Results with added 'passage' field
        """
        for result in results:
            full_text = result.get('text', '')
            
            # Extract relevant passage
            passage = self.extract_passage(full_text, query)
            
            # Add to result
            result['passage'] = passage
            result['full_text'] = full_text  # Keep original for reference
            result['passage_extracted'] = len(passage) < len(full_text)
        
        return results

class HybridRetriever:
    """Hybrid retrieval combining local datasets and web search"""
    
    def __init__(self):
        """Initialize hybrid retriever"""
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.loaded_indices = {}
        self.loaded_metadata = {}

        self.passage_extractor = PassageExtractor(max_passage_length=1000)
        
        print(f"Hybrid Retriever initialized")
        print(f"   Embedding model: {EMBEDDING_MODEL}")
        print(f"   Hybrid mode: {USE_HYBRID_RETRIEVAL}")
        print(f"   Serper fallback: {HYBRID_FALLBACK_TO_SERPER}")
        print(f"   Passage extraction: Enabled (max 1000 chars)")
        
        # Load available indices
        self._load_available_indices()
    
    def _load_available_indices(self):
        """Load all available FAISS indices"""
        print(f"\nLoading available indices from {INDICES_DIR}...")
        
        if not INDICES_DIR.exists():
            print(f"   ERROR: Indices directory not found: {INDICES_DIR}")
            return
        
        for index_file in INDICES_DIR.glob("*.index"):
            dataset_name = index_file.stem
            metadata_file = INDICES_DIR / f"{dataset_name}.metadata.pkl"
            
            if metadata_file.exists():
                try:
                    # Load FAISS index
                    index = faiss.read_index(str(index_file))
                    
                    # Load metadata
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                    
                    self.loaded_indices[dataset_name] = index
                    self.loaded_metadata[dataset_name] = metadata
                    
                    doc_count = metadata['total_documents']
                    print(f"{dataset_name}: {doc_count:,} documents")
                    
                except Exception as e:
                    print(f"Failed to load {dataset_name}: {e}")
        
        total_docs = sum(meta['total_documents'] for meta in self.loaded_metadata.values())
        print(f"Total: {len(self.loaded_indices)} indices, {total_docs:,} documents")
    
    def search_datasets(self, query: str, claim_types: List[str] = None, top_k: int = None) -> List[Dict]:
        """Search local datasets for relevant documents with passage extraction"""
        if not USE_HYBRID_RETRIEVAL or not self.loaded_indices:
            return []
        
        if top_k is None:
            top_k = TOP_K_PER_DATASET
        
        print(f"\nSearching local datasets for: '{query[:100]}...'")
        
        # Determine which datasets to search
        if claim_types:
            relevant_datasets = get_datasets_for_claim_type(claim_types)
            print(f"   Claim types {claim_types} → datasets: {relevant_datasets}")
        else:
            relevant_datasets = list(self.loaded_indices.keys())
            print(f"   Searching all available datasets: {relevant_datasets}")
        
        # Filter to loaded datasets
        searchable_datasets = [name for name in relevant_datasets if name in self.loaded_indices]
        
        if not searchable_datasets:
            print(f"No relevant datasets loaded")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        
        all_results = []
        
        # Search each relevant dataset
        for dataset_name in searchable_datasets:
            try:
                index = self.loaded_indices[dataset_name]
                metadata = self.loaded_metadata[dataset_name]
                
                # Search FAISS index
                scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
                
                # Process results
                dataset_results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:  # FAISS returns -1 for missing results
                        continue
                    
                    if score < RELEVANCE_THRESHOLD:
                        continue
                    
                    # Get document metadata and text
                    doc_metadata = metadata['metadata'][idx]
                    doc_text = metadata['texts'][idx]
                    
                    result = {
                        'text': doc_text,
                        'title': doc_metadata.get('title', ''),
                        'url': doc_metadata.get('url', ''),
                        'date': doc_metadata.get('date', ''),
                        'source': f"{dataset_name} ({DATASETS_CONFIG[dataset_name]['description']})",
                        'authority': DATASETS_CONFIG[dataset_name]['authority'],
                        'similarity_score': float(score),
                        'dataset': dataset_name,
                        'retrieval_method': 'local_dataset'
                    }
                    
                    dataset_results.append(result)
                
                print(f"{dataset_name}: {len(dataset_results)} results")
                all_results.extend(dataset_results)
                
            except Exception as e:
                print(f"Error searching {dataset_name}: {e}")
        
        #Extract passages from all results
        print(f"\nExtracting relevant passages from {len(all_results)} results...")
        all_results = self.passage_extractor.extract_passages_from_results(all_results, query)
        
        # Count how many needed extraction
        extracted_count = sum(1 for r in all_results if r.get('passage_extracted', False))
        print(f"Extracted passages from {extracted_count}/{len(all_results)} long documents")
        
        # Sort by authority and similarity
        all_results.sort(key=lambda x: (x['authority'], x['similarity_score']), reverse=True)
        
        print(f"Total dataset results: {len(all_results)}")
        
        return all_results
    
    def _generate_query_variations(self, query: str, max_variations: int = 3) -> List[str]:
        """
        Generate query variations for better coverage
        
        Args:
            query: Original query
            max_variations: Maximum number of variations to generate
            
        Returns:
            List of query variations including original
        """
        variations = [query]  # Always include original
        
        # Simple synonym replacement
        # You can expand this with more sophisticated methods
        synonyms = {
            'said': ['stated', 'announced', 'declared', 'mentioned'],
            'president': ['presidential', 'head of state'],
            'government': ['administration', 'cabinet', 'authorities'],
            'animals': ['pets', 'livestock'],
            'shortage': ['scarcity', 'deficit', 'lack'],
            'allocated': ['approved', 'designated', 'assigned', 'earmarked'],
            'billion': ['bn', 'b'],
            'parliament': ['parliamentary', 'legislature'],
            'approval': ['authorization', 'consent', 'permission']
        }
        
        query_lower = query.lower()
        
        for original, replacements in synonyms.items():
            if original in query_lower:
                # Create variation with first synonym
                if len(variations) < max_variations:
                    variation = query_lower.replace(original, replacements[0])
                    if variation not in variations:
                        variations.append(variation)
        
        return variations[:max_variations]
    
    def search_web(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search web using existing Serper integration"""
        print(f"\nSearching web for: '{query[:100]}...'")
        
        try:
            # Try to import your existing evidence retrieval
            try:
                from evidence_retrieval import retrieve_evidence
                
                # Use your existing Serper-based search
                web_results = retrieve_evidence(query, num_sources=num_results)
                
                # Convert to consistent format
                formatted_results = []
                for result in web_results:
                    formatted_result = {
                        'text': result.get('snippet', result.get('content', '')),
                        'title': result.get('title', ''),
                        'url': result.get('url', result.get('link', '')),
                        'date': result.get('date', ''),
                        'source': result.get('source', 'Web Search'),
                        'authority': 0.5,  # Lower authority than government sources
                        'similarity_score': 0.8,  # Default web relevance
                        'dataset': 'web_search',
                        'retrieval_method': 'web_search'
                    }
                    formatted_results.append(formatted_result)
                
                print(f"Web results: {len(formatted_results)}")
                return formatted_results
                
            except ImportError:
                print(f" evidence_retrieval module not found - skipping web search")
                return []
                
        except Exception as e:
            print(f" Web search error: {e}")
            return []
    
    def hybrid_search(self, query: str, claim_types: List[str] = None, 
                     total_results: int = 20, use_query_expansion: bool = True) -> List[Dict]:
        """
        Perform hybrid search combining local datasets and web search
        
        Args:
            query: Search query
            claim_types: List of claim types to determine dataset priority
            total_results: Total number of results to return
            use_query_expansion: Whether to use query expansion for better recall
            
        Returns:
            List of relevant documents with metadata
        """
        print(f"\n{'='*80}")
        print(f"HYBRID SEARCH")
        print(f"{'='*80}")
        print(f"Query: '{query}'")
        print(f"Claim types: {claim_types}")
        print(f"Target results: {total_results}")
        print(f"Query expansion: {use_query_expansion}")
        
        all_results = []
        
        # NEW: Generate query variations for better coverage
        if use_query_expansion:
            queries = self._generate_query_variations(query, max_variations=3)
            print(f"\nQuery expansion enabled:")
            for i, q in enumerate(queries, 1):
                print(f"   {i}. {q}")
        else:
            queries = [query]
        
        # 1. Search local datasets with each query variation
        if USE_HYBRID_RETRIEVAL and self.loaded_indices:
            print(f"\nSearching local datasets...")
            
            for q in queries:
                dataset_results = self.search_datasets(
                    q, 
                    claim_types, 
                    top_k=TOP_K_PER_DATASET
                )
                all_results.extend(dataset_results)
            
            print(f"\nTotal dataset results (before dedup): {len(all_results)}")
        
        # 2. Search web if needed
        web_results_needed = max(0, total_results - len(all_results))
        
        if (web_results_needed > 0 and HYBRID_FALLBACK_TO_SERPER) or not USE_HYBRID_RETRIEVAL:
            print(f"\nSearching web (need {web_results_needed} more results)...")
            web_results = self.search_web(query, num_results=web_results_needed)
            
            # Extract passages from web results too
            if web_results:
                web_results = self.passage_extractor.extract_passages_from_results(
                    web_results, 
                    query
                )
            
            all_results.extend(web_results)
            print(f"Web results: {len(web_results)}")
        
        # 3. Deduplicate
        print(f"\nDeduplicating results...")
        unique_results = self._deduplicate_results(all_results)
        print(f"   {len(all_results)} → {len(unique_results)} unique results")
        
        # 4. Rank by authority and similarity
        unique_results.sort(
            key=lambda x: (x['authority'], x['similarity_score']), 
            reverse=True
        )
        
        # 5. Limit to requested number
        final_results = unique_results[:total_results]
        
        # 6. Add metadata
        for i, result in enumerate(final_results):
            result['rank'] = i + 1
            result['combined_score'] = result['authority'] * result['similarity_score']
        
        # 7. Print summary
        print(f"\n{'='*80}")
        print(f"SEARCH RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Total results: {len(final_results)}")
        print(f"\nSource breakdown:")
        
        source_counts = {}
        for result in final_results:
            dataset = result['dataset']
            source_counts[dataset] = source_counts.get(dataset, 0) + 1
        
        for dataset, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            authority = DATASETS_CONFIG.get(dataset, {}).get('authority', 0.5)
            print(f"   {dataset}: {count} results (authority: {authority})")
        
        print(f"\nTop 3 results:")
        for i, result in enumerate(final_results[:3], 1):
            print(f"   {i}. [{result['dataset']}] {result['title'][:60]}...")
            print(f"      Authority: {result['authority']}, Score: {result['similarity_score']:.3f}")
            print(f"      Passage length: {len(result.get('passage', ''))} chars")
        
        return final_results
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on URL and title similarity"""
        unique_results = []
        seen_urls = set()
        seen_titles = set()
        
        for result in results:
            url = result.get('url', '').strip()
            title = result.get('title', '').strip().lower()
            
            # Skip if we've seen this URL
            if url and url in seen_urls:
                continue
            
            # Skip if we've seen this exact title
            if title and title in seen_titles:
                continue
            
            # Add to unique results
            unique_results.append(result)
            
            if url:
                seen_urls.add(url)
            if title:
                seen_titles.add(title)
        
        return unique_results
    
    def explain_result(self, result: Dict) -> str:
        """
        Generate human-readable explanation for why a result was retrieved
        
        Args:
            result: Search result dictionary
            
        Returns:
            Explanation string
        """
        dataset = result['dataset']
        authority = result['authority']
        score = result['similarity_score']
        
        # Authority description
        if authority >= 0.95:
            auth_desc = "highest authority (official government source)"
        elif authority >= 0.8:
            auth_desc = "high authority (verified source)"
        elif authority >= 0.6:
            auth_desc = "medium authority (news source)"
        else:
            auth_desc = "lower authority (web source)"
        
        # Similarity description
        if score >= 0.8:
            sim_desc = "very high semantic similarity"
        elif score >= 0.6:
            sim_desc = "high semantic similarity"
        elif score >= 0.4:
            sim_desc = "moderate semantic similarity"
        else:
            sim_desc = "low semantic similarity"
        
        # Was passage extracted?
        if result.get('passage_extracted', False):
            extraction_note = f"Relevant passage extracted ({len(result['passage'])} chars from {len(result['full_text'])} chars)"
        else:
            extraction_note = "Full document included (already concise)"
        
        explanation = f"""
        Retrieved from: {dataset} ({auth_desc})
        Similarity: {score:.2%} ({sim_desc})
        Extraction: {extraction_note}
        Combined score: {result.get('combined_score', 0):.3f}
        """.strip()
        
        return explanation
    
    def get_search_summary(self) -> Dict:
        """Get summary of search capabilities"""
        return {
            'hybrid_mode_enabled': USE_HYBRID_RETRIEVAL,
            'web_fallback_enabled': HYBRID_FALLBACK_TO_SERPER,
            'loaded_datasets': list(self.loaded_indices.keys()),
            'total_local_documents': sum(meta['total_documents'] for meta in self.loaded_metadata.values()),
            'embedding_model': EMBEDDING_MODEL,
            'available_claim_types': list(set().union(*[config['use_for'] for config in DATASETS_CONFIG.values()]))
        }

def main():
    """Test hybrid retriever"""
    retriever = HybridRetriever()
    
    # Show capabilities
    summary = retriever.get_search_summary()
    print(f"\nSEARCH CAPABILITIES:")
    print(f"   Hybrid mode: {summary['hybrid_mode_enabled']}")
    print(f"   Loaded datasets: {summary['loaded_datasets']}")
    print(f"   Total documents: {summary['total_local_documents']:,}")
    print(f"   Available claim types: {summary['available_claim_types']}")
    
    # Test searches
    test_queries = [
        ("cabinet decision on education", ['policy', 'government_policy']),
        ("presidential statement on economy", ['presidential', 'economic']),
        ("Sri Lanka parliament budget", ['parliamentary', 'economic'])
    ]
    
    print(f"\nTESTING SEARCHES:")
    
    for query, claim_types in test_queries:
        print(f"\n" + "="*60)
        print(f"Query: {query}")
        print(f"Claim types: {claim_types}")
        
        results = retriever.hybrid_search(query, claim_types, total_results=5)
        
        print(f"\nResults ({len(results)}):")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. [{result['dataset']}] {result['title'][:80]}...")
            print(f"     Authority: {result['authority']}, Score: {result['similarity_score']:.3f}")
            print(f"     Source: {result['source']}")
        
        if len(results) > 3:
            print(f"  ... and {len(results) - 3} more results")

if __name__ == "__main__":
    main()