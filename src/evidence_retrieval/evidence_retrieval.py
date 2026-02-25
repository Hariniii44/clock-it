import spacy
import requests
import pandas as pd
from typing import List, Dict, Set
import re
from datetime import datetime
from urllib.parse import urlparse

from src.evidence_retrieval.claim_decomposer import ClaimDecomposer

class AdvancedEvidenceRetriever:
    def __init__(self, serper_key: str, gemini_key: str = None, groq_key: str = None):
        # self.serpapi_key = serpapi_key
        self.serper_key = serper_key
        self.groq_key = groq_key
        # self.base_url = "https://serpapi.com/search"
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print(" spaCy NER model loaded")
        except Exception as e:
            print(f" Could not load spaCy model: {e}")
            self.nlp = None

        self.source_categories = {
            # TIER 1: Official/Authority sources (highest priority)
            'official_primary': [
                'parliament.lk',           # Parliament Hansards
                'president.gov.lk',        # President Media Division
                'pmoffice.gov.lk',        # PM Office
                'cbsl.lk',                # Central Bank of Sri Lanka
                'statistics.gov.lk'        # Department of Statistics
            ],
            
            # TIER 2: Fact-checkers and verification sources
            'fact_checkers': [
                'factcheck.lk',           # Local fact-checker
                'factcrescendo.com',      # Regional fact-checker
                'boomlive.in',           # Regional verification
                'factly.in'              # Regional verification
            ],
            
            # TIER 3: Quality Sri Lankan news sources
            'sri_lankan_news': [
                'dailymirror.lk',
                'economynext.com', 
                'island.lk',
                'newsfirst.lk',
                'adaderana.lk',
                'ft.lk',
                'themorning.lk',
                'sundayobserver.lk'
            ],

            # TIER 4: Government information sources
            'government': [
                'dailynews.lk', 
                'news.lk',
                'agrimin.gov.lk',
                'treasury.gov.lk'
            ],
            
            # TIER 5: International sources (backup)
            'international': [
                'worldbank.org', 
                'imf.org', 
                'reuters.com', 
                'bbc.com',
                'ap.org'
            ]
        }

        # Priority weights for scoring
        self.priority_weights = {
            'official_primary': 3.0,    # Highest priority
            'fact_checkers': 2.5,      # Very high priority  
            'sri_lankan_news': 2.0,    # High priority
            'government': 1.5,         # Medium priority
            'international': 1.0,      # Normal priority
            'unknown': 0.5            # Lowest priority
        }

    def _advanced_deduplication(self, evidence_list: List[Dict]) -> List[Dict]:
        """Advanced deduplication using content similarity"""
        from difflib import SequenceMatcher
        
        unique_evidence = []
        seen_content = []
        
        for evidence in evidence_list:
            content = f"{evidence.get('title', '')} {evidence.get('snippet', '')}"
            
            # Check URL deduplication
            url = evidence.get('link', '')
            if any(url == seen.get('link', '') for seen in unique_evidence):
                continue
            
            # Check content similarity
            is_duplicate = False
            for seen_content_text in seen_content:
                similarity = SequenceMatcher(None, content.lower(), seen_content_text.lower()).ratio()
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_evidence.append(evidence)
                seen_content.append(content)
        
        print(f"Deduplication: {len(evidence_list)} → {len(unique_evidence)} unique pieces")
        return unique_evidence
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc.lower()
        except:
            return url.lower()


    def _simple_serper_query(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Simple Serper query for a single search term
        """
        import requests
        
        url = "https://google.serper.dev/search"
        payload = {
            "q": query,
            "num": max_results,
            "gl": "lk",  # Sri Lanka
            "hl": "en"   # English
        }
        headers = {
            "X-API-KEY": self.serper_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if "organic" in data:
                    for item in data["organic"]:
                        result = {
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "link": item.get("link", ""),
                            "source": self._extract_domain(item.get("link", "")),
                            "date": item.get("date", ""),
                            "position": item.get("position", 0),
                            "search_type": "hybrid_serper"
                        }
                        results.append(result)
                        
                filtered_results = self._filter_out_social_media(results)
                return filtered_results[:max_results]
                
            else:
                print(f" Search failed for query: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"  Search error: {e}")
            return []
        


    def retrieve_hybrid_serper_decomposition(self, claim: str, num_results: int = 20, 
                                   results_per_query: int = 10) -> List[Dict]:
        """
        Enhanced hybrid search with cross-encoder relevance filtering
        """
        print(f" Target final results: {num_results}")
        
        all_results = []
        
        # Step 1: Decompose claim into questions
        print(f" Decomposing claim...")
        decomposer = ClaimDecomposer(self.groq_key)
        questions = decomposer.decompose_claim(claim, num_questions=4)
        
        print(f" Generated {len(questions)} verification questions:")
        for i, q in enumerate(questions, 1):
            print(f"      Q{i}: {q[:80]}...")
        
        # Step 2: Search for original claim + each question
        search_queries = [claim] + questions
        
        print(f"  Searching {results_per_query} results per query...")
        
        for i, query in enumerate(search_queries):
            if i == 0:
                print(f" Original claim search...")
            else:
                print(f"  Question {i} search...")
            
            query_results = self._simple_serper_query(query, results_per_query)
            
            for result in query_results:
                result['query_source'] = 'original_claim' if i == 0 else f'question_{i}'
                result['query_text'] = query
            
            all_results.extend(query_results)
            print(f"    Retrieved {len(query_results)} results")
            
            import time
            time.sleep(0.5)
        
        print(f"    Retrieved {len(all_results)} total results from {len(search_queries)} queries")
        
        # Step 3: FILTER SOCIAL MEDIA FIRST
        print(f"      Pre-filtering social media sources...")
        all_results = self._filter_out_social_media(all_results)
        print(f"      After social media filtering: {len(all_results)} sources")

        # Step 3: Deduplicate
        unique_results = self._advanced_deduplication(all_results)
        print(f"    After deduplication: {len(unique_results)} unique results")
        
        # Step 4: Cross-encoder relevance filtering
        relevant_results = self._filter_by_cross_encoder_relevance(
            claim, 
            unique_results, 
            top_k=num_results,
            relevance_threshold=0.15  # Lower threshold to ensure we get results
        )
        
        # Step 5: Final ranking (optional, since cross-encoder already ranked)
        if len(relevant_results) > num_results:
            print(f"    Final ranking selection...")
            final_results = relevant_results[:num_results]
        else:
            final_results = relevant_results
        
        print(f"    Final selection: {len(final_results)} highly relevant evidence pieces")
        return final_results
    
    def _is_social_media_source(self, url: str) -> bool:
        """Check if a source is from social media platforms"""
        social_media_domains = [
            'facebook.com',
            'twitter.com', 
            'x.com',
            'instagram.com',
            'tiktok.com',
            'youtube.com',
            'youtu.be',
            'linkedin.com',
            'reddit.com',
            'pinterest.com',
            'snapchat.com',
            'telegram.org',
            'whatsapp.com'
        ]
        
        domain = self._extract_domain(url)
        return any(social_domain in domain for social_domain in social_media_domains)

    def _filter_out_social_media(self, results: List[Dict]) -> List[Dict]:
        """Remove social media sources completely"""
        filtered_results = []
        removed_count = 0
        
        for result in results:
            url = result.get('link', '')
            if self._is_social_media_source(url):
                removed_count += 1
                print(f"   Removed social media: {self._extract_domain(url)}")
            else:
                filtered_results.append(result)
        
        if removed_count > 0:
            print(f" Filtered out {removed_count} social media sources")
        
        return filtered_results

    def _filter_by_cross_encoder_relevance(self, claim: str, evidence_list: List[Dict], 
                                     top_k: int = 10, relevance_threshold: float = 0.3) -> List[Dict]:
        """
        Use cross-encoder to score relevance and filter evidence before verification
        """
        print(f"   Cross-encoder relevance filtering...")
        print(f"      Threshold: {relevance_threshold}, Target: {top_k} sources")
        
        if not evidence_list:
            return []

        # FIRST: Remove social media sources completely
        pre_filtered = self._filter_out_social_media(evidence_list)
        if len(pre_filtered) < len(evidence_list):
            print(f"      Pre-filtered: {len(evidence_list)} → {len(pre_filtered)} (removed social media)")

        # Import cross-encoder if not available
        try:
            from sentence_transformers import CrossEncoder
            if not hasattr(self, 'relevance_ranker'):
                print(f"      Loading cross-encoder for relevance...")
                self.relevance_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
                print(f"      Cross-encoder loaded")
        except Exception as e:
            print(f"     Cross-encoder loading failed: {e}")
            return evidence_list[:top_k]  # Fallback to first k results
        
        # Prepare claim-evidence pairs for scoring
        pairs = []
        valid_evidence = []
        
        for evidence in evidence_list:
            # Combine title and snippet for better context
            evidence_text = f"{evidence.get('title', '')} {evidence.get('snippet', '')}"
            evidence_text = evidence_text.strip()
            
            if evidence_text and len(evidence_text) > 10:  # Skip empty or very short evidence
                pairs.append([claim, evidence_text])
                valid_evidence.append(evidence)
        
        if not pairs:
            print(f"No valid evidence text found")
            return []
        
        print(f"Scoring {len(pairs)} evidence pieces...")
        
        # Get relevance scores
        try:
            scores = self.relevance_ranker.predict(pairs)
            
            # Add scores to evidence and sort by relevance
            scored_evidence = []
            for evidence, score in zip(valid_evidence, scores):
                base_score = float(score)  # Get base cross-encoder score
                
                # quality source binus
                source = evidence.get('source', '').lower()
                link = evidence.get('link', '').lower()
                
                # Academic/Research sources
                if any(academic in source for academic in [
                    'researchgate.net', 'academia.edu', 'scholar.google', 
                    'jstor.org', 'pubmed.ncbi', 'arxiv.org'
                ]):
                    base_score += 0.2
                    print(f" Academic bonus: {source}")
                
                # Government/Official sources (.gov.lk domains)
                elif any(official in source for official in [
                    'gov.lk', 'parliament.lk', 'president.gov.lk', 'cbsl.lk',
                    'statistics.gov.lk', 'treasury.gov.lk', 'pmoffice.gov.lk'
                ]):
                    base_score += 0.3
                    print(f"  Official source bonus: {source}")

                # International Organizations
                elif any(intl in source for intl in [
                    'worldbank.org', 'imf.org', 'fao.org', 'un.org',
                    'oecd.org', 'adb.org'
                ]):
                    base_score += 0.25
                    print(f"  International org bonus: {source}")
                
                # Quality News Sources (Sri Lankan)
                elif any(quality in source for quality in [
                    'economynext.com', 'ft.lk', 'themorning.lk'
                ]):
                    base_score += 0.1
                    print(f"  Quality news bonus: {source}")
                
                # Penalty for low-quality sources
                elif any(low_qual in source for low_qual in [
                    'facebook.com', 'twitter.com', 'instagram.com',
                    'tiktok.com', 'youtube.com'
                ]):
                    base_score -= 0.15
                    print(f"  Social media penalty: {source}")
                
                evidence['relevance_score'] = min(base_score, 10.0)
                
                # evidence['relevance_score'] = float(score)
                scored_evidence.append(evidence)
            
            # Sort by relevance score (highest first)
            scored_evidence.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Apply threshold and top-k filtering
            filtered_evidence = []
            for evidence in scored_evidence:
                if (len(filtered_evidence) < top_k and 
                    evidence['relevance_score'] >= relevance_threshold):
                    filtered_evidence.append(evidence)
            
            # If we don't have enough above threshold, take top results anyway
            if len(filtered_evidence) < min(5, top_k):  # Ensure at least 5 sources
                print(f" Only {len(filtered_evidence)} above threshold, taking top {top_k}")
                filtered_evidence = scored_evidence[:top_k]
            
            # Display filtering results
            print(f" Relevance scores:")
            for i, evidence in enumerate(filtered_evidence[:5], 1):
                score = evidence['relevance_score']
                title = evidence.get('title', 'No title')[:50]
                print(f"         {i}. {title}... (score: {score:.3f})")
            
            print(f"Selected {len(filtered_evidence)} most relevant sources")
            return filtered_evidence
            
        except Exception as e:
            print(f"Relevance scoring failed: {e}")
            return evidence_list[:top_k]  # Fallback

    def retrieve_with_hansard_fallback(self, claim: str, num_results: int = 10) -> List[Dict]:
        """
        Use DatasetManager as primary source for Sri Lankan parliamentary content
        """
        print(f" Searching parliamentary and news databases...")
        
        try:
            # Import and initialize DatasetManager
            from src.database_retrieval.dataset_manager import DatasetManager
            db_manager = DatasetManager()
            
            # Load hansard dataset (parliamentary debates)
            hansard_df = db_manager.get_dataset('hansard')
            if hansard_df is None or len(hansard_df) == 0:
                print(f"  Loading hansard dataset...")
                hansard_df = db_manager.download_dataset('hansard', max_samples=500)  # Increase sample size
            
            print(f"  Hansard dataset: {len(hansard_df)} parliamentary chunks available")
            
            # Search in parliamentary content
            parliamentary_results = self._search_parliamentary_content(claim, hansard_df, max_results=num_results)
            
            print(f" Found {len(parliamentary_results)} relevant parliamentary references")
            
            # Display top matches for debugging
            for i, result in enumerate(parliamentary_results[:3], 1):
                print(f"      {i}. {result['title'][:80]}... (score: {result['relevance_score']:.3f})")
            
            return parliamentary_results
            
        except Exception as e:
            print(f"  Parliamentary database search failed: {e}")
            return []

    def _search_parliamentary_content(self, claim: str, hansard_df: pd.DataFrame, max_results: int = 10) -> List[Dict]:
        """
        Search parliamentary hansard content for claim verification
        """
        import pandas as pd
        from difflib import SequenceMatcher
        
        claim_lower = claim.lower()
        claim_keywords = self._extract_claim_keywords(claim)
        
        scored_results = []
        
        print(f"  Searching {len(hansard_df)} parliamentary chunks for: '{claim[:50]}...'")
        
        for _, row in hansard_df.iterrows():
            chunk_text = str(row.get('text', ''))  # text is standardized to chunk_text
            title = str(row.get('title', ''))
            date = str(row.get('date', ''))
            chunk_id = str(row.get('url', ''))  # url is mapped to doc_id
            
            # Calculate parliamentary relevance
            relevance_score = self._calculate_parliamentary_relevance(
                claim_lower, chunk_text.lower(), title.lower(), claim_keywords
            )
            
            if relevance_score > 0.1:  # Lower threshold for parliamentary content
                scored_results.append({
                    'title': f"Parliamentary Debate - {date}" if date else "Parliamentary Debate",
                    'snippet': chunk_text[:400] + "..." if len(chunk_text) > 400 else chunk_text,
                    'source': 'Sri Lanka Parliament (Hansard)',
                    'link': f"parliament.lk/hansard/{chunk_id}",
                    'date': date,
                    'relevance_score': relevance_score,
                    'source_alignment': 'official_primary',  # Highest authority
                    'dataset_source': 'hansard'
                })
        
        # Sort by relevance
        scored_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        print(f" Parliamentary search complete: {len(scored_results)} relevant chunks found")
        return scored_results[:max_results]

    def _calculate_parliamentary_relevance(self, claim: str, content: str, title: str, keywords: List[str]) -> float:
        """
        Calculate relevance for parliamentary content
        """
        score = 0.0
        
        # 1. Direct keyword matching (weighted for parliamentary context)
        for keyword in keywords:
            if keyword in content:
                # Count occurrences but cap contribution
                count = content.count(keyword)
                score += min(count * 0.15, 0.4)
        
        # 2. Parliamentary context boost
        parliamentary_terms = [
            'president', 'minister', 'government', 'parliament', 'member',
            'budget', 'bill', 'motion', 'debate', 'committee', 'policy',
            'sri lanka', 'country', 'people', 'nation'
        ]
        
        for term in parliamentary_terms:
            if term in content:
                score += 0.1
        
        # 3. Question/answer patterns (common in parliamentary debates)
        if any(pattern in content for pattern in ['question:', 'answer:', 'hon.', 'minister of']):
            score += 0.2
        
        # 4. Phrase similarity for longer claims
        if len(claim) > 20:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, claim, content[:200]).ratio()
            score += similarity * 0.3
        
        return min(score, 1.0)

    def _extract_claim_keywords(self, claim: str) -> List[str]:
        """Extract meaningful keywords from claim for parliamentary search"""
        import re
        
        # Remove common question words and extract meaningful terms
        words = re.findall(r'\b\w+\b', claim.lower())
        
        stop_words = {
            'is', 'are', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'giving', 'reason', 'reasons', 'why'
        }
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Boost Sri Lankan context terms
        if 'sri' in keywords and 'lanka' in keywords:
            keywords = ['sri lanka'] + [w for w in keywords if w not in ['sri', 'lanka']]
        
        return keywords[:8]  # Top 8 keywords

