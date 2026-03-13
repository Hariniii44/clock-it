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

    # def _advanced_deduplication(self, evidence_list: List[Dict]) -> List[Dict]:
    #     """Advanced deduplication using content similarity"""
    #     from difflib import SequenceMatcher
        
    #     unique_evidence = []
    #     seen_content = []
        
    #     for evidence in evidence_list:
    #         content = f"{evidence.get('title', '')} {evidence.get('snippet', '')}"
            
    #         # Check URL deduplication
    #         url = evidence.get('link', '')
    #         if any(url == seen.get('link', '') for seen in unique_evidence):
    #             continue
            
    #         # Check content similarity
    #         is_duplicate = False
    #         for seen_content_text in seen_content:
    #             similarity = SequenceMatcher(None, content.lower(), seen_content_text.lower()).ratio()
    #             if similarity > 0.8:  # 80% similarity threshold
    #                 is_duplicate = True
    #                 break
            
    #         if not is_duplicate:
    #             unique_evidence.append(evidence)
    #             seen_content.append(content)
        
    #     print(f"Deduplication: {len(evidence_list)} -> {len(unique_evidence)} unique pieces")
    #     return unique_evidence

    def _advanced_deduplication(self, evidence_list: List[Dict]) -> List[Dict]:
        """Advanced deduplication using content similarity - FIXED VERSION"""
        from difflib import SequenceMatcher
        
        unique_evidence = []
        seen_content = []
        seen_urls = set()
        
        for evidence in evidence_list:
            # First check URL deduplication (exact matches only)
            url = evidence.get('link', evidence.get('url', ''))
            if url and url in seen_urls:
                continue
            
            # Content similarity check - LESS AGGRESSIVE
            content = f"{evidence.get('title', '')} {evidence.get('snippet', evidence.get('text', ''))}"
            
            # Skip very short content
            if len(content.strip()) < 50:
                continue
            
            # Check content similarity with LOWER threshold
            is_duplicate = False
            for seen_content_text in seen_content:
                similarity = SequenceMatcher(None, content.lower(), seen_content_text.lower()).ratio()
                
                # FIXED: More lenient similarity threshold
                # Government documents can have similar structure but different content
                if similarity > 0.95:  # Changed from 0.8 to 0.95 - only remove near-identical content
                    is_duplicate = True
                    break
            
            # Special handling for government sources - be even more lenient
            if evidence.get('dataset') or evidence.get('dataset_source'):
                # For government sources, only remove if >98% similar (almost identical)
                is_duplicate = False
                for seen_content_text in seen_content:
                    similarity = SequenceMatcher(None, content.lower(), seen_content_text.lower()).ratio()
                    if similarity > 0.98:  # Very strict for government sources
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_evidence.append(evidence)
                seen_content.append(content)
                if url:
                    seen_urls.add(url)
        
        print(f"Deduplication: {len(evidence_list)} -> {len(unique_evidence)} unique pieces")
        
        # ADDITIONAL: Show what was deduplicated for debugging
        if len(evidence_list) - len(unique_evidence) > 2:
            print(f"   Removed {len(evidence_list) - len(unique_evidence)} duplicates")
            # Show dataset breakdown of what remained
            remaining_datasets = {}
            for evidence in unique_evidence:
                dataset = evidence.get('dataset', evidence.get('dataset_source', 'web'))
                remaining_datasets[dataset] = remaining_datasets.get(dataset, 0) + 1
            
            if remaining_datasets:
                print(f"   Remaining by source: {dict(remaining_datasets)}")
        
        return unique_evidence
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc.lower()
        except:
            return url.lower()


    def _parse_serper_date(self, date_str: str) -> str:
        """
        Normalize Serper date strings to ISO format YYYY-MM-DD.
        Serper returns dates as relative ('3 days ago') or formatted ('Mar 11, 2026').
        Returns empty string if parsing fails.
        """
        if not date_str:
            return ''

        from datetime import datetime, timedelta
        import re

        date_str = date_str.strip()

        #Relative dates
        relative = re.match(
            r'(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago', date_str, re.IGNORECASE
        )
        if relative:
            n, unit = int(relative.group(1)), relative.group(2).lower()
            delta_map = {
                'minute': timedelta(minutes=n),
                'hour':   timedelta(hours=n),
                'day':    timedelta(days=n),
                'week':   timedelta(weeks=n),
                'month':  timedelta(days=n * 30),
                'year':   timedelta(days=n * 365),
            }
            result = datetime.now() - delta_map[unit]
            return result.strftime('%Y-%m-%d')

        # Named-month formats: "Mar 11, 2026" / "March 11, 2026"
        for fmt in ('%b %d, %Y', '%B %d, %Y', '%d %b %Y', '%d %B %Y'):
            try:
                return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue

        # ISO and slash formats already handled downstream but normalise anyway
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y'):
            try:
                return datetime.strptime(date_str[:10], fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue

        return ''  # Could not parse

    def _simple_serper_query(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Simplified Serper query - no social media filtering, just get results
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
                        raw_date = item.get("date", "")
                        result = {
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "link": item.get("link", ""),
                            "source": self._extract_domain(item.get("link", "")),
                            "date": self._parse_serper_date(raw_date),
                            "date_raw": raw_date,
                            "position": item.get("position", 0),
                            "search_type": "hybrid_serper"
                        }
                        results.append(result)
                    
                return results[:max_results]
                
            else:
                print(f" Search failed for query: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"  Search error: {e}")
            return []
        


    def retrieve_hybrid_serper_decomposition(self, claim: str, num_results: int = 20, 
                                   results_per_query: int = 10) -> List[Dict]:
        """
        Simplified hybrid search focused purely on relevance
        No complex social media filtering - getting the most relevant results
        """
        print(f" Target final results: {num_results}")
        
        all_results = []
        
        # Step 1: Generate search-optimized queries
        print(f" Generating search-optimized queries...")
        decomposer = ClaimDecomposer(self.groq_key)
        search_queries = decomposer.generate_search_queries(claim, num_queries=5)
        
        print(f" Generated {len(search_queries)} search-optimized queries:")
        for i, q in enumerate(search_queries, 1):
            print(f"      Query {i}: {q[:80]}...")
        
        # Step 2: Search using optimized queries
        print(f"  Searching {results_per_query} results per query...")
        
        for i, query in enumerate(search_queries):
            print(f"  Optimized query {i+1} search...")
            
            query_results = self._simple_serper_query(query, results_per_query)
            
            for result in query_results:
                result['query_source'] = f'optimized_query_{i+1}'
                result['query_text'] = query
            
            all_results.extend(query_results)
            print(f"    Retrieved {len(query_results)} results")
            
            import time
            time.sleep(0.5)
        
        print(f"    Retrieved {len(all_results)} total results from {len(search_queries)} queries")
        
        # Step 3: Deduplicate - no social media filtering
        unique_results = self._advanced_deduplication(all_results)
        print(f"    After deduplication: {len(unique_results)} unique results")
        
        # Step 4: Pure relevance filtering
        relevant_results = self._filter_by_relevance_only(claim, unique_results, top_k=num_results)
        
        print(f"    Final selection: {len(relevant_results)} highly relevant evidence pieces")
        return relevant_results
    
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
    
    def _is_verified_social_media(self, url: str, content: str) -> bool:
        """Check if social media source is from verified/official accounts"""
        domain = self._extract_domain(url)
        
        # Verified Sri Lankan government/official patterns in URLs or content
        verified_patterns = [
            # Government social media accounts
            'officialpresident', 'presidentsoffice', 'pmdnews', 'srilankagov',
            'srilankapolice', 'srilankanavy', 'srilankanairforce', 'srilankaarmy',
            # Official news organizations
            'adaderanalk', 'economynextcom', 'dailymirrornews', 'themorninglk',
            # Verified politicians (common patterns)
            'mpssrilanka', 'parliamentlk'
        ]
        
        # Check URL patterns
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in verified_patterns):
            return True
            
        # Check content for verification indicators
        content_lower = content.lower() if content else ''
        verification_indicators = [
            'official statement', 'press release', 'government announcement',
            'ministry of', 'secretary', 'director general', 'commissioner'
        ]
        
        return any(indicator in content_lower for indicator in verification_indicators)
    
    def _filter_social_media_selectively(self, results: List[Dict], temporal_context: dict = None) -> List[Dict]:
        """Filter social media but keep verified accounts for breaking news"""
        filtered_results = []
        removed_count = 0
        kept_verified = 0
        
        for result in results:
            url = result.get('link', '')
            content = f"{result.get('title', '')} {result.get('snippet', '')}"
            
            if self._is_social_media_source(url):
                # Check if it's a verified account
                if self._is_verified_social_media(url, content):
                    result['source_type'] = 'verified_social_media'
                    result['authority_weight'] = 0.3  # Lower than traditional media but not zero
                    filtered_results.append(result)
                    kept_verified += 1
                    print(f"   Kept verified social media: {self._extract_domain(url)}")
                else:
                    removed_count += 1
                    print(f"   Removed unverified social media: {self._extract_domain(url)}")
            else:
                filtered_results.append(result)
        
        if kept_verified > 0:
            print(f" Kept {kept_verified} verified social media sources")
        if removed_count > 0:
            print(f" Filtered out {removed_count} unverified social media sources")
        
        return filtered_results

    def _filter_by_relevance_only(self, claim: str, evidence_list: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Pure relevance-based filtering with cross-encoder - no source type filtering
        """
        print(f"   Cross-encoder relevance filtering (pure relevance approach)...")
        print(f"      Target: {top_k} most relevant sources")
        
        if not evidence_list:
            return []

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
            
            if evidence_text and len(evidence_text) > 10:
                pairs.append([claim, evidence_text])
                valid_evidence.append(evidence)
        
        if not pairs:
            print(f"      No valid text content found")
            return evidence_list[:top_k]
        
        print(f"Scoring {len(pairs)} evidence pieces...")
        
        # Get relevance scores using cross-encoder
        try:
            scores = self.relevance_ranker.predict(pairs)
            
            # Combine evidence with scores
            scored_evidence = list(zip(valid_evidence, scores))
            
            # Sort by relevance score (descending)
            scored_evidence.sort(key=lambda x: x[1], reverse=True)
            
            # Show top results with scores for debugging
            print(f" Relevance scores:")
            for i, (evidence, score) in enumerate(scored_evidence[:5], 1):
                title = evidence.get('title', 'No title')[:50]
                print(f"         {i}. {title}... (score: {score:.3f})")
            
            # Return top k most relevant
            top_evidence = [evidence for evidence, score in scored_evidence[:top_k]]
            
            print(f"Selected {len(top_evidence)} most relevant sources")
            return top_evidence
            
        except Exception as e:
            print(f"      Scoring failed: {e}")
            return evidence_list[:top_k]
            
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
        ENHANCED: Use comprehensive DatasetManager with PMD routing as primary source
        """
        print(f"Searching enhanced government databases (PMD, Cabinet, Parliament, Courts)...")
        
        try:
            # Import and initialize enhanced DatasetManager
            from src.database_retrieval.dataset_manager import DatasetManager
            db_manager = DatasetManager()
            
            # Load priority datasets based on claim type
            priority_datasets = ['pmd_press_releases', 'pmd_press_release', 'cabinet_decisions', 'hansard', 'supreme_court']
            
            loaded_datasets = 0
            for dataset_key in priority_datasets:
                try:
                    dataset = db_manager.get_dataset(dataset_key)
                    if dataset is None or len(dataset) == 0:
                        print(f"   Loading {dataset_key} dataset...")
                        dataset = db_manager.download_dataset(dataset_key, max_samples=200)
                        if len(dataset) > 0:
                            loaded_datasets += 1
                            print(f"   {dataset_key}: {len(dataset)} documents")
                    else:
                        loaded_datasets += 1
                        print(f"   {dataset_key}: {len(dataset)} documents (cached)")
                        
                except Exception as e:
                    print(f"   Failed to load {dataset_key}: {str(e)[:100]}...")
            
            print(f"Successfully loaded {loaded_datasets} datasets")
            
            if loaded_datasets == 0:
                print("No datasets available, falling back to web search")
                return []
            
            # Use enhanced hybrid search (semantic + keyword)
            print(f"Performing enhanced search with claim routing...")
            
            # Try hybrid search first (combines semantic + keyword search)
            try:
                database_results = db_manager.hybrid_search_datasets(
                    query=claim, 
                    max_results=num_results,
                    claim=claim  # This enables claim routing
                )
                print(f"Hybrid search found {len(database_results)} results")
                
            except Exception as e:
                print(f"Hybrid search failed ({e}), using keyword search...")
                # Fallback to enhanced keyword search with routing
                database_results = db_manager.search_datasets(
                    query=claim, 
                    max_results=num_results,
                    claim=claim
                )
                print(f"Keyword search found {len(database_results)} results")
            
            # Convert database results to consistent format
            formatted_results = []
            for result in database_results:
                formatted_result = {
                    'title': result['title'],
                    'snippet': result['snippet'],
                    'source': f"Sri Lanka Government Database - {result['source']}",
                    'link': result['link'],
                    'date': result.get('date', ''),
                    'authority_weight': result['authority_weight'],
                    'relevance_score': result.get('final_score', result.get('similarity_score', result.get('relevance_score', 0.5))),
                    'search_method': result.get('search_method', 'database'),
                    'dataset_source': result['dataset_source']
                }
                formatted_results.append(formatted_result)
            
            # Display top matches 
            print(f"Top database matches:")
            for i, result in enumerate(formatted_results[:3], 1):
                search_type = result.get('search_method', 'keyword')
                score = result['relevance_score']
                dataset = result['dataset_source'].replace('_', ' ').title()
                print(f"   {i}. {result['title'][:70]}...")
                print(f"      {dataset} | {search_type} | Score: {score:.3f}")
            
            return formatted_results
            
        except Exception as e:
            print(f"Enhanced database search failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def retrieve_evidence_dataset_first(self, claim: str, num_results: int = 20) -> List[Dict]:
        """
        NEW: Dataset-first evidence retrieval with Serper fallback
        This is the main method that prioritizes authoritative datasets over web search
        """
        print(f"DATASET-FIRST Evidence Retrieval for: '{claim[:60]}...'")
        print(f"Target results: {num_results}")
        
        all_results = []
        
        # STEP 1: Search Enhanced Government Datasets (PRIMARY)
        print(f"\nSTEP 1: Searching Government Databases...")
        database_results = self.retrieve_with_hansard_fallback(claim, num_results=num_results//2)
        
        if len(database_results) > 0:
            print(f"Found {len(database_results)} authoritative database results")
            all_results.extend(database_results)
            
            # If we have good database results, we may not need web search
            high_quality_results = [r for r in database_results if r['relevance_score'] > 0.4]
            
            if len(high_quality_results) >= num_results//2:
                print(f"Sufficient high-quality database results found ({len(high_quality_results)})")
                print(f"   Skipping web search to prioritize authoritative sources")
                return all_results[:num_results]
        else:
            print(f"No database results found")
        
        # STEP 2: Web Search Fallback (SECONDARY)
        remaining_results = num_results - len(all_results)
        if remaining_results > 0:
            print(f"\nSTEP 2: Web Search Fallback ({remaining_results} additional results needed)...")
            
            try:
                # Use decomposed search for better coverage
                web_results = self.retrieve_hybrid_serper_decomposition(
                    claim, 
                    num_results=remaining_results,
                    results_per_query=5
                )
                
                # Mark web results and lower their authority
                for result in web_results:
                    result['source'] = f"Web Search - {result.get('source', 'Unknown')}"
                    result['authority_weight'] = result.get('authority_weight', 0.5) * 0.8  # Reduce web authority
                    result['search_method'] = 'web_fallback'
                
                all_results.extend(web_results[:remaining_results])
                print(f"Added {len(web_results[:remaining_results])} web search results")
                
            except Exception as e:
                print(f"Web search fallback failed: {e}")
        
        # STEP 3: Final Processing
        print(f"\nSTEP 3: Final Processing...")
        
        # Deduplicate and re-rank
        all_results = self._advanced_deduplication(all_results)
        
        # Re-sort by authority weight and relevance (database results naturally rank higher)
        all_results.sort(key=lambda x: (
            x.get('authority_weight', 0.5) * x.get('relevance_score', 0.5)
        ), reverse=True)
        
        final_results = all_results[:num_results]
        
        # Summary
        database_count = sum(1 for r in final_results if r.get('search_method') != 'web_fallback')
        web_count = len(final_results) - database_count
        
        print(f"\nFINAL RESULTS SUMMARY:")
        print(f"   Database sources: {database_count}")
        print(f"   Web sources: {web_count}")
        print(f"   Total results: {len(final_results)}")
        
        return final_results

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

