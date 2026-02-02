import requests
import time
from typing import List, Dict
from urllib.parse import quote_plus

class MultiSourceRetriever:
    def __init__(self, serper_key: str, bing_key: str = None):
        self.serper_key = serper_key
        self.bing_key = bing_key
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        # Search endpoints
        self.serper_url = "https://google.serper.dev/search"
        self.bing_url = "https://api.bing.microsoft.com/v7.0/search"
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            print(f"         ⏱️ Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def retrieve_from_multiple_sources_priority(self, questions: List[str], max_per_source: int = 10, original_claim: str = None) -> List[Dict]:
        """Enhanced retrieval with priority source targeting"""
        
        all_evidence = []
        
        # PRIORITY 0: Search for the original claim first if provided
        if original_claim:
            print(f"      🎯 Original Claim: {original_claim[:60]}...")
            
            # Search official sources for original claim
            official_results = self._search_official_sources(original_claim, 4)
            all_evidence.extend(official_results)
            
            # Search fact-checkers for original claim  
            factcheck_results = self._search_factcheck_sources(original_claim, 3)
            all_evidence.extend(factcheck_results)
            
            # Search Sri Lankan news for original claim
            news_results = self._search_sri_lankan_news(original_claim, 6)
            all_evidence.extend(news_results)
            
            # General search for original claim
            general_results = self._search_google(original_claim, 5)
            for result in general_results:
                result['source_engine'] = 'google_original'
                result['question_id'] = 0  # Mark as original claim
                result['question'] = original_claim
                result['priority_tier'] = self._get_source_priority_tier(result.get('link', ''))
            all_evidence.extend(general_results)
        
        # Continue with decomposed questions
        for i, question in enumerate(questions, 1):
            print(f"      📋 Question {i}: {question[:60]}...")
            
            # PRIORITY 1: Official sources first
            official_results = self._search_official_sources(question, 2)
            all_evidence.extend(official_results)
            
            # PRIORITY 2: Fact-checkers
            factcheck_results = self._search_factcheck_sources(question, 1)
            all_evidence.extend(factcheck_results)
            
            # PRIORITY 3: Quality Sri Lankan news
            news_results = self._search_sri_lankan_news(question, max_per_source // 3)
            all_evidence.extend(news_results)
            
            # PRIORITY 4: General search (Google/Bing) 
            google_results = self._search_google(question, max_per_source // 4)
            for result in google_results:
                result['source_engine'] = 'google'
                result['question_id'] = i
                result['question'] = question
                result['priority_tier'] = self._get_source_priority_tier(result.get('link', ''))
            all_evidence.extend(google_results)
        
        print(f"         📊 Retrieved {len(all_evidence)} pieces from priority sources (including original claim)")
        return all_evidence
    
    def _search_official_sources(self, query: str, max_results: int) -> List[Dict]:
        """Search official government sources"""
        official_queries = [
            f'site:parliament.lk {query}',
            f'site:president.gov.lk {query}', 
            f'site:cbsl.lk {query}',
            f'site:pmoffice.gov.lk {query}'
        ]
        
        results = []
        for official_query in official_queries:
            print(f"            🔍 Searching: {official_query}")
            try:
                search_results = self._search_google(official_query, max_results // len(official_queries))
                print(f"            📊 Found {len(search_results)} results")
                for result in search_results:
                    result['source_engine'] = 'official'
                    result['priority_tier'] = 'official_primary'
                results.extend(search_results)
            except Exception as e:
                print(f"            ⚠️ Official search failed for {official_query}: {e}")
        
        print(f"            📋 Total official results: {len(results)}")
        return results

    def _search_factcheck_sources(self, query: str, max_results: int) -> List[Dict]:
        """Search fact-checking sources"""
        factcheck_queries = [
            f'site:factcheck.lk {query}',
            f'fact check verify "{query}" Sri Lanka'
        ]
        
        results = []
        for fc_query in factcheck_queries:
            try:
                search_results = self._search_google(fc_query, max_results // len(factcheck_queries))
                for result in search_results:
                    result['source_engine'] = 'factcheck'
                    result['priority_tier'] = 'fact_checkers'
                results.extend(search_results)
            except Exception as e:
                print(f"            ⚠️ Fact-check search failed: {e}")
        
        return results

    def _search_sri_lankan_news(self, query: str, max_results: int) -> List[Dict]:
        """Search quality Sri Lankan news sources"""
        news_query = f'site:dailymirror.lk OR site:economynext.com OR site:island.lk OR site:newsfirst.lk OR site:adaderana.lk OR site:ft.lk {query}'
        
        print(f"            🔍 News search: {news_query}")
        try:
            results = self._search_google(news_query, max_results)
            print(f"            📊 News results: {len(results)}")
            for result in results:
                result['source_engine'] = 'sri_lankan_news'
                result['priority_tier'] = 'sri_lankan_news'
            return results
        except Exception as e:
            print(f"            ⚠️ Sri Lankan news search failed: {e}")
            return []

    def _search_google(self, query: str, max_results: int) -> List[Dict]:
        """Search Google via Serper.dev"""
        self._rate_limit()
        
        headers = {
            'X-API-KEY': self.serper_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': query,
            'gl': 'lk',
            'hl': 'en',
            'num': max_results,
            'type': 'search'
        }
        
        print(f"            🔍 Serper search: {query[:50]}...")
        
        try:
            response = requests.post(self.serper_url, headers=headers, json=payload)
            print(f"            📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                results = response.json()
                print(f"            📊 Raw results keys: {list(results.keys())}")
                
                evidence = []
                
                # Serper returns results in 'organic' key
                for result in results.get("organic", []):
                    link = result.get("link", "")
                    evidence.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "source": self._extract_domain(link),
                        "link": link,
                        "date": result.get("date", ""),
                        "source_alignment": self._classify_source_alignment(link),
                        "relevance_score": 0.0
                    })
                
                # Also check for news results if available
                for result in results.get("news", []):
                    link = result.get("link", "")
                    evidence.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "source": result.get("source", ""),
                        "link": link,
                        "date": result.get("date", ""),
                        "source_alignment": self._classify_source_alignment(link),
                        "relevance_score": 0.0
                    })
                
                print(f"            ✅ Extracted {len(evidence)} pieces")
                return evidence
                
            elif response.status_code == 429:
                print(f"            ⚠️ Rate limited. Waiting 2 seconds...")
                time.sleep(2)
                return []
            else:
                print(f"            ⚠️ Serper error {response.status_code}: {response.text[:100]}")
                return []
                
        except Exception as e:
            print(f"            ⚠️ Serper search failed: {e}")
            return []

    def _get_source_priority_tier(self, url: str) -> str:
        """Determine priority tier for a source"""
        domain = self._extract_domain(url)
        
        tier_mapping = {
            'official_primary': ['parliament.lk', 'president.gov.lk', 'cbsl.lk', 'pmoffice.gov.lk'],
            'fact_checkers': ['factcheck.lk', 'factcrescendo.com', 'boomlive.in'],
            'sri_lankan_news': ['dailymirror.lk', 'economynext.com', 'island.lk', 'newsfirst.lk', 'adaderana.lk', 'ft.lk'],
            'government': ['news.lk', 'dailynews.lk'],
            'international': ['reuters.com', 'bbc.com', 'worldbank.org']
        }
        
        for tier, domains in tier_mapping.items():
            if any(d in domain for d in domains):
                return tier
        
        return 'unknown'

    def _classify_source_alignment(self, url: str) -> str:
        """Classify source political alignment"""
        source_categories = {
            'government': ['dailynews.lk', 'news.lk', 'parliament.lk'],
            'opposition': ['themorning.lk', 'island.lk', 'economynext.com'],
            'neutral': ['adaderana.lk', 'newsfirst.lk', 'ft.lk', 'dailymirror.lk'],
            'international': ['worldbank.org', 'imf.org', 'reuters.com', 'bbc.com'],
            'fact_checkers': ['factcheck.org', 'politifact.com', 'snopes.com']
        }
        
        domain = self._extract_domain(url)
        
        for alignment, domains in source_categories.items():
            if any(d in domain for d in domains):
                return alignment
        
        return 'unknown'

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lower()
        except:
            return url.lower()