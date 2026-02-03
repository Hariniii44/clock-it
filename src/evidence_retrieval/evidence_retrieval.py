import spacy
import requests
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
            print("   ✅ spaCy NER model loaded")
        except Exception as e:
            print(f"   ⚠️ Could not load spaCy model: {e}")
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

    # def _build_priority_queries(self, claim: str, entities: Dict) -> List[str]:
    #     """Build queries that prioritize official and quality sources"""
        
    #     priority_queries = []
    #     base_keywords = self._extract_key_terms(claim)
        
    #     # TIER 1: Target official sources first
    #     official_queries = [
    #         f'site:parliament.lk "{" ".join(base_keywords[:3])}"',
    #         f'site:president.gov.lk {" ".join(base_keywords[:2])}',
    #         f'site:cbsl.lk {" ".join(base_keywords[:2])}',
    #         f'site:pmoffice.gov.lk {" ".join(base_keywords[:2])}'
    #     ]
        
    #     # TIER 2: Target fact-checkers
    #     factcheck_queries = [
    #         f'site:factcheck.lk "{" ".join(base_keywords[:2])}"',
    #         f'fact check verify "{" ".join(base_keywords[:3])}" Sri Lanka'
    #     ]
        
    #     # TIER 3: Target quality Sri Lankan news
    #     news_queries = [
    #         f'site:dailymirror.lk OR site:economynext.com OR site:island.lk {" ".join(base_keywords[:2])}',
    #         f'site:newsfirst.lk OR site:adaderana.lk OR site:ft.lk {" ".join(base_keywords[:2])}'
    #     ]
        
    #     # TIER 4: General search without site restrictions
    #     general_queries = [
    #         f'"{claim}" Sri Lanka',
    #         f'{" ".join(base_keywords[:3])} Sri Lanka news'
    #     ]
        
    #     # Combine in priority order
    #     priority_queries.extend(official_queries[:2])      # 2 official queries
    #     priority_queries.extend(factcheck_queries[:1])     # 1 fact-check query  
    #     priority_queries.extend(news_queries[:2])          # 2 news queries
    #     priority_queries.extend(general_queries[:2])       # 2 general queries
        
    #     return priority_queries[:7]  # Return top 7 priority queries

    # def _extract_key_terms(self, claim: str) -> List[str]:
    #     """Extract key terms for query building"""
    #     # Remove common stop words and extract meaningful terms
    #     stop_words = {'said', 'that', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'has', 'have'}
    #     words = [word for word in claim.split() if word.lower() not in stop_words and len(word) > 2]
    #     return words[:5]  # Return top 5 key terms
    
    def extract_claim_entities(self, claim: str) -> Dict[str, List[str]]:
        """Extract key entities from the claim for targeted search"""
        if not self.nlp:
            return self._fallback_entity_extraction(claim)
        
        doc = self.nlp(claim)
        
        entities = {
            'persons': [ent.text for ent in doc.ents if ent.label_ == 'PERSON'],
            'organizations': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
            'locations': [ent.text for ent in doc.ents if ent.label_ == 'GPE'],
            'dates': [ent.text for ent in doc.ents if ent.label_ == 'DATE'],
            'money': [ent.text for ent in doc.ents if ent.label_ == 'MONEY'],
            'percentages': [ent.text for ent in doc.ents if ent.label_ == 'PERCENT'],
            'actions': [token.lemma_ for token in doc if token.pos_ == 'VERB' and not token.is_stop]
        }
        
        # Clean empty lists
        entities = {k: v for k, v in entities.items() if v}
        
        return entities
    
    def _fallback_entity_extraction(self, claim: str) -> Dict[str, List[str]]:
        """Simple regex-based extraction if spaCy fails"""
        entities = {}
        
        # Extract percentages
        percentages = re.findall(r'\d+(?:\.\d+)?%', claim)
        if percentages:
            entities['percentages'] = percentages
        
        # Extract money amounts
        money = re.findall(r'Rs\.?\s*[\d,]+(?:\.\d+)?(?:\s*(?:bn|million|trillion|lakh|crore))?', claim)
        if money:
            entities['money'] = money
        
        # Extract years
        dates = re.findall(r'\b(?:19|20)\d{2}\b', claim)
        if dates:
            entities['dates'] = dates
        
        return entities
    
    def retrieve_diverse_evidence(
        self, 
        claim: str, 
        num_results: int = 10,
        include_fact_checkers: bool = True,
        use_ai_filtering: bool = True
    ) -> List[Dict]:
        """Retrieve evidence with source diversity"""
        
        # Extract entities
        entities = self.extract_claim_entities(claim)
        print(f"   Extracted entities: {entities}")
        
        # Build targeted queries
        queries = self.build_targeted_queries(claim, entities)
        print(f"   🔍 Built {len(queries)} targeted queries")
        
        all_results = []
        seen_urls = set()
        
        # Execute multiple queries
        for i, query in enumerate(queries):
            print(f"      Query {i+1}: {query[:50]}...")
            
            results = self._search_single_query(
                query, 
                max_results=min(15, num_results * 3)
            )
            
            # Add to results with deduplication
            for result in results:
                if result['link'] not in seen_urls:
                    result['query_used'] = query
                    result['source_alignment'] = self._classify_source_alignment(result['link'])
                    result['relevance_score'] = self._calculate_relevance(claim, result, entities)
                    all_results.append(result)
                    seen_urls.add(result['link'])

        print(f"   📋 Found {len(all_results)} initial results")

        print(f"   DEBUG: self.query_builder exists={self.query_builder is not None}")
        if self.query_builder:
            print(f"   DEBUG: has filter method={hasattr(self.query_builder, 'filter_relevant_evidence')}")
        
        # Sort by relevance and diversity
        diverse_results = self._ensure_source_diversity(all_results, num_results)
        
        return diverse_results[:num_results]

    def retrieve_diverse_evidence_livefc(
        self, 
        claim: str, 
        num_results: int = 10,
        use_decomposition: bool = True
    ) -> List[Dict]:
        """LiveFC-style retrieval with priority source targeting"""
        
        if use_decomposition:
            from .claim_decomposer import ClaimDecomposer
            from .multi_source_retriever import MultiSourceRetriever
            from .evidence_ranker import EvidenceRanker
            
            decomposer = ClaimDecomposer(self.groq_key)
            questions = decomposer.decompose_claim(claim, num_questions=4)  # Reduce to 4 since we're adding original
            
            print(f"   Decomposed into {len(questions)} verification questions")
            for i, q in enumerate(questions, 1):
                print(f"      Q{i}: {q}")
            
            # Step 2: Priority-aware multi-source retrieval WITH original claim
            multi_retriever = MultiSourceRetriever(self.serper_key)
            all_evidence = multi_retriever.retrieve_from_multiple_sources_priority(
                questions, 
                max_per_source=6,
                original_claim=claim  
            )
            
            print(f"   📋 Retrieved {len(all_evidence)} total pieces from priority sources")
            
        else:
            return self.retrieve_diverse_evidence(claim, num_results)
        
        # Step 3: Advanced deduplication
        unique_evidence = self._advanced_deduplication(all_evidence)
        
        # Step 4: Priority-aware cross-encoder ranking
        ranker = EvidenceRanker()
        ranked_evidence = ranker.rank_evidence_with_priority(claim, unique_evidence, top_k=num_results * 2)
        
        # Step 5: Priority-aware source selection
        priority_results = self._ensure_priority_source_diversity(ranked_evidence, num_results)
        
        return priority_results[:num_results]
    
    def _ensure_priority_source_diversity(self, results: List[Dict], target_count: int) -> List[Dict]:
        """Ensure priority sources are included first"""
        
        selected = []
        
        # STEP 1: Guarantee at least 1 from each priority tier (if available)
        tier_counts = {'official_primary': 0, 'fact_checkers': 0, 'sri_lankan_news': 0}
        
        # First pass: Take top result from each priority tier
        for tier in ['official_primary', 'fact_checkers', 'sri_lankan_news']:
            tier_results = [r for r in results if r.get('priority_tier') == tier]
            if tier_results and len(selected) < target_count:
                selected.append(tier_results[0])  # Take best from this tier
                tier_counts[tier] += 1
        
        # STEP 2: Fill remaining slots with best overall scores
        remaining_slots = target_count - len(selected)
        for result in results:
            if result not in selected and len(selected) < target_count:
                tier = result.get('priority_tier', 'unknown')
                
                # Limit per tier to maintain diversity
                if tier in tier_counts and tier_counts[tier] < 2:  # Max 2 per priority tier
                    selected.append(result)
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
                elif tier not in tier_counts:  # Unknown/other sources
                    selected.append(result)
        
        # Print source breakdown
        tier_breakdown = {}
        for result in selected:
            tier = result.get('priority_tier', 'unknown')
            tier_breakdown[tier] = tier_breakdown.get(tier, 0) + 1
        
        print(f"   📊 Priority source distribution: {tier_breakdown}")
        
        return selected
    
    
    def _classify_source_alignment(self, url: str) -> str:
        """Classify source political alignment"""
        domain = self._extract_domain(url)
        
        for alignment, domains in self.source_categories.items():
            if any(d in domain for d in domains):
                return alignment
        
        return 'unknown'
    
    def _calculate_relevance(self, claim: str, result: Dict, entities: Dict) -> float:
        """Calculate relevance score for ranking"""
        score = 0.0
        
        text = f"{result['title']} {result['snippet']}".lower()
        claim_lower = claim.lower()
        
        # Exact phrase match
        if claim_lower in text:
            score += 1.0
        
        # Entity matches
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if entity.lower() in text:
                    score += 0.5
        
        # Keyword overlap
        claim_words = set(claim_lower.split())
        text_words = set(text.split())
        overlap = len(claim_words.intersection(text_words)) / len(claim_words)
        score += overlap
        
        # Boost for known quality sources
        domain = self._extract_domain(result['link'])
        if any(quality_domain in domain for quality_domain in [
            'worldbank.org', 'imf.org', 'reuters.com', 'bbc.com', 
            'centralbank.lk', 'statistics.gov.lk'
        ]):
            score += 0.5
        
        return min(score, 2.0)  # Cap at 2.0
    
    def _ensure_source_diversity(self, results: List[Dict], target_count: int) -> List[Dict]:
        """Enhanced diversity with AI relevance priority"""
        
        # Create priority scoring system
        for result in results:
            priority_score = result.get('relevance_score', 0)
            
            # Boost for AI-determined high relevance
            if result.get('ai_relevance') == 'HIGHLY_RELEVANT':
                priority_score += 2.0
            elif result.get('ai_relevance') == 'MODERATELY_RELEVANT':
                priority_score += 1.0
            
            result['priority_score'] = priority_score

            # Ensure source_alignment exists
            if 'source_alignment' not in result:
                result['source_alignment'] = self._classify_source_alignment(result.get('link', ''))

        # Sort by priority (AI relevance + traditional relevance)
        results.sort(key=lambda x: x['priority_score'], reverse=True)
        
        selected = []
        alignment_counts = {alignment: 0 for alignment in self.source_categories.keys()}
        alignment_counts['unknown'] = 0
        
        # First pass: Take highly relevant evidence regardless of diversity
        for result in results:
            if result.get('ai_relevance') == 'HIGHLY_RELEVANT' and len(selected) < target_count:
                selected.append(result)
                alignment_counts[result['source_alignment']] += 1
        
        # Second pass: Fill remaining slots with diversity considerations
        for result in results:
            if result not in selected and len(selected) < target_count:
                alignment = result['source_alignment']
                if alignment_counts[alignment] < 2:  # Max 2 per alignment
                    selected.append(result)
                    alignment_counts[alignment] += 1
        
        return selected

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
        
        print(f"         🔄 Deduplication: {len(evidence_list)} → {len(unique_evidence)} unique pieces")
        return unique_evidence
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc.lower()
        except:
            return url.lower()
        

    def retrieve_simple_serper(self, claim: str, num_results: int = 10, get_all: bool = False) -> List[Dict]:
        """
        Simple direct Serper search
        Args:
            claim: The claim to search
            num_results: Final number of results to return
            get_all: If True, retrieve many results and rank them
        """
        if get_all:
            # Comprehensive search - get many results
            max_results = 50  # Serper max is usually 100
            print(f"   🔍 Comprehensive Serper search (up to {max_results} results)...")
        else:
            max_results = num_results * 2
            print(f"   🔍 Simple Serper search: {claim[:50]}...")
        
        import requests
        import time
        
        # Rate limiting
        time.sleep(1)
        
        url = "https://google.serper.dev/search"
        payload = {
            "q": claim,
            "num": max_results,  # Request many more results
            "gl": "lk",
            "hl": "en"
        }
        headers = {
            "X-API-KEY": self.serper_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            print(f"      📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"      📊 Raw results keys: {list(data.keys())}")
                
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
                            "search_type": "simple_serper"  # Add search method tracking
                        }
                        results.append(result)
                
                print(f"      ✅ Retrieved {len(results)} total results")
                
                if get_all:
                    # Apply ranking and then select top results
                    print(f"      🏆 Ranking all {len(results)} results...")
                    ranked_results = self._rank_and_select_evidence(claim, results, num_results)
                    print(f"      📊 Selected top {len(ranked_results)} after ranking")
                    return ranked_results
                else:
                    return results[:num_results]
                    
            else:
                print(f"      ❌ Search failed: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"      ❌ Search error: {e}")
            return []
        
    def _rank_and_select_evidence(self, claim: str, all_results: List[Dict], num_final: int) -> List[Dict]:
        """
        Rank all retrieved results and select the best ones
        """

        print(f"      🎯 _rank_and_select_evidence: Targeting {num_final} final results")  # DEBUG

        if not all_results:
            return []
        
        # Import ranker if not already available
        from .evidence_ranker import EvidenceRanker
        if not hasattr(self, 'ranker'):
            self.ranker = EvidenceRanker()
        
        ranking_target = min(len(all_results), num_final * 3)  # Get 3x target for diversity
        print(f"      📊 Cross-encoder ranking top {ranking_target} from {len(all_results)} total")

        # Apply cross-encoder ranking
        ranked_evidence = self.ranker.rank_evidence_with_priority(
            claim, 
            all_results,
            top_k=ranking_target
            # prioritize_sri_lankan=True
        )
        
        print(f"      📈 Cross-encoder returned {len(ranked_evidence)} ranked results")

        # Apply diversity selection
        diverse_evidence = self._ensure_source_diversity(ranked_evidence, num_final)
        
        print(f"      🎯 Diversity selection returned {len(diverse_evidence)} final results")
        return diverse_evidence
    
    # def retrieve_hybrid_serper_decomposition(self, claim: str, num_results: int = 10) -> List[Dict]:
    #     """
    #     Hybrid approach: Claim decomposition + Simple Serper search
    #     """
    #     print(f"    Hybrid: Decomposition + Serper search...")
        
    #     all_results = []
        
    #     # Step 1: Decompose claim into questions
    #     print(f"    Decomposing claim...")
    #     decomposer = ClaimDecomposer(self.groq_key)
    #     questions = decomposer.decompose_claim(claim, num_questions=4)
        
    #     print(f"    Generated {len(questions)} verification questions:")
    #     for i, q in enumerate(questions, 1):
    #         print(f"      Q{i}: {q[:80]}...")
        
    #     # Step 2: Search for original claim + each question using simple Serper
    #     search_queries = [claim] + questions

    #     print(f"   🔍 Searching {results_per_query} results per query...")

    #     results_per_query = max(2, num_results // len(search_queries))  # Distribute results
        
    #     for i, query in enumerate(search_queries):
    #         if i == 0:
    #             print(f"    Original claim search...")
    #         else:
    #             print(f"    Question {i} search...")
            
    #         # Use simple Serper search for each query
    #         query_results = self._simple_serper_query(query, results_per_query)
            
    #         # Tag results with query source
    #         for result in query_results:
    #             result['query_source'] = 'original_claim' if i == 0 else f'question_{i}'
    #             result['query_text'] = query
            
    #         all_results.extend(query_results)
            
    #         # Rate limiting between queries
    #         import time
    #         time.sleep(0.5)
        
    #     print(f"    Retrieved {len(all_results)} total results from {len(search_queries)} queries")
        
    #     # Step 3: Deduplicate
    #     unique_results = self._advanced_deduplication(all_results)
    #     print(f"    After deduplication: {len(unique_results)} unique results")
        
    #     # Step 4: Rank and select top results
    #     if len(unique_results) > num_results:
    #         print(f"    Ranking and selecting top {num_results} results...")
    #         final_results = self._rank_and_select_evidence(claim, unique_results, num_results)
    #     else:
    #         final_results = unique_results
        
    #     print(f"   ✅ Final selection: {len(final_results)} evidence pieces")
    #     return final_results

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
                print(f"      ⚠️ Search failed for query: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"      ❌ Search error: {e}")
            return []
        
    # def retrieve_hybrid_serper_decomposition(self, claim: str, num_results: int = 10, 
    #                                    results_per_query: int = 10) -> List[Dict]:
    #     """
    #     Hybrid approach: Claim decomposition + Simple Serper search
    #     Args:
    #         claim: The claim to fact-check
    #         num_results: FINAL number of results to return after ranking
    #         results_per_query: Number of results to get PER QUERY (NEW!)
    #     """
    #     print(f"   🔍 Hybrid: Decomposition + Serper search...")
    #     print(f"   🎯 Target final results: {num_results}")  # DEBUG

        
    #     all_results = []
        
    #     # Step 1: Decompose claim into questions
    #     print(f"   🧠 Decomposing claim...")
    #     decomposer = ClaimDecomposer(self.groq_key)
    #     questions = decomposer.decompose_claim(claim, num_questions=4)
        
    #     print(f"   📋 Generated {len(questions)} verification questions:")
    #     for i, q in enumerate(questions, 1):
    #         print(f"      Q{i}: {q[:80]}...")
        
    #     # Step 2: Search for original claim + each question
    #     search_queries = [claim] + questions
        
    #     print(f"   🔍 Searching {results_per_query} results per query...")
        
    #     for i, query in enumerate(search_queries):
    #         if i == 0:
    #             print(f"   🔍 Original claim search...")
    #         else:
    #             print(f"   🔍 Question {i} search...")
            
    #         # Get FULL results_per_query for EACH query
    #         query_results = self._simple_serper_query(query, results_per_query)
            
    #         # Tag results with query source
    #         for result in query_results:
    #             result['query_source'] = 'original_claim' if i == 0 else f'question_{i}'
    #             result['query_text'] = query
            
    #         all_results.extend(query_results)
    #         print(f"      📊 Retrieved {len(query_results)} results")
            
    #         # Rate limiting between queries
    #         import time
    #         time.sleep(0.5)
        
    #     print(f"   📊 Retrieved {len(all_results)} total results from {len(search_queries)} queries")
        
    #     # Step 3: Deduplicate
    #     unique_results = self._advanced_deduplication(all_results)
    #     print(f"   🔄 After deduplication: {len(unique_results)} unique results")
        
    #     # Step 4: Rank and select top results (only if we have more than needed)
    #     if len(unique_results) > num_results:
    #         print(f"   🏆 Ranking and selecting top {num_results} results...")
    #         final_results = self._rank_and_select_evidence(claim, unique_results, num_results)
    #         print(f"   📊 After ranking: {len(final_results)} results") 
    #     else:
    #         print(f"   ✅ Keeping all {len(unique_results)} unique results")
    #         final_results = unique_results


    #     if len(final_results) < num_results and len(unique_results) >= num_results:
    #         print(f"    Ranking returned {len(final_results)} but we have {len(unique_results)} available")
    #         print(f"    Taking top {num_results} from unique results directly")
    #         final_results = unique_results[:num_results]
        
    #     print(f"   ✅ Final selection: {len(final_results)} evidence pieces")
    #     return final_results

    def retrieve_hybrid_serper_decomposition(self, claim: str, num_results: int = 20, 
                                   results_per_query: int = 10) -> List[Dict]:
        """
        Enhanced hybrid search with cross-encoder relevance filtering
        """
        print(f"   🔍 Hybrid: Decomposition + Serper + Relevance Filtering...")
        print(f"   🎯 Target final results: {num_results}")
        
        all_results = []
        
        # Step 1: Decompose claim into questions
        print(f"   🧠 Decomposing claim...")
        decomposer = ClaimDecomposer(self.groq_key)
        questions = decomposer.decompose_claim(claim, num_questions=4)
        
        print(f"   📋 Generated {len(questions)} verification questions:")
        for i, q in enumerate(questions, 1):
            print(f"      Q{i}: {q[:80]}...")
        
        # Step 2: Search for original claim + each question
        search_queries = [claim] + questions
        
        print(f"   🔍 Searching {results_per_query} results per query...")
        
        for i, query in enumerate(search_queries):
            if i == 0:
                print(f"   🔍 Original claim search...")
            else:
                print(f"   🔍 Question {i} search...")
            
            query_results = self._simple_serper_query(query, results_per_query)
            
            for result in query_results:
                result['query_source'] = 'original_claim' if i == 0 else f'question_{i}'
                result['query_text'] = query
            
            all_results.extend(query_results)
            print(f"      📊 Retrieved {len(query_results)} results")
            
            import time
            time.sleep(0.5)
        
        print(f"   📊 Retrieved {len(all_results)} total results from {len(search_queries)} queries")
        
        # Step 3: FILTER SOCIAL MEDIA FIRST
        print(f"         🔄 Pre-filtering social media sources...")
        all_results = self._filter_out_social_media(all_results)
        print(f"         📊 After social media filtering: {len(all_results)} sources")

        # Step 3: Deduplicate
        unique_results = self._advanced_deduplication(all_results)
        print(f"   🔄 After deduplication: {len(unique_results)} unique results")
        
        # NEW: Step 4: Cross-encoder relevance filtering
        relevant_results = self._filter_by_cross_encoder_relevance(
            claim, 
            unique_results, 
            top_k=num_results,
            relevance_threshold=0.15  # Lower threshold to ensure we get results
        )
        
        # Step 5: Final ranking (optional, since cross-encoder already ranked)
        if len(relevant_results) > num_results:
            print(f"   🏆 Final ranking selection...")
            final_results = relevant_results[:num_results]
        else:
            final_results = relevant_results
        
        print(f"   ✅ Final selection: {len(final_results)} highly relevant evidence pieces")
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
                print(f"         🚫 Removed social media: {self._extract_domain(url)}")
            else:
                filtered_results.append(result)
        
        if removed_count > 0:
            print(f"      🧹 Filtered out {removed_count} social media sources")
        
        return filtered_results

    def _filter_by_cross_encoder_relevance(self, claim: str, evidence_list: List[Dict], 
                                     top_k: int = 10, relevance_threshold: float = 0.3) -> List[Dict]:
        """
        Use cross-encoder to score relevance and filter evidence before verification
        """
        print(f"   🎯 Cross-encoder relevance filtering...")
        print(f"      Threshold: {relevance_threshold}, Target: {top_k} sources")
        
        if not evidence_list:
            return []

        # FIRST: Remove social media sources completely
        pre_filtered = self._filter_out_social_media(evidence_list)
        if len(pre_filtered) < len(evidence_list):
            print(f"      🧹 Pre-filtered: {len(evidence_list)} → {len(pre_filtered)} (removed social media)")

        # Import cross-encoder if not available
        try:
            from sentence_transformers import CrossEncoder
            if not hasattr(self, 'relevance_ranker'):
                print(f"      📊 Loading cross-encoder for relevance...")
                self.relevance_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
                print(f"      ✅ Cross-encoder loaded")
        except Exception as e:
            print(f"      ❌ Cross-encoder loading failed: {e}")
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
            print(f"      ⚠️ No valid evidence text found")
            return []
        
        print(f"      📊 Scoring {len(pairs)} evidence pieces...")
        
        # Get relevance scores
        try:
            scores = self.relevance_ranker.predict(pairs)
            
            # Add scores to evidence and sort by relevance
            scored_evidence = []
            for evidence, score in zip(valid_evidence, scores):
                base_score = float(score)  # Get base cross-encoder score
                
                # ADD QUALITY SOURCE BONUS HERE
                source = evidence.get('source', '').lower()
                link = evidence.get('link', '').lower()
                
                # Academic/Research sources
                if any(academic in source for academic in [
                    'researchgate.net', 'academia.edu', 'scholar.google', 
                    'jstor.org', 'pubmed.ncbi', 'arxiv.org'
                ]):
                    base_score += 0.2
                    print(f"         📚 Academic bonus: {source}")
                
                # Government/Official sources (.gov.lk domains)
                elif any(official in source for official in [
                    'gov.lk', 'parliament.lk', 'president.gov.lk', 'cbsl.lk',
                    'statistics.gov.lk', 'treasury.gov.lk', 'pmoffice.gov.lk'
                ]):
                    base_score += 0.3
                    print(f"         🏛️ Official source bonus: {source}")

                # International Organizations
                elif any(intl in source for intl in [
                    'worldbank.org', 'imf.org', 'fao.org', 'un.org',
                    'oecd.org', 'adb.org'
                ]):
                    base_score += 0.25
                    print(f"         🌍 International org bonus: {source}")
                
                # Quality News Sources (Sri Lankan)
                elif any(quality in source for quality in [
                    'economynext.com', 'ft.lk', 'themorning.lk'
                ]):
                    base_score += 0.1
                    print(f"         📰 Quality news bonus: {source}")
                
                # Penalty for low-quality sources
                elif any(low_qual in source for low_qual in [
                    'facebook.com', 'twitter.com', 'instagram.com',
                    'tiktok.com', 'youtube.com'
                ]):
                    base_score -= 0.15
                    print(f"         📱 Social media penalty: {source}")
                
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
                print(f"      ⚠️ Only {len(filtered_evidence)} above threshold, taking top {top_k}")
                filtered_evidence = scored_evidence[:top_k]
            
            # Display filtering results
            print(f"      📈 Relevance scores:")
            for i, evidence in enumerate(filtered_evidence[:5], 1):
                score = evidence['relevance_score']
                title = evidence.get('title', 'No title')[:50]
                print(f"         {i}. {title}... (score: {score:.3f})")
            
            print(f"      ✅ Selected {len(filtered_evidence)} most relevant sources")
            return filtered_evidence
            
        except Exception as e:
            print(f"      ❌ Relevance scoring failed: {e}")
            return evidence_list[:top_k]  # Fallback

