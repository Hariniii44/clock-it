import spacy
import requests
from typing import List, Dict, Set
import re
from datetime import datetime
from urllib.parse import urlparse
# from .query_builder import GeminiQueryBuilder
from .groq_query_builder import GroqQueryBuilder

class AdvancedEvidenceRetriever:
    def __init__(self, serper_key: str, gemini_key: str = None, groq_key: str = None):
        # self.serpapi_key = serpapi_key
        self.serper_key = serper_key
        self.base_url = "https://serpapi.com/search"

        # Initialize query builders in order of preference
        self.query_builder = None

        # Try Groq first (faster, more reliable)
        if groq_key:
            try:
                self.query_builder = GroqQueryBuilder(groq_key)
                self.query_provider = "Groq"
                print("   ✅ Groq query builder initialized")
            except Exception as e:
                print(f"   ⚠️ Groq initialization failed: {e}")
        
        # Fallback to Gemini
        # if not self.query_builder and gemini_key:
        #     try:
        #         self.query_builder = GeminiQueryBuilder(gemini_key)
        #         self.query_provider = "Gemini"
        #         print("   ✅ Gemini query builder initialized")
        #     except Exception as e:
        #         print(f"   ⚠️ Gemini initialization failed: {e}")
        
        if not self.query_builder:
            self.query_provider = "Rule-based"
            print("   ✅ Using rule-based query generation")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("   ✅ spaCy NER model loaded")
        except Exception as e:
            print(f"   ⚠️ Could not load spaCy model: {e}")
            self.nlp = None
        
        # Source diversity categories
        # self.source_categories = {
        #     'government': ['dailynews.lk', 'news.lk', 'parliament.lk'],
        #     'opposition': ['themorning.lk', 'island.lk', 'economynext.com'],
        #     'neutral': ['adaderana.lk', 'newsfirst.lk', 'ft.lk', 'dailymirror.lk'],
        #     'international': ['worldbank.org', 'imf.org', 'reuters.com', 'bbc.com'],
        #     'fact_checkers': ['factcheck.org', 'politifact.com', 'snopes.com']
        # }

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

    def _build_priority_queries(self, claim: str, entities: Dict) -> List[str]:
        """Build queries that prioritize official and quality sources"""
        
        priority_queries = []
        base_keywords = self._extract_key_terms(claim)
        
        # TIER 1: Target official sources first
        official_queries = [
            f'site:parliament.lk "{" ".join(base_keywords[:3])}"',
            f'site:president.gov.lk {" ".join(base_keywords[:2])}',
            f'site:cbsl.lk {" ".join(base_keywords[:2])}',
            f'site:pmoffice.gov.lk {" ".join(base_keywords[:2])}'
        ]
        
        # TIER 2: Target fact-checkers
        factcheck_queries = [
            f'site:factcheck.lk "{" ".join(base_keywords[:2])}"',
            f'fact check verify "{" ".join(base_keywords[:3])}" Sri Lanka'
        ]
        
        # TIER 3: Target quality Sri Lankan news
        news_queries = [
            f'site:dailymirror.lk OR site:economynext.com OR site:island.lk {" ".join(base_keywords[:2])}',
            f'site:newsfirst.lk OR site:adaderana.lk OR site:ft.lk {" ".join(base_keywords[:2])}'
        ]
        
        # TIER 4: General search without site restrictions
        general_queries = [
            f'"{claim}" Sri Lanka',
            f'{" ".join(base_keywords[:3])} Sri Lanka news'
        ]
        
        # Combine in priority order
        priority_queries.extend(official_queries[:2])      # 2 official queries
        priority_queries.extend(factcheck_queries[:1])     # 1 fact-check query  
        priority_queries.extend(news_queries[:2])          # 2 news queries
        priority_queries.extend(general_queries[:2])       # 2 general queries
        
        return priority_queries[:7]  # Return top 7 priority queries

    def _extract_key_terms(self, claim: str) -> List[str]:
        """Extract key terms for query building"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'said', 'that', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'has', 'have'}
        words = [word for word in claim.split() if word.lower() not in stop_words and len(word) > 2]
        return words[:5]  # Return top 5 key terms
    
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
    
    def build_targeted_queries(self, claim: str, entities: Dict) -> List[str]:
        if self.query_builder:
            print(f"   🤖 Generating queries with {self.query_provider}...")
            queries = self.query_builder.generate_search_queries(claim, num_queries=5)
            
            if queries and len(queries) > 0:
                return queries
        
        print("   📝 Using rule-based query generation...")
        return self._build_rule_based_queries(claim, entities)


    def _build_rule_based_queries(self, claim: str, entities: Dict) -> List[str]:
        """Original rule-based query building (as backup)"""
        # Your existing build_targeted_queries logic here
        base_claim = claim
        queries = []

        queries.append(f'"{base_claim}" Sri Lanka')

        if 'money' in entities:
            for money in entities['money'][:2]:
                queries.append(f'"{money}" Sri Lanka economy financial')
    
        if 'percentages' in entities:
            for pct in entities['percentages'][:2]:
                queries.append(f'"{pct}" Sri Lanka growth rate economic')

        # Topic-specific queries
        if any(word in claim.lower() for word in ['gdp', 'growth', 'economy']):
            queries.append('Sri Lanka GDP growth economic performance')
        
        if any(word in claim.lower() for word in ['drug', 'mafia', 'crime']):
            queries.append('Sri Lanka drug problem criminal gangs')
        
        # Ensure we have at least 5 queries
        while len(queries) < 5:
            queries.append(f'Sri Lanka {claim.split()[0]} news')
        
        return queries[:5]

    
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
        print(f"   📊 Extracted entities: {entities}")
        
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

        print(f"   DEBUG: use_ai_filtering={use_ai_filtering}")
        print(f"   DEBUG: self.query_builder exists={self.query_builder is not None}")
        if self.query_builder:
            print(f"   DEBUG: has filter method={hasattr(self.query_builder, 'filter_relevant_evidence')}")

        #ai powred filtering
        if use_ai_filtering and self.query_builder and hasattr(self.query_builder, 'filter_relevant_evidence'):
            print(f"   🤖 Applying AI relevance filtering...")
            all_results = self.query_builder.filter_relevant_evidence(claim, all_results)
            print(f"   📋 {len(all_results)} results after AI filtering")

        else:
            print(f"   ⚠️ AI filtering skipped - conditions not met")
        
        # Sort by relevance and diversity
        diverse_results = self._ensure_source_diversity(all_results, num_results)
        
        return diverse_results[:num_results]

    # def retrieve_diverse_evidence_livefc(
    #     self, 
    #     claim: str, 
    #     num_results: int = 10,
    #     use_decomposition: bool = True
    # ) -> List[Dict]:
    #     """LiveFC-style evidence retrieval with claim decomposition"""
        
    #     if use_decomposition:
    #         # Step 1: Decompose claim into verification questions
    #         from .claim_decomposer import ClaimDecomposer
    #         from .multi_source_retriever import MultiSourceRetriever
    #         from .evidence_ranker import EvidenceRanker
            
    #         decomposer = ClaimDecomposer(self.query_builder.client.api_key)
    #         questions = decomposer.decompose_claim(claim, num_questions=5)
            
    #         print(f"   🔍 Decomposed into {len(questions)} verification questions")
    #         for i, q in enumerate(questions, 1):
    #             print(f"      Q{i}: {q}")
            
    #         # Step 2: Multi-source retrieval
    #         multi_retriever = MultiSourceRetriever(self.serpapi_key)
    #         all_evidence = multi_retriever.retrieve_from_multiple_sources(questions, max_per_source=8)
            
    #         print(f"   📋 Retrieved {len(all_evidence)} total pieces from multiple sources")
            
    #     else:
    #         # Fallback to original method
    #         return self.retrieve_diverse_evidence(claim, num_results)
        
    #     # Step 3: Advanced deduplication
    #     unique_evidence = self._advanced_deduplication(all_evidence)
        
    #     # Step 4: Cross-encoder ranking
    #     ranker = EvidenceRanker()
    #     ranked_evidence = ranker.rank_evidence(claim, unique_evidence, top_k=num_results * 2)
        
    #     # Step 5: Apply source diversity
    #     diverse_results = self._ensure_source_diversity(ranked_evidence, num_results)
        
    #     return diverse_results[:num_results]

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
            
            decomposer = ClaimDecomposer(self.query_builder.client.api_key)
            questions = decomposer.decompose_claim(claim, num_questions=4)  # Reduce to 4 since we're adding original
            
            print(f"   🔍 Decomposed into {len(questions)} verification questions")
            for i, q in enumerate(questions, 1):
                print(f"      Q{i}: {q}")
            
            # Step 2: Priority-aware multi-source retrieval WITH original claim
            multi_retriever = MultiSourceRetriever(self.serper_key)
            all_evidence = multi_retriever.retrieve_from_multiple_sources_priority(
                questions, 
                max_per_source=6,  # Slightly reduced since we're adding original claim searches
                original_claim=claim  # ADD THIS - pass the original claim
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
    
    def _search_single_query(self, query: str, max_results: int = 10) -> List[Dict]:
        """Execute a single search query"""
        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "num": max_results,
            "gl": "lk",
            "tbm": "nws"
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            results = response.json()
            
            evidence = []
            for result in results.get("news_results", []):
                evidence.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "source": result.get("source", ""),
                    "link": result.get("link", ""),
                    "date": result.get("date", "")
                })
            
            return evidence
        except Exception as e:
            print(f"      ⚠️ Query failed: {e}")
            return []
    
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

# Backward compatibility wrapper
class EvidenceRetriever(AdvancedEvidenceRetriever):
    def retrieve_evidence(self, claim: str, num_sources: int = 5) -> List[Dict]:
        """Retrieve and filter evidence for a given claim"""
        
        # Step 1: Extract entities
        entities = self.extract_claim_entities(claim)
        print(f"   📊 Extracted entities: {entities}")
        
        # Step 2: Generate targeted queries
        if self.query_builder:
            print("   🤖 Generating queries with Groq...")
            queries = self.query_builder.generate_search_queries(claim, num_queries=5)
            print(f"   🔍 Built {len(queries)} targeted queries")
            for i, query in enumerate(queries, 1):
                print(f"      Query {i}: {query[:50]}...")
        else:
            queries = self.build_targeted_queries(claim, entities)
        
        # Step 3: Execute searches and collect raw results
        all_results = []
        for query in queries:
            try:
                results = self._execute_search(query)
                all_results.extend(results)
            except Exception as e:
                print(f"      ⚠️ Search failed for query: {e}")
                continue
        
        print(f"   📋 Found {len(all_results)} initial results")
        
        # Step 4: Remove duplicates
        unique_results = self._remove_duplicates(all_results)
        
        # NEW STEP 5: Apply AI relevance filtering (THIS IS MISSING!)
        if self.query_builder and len(unique_results) > 0:
            print("   🤖 Applying AI relevance filtering...")
            filtered_evidence = self.query_builder.filter_relevant_evidence(claim, unique_results)
            print(f"   📋 {len(filtered_evidence)} results after AI filtering")
        else:
            filtered_evidence = unique_results
        
        # Step 6: Apply traditional relevance scoring and select top results
        if len(filtered_evidence) > 0:
            # Score remaining evidence
            for evidence in filtered_evidence:
                evidence['relevance_score'] = self._calculate_relevance(claim, evidence, entities)
            
            # Sort by relevance and take top results
            sorted_evidence = sorted(filtered_evidence, key=lambda x: x['relevance_score'], reverse=True)
            final_evidence = sorted_evidence[:num_sources]
            
            print(f"   ✅ Found {len(final_evidence)} evidence sources")
            return final_evidence
        else:
            print("   ❌ No relevant evidence found")
            return []