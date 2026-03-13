"""
TavilyEvidenceRetriever
-----------------------
Tavily API for web search instead of Serper/Google.

Flow is identical to the Serper-based retriever:
  retrieve_evidence_dataset_first()
      ├── STEP 1: retrieve_with_hansard_fallback()   [govt DB — unchanged]
      └── STEP 2: retrieve_hybrid_tavily_decomposition()  [Tavily web search]
                      ├── ClaimDecomposer.generate_search_queries()
                      ├── _simple_tavily_query()           [replaces Serper]
                      ├── _advanced_deduplication()
                      └── _filter_by_relevance_only()      [cross-encoder]

Original Serper flow is preserved in evidence_retrieval.py — this file
adds a parallel Tavily-backed flow without touching the original.
"""

import requests
import pandas as pd
from typing import List, Dict
import re
from datetime import datetime
from urllib.parse import urlparse

from src.evidence_retrieval.claim_decomposer import ClaimDecomposer


class TavilyEvidenceRetriever:
    def __init__(self, tavily_key: str, gemini_key: str = None, groq_key: str = None):
        self.tavily_key = tavily_key
        self.groq_key = groq_key

        # Lazy-init spaCy (same as original)
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            print(" spaCy NER model loaded")
        except Exception as e:
            print(f" Could not load spaCy model: {e}")
            self.nlp = None

        # Source categories — identical to AdvancedEvidenceRetriever
        self.source_categories = {
            'official_primary': [
                'parliament.lk',
                'president.gov.lk',
                'pmoffice.gov.lk',
                'cbsl.lk',
                'statistics.gov.lk'
            ],
            'fact_checkers': [
                'factcheck.lk',
                'factcrescendo.com',
                'boomlive.in',
                'factly.in'
            ],
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
            'government': [
                'dailynews.lk',
                'news.lk',
                'agrimin.gov.lk',
                'treasury.gov.lk'
            ],
            'international': [
                'worldbank.org',
                'imf.org',
                'reuters.com',
                'bbc.com',
                'ap.org'
            ]
        }

        self.priority_weights = {
            'official_primary': 3.0,
            'fact_checkers': 2.5,
            'sri_lankan_news': 2.0,
            'government': 1.5,
            'international': 1.0,
            'unknown': 0.5
        }

    # ------------------------------------------------------------------
    # DATE EXTRACTION: fallback when Tavily omits published_date
    # ------------------------------------------------------------------

    def _extract_date_from_content(self, content: str) -> str:
        """
        Tavily often omits published_date. This method scans the content
        text for common date patterns and returns the first match in
        ISO YYYY-MM-DD format, or '' if none found.
        """
        if not content:
            return ''

        # Named-month patterns: "March 11, 2026", "11 March 2026", "Mar 2, 2024"
        named = re.findall(
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|'
            r'September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|'
            r'Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b'
            r'|'
            r'\b(January|February|March|April|May|June|July|August|'
            r'September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|'
            r'Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b',
            content
        )

        for match in named:
            # Two groups per alternation — try both
            if match[0]:  # "11 March 2026" form
                day, month_str, year = match[0], match[1], match[2]
            else:          # "March 11, 2026" form
                month_str, day, year = match[3], match[4], match[5]
            try:
                for fmt in ('%d %B %Y', '%d %b %Y', '%B %d %Y', '%b %d %Y'):
                    try:
                        dt = datetime.strptime(f"{day} {month_str} {year}", '%d %B %Y'
                                               if len(month_str) > 3 else '%d %b %Y')
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        pass
                # Try month-first
                for fmt in ('%B %d %Y', '%b %d %Y'):
                    try:
                        dt = datetime.strptime(f"{month_str} {day} {year}", fmt)
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        pass
            except Exception:
                continue

        # ISO pattern: "2026-03-11" or "2024/03/02"
        iso = re.search(r'\b(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b', content)
        if iso:
            return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"

        # EconomyNext / wire format: "(Colombo/Mar11/2022)" or "(Colombo/Dec31/2025)"
        wire = re.search(
            r'\([A-Za-z]+/(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\d{1,2})/(\d{4})\)',
            content
        )
        if wire:
            try:
                dt = datetime.strptime(f"{wire.group(1)} {wire.group(2)} {wire.group(3)}", '%b %d %Y')
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass

        # Relative dates: "1 day ago", "2 days ago", "3h ago", "2d ·", "1 day ago"
        relative = re.search(
            r'\b(\d+)\s*(minute|hour|day|week|month|year)s?\s*ago\b'
            r'|'
            r'\b(\d+)(h|d|w|mo)\b',
            content, re.IGNORECASE
        )
        if relative:
            from datetime import timedelta
            now = datetime.now()
            if relative.group(1):  # "N unit ago" form
                n, unit = int(relative.group(1)), relative.group(2).lower()
            else:                   # "Nh / Nd / Nw / Nmo" form
                n, unit = int(relative.group(3)), relative.group(4).lower()
                unit = {'h': 'hour', 'd': 'day', 'w': 'week', 'mo': 'month'}.get(unit, 'day')

            delta_map = {
                'minute': timedelta(minutes=n),
                'hour':   timedelta(hours=n),
                'day':    timedelta(days=n),
                'week':   timedelta(weeks=n),
                'month':  timedelta(days=n * 30),
                'year':   timedelta(days=n * 365),
            }
            return (now - delta_map.get(unit, timedelta(days=n))).strftime('%Y-%m-%d')

        return ''

    # ------------------------------------------------------------------
    # CORE: Tavily web search  (replaces _simple_serper_query)
    # ------------------------------------------------------------------

    def _simple_tavily_query(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Execute a single Tavily search and normalise results to the same
        dict shape used throughout the pipeline:
            title, snippet, link, source, date, date_raw, position, search_type
        """
        try:
            from tavily import TavilyClient
        except ImportError:
            print(" tavily-python not installed. Run: pip install tavily-python")
            return []

        if not hasattr(self, '_tavily_client'):
            self._tavily_client = TavilyClient(api_key=self.tavily_key)

        try:
            response = self._tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",      # deeper crawl, same cost tier
                include_raw_content=False,    # snippets sufficient for NLI
                include_answer=False          # we do our own synthesis
            )

            #print raw Tavily response before mapping
            import json
            # print("\n--- RAW TAVILY RESPONSE ---")
            # print(json.dumps(response, indent=2, default=str))
            # print("--- END RAW TAVILY RESPONSE ---\n")

            results = []
            for item in response.get("results", []):
                content = item.get("content", "")
                # Use Tavily's published_date if present, otherwise extract from content
                pub_date = item.get("published_date", "") or ""
                if not pub_date:
                    pub_date = self._extract_date_from_content(content)

                results.append({
                    "title":       item.get("title", ""),
                    "snippet":     content,               # Tavily: 'content'
                    "link":        item.get("url", ""),   # Tavily: 'url'
                    "source":      self._extract_domain(item.get("url", "")),
                    "date":        pub_date,
                    "date_raw":    pub_date,
                    "position":    0,
                    "search_type": "tavily"
                })

            return results[:max_results]

        except Exception as e:
            print(f"  Tavily search error: {e}")
            return []

    # ------------------------------------------------------------------
    # Helpers (identical to AdvancedEvidenceRetriever)
    # ------------------------------------------------------------------

    def _extract_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except:
            return url.lower()

    def _advanced_deduplication(self, evidence_list: List[Dict]) -> List[Dict]:
        """Advanced deduplication using content similarity"""
        from difflib import SequenceMatcher

        unique_evidence = []
        seen_content = []
        seen_urls = set()

        for evidence in evidence_list:
            url = evidence.get('link', evidence.get('url', ''))
            if url and url in seen_urls:
                continue

            content = f"{evidence.get('title', '')} {evidence.get('snippet', evidence.get('text', ''))}"
            if len(content.strip()) < 50:
                continue

            is_duplicate = False
            for seen_content_text in seen_content:
                similarity = SequenceMatcher(None, content.lower(), seen_content_text.lower()).ratio()
                if similarity > 0.95:
                    is_duplicate = True
                    break

            # Be even more lenient for government/dataset sources
            if evidence.get('dataset') or evidence.get('dataset_source'):
                is_duplicate = False
                for seen_content_text in seen_content:
                    similarity = SequenceMatcher(None, content.lower(), seen_content_text.lower()).ratio()
                    if similarity > 0.98:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_evidence.append(evidence)
                seen_content.append(content)
                if url:
                    seen_urls.add(url)

        print(f"Deduplication: {len(evidence_list)} -> {len(unique_evidence)} unique pieces")

        if len(evidence_list) - len(unique_evidence) > 2:
            print(f"   Removed {len(evidence_list) - len(unique_evidence)} duplicates")
            remaining_datasets = {}
            for evidence in unique_evidence:
                dataset = evidence.get('dataset', evidence.get('dataset_source', 'web'))
                remaining_datasets[dataset] = remaining_datasets.get(dataset, 0) + 1
            if remaining_datasets:
                print(f"   Remaining by source: {dict(remaining_datasets)}")

        return unique_evidence

    def _is_social_media_source(self, url: str) -> bool:
        social_media_domains = [
            'facebook.com', 'twitter.com', 'x.com', 'instagram.com',
            'tiktok.com', 'youtube.com', 'youtu.be', 'linkedin.com',
            'reddit.com', 'pinterest.com', 'snapchat.com',
            'telegram.org', 'whatsapp.com'
        ]
        domain = self._extract_domain(url)
        return any(s in domain for s in social_media_domains)

    def _filter_by_relevance_only(self, claim: str, evidence_list: List[Dict], top_k: int = 10) -> List[Dict]:
        """Cross-encoder relevance filtering — identical to original"""
        print(f"   Cross-encoder relevance filtering (pure relevance approach)...")
        print(f"      Target: {top_k} most relevant sources")

        if not evidence_list:
            return []

        try:
            from sentence_transformers import CrossEncoder
            if not hasattr(self, 'relevance_ranker'):
                print(f"      Loading cross-encoder for relevance...")
                self.relevance_ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
                print(f"      Cross-encoder loaded")
        except Exception as e:
            print(f"     Cross-encoder loading failed: {e}")
            return evidence_list[:top_k]

        pairs = []
        valid_evidence = []

        for evidence in evidence_list:
            evidence_text = f"{evidence.get('title', '')} {evidence.get('snippet', '')}".strip()
            if evidence_text and len(evidence_text) > 10:
                pairs.append([claim, evidence_text])
                valid_evidence.append(evidence)

        if not pairs:
            print(f"      No valid text content found")
            return evidence_list[:top_k]

        print(f"Scoring {len(pairs)} evidence pieces...")

        try:
            scores = self.relevance_ranker.predict(pairs)
            scored_evidence = sorted(zip(valid_evidence, scores), key=lambda x: x[1], reverse=True)

            print(f" Relevance scores:")
            for i, (evidence, score) in enumerate(scored_evidence[:5], 1):
                title = evidence.get('title', 'No title')[:50]
                print(f"         {i}. {title}... (score: {score:.3f})")

            top_evidence = [e for e, _ in scored_evidence[:top_k]]
            print(f"Selected {len(top_evidence)} most relevant sources")
            return top_evidence

        except Exception as e:
            print(f"      Scoring failed: {e}")
            return evidence_list[:top_k]

    # ------------------------------------------------------------------
    # WEB RETRIEVAL: Tavily-backed decomposition search
    # ------------------------------------------------------------------

    def retrieve_hybrid_tavily_decomposition(self, claim: str, num_results: int = 20,
                                              results_per_query: int = 10) -> List[Dict]:
        """
        Multi-query Tavily search — mirrors retrieve_hybrid_serper_decomposition().

        Steps:
          1. ClaimDecomposer generates 5 search-optimised queries
          2. Each query hits Tavily independently
          3. Deduplicate
          4. Cross-encoder relevance filter → top num_results
        """
        print(f" Target final results: {num_results}")

        all_results = []

        # Step 1: Generate search-optimised queries (same ClaimDecomposer as before)
        print(f" Generating search-optimised queries...")
        decomposer = ClaimDecomposer(self.groq_key)
        search_queries = decomposer.generate_search_queries(claim, num_queries=5)

        print(f" Generated {len(search_queries)} search-optimised queries:")
        for i, q in enumerate(search_queries, 1):
            print(f"      Query {i}: {q[:80]}...")

        # Step 2: Search via Tavily
        print(f"  Searching {results_per_query} results per query...")

        import time
        for i, query in enumerate(search_queries):
            print(f"  Tavily query {i+1} search...")
            query_results = self._simple_tavily_query(query, results_per_query)

            for result in query_results:
                result['query_source'] = f'tavily_query_{i+1}'
                result['query_text'] = query

            all_results.extend(query_results)
            print(f"    Retrieved {len(query_results)} results")
            time.sleep(0.3)   # slightly shorter wait; Tavily is lenient on rate limits

        print(f"    Retrieved {len(all_results)} total results from {len(search_queries)} queries")

        # Step 3: Deduplicate
        unique_results = self._advanced_deduplication(all_results)
        print(f"    After deduplication: {len(unique_results)} unique results")

        # Step 4: Cross-encoder relevance filter
        relevant_results = self._filter_by_relevance_only(claim, unique_results, top_k=num_results)
        print(f"    Final selection: {len(relevant_results)} highly relevant evidence pieces")
        return relevant_results

    # Alias so core_pipeline.py can call either name interchangeably
    def retrieve_hybrid_serper_decomposition(self, claim: str, num_results: int = 20,
                                              results_per_query: int = 10) -> List[Dict]:
        return self.retrieve_hybrid_tavily_decomposition(claim, num_results, results_per_query)

    # ------------------------------------------------------------------
    # DATABASE RETRIEVAL: unchanged from original
    # ------------------------------------------------------------------

    def retrieve_with_hansard_fallback(self, claim: str, num_results: int = 10) -> List[Dict]:
        """
        Government dataset search (PMD, Cabinet, Parliament, Courts).
        Identical to AdvancedEvidenceRetriever.retrieve_with_hansard_fallback().
        """
        print(f"Searching enhanced government databases (PMD, Cabinet, Parliament, Courts)...")

        try:
            from src.database_retrieval.dataset_manager import DatasetManager
            db_manager = DatasetManager()

            priority_datasets = ['pmd_press_releases', 'pmd_press_release',
                                  'cabinet_decisions', 'hansard', 'supreme_court']

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

            print(f"Performing enhanced search with claim routing...")

            try:
                database_results = db_manager.hybrid_search_datasets(
                    query=claim,
                    max_results=num_results,
                    claim=claim
                )
                print(f"Hybrid search found {len(database_results)} results")
            except Exception as e:
                print(f"Hybrid search failed ({e}), using keyword search...")
                database_results = db_manager.search_datasets(
                    query=claim,
                    max_results=num_results,
                    claim=claim
                )
                print(f"Keyword search found {len(database_results)} results")

            formatted_results = []
            for result in database_results:
                formatted_results.append({
                    'title':           result['title'],
                    'snippet':         result['snippet'],
                    'source':          f"Sri Lanka Government Database - {result['source']}",
                    'link':            result['link'],
                    'date':            result.get('date', ''),
                    'authority_weight': result['authority_weight'],
                    'relevance_score': result.get('final_score',
                                        result.get('similarity_score',
                                        result.get('relevance_score', 0.5))),
                    'search_method':   result.get('search_method', 'database'),
                    'dataset_source':  result['dataset_source']
                })

            print(f"Top database matches:")
            for i, result in enumerate(formatted_results[:3], 1):
                score = result['relevance_score']
                dataset = result['dataset_source'].replace('_', ' ').title()
                print(f"   {i}. {result['title'][:70]}...")
                print(f"      {dataset} | Score: {score:.3f}")

            return formatted_results

        except Exception as e:
            print(f"Enhanced database search failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------

    def retrieve_evidence_dataset_first(self, claim: str, num_results: int = 20) -> List[Dict]:
        """
        Dataset-first evidence retrieval with Tavily web fallback.

        Mirrors AdvancedEvidenceRetriever.retrieve_evidence_dataset_first()
        exactly — only the web search backend has changed (Serper → Tavily).

        Steps
        -----
        STEP 1  Search government databases (primary)
        STEP 2  Tavily web search fallback (if DB results insufficient)
        STEP 3  Deduplicate + re-rank by authority × relevance
        """
        print(f"[Tavily] DATASET-FIRST Evidence Retrieval for: '{claim[:60]}...'")
        print(f"Target results: {num_results}")

        all_results = []

        # STEP 1: Government databases (primary)
        print(f"\nSTEP 1: Searching Government Databases...")
        database_results = self.retrieve_with_hansard_fallback(claim, num_results=num_results // 2)

        if len(database_results) > 0:
            print(f"Found {len(database_results)} authoritative database results")
            all_results.extend(database_results)

            high_quality_results = [r for r in database_results if r['relevance_score'] > 0.4]
            if len(high_quality_results) >= num_results // 2:
                print(f"Sufficient high-quality database results ({len(high_quality_results)})")
                print(f"   Skipping web search to prioritise authoritative sources")
                return all_results[:num_results]
        else:
            print(f"No database results found")

        # STEP 2: Tavily web fallback
        remaining_results = num_results - len(all_results)
        if remaining_results > 0:
            print(f"\nSTEP 2: Tavily Web Search Fallback ({remaining_results} additional results needed)...")

            try:
                web_results = self.retrieve_hybrid_tavily_decomposition(
                    claim,
                    num_results=remaining_results,
                    results_per_query=5
                )

                for result in web_results:
                    result['source'] = f"Web Search (Tavily) - {result.get('source', 'Unknown')}"
                    result['authority_weight'] = result.get('authority_weight', 0.5) * 0.8
                    result['search_method'] = 'tavily_web_fallback'

                all_results.extend(web_results[:remaining_results])
                print(f"Added {len(web_results[:remaining_results])} Tavily web results")

            except Exception as e:
                print(f"Tavily web fallback failed: {e}")

        # STEP 3: Final deduplication + re-rank
        print(f"\nSTEP 3: Final Processing...")
        all_results = self._advanced_deduplication(all_results)
        all_results.sort(
            key=lambda x: x.get('authority_weight', 0.5) * x.get('relevance_score', 0.5),
            reverse=True
        )
        final_results = all_results[:num_results]

        database_count = sum(1 for r in final_results if r.get('search_method') != 'tavily_web_fallback')
        web_count = len(final_results) - database_count

        print(f"\nFINAL RESULTS SUMMARY (Tavily flow):")
        print(f"   Database sources: {database_count}")
        print(f"   Tavily web sources: {web_count}")
        print(f"   Total results: {len(final_results)}")

        return final_results
