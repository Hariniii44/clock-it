"""
GoogleAIModeRetriever
---------------------
Uses SerpAPI's Google AI Mode engine as the web evidence source.

Google AI Mode returns a pre-curated set of references (typically 7) that
Google's own AI has already ranked for temporal relevance and factual
authority.  This eliminates the stale-article problem that affects Tavily
(2022/2024 kottu articles polluting 2026 results) because Google surfaces
the right time period automatically.

Compared to TavilyEvidenceRetriever:
  - 1 API call per claim instead of 5 × decomposed queries
  - Fewer sources (~7) but higher average quality
  - Also returns Google's own synthesised answer (stored as `google_synthesis`)
    which can be used as a baseline comparison in evaluation

Interface is identical to TavilyEvidenceRetriever so it can be swapped into
core_pipeline.py with a one-line change.

Original Tavily and Serper flows are preserved in their own files — this adds
a third parallel retrieval option without touching anything else.
"""

from typing import Dict, List
from urllib.parse import urlparse

from src.evidence_retrieval.claim_decomposer import ClaimDecomposer


class GoogleAIModeRetriever:
    """
    Drop-in replacement for TavilyEvidenceRetriever that uses SerpAPI's
    Google AI Mode engine.
    """

    def __init__(self, serp_api_key: str, groq_key: str = None, tavily_key: str = None):
        self.serp_api_key = serp_api_key
        self.groq_key     = groq_key
        self.tavily_key   = tavily_key  # kept for interface compatibility, no longer used

        # Lazy-init spaCy (same as other retrievers)
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            print(" spaCy NER model loaded")
        except Exception as e:
            print(f" Could not load spaCy model: {e}")
            self.nlp = None

        # Source priority tiers — identical to TavilyEvidenceRetriever
        self.source_categories = {
            'official_primary': [
                'parliament.lk', 'president.gov.lk', 'pmoffice.gov.lk',
                'cbsl.lk', 'statistics.gov.lk',
            ],
            'fact_checkers': [
                'factcheck.lk', 'factcrescendo.com', 'boomlive.in', 'factly.in',
            ],
            'sri_lankan_news': [
                'dailymirror.lk', 'economynext.com', 'island.lk', 'newsfirst.lk',
                'adaderana.lk', 'ft.lk', 'themorning.lk', 'sundayobserver.lk',
            ],
            'government': [
                'dailynews.lk', 'news.lk', 'agrimin.gov.lk', 'treasury.gov.lk',
            ],
            'international': [
                'worldbank.org', 'imf.org', 'reuters.com', 'bbc.com', 'ap.org',
            ],
        }

        self.priority_weights = {
            'official_primary': 3.0,
            'fact_checkers': 2.5,
            'sri_lankan_news': 2.0,
            'government': 1.5,
            'international': 1.0,
            'unknown': 0.5,
        }

        # Domains that are irrelevant to news fact-checking
        # (restaurant menus, food delivery, e-commerce, etc.)
        self.blocked_domains = [
            'ubereats.com', 'uber.com', 'deliveroo.com', 'foodpanda.com',
            'zomato.com', 'swiggy.com', 'tripadvisor.com', 'yelp.com',
            'booking.com', 'airbnb.com', 'amazon.com', 'ebay.com',
            'wikipedia.org',  # keep out; too generic for claim verification
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ''

    def _clean_url(self, url: str) -> str:
        """Strip Google text-fragment anchors (#:~:text=...) before fetching."""
        return url.split('#')[0]

    # Social media domains — can't extract past auth walls, use snippet as-is
    SOCIAL_MEDIA_DOMAINS = {
        'facebook.com', 'x.com', 'twitter.com', 'instagram.com',
        'tiktok.com', 'linkedin.com', 'threads.net',
    }

    def _is_social_media(self, url: str) -> bool:
        domain = self._extract_domain(url)
        return any(s in domain for s in self.SOCIAL_MEDIA_DOMAINS)

    def _fetch_with_trafilatura(self, url: str) -> str:
        """
        Fetch and extract clean article text using trafilatura.

        Strategy:
          1. Try AMP version first (pure static HTML, no JS rendering needed).
          2. Fall back to requests with browser-like headers (bypasses basic bot detection).
          3. Fall back to trafilatura's built-in fetch.
        Returns empty string on failure.
        """
        import trafilatura

        def _extract(html: str) -> str:
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
            return (text or '').strip()

        # --- Strategy 1: AMP version (cleaner static HTML on many news sites) ---
        try:
            amp_url = url.rstrip('/') + '/amp/'
            import requests
            headers = {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/122.0.0.0 Safari/537.36'
                ),
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Referer': 'https://www.google.com/',
            }
            resp = requests.get(amp_url, headers=headers, timeout=10)
            if resp.status_code == 200 and len(resp.text) > 500:
                text = _extract(resp.text)
                if text and len(text) > 200:
                    return text
        except Exception:
            pass

        # --- Strategy 2: requests with browser headers (bypasses basic bot detection) ---
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                text = _extract(resp.text)
                if text and len(text) > 200:
                    return text
        except Exception:
            pass

        # --- Strategy 3: trafilatura built-in fetch (fallback) ---
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return ''
            return _extract(downloaded)
        except Exception:
            return ''

    def _enrich_with_full_content(self, references: List[Dict]) -> List[Dict]:
        """
        Fetch full article text for each reference in parallel:
          - Social media URLs  → keep original SerpAPI snippet (auth walls)
          - News/article URLs  → fetch via trafilatura (clean article extraction)
        Deduplicates by clean base URL so the same article is only fetched once.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Build de-duplicated map: clean_url → [refs that share this base URL]
        seen_urls: dict = {}
        for ref in references:
            key = self._clean_url(ref['link'])
            seen_urls.setdefault(key, []).append(ref)

        news_urls = [u for u in seen_urls if not self._is_social_media(u)]
        social_urls = [u for u in seen_urls if self._is_social_media(u)]

        print(f"  Content fetch: {len(news_urls)} news URLs via trafilatura (parallel), "
              f"{len(social_urls)} social media URLs kept as snippets")

        # Fetch all news URLs in parallel (max 8 workers, 15s per request)
        results: dict = {}  # clean_url → text or ''
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_url = {
                executor.submit(self._fetch_with_trafilatura, url): url
                for url in news_urls
            }
            for future in as_completed(future_to_url, timeout=60):
                url = future_to_url[future]
                try:
                    results[url] = future.result()
                except Exception:
                    results[url] = ''

        enriched = 0
        for clean_url in news_urls:
            text = results.get(clean_url, '')
            if text:
                content = text[:2000]
                for ref in seen_urls[clean_url]:
                    ref['snippet'] = content
                    ref['full_content_fetched'] = True
                enriched += len(seen_urls[clean_url])

                # --- TRAFILATURA EXTRACT OUTPUT ---
                print("\n" + "-"*60)
                print(f"TRAFILATURA — {clean_url}")
                print("-"*60)
                print(content)
                print("-"*60)
            else:
                print(f"  trafilatura: no content extracted for {clean_url} — keeping snippet")

        print(f"  Full content fetched for {enriched}/{len(references)} sources "
              f"({len(references) - enriched} kept original snippets)")
        return references

    def _is_blocked(self, url: str) -> bool:
        domain = self._extract_domain(url)
        return any(blocked in domain for blocked in self.blocked_domains)

    def _get_source_priority(self, url: str) -> float:
        domain = self._extract_domain(url)
        for category, domains in self.source_categories.items():
            if any(d in domain for d in domains):
                return self.priority_weights[category]
        return self.priority_weights['unknown']

    # ------------------------------------------------------------------
    # Core: single Google AI Mode query
    # ------------------------------------------------------------------

    def _google_ai_mode_query(self, query: str) -> Dict:
        """
        Execute a single Google AI Mode search via SerpAPI.

        Returns a dict with keys:
          references      — list of normalised evidence dicts
          synthesis       — Google's own synthesised answer text (plain string)
          raw_response    — full SerpAPI response (for debugging)
        """
        try:
            from serpapi import GoogleSearch
        except ImportError:
            print(" google-search-results not installed. Run: pip install google-search-results")
            return {'references': [], 'synthesis': '', 'raw_response': {}}

        params = {
            'engine': 'google_ai_mode',
            'q': query,
            'hl': 'en',
            'gl': 'lk',          # Sri Lanka locale for better local results
            'api_key': self.serp_api_key,
        }

        try:
            search = GoogleSearch(params)
            raw = search.get_dict()

            # Build normalised evidence list from references
            references = []
            for ref in raw.get('references', []):
                url = ref.get('link', '')
                references.append({
                    'title':       ref.get('title', ''),
                    'snippet':     ref.get('snippet', ''),
                    'link':        url,
                    'source':      self._extract_domain(url),
                    'date':        self._parse_date_from_snippet(ref.get('snippet', '')),
                    'date_raw':    ref.get('snippet', '')[:50],
                    'position':    ref.get('index', 0),
                    'search_type': 'google_ai_mode',
                })

            # Extract synthesised answer from text_blocks
            synthesis_parts = []
            for block in raw.get('text_blocks', []):
                if block.get('type') == 'paragraph':
                    text = block.get('snippet', '').strip()
                    if text:
                        synthesis_parts.append(text)
                elif block.get('type') == 'list':
                    for item in block.get('list', []):
                        text = item.get('snippet', '').strip()
                        if text:
                            synthesis_parts.append(f"• {text}")

            synthesis = '\n'.join(synthesis_parts)

            print(f"  Google AI Mode: {len(references)} references retrieved")
            if synthesis:
                print(f"  Google synthesis: {synthesis[:150]}{'...' if len(synthesis) > 150 else ''}")

            # --- RAW SERP OUTPUT ---
            import json
            print("\n" + "="*60)
            print("RAW SERP API RESPONSE")
            print("="*60)
            print(json.dumps(raw, indent=2, default=str))
            print("="*60 + "\n")

            return {
                'references': references,
                'synthesis': synthesis,
                'raw_response': raw,
            }

        except Exception as e:
            print(f"  Google AI Mode search error: {e}")
            return {'references': [], 'synthesis': '', 'raw_response': {}}

    def _parse_date_from_snippet(self, snippet: str) -> str:
        """
        Extract a date from the reference snippet.
        SerpAPI snippets often start with 'Mar 11, 2026 —' or similar.
        """
        import re
        from datetime import datetime

        # Pattern: "Mar 11, 2026" or "March 11, 2026" at start of snippet
        match = re.search(
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
            r'January|February|March|April|May|June|July|August|'
            r'September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
            snippet
        )
        if match:
            try:
                month_str, day, year = match.group(1), match.group(2), match.group(3)
                fmt = '%B %d %Y' if len(month_str) > 3 else '%b %d %Y'
                dt = datetime.strptime(f"{month_str} {day} {year}", fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass

        # ISO pattern
        iso = re.search(r'\b(20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b', snippet)
        if iso:
            return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}"

        return ''

    # ------------------------------------------------------------------
    # Government database retrieval (unchanged from Tavily retriever)
    # ------------------------------------------------------------------

    def retrieve_with_hansard_fallback(self, claim: str, num_results: int = 10) -> List[Dict]:
        """
        Search government document datasets (Hansard, PMD, Cabinet, etc.)
        via HybridRetriever.  Identical to TavilyEvidenceRetriever.
        """
        try:
            from src.retrieval.hybrid_retriever import HybridRetriever
            retriever = HybridRetriever()
            results = retriever.hybrid_search(
                query=claim,
                claim_types=None,
                total_results=num_results,
                use_query_expansion=True,
            )
            formatted = []
            for r in results:
                formatted.append({
                    'title':          r['title'],
                    'snippet':        r.get('passage', r['text']),
                    'link':           r['url'],
                    'source':         r['source'],
                    'date':           r.get('date', ''),
                    'date_raw':       r.get('date', ''),
                    'dataset_source': r['dataset'],
                    'authority':      r['authority'],
                    'relevance_score': r['similarity_score'],
                    'search_type':    'database',
                })
            return formatted
        except Exception as e:
            print(f"  Database retrieval failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Main web retrieval: single Google AI Mode call
    # ------------------------------------------------------------------

    def retrieve_google_ai_mode(
        self, claim: str, num_results: int = 20
    ) -> List[Dict]:
        """
        Retrieve evidence using a single Google AI Mode query.

        Stores Google's synthesis on the first result as `google_synthesis`
        so the pipeline can surface it alongside the verdict.
        """
        print(f"\n  Querying Google AI Mode...")
        result = self._google_ai_mode_query(claim)
        references = result['references']
        synthesis = result['synthesis']

        if not references:
            print("  Google AI Mode returned no references.")
            return []

        # Drop irrelevant domains (restaurant menus, food delivery, e-commerce)
        before = len(references)
        references = [r for r in references if not self._is_blocked(r['link'])]
        dropped = before - len(references)
        if dropped:
            print(f"  Filtered {dropped} irrelevant domain(s) (menus/delivery/e-commerce)")

        # Sort by source priority (official > fact-checkers > news > unknown)
        references.sort(key=lambda r: self._get_source_priority(r['link']), reverse=True)

        # Fetch full article content via Tavily Extract
        references = self._enrich_with_full_content(references[:num_results])

        # Deduplicate by clean URL — same article shouldn't cast multiple NLI votes
        seen = set()
        deduped = []
        for ref in references:
            key = self._clean_url(ref['link'])
            if key not in seen:
                seen.add(key)
                deduped.append(ref)
        if len(deduped) < len(references):
            print(f"  Evidence dedup: {len(references)} → {len(deduped)} unique sources")
        references = deduped

        # Attach synthesis to first result so it surfaces in output
        if synthesis and references:
            references[0]['google_synthesis'] = synthesis

        return references

    # ------------------------------------------------------------------
    # Interface aliases — identical names to TavilyEvidenceRetriever
    # so core_pipeline.py needs zero changes when swapping retrievers
    # ------------------------------------------------------------------

    def retrieve_hybrid_serper_decomposition(
        self, claim: str, num_results: int = 20, results_per_query: int = 8
    ) -> List[Dict]:
        """
        Pipeline-compatible alias.  Google AI Mode doesn't need query
        decomposition — a single query returns curated references.
        The num_results / results_per_query params are accepted for
        compatibility but only num_results is used as an upper bound.
        """
        return self.retrieve_google_ai_mode(claim, num_results=num_results)

    def retrieve_evidence_dataset_first(
        self, claim: str, num_results: int = 20
    ) -> List[Dict]:
        """
        Full retrieval: government database first, then Google AI Mode web.
        Drop-in replacement for TavilyEvidenceRetriever.retrieve_evidence_dataset_first().
        """
        print("\n  [GoogleAIModeRetriever] Dataset-first evidence retrieval")

        # Step 1: government document datasets
        db_evidence = self.retrieve_with_hansard_fallback(claim, num_results=10)
        print(f"  Database sources: {len(db_evidence)}")

        # Step 2: Google AI Mode web evidence
        web_evidence = self.retrieve_google_ai_mode(claim, num_results=num_results)
        print(f"  Web sources (Google AI Mode): {len(web_evidence)}")

        combined = db_evidence + web_evidence
        print(f"  Total: {len(combined)} sources")
        return combined
