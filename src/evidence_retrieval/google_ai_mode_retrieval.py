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

    def _extract_key_entities(self, text: str) -> set:
        """
        Extract key entities (proper nouns, dates, locations) from text.
        Used for title-content coherence check.
        """
        if not text or not self.nlp:
            return set()
        
        # Simple keyword extraction using spaCy NER
        doc = self.nlp(text[:500])  # First 500 chars for speed
        entities = set()
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in {'GPE', 'ORG', 'PERSON', 'DATE', 'EVENT', 'PRODUCT'}:
                entities.add(ent.text.lower())
        
        # Also extract capitalized words (potential proper nouns)
        for token in doc:
            if token.is_alpha and token.text[0].isupper() and len(token.text) > 3:
                entities.add(token.text.lower())
        
        return entities
    
    def _title_content_coherence(self, title: str, content: str) -> float:
        """
        Check if fetched content actually matches what the title promises.
        Returns overlap score (0.0 to 1.0).
        
        This catches cases where content fetch grabbed the wrong article
        (e.g., title: "Sri Lanka fuel prices", content: "Malaysia cosmetics ban").
        """
        if not title or not content:
            return 1.0  # No title to compare, assume coherent
        
        title_entities = self._extract_key_entities(title)
        content_entities = self._extract_key_entities(content[:1000])
        
        if not title_entities:
            return 1.0  # No entities in title, can't verify
        
        # Calculate Jaccard similarity
        overlap = len(title_entities & content_entities)
        union = len(title_entities | content_entities)
        
        return overlap / union if union > 0 else 0.0
    
    def _semantic_similarity(self, claim: str, content: str) -> float:
        """
        Calculate semantic similarity between claim and fetched content.
        Uses sentence embeddings (sentence-transformers if available).
        
        Returns similarity score (0.0 to 1.0).
        """
        try:
            # Try to use sentence-transformers if available
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            # Lazy-load model (cache it as instance variable)
            if not hasattr(self, '_embedding_model'):
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get embeddings
            claim_emb = self._embedding_model.encode(claim, convert_to_tensor=False)
            content_emb = self._embedding_model.encode(content[:500], convert_to_tensor=False)
            
            # Cosine similarity
            similarity = np.dot(claim_emb, content_emb) / (
                np.linalg.norm(claim_emb) * np.linalg.norm(content_emb)
            )
            
            return float(similarity)
        
        except ImportError:
            # Fallback: simple keyword overlap if sentence-transformers not available
            claim_words = set(claim.lower().split())
            content_words = set(content[:500].lower().split())
            
            overlap = len(claim_words & content_words)
            return overlap / max(len(claim_words), 1)
    
    def _is_navigation_content(self, text: str) -> bool:
        """
        Detect if extracted text is navigation/boilerplate rather than article body.

        Jina AI Reader sometimes returns full page Markdown including nav menus
        (e.g. '- [Home](/) - [About](/about) ...') for JS-heavy sites.
        PDF readers sometimes return institutional headers before the actual text.

        Returns True (= poor quality, discard) when:
          • >65 % of non-empty lines are short (<60 chars) — nav menu pattern
          • URL density >1 URL per 80 chars — link-list pages
          • >30 % of lines are Markdown hyperlink bullets ('- [text](url)')
        """
        if not text:
            return True
        lines = [l for l in text.split('\n') if l.strip()]
        if not lines:
            return True

        short_lines   = sum(1 for l in lines if len(l.strip()) < 60)
        md_link_lines = sum(1 for l in lines if l.strip().startswith(('- [', '* [', '+ [')))
        url_count     = text.count('http')

        # Navigation-menu heuristic
        if short_lines / len(lines) > 0.65:
            return True
        # Link-list / sitemap heuristic
        if url_count > len(text) / 80:
            return True
        # Jina markdown-nav bullets heuristic
        if len(lines) > 5 and md_link_lines / len(lines) > 0.30:
            return True

        return False

    def _fetch_with_trafilatura(self, url: str) -> str:
        """
        Fetch and extract clean article text.

        Strategy:
          1. Try AMP version first (pure static HTML, no JS rendering needed).
          2. Fall back to requests with browser-like headers (bypasses basic bot detection).
          3. Fall back to trafilatura's built-in fetch.
          4. Fall back to Jina AI Reader (handles JS-rendered / dynamic pages).
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
            resp = requests.get(amp_url, headers=headers, timeout=3)
            if resp.status_code == 200 and len(resp.text) > 500:
                text = _extract(resp.text)
                if text and len(text) > 200:
                    return text
        except Exception:
            pass

        # --- Strategy 2: requests with browser headers (bypasses basic bot detection) ---
        try:
            resp = requests.get(url, headers=headers, timeout=3)
            if resp.status_code == 200:
                text = _extract(resp.text)
                if text and len(text) > 200:
                    return text
        except Exception:
            pass

        # --- Strategy 3: trafilatura built-in fetch (fallback) ---
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = _extract(downloaded)
                if text and len(text) > 200:
                    return text
        except Exception:
            pass

        # --- Strategy 4: Jina AI Reader (handles JS-rendered / dynamic pages) ---
        try:
            jina_resp = requests.get(
                f'https://r.jina.ai/{url}',
                headers={'Accept': 'text/plain', 'User-Agent': 'Mozilla/5.0'},
                timeout=5,
            )
            if jina_resp.status_code == 200 and len(jina_resp.text) > 200:
                print(f"  jina: content fetched for {url}")
                return jina_resp.text.strip()
        except Exception:
            pass

        return ''

    def _enrich_with_full_content(self, references: List[Dict], claim: str = "") -> List[Dict]:
        """
        Fetch full article text for each reference in parallel:
          - Social media URLs  → keep original SerpAPI snippet (auth walls)
          - News/article URLs  → fetch via trafilatura (clean article extraction)
        Deduplicates by clean base URL so the same article is only fetched once.
        
        Hybrid relevance check before replacing snippet:
          1. Title-content coherence: Does fetched content match the title?
          2. Semantic similarity: Is fetched content relevant to the claim?
        If either check fails, keep original SERP snippet instead.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Build de-duplicated map: clean_url → [refs that share this base URL]
        seen_urls: dict = {}
        for ref in references:
            key = self._clean_url(ref['link'])
            seen_urls.setdefault(key, []).append(ref)

        # URLs with a substantial SerpAPI snippet (>250 chars) already give NLI
        # enough to work with — skip full-content fetch for these.
        # Google AI Mode snippets are typically 200-400 chars, so most are skipped.
        SKIP_FETCH_SNIPPET_LEN = 250
        news_urls_all = [u for u in seen_urls if not self._is_social_media(u)]
        social_urls   = [u for u in seen_urls if self._is_social_media(u)]

        news_urls_fetch = [
            u for u in news_urls_all
            if len(seen_urls[u][0].get('snippet', '')) < SKIP_FETCH_SNIPPET_LEN
        ]
        news_urls_skip  = [u for u in news_urls_all if u not in set(news_urls_fetch)]
        news_urls = news_urls_all  # keep original name for result loop below

        print(f"  Content fetch: {len(news_urls_fetch)}/{len(news_urls_all)} news URLs need fetch "
              f"({len(news_urls_skip)} already have sufficient snippets), "
              f"{len(social_urls)} social media kept as-is")

        # Fetch only URLs that need it, in parallel
        results: dict = {}  # clean_url → text or ''
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_url = {
                executor.submit(self._fetch_with_trafilatura, url): url
                for url in news_urls_fetch
            }
            try:
                for future in as_completed(future_to_url, timeout=15):
                    url = future_to_url[future]
                    try:
                        results[url] = future.result()
                    except Exception:
                        results[url] = ''
            except TimeoutError:
                # One or more fetches exceeded the wall-clock budget.
                # Mark unfinished URLs as empty so partial results are kept.
                for url in news_urls_fetch:
                    if url not in results:
                        results[url] = ''
                        print(f"  trafilatura: fetch timed out for {url} — keeping snippet")

        # Fallback statistics
        enriched = 0
        navigation_fallback = 0
        irrelevance_fallback = 0
        
        for clean_url in news_urls:
            text = results.get(clean_url, '')
            if text and not self._is_navigation_content(text):
                content = text[:2000]
                
                # HYBRID RELEVANCE CHECK (NEW)
                # Stage 1: Title-content coherence
                first_ref = seen_urls[clean_url][0]
                title = first_ref.get('title', '')
                coherence_score = self._title_content_coherence(title, content)
                
                # Stage 2: Semantic similarity (if claim provided)
                semantic_score = 1.0  # default: assume relevant
                if claim:
                    semantic_score = self._semantic_similarity(claim, content)
                
                # Thresholds
                COHERENCE_THRESHOLD = 0.15  # At least 15% entity overlap between title and content
                SEMANTIC_THRESHOLD = 0.25   # At least 25% similarity to claim
                
                # Decision: Use fetched content or fallback to SERP snippet?
                if coherence_score < COHERENCE_THRESHOLD:
                    # Fetched content doesn't match title (probably wrong article)
                    print(f"  [Relevance] Title-content mismatch for {clean_url[:60]} "
                          f"(coherence={coherence_score:.2f}) — keeping SERP snippet")
                    irrelevance_fallback += len(seen_urls[clean_url])
                    
                elif claim and semantic_score < SEMANTIC_THRESHOLD:
                    # Fetched content not relevant to claim
                    print(f"  [Relevance] Content irrelevant for {clean_url[:60]} "
                          f"(similarity={semantic_score:.2f}) — keeping SERP snippet")
                    irrelevance_fallback += len(seen_urls[clean_url])
                    
                else:
                    # Content passed both checks — use it
                    for ref in seen_urls[clean_url]:
                        ref['snippet'] = content
                        ref['full_content_fetched'] = True
                        ref['_coherence_score'] = coherence_score
                        ref['_semantic_score'] = semantic_score
                    enriched += len(seen_urls[clean_url])
                    
            elif text:
                print(f"  trafilatura: navigation/boilerplate detected for {clean_url} — keeping original snippet")
                navigation_fallback += len(seen_urls[clean_url])
            else:
                print(f"  trafilatura: no content extracted for {clean_url} — keeping snippet")

        print(f"  Content enrichment: {enriched} used fetched content, "
              f"{navigation_fallback} navigation fallback, "
              f"{irrelevance_fallback} irrelevance fallback, "
              f"{len(references) - enriched - navigation_fallback - irrelevance_fallback} kept SERP snippets")
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

    def _google_search_fallback(self, query: str) -> Dict:
        """
        Fallback: regular Google Search via SerpAPI (engine=google).
        Parses organic_results instead of references/text_blocks.
        Used when google_ai_mode is temporarily unavailable.
        """
        from serpapi import GoogleSearch

        params = {
            'engine': 'google',
            'q': query,
            'hl': 'en',
            'gl': 'lk',
            'num': 10,
            'api_key': self.serp_api_key,
        }

        search = GoogleSearch(params)
        raw = search.get_dict()

        if 'error' in raw:
            print(f"  Google Search fallback error: {raw['error']}")
            return {'references': [], 'synthesis': '', 'raw_response': raw}

        references = []
        for i, result in enumerate(raw.get('organic_results', [])):
            url = result.get('link', '')
            snippet = result.get('snippet', '')
            references.append({
                'title':       result.get('title', ''),
                'snippet':     snippet,
                'link':        url,
                'source':      self._extract_domain(url),
                'date':        result.get('date', '') or self._parse_date_from_snippet(snippet),
                'date_raw':    snippet[:50],
                'position':    i,
                'search_type': 'google_search',
            })

        print(f"  Google Search fallback: {len(references)} organic results retrieved")
        return {'references': references, 'synthesis': '', 'raw_response': raw}

    def _google_ai_mode_query(self, query: str) -> Dict:
        """
        Execute a single Google AI Mode search via SerpAPI.
        Falls back to regular Google Search if AI Mode is unavailable.

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

            # If AI Mode returned an error, fall back to regular Google Search
            if 'error' in raw or not raw.get('references') and not raw.get('text_blocks'):
                if 'error' in raw:
                    print(f"  Google AI Mode unavailable ({raw['error']}) — falling back to Google Search")
                else:
                    print(f"  Google AI Mode returned no content — falling back to Google Search")
                return self._google_search_fallback(query)

            # Collect which reference indexes Google's synthesis actually cited
            cited_indexes: set = set()
            for block in raw.get('text_blocks', []):
                for idx in block.get('reference_indexes', []):
                    cited_indexes.add(idx)
                for item in block.get('list', []):
                    for idx in item.get('reference_indexes', []):
                        cited_indexes.add(idx)

            # Build normalised evidence list — only keep cited references
            references = []
            for ref in raw.get('references', []):
                ref_index = ref.get('index', -1)
                if cited_indexes and ref_index not in cited_indexes:
                    continue  # Google didn't use this source in its synthesis
                url = ref.get('link', '')
                references.append({
                    'title':       ref.get('title', ''),
                    'snippet':     ref.get('snippet', ''),
                    'link':        url,
                    'source':      self._extract_domain(url),
                    'date':        ref.get('date', '') or self._parse_date_from_snippet(ref.get('snippet', '')),
                    'date_raw':    ref.get('snippet', '')[:50],
                    'position':    ref.get('index', 0),
                    'search_type': 'google_ai_mode',
                })

            total_refs = len(raw.get('references', []))
            if cited_indexes and len(references) < total_refs:
                print(f"  Google AI Mode: {len(references)}/{total_refs} references kept "
                      f"(only sources cited in synthesis)")

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

            return {
                'references': references,
                'synthesis': synthesis,
                'raw_response': raw,
            }

        except Exception as e:
            print(f"  Google AI Mode search error: {e}")
            print(f"  Falling back to Google Search...")
            try:
                return self._google_search_fallback(query)
            except Exception as e2:
                print(f"  Google Search fallback also failed: {e2}")
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
            from src.retrieval.qdrant_hybrid_retriever import QdrantHybridRetriever as HybridRetriever
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
        self, claim: str, num_results: int = 20, source_callback=None
    ) -> List[Dict]:
        """
        Retrieve evidence using a single Google AI Mode query.

        Stores Google's synthesis on the first result as `google_synthesis`
        so the pipeline can surface it alongside the verdict.

        source_callback: optional callable(slim_ref) fired for each reference
        immediately after the Serper API returns — before content enrichment.
        Callers can use this to stream sources to the UI while enrichment runs.
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

        # Fire callback immediately — before enrichment (~1-2 s after API call).
        # This lets callers stream each source to the UI while enrichment runs.
        if source_callback:
            for ref in references[:num_results]:
                source_callback({
                    "title":         ref.get("title", ""),
                    "link":          ref.get("link", ""),
                    "source":        ref.get("source", "") or urlparse(ref.get("link", "")).netloc,
                    "evidence_type": "web",
                    "snippet":       (ref.get("snippet", "") or "")[:300],
                    "date":          ref.get("date", "") or ref.get("date_raw", ""),
                })

        # Fetch full article content with hybrid relevance check
        references = self._enrich_with_full_content(references[:num_results], claim=claim)

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
        self, claim: str, num_results: int = 20, results_per_query: int = 8,
        source_callback=None
    ) -> List[Dict]:
        """
        Pipeline-compatible alias.  Google AI Mode doesn't need query
        decomposition — a single query returns curated references.
        The num_results / results_per_query params are accepted for
        compatibility but only num_results is used as an upper bound.
        """
        return self.retrieve_google_ai_mode(claim, num_results=num_results,
                                            source_callback=source_callback)

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
