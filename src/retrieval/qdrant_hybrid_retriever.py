"""
Qdrant-backed hybrid retriever — drop-in replacement for HybridRetriever.
Returns results in the same dict format the pipeline expects.
"""
import sys
sys.path.append('.')

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict

from config import (
    QDRANT_URL, QDRANT_API_KEY, EMBEDDING_MODEL, DATASETS_CONFIG,
    TOP_K_PER_DATASET, RELEVANCE_THRESHOLD,
    get_datasets_for_claim_type,
    USE_HYBRID_RETRIEVAL, HYBRID_FALLBACK_TO_SERPER,
)


class QdrantHybridRetriever:
    """
    Hybrid retriever using Qdrant for local dataset search.
    Identical interface to HybridRetriever — swap with one import change.
    """

    def __init__(self):
        print(f"Connecting to Qdrant at {QDRANT_URL} ...")
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.available_collections = {
            c.name for c in self.client.get_collections().collections
        }
        print(f"   {len(self.available_collections)} collections available: "
              f"{sorted(self.available_collections)}")

        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"QdrantHybridRetriever ready")

    # ------------------------------------------------------------------
    # Public API (matches HybridRetriever)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        claim_types: List[str] = None,
        total_results: int = 20,
        use_query_expansion: bool = True,
    ) -> List[Dict]:
        """
        Search Qdrant collections + optional web fallback.
        Returns list of result dicts compatible with core_pipeline.py.
        """
        print(f"\n{'='*70}")
        print(f"QDRANT HYBRID SEARCH")
        print(f"{'='*70}")
        print(f"Query: '{query}'")

        all_results = []

        # 1. Local Qdrant search
        if USE_HYBRID_RETRIEVAL and self.available_collections:
            queries = self._query_variations(query) if use_query_expansion else [query]
            print(f"Query variations: {queries}")

            seen_ids = set()
            for q in queries:
                results = self._search_qdrant(q, claim_types, top_k=TOP_K_PER_DATASET)
                for r in results:
                    # deduplicate across query variations by (collection, point_id)
                    uid = (r['dataset'], r['_point_id'])
                    if uid not in seen_ids:
                        seen_ids.add(uid)
                        all_results.append(r)

            print(f"Qdrant results (before web): {len(all_results)}")

        # 2. Web fallback if needed
        if len(all_results) < total_results and HYBRID_FALLBACK_TO_SERPER:
            needed = total_results - len(all_results)
            print(f"Fetching {needed} web results to fill gap...")
            web_results = self._search_web(query, num_results=needed)
            all_results.extend(web_results)

        # 3. Deduplicate by URL / title
        all_results = self._deduplicate(all_results)

        # 4. Sort: authority desc, then similarity_score desc
        all_results.sort(
            key=lambda x: (x['authority'], x['similarity_score']),
            reverse=True
        )

        # 5. Trim and add rank
        final = all_results[:total_results]
        for i, r in enumerate(final):
            r['rank'] = i + 1
            r.pop('_point_id', None)   # remove internal field

        print(f"Final results returned: {len(final)}")
        return final

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_qdrant(
        self,
        query: str,
        claim_types: List[str],
        top_k: int,
    ) -> List[Dict]:
        """Query Qdrant collections and return pipeline-compatible dicts."""

        # Decide which collections to search
        if claim_types:
            relevant = get_datasets_for_claim_type(claim_types)
        else:
            relevant = list(self.available_collections)

        searchable = [c for c in relevant if c in self.available_collections]
        if not searchable:
            print("No matching collections in Qdrant.")
            return []

        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()

        results = []
        for collection in searchable:
            hits = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                score_threshold=RELEVANCE_THRESHOLD,
                limit=top_k,
                with_payload=True,
            ).points

            cfg = DATASETS_CONFIG.get(collection, {})

            for hit in hits:
                p = hit.payload
                text = p.get('text', '')
                results.append({
                    # Keys expected by core_pipeline.py
                    'source':           f"{collection} ({cfg.get('description', '')})",
                    'title':            p.get('title', ''),
                    'text':             text,
                    'passage':          text,   # already a chunk, no further extraction needed
                    'link':             p.get('url', ''),
                    'url':              p.get('url', ''),
                    'date':             p.get('date', ''),
                    'dataset':          collection,
                    'dataset_source':   collection,
                    'authority':        cfg.get('authority', 0.5),
                    'similarity_score': round(float(hit.score), 4),
                    'retrieval_method': 'qdrant',
                    # Internal — removed before returning
                    '_point_id':        hit.id,
                })

        return results

    def _search_web(self, query: str, num_results: int) -> List[Dict]:
        """Web search fallback — mirrors HybridRetriever.search_web()."""
        try:
            from src.evidence_retrieval.evidence_retrieval import retrieve_evidence
            web_results = retrieve_evidence(query, num_sources=num_results)
            formatted = []
            for r in web_results:
                text = r.get('snippet', r.get('content', ''))
                formatted.append({
                    'source':           r.get('source', 'Web Search'),
                    'title':            r.get('title', ''),
                    'text':             text,
                    'passage':          text,
                    'link':             r.get('url', r.get('link', '')),
                    'url':              r.get('url', r.get('link', '')),
                    'date':             r.get('date', ''),
                    'dataset':          'web_search',
                    'dataset_source':   'web_search',
                    'authority':        0.5,
                    'similarity_score': 0.8,
                    'retrieval_method': 'web_search',
                    '_point_id':        None,
                })
            return formatted
        except Exception as e:
            print(f"Web search unavailable: {e}")
            return []

    def _query_variations(self, query: str, max_variations: int = 3) -> List[str]:
        """Generate simple query variations to improve recall."""
        synonyms = {
            'said':       'stated',
            'president':  'presidential',
            'government': 'administration',
            'shortage':   'scarcity',
            'allocated':  'approved',
            'parliament': 'parliamentary',
        }
        variations = [query]
        q = query.lower()
        for original, replacement in synonyms.items():
            if original in q and len(variations) < max_variations:
                v = q.replace(original, replacement)
                if v not in variations:
                    variations.append(v)
        return variations

    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results by URL and title."""
        seen_urls, seen_titles, unique = set(), set(), []
        for r in results:
            url   = r.get('url', '').strip()
            title = r.get('title', '').strip().lower()
            if url and url in seen_urls:
                continue
            if title and title in seen_titles:
                continue
            unique.append(r)
            if url:   seen_urls.add(url)
            if title: seen_titles.add(title)
        return unique
