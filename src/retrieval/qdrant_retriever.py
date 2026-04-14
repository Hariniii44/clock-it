# """
# Standalone Qdrant retriever — for testing before pipeline integration.

# Run directly to test:
#     python src/retrieval/qdrant_retriever.py
#     python src/retrieval/qdrant_retriever.py --claim "Sri Lanka economy contracted 2020"
#     python src/retrieval/qdrant_retriever.py --list
# """
# import sys
# import argparse
# sys.path.append('.')

# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer
# from typing import List, Dict

# from config import QDRANT_URL, EMBEDDING_MODEL, DATASETS_CONFIG, TOP_K_PER_DATASET, RELEVANCE_THRESHOLD


# class QdrantRetriever:
#     """Simple Qdrant-based retriever for testing."""

#     def __init__(self):
#         print(f"Connecting to Qdrant at {QDRANT_URL} ...")
#         self.client = QdrantClient(url=QDRANT_URL)

#         print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
#         self.model = SentenceTransformer(EMBEDDING_MODEL)

#         # Fetch available collections
#         self.collections = {c.name for c in self.client.get_collections().collections}
#         print(f"Available collections: {sorted(self.collections)}")

#     def search(self, claim: str, top_k: int = None, collections: List[str] = None) -> List[Dict]:
#         """
#         Search Qdrant for chunks relevant to the claim.

#         Args:
#             claim:       The claim text to search for
#             top_k:       Results per collection (default: TOP_K_PER_DATASET from config)
#             collections: Which collections to search (default: all available)

#         Returns:
#             List of result dicts sorted by similarity score
#         """
#         if top_k is None:
#             top_k = TOP_K_PER_DATASET

#         if collections is None:
#             collections = sorted(self.collections)

#         # Only search collections that exist in Qdrant
#         searchable = [c for c in collections if c in self.collections]
#         if not searchable:
#             print("No matching collections found in Qdrant.")
#             return []

#         # Encode the claim into a vector
#         query_vector = self.model.encode(claim, normalize_embeddings=True).tolist()

#         all_results = []

#         for collection_name in searchable:
#             hits = self.client.query_points(
#                 collection_name=collection_name,
#                 query=query_vector,
#                 score_threshold=RELEVANCE_THRESHOLD,
#                 limit=top_k,
#                 with_payload=True,
#             ).points

#             for hit in hits:
#                 p = hit.payload
#                 config = DATASETS_CONFIG.get(collection_name, {})
#                 all_results.append({
#                     'collection':       collection_name,
#                     'score':            round(float(hit.score), 4),
#                     'authority':        config.get('authority', 0.5),
#                     'text':             p.get('text', ''),
#                     'title':            p.get('title', ''),
#                     'date':             p.get('date', ''),
#                     'url':              p.get('url', ''),
#                     'description':      config.get('description', ''),
#                 })

#         # Sort by score descending
#         all_results.sort(key=lambda x: x['score'], reverse=True)
#         return all_results

#     def list_collections(self):
#         """Print all Qdrant collections with their point counts."""
#         collections = self.client.get_collections().collections
#         if not collections:
#             print("No collections found. Run scripts/build_qdrant_index.py first.")
#             return
#         print(f"\n{'Collection':<40} {'Points':>10}  {'Authority':>10}")
#         print("-" * 64)
#         for c in sorted(collections, key=lambda x: x.name):
#             info   = self.client.get_collection(c.name)
#             config = DATASETS_CONFIG.get(c.name, {})
#             auth   = config.get('authority', '?')
#             print(f"{c.name:<40} {info.points_count:>10,}  {str(auth):>10}")


# def main():
#     parser = argparse.ArgumentParser(description='Test Qdrant retrieval')
#     parser.add_argument('--claim', type=str, default="Sri Lanka's economy contracted by 3.6% in 2020",
#                         help='Claim to search for')
#     parser.add_argument('--top-k', type=int, default=3,
#                         help='Results per collection')
#     parser.add_argument('--list', action='store_true',
#                         help='List available collections and exit')
#     args = parser.parse_args()

#     retriever = QdrantRetriever()

#     if args.list:
#         retriever.list_collections()
#         return

#     print(f"\nSearching for: '{args.claim}'")
#     print("=" * 70)

#     results = retriever.search(args.claim, top_k=args.top_k)

#     if not results:
#         print("No results found.")
#         return

#     print(f"Found {len(results)} results:\n")
#     for i, r in enumerate(results, 1):
#         print(f"[{i}] Collection : {r['collection']} ({r['description']})")
#         print(f"    Score     : {r['score']}  |  Authority: {r['authority']}")
#         print(f"    Title     : {r['title'][:80]}")
#         print(f"    Date      : {r['date']}")
#         print(f"    Text      : {r['text'][:300]}...")
#         print()


# if __name__ == "__main__":
#     main()
