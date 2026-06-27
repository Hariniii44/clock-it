import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

from ..state import FactCheckState, RawSource

# ---------------------------------------------------------------------------
# Lazy singleton retrievers — initialized once on first call, reused after.
# Avoids reloading SentenceTransformer and spaCy on every node invocation.
# ---------------------------------------------------------------------------
_db_retriever = None
_web_retriever = None
_init_lock = threading.Lock()  # threading.Lock for use inside run_in_executor


def _init_retrievers():
    global _db_retriever, _web_retriever
    with _init_lock:
        if _db_retriever is None:
            try:
                from src.retrieval.qdrant_hybrid_retriever import QdrantHybridRetriever
                _db_retriever = QdrantHybridRetriever()
                print("  [retrieve] QdrantHybridRetriever ready")
            except Exception as e:
                print(f"  [retrieve] Qdrant unavailable: {e}")
                _db_retriever = False  # sentinel: don't retry init

        if _web_retriever is None:
            try:
                from src.evidence_retrieval.google_ai_mode_retrieval import GoogleAIModeRetriever
                from config import Config
                _web_retriever = GoogleAIModeRetriever(
                    serp_api_key=Config.SERP_API_KEY,
                    groq_key=Config.GROQ_API_KEY,
                    tavily_key=Config.TAVILY_API_KEY,
                )
                print("  [retrieve] GoogleAIModeRetriever ready")
            except Exception as e:
                print(f"  [retrieve] Web retriever unavailable: {e}")
                _web_retriever = False


def _run_retrieval(query: str) -> List[RawSource]:
    """Synchronous retrieval — called inside run_in_executor."""
    sources: List[RawSource] = []

    # DB retrieval
    if _db_retriever and _db_retriever is not False:
        try:
            results = _db_retriever.hybrid_search(
                query=query,
                claim_types=None,
                total_results=5,
                use_query_expansion=False,  # query is already targeted
            )
            for r in results:
                sources.append(RawSource(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=r.get("passage") or r.get("text", ""),
                    source_type="db",
                    authority_tier=str(r.get("authority", "")),
                ))
        except Exception as e:
            print(f"  [retrieve] DB retrieval error: {e}")

    # Web retrieval
    if _web_retriever and _web_retriever is not False:
        try:
            results = _web_retriever.retrieve_hybrid_serper_decomposition(
                query, num_results=8, results_per_query=8
            )
            for r in results:
                url = r.get("link") or r.get("url", "")
                if not url:
                    continue
                sources.append(RawSource(
                    url=url,
                    title=r.get("title", ""),
                    snippet=r.get("snippet") or r.get("content", ""),
                    source_type="web",
                    authority_tier=r.get("source", ""),
                ))
        except Exception as e:
            print(f"  [retrieve] Web retrieval error: {e}")

    # Deduplicate by URL
    seen = set()
    deduped = []
    for s in sources:
        key = s["url"].rstrip("/").split("?")[0]
        if key and key not in seen:
            seen.add(key)
            deduped.append(s)

    return deduped


async def retrieve_evidence(state: FactCheckState) -> dict:
    idx = state["current_sub_claim_index"]
    sub_claims = state.get("sub_claims", [])
    retry_count = state.get("retry_count", 0)

    if not sub_claims or idx >= len(sub_claims):
        return {"current_raw_sources": [], "current_investigated": [], "retry_count": retry_count + 1}

    base_query = sub_claims[idx]["search_query"]

    # On retries, broaden the query to find primary sources the first pass missed
    if retry_count == 1:
        query = f"{base_query} primary source report"
    elif retry_count >= 2:
        query = f"{base_query} original study data"
    else:
        query = base_query

    print(f"\n[retrieve_evidence] sub_claim={idx} retry={retry_count} query='{query}'")

    # Lazy-init retrievers (thread-safe via threading.Lock)
    if _db_retriever is None or _web_retriever is None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _init_retrievers)

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=2) as pool:
        sources = await loop.run_in_executor(pool, _run_retrieval, query)

    print(f"  [retrieve_evidence] {len(sources)} sources ({sum(1 for s in sources if s['source_type']=='db')} db, {sum(1 for s in sources if s['source_type']=='web')} web)")

    return {
        "current_raw_sources": sources,
        "current_investigated": [],
        "retry_count": retry_count + 1,
    }
