import asyncio
from typing import Optional

from ..state import FactCheckState, InvestigatedSource
from .investigate import _fetch_page_text, _analyse_source


async def follow_citation(state: FactCheckState) -> dict:
    idx = state["current_sub_claim_index"]
    sub_claims = state.get("sub_claims", [])
    investigated = state.get("current_investigated", [])

    if not sub_claims or idx >= len(sub_claims):
        return {"current_investigated": investigated}

    sub_claim = sub_claims[idx]

    # Collect citation URLs to follow:
    # - source must be "citing"
    # - must have a citation_url
    # - depth must be < 2 (don't follow beyond two hops)
    # - URL must not already be in investigated (avoid re-fetching)
    already_fetched = {s["url"].rstrip("/").split("?")[0] for s in investigated}

    to_follow: list[tuple[str, Optional[str], int]] = []  # (url, citation_name, depth)
    seen_citation_urls: set[str] = set()

    for source in investigated:
        if (
            source.get("is_original_or_citing") == "citing"
            and source.get("citation_url")
            and source.get("depth", 0) < 2
        ):
            citation_url = source["citation_url"].rstrip("/").split("?")[0]
            if citation_url not in already_fetched and citation_url not in seen_citation_urls:
                seen_citation_urls.add(citation_url)
                to_follow.append((
                    source["citation_url"],
                    source.get("citation_name"),
                    source.get("depth", 0) + 1,
                ))

    if not to_follow:
        return {"current_investigated": investigated}

    print(f"\n[follow_citation] following {len(to_follow)} citation(s) for '{sub_claim['text'][:50]}'")
    for url, name, depth in to_follow:
        print(f"  depth={depth} name='{name}' url={url[:70]}")

    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(4)

    async def _fetch_one(
        url: str, name: Optional[str], depth: int
    ) -> Optional[InvestigatedSource]:
        async with semaphore:
            full_text = await loop.run_in_executor(None, _fetch_page_text, url)

            # Build a minimal RawSource so _analyse_source can use it
            from ..state import RawSource
            raw = RawSource(
                url=url,
                title=name or "",
                snippet="",
                source_type="web",
                authority_tier="",
            )
            result = await loop.run_in_executor(
                None, _analyse_source,
                sub_claim["text"], sub_claim["type"], raw, full_text,
            )
            if result is not None:
                # Overwrite depth — _analyse_source always sets depth=0
                result = InvestigatedSource(**{**result, "depth": depth})
                print(f"  [follow_citation] [{result['nli_label']} {result['confidence']:.2f}] "
                      f"[{result['is_original_or_citing']}] [{result['precision_level']}] {url[:60]}")
            return result

    followed = await asyncio.gather(
        *[_fetch_one(url, name, depth) for url, name, depth in to_follow],
        return_exceptions=False,
    )

    additional = [r for r in followed if r is not None]
    print(f"  [follow_citation] added {len(additional)} primary source(s)")

    return {"current_investigated": investigated + additional}
