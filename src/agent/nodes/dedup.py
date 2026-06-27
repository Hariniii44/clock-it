from ..state import FactCheckState, InvestigatedSource


def _normalise_url(url: str) -> str:
    return url.rstrip("/").split("?")[0].split("#")[0].lower()


def dedup_by_origin(state: FactCheckState) -> dict:
    """
    Group investigated sources by their ultimate origin.
    Sources that all trace back to the same root are not independent.

    Origin rules:
    - "original" sources: their own URL is the origin key
    - "citing" sources: their citation_url is the origin key (they share the
      origin of what they cite); if no citation_url, fall back to their own URL
    """
    sources = state.get("current_investigated", [])

    if not sources:
        return {"current_independent": []}

    seen_origins: set[str] = set()
    independent: list[InvestigatedSource] = []

    # Prefer original sources over citing ones — sort so originals come first
    sorted_sources = sorted(
        sources,
        key=lambda s: (0 if s.get("is_original_or_citing") == "original" else 1, s.get("depth", 0)),
    )

    for source in sorted_sources:
        if source.get("is_original_or_citing") == "original":
            origin = _normalise_url(source["url"])
        else:
            citation_url = source.get("citation_url") or source["url"]
            origin = _normalise_url(citation_url)

        if origin not in seen_origins:
            seen_origins.add(origin)
            independent.append(source)

    citing_count = sum(1 for s in sources if s.get("is_original_or_citing") == "citing")
    collapsed = len(sources) - len(independent)

    print(f"\n[dedup_by_origin] {len(sources)} sources -> {len(independent)} independent "
          f"({collapsed} collapsed, {citing_count} were citing)")
    for s in independent:
        tag = "ORIG" if s.get("is_original_or_citing") == "original" else "CITE"
        print(f"  [{tag} depth={s.get('depth',0)}] [{s['nli_label']}] {s['url'][:65]}")

    return {"current_independent": independent}
