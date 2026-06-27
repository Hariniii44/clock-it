from ..state import FactCheckState


def should_follow_citation(state: FactCheckState) -> str:
    """After investigate_source: route to follow_citation if any source cites another and has a URL."""
    needs_following = any(
        s.get("is_original_or_citing") == "citing"
        and s.get("citation_url")
        and s.get("depth", 0) < 2
        for s in state.get("current_investigated", [])
    )
    return "follow_citation" if needs_following else "dedup_by_origin"


def has_enough_independent_sources(state: FactCheckState) -> str:
    """After dedup_by_origin: retry retrieval if fewer than 2 independent sources (max 2 retries)."""
    independent = state.get("current_independent", [])
    retry_count = state.get("retry_count", 0)

    if len(independent) >= 2 or retry_count >= 3:
        return "assess_sub_claim"
    return "retrieve_evidence"


def more_sub_claims_remaining(state: FactCheckState) -> str:
    """After assess_sub_claim: loop to next sub-claim or synthesise if all done."""
    next_index = state.get("current_sub_claim_index", 0)
    total = len(state.get("sub_claims", []))

    if next_index < total:
        return "retrieve_evidence"
    return "synthesise_verdict"
