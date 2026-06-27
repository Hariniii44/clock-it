from typing import TypedDict, List, Optional


class SubClaim(TypedDict):
    text: str
    type: str           # "number" | "category" | "geography" | "attribution" | "event"
    search_query: str   # targeted query for this specific sub-claim


class RawSource(TypedDict):
    url: str
    title: str
    snippet: str
    source_type: str    # "web" | "db"
    authority_tier: str


class InvestigatedSource(TypedDict):
    url: str
    title: str
    full_text: str
    what_claim_does_source_make: str
    is_original_or_citing: str      # "original" | "citing"
    citation_url: Optional[str]
    citation_name: Optional[str]    # e.g. "UN Panel of Experts 2011"
    precision_level: str            # "exact_count" | "estimate" | "upper_bound" | "vague_magnitude" | "recognition" | "no_number"
    precision_match: bool
    nli_label: str                  # "SUPPORTED" | "REFUTED" | "NEUTRAL"
    confidence: float
    reasoning: str
    depth: int                      # 0 = directly retrieved, 1 = followed one citation, 2 = followed two


class SubClaimResult(TypedDict):
    sub_claim: SubClaim
    all_sources: List[InvestigatedSource]
    independent_sources: List[InvestigatedSource]
    verdict: str        # "SUPPORTED" | "PARTIALLY_SUPPORTED" | "NOT_VERIFIED" | "REFUTED" | "UNCERTAIN"
    reasoning: str
    precision_gap: bool  # True if the claim is more precise than any source supports


class FactCheckState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    claim: str

    # ── Built up during investigation ──────────────────────────────────────
    sub_claims: List[SubClaim]
    current_sub_claim_index: int
    current_raw_sources: List[RawSource]
    current_investigated: List[InvestigatedSource]
    current_independent: List[InvestigatedSource]
    retry_count: int

    # ── Completed results (one entry per sub-claim) ────────────────────────
    sub_claim_results: List[SubClaimResult]

    # ── Final output ────────────────────────────────────────────────────────
    final_verdict: str
    final_confidence: float
    final_explanation: str
    precision_issues: List[str]
    geography_issues: List[str]
