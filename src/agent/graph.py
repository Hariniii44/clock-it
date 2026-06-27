from langgraph.graph import StateGraph, END

from .state import FactCheckState
from .nodes.decompose import decompose_claim
from .nodes.retrieve import retrieve_evidence
from .nodes.investigate import investigate_source
from .nodes.follow_citation import follow_citation
from .nodes.dedup import dedup_by_origin
from .nodes.assess import assess_sub_claim
from .nodes.synthesise import synthesise_verdict
from .edges.routing import (
    should_follow_citation,
    has_enough_independent_sources,
    more_sub_claims_remaining,
)


def build_graph():
    graph = StateGraph(FactCheckState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    graph.add_node("decompose_claim", decompose_claim)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("investigate_source", investigate_source)
    graph.add_node("follow_citation", follow_citation)
    graph.add_node("dedup_by_origin", dedup_by_origin)
    graph.add_node("assess_sub_claim", assess_sub_claim)
    graph.add_node("synthesise_verdict", synthesise_verdict)

    # ── Entry point ─────────────────────────────────────────────────────────
    graph.set_entry_point("decompose_claim")

    # ── Fixed edges ─────────────────────────────────────────────────────────
    graph.add_edge("decompose_claim", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "investigate_source")
    graph.add_edge("follow_citation", "dedup_by_origin")
    graph.add_edge("synthesise_verdict", END)

    # ── Conditional edges ───────────────────────────────────────────────────
    graph.add_conditional_edges(
        "investigate_source",
        should_follow_citation,
        {
            "follow_citation": "follow_citation",
            "dedup_by_origin": "dedup_by_origin",
        },
    )
    graph.add_conditional_edges(
        "dedup_by_origin",
        has_enough_independent_sources,
        {
            "retrieve_evidence": "retrieve_evidence",
            "assess_sub_claim": "assess_sub_claim",
        },
    )
    graph.add_conditional_edges(
        "assess_sub_claim",
        more_sub_claims_remaining,
        {
            "retrieve_evidence": "retrieve_evidence",
            "synthesise_verdict": "synthesise_verdict",
        },
    )

    return graph.compile()


# Compiled once at import time — reused across all requests
investigative_agent = build_graph()
