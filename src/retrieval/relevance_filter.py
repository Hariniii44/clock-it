"""
CrossEncoderRelevanceFilter
---------------------------
Gates the evidence pool between retrieval and NLI verification.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 — a ~80 MB model trained on
Microsoft MARCO passage retrieval — to score how relevant each retrieved
source snippet is to the claim being verified.

Sources below the threshold are dropped before NLI, which:
  - Removes noise from keyword-matched-but-irrelevant sources
  - Reduces the NLI batch size (faster verification)
  - Prevents irrelevant NEUTRAL votes from diluting the weighted verdict

The raw relevance score is also stored on each evidence dict as
``cross_encoder_relevance`` so EvidenceWeighter can use it as a
multiplicative factor in the final weight formula.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict


class CrossEncoderRelevanceFilter:
    """
    Scores and filters evidence sources by relevance to a claim.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID for the cross-encoder.
    threshold  : float
        Minimum relevance score (0–1) to keep a source.
        Sources below this are dropped before NLI.
    max_snippet_tokens : int
        Truncate snippets to this many characters before scoring
        (cross-encoders have a 512-token limit).
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        threshold: float = 0.4,
        max_snippet_chars: int = 512,
    ):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        self.threshold = threshold
        self.max_snippet_chars = max_snippet_chars
        print(f"  CrossEncoderRelevanceFilter loaded ({model_name})")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Convert raw logits → probabilities in [0, 1]."""
        return 1.0 / (1.0 + np.exp(-x))

    def _snippet(self, ev: Dict) -> str:
        """Extract and truncate the text to score."""
        text = ev.get("snippet", "") or ev.get("passage", "") or ev.get("text", "")
        return text[: self.max_snippet_chars]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_all(self, claim: str, sources: List[Dict]) -> List[Dict]:
        """
        Score every source for relevance to *claim*.

        Adds ``cross_encoder_relevance`` (float 0–1) to each source dict.
        Sources are NOT filtered — call ``filter_relevant`` to drop low scorers.
        """
        if not sources:
            return sources

        pairs = [(claim, self._snippet(ev)) for ev in sources]
        raw_scores = self.model.predict(pairs)                    # numpy array of logits
        relevance_scores = self._sigmoid(np.array(raw_scores))   # → [0, 1]

        for ev, score in zip(sources, relevance_scores):
            ev["cross_encoder_relevance"] = float(score)

        return sources

    def filter_relevant(self, claim: str, sources: List[Dict]) -> List[Dict]:
        """
        Score all sources and return only those at or above ``self.threshold``.

        Always keeps at least 3 sources so the pipeline is never starved,
        even if all scores fall below the threshold (e.g. very niche claims).
        """
        scored = self.score_all(claim, sources)

        passing   = [s for s in scored if s["cross_encoder_relevance"] >= self.threshold]
        filtered  = [s for s in scored if s["cross_encoder_relevance"] <  self.threshold]

        # Safety floor — never drop everything
        if len(passing) < 3 and scored:
            # Take the top-3 by score even if below threshold
            scored_sorted = sorted(scored, key=lambda s: s["cross_encoder_relevance"], reverse=True)
            passing = scored_sorted[:3]
            filtered = scored_sorted[3:]

        if filtered:
            dropped_info = ", ".join(
                f"{s.get('source', '?')} ({s['cross_encoder_relevance']:.2f})"
                for s in sorted(filtered, key=lambda s: s["cross_encoder_relevance"], reverse=True)[:5]
            )
            print(
                f"  [RelevanceFilter] {len(filtered)} source(s) dropped (below {self.threshold}): {dropped_info}"
            )

        print(
            f"  [RelevanceFilter] {len(passing)}/{len(scored)} sources passed "
            f"(threshold={self.threshold})"
        )
        return passing
