# Context-Adaptive Evidence Weighting Framework
### Architecture & Pipeline Documentation

Sri Lankan political fact-checking system. Core research contribution: a **bias-aware evidence weighting algorithm** that dynamically adjusts source credibility based on how a source's political framing aligns with the claim being checked.

---

## System Overview

```
User Claim (text)
      │
      ▼
┌─────────────────────────────────────────────────┐
│                  FastAPI  (api.py)               │
│                                                 │
│  POST /verify        — blocking response        │
│  POST /verify/stream — Server-Sent Events       │
│  POST /explain       — blocking Gemini narration│
│  POST /explain/stream— streaming Gemini         │
│  GET  /health                                   │
└─────────────────────────────────────────────────┘
      │
      ▼  _pipeline_steps()  ← single async generator
      │     driving all steps below
```

All heavy models are loaded **once at startup** into a global `_models` dict and reused across every request. There is no per-request model loading.

---

## Pipeline Steps (in order)

### Step 0 — Temporal Claim Analysis
*File: `core_pipeline.py`*

Fast, no I/O. Runs before any retrieval.

- `detect_temporal_claim(claim)` — classifies whether the claim is time-bound (e.g., "in 2020", "last week") and estimates the claimed time period.
- `analyze_content_freshness()` / `calculate_temporal_confidence()` — used later to apply a confidence multiplier if temporal context is uncertain.
- If a majority of retrieved evidence URLs contain recent dates, recency can be **inferred** even when the claim text has no explicit time markers.

---

### Step 1 — Dual Evidence Retrieval (parallel)
*Runs DB and web concurrently via `ThreadPoolExecutor(max_workers=2)`*

#### 1a. Database Retrieval — `QdrantHybridRetriever`
`src/retrieval/qdrant_hybrid_retriever.py`

- Connects to a local **Qdrant** vector database containing ~140k Sri Lankan government documents.
- Uses `SentenceTransformer` (configured in `config.py` → `EMBEDDING_MODEL`) to embed the claim.
- Performs **hybrid search** (semantic + keyword) across configured dataset collections.
- Returns up to 10 passages with source metadata, authority tier, and similarity score.

Qdrant collections (from `config.py` → `DATASETS_CONFIG`):

| Collection | Content |
|---|---|
| `hansard_2020s` | Parliamentary debates 2020s (53k docs) |
| `central_bank_reports` | Central Bank annual reports (50.8k docs) |
| `acts` | Sri Lankan Acts and Laws (23.5k docs) |
| `cabinet_decisions` | Cabinet decisions (10.5k docs) |
| `pmd_press_releases` | Presidential press releases (3.38k docs) |

#### 1b. Web Retrieval — `GoogleAIModeRetriever` (primary) / `TavilyEvidenceRetriever` (fallback)
`src/evidence_retrieval/google_ai_mode_retrieval.py` / `src/evidence_retrieval/tavily_evidence_retrieval.py`

Selection logic in `api.py`: uses `GoogleAIModeRetriever` if `Config.SERP_API_KEY` is set, otherwise falls back to Tavily.

**GoogleAIModeRetriever** (current primary):
- Calls SerpAPI's Google AI Mode engine — 1 API call per claim.
- Google's AI pre-selects ~7 temporally relevant, authoritative references.
- Also returns Google's own synthesised answer (`google_synthesis`) for evaluation baseline comparison.
- Uses `ClaimDecomposer` (Groq) to optionally decompose complex claims into sub-queries.
- **Fallback**: if SerpAPI returns no results, automatically retries via Google Search engine (`engine=google`) before giving up.

**TavilyEvidenceRetriever** (fallback):
- 5 × decomposed queries via Tavily Search API.
- More sources (~15) but lower average temporal relevance.

Both retrievers tag results with `evidence_type = "web"` and use the same tiered source priority system.

#### Post-Retrieval Filters (Step 1 continued)

1. **Non-Latin filter** — drops sources where <60% of characters are ASCII (Sinhala/Tamil UI text scraped from YouTube).
2. **Stale-source filter** — if the claim has explicit temporal markers AND recent sources outnumber stale ones, sources dated before the claim's time window are dropped.
3. **CrossEncoder Stage 1 filter** — `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranks DB sources. Keeps top 3–5 above score threshold 1.0; web sources are exempt.

---

### Step 2+3 — NLI Verification + Bias Analysis (parallel)
*Runs two tasks concurrently via `ThreadPoolExecutor(max_workers=2)`*

#### Task A — NLI Verification: `GeminiNLIVerifier` (primary)
`src/verification/gemini_nli_verification.py`
Model: **Google Gemini** (model fallback chain: `gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-1.5-flash`)

- Single batched API call — sends **all** evidence sources in one prompt with `response_mime_type="application/json"` to enforce valid JSON output.
- Returns per-source: `label` (SUPPORTED / REFUTED / NEUTRAL / **IRRELEVANT**), `confidence` (0–1), `reason`.
  - `relevant` (bool) is derived programmatically from the label (`IRRELEVANT` → `relevant=False`); it is no longer a separate model output.
- Four-label classification: SUPPORTED and REFUTED require verified factual evidence; NEUTRAL means the source engages with the claim's question but is inconclusive; **IRRELEVANT** means the source addresses a different question than the claim, even within the same domain (e.g. a source explaining how permits are issued when the claim asks about enforcement).
- Key rules encoded in system prompt: accusations/allegations always NEUTRAL; superlative claims (first/only/never) require explicit source confirmation; historical facts not undone by later events; NEUTRAL vs IRRELEVANT decided by whether the source could help confirm or refute the specific claim.
- Model fallback with exponential backoff: distinguishes daily quota exhaustion (skip model immediately), rate limits (backoff retry), and connection drops (short retry).

Fallback: `ClaimVerifier` (`src/verification/verification.py`) — local **DeBERTa-v3-base-mnli**, runs per-source if all Gemini models fail.

#### Stage 2 Relevance Filter (after NLI)
Sources labelled **IRRELEVANT** by the NLI verifier are removed from the evidence pool before weighting. This replaces the previous `relevant=False` boolean flag approach.

#### Task B — Claim-Level Bias Analysis: `ClaimBiasAnalyzer`
`src/bias_detection/claim_bias_analyzer.py`
Model: **Google Gemini** (model fallback chain: `gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-1.5-flash`)

> **Note:** Per-source RoBERTa sentiment and BART zero-shot framing (`BiasDetector`) were removed. All bias alignment now comes from `ClaimBiasAnalyzer` via pre-computed source profiles and Gemini-extracted dimensions. This eliminates a redundant CPU-bound step whose results were overridden by `precomputed_alignments` in every request.

Two-stage design (important for dissertation reproducibility):
- **Stage 1 (Gemini)**: extracts raw bias dimensions for the claim and each source — `emotional_tone`, `political_direction`, `framing_type`, `loaded_phrases`, and three Sri Lanka-specific flags: `gender_bias_signal`, `ethnic_bias_signal`, `trauma_trivialization`. Gemini does language interpretation only — no scoring.
- **Stage 2 (Python)**: computes `alignment` deterministically using an explicit weighted formula:
  ```
  alignment = emotional_proximity × w_e
            + political_match     × w_p
            + framing_match       × w_f
            + gender_bias_match   × w_g
            + ethnic_bias_match   × w_et
            + trauma_bias_match   × w_t
  ```
  Key constraints applied before gating:
  - `political_match` fires only when both claim and source share a **non-neutral** political direction (neutral==neutral is not bias amplification).
  - `framing_match` fires only when both claim and source share a **non-neutral** framing type (same rule).
  - `emotional_proximity` is scaled by `max(|claim_tone|, |source_tone| × 0.4)` so a neutral claim (tone≈0) cannot produce a high proximity score.
  - A sigmoid soft gate scales the raw alignment by overall bias signal strength, replacing the previous hard threshold.

  This means alignment is fully reproducible and precisely describable — Gemini only handles interpretation.
- Same model fallback and retry logic as GeminiNLIVerifier: daily quota → skip model; 503 UNAVAILABLE → retry with backoff; 404 NOT_FOUND → skip model immediately.

---

### Step 4 — Bias-Aware Evidence Weighting + Verdict
*Fast, no I/O — pure computation*

#### `EvidenceWeighter`
`src/evidence_weighting/evidence_weighting.py`

Computes a weight for each source using:

```
weight = base_credibility
       × NLI_confidence
       × (1 − bias_penalty × 0.5)
       × authority_multiplier
       × recency_multiplier
       × temporal_factor
```

**Authority multipliers** (from dataset/domain):

DB collections — three-tier system grounded in Sri Lanka's constitutional structure
and FactCheck.lk practitioner interview (Mahoshadi):

| Tier | Collections | Multiplier | Rationale |
|------|-------------|------------|-----------|
| 1 — Statutory primary | `central_bank_reports`, `supreme_court` | 3.0 | Legally designated sole producers of authoritative data in their domain |
| 2 — Enacted law / fiscal | `acts`, `treasury_press_releases` | 2.7 | Definitive for legal/fiscal claims; one step removed from primary institution |
| 3 — Administrative records | `cabinet_decisions`, `pmd_press_releases`, `appeal_court` | 2.5 | Official executive records and subordinate court judgments |
| 4 — Notifications | `extraordinary_gazettes_*` | 2.3 | Official but procedural/regulatory |
| 5 — Deliberative / unenacted | `hansard_*`, `bills` | 2.0 | Context-only per Mahoshadi; MPs can assert unverified claims; bills have no legal force |
| 6 — Sector statistics | `fisheries_statistics`, `tourism_reports` | 1.8 | Authoritative within narrow domain only |
| 7 — Operational releases | `police_press_releases` | 1.5 | Official but low evidentiary weight |
| 8 — Educational | `education_publications` | 1.3 | Background material |

Web sources — domain-based lookup:

| Source | Multiplier |
|---|---|
| official .gov.lk | 2.5 |
| fact-checkers (AFP, BBC Verify, FactCrescendo) | 1.8 |
| quality Sri Lankan news | 1.5 |
| international outlets | 1.0 |
| unknown / social media | 0.5 |

**Recency multipliers** (source publication date):

| Age | Multiplier |
|---|---|
| < 3 months | 1.0 |
| 3–6 months | 0.8 |
| 6–12 months | 0.6 |
| > 1 year | 0.3 |

**Temporal factor** (claim-content alignment):
- 1.0 if the source content explicitly mentions the claim's time period.
- 0.4 if the source content is about a different time period entirely.

**Key insight — bias alignment penalty**: sources whose political framing *aligns* with the claim's own framing receive a **lower** weight. A source that independently frames the same facts differently is treated as more credible.

- Political claims: 80% pre-computed profile weight, 20% real-time text analysis.
- Non-political claims: 40% profile / 60% real-time text.

#### `VerdictGenerator`
`src/evidence_weighting/evidence_weighting.py`

Weighted vote aggregation across sources → **6-label verdict system**:

| Label | Condition | Display |
|-------|-----------|---------|
| `VERIFIED` | support_pct ≥ 0.80 | True / Verified |
| `MOSTLY_TRUE` | ≥ 0.60 | Mostly True / Largely Supported |
| `PARTIALLY_TRUE` | ≥ 0.35 | Partially True / Mixed Evidence |
| `MOSTLY_FALSE` | ≥ 0.20 (or compound partial) | Mostly False / Largely Contradicted |
| `REFUTED` | < 0.20 | False / Refuted |
| `UNCERTAIN` | active_count < 2 (off-scale) | Unverified / Insufficient Evidence |

`support_pct` = fraction of active weighted votes that support the claim. **NEUTRAL and IRRELEVANT sources are both excluded from the active count** — NEUTRAL sources are abstentions (they engage with the claim but are inconclusive); IRRELEVANT sources are filtered out entirely before verdict aggregation in `weight_all_evidence`.

Also produces **uncertainty decomposition**:
- **Epistemic**: std dev of NLI confidence scores across sources.
- **Aleatoric**: evidence scarcity (ideal = 5 sources).
- **Bias-induced**: ratio of high-alignment sources × label diversity.

Final confidence = `base_confidence × temporal_confidence_multiplier`.
If majority of sources are from the last 2 weeks, verdict is tagged `(PRELIMINARY)`.

---

### Step 5 — Gemini Explanation (on-demand, optional)
`src/verification/gemini_verification.py`
Model: **Google Gemini** via `google.genai`

- **Not run during `/verify`** — skipped by default (`skip_gemini=True`).
- Called explicitly via `POST /explain` or `POST /explain/stream` after the verify result is cached.
- Receives the full weighted evidence, final verdict, temporal context, and claim bias as input.
- Generates a natural-language narrative explaining *why* the verdict was reached and how weighting decisions were made.
- `explain/stream` uses SSE to stream tokens token-by-token as Gemini generates them.

Result is cached in-memory (`_result_cache`, max 10 entries) so `/explain` can be called without re-running the full pipeline.

---

## Model Inventory

| Purpose | Model | Where |
|---|---|---|
| NLI Verification (primary) | **Google Gemini** (`gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-1.5-flash`) | `gemini_nli_verification.py` |
| NLI Verification (fallback) | **DeBERTa-v3-base-mnli** (local) | `verification.py` |
| Claim Bias Dimensions | **Google Gemini** (`gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-1.5-flash`) | `claim_bias_analyzer.py` |
| ~~Per-Source Sentiment~~ | ~~RoBERTa (BiasDetector)~~ | *removed — superseded by ClaimBiasAnalyzer* |
| ~~Per-Source Framing~~ | ~~BART zero-shot (BiasDetector)~~ | *removed — superseded by ClaimBiasAnalyzer* |
| DB Embedding | **SentenceTransformer** (config-specified) | `qdrant_hybrid_retriever.py` |
| Relevance Filtering | **ms-marco-MiniLM-L-6-v2** CrossEncoder (local) | `api.py` |
| Claim Decomposition | Llama 3 via **Groq API** | `claim_decomposer.py` |
| AI Explanation | **Google Gemini** | `gemini_verification.py` |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/verify` | Blocking full pipeline run |
| `POST` | `/verify/stream` | SSE streaming — emits status events then result |
| `POST` | `/explain` | Blocking Gemini explanation (requires prior `/verify`) |
| `POST` | `/explain/stream` | SSE streaming Gemini explanation |

### `/verify` request body
```json
{
  "claim": "Sri Lanka's economy contracted by 3.6% in 2020",
  "skip_gemini": true
}
```

### `/verify` response (key fields)
```json
{
  "verdict": "VERIFIED",
  "confidence": 0.95,
  "total_sources": 21,
  "db_sources": 10,
  "web_sources": 11,
  "temporal_status": "historical",
  "is_breaking_news": false,
  "temporal_warnings": [],
  "uncertainty": { "epistemic": 0.1, "aleatoric": 0.05, "bias_induced": 0.02 },
  "evidence_quality": { "level": "high", ... },
  "claim_bias": { "claim_type": "economic", "framing_type": "...", ... },
  "divergence_level": "low",
  "bias_sources": [ ... ],
  "sources": [ { "title": "...", "verdict": "SUPPORTED", "weight": 0.529, "bias_alignment": 0.0, ... } ],
  "gemini_explanation": null
}
```

---

## Data

| File | Purpose |
|---|---|
| `data/bias_profiles.json` | Pre-computed bias scores for ~99k Sri Lankan news sources. Score −50 (opposition-leaning) to +50 (pro-government). Used by `EvidenceWeighter` and `BiasDetector`. |
| `config.py` | All API keys, Qdrant URL, embedding model name, dataset config, retrieval thresholds. |

---

## How to Run

```bash
# Start Qdrant first (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Start the API
uvicorn api:app --reload
```

Required environment variables (set in `.env` or `config.py`):
- `GROQ_API_KEY`
- `SERP_API_KEY` (for GoogleAIModeRetriever) or `TAVILY_API_KEY` (fallback)
- `GEMINI_API_KEY`
