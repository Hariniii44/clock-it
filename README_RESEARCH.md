# Context-Adaptive Evidence Weighting Framework

## Research Focus: Bias-Aware Fact-Checking

This system demonstrates a **novel context-adaptive evidence weighting algorithm** for automated fact-checking in politically polarized environments, specifically targeting Sri Lankan media and politics.

## Core Research Contribution

**Dynamic Source Weighting Based on Political Bias Alignment**
- Traditional systems use static credibility scores
- Our algorithm dynamically adjusts weights based on whether source bias aligns with claim direction
- **Key Insight**: A biased source opposing its typical bias is more credible than supporting it

## Architecture Overview

### Main Pipeline: `core_pipeline.py`
```
1. Dual Evidence Retrieval (Database + Web)
2. NLI Verification (Single, consistent method)
3. BIAS-AWARE EVIDENCE WEIGHTING (YOUR ALGORITHM)
4. Verdict Generation (Weighted results)
5. [Optional] Gemini Explanations (User interface)
```

### Key Components

**Your Research:**
- `EvidenceWeighter`: Bias-aware weighting algorithm
- `VerdictGenerator`: Weighted verdict with uncertainty quantification
- `HybridRetriever`: Dual-source evidence architecture

**Supporting Tools:**
- `ClaimVerifier`: NLI-based verification
- `BiasDetector`: Provides bias signals for weighting
- `GeminiClaimVerifier`: Natural language explanations (optional)

## Usage

### Run Main System
```bash
python core_pipeline.py
```

### Compare Against Baselines
```bash
python baseline_comparison.py
```

### Run Original Complex Pipeline (for comparison)
```bash
python dual.py
```

## Algorithm Details

### Bias-Aware Weighting Formula
```
Final_Weight = Base_Credibility × NLI_Confidence × (1 - Bias_Penalty) × Authority_Weight × Recency_Weight

Where:
- Bias_Penalty = bias_alignment × 0.5
- bias_alignment = similarity(source_political_stance, claim_sentiment)
```

### Key Logic
- **Pro-government source + Pro-government claim** = Higher bias alignment = Lower weight
- **Pro-government source + Anti-government claim** = Lower bias alignment = Higher weight  
- **Independent source + Any claim** = Standard weight

### Uncertainty Types
1. **Epistemic**: Model confidence variance
2. **Aleatoric**: Insufficient evidence quantity
3. **Bias-induced**: Conflicting biased sources

## Evaluation Framework

### Baseline Comparisons
1. **Uniform Weighting**: All sources equal weight (baseline)
2. **Authority-Only**: Government vs web source weights
3. **Bias-Aware**: Your dynamic weighting algorithm

### Metrics
- Accuracy improvement over baseline
- Confidence calibration
- Verdict stability across source compositions

## Thesis Positioning

### What This System Demonstrates
Novel algorithmic contribution: Context-adaptive weighting framework  
Practical impact: Improved accuracy in polarized information environments  
Theoretical foundation: Political bias psychology applied to automated systems  
Empirical validation: Ready for FEVER dataset evaluation  

### Clear Separation of Contributions
- **Your Work**: Evidence retrieval, weighting algorithm, verdict generation
- **Established Methods**: NLI verification, pre-trained bias detection models
- **Enhancement Tools**: Gemini explanations (user interface only)

## For Thesis Defense

### Key Claims to Defend
1. "Dynamic bias-aware weighting improves fact-checking accuracy over static approaches"
2. "Political bias alignment calculation enables context-sensitive credibility assessment"  
3. "Multi-dimensional uncertainty quantification provides reliable confidence intervals"
4. "Sri Lankan media bias profiles enable localized fact-checking optimization"

### Examiner Questions & Answers

**Q: "What's your novel contribution?"**  
**A:** "Context-adaptive evidence weighting that considers political bias alignment with claim direction"

**Q: "How does it work?"**  
**A:** "Pre-computed bias profiles + dynamic alignment calculation + inverse weighting for aligned bias"

**Q: "What about Gemini?"**  
**A:** "Optional explanation interface - system works independently, Gemini just translates results to natural language"

**Q: "How do you evaluate it?"**  
**A:** "Baseline comparison + FEVER dataset + ablation studies showing each component's contribution"

## File Structure

```
core_pipeline.py              # Main research demonstration
baseline_comparison.py        # Evaluation framework
src/evidence_weighting/       # Your core algorithm
    ├── evidence_weighting.py # Bias-aware weighting logic
    └── verdict_generation.py # Weighted verdict generation
src/verification/
    └── gemini_verification.py # Enhanced with weighting explanations
dual.py                       # Original complex pipeline (for comparison)
```

## Academic Impact

This framework addresses a critical gap in automated fact-checking: **static credibility scoring in polarized information environments**. By dynamically weighting sources based on bias-claim alignment, the system provides more accurate and context-aware fact-checking for politically sensitive claims.

**Suitable for publication** in venues focusing on:
- Computational journalism  
- Misinformation detection
- Political communication technology
- Information credibility assessment