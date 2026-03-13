# Bias-Aware Fact-Checking System

A comprehensive fact-checking system that integrates bias profiling of Sri Lankan news sources with advanced evidence retrieval and verification to provide more accurate and contextually-aware fact-checking results.

## Features

- **Bias-Aware Evidence Weighting**: Automatically adjusts evidence reliability based on pre-computed bias profiles of Sri Lankan news sources
- **Advanced Evidence Retrieval**: Uses hybrid approach combining web search and claim decomposition
- **Multi-AI Verification**: Employs multiple AI models for cross-verification
- **Comprehensive Bias Analysis**: Real-time bias detection plus pre-computed source profiles
- **Uncertainty Quantification**: Provides detailed uncertainty breakdown (epistemic, aleatoric, bias-induced)
- **Intelligent Synthesis**: AI-powered explanation generation with citation support

## System Requirements

- Python 3.8 or higher
- Internet connection for API calls
- API keys for external services

## Installation

1. **Clone/Download the project**
   ```bash
   cd path/to/your/project/directory
   ```

2. **Set up Python virtual environment** (recommended)
   ```bash
   python -m venv bias_aware_env
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     bias_aware_env\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source bias_aware_env/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Create API accounts and get keys:**
   - **Groq API**: Visit [Groq Console](https://console.groq.com/) for free API access
   - **Serper API**: Visit [Serper.dev](https://serper.dev/) for web search API (free tier available)

2. **Configure your API keys in `config.py`:**
   ```python
   # Update these with your actual API keys
   GROQ_API_KEY = "your_groq_api_key_here"
   SERPER_KEY = "your_serper_api_key_here"
   ```

3. **Verify configuration:**
   ```bash
   python -c "from config import Config; Config.validate(); print('Configuration OK')"
   ```

## Usage

### Quick Start - Interactive Testing

Run the main test pipeline for interactive fact-checking:

```bash
python test_pipeline.py
```

This will:
1. Validate your configuration
2. Initialize all system components
3. Prompt you to enter a claim to fact-check
4. Process the claim and display comprehensive results

### Example Claims to Test

- "Parliament of Sri Lanka has one of the lowest representations of female MPs in all of South Asia."
- "Sri Lanka's IMF bailout program has improved the country's economic stability."

### Understanding the Output

The system provides several layers of analysis:

1. **Verdict Explanation**: AI-generated summary with key findings
2. **Fact-Check Verdict**: Final determination (SUPPORTS/REFUTES/NEUTRAL) with confidence
3. **Evidence Breakdown**: Details about each source, verification, and weighting
4. **Bias Analysis**: How bias profiling affected the final verdict
5. **Uncertainty Analysis**: Breakdown of different uncertainty types

### Expected Output Structure

```
==============================================================
VERDICTS EXPLANATION
==============================================================
[AI-generated explanation of findings]

FACT-CHECK VERDICT: SUPPORTS (85% confidence)

==============================================================
DETAILED ANALYSIS
==============================================================

EVIDENCE BREAKDOWN:
[Source-by-source analysis with weights and verification]

==============================================================
BIAS PROFILING INTEGRATION SUMMARY
==============================================================
[Summary of how bias profiling influenced the results]
```

## Component Overview

### Core Modules

- **`src/evidence_retrieval/`**: Advanced evidence gathering with claim decomposition
- **`src/verification/`**: Multi-AI claim verification system
- **`src/bias_detection/`**: Real-time bias analysis and source profiling
- **`src/evidence_weighting/`**: Bias-aware evidence weighting and verdict generation
- **`src/synthesis/`**: AI-powered explanation synthesis

### Bias Profiling System

- **`src/bias_profiling/`**: Complete bias profiling pipeline
  - `bias_data_preparation.py`: Loads and filters Sri Lankan news data
  - `politician_names.py`: Identifies political content
  - `bias_analyzer.py`: Analyzes bias patterns
  - `bias_scorer.py`: Generates final bias profiles

### Data Files

- **`data/bias_profiles.json`**: Pre-computed bias profiles for Sri Lankan sources
- **`data/political_articles.csv`**: Filtered political articles for analysis
- **`data/bias_matrix.json`**: Bias comparison matrix between sources

## Bias Profiling Data

The system includes pre-computed bias profiles for major Sri Lankan news sources:

- `adaderana.lk`
- `dailymirror.lk` 
- `economynext.com`
- `ft.lk`
- `island.lk`
- `newsfirst.lk`
- `sundayobserver.lk`

These profiles enable more accurate weighting of evidence based on historical bias patterns.

### Performance Tips

- **For best results**: Use claims that are likely to have coverage in Sri Lankan media
- **Testing efficiency**: Start with shorter, factual claims
- **API usage**: The system makes multiple API calls per fact-check (10-30 depending on evidence found)

## API Usage and Costs

- **Groq API**: Free tier sufficient for testing (30 req/min)
- **Serper API**: Free tier includes 2500 searches/month

## Academic Context

This system implements bias-aware fact-checking as described in academic literature on:
- Source reliability assessment
- Bias quantification in news media
- Multi-source evidence aggregation
- Uncertainty quantification in automated fact-checking