# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Centralized configuration for the bias-aware fact-checking system
    All settings in one place for easy management
    """
    
    # ============================================
    # API KEYS
    # ============================================
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # ============================================
    # MODEL SETTINGS
    # ============================================
    # NLI model for claim verification (with automatic fallback)
    # Primary: DeBERTa models, Fallback: BART/DistilBERT
    NLI_MODEL = "microsoft/deberta-v3-large-mnli"  # Primary choice (auto-fallback enabled)
    
    # Sentiment model for emotional bias detection
    SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Similarity model for semantic matching
    SIMILARITY_MODEL = "all-MiniLM-L6-v2"
    
    # ============================================
    # EVIDENCE RETRIEVAL SETTINGS
    # ============================================
    NUM_EVIDENCE_SOURCES = 5  # Number of sources to retrieve per claim
    SEARCH_REGION = "lk"      # Sri Lanka
    SEARCH_TYPE = "nws"       # News search
    
    # ============================================
    # BIAS DETECTION - KNOWN SOURCE DATABASE
    # ============================================
    KNOWN_SOURCES = {
        "dailynews.lk": {
            "political_stance": "pro-government",
            "stance_score": 0.8,
            "description": "State-owned, typically pro-government"
        },
        "island.lk": {
            "political_stance": "opposition-leaning",
            "stance_score": 0.6,
            "description": "Independent, critical of government"
        },
        "adaderana.lk": {
            "political_stance": "neutral",
            "stance_score": 0.3,
            "description": "Mainstream, relatively balanced"
        },
        "newsfirst.lk": {
            "political_stance": "neutral",
            "stance_score": 0.3,
            "description": "Commercial broadcaster"
        },
        "economynext.com": {
            "political_stance": "opposition-leaning",
            "stance_score": 0.5,
            "description": "Economic focus, critical analysis"
        },
        "ceylontoday.lk": {
            "political_stance": "neutral",
            "stance_score": 0.4,
            "description": "Independent news outlet"
        }
    }
    
    # ============================================
    # FRAMING BIAS KEYWORDS
    # ============================================
    FRAMING_PATTERNS = {
        "positive_government": [
            "achievement", "success", "progress", "improvement",
            "milestone", "growth", "development", "reform",
            "breakthrough", "accomplishment", "triumph"
        ],
        "negative_government": [
            "failure", "crisis", "scandal", "corruption",
            "mismanagement", "decline", "controversial", "criticism",
            "setback", "controversy", "allegations"
        ]
    }
    
    # ============================================
    # EVIDENCE WEIGHTING PARAMETERS
    # ============================================
    BASE_CREDIBILITY = 0.7        # Default source credibility
    MAX_BIAS_PENALTY = 0.5        # Maximum weight reduction due to bias
    MIN_EVIDENCE_WEIGHT = 0.1     # Minimum weight (never fully discard)
    
    # ============================================
    # UNCERTAINTY THRESHOLDS
    # ============================================
    HIGH_UNCERTAINTY_THRESHOLD = 0.6
    MEDIUM_UNCERTAINTY_THRESHOLD = 0.3
    
    # ============================================
    # VERDICT GENERATION SETTINGS
    # ============================================
    CONFLICTING_THRESHOLD = 0.2   # If support/refute scores within this range → CONFLICTING
    UNCERTAIN_THRESHOLD = 0.5     # If max score below this → UNCERTAIN
    
    # ============================================
    # VALIDATION
    # ============================================
    @classmethod
    def validate(cls):
        """
        Validate that all required configuration is present
        Call this at startup to catch configuration errors early
        """
        errors = []
        
        # Check API key
        if not cls.SERPAPI_KEY:
            errors.append("SERPAPI_KEY not found in environment variables")
        
        # Check if API key looks valid (basic format check)
        if cls.SERPAPI_KEY and len(cls.SERPAPI_KEY) < 20:
            errors.append("SERPAPI_KEY appears to be invalid (too short)")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            error_msg += "\n\nPlease check your .env file and ensure SERPAPI_KEY is set correctly."
            raise ValueError(error_msg)
        
        return True
    
    @staticmethod
    def print_config():
        print("="*60)
        print("CONFIGURATION")
        print("="*60)
        print(f"API Key: {Config.SERPAPI_KEY[:10]}...{Config.SERPAPI_KEY[-5:]}")
        print(f"NLI Model: {Config.NLI_MODEL}")
        print(f"Sentiment Model: {Config.SENTIMENT_MODEL}")
        print(f"Evidence Sources: {Config.NUM_EVIDENCE_SOURCES}")
        print(f"Search Region: {Config.SEARCH_REGION}")
        print(f"Known Sources: {len(Config.KNOWN_SOURCES)}")
        print("="*60)