# config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration for the bias-aware fact-checking system
    """
    
    # API KEYS
    SERPER_KEY = os.getenv('SERPER_API_KEY')
    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    SERP_API_KEY = os.getenv('SERP_API_KEY', '')       # SerpAPI for Google AI Mode

    # QDRANT SETTINGS
    QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')
    
    # EVIDENCE RETRIEVAL SETTINGS
    NUM_EVIDENCE_SOURCES = 20  # Number of sources to retrieve per claim

    # VALIDATION
    @classmethod
    def validate(cls):
        """
        Validate that all required configuration is present
        """
        errors = []
        
        if not cls.SERPER_KEY:
            errors.append("SERPER_API_KEY not found in environment variables")
        
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY not found in environment variables")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            error_msg += "\n\nPlease check your .env file and ensure API keys are set correctly."
            raise ValueError(error_msg)
        
        return True
    
# dataset retrieval configurations

#paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INDICES_DIR = DATA_DIR / "indices"
DATASETS_DIR = DATA_DIR / "datasets"

# Create directories
INDICES_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration - Nuuwan's Sri Lankan government datasets

# Replace your DATASETS_CONFIG with this expanded version:

DATASETS_CONFIG = {
    # PARLIAMENTARY DATA (High Authority)
    # 'hansard': {  # 354k docs - too large, skipped
    #     'hf_name': 'nuuuwan/lk-hansard-chunks',
    #     'description': 'Parliamentary debates and speeches (354k docs)',
    #     'authority': 1.0,
    #     'use_for': ['government_statement', 'policy', 'parliamentary', 'legislative', 'all']
    # },
    'hansard_2000s': {
        'hf_name': 'nuuuwan/lk-hansard-2000s-chunks',
        'description': 'Parliamentary debates 2000s (2.83k docs)',
        'authority': 0.95,
        'use_for': ['government_statement', 'policy', 'parliamentary', 'legislative', 'all']
    },
    'hansard_2020s': {
        'hf_name': 'nuuuwan/lk-hansard-2020s-chunks',
        'description': 'Parliamentary debates 2020s (53k docs)',
        'authority': 1.0,
        'use_for': ['government_statement', 'policy', 'parliamentary', 'legislative', 'all']
    },
    'hansard_2010s': {
        'hf_name': 'nuuuwan/lk-hansard-2010s-chunks',
        'description': 'Parliamentary debates 2010s (1k docs)',
        'authority': 0.95,
        'use_for': ['government_statement', 'policy', 'parliamentary', 'legislative', 'all']
    },
    
    # PRESIDENTIAL & CABINET (High Authority)
    'pmd_press_releases': {
        'hf_name': 'nuuuwan/lk-pmd-press-releases-chunks',
        'description': 'Presidential press releases (3.38k docs)',
        'authority': 1.0,
        'use_for': ['government_statement', 'presidential', 'president_said', 'all']
    },
    'cabinet_decisions': {
        'hf_name': 'nuuuwan/lk-cabinet-decisions-chunks',
        'description': 'Cabinet decisions (10.5k docs)',
        'authority': 1.0,
        'use_for': ['policy', 'economic', 'government_policy', 'cabinet_decision', 'all']
    },
    'treasury_press_releases': {
        'hf_name': 'nuuuwan/lk-treasury-press-releases-chunks',
        'description': 'Treasury press releases (210 docs)',
        'authority': 1.0,
        'use_for': ['economic', 'financial', 'monetary', 'budget', 'all']
    },
    
    # LEGAL & JUDICIAL (High Authority)
    'supreme_court': {
        'hf_name': 'nuuuwan/lk-supreme-court-judgements-chunks',
        'description': 'Supreme Court judgments (24.8k docs)',
        'authority': 1.0,
        'use_for': ['legal', 'judicial', 'constitutional', 'court_ruling']
    },
    'appeal_court': {
        'hf_name': 'nuuuwan/lk-appeal-court-judgements-chunks',
        'description': 'Appeal Court judgments (73.8k docs)',
        'authority': 0.95,
        'use_for': ['legal', 'judicial', 'court_ruling']
    },
    
    # LAWS & ACTS (High Authority)
    'acts': {
        'hf_name': 'nuuuwan/lk-acts-chunks',
        'description': 'Sri Lankan Acts and Laws (23.5k docs)',
        'authority': 1.0,
        'use_for': ['legal', 'legislative', 'law', 'statute', 'all']
    },
    'bills': {
        'hf_name': 'nuuuwan/lk-bills-chunks',
        'description': 'Parliamentary Bills (51k docs)',
        'authority': 0.95,
        'use_for': ['legislative', 'law', 'parliamentary', 'policy']
    },
    
    # ECONOMIC & FINANCIAL (High Authority)
    'central_bank_reports': {
        'hf_name': 'nuuuwan/cbsl-annual-reports-chunks',
        'description': 'Central Bank annual reports (50.8k docs)',
        'authority': 1.0,
        'use_for': ['economic', 'financial', 'monetary', 'statistics', 'all']
    },
    'fisheries_statistics': {
        'hf_name': 'nuuuwan/lk-fisheries-annual-statistics-reports-chunks',
        'description': 'Fisheries annual statistics reports (450 docs)',
        'authority': 0.95,
        'use_for': ['economic', 'statistics', 'fisheries', 'all']
    },
    'tourism_reports': {
        'hf_name': 'nuuuwan/lk-tourism-monthly-reports-chunks',
        'description': 'Tourism monthly reports (1.1k docs)',
        'authority': 0.95,
        'use_for': ['economic', 'statistics', 'tourism', 'all']
    },

    # GOVERNMENT GAZETTES (Medium-High Authority)
    'extraordinary_gazettes_2020s': {
        'hf_name': 'nuuuwan/lk-extraordinary-gazettes-2020s-chunks',
        'description': 'Extraordinary Gazettes 2020s (54.3k docs)',
        'authority': 0.9,
        'use_for': ['government_statement', 'policy', 'legal', 'administrative', 'all']
    },
    # 'extraordinary_gazettes_2010s': {  # 118k docs - too large, skipped
    #     'hf_name': 'nuuuwan/lk-extraordinary-gazettes-2010s-chunks',
    #     'description': 'Extraordinary Gazettes 2010s (118k docs)',
    #     'authority': 0.9,
    #     'use_for': ['government_statement', 'policy', 'legal', 'administrative', 'all']
    # },
    
    # LAW ENFORCEMENT 
    'police_press_releases': {
        'hf_name': 'nuuuwan/lk-police-press-releases-chunks',
        'description': 'Police press releases (2.16k docs)',
        'authority': 0.85,
        'use_for': ['law_enforcement', 'crime', 'police', 'security', 'all']
    },
    
    # NEWS & MEDIA (Lower Authority but High Coverage)
    # 'news': {  # 164k docs, authority 0.7 - too large, low authority, skipped
    #     'hf_name': 'nuuuwan/lk-news-chunks',
    #     'description': 'Sri Lankan news articles (164k docs)',
    #     'authority': 0.7,
    #     'use_for': ['all']
    # },
    
    # EDUCATION
    'education_publications': {
        'hf_name': 'nuuuwan/lk-edupub-chunks',
        'description': 'Education publications (8.31k docs)',
        'authority': 0.8,
        'use_for': ['education', 'policy', 'academic', 'all']
    }
}

# DATASETS_CONFIG = {
#     'hansard': {
#         'hf_name': 'nuuuwan/lk-hansard-chunks', 
#         'description': 'Parliamentary debates and speeches',
#         'authority': 0.9,
#         'use_for': ['government_statement', 'policy', 'parliamentary', 'legislative', 'all']
#     },
#     'hansard_2020s': {
#         'hf_name': 'nuuuwan/lk-hansard-2020s-chunks',
#         'description': 'Parliamentary debates and speeches',
#         'authority': 1.0,
#         'use_for': ['government_statement', 'policy', 'parliamentary', 'legislative', 'all']
#     },
#     'hansard_2010s': {
#         'hf_name': 'nuuuwan/lk-hansard-2010s-chunks',
#         'description': 'Parliamentary debates 2010s',
#         'authority': 0.95,
#         'use_for': ['government_statement', 'policy', 'parliamentary', 'legislative', 'all']
#     },
#     'pmd_press_releases': {
#         'hf_name': 'nuuuwan/lk-pmd-press-releases-chunks',
#         'description': 'Presidential press releases (chunked)',
#         'authority': 1.0,
#         'use_for': ['government_statement', 'presidential', 'president_said', 'all']
#     },
#     'pmd_press_release': {
#         'hf_name': 'nuuuwan/lk-pmd-press-releases-chunks', 
#         'description': 'Presidential press releases (alternative)',
#         'authority': 1.0,
#         'use_for': ['government_statement', 'presidential', 'president_said', 'all']
#     },
#     'cabinet_decisions': {
#         'hf_name': 'nuuuwan/lk-cabinet-decisions-chunks',
#         'description': 'Cabinet decisions',
#         'authority': 1.0,
#         'use_for': ['policy', 'economic', 'government_policy', 'cabinet_decision', 'all']
#     },
#     'supreme_court': {
#         'hf_name': 'nuuuwan/lk-supreme-court-judgements-chunks',
#         'description': 'Supreme Court judgments',
#         'authority': 1.0,
#         'use_for': ['legal', 'judicial', 'constitutional']
#     },
#     'appeal_court': {
#         'hf_name': 'nuuuwan/lk-news-chunks',
#         'description': 'Appeal Court judgments', 
#         'authority': 0.95,
#         'use_for': ['news']
#     },
#     'news': {
#         'hf_name': 'nuuuwan/lk-news-docs',
#         'description': 'Sri Lankan news articles',
#         'authority': 0.7,
#         'use_for': ['all']  #always searching news as fallback
#     }
# }

# Qdrant settings
QDRANT_URL     = os.getenv('QDRANT_URL', 'http://localhost:6333')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', None)

# Embedding and retrieval settings
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'  # Fast, good quality
TOP_K_PER_DATASET = 5  # Retrieve top 5 from each dataset
RELEVANCE_THRESHOLD = 0.5  # Minimum similarity score
MAX_PASSAGE_LENGTH = 1000  # Maximum characters per passage

# Hybrid retrieval settings
USE_HYBRID_RETRIEVAL = True  # Set to False to use only Serper (existing method)
HYBRID_FALLBACK_TO_SERPER = True  # Fall back to Serper if no dataset results


def get_datasets_for_claim_type(claim_types):
    """
    Get relevant datasets for given claim types
    
    Args:
        claim_types: List of claim types (e.g., ['presidential', 'policy'])
        
    Returns:
        List of dataset names to search
    """
    relevant_datasets = []
    
    for dataset_name, config in DATASETS_CONFIG.items():
        use_for = config['use_for']
        
        # Check if any claim type matches this dataset
        if 'all' in use_for or any(ctype in use_for for ctype in claim_types):
            relevant_datasets.append(dataset_name)
    
    # Sort by authority (highest first)
    relevant_datasets.sort(
        key=lambda x: DATASETS_CONFIG[x]['authority'], 
        reverse=True
    )
    
    return relevant_datasets

def print_config():
    """Print configuration summary"""
    print("="*60)
    print("BIAS-AWARE FACT-CHECKING CONFIGURATION")
    print("="*60)
    print("EXISTING PIPELINE:")
    print(f"  Serper API Key: {'Set' if Config.SERPER_KEY else 'Missing'}")
    print(f"  Groq API Key: {'Set' if Config.GROQ_API_KEY else 'Missing'}")
    print(f"  Evidence Sources: {Config.NUM_EVIDENCE_SOURCES}")
    
    print("\nNEW HYBRID RETRIEVAL:")
    print(f"  Hybrid Mode: {' Enabled' if USE_HYBRID_RETRIEVAL else ' Disabled'}")
    print(f"  Serper Fallback: {' Yes' if HYBRID_FALLBACK_TO_SERPER else ' No'}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Available Datasets: {len(DATASETS_CONFIG)}")
    
    print(f"\nDATA DIRECTORIES:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Dir: {DATA_DIR}")
    print(f"  Indices Dir: {INDICES_DIR} ({' Exists' if INDICES_DIR.exists() else ' Missing'})")
    print(f"  Datasets Dir: {DATASETS_DIR} ({' Exists' if DATASETS_DIR.exists() else ' Missing'})")
    
    print(f"\nDATASET CATALOG:")
    for name, config in DATASETS_CONFIG.items():
        print(f"   {name}")
        print(f"     Authority: {config['authority']}")
        print(f"     Use for: {config['use_for']}")
        print(f"     HuggingFace: {config['hf_name']}")
        print()

    print("="*60)

def get_serper_key():
    """Get Serper API key (for backward compatibility)"""
    return Config.SERPER_KEY

def get_groq_key():
    """Get Groq API key (for backward compatibility)"""
    return Config.GROQ_API_KEY

def get_num_evidence_sources():
    """Get number of evidence sources (for backward compatibility)"""
    return Config.NUM_EVIDENCE_SOURCES



