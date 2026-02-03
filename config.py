# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration for the bias-aware fact-checking system
    """
    
    # API KEYS
    SERPER_KEY = os.getenv('SERPER_API_KEY')
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
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
    
    # @staticmethod
    # def print_config():
    #     print("="*60)
    #     print("CONFIGURATION")
    #     print("="*60)
    #     print(f"Serper API Key: {Config.SERPER_KEY[:10] if Config.SERPER_KEY else 'Not Set'}...{Config.SERPER_KEY[-5:] if Config.SERPER_KEY else ''}")
    #     print(f"Groq API Key: {Config.GROQ_API_KEY[:10] if Config.GROQ_API_KEY else 'Not Set'}...{Config.GROQ_API_KEY[-5:] if Config.GROQ_API_KEY else ''}")
    #     print(f"Evidence Sources: {Config.NUM_EVIDENCE_SOURCES}")
    #     print("="*60)