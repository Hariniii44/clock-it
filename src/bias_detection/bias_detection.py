import json
from transformers import pipeline
from typing import Dict, List
import os
import warnings
warnings.filterwarnings('ignore')

class BiasDetector:
    def __init__(self):
        """Initialize bias detection with both real-time analysis and pre-computed profiles"""
        print("loading bias detection models...")
        
        # Load sentiment analysis model
        hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1,
            token=hf_token
        )
        print("Sentiment model loaded successfully")
        
        # Load pre-computed bias profiles
        self.bias_profiles = self._load_bias_profiles()
        print(f"Loaded bias profiles for {len(self.bias_profiles)} sources")
    
    def _load_bias_profiles(self) -> dict:
        """Load pre-computed bias profiles from bias detection pipeline"""
        try:
            with open("data/bias_profiles.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Bias profiles not found. Using real-time analysis only.")
            return {}
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            if url.startswith('http'):
                url = url.split('://', 1)[1]
            domain = url.split('/')[0]
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return ""
    
    def get_source_bias_profile(self, url: str) -> Dict:
        """Get pre-computed bias profile for a source"""
        domain = self._extract_domain(url)
        
        if domain in self.bias_profiles:
            profile = self.bias_profiles[domain]
            return {
                "has_profile": True,
                "source": domain,
                "bias_score": profile["bias_score"],
                "bias_interpretation": profile["interpretation"],
                "confidence": profile["confidence"],
                "articles_analyzed": profile["articles_analyzed"]
            }
        else:
            return {
                "has_profile": False,
                "source": domain,
                "bias_score": 0.0,
                "bias_interpretation": "Unknown source",
                "confidence": 0.0,
                "articles_analyzed": 0
            }
    
    def analyze_source(self, text: str, url: str = None) -> Dict:
        """
        Comprehensive bias analysis combining pre-computed profiles and real-time analysis
        """
        try:
            # Get pre-computed bias profile if available
            source_profile = self.get_source_bias_profile(url) if url else {"has_profile": False}
            
            # Perform real-time text analysis with safety checks
            text_analysis = self._analyze_text_bias(text)
            
            # Combine both analyses
            combined_analysis = {
                "source_profile": source_profile,
                "text_analysis": text_analysis,
                "combined_score": self._combine_bias_scores(source_profile, text_analysis)
            }
            
            # For backward compatibility, maintain original structure
            combined_analysis.update(text_analysis)
            
            return combined_analysis
            
        except Exception as e:
            # Graceful fallback for parliamentary sources or other errors
            print(f"⚠️ Bias analysis failed: {e}")
            return {
                'emotional_tone': {'emotion': 'neutral', 'emotion_score': 0.0},
                'framing_bias': {'framing_type': 'neutral', 'framing_score': 0.0},
                'source_profile': {'has_profile': False},
                'combined_score': {'method': 'fallback', 'overall_bias': 0.0}
            }
    
    def _analyze_text_bias(self, text: str) -> Dict:
        """Original text-based bias analysis with enhanced safety checks"""
        if not text or len(text.strip()) == 0:
            return self._default_bias_analysis()
        
        try:
            # More aggressive text truncation to prevent tensor issues
            # RoBERTa tokenizer creates ~1.3 tokens per word on average
            words = text.split()
            if len(words) > 300:  # Even more conservative limit
                text = ' '.join(words[:300])
            
            # Character limit as additional safety
            if len(text) > 1500:
                text = text[:1500]
            
            # Clean text to remove special characters that may cause tokenization issues
            import re
            text = re.sub(r'[^\w\s.,!?-]', ' ', text)
            text = ' '.join(text.split())  # Remove extra whitespace
            
            # Skip analysis if text is too short after cleaning
            if len(text.strip()) < 10:
                return self._default_bias_analysis()
            
            # Sentiment analysis
            sentiment_result = self.sentiment_model(text)[0]
            sentiment_label = sentiment_result['label'].lower()
            sentiment_score = sentiment_result['score']
            
            # Map sentiment to emotional tone
            if 'positive' in sentiment_label:
                emotion = 'positive'
                emotion_score = sentiment_score
            elif 'negative' in sentiment_label:
                emotion = 'negative' 
                emotion_score = -sentiment_score
            else:
                emotion = 'neutral'
                emotion_score = 0.0
                        
            # Simple framing analysis
            framing_type, framing_score = self._analyze_framing(text)
            
            return {
                "emotional_tone": {
                    "emotion": emotion,
                    "emotion_score": emotion_score
                },
                "framing_bias": {
                    "framing_type": framing_type,
                    "framing_score": framing_score
                }
            }
            
        except Exception as e:
            print(f" Text bias analysis failed: {e}")
            return self._default_bias_analysis()
    
    def _combine_bias_scores(self, source_profile: Dict, text_analysis: Dict) -> Dict:
        """Combine source-level and text-level bias scores"""
        # Get text-level bias scores from available analysis
        text_emotional = abs(text_analysis["emotional_tone"]["emotion_score"])
        text_framing = text_analysis["framing_bias"]["framing_score"]
        
        if not source_profile.get("has_profile", False):
            # No source profile, use text analysis only
            text_overall_bias = (text_emotional + text_framing) / 2
            return {
                "method": "text_only",
                "emotional_bias": text_emotional,
                "framing_bias": text_framing,
                "overall_bias": text_overall_bias
            }
        
        # Combine source profile with text analysis
        source_bias = abs(source_profile["bias_score"]) / 50  # Normalize from -50:+50 to 0:1
        
        # Weighted combination (source profile gets more weight for political bias)
        combined_political = source_bias  # Use source profile for political stance
        combined_emotional = text_emotional  # Use text analysis for emotional tone
        combined_framing = text_framing  # Use text analysis for framing
        
        return {
            "method": "combined",
            "source_bias_score": source_profile["bias_score"],
            "source_interpretation": source_profile["bias_interpretation"],
            "political_bias": combined_political,
            "emotional_bias": combined_emotional,
            "framing_bias": combined_framing,
            "overall_bias": (combined_political + combined_emotional + combined_framing) / 3,
            "confidence": source_profile["confidence"]
        }
    
    def _analyze_framing(self, text: str) -> tuple:
        """Simple framing analysis"""
        text_lower = text.lower()
        
        # Check for loaded/emotional language
        emotional_words = ["crisis", "disaster", "scandal", "amazing", "terrible", 
                          "shocking", "outrageous", "brilliant", "devastating"]
        
        emotional_count = sum(1 for word in emotional_words if word in text_lower)
        
        if emotional_count > 2:
            return "emotional", min(emotional_count * 0.2, 1.0)
        else:
            return "neutral", 0.0
    
    def _default_bias_analysis(self) -> Dict:
        """Default analysis when bias detection fails"""
        return {
            "emotional_tone": {
                "emotion": "neutral", 
                "emotion_score": 0.0
            },
            "framing_bias": {
                "framing_type": "neutral",
                "framing_score": 0.0
            }
        }
