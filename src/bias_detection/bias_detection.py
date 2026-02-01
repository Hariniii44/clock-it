# from transformers import pipeline
# import re
# from typing import Dict
# import os

# class BiasDetector:
#     def __init__(self):
#         """
#         multi-dimensional bias detection
#         -political stance(manual and heuristics)
#         -emotional tone (sentiment model)
#         -framing patterns (keyword-based)
#         -selection bias - NO
#         """
#         print("loading bias detection models...")

#         #hugging face token from environment variable
#         hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')

#         try:
#             self.sentiment_model = pipeline(
#                 "sentiment-analysis",
#                 model="cardiffnlp/twitter-roberta-base-sentiment-latest",
#                 token=hf_token
#             )
#             print("✅ Sentiment model loaded successfully")
#         except Exception as e:
#             print(f"⚠️ Error loading sentiment model: {e}")
#             print("Falling back to alternative model...")
#             #fallback to a model that doesn't require authentication
#             self.sentiment_model = pipeline(
#                 "sentiment-analysis",
#                 model="distilbert-base-uncased-finetuned-sst-2-english"
#             )
#             print("✅ Fallback sentiment model loaded successfully")

#         #political stance - manual mapping for SL
#         self.known_sources = {
#             "dailynews.lk": {
#                 "political_stance": "pro-government",
#                 "stance_score": 0.8,
#                 "description": "State-owned, typically pro-government"
#             },
#             "island.lk": {
#                 "political_stance": "opposition-leaning",
#                 "stance_score": 0.6,
#                 "description": "Independent, critical of government"
#             },
#             "adaderana.lk": {
#                 "political_stance": "neutral",
#                 "stance_score": 0.3,
#                 "description": "Mainstream, relatively balanced"
#             },
#             "newsfirst.lk": {
#                 "political_stance": "neutral",
#                 "stance_score": 0.3,
#                 "description": "Commercial broadcaster"
#             },
#             "economynext.com": {
#                 "political_stance": "opposition-leaning",
#                 "stance_score": 0.5,
#                 "description": "Economic focus, critical analysis"
#             },
#             "worldbank.org": {
#                 "political_stance": "neutral",
#                 "stance_score": 0.2,
#                 "description": "International organization, data-driven"
#             },
#             "ft.lk": {
#                 "political_stance": "neutral",
#                 "stance_score": 0.4,
#                 "description": "Business-focused, relatively balanced"
#             },
#             "themorning.lk": {
#                 "political_stance": "opposition-leaning",
#                 "stance_score": 0.6,
#                 "description": "Critical of government policies"
#             },
#             "srilankamirror.com": {
#                 "political_stance": "pro-government",
#                 "stance_score": 0.7,
#                 "description": "Pro-government, nationalist-leaning"
#             },
#             "dailymirror.lk": {
#                 "political_stance": "opposition-leaning",
#                 "stance_score": 0.3,
#                 "description": "Mainstream, relatively balanced"
#             }
#         }

#         # Framing keywords
#         self.framing_patterns = {
#             "positive_government": [
#                 "achievement", "success", "progress", "improvement",
#                 "milestone", "growth", "development", "reform"
#             ],
#             "negative_government": [
#                 "failure", "crisis", "scandal", "corruption",
#                 "mismanagement", "decline", "controversial"
#             ]
#         }

#     def detect_political_stance(self, source_url: str) -> Dict:
#             """
#             Detect political bias of a source based on known mappings
#             """

#             domain = self._extract_domain(source_url)

#             return self.known_sources.get(domain, {
#                 "political_stance": "unknown",
#                 "stance_score": 0.0,
#                 "description": "Source not in known mappings"
#             })
        
#     def detect_emotional_tone(self, text: str) -> Dict:
#             """
#             detect emotional tone in text
#             """
#             text_short = text[:512]  #truncate to model limit

#             result = self.sentiment_model(text_short)[0]

#             #map to bias scale 
#             emotion_map = {
#             "POSITIVE": 1.0,
#             "NEGATIVE": -1.0,
#             "NEUTRAL": 0.0,
#             "positive": 1.0,
#             "negative": -1.0,
#             "neutral": 0.0
#             }

#             return {
#                 "emotion": result["label"],
#                 "emotion_score": emotion_map.get(result["label"], 0.0),
#                 "confidence": result["score"]
#             }
        
#     def detect_framing_bias(self, text: str, claim_topic: str = None) -> Dict:
#             """
#             detect framing patterns in text
#             """
#             text_lower = text.lower()

#             positive_count = sum(
#                 1 for word in self.framing_patterns["positive_government"]
#                 if word in text_lower
#             )

#             negative_count = sum(
#                 1 for word in self.framing_patterns["negative_government"]
#                 if word in text_lower
#             )

#             total = positive_count + negative_count
#             if total == 0:
#                 framing_score = 0.0
#                 framing_type = "neutral"
#             else:
#                 framing_score = (positive_count - negative_count) / total
#                 if framing_score > 0.3:
#                     framing_type = "positive"
#                 elif framing_score < -0.3:
#                     framing_type = "negative"
#                 else:
#                     framing_type = "balanced"

#             return {
#                 "framing_type": framing_type,
#                 "framing_score": framing_score,
#                 "positive_words": positive_count,
#                 "negative_words": negative_count
#             }
        
#     def analyze_source(self, text: str, source_url: str) -> Dict:
#             """
#             Comprehensive bias analysis of a source
#             """
            
#             return {
#                 "political_stance": self.detect_political_stance(source_url),
#                 "emotional_tone": self.detect_emotional_tone(text),
#                 "framing_bias": self.detect_framing_bias(text)
#             }
        
#     def _extract_domain(self, url: str) -> str:
#         """
#         Extract domain from URL
#         """
#         match = re.search(r'https?://(?:www\.)?([^/]+)', url)
#         return match.group(1) if match else url
    
    

            

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
        print("✅ Sentiment model loaded successfully")
        
        # Load pre-computed bias profiles
        self.bias_profiles = self._load_bias_profiles()
        print(f"✅ Loaded bias profiles for {len(self.bias_profiles)} sources")
    
    def _load_bias_profiles(self) -> dict:
        """Load pre-computed bias profiles from bias detection pipeline"""
        try:
            with open("data/bias_profiles.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ Bias profiles not found. Using real-time analysis only.")
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
        # Get pre-computed bias profile if available
        source_profile = self.get_source_bias_profile(url) if url else {"has_profile": False}
        
        # Perform real-time text analysis
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
    
    def _analyze_text_bias(self, text: str) -> Dict:
        """Original text-based bias analysis"""
        if not text or len(text.strip()) == 0:
            return self._default_bias_analysis()
        
        # Truncate text to model's maximum length
        text = text[:512]
        
        try:
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
            
            # Simple political stance detection based on keywords
            political_stance, stance_score = self._detect_political_stance(text)
            
            # Simple framing analysis
            framing_type, framing_score = self._analyze_framing(text)
            
            return {
                "political_stance": {
                    "political_stance": political_stance,
                    "stance_score": stance_score
                },
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
            print(f"⚠️ Bias analysis failed: {e}")
            return self._default_bias_analysis()
    
    def _combine_bias_scores(self, source_profile: Dict, text_analysis: Dict) -> Dict:
        """Combine source-level and text-level bias scores"""
        if not source_profile.get("has_profile", False):
            # No source profile, use text analysis only
            return {
                "method": "text_only",
                "political_bias": text_analysis["political_stance"]["stance_score"],
                "emotional_bias": abs(text_analysis["emotional_tone"]["emotion_score"]),
                "overall_bias": (
                    text_analysis["political_stance"]["stance_score"] + 
                    abs(text_analysis["emotional_tone"]["emotion_score"])
                ) / 2
            }
        
        # Combine source profile with text analysis
        source_bias = abs(source_profile["bias_score"]) / 50  # Normalize from -50:+50 to 0:1
        text_political = text_analysis["political_stance"]["stance_score"]
        text_emotional = abs(text_analysis["emotional_tone"]["emotion_score"])
        
        # Weighted combination (source profile gets more weight)
        combined_political = (0.7 * source_bias) + (0.3 * text_political)
        combined_emotional = text_emotional  # Text-level only
        
        return {
            "method": "combined",
            "source_bias_score": source_profile["bias_score"],
            "source_interpretation": source_profile["bias_interpretation"],
            "political_bias": combined_political,
            "emotional_bias": combined_emotional,
            "overall_bias": (combined_political + combined_emotional) / 2,
            "confidence": source_profile["confidence"]
        }
    
    def _detect_political_stance(self, text: str) -> tuple:
        """Simple keyword-based political stance detection"""
        text_lower = text.lower()
        
        # Government/pro-government keywords
        gov_keywords = ["government", "administration", "policy", "development", "progress", 
                       "achievement", "success", "improvement"]
        
        # Opposition/critical keywords  
        opp_keywords = ["opposition", "criticism", "failure", "corruption", "protest",
                       "crisis", "problem", "scandal"]
        
        gov_count = sum(1 for kw in gov_keywords if kw in text_lower)
        opp_count = sum(1 for kw in opp_keywords if kw in text_lower)
        
        if gov_count > opp_count:
            stance_score = min(gov_count * 0.1, 1.0)
            return "pro-government", stance_score
        elif opp_count > gov_count:
            stance_score = min(opp_count * 0.1, 1.0) 
            return "opposition-leaning", stance_score
        else:
            return "unknown", 0.0
    
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
            "political_stance": {
                "political_stance": "unknown",
                "stance_score": 0.0
            },
            "emotional_tone": {
                "emotion": "neutral", 
                "emotion_score": 0.0
            },
            "framing_bias": {
                "framing_type": "neutral",
                "framing_score": 0.0
            }
        }
