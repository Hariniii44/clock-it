from transformers import pipeline
import re
from typing import Dict
import os

class BiasDetector:
    def __init__(self):
        """
        multi-dimensional bias detection
        -political stance(manual and heuristics)
        -emotional tone (sentiment model)
        -framing patterns (keyword-based)
        -selection bias - NO
        """
        print("loading bias detection models...")

        #hugging face token from environment variable
        hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')

        try:
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                token=hf_token
            )
            print("✅ Sentiment model loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading sentiment model: {e}")
            print("Falling back to alternative model...")
            #fallback to a model that doesn't require authentication
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            print("✅ Fallback sentiment model loaded successfully")

        #political stance - manual mapping for SL
        self.known_sources = {
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
            "worldbank.org": {
                "political_stance": "neutral",
                "stance_score": 0.2,
                "description": "International organization, data-driven"
            },
            "ft.lk": {
                "political_stance": "neutral",
                "stance_score": 0.4,
                "description": "Business-focused, relatively balanced"
            },
            "themorning.lk": {
                "political_stance": "opposition-leaning",
                "stance_score": 0.6,
                "description": "Critical of government policies"
            },
            "srilankamirror.com": {
                "political_stance": "pro-government",
                "stance_score": 0.7,
                "description": "Pro-government, nationalist-leaning"
            },
            "dailymirror.lk": {
                "political_stance": "opposition-leaning",
                "stance_score": 0.3,
                "description": "Mainstream, relatively balanced"
            }
        }

        # Framing keywords
        self.framing_patterns = {
            "positive_government": [
                "achievement", "success", "progress", "improvement",
                "milestone", "growth", "development", "reform"
            ],
            "negative_government": [
                "failure", "crisis", "scandal", "corruption",
                "mismanagement", "decline", "controversial"
            ]
        }

    def detect_political_stance(self, source_url: str) -> Dict:
            """
            Detect political bias of a source based on known mappings
            """

            domain = self._extract_domain(source_url)

            return self.known_sources.get(domain, {
                "political_stance": "unknown",
                "stance_score": 0.0,
                "description": "Source not in known mappings"
            })
        
    def detect_emotional_tone(self, text: str) -> Dict:
            """
            detect emotional tone in text
            """
            text_short = text[:512]  #truncate to model limit

            result = self.sentiment_model(text_short)[0]

            #map to bias scale 
            emotion_map = {
            "POSITIVE": 1.0,
            "NEGATIVE": -1.0,
            "NEUTRAL": 0.0,
            "positive": 1.0,
            "negative": -1.0,
            "neutral": 0.0
            }

            return {
                "emotion": result["label"],
                "emotion_score": emotion_map.get(result["label"], 0.0),
                "confidence": result["score"]
            }
        
    def detect_framing_bias(self, text: str, claim_topic: str = None) -> Dict:
            """
            detect framing patterns in text
            """
            text_lower = text.lower()

            positive_count = sum(
                1 for word in self.framing_patterns["positive_government"]
                if word in text_lower
            )

            negative_count = sum(
                1 for word in self.framing_patterns["negative_government"]
                if word in text_lower
            )

            total = positive_count + negative_count
            if total == 0:
                framing_score = 0.0
                framing_type = "neutral"
            else:
                framing_score = (positive_count - negative_count) / total
                if framing_score > 0.3:
                    framing_type = "positive"
                elif framing_score < -0.3:
                    framing_type = "negative"
                else:
                    framing_type = "balanced"

            return {
                "framing_type": framing_type,
                "framing_score": framing_score,
                "positive_words": positive_count,
                "negative_words": negative_count
            }
        
    def analyze_source(self, text: str, source_url: str) -> Dict:
            """
            Comprehensive bias analysis of a source
            """
            
            return {
                "political_stance": self.detect_political_stance(source_url),
                "emotional_tone": self.detect_emotional_tone(text),
                "framing_bias": self.detect_framing_bias(text)
            }
        
    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL
        """
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return match.group(1) if match else url
    
    

            
