import json
from transformers import pipeline
from typing import Dict, List
import os
import warnings
warnings.filterwarnings('ignore')

class BiasDetector:
    def __init__(self):
        """Initialize bias detection with both real-time analysis and pre-computed profiles"""
        print("Loading bias detection models...")
        
        # Memory management flags
        self.bart_model_failed = False
        self.zero_shot_classifier = None
        
        # Load sentiment analysis model
        hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1,
            token=hf_token
        )
        print("Sentiment model loaded successfully")

        try:
            self.emotion_model = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-emotion",
                device=-1,
                token=hf_token
            )
            print("Emotion model loaded successfully")
        except Exception as e:
            print(f"Emotion model loading failed: {e}")
            self.emotion_model = None
        
        # Attempt to load BART model once during initialization
        try:
            print("Loading BART model for framing analysis...")
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
            print("BART model loaded successfully")
        except Exception as e:
            error_str = str(e).lower()
            if any(mem_indicator in error_str for mem_indicator in 
                   ['paging file', 'memory', 'out of memory', 'cuda out of memory', 'not enough memory']):
                print(f"MEMORY WARNING: BART model unavailable due to insufficient memory")
                print(f"System will use lightweight fallback analysis for framing detection")
            else:
                print(f"BART model loading failed: {str(e)[:100]}...")
                print(f"Using fallback framing analysis")
            self.bart_model_failed = True
            self.zero_shot_classifier = None
        
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
            # Graceful fallback for memory constraints or other errors
            error_str = str(e).lower()
            if any(mem_indicator in error_str for mem_indicator in 
                   ['paging file', 'memory', 'out of memory', 'cuda out of memory']):
                print(f"  MEMORY CONSTRAINT: Using lightweight bias analysis")
            else:
                print(f"  Bias analysis error: {e}")
                
            return {
                'emotional_tone': {'emotion': 'neutral', 'emotion_score': 0.0},
                'framing_bias': {'framing_type': 'neutral', 'framing_score': 0.0},
                'political_stance': {'stance': 'neutral', 'stance_score': 0.0},
                'source_profile': {'has_profile': False},
                'combined_score': {'method': 'fallback', 'overall_bias': 0.0},
                'analysis_mode': 'lightweight_fallback'
            }
    
    def _analyze_text_bias(self, text: str) -> Dict:
        """Original text-based bias analysis with enhanced safety checks"""
        if not text or len(text.strip()) == 0:
            return self._default_bias_analysis()
        
        try:
            # More aggressive text truncation to prevent tensor issues
            # RoBERTa tokenizer creates ~1.3 tokens per word on average
            words = text.split()
            if len(words) > 50:  # Very conservative word limit
                text = ' '.join(words[:50])
            
            # Character limit as additional safety
            if len(text) > 400:  # Much more conservative character limit
                text = text[:400]
            
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
        """ML-based framing analysis with memory-aware fallback"""
        # Use pre-loaded BART model or fallback if unavailable
        if self.bart_model_failed or self.zero_shot_classifier is None:
            return self._analyze_framing_fallback(text)
        
        try:
            # Define framing categories
            framing_labels = [
                "neutral reporting",
                "emotional manipulation", 
                "propaganda language",
                "objective description"
            ]
            
            # Classify the text using pre-loaded model
            result = self.zero_shot_classifier(text, framing_labels)
            
            # Get the top prediction and confidence
            top_label = result['labels'][0]
            confidence = result['scores'][0]
            
            # Map to framing type and score
            if "emotional" in top_label or "propaganda" in top_label:
                return "emotional", confidence
            else:
                return "neutral", max(0.0, 1.0 - confidence)  # Low confidence in neutral = some bias
                
        except Exception as e:
            print(f"  ML framing analysis failed: {str(e)[:50]}...")
            # Use fallback method
            return self._analyze_framing_fallback(text)

    def _analyze_framing_fallback(self, text: str) -> tuple:
        """Lightweight fallback framing analysis for memory-constrained environments"""
        text_lower = text.lower()
        
        # Enhanced emotional/biased language detection
        emotional_words = [
            "crisis", "disaster", "scandal", "amazing", "terrible", 
            "shocking", "outrageous", "brilliant", "devastating", 
            "alarming", "unprecedented", "historic", "dramatic",
            "explosive", "controversial", "sensational", "urgent"
        ]
        
        # Bias indicator words
        bias_indicators = [
            "allegedly", "reportedly", "sources claim", "it is said",
            "critics argue", "supporters believe", "opponents claim"
        ]
        
        # Count occurrences
        emotional_count = sum(1 for word in emotional_words if word in text_lower)
        bias_count = sum(1 for indicator in bias_indicators if indicator in text_lower)
        
        # Calculate framing score
        total_bias_score = emotional_count * 0.15 + bias_count * 0.25
        
        if total_bias_score > 0.4:
            return "emotional", min(total_bias_score, 1.0)
        else:
            return "neutral", max(0.0, total_bias_score)
    
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
    




    def analyze_cross_source_framing(self, claim: str, all_evidence: List[Dict], all_bias_analyses: List[Dict]) -> Dict:
        """
        Compare framing patterns across ALL sources to identify systematic differences
        """
        try:
            # Extract key entities from the claim
            entities = self._extract_claim_entities(claim)
            
            # Analyze how each source describes these entities
            source_descriptions = {}
            source_emotional_scores = {}
            
            for i, (evidence, bias) in enumerate(zip(all_evidence, all_bias_analyses)):
                source_id = f"source_{i}"
                text = evidence.get('snippet', evidence.get('text', ''))
                
                # Extract entity descriptions
                descriptions = self._extract_entity_descriptions(text, entities)
                source_descriptions[source_id] = {
                    'descriptions': descriptions,
                    'full_text': text,
                    'source': evidence.get('source', 'unknown'),
                    'evidence_type': evidence.get('evidence_type', 'unknown')
                }
                
                # Get emotional intensity score
                emotional_score = abs(bias.get('emotional_tone', {}).get('emotion_score', 0.0))
                source_emotional_scores[source_id] = emotional_score
            
            # Calculate comparative analysis
            framing_analysis = self._compare_emotional_intensity(source_emotional_scores)
            semantic_analysis = self._compare_semantic_choices(source_descriptions, entities)
            
            return {
                'emotional_outliers': framing_analysis['outliers'],
                'emotional_baseline': framing_analysis['baseline'],
                'semantic_differences': semantic_analysis,
                'systematic_patterns': self._identify_systematic_patterns(source_descriptions, source_emotional_scores),
                'cross_source_summary': self._generate_cross_source_summary(framing_analysis, semantic_analysis)
            }
            
        except Exception as e:
            print(f"Cross-source framing analysis failed: {e}")
            return self._default_cross_source_analysis()


    def analyze_cross_source_framing_with_evidence(self, claim: str, all_evidence: List[Dict], all_bias_analyses: List[Dict]) -> Dict:
        """
        Enhanced cross-source analysis that captures specific examples of emotional language
        """
        try:
            # Get basic cross-source analysis
            base_analysis = self.analyze_cross_source_framing(claim, all_evidence, all_bias_analyses)
            
            # Extract specific emotional evidence for each outlier
            detailed_outlier_analysis = {}
            
            for outlier in base_analysis.get('emotional_outliers', []):
                source_idx = int(outlier['source_id'].split('_')[1])
                evidence = all_evidence[source_idx]
                bias_analysis = all_bias_analyses[source_idx]
                
                # Extract specific emotional evidence
                emotional_evidence = self._extract_emotional_evidence(
                    evidence.get('snippet', evidence.get('text', '')),
                    claim,
                    outlier['type']
                )
                
                # Analyze manipulation techniques
                manipulation_analysis = self._analyze_manipulation_techniques(
                    evidence.get('snippet', evidence.get('text', '')),
                    emotional_evidence
                )
                
                detailed_outlier_analysis[outlier['source_id']] = {
                    'source_name': evidence.get('source', 'Unknown'),
                    'evidence_type': evidence.get('evidence_type', 'unknown'),
                    'outlier_info': outlier,
                    'emotional_evidence': emotional_evidence,
                    'manipulation_analysis': manipulation_analysis,
                    'bias_explanation': self._generate_bias_explanation(emotional_evidence, manipulation_analysis)
                }
            
            # Add detailed analysis to base result
            base_analysis['detailed_outlier_analysis'] = detailed_outlier_analysis
            
            return base_analysis
            
        except Exception as e:
            print(f"Enhanced cross-source analysis failed: {e}")
            return self.analyze_cross_source_framing(claim, all_evidence, all_bias_analyses)

    def _extract_emotional_evidence(self, text: str, claim: str, outlier_type: str) -> Dict:
        """ML-based emotional evidence extraction"""
        
        emotional_evidence = {
            'emotional_words': [],
            'loaded_phrases': [],
            'emotion_scores': {},
            'sentiment_analysis': {},
            'context_appropriateness': {}
        }
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        try:
            # Use emotion model if available
            if self.emotion_model:
                # Analyze overall emotional content
                emotion_result = self.emotion_model(text)
                
                # Extract high-intensity emotions
                for result in emotion_result:
                    emotion_label = result['label'].lower()
                    emotion_score = result['score']
                    
                    if emotion_score > 0.3:  # Threshold for significant emotion
                        emotional_evidence['emotion_scores'][emotion_label] = emotion_score
            
            # Analyze each sentence for emotional content
            for sentence in sentences:
                if len(sentence) > 10:  # Skip very short sentences
                    try:
                        # Truncate sentence for sentiment analysis
                        if len(sentence) > 300:
                            sentence = sentence[:300]
                        
                        # Get sentiment for this sentence
                        sentiment_result = self.sentiment_model(sentence)[0]
                        sentiment_score = sentiment_result['score']
                        
                        # If high emotional intensity, extract it
                        if sentiment_score > 0.7:  # High confidence threshold
                            emotional_evidence['emotional_words'].append({
                                'sentence': sentence,
                                'sentiment': sentiment_result['label'],
                                'intensity': sentiment_score,
                                'context_appropriate': self._assess_emotional_appropriateness(sentence, claim)
                            })
                            
                    except Exception as e:
                        continue  # Skip problematic sentences
            
            # Extract loaded phrases using existing regex patterns
            loaded_phrase_patterns = [
                r'\b(complete|total|utter|absolute)\s+(failure|disaster|success|triumph)',
                r'\b(shocking|stunning|amazing|terrible|horrific|wonderful)\s+\w+',
                r'\b(unprecedented|historic|record-breaking|never-before-seen)\s+\w+',
                r'\b(desperately|urgently|critically|immediately)\s+need'
            ]
            
            import re
            for pattern in loaded_phrase_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    phrase = match.group()
                    for sentence in sentences:
                        if phrase.lower() in sentence.lower():
                            emotional_evidence['loaded_phrases'].append({
                                'phrase': phrase,
                                'context': sentence,
                                'manipulation_type': 'intensification'
                            })
                            break
            
        except Exception as e:
            print(f"ML emotional evidence extraction failed: {e}")
            # Fallback to the original hardcoded approach if needed
        
        return emotional_evidence


    def _assess_emotional_appropriateness(self, sentence: str, claim: str) -> Dict:
        """Assess whether emotional language is contextually appropriate"""
        
        try:
            # Classify claim context
            claim_context = self._classify_claim_context(claim)
            
            # Context-specific appropriate emotional terms
            appropriate_terms = {
                'military': ['attacked', 'struck', 'bombed', 'targeted', 'engaged', 'deployed'],
                'economic': ['declined', 'fell', 'rose', 'increased', 'decreased', 'fluctuated'],
                'political': ['debated', 'contested', 'opposed', 'supported', 'criticized'],
                'legal': ['convicted', 'acquitted', 'sentenced', 'charged', 'prosecuted']
            }
            
            sentence_lower = sentence.lower()
            context_terms = appropriate_terms.get(claim_context, [])
            
            # Check if emotional language is contextually appropriate
            has_appropriate_terms = any(term in sentence_lower for term in context_terms)
            
            # Check for inappropriate amplification
            amplification_terms = ['brutally', 'mercilessly', 'devastatingly', 'catastrophically']
            has_amplification = any(term in sentence_lower for term in amplification_terms)
            
            return {
                'context': claim_context,
                'appropriate': has_appropriate_terms and not has_amplification,
                'amplification_detected': has_amplification,
                'explanation': f"Context: {claim_context}. Appropriate terms used: {has_appropriate_terms}. Amplification: {has_amplification}"
            }
            
        except Exception as e:
            return {
                'context': 'unknown',
                'appropriate': True,  # Default to appropriate to avoid false flags
                'amplification_detected': False,
                'explanation': f"Assessment failed: {e}"
            }

    def _classify_claim_context(self, claim: str) -> str:
        """Classify the context/domain of the claim"""
        claim_lower = claim.lower()
        
        # Simple keyword-based classification
        if any(word in claim_lower for word in ['attack', 'military', 'war', 'defense', 'army', 'navy', 'bomb']):
            return 'military'
        elif any(word in claim_lower for word in ['economy', 'gdp', 'inflation', 'budget', 'financial', 'market']):
            return 'economic'
        elif any(word in claim_lower for word in ['election', 'parliament', 'government', 'minister', 'policy']):
            return 'political'
        elif any(word in claim_lower for word in ['court', 'legal', 'law', 'judge', 'conviction', 'trial']):
            return 'legal'
        else:
            return 'general'

    def _analyze_manipulation_techniques(self, text: str, emotional_evidence: Dict) -> Dict:
        """Analyze specific manipulation techniques being used"""
        
        manipulation_techniques = {
            'emotional_amplification': False,
            'selective_emphasis': False,
            'false_urgency': False,
            'loaded_language': False,
            'cherry_picking': False
        }
        
        explanations = []
        
        # Check for emotional amplification
        emotional_word_count = len(emotional_evidence.get('emotional_words', []))
        if emotional_word_count > 3:
            manipulation_techniques['emotional_amplification'] = True
            explanations.append(f"Uses excessive emotional language ({emotional_word_count} emotional words) to amplify impact")
        
        # Check for loaded language
        loaded_phrases = emotional_evidence.get('loaded_phrases', [])
        if loaded_phrases:
            manipulation_techniques['loaded_language'] = True
            phrase_examples = [p['phrase'] for p in loaded_phrases[:2]]
            explanations.append(f"Uses loaded phrases like '{', '.join(phrase_examples)}' to bias perception")
        
        # Check for superlatives overuse
        # Handle both old and new format for emotional words
        superlatives = []
        for w in emotional_evidence.get('emotional_words', []):
            if isinstance(w, dict):
                # Old format has 'category' field
                if 'category' in w and w['category'] == 'superlatives':
                    superlatives.append(w)
                # New format - check if sentence contains superlative words  
                elif 'sentence' in w:
                    superlative_words = ['worst', 'best', 'unprecedented', 'historic', 'massive', 'enormous']
                    if any(sup_word in w['sentence'].lower() for sup_word in superlative_words):
                        superlatives.append(w)
        
        if len(superlatives) > 1:
            manipulation_techniques['selective_emphasis'] = True
            # Extract words for explanation
            sup_examples = []
            for s in superlatives[:2]:
                if 'word' in s:
                    sup_examples.append(s['word'])
                elif 'sentence' in s:
                    # Extract the superlative word from sentence
                    superlative_words = ['worst', 'best', 'unprecedented', 'historic', 'massive', 'enormous']
                    for sup_word in superlative_words:
                        if sup_word in s['sentence'].lower():
                            sup_examples.append(sup_word)
                            break
            
            explanations.append(f"Overuses superlatives like '{', '.join(sup_examples)}' for dramatic effect")
        
        # Check for false urgency
        # Handle both old and new format for urgency words
        urgency_words = []
        for w in emotional_evidence.get('emotional_words', []):
            if isinstance(w, dict):
                # Old format has 'category' field
                if 'category' in w and w['category'] == 'urgency':
                    urgency_words.append(w)
                # New format - check if sentence contains urgency words
                elif 'sentence' in w:
                    urgency_terms = ['immediate', 'urgent', 'critical', 'emergency', 'must', 'cannot wait']
                    if any(urgency_term in w['sentence'].lower() for urgency_term in urgency_terms):
                        urgency_words.append(w)
        
        if urgency_words:
            manipulation_techniques['false_urgency'] = True
            explanations.append("Creates false sense of urgency to pressure reader reaction")
        
        return {
            'techniques_detected': manipulation_techniques,
            'explanations': explanations,
            'manipulation_score': sum(manipulation_techniques.values()) / len(manipulation_techniques)
        }

    def _generate_bias_explanation(self, emotional_evidence: Dict, manipulation_analysis: Dict) -> str:
        """Generate human-readable explanation of detected bias"""
        
        explanations = []
        
        # Explain emotional words
        emotional_words = emotional_evidence.get('emotional_words', [])
        if emotional_words:
            # Handle both old format (with 'word' and 'category') and new format (with 'sentence')
            high_intensity_items = []
            for w in emotional_words:
                if isinstance(w, dict):
                    if 'intensity' in w and w.get('intensity', 0) > 0.7:
                        # New ML format
                        if 'sentence' in w:
                            high_intensity_items.append(f"sentence: '{w['sentence'][:50]}...'")
                        # Old format fallback
                        elif 'word' in w:
                            high_intensity_items.append(f"'{w['word']}'")

            if high_intensity_items:
                explanations.append(f"Uses highly emotional language: {', '.join(high_intensity_items[:3])}")        
        # Explain loaded phrases
        loaded_phrases = emotional_evidence.get('loaded_phrases', [])
        if loaded_phrases:
            phrase_examples = [f"'{p['phrase']}'" for p in loaded_phrases[:2]]
            explanations.append(f"Employs loaded phrases such as {', '.join(phrase_examples)} to frame the narrative")
        
        # Add manipulation technique explanations
        explanations.extend(manipulation_analysis.get('explanations', []))
        
        if not explanations:
            return "Source uses emotionally neutral language appropriate for the content"
        
        return ". ".join(explanations) + "."

    def _calculate_word_intensity(self, word: str, category: str) -> float:
        """Calculate emotional intensity using model confidence rather than hardcoded values"""
        
        try:
            # Use sentiment model to get intensity
            context_sentence = f"This is {word}."  # Simple context for the word
            sentiment_result = self.sentiment_model(context_sentence)[0]
            
            # Use model confidence as intensity (0.5-1.0 range)
            base_intensity = sentiment_result['score']
            
            # Apply category multipliers for certain types
            category_multipliers = {
                'crisis_words': 1.0,
                'violence_words': 1.0, 
                'superlatives': 0.8,
                'urgency': 0.9,
                'certainty': 0.7
            }
            
            multiplier = category_multipliers.get(category, 0.8)
            return min(base_intensity * multiplier, 1.0)
            
        except Exception as e:
            # Fallback to hardcoded values
            intensity_mapping = {
                'crisis_words': {'crisis': 0.8, 'collapse': 0.9, 'disaster': 0.9, 'catastrophe': 1.0, 'devastation': 0.9, 'chaos': 0.7},
                'success_words': {'triumph': 0.9, 'success': 0.6, 'victory': 0.8, 'achievement': 0.5, 'breakthrough': 0.7, 'remarkable': 0.8},
                'violence_words': {'killed': 0.7, 'murdered': 0.9, 'slaughtered': 1.0, 'attacked': 0.6, 'violence': 0.7, 'brutal': 0.8},
                'superlatives': {'worst': 0.9, 'best': 0.8, 'unprecedented': 0.9, 'historic': 0.7, 'massive': 0.8, 'enormous': 0.8},
                'certainty': {'definitely': 0.7, 'clearly': 0.6, 'obviously': 0.8, 'undoubtedly': 0.9, 'certainly': 0.7, 'absolutely': 0.9},
                'urgency': {'immediate': 0.8, 'urgent': 0.9, 'critical': 0.9, 'emergency': 1.0, 'now': 0.6, 'must': 0.7}
            }
            return intensity_mapping.get(category, {}).get(word, 0.5)





    def _extract_claim_entities(self, claim: str) -> List[str]:
        """Extract key entities from claim for comparison"""
        import re
        
        # Basic entity extraction - you could use spaCy here for better results
        entities = []
        
        # Common entities in Sri Lankan political context
        political_entities = ['government', 'parliament', 'president', 'minister', 'MPs', 'cabinet', 
                            'opposition', 'party', 'election', 'policy', 'budget', 'economy']
        
        claim_lower = claim.lower()
        for entity in political_entities:
            if entity in claim_lower:
                entities.append(entity)
        
        # Extract proper nouns (basic approach)
        words = re.findall(r'\b[A-Z][a-z]+\b', claim)
        entities.extend(words)
        
        return list(set(entities))  # Remove duplicates

    def _extract_entity_descriptions(self, text: str, entities: List[str]) -> Dict:
        """Find how text describes each entity"""
        descriptions = {}
        text_lower = text.lower()
        
        for entity in entities:
            entity_lower = entity.lower()
            if entity_lower in text_lower:
                # Find sentences containing the entity
                sentences = text.split('.')
                relevant_sentences = [s.strip() for s in sentences if entity_lower in s.lower()]
                descriptions[entity] = relevant_sentences
        
        return descriptions

    def _compare_emotional_intensity(self, source_scores: Dict[str, float]) -> Dict:
        """Compare emotional intensity across sources"""
        if not source_scores:
            return {'baseline': 0.0, 'outliers': []}
        
        scores = list(source_scores.values())
        baseline = sum(scores) / len(scores)
        
        # Identify outliers (sources significantly above/below baseline)
        outliers = []
        threshold = 0.3  # Adjust as needed
        
        for source_id, score in source_scores.items():
            deviation = abs(score - baseline)
            if deviation > threshold:
                outlier_type = 'high_emotional' if score > baseline else 'low_emotional'
                outliers.append({
                    'source_id': source_id,
                    'score': score,
                    'deviation': deviation,
                    'type': outlier_type
                })
        
        return {
            'baseline': baseline,
            'outliers': outliers,
            'score_distribution': scores
        }

    def _compare_semantic_choices(self, source_descriptions: Dict, entities: List[str]) -> Dict:
        """Compare semantic choices across sources for same entities"""
        semantic_differences = {}
        
        # Define semantic intensity patterns
        intensity_patterns = {
            'violence': {
                'extreme': ['killed', 'murdered', 'assassinated', 'slaughtered', 'massacred'],
                'moderate': ['attacked', 'harmed', 'injured', 'hurt'],
                'neutral': ['died', 'deceased', 'passed away', 'lost life'],
                'euphemistic': ['fell', 'lost their lives', 'paid ultimate price']
            },
            'conflict': {
                'extreme': ['riot', 'chaos', 'violence', 'rampage', 'mayhem'],
                'moderate': ['protest', 'demonstration', 'unrest', 'disturbance'],
                'neutral': ['gathering', 'assembly', 'meeting', 'event'],
                'euphemistic': ['public order situation', 'incident']
            },
            'economy': {
                'extreme': ['collapsed', 'crashed', 'devastated', 'destroyed'],
                'moderate': ['declined', 'weakened', 'struggled', 'challenged'],
                'neutral': ['changed', 'adjusted', 'shifted', 'moved'],
                'euphemistic': ['adjusted', 'restructured', 'reformed']
            }
        }
        
        for entity in entities:
            entity_comparisons = {}
            
            for source_id, data in source_descriptions.items():
                descriptions = data['descriptions'].get(entity, [])
                if descriptions:
                    # Analyze word choices for this entity
                    intensity_level = self._classify_word_intensity(' '.join(descriptions), intensity_patterns)
                    entity_comparisons[source_id] = {
                        'descriptions': descriptions,
                        'intensity_level': intensity_level,
                        'source_type': data['evidence_type']
                    }
            
            if len(entity_comparisons) > 1:  # Only compare if multiple sources mention entity
                semantic_differences[entity] = entity_comparisons
        
        return semantic_differences

    def _classify_word_intensity(self, text: str, patterns: Dict) -> str:
        """Classify the intensity level of word choices in text"""
        text_lower = text.lower()
        intensity_scores = {'extreme': 0, 'moderate': 0, 'neutral': 0, 'euphemistic': 0}
        
        for category, levels in patterns.items():
            for level, words in levels.items():
                for word in words:
                    if word in text_lower:
                        intensity_scores[level] += 1
        
        # Return the level with highest score
        max_level = max(intensity_scores.items(), key=lambda x: x[1])
        return max_level[0] if max_level[1] > 0 else 'neutral'

    def _identify_systematic_patterns(self, source_descriptions: Dict, emotional_scores: Dict) -> Dict:
        """Identify systematic patterns in framing across source types"""
        patterns = {
            'database_vs_web': {'database_emotional_avg': 0, 'web_emotional_avg': 0},
            'government_vs_independent': {'government_sources': [], 'independent_sources': []},
            'consistent_framers': []
        }
        
        # Compare database vs web sources
        db_scores = []
        web_scores = []
        
        for source_id, data in source_descriptions.items():
            score = emotional_scores.get(source_id, 0)
            
            if data['evidence_type'] == 'database':
                db_scores.append(score)
            elif data['evidence_type'] == 'web':
                web_scores.append(score)
        
        if db_scores:
            patterns['database_vs_web']['database_emotional_avg'] = sum(db_scores) / len(db_scores)
        if web_scores:
            patterns['database_vs_web']['web_emotional_avg'] = sum(web_scores) / len(web_scores)
        
        # Identify sources with consistently high/low emotional language
        baseline = sum(emotional_scores.values()) / len(emotional_scores) if emotional_scores else 0
        
        for source_id, score in emotional_scores.items():
            if score > baseline + 0.3:
                patterns['consistent_framers'].append({
                    'source_id': source_id,
                    'pattern': 'high_emotional',
                    'score': score
                })
            elif score < baseline - 0.2:  # Different threshold for low emotional
                patterns['consistent_framers'].append({
                    'source_id': source_id,
                    'pattern': 'low_emotional', 
                    'score': score
                })
        
        return patterns

    def _generate_cross_source_summary(self, framing_analysis: Dict, semantic_analysis: Dict) -> Dict:
        """Generate summary of cross-source framing patterns"""
        summary = {
            'total_outliers': len(framing_analysis['outliers']),
            'emotional_baseline': framing_analysis['baseline'],
            'entities_with_different_framing': len(semantic_analysis),
            'framing_consistency': 'high' if len(framing_analysis['outliers']) <= 1 else 'low'
        }
        
        # Identify most problematic sources
        high_emotional_outliers = [o for o in framing_analysis['outliers'] if o['type'] == 'high_emotional']
        if high_emotional_outliers:
            summary['high_emotion_sources'] = len(high_emotional_outliers)
        
        return summary

    def _default_cross_source_analysis(self) -> Dict:
        """Default analysis when cross-source comparison fails"""
        return {
            'emotional_outliers': [],
            'emotional_baseline': 0.0,
            'semantic_differences': {},
            'systematic_patterns': {},
            'cross_source_summary': {'total_outliers': 0, 'framing_consistency': 'unknown'}
        }
