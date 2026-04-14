"""
Claim verification using Google Gemini AI API
Provides evidence-based fact-checking with structured responses
"""

import google.genai as genai
import os
import json
import time
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

@dataclass
class VerificationResult:
    """Structured verification result from Gemini"""
    verdict: str  # "SUPPORTS", "REFUTES", "INSUFFICIENT_EVIDENCE", "PARTIALLY_SUPPORTS"
    confidence: float  # 0.0 to 1.0
    explanation: str
    evidence_assessment: Dict[str, Any]
    reasoning_steps: List[str]
    limitations: List[str]

@dataclass
class BiasAnalysisResult:
    """Structured bias analysis result from Gemini"""
    overall_bias_assessment: str  # "HIGH", "MODERATE", "LOW", "MINIMAL"
    emotional_bias_score: float  # 0.0 to 1.0
    framing_bias_score: float  # 0.0 to 1.0
    detailed_analysis: Dict[str, Any]  # Per-source analysis
    cross_source_patterns: Dict[str, Any]  # Patterns across sources
    specific_examples: List[Dict]  # Concrete bias examples
    recommendations: List[str]  # User recommendations

class GeminiClaimVerifier:
    """Verify claims using Google Gemini AI with evidence context"""
    
    def __init__(self, api_key: Optional[str] = None, model_names: List[str] = None, max_retries: int = 3):
        """
        Initialize Gemini verifier with model fallback support
        
        Args:
            api_key: Google AI API key (if None, reads from GEMINI_API_KEY env var)
            model_names: List of Gemini models to try in order (fallback on rate limits)
            max_retries: Maximum number of retries per model for rate-limited requests
        """
        # Configure API
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Google AI API key required. Set GEMINI_API_KEY environment variable.")
        
        # Initialize client with new API
        self.client = genai.Client(api_key=api_key)
        
        # Model fallback configuration - using confirmed available models
        self.model_names = model_names or [
            "models/gemini-2.5-flash",          # Primary
            "models/gemini-2.5-flash-lite",     # Fallback 1: lighter 2.5, separate quota
            "models/gemini-3-flash-preview",    # Fallback 2: preview, separate quota
        ]
        self.current_model_index = 0  # Start with first model
        self.model_usage_stats = {model: 0 for model in self.model_names}
        
        self.max_retries = max_retries
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized Gemini verifier with {len(self.model_names)} fallback models: {self.model_names}")
        
        # Verification prompts
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for claim verification"""
        return """You are an expert fact-checker analyzing claims about Sri Lankan politics, economics, and current events.

Your task is to verify claims based on provided evidence and return structured assessments.

VERIFICATION LEVELS:
- SUPPORTS: Evidence clearly supports the claim
- REFUTES: Evidence clearly contradicts the claim  
- PARTIALLY_SUPPORTS: Evidence supports some parts but not others
- INSUFFICIENT_EVIDENCE: Not enough reliable evidence to make determination

CONFIDENCE SCALE (0.0 to 1.0):
- 0.9-1.0: Very high confidence (multiple authoritative sources)
- 0.7-0.8: High confidence (reliable sources with minor gaps)
- 0.5-0.6: Moderate confidence (mixed or limited evidence)
- 0.3-0.4: Low confidence (weak or contradictory evidence)
- 0.0-0.2: Very low confidence (insufficient reliable evidence)

EVIDENCE AUTHORITY RANKING:
1. Official government documents (Parliament Hansard, PMD releases, Cabinet decisions)
2. Legal documents (Supreme Court judgments, official gazettes)
3. Central Bank and Treasury reports
4. Established news organizations
5. Web sources and social media (lowest authority)

Always consider:
- Source reliability and potential bias
- Date relevance (recent vs outdated information)
- Context and nuance in statements
- Distinction between direct quotes vs paraphrasing
"""

    def _make_api_request_with_retry(self, prompt: str, operation_name: str = "API request") -> str:
        """
        Make API request with model fallback and exponential backoff for rate limiting
        
        Args:
            prompt: The prompt to send to Gemini
            operation_name: Name of the operation for logging
            
        Returns:
            Response text from Gemini
            
        Raises:
            Exception: If all models and retries are exhausted
        """
        # Try each model in the fallback list
        for model_index, model_name in enumerate(self.model_names):
            self.logger.info(f"Trying model {model_index + 1}/{len(self.model_names)}: {model_name} for {operation_name}")
            
            # Try current model with retries
            for attempt in range(self.max_retries + 1):
                try:
                    self.logger.debug(f"Model {model_name} attempt {attempt + 1}/{self.max_retries + 1} for {operation_name}")
                    
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    
                    # Success! Track usage and return
                    self.model_usage_stats[model_name] += 1
                    self.current_model_index = model_index  # Remember successful model for next request
                    
                    self.logger.info(f"✓ {operation_name} successful with {model_name} (usage: {self.model_usage_stats[model_name]})")
                    return response.text
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Check if it's a rate limit error
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        if attempt < self.max_retries:
                            # Retry with same model after delay
                            retry_delay = self._extract_retry_delay(error_str)
                            if retry_delay is None:
                                retry_delay = (2 ** attempt) + random.uniform(0, 1)
                            
                            self.logger.warning(f"Rate limit for {model_name}. Retrying in {retry_delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                            time.sleep(retry_delay)
                            continue
                        else:
                            # Exhausted retries for this model, try next model
                            self.logger.warning(f"Rate limit exhausted for {model_name} after {self.max_retries + 1} attempts. Trying next model...")
                            break  # Break to try next model
                    
                    # For non-rate-limit errors, don't retry this model
                    self.logger.error(f"Non-rate-limit error for {model_name}: {e}")
                    break  # Break to try next model
        
        # If we get here, all models failed
        usage_summary = ", ".join([f"{model.split('/')[-1]}: {count} requests" for model, count in self.model_usage_stats.items()])
        error_msg = f"All Gemini models exhausted for {operation_name}. Tried {len(self.model_names)} models with {self.max_retries + 1} attempts each. Usage today: {usage_summary}"
        
        self.logger.error(error_msg)
        raise Exception(f"All Gemini API models rate-limited or failed. Models tried: {[m.split('/')[-1] for m in self.model_names]}. Please wait for quota reset or upgrade your plan.")
    
    def _extract_retry_delay(self, error_message: str) -> Optional[float]:
        """
        Extract retry delay from Gemini error message
        
        Args:
            error_message: The error message from Gemini API
            
        Returns:
            Retry delay in seconds, or None if not found
        """
        try:
            # Look for patterns like "retry in 54.319346979s"
            import re
            match = re.search(r'retry in ([0-9]+\.?[0-9]*)s', error_message)
            if match:
                return float(match.group(1))
            
            # Look for retryDelay in JSON error details
            if 'retryDelay' in error_message:
                # Try to extract from JSON structure
                match = re.search(r'retryDelay["\']?:\s*["\']?([0-9]+)', error_message)
                if match:
                    return float(match.group(1))
            
        except (ValueError, AttributeError):
            pass
        
        return None
    
    def get_model_usage_stats(self) -> Dict[str, int]:
        """
        Get usage statistics for each model
        
        Returns:
            Dictionary mapping model names to request counts
        """
        return self.model_usage_stats.copy()
    
    def get_current_model(self) -> str:
        """
        Get the currently preferred model (last successful model)
        
        Returns:
            Current model name
        """
        return self.model_names[self.current_model_index]
    
    def reset_model_preference(self) -> None:
        """
        Reset to using the primary (first) model
        Useful for starting fresh after rate limit cooldowns
        """
        self.current_model_index = 0
        self.logger.info(f"Reset to primary model: {self.model_names[0]}")
    
    def list_available_models(self) -> List[str]:
        """
        List all available Gemini models for the current API version
        Useful for discovering which models can be used as fallbacks
        
        Returns:
            List of available model names
        """
        try:
            # Use the models.list endpoint to get available models
            models_response = self.client.models.list()
            
            available_models = []
            for model in models_response:
                model_name = model.name
                # Filter to generative models that support generateContent
                if 'generateContent' in getattr(model, 'supported_generation_methods', []):
                    available_models.append(model_name)
            
            self.logger.info(f"Found {len(available_models)} available models: {available_models}")
            return available_models
            
        except Exception as e:
            self.logger.error(f"Error listing available models: {e}")
            return []

    def verify_claim(self, claim: str, evidence_pieces: List[Dict]) -> VerificationResult:
        """
        Verify a claim against provided evidence
        
        Args:
            claim: The claim to verify
            evidence_pieces: List of evidence from hybrid retrieval
            
        Returns:
            VerificationResult with structured assessment
        """
        try:
            # Build verification prompt
            prompt = self._build_verification_prompt(claim, evidence_pieces)
            
            # Generate response with retry logic
            self.logger.info(f"Verifying claim with Gemini: {claim[:100]}...")
            
            # Use API request with retry logic
            response_text = self._make_api_request_with_retry(prompt, "claim verification")
            
            # Parse structured response
            result = self._parse_verification_response(response_text)
            
            self.logger.info(f"Verification complete: {result.verdict} (confidence: {result.confidence})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during Gemini verification: {e}")
            return self._create_error_result(str(e))
    
    def _build_verification_prompt(self, claim: str, evidence_pieces: List[Dict]) -> str:
        """Build detailed verification prompt with evidence"""
        
        # Organize evidence by source authority
        evidence_by_authority = self._organize_evidence_by_authority(evidence_pieces)
        
        prompt = f"""
CLAIM TO VERIFY:
"{claim}"

AVAILABLE EVIDENCE:
{self._format_evidence_for_prompt(evidence_by_authority)}

ANALYSIS REQUIRED:
1. Evaluate each piece of evidence for relevance and reliability
2. Look for direct quotes, official statements, or documented facts
3. Consider source bias and authority levels
4. Assess temporal relevance (when was this said/reported?)
5. Check for context that might change interpretation

Provide your response in this exact JSON format:
{{
    "verdict": "SUPPORTS|REFUTES|PARTIALLY_SUPPORTS|INSUFFICIENT_EVIDENCE",
    "confidence": 0.0-1.0,
    "explanation": "Detailed explanation of your assessment",
    "evidence_assessment": {{
        "strongest_supporting": "Description of best supporting evidence",
        "strongest_contradicting": "Description of best contradicting evidence", 
        "source_reliability": "Assessment of source quality",
        "temporal_relevance": "How recent/relevant is the evidence"
    }},
    "reasoning_steps": [
        "Step 1: What you checked first",
        "Step 2: What you verified next",
        "Step 3: How you reached conclusion"
    ],
    "limitations": [
        "What information is missing",
        "What assumptions were made",
        "What could change this assessment"
    ]
}}
"""
        return prompt
    
    def _organize_evidence_by_authority(self, evidence_pieces: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize evidence by authority level"""
        authority_groups = {
            "Official Government (Authority: 1.0)": [],
            "Legal/Judicial (Authority: 0.9-1.0)": [],
            "Economic/Financial (Authority: 0.8-0.9)": [],
            "News Media (Authority: 0.6-0.8)": [],
            "Web Sources (Authority: <0.6)": []
        }
        
        for evidence in evidence_pieces:
            authority = evidence.get('authority', 0.5)
            dataset = evidence.get('dataset', 'unknown')
            
            if authority >= 0.95 and dataset in ['hansard', 'pmd', 'cabinet']:
                authority_groups["Official Government (Authority: 1.0)"].append(evidence)
            elif authority >= 0.9:
                authority_groups["Legal/Judicial (Authority: 0.9-1.0)"].append(evidence)
            elif authority >= 0.8:
                authority_groups["Economic/Financial (Authority: 0.8-0.9)"].append(evidence)
            elif authority >= 0.6:
                authority_groups["News Media (Authority: 0.6-0.8)"].append(evidence)
            else:
                authority_groups["Web Sources (Authority: <0.6)"].append(evidence)
        
        return authority_groups
    
    def _format_evidence_for_prompt(self, evidence_by_authority: Dict[str, List[Dict]]) -> str:
        """Format evidence for the prompt"""
        formatted = []
        
        for authority_level, evidence_list in evidence_by_authority.items():
            if not evidence_list:
                continue
            
            formatted.append(f"\n=== {authority_level} ===")
            
            for i, evidence in enumerate(evidence_list, 1):
                title = evidence.get('title', 'No title')
                text = evidence.get('passage', evidence.get('text', ''))[:500]
                date = evidence.get('date', 'Unknown date')
                source = evidence.get('source', 'Unknown source')
                score = evidence.get('score', 0)
                
                formatted.append(f"""
{i}. Title: {title}
   Source: {source} | Date: {date} | Relevance: {score:.3f}
   Content: {text}...
""")
        
        return "\n".join(formatted)
    
    def _parse_verification_response(self, response_text: str) -> VerificationResult:
        """Parse Gemini's JSON response into VerificationResult"""
        try:
            # Extract JSON from response (in case there's extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_text = response_text[json_start:json_end]
            data = json.loads(json_text)
            
            # Validate required fields
            required_fields = ['verdict', 'confidence', 'explanation', 'evidence_assessment', 'reasoning_steps', 'limitations']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate verdict values
            valid_verdicts = ['SUPPORTS', 'REFUTES', 'PARTIALLY_SUPPORTS', 'INSUFFICIENT_EVIDENCE']
            if data['verdict'] not in valid_verdicts:
                raise ValueError(f"Invalid verdict: {data['verdict']}")
            
            # Validate confidence range
            confidence = float(data['confidence'])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got: {confidence}")
            
            return VerificationResult(
                verdict=data['verdict'],
                confidence=confidence,
                explanation=data['explanation'],
                evidence_assessment=data['evidence_assessment'],
                reasoning_steps=data['reasoning_steps'],
                limitations=data['limitations']
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"Error parsing Gemini response: {e}")
            # Return fallback result
            return VerificationResult(
                verdict="INSUFFICIENT_EVIDENCE",
                confidence=0.1,
                explanation=f"Error parsing AI response: {str(e)}",
                evidence_assessment={"error": "Could not parse structured response"},
                reasoning_steps=["Error in response parsing"],
                limitations=["AI response could not be properly parsed"]
            )
    
    def _create_error_result(self, error_message: str) -> VerificationResult:
        """Create error result when verification fails"""
        # Provide more helpful error messages for common issues
        if "rate limit" in error_message.lower() or "429" in error_message or "All Gemini" in error_message:
            explanation = f"Unable to verify claim due to API rate limits across all available models. {error_message}"
            limitations = [
                "All Gemini models rate-limited - verification temporarily unavailable",
                f"Models tried: {[m.split('/')[-1] for m in self.model_names]}",
                "Each model has separate quotas (typically 20 requests/day for free tier)",
                "Consider upgrading to paid plan for higher limits or wait for quota reset"
            ]
        else:
            explanation = f"Verification failed due to error: {error_message}"
            limitations = ["Could not complete verification due to technical error"]
        
        return VerificationResult(
            verdict="INSUFFICIENT_EVIDENCE",
            confidence=0.0,
            explanation=explanation,
            evidence_assessment={"error": error_message},
            reasoning_steps=["Error occurred during verification process"],
            limitations=limitations
        )
    
    def batch_verify(self, claims_with_evidence: List[Dict]) -> List[VerificationResult]:
        """
        Verify multiple claims in batch
        
        Args:
            claims_with_evidence: List of dicts with 'claim' and 'evidence' keys
            
        Returns:
            List of VerificationResults
        """
        results = []
        
        for item in claims_with_evidence:
            claim = item['claim']
            evidence = item['evidence']
            
            result = self.verify_claim(claim, evidence)
            results.append(result)
        
        return results
    
    def explain_bias_patterns(self, claim: str, evidence_pieces: List[Dict], bias_analyses: List[Dict] = None, temporal_context: Dict = None) -> BiasAnalysisResult:
        """
        Use Gemini to identify and explain emotional/framing biases in evidence sources
        
        Args:
            claim: The original claim being fact-checked
            evidence_pieces: List of evidence from hybrid retrieval
            bias_analyses: Optional ML-based bias analysis results
            temporal_context: Information about breaking news and temporal factors
            
        Returns:
            BiasAnalysisResult with detailed bias explanations including temporal warnings
        """
        try:
            # Build bias analysis prompt with temporal context
            prompt = self._build_bias_analysis_prompt(claim, evidence_pieces, bias_analyses, temporal_context)
            
            self.logger.info(f"Analyzing bias patterns with Gemini for claim: {claim[:100]}...")
            
            # Generate response with retry logic
            response_text = self._make_api_request_with_retry(prompt, "bias analysis")
            
            # Parse structured response
            result = self._parse_bias_response(response_text)
            
            self.logger.info(f"Bias analysis complete: {result.overall_bias_assessment} bias level")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during Gemini bias analysis: {e}")
            return self._create_error_bias_result(str(e))
    
    def _build_bias_analysis_prompt(self, claim: str, evidence_pieces: List[Dict], bias_analyses: List[Dict] = None, temporal_context: Dict = None) -> str:
        """Build detailed bias analysis prompt with temporal awareness"""
        
        # Organize evidence by type
        evidence_by_type = self._organize_evidence_by_type(evidence_pieces)
        
        # Build temporal warning section for bias analysis
        temporal_warning = ""
        if temporal_context:
            is_breaking = temporal_context.get('is_breaking_news', False)
            recency_level = temporal_context.get('recency_level', 'historical')
            hours_old = temporal_context.get('estimated_hours_old', 0)
            freshness_score = temporal_context.get('content_freshness_score', 0)
            
            if is_breaking or recency_level in ['immediate', 'recent'] or freshness_score > 0.5:
                temporal_warning = f"""

CRITICAL TEMPORAL CONTEXT:
- This is BREAKING/RECENT NEWS (estimated {hours_old} hours old)
- Content freshness: {freshness_score:.2f}/1.0
- Evidence is from very recent sources (social media, breaking news)
- Information is PRELIMINARY and rapidly evolving

TEMPORAL FACTORS TO EMPHASIZE IN BIAS ANALYSIS:
1. Breaking news often has higher emotional intensity - this may be situational, not just bias
2. Social media sources in breaking news context have different standards than established journalism
3. Rapid reporting can lead to less careful language choices
4. Users need to understand this is DEVELOPING STORY with preliminary information
5. Recommend caution and checking back for updates as story evolves

YOUR RECOMMENDATIONS MUST EMPHASIZE:
- This is breaking news - higher uncertainty expected
- Information is preliminary and may change
- Users should be extra cautious with breaking news claims
- Story is still developing - check back for updates
"""
        
        prompt = f"""
You are an expert media analyst specializing in identifying emotional manipulation and framing bias in news sources and official communications.

ORIGINAL CLAIM:
"{claim}"
{temporal_warning}

SOURCE EVIDENCE TO ANALYZE:
{self._format_evidence_for_bias_analysis(evidence_by_type)}

BIAS ANALYSIS FRAMEWORK:

1. EMOTIONAL MANIPULATION:
   - Loaded language (inflammatory adjectives, emotional appeals)
   - Sensationalism vs factual reporting
   - Fear-mongering or excessive optimism
   - Inappropriate emotional intensity for the context

2. FRAMING BIAS:
   - Selective emphasis on certain aspects
   - Omission of relevant context
   - Characterization of events/people in biased ways
   - Different framings of the same facts across sources

3. CROSS-SOURCE PATTERNS:
   - Government vs independent media differences
   - Systematic patterns in language choices
   - Consistency or inconsistency in framing approaches

For each source, identify:
- Specific examples of biased language (quote exact phrases)
- The emotional/framing technique being used
- Why it's problematic or appropriate
- How it differs from other sources

Provide your analysis in this exact JSON format:
{{
    "overall_bias_assessment": "HIGH|MODERATE|LOW|MINIMAL",
    "emotional_bias_score": 0.0-1.0,
    "framing_bias_score": 0.0-1.0,
    "detailed_analysis": {{
        "source_1": {{
            "source_name": "Name of source",
            "evidence_type": "government|web|news",
            "emotional_techniques": ["List of techniques used"],
            "framing_techniques": ["List of framing approaches"],
            "bias_examples": [
                {{
                    "quote": "Exact quote from source",
                    "technique": "Specific bias technique",
                    "explanation": "Why this is biased",
                    "better_alternative": "How it could be phrased neutrally"
                }}
            ],
            "bias_score": 0.0-1.0,
            "appropriateness_assessment": "Explanation of context appropriateness"
        }}
    }},
    "cross_source_patterns": {{
        "government_vs_independent": "Analysis of systematic differences",
        "emotional_intensity_comparison": "Which sources use more emotional language",
        "framing_consistency": "Whether sources frame similarly or differently",
        "credibility_impact": "How bias affects source credibility"
    }},
    "specific_examples": [
        {{
            "comparison_type": "Same fact, different framing",
            "source_a": "Source A said: 'quote'",
            "source_b": "Source B said: 'quote'",
            "bias_difference": "Explanation of the bias difference"
        }}
    ],
    "recommendations": [
        "Actionable advice for users about source reliability",
        "How to interpret biased sources",
        "What to watch out for"
    ]
}}
"""
        return prompt
    
    def _organize_evidence_by_type(self, evidence_pieces: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize evidence by source type for bias analysis"""
        type_groups = {
            "Government Sources": [],
            "Independent News": [],
            "Web Sources": []
        }
        
        for evidence in evidence_pieces:
            evidence_type = evidence.get('evidence_type', 'unknown')
            dataset = evidence.get('dataset', evidence.get('dataset_source', 'unknown'))
            
            if evidence_type == 'database' or dataset in ['hansard', 'pmd', 'cabinet_decisions']:
                type_groups["Government Sources"].append(evidence)
            elif evidence_type == 'web' and evidence.get('authority', 0) > 0.6:
                type_groups["Independent News"].append(evidence)
            else:
                type_groups["Web Sources"].append(evidence)
        
        return type_groups
    
    def _format_evidence_for_bias_analysis(self, evidence_by_type: Dict[str, List[Dict]]) -> str:
        """Format evidence specifically for bias analysis"""
        formatted = []
        source_counter = 1
        
        for source_type, evidence_list in evidence_by_type.items():
            if not evidence_list:
                continue
            
            formatted.append(f"\n=== {source_type} ===")
            
            for evidence in evidence_list:
                title = evidence.get('title', 'No title')
                text = evidence.get('snippet', evidence.get('passage', evidence.get('text', '')))
                source = evidence.get('source', 'Unknown source')
                
                # Truncate text for analysis
                if len(text) > 800:
                    text = text[:800] + "..."
                
                formatted.append(f"""
Source {source_counter}: {source}
Title: {title}
Content: {text}
""")
                source_counter += 1
        
        return "\n".join(formatted)
    
    def _parse_bias_response(self, response_text: str) -> BiasAnalysisResult:
        """Parse Gemini's bias analysis JSON response"""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_text = response_text[json_start:json_end]
            data = json.loads(json_text)
            
            # Validate required fields
            required_fields = ['overall_bias_assessment', 'emotional_bias_score', 'framing_bias_score', 
                             'detailed_analysis', 'cross_source_patterns', 'specific_examples', 'recommendations']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate assessment values
            valid_assessments = ['HIGH', 'MODERATE', 'LOW', 'MINIMAL']
            if data['overall_bias_assessment'] not in valid_assessments:
                raise ValueError(f"Invalid bias assessment: {data['overall_bias_assessment']}")
            
            return BiasAnalysisResult(
                overall_bias_assessment=data['overall_bias_assessment'],
                emotional_bias_score=float(data['emotional_bias_score']),
                framing_bias_score=float(data['framing_bias_score']),
                detailed_analysis=data['detailed_analysis'],
                cross_source_patterns=data['cross_source_patterns'],
                specific_examples=data['specific_examples'],
                recommendations=data['recommendations']
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"Error parsing Gemini bias response: {e}")
            return BiasAnalysisResult(
                overall_bias_assessment="MINIMAL",
                emotional_bias_score=0.1,
                framing_bias_score=0.1,
                detailed_analysis={"error": "Could not parse bias analysis"},
                cross_source_patterns={"error": str(e)},
                specific_examples=[],
                recommendations=["Could not analyze bias due to parsing error"]
            )
    
    def _create_error_bias_result(self, error_message: str) -> BiasAnalysisResult:
        """Create error result when bias analysis fails"""
        # Provide more helpful error messages for rate limits
        if "rate limit" in error_message.lower() or "429" in error_message or "All Gemini" in error_message:
            recommendations = [
                "Bias analysis temporarily unavailable - all Gemini models rate-limited",
                f"Models attempted: {[m.split('/')[-1] for m in self.model_names]}",
                "Each model has separate quotas (free tier: ~20 requests/day per model)",
                "Wait for quota reset or upgrade to paid plan for higher limits",
                "Manual review recommended: Look for emotional language and source framing differences"
            ]
            error_context = "All models quota exceeded"
        else:
            recommendations = ["Bias analysis could not be completed due to technical error"]
            error_context = "Analysis failed"
        
        return BiasAnalysisResult(
            overall_bias_assessment="MINIMAL",
            emotional_bias_score=0.0,
            framing_bias_score=0.0,
            detailed_analysis={"error": error_message},
            cross_source_patterns={"error": error_context},
            specific_examples=[],
            recommendations=recommendations
        )
    
    def get_verification_summary(self, result: VerificationResult) -> str:
        """Generate human-readable summary of verification result"""
        verdict_markers = {
            "SUPPORTS": "[SUPPORTED]",
            "REFUTES": "[REFUTED]", 
            "PARTIALLY_SUPPORTS": "[PARTIALLY SUPPORTED]",
            "INSUFFICIENT_EVIDENCE": "[INSUFFICIENT EVIDENCE]"
        }
        
        confidence_level = (
            "Very High" if result.confidence >= 0.9 else
            "High" if result.confidence >= 0.7 else
            "Moderate" if result.confidence >= 0.5 else
            "Low" if result.confidence >= 0.3 else
            "Very Low"
        )
        
        summary = f"""
{verdict_markers.get(result.verdict, '[UNKNOWN]')} VERDICT: {result.verdict}
CONFIDENCE: {confidence_level} ({result.confidence:.2f})

EXPLANATION:
{result.explanation}

KEY REASONING:
{chr(10).join(f"- {step}" for step in result.reasoning_steps)}

LIMITATIONS:
{chr(10).join(f"- {limitation}" for limitation in result.limitations)}
"""
        return summary.strip()
    
    def get_bias_summary(self, result: BiasAnalysisResult) -> str:
        """Generate human-readable summary of bias analysis result"""
        bias_markers = {
            "HIGH": "HIGH BIAS DETECTED",
            "MODERATE": "MODERATE BIAS",
            "LOW": "LOW BIAS",
            "MINIMAL": "MINIMAL BIAS"
        }
        
        marker = bias_markers.get(result.overall_bias_assessment, result.overall_bias_assessment)
        
        summary = f"""
{marker}
EMOTIONAL BIAS: {result.emotional_bias_score:.1%} | FRAMING BIAS: {result.framing_bias_score:.1%}

CROSS-SOURCE PATTERNS:
- Government vs Independent: {result.cross_source_patterns.get('government_vs_independent', 'No comparison available')}
- Emotional Intensity: {result.cross_source_patterns.get('emotional_intensity_comparison', 'No analysis available')}
- Framing Consistency: {result.cross_source_patterns.get('framing_consistency', 'No analysis available')}

SPECIFIC BIAS EXAMPLES:
{chr(10).join(f"- {example.get('comparison_type', 'Unknown')}: {example.get('bias_difference', 'No description')}" for example in result.specific_examples[:3])}

RECOMMENDATIONS:
{chr(10).join(f"- {rec}" for rec in result.recommendations[:3])}
"""
        return summary.strip()
    
    def explain_weighting_decisions(self, claim: str, weighted_evidence: List[Dict], final_verdict: Dict, temporal_context: Dict = None, claim_bias: Dict = None) -> str:
        """
        Generate natural language explanations for why sources received specific weights.
        claim_bias: dict of claim-level bias dimensions from ClaimAwareBiasAnalyzer.
        """
        try:
            # Build explanation prompt with temporal context
            prompt = self._build_weighting_explanation_prompt(claim, weighted_evidence, final_verdict, temporal_context, claim_bias)

            # Get Gemini's explanation with retry logic
            response_text = self._make_api_request_with_retry(prompt, "weighting explanation")

            return response_text.strip()

        except Exception as e:
            return f"Error generating weighting explanations: {str(e)}"

    def stream_explanation(self, claim: str, weighted_evidence: List[Dict], final_verdict: Dict, temporal_context: Dict = None, claim_bias: Dict = None):
        """
        Stream the explanation token-by-token using Gemini's streaming API.
        Yields text chunks as they arrive so the frontend can display them immediately.
        Falls back to the blocking method if streaming fails.
        """
        prompt = self._build_weighting_explanation_prompt(claim, weighted_evidence, final_verdict, temporal_context, claim_bias)

        for model_name in self.model_names:
            try:
                for chunk in self.client.models.generate_content_stream(
                    model=model_name,
                    contents=prompt,
                ):
                    if chunk.text:
                        yield chunk.text
                return  # success — stop trying other models
            except Exception as e:
                self.logger.warning(f"Streaming failed for {model_name}: {e}")
                continue

        # All streaming attempts failed — fall back to blocking call
        yield self.explain_weighting_decisions(claim, weighted_evidence, final_verdict, temporal_context, claim_bias)


    def _build_weighting_explanation_prompt(self, claim: str, weighted_evidence: List[Dict], final_verdict: Dict, temporal_context: Dict = None, claim_bias: Dict = None) -> str:
        """Build prompt for explaining weighting decisions with temporal awareness"""
        
        # Format weighted evidence for explanation
        evidence_summary = []
        for i, item in enumerate(weighted_evidence, 1):
            evidence = item['evidence']
            weight = item['weight']
            bias_alignment = item['bias_alignment']
            verification = item['verification']
            
            source_name = evidence.get('source', 'Unknown Source')
            source_type = evidence.get('evidence_type', 'unknown')
            verdict = verification.get('label', 'unknown')
            confidence = verification.get('confidence', 0)
            
            actual_content = evidence.get('content', evidence.get('snippet', ''))
            content_preview = actual_content[:400].strip() if actual_content else '[no content retrieved]'

            evidence_summary.append(f"""
Source {i}: {source_name}
  - Type: {source_type} source
  - Verdict: {verdict} ({confidence:.1%} confidence)
  - Final Weight: {weight:.3f}
  - Bias Alignment: {bias_alignment:.3f}
  - Actual content: "{content_preview}"
""")
        
        evidence_text = "\n".join(evidence_summary)
        
        # Build temporal warning section
        temporal_warning = ""
        if temporal_context:
            is_breaking = temporal_context.get('is_breaking_news', False)
            recency_level = temporal_context.get('recency_level', 'historical')
            hours_old = temporal_context.get('estimated_hours_old', 0)
            freshness_score = temporal_context.get('content_freshness_score', 0)
            
            if is_breaking or recency_level in ['immediate', 'recent'] or freshness_score > 0.5:
                temporal_warning = f"""

CRITICAL TEMPORAL NOTE:
- This is BREAKING/RECENT NEWS (estimated {hours_old} hours old)
- Content freshness score: {freshness_score:.2f}/1.0  
- Recency level: {recency_level}
- Sources are very fresh and preliminary
- Story is likely still developing
- Information may be incomplete or changing rapidly

YOU MUST EMPHASIZE IN YOUR EXPLANATION:
1. This verdict is HIGHLY PRELIMINARY due to breaking news nature
2. Users should be CAUTIOUS and expect updates
3. Evidence is temporal and may evolve quickly
4. Recommend checking back for updates as story develops
"""
        
        # Build claim bias section
        claim_bias_section = ""
        if claim_bias:
            bias_lines = []

            # Framing and tone
            framing = claim_bias.get("framing_type", "")
            tone = claim_bias.get("emotional_tone", None)
            political = claim_bias.get("political_direction", "")
            if framing:
                bias_lines.append(f"- Framing type: {framing}")
            if tone is not None:
                tone_label = "positive" if tone > 0.3 else "negative" if tone < -0.3 else "neutral"
                bias_lines.append(f"- Emotional tone: {tone_label} ({tone:+.2f})")
            if political:
                bias_lines.append(f"- Political direction: {political}")

            # Loaded phrases
            loaded_phrases = claim_bias.get("loaded_phrases", [])
            if loaded_phrases:
                phrase_list = ", ".join(f'"{p}"' for p in loaded_phrases[:5])
                bias_lines.append(f"- Loaded/charged language: {phrase_list}")

            # Sri Lankan-specific flags
            sl_flags = []
            if claim_bias.get("gender_bias_signal"):
                sl_flags.append("gender-first framing or gender stereotyping")
            if claim_bias.get("ethnic_bias_signal"):
                sl_flags.append("ethnic/communal framing")
            if claim_bias.get("trauma_trivialization"):
                sl_flags.append("trivialisation of historical trauma")
            if sl_flags:
                bias_lines.append(f"- Sri Lankan-specific bias patterns detected: {'; '.join(sl_flags)}")

            # LLaMA explanation of the claim's bias
            llm_explanation = claim_bias.get("explanation", "")
            if llm_explanation:
                bias_lines.append(f"- AI bias analysis: {llm_explanation}")

            if bias_lines:
                claim_bias_section = "\n\nCLAIM-LEVEL BIAS ANALYSIS:\nThe fact-checking system also analysed the claim itself for bias:\n" + "\n".join(bias_lines) + "\n"

        prompt = f"""
You are an expert in explaining automated fact-checking algorithms to users.

ORIGINAL CLAIM:
"{claim}"

WEIGHTING ALGORITHM RESULTS:
{evidence_text}

FINAL VERDICT:
- Decision: {final_verdict.get('verdict', 'Unknown')}
- Confidence: {final_verdict.get('confidence', 0):.1%}
- Support: {final_verdict.get('support_score', 0):.1%}
- Refute: {final_verdict.get('refute_score', 0):.1%}
{temporal_warning}{claim_bias_section}

ALGORITHM EXPLANATION TASK:
Your job is to explain in simple terms WHY each source received its specific weight. For each key source, show:

1. **What the source actually said** (quote directly from the "Actual content" field above — do NOT infer or fabricate quotes)
2. **How NLI interpreted it** (what verdict + confidence it gave)
3. **Bias-claim alignment calculation** (why the bias score was high/low)
4. **Final weight explanation** (how all factors combined)

Focus on 3-4 most influential sources as examples. Show the actual content that drove the algorithm's decisions.

EXAMPLE FORMAT FOR EACH SOURCE:
"Source X (government website) said: 'The Standing Orders clearly state that the Speaker has authority to remove DSGs without parliamentary approval.'

NLI Analysis: This directly contradicts the claim (confidence: 95%) because it states the Speaker DOES have authority.

Bias Alignment: As a government source, this statement goes against typical government bias (alignment: 0.15), so it gets higher weight.

Final Weight: 1.05 (high authority × high confidence × low bias penalty)"

Keep it educational and show how the research algorithm works step-by-step. Use real examples from the sources above.
{f'''
IMPORTANT — CLAIM BIAS EXPLANATION:
After explaining the source weights, add a dedicated section titled "## Bias in the Claim Itself" that explains to the reader:
- What biases or loaded language were detected in the claim as written
- Why these patterns matter for accurate fact-checking
- How this may influence how the claim is interpreted (e.g., gender-first framing foregrounds gender over other attributes; loaded language can create emotional bias before evidence is considered)
Use plain, accessible language so a general reader can understand.
''' if claim_bias_section else ''}"""
        return prompt

def main():
    """Test Gemini claim verifier with model fallback"""
    verifier = GeminiClaimVerifier()
    
    # Check available models first
    print("Checking available Gemini models...")
    available_models = verifier.list_available_models()
    print(f"Available models: {available_models}")
    
    # Show current configuration
    print(f"\nConfigured models: {[m.split('/')[-1] for m in verifier.model_names]}")
    print(f"Current preferred model: {verifier.get_current_model().split('/')[-1]}")
    print(f"Usage stats: {verifier.get_model_usage_stats()}")
    
    # Test claim
    # claim = "President said giving rice to animals causes shortage"
    claim = "funds from the previous year’s budget are available for disaster-related spending without requiring fresh parliamentary approval"
    
    # Mock evidence (in real usage, this comes from hybrid retrieval)
    evidence = [
        {
            'dataset': 'hansard',
            'authority': 1.0,
            'title': 'Parliamentary Session - Food Security Discussion',
            'passage': 'The President mentioned that pets consuming rice has increased and affects food security calculations.',
            'date': '2024-01-15',
            'source': 'Parliament of Sri Lanka',
            'score': 0.85
        },
        {
            'dataset': 'news',
            'authority': 0.7,
            'title': 'President Comments on Rice Shortage',
            'passage': 'In a press briefing, officials discussed various factors contributing to rice shortage including animal feed usage.',
            'date': '2024-01-16',
            'source': 'Daily Mirror',
            'score': 0.72
        }
    ]
    
    # Verify claim
    result = verifier.verify_claim(claim, evidence)
    
    # Display results
    print("=" * 80)
    print("GEMINI CLAIM VERIFICATION RESULTS")
    print("=" * 80)
    print(f"Claim: {claim}")
    print("\n" + verifier.get_verification_summary(result))
    
    # Show model usage after verification
    print("\n" + "=" * 80)
    print("MODEL USAGE STATISTICS")
    print("=" * 80)
    for model, count in verifier.get_model_usage_stats().items():
        model_short = model.split('/')[-1]
        print(f"{model_short}: {count} requests")
    print(f"\nCurrent preferred model: {verifier.get_current_model().split('/')[-1]}")

if __name__ == "__main__":
    main()