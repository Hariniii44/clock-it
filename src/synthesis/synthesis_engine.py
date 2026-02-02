from groq import Groq
from typing import List, Dict, Any
import json

class SynthesisEngine:
    def __init__(self, groq_api_key: str):
        """Initialize synthesis engine with Groq client"""
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.3-70b-versatile"  # High-quality model for synthesis
    
    def generate_synthesis(self, claim: str, evidence: List[Dict], 
                          verification_results: List[Dict], 
                          bias_analyses: List[Dict], 
                          final_verdict: Dict) -> str:
        """
        Generate Perplexity-style synthesis with bias awareness
        """
        
        # Format evidence for the prompt
        evidence_summary = self._format_evidence_for_prompt(
            evidence, verification_results, bias_analyses
        )
        
        # Create bias context summary
        bias_context = self._format_bias_context(bias_analyses)
        
        # Generate synthesis prompt
        prompt = self._create_synthesis_prompt(
            claim, evidence_summary, bias_context, final_verdict
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Balanced creativity and accuracy
                max_tokens=300,   # 2-3 sentences as planned
                top_p=0.9
            )
            
            synthesis = response.choices[0].message.content.strip()
            return self._clean_synthesis_output(synthesis)
            
        except Exception as e:
            print(f"⚠️ Synthesis generation failed: {e}")
            return self._fallback_synthesis(claim, final_verdict)
    
    def _format_evidence_for_prompt(self, evidence: List[Dict], 
                                   verification_results: List[Dict], 
                                   bias_analyses: List[Dict]) -> str:
        """Format evidence with verification and bias info for LLM"""
        
        evidence_lines = []
        
        for i, (e, verification, bias) in enumerate(zip(evidence, verification_results, bias_analyses), 1):
            # Extract key information
            source = e.get('source', 'Unknown Source')
            title = e.get('title', 'No title')[:60]
            verdict = verification.get('label', 'UNKNOWN')
            confidence = verification.get('confidence', 0)
            
            # Get bias information
            bias_info = self._extract_bias_summary(bias)
            
            # Format evidence entry
            evidence_lines.append(
                f"{i}. {source}: \"{title}...\"\n"
                f"   Verification: {verdict} ({confidence:.0%} confidence)\n"
                f"   Bias: {bias_info}"
            )
        
        return "\n\n".join(evidence_lines)
    
    def _extract_bias_summary(self, bias_analysis: Dict) -> str:
        """Extract concise bias summary from analysis"""
        
        # Check if source profile exists
        if "source_profile" in bias_analysis and bias_analysis["source_profile"]["has_profile"]:
            profile = bias_analysis["source_profile"]
            return f"{profile['bias_interpretation']} ({profile['bias_score']:+.1f})"
        
        # Use real-time analysis
        political = bias_analysis.get('political_stance', {})
        stance = political.get('political_stance', 'unknown')
        score = political.get('stance_score', 0)
        
        return f"{stance} ({score:+.1f})"
    
    def _format_bias_context(self, bias_analyses: List[Dict]) -> str:
        """Create bias context summary"""
        
        profiled_sources = 0
        bias_types = {"opposition": 0, "pro-government": 0, "neutral": 0}
        
        for bias in bias_analyses:
            if "source_profile" in bias and bias["source_profile"]["has_profile"]:
                profiled_sources += 1
                score = bias["source_profile"]["bias_score"]
                
                if score < -10:
                    bias_types["opposition"] += 1
                elif score > 10:
                    bias_types["pro-government"] += 1
                else:
                    bias_types["neutral"] += 1
        
        context_parts = [f"{profiled_sources}/{len(bias_analyses)} sources have bias profiles"]
        
        if profiled_sources > 0:
            active_biases = [f"{count} {bias_type}" for bias_type, count in bias_types.items() if count > 0]
            if active_biases:
                context_parts.append(f"Distribution: {', '.join(active_biases)}")
        
        return "; ".join(context_parts)
    
    def _create_synthesis_prompt(self, claim: str, evidence_summary: str, 
                               bias_context: str, final_verdict: Dict) -> str:
        """Create the synthesis prompt for the LLM"""
        
        prompt = f"""You are an expert fact-checker creating a concise synthesis of evidence about a claim.

CLAIM TO VERIFY: "{claim}"

EVIDENCE ANALYSIS:
{evidence_summary}

BIAS CONTEXT: {bias_context}

FACT-CHECK VERDICT: {final_verdict['verdict']} ({final_verdict['confidence']:.0%} confidence)

Your task: Write a clear, factual 2-3 sentence synthesis that:

1. States what the evidence shows about the claim's accuracy
2. Notes key discrepancies or patterns between sources  
3. Mentions significant bias concerns if they affect the conclusion
4. Maintains neutrality while being informative

Guidelines:
- Be direct and factual, not speculative
- If sources disagree, explain the disagreement briefly
- If bias patterns are significant, mention them naturally
- If evidence is limited/unclear, state that honestly
- Use accessible language, avoid jargon

Focus on what a reader needs to know about this claim's accuracy based on the available evidence."""

        return prompt
    
    def _clean_synthesis_output(self, synthesis: str) -> str:
        """Clean and format the synthesis output"""
        
        # Remove any unwanted prefixes/suffixes
        synthesis = synthesis.strip()
        
        # Remove common LLM artifacts
        unwanted_prefixes = [
            "Based on the evidence,", "According to the analysis,", 
            "The synthesis shows that", "In summary,", "To summarize,"
        ]
        
        for prefix in unwanted_prefixes:
            if synthesis.startswith(prefix):
                synthesis = synthesis[len(prefix):].strip()
        
        # Ensure proper capitalization
        if synthesis and synthesis[0].islower():
            synthesis = synthesis[0].upper() + synthesis[1:]
        
        return synthesis
    
    def _fallback_synthesis(self, claim: str, final_verdict: Dict) -> str:
        """Fallback synthesis if LLM fails"""
        
        verdict = final_verdict['verdict'].lower()
        confidence = final_verdict['confidence']
        
        if verdict == 'supported':
            return f"Available evidence supports the claim with {confidence:.0%} confidence, though source limitations may apply."
        elif verdict == 'refuted':
            return f"Available evidence contradicts the claim with {confidence:.0%} confidence, though source limitations may apply."
        else:
            return f"Evidence is insufficient or conflicting to verify the claim, resulting in {confidence:.0%} uncertainty."