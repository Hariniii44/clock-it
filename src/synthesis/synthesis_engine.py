from groq import Groq
from typing import List, Dict, Any
import json

class SynthesisEngine:
    def __init__(self, groq_api_key: str):
        """Initialize synthesis engine with Groq client"""
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.3-70b-versatile"  # High-quality model for synthesis
    
    # def generate_synthesis(self, claim: str, evidence: List[Dict], 
    #                       verification_results: List[Dict], 
    #                       bias_analyses: List[Dict], 
    #                       final_verdict: Dict) -> str:
    #     """
    #     Generate Perplexity-style synthesis with bias awareness
    #     """
        
    #     # Format evidence for the prompt
    #     evidence_summary = self._format_evidence_for_prompt(
    #         evidence, verification_results, bias_analyses
    #     )
        
    #     # Create bias context summary
    #     bias_context = self._format_bias_context(bias_analyses)
        
    #     # Generate synthesis prompt
    #     prompt = self._create_synthesis_prompt(
    #         claim, evidence_summary, bias_context, final_verdict
    #     )
        
    #     try:
    #         response = self.client.chat.completions.create(
    #             model=self.model,
    #             messages=[{"role": "user", "content": prompt}],
    #             temperature=0.3,  # Balanced creativity and accuracy
    #             max_tokens=300,   # 2-3 sentences as planned
    #             top_p=0.9
    #         )
            
    #         synthesis = response.choices[0].message.content.strip()
    #         return self._clean_synthesis_output(synthesis)
            
    #     except Exception as e:
    #         print(f"⚠️ Synthesis generation failed: {e}")
    #         return self._fallback_synthesis(claim, final_verdict)

    def generate_synthesis(self, claim: str, evidence: List[Dict], 
                      verification_results: List[Dict], 
                      bias_analyses: List[Dict], 
                      final_verdict: Dict) -> str:
        """
        Generate BOTH AI synthesis AND enhanced bias-aware explanations
        """
        
        # Generate AI synthesis first (the original one)
        ai_synthesis = self._generate_ai_synthesis(
            claim, evidence, verification_results, bias_analyses, final_verdict
        )
        
        # Generate detailed bias-aware analysis
        bias_analysis = self._create_bias_aware_analysis(
            claim, evidence, verification_results, bias_analyses, final_verdict
        )
        
        # Combine both
        combined_output = f"{ai_synthesis}\n\n{'='*60}\nVERDICTS EXPLANATION\n{'='*60}\n{bias_analysis}"
        
        return combined_output

    def _generate_ai_synthesis(self, claim: str, evidence: List[Dict], 
                            verification_results: List[Dict], 
                            bias_analyses: List[Dict], 
                            final_verdict: Dict) -> str:
        """
        Generate the original AI synthesis using Groq
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
                temperature=0.3,
                max_tokens=300,
                top_p=0.9
            )
            
            synthesis = response.choices[0].message.content.strip()
            return self._clean_synthesis_output(synthesis)
            
        except Exception as e:
            print(f"⚠️ Synthesis generation failed: {e}")
            return self._fallback_synthesis(claim, final_verdict)

    # def _create_bias_aware_analysis(self, claim: str, evidence: List[Dict], 
    #                             verification_results: List[Dict], 
    #                             bias_analyses: List[Dict], 
    #                             final_verdict: Dict) -> str:
    #     """
    #     Create detailed bias-aware analysis with source categorization
    #     """
        
    #     # DEBUG: Check input data
    #     print(f"DEBUG: Processing {len(evidence)} evidence pieces")
    #     print(f"DEBUG: Processing {len(verification_results)} verification results") 
    #     print(f"DEBUG: Processing {len(bias_analyses)} bias analyses")

    #     # Categorize sources by verdict and bias
    #     supporting_sources = []
    #     refuting_sources = []
    #     neutral_sources = []
        
    #     for i, (evidence_item, verification, bias) in enumerate(zip(evidence, verification_results, bias_analyses)):
    #         print(f"DEBUG: Processing source {i+1}: {evidence_item.get('source', 'Unknown')}")

    #         try:
    #             source_info = self._extract_source_info(evidence_item, bias)
    #             source_info['verification'] = verification
                
    #             verdict = verification.get('label', 'NEUTRAL')
    #             print(f"DEBUG: Verdict for {source_info['name']}: {verdict}")
                
    #             if verdict == 'SUPPORTED':
    #                 supporting_sources.append(source_info)
    #             elif verdict == 'REFUTED':
    #                 refuting_sources.append(source_info)
    #             else:
    #                 neutral_sources.append(source_info)
                    
    #         except Exception as e:
    #             print(f"DEBUG ERROR processing source {i+1}: {e}")

    #     print(f"DEBUG: Final counts - Support: {len(supporting_sources)}, Refute: {len(refuting_sources)}, Neutral: {len(neutral_sources)}")

    #         # source_info = self._extract_source_info(evidence_item, bias)
    #         # source_info['verification'] = verification
            
    #         # verdict = verification.get('label', 'NEUTRAL')
    #         # if verdict == 'SUPPORTED':
    #         #     supporting_sources.append(source_info)
    #         # elif verdict == 'REFUTED':
    #         #     refuting_sources.append(source_info)
    #         # else:
    #         #     neutral_sources.append(source_info)
        
    #     # Generate the human-readable analysis
    #     analysis_parts = []
        
    #     # Header with main verdict
    #     confidence = final_verdict.get('confidence', 0) * 100
    #     verdict = final_verdict.get('verdict', 'UNCERTAIN').upper()
    #     analysis_parts.append(f"VERDICT: {verdict} ({confidence:.0f}% confidence)")
    #     analysis_parts.append("")
        
    #     # Supporting sources section with detailed bias info
    #     if supporting_sources:
    #         analysis_parts.append(f"SOURCES SUPPORTING ({len(supporting_sources)}):")
    #         for source in supporting_sources:
    #             line = f"  {source['name']} ({source['bias_label']}) {source['reliability_icon']}"
                
    #             # Add emotional/framing context if significant
    #             bias_context = self._get_bias_context_summary(source)
    #             if bias_context:
    #                 line += f" {bias_context}"
                    
    #             analysis_parts.append(line)
    #         analysis_parts.append("")
        
    #     # Refuting sources section with detailed bias info
    #     if refuting_sources:
    #         analysis_parts.append(f"SOURCES REFUTING ({len(refuting_sources)}):")
    #         for source in refuting_sources:
    #             line = f"  {source['name']} ({source['bias_label']}) {source['reliability_icon']}"
                
    #             # Add emotional/framing context if significant
    #             bias_context = self._get_bias_context_summary(source)
    #             if bias_context:
    #                 line += f" {bias_context}"
                    
    #             analysis_parts.append(line)
    #         analysis_parts.append("")
        
    #     # Neutral sources (if significant)
    #     if len(neutral_sources) > 2:
    #         analysis_parts.append(f"NEUTRAL/INCONCLUSIVE ({len(neutral_sources)}):")
    #         for source in neutral_sources[:3]:  # Show top 3
    #             line = f"  {source['name']} ({source['bias_label']}) {source['reliability_icon']}"
                
    #             # Add bias context for neutrals too
    #             bias_context = self._get_bias_context_summary(source)
    #             if bias_context:
    #                 line += f" {bias_context}"
                    
    #             analysis_parts.append(line)
    #         if len(neutral_sources) > 3:
    #             analysis_parts.append(f"  ... and {len(neutral_sources) - 3} more")
    #         analysis_parts.append("")
        
    #     # Enhanced bias alert section
    #     bias_warning = self._generate_bias_warning(supporting_sources, refuting_sources, claim)
    #     if bias_warning:
    #         analysis_parts.append("⚠️ BIAS ALERT:")
    #         analysis_parts.extend(bias_warning)
    #         analysis_parts.append("")
        
    #     # Recommendation with bias context
    #     recommendation = self._generate_recommendation(
    #         claim, supporting_sources, refuting_sources, final_verdict
    #     )
    #     analysis_parts.append("RECOMMENDATION:")
    #     analysis_parts.append(recommendation)
        
    #     return "\n".join(analysis_parts)

    def _create_bias_aware_analysis(self, claim: str, evidence: List[Dict], 
                               verification_results: List[Dict], 
                               bias_analyses: List[Dict], 
                               final_verdict: Dict) -> str:
        """
        Create detailed bias-aware analysis showing only sources with identified bias
        """
        
        # Categorize sources by verdict and bias
        supporting_sources = []
        refuting_sources = []
        neutral_sources = []
        
        for i, (evidence_item, verification, bias) in enumerate(zip(evidence, verification_results, bias_analyses)):
            source_info = self._extract_source_info(evidence_item, bias)
            source_info['verification'] = verification
            
            verdict = verification.get('label', 'NEUTRAL')
            
            # Only include sources with IDENTIFIED political bias (not unknown)
            has_identified_bias = (
                'pro-government' in source_info['bias_label'].lower() or 
                'opposition' in source_info['bias_label'].lower() or
                'official' in source_info['bias_label'].lower() or
                'academic' in source_info['bias_label'].lower()
            )
            
            if has_identified_bias:
                if verdict == 'SUPPORTED':
                    supporting_sources.append(source_info)
                elif verdict == 'REFUTED':
                    refuting_sources.append(source_info)
                else:
                    neutral_sources.append(source_info)
        
        # Generate the human-readable analysis
        analysis_parts = []
        
        # Header with main verdict
        confidence = final_verdict.get('confidence', 0) * 100
        verdict = final_verdict.get('verdict', 'UNCERTAIN').upper()
        analysis_parts.append(f"VERDICT: {verdict} ({confidence:.0f}% confidence)")
        analysis_parts.append("")
        
        # Supporting sources section - show ALL with bias
        if supporting_sources:
            analysis_parts.append(f"SOURCES SUPPORTING ({len(supporting_sources)}):")
            for source in supporting_sources:
                line = f"  {source['name']} ({source['bias_label']}) {source['reliability_icon']}"
                bias_context = self._get_bias_context_summary(source)
                if bias_context:
                    line += f" {bias_context}"
                analysis_parts.append(line)
            analysis_parts.append("")
        
        # Refuting sources section - show ALL with bias
        if refuting_sources:
            analysis_parts.append(f"SOURCES REFUTING ({len(refuting_sources)}):")
            for source in refuting_sources:
                line = f"  {source['name']} ({source['bias_label']}) {source['reliability_icon']}"
                bias_context = self._get_bias_context_summary(source)
                if bias_context:
                    line += f" {bias_context}"
                analysis_parts.append(line)
            analysis_parts.append("")
        
        # Neutral sources - show ALL with bias (no "...and more")
        if neutral_sources:
            analysis_parts.append(f"NEUTRAL/INCONCLUSIVE ({len(neutral_sources)}):")
            for source in neutral_sources:
                line = f"  {source['name']} ({source['bias_label']}) {source['reliability_icon']}"
                bias_context = self._get_bias_context_summary(source)
                if bias_context:
                    line += f" {bias_context}"
                analysis_parts.append(line)
            analysis_parts.append("")
        
        # If no sources with identified bias at all
        if not supporting_sources and not refuting_sources and not neutral_sources:
            analysis_parts.append("NO SOURCES WITH IDENTIFIED POLITICAL BIAS")
            analysis_parts.append("All sources show unknown political stance")
            analysis_parts.append("")
        
        # Enhanced bias alert section
        bias_warning = self._generate_enhanced_bias_warning(
            supporting_sources, refuting_sources, neutral_sources, claim
        )
        if bias_warning:
            analysis_parts.append("⚠️ BIAS ALERT:")
            analysis_parts.extend(bias_warning)
            analysis_parts.append("")
        
        # Recommendation with bias context
        recommendation = self._generate_recommendation(
            claim, supporting_sources, refuting_sources, final_verdict
        )
        analysis_parts.append("RECOMMENDATION:")
        analysis_parts.append(recommendation)
        
        return "\n".join(analysis_parts)

    # def _get_bias_context_summary(self, source: Dict) -> str:
    #     """Get brief bias context for display"""
    #     context_parts = []
        
    #     bias_details = source.get('bias_details', {})
        
    #     # Emotional context
    #     if 'emotional' in bias_details:
    #         emotional = bias_details['emotional']
    #         if emotional['tone'] != 'neutral' and abs(emotional['score']) > 0.5:
    #             context_parts.append(f"[{emotional['tone']} tone]")
        
    #     # Framing context
    #     if 'framing' in bias_details:
    #         framing = bias_details['framing']
    #         if framing['type'] != 'neutral' and abs(framing['score']) > 0.3:
    #             context_parts.append(f"[{framing['type']} framing]")
        
    #     return " ".join(context_parts) if context_parts else ""

    # def _extract_source_info(self, evidence_item: Dict, bias_analysis: Dict) -> Dict:
    #     """Extract comprehensive source information with political and emotional bias"""
        
    #     source_name = evidence_item.get('source', 'Unknown Source')
    #     clean_name = source_name.replace('www.', '').replace('.lk', '').replace('.com', '')
        
    #     # Initialize source info
    #     source_info = {
    #         'name': clean_name.title(),
    #         'full_name': source_name,
    #         'bias_details': {}
    #     }
        
    #     # Get political bias from profile
    #     if "source_profile" in bias_analysis and bias_analysis["source_profile"]["has_profile"]:
    #         profile = bias_analysis["source_profile"]
    #         bias_score = profile["bias_score"]
            
    #         # Categorize political bias
    #         if bias_score < -30:
    #             political_label = "Strong Opposition"
    #             reliability_icon = "🔴 High Bias Risk"
    #         elif bias_score < -15:
    #             political_label = "Opposition-leaning"
    #             reliability_icon = "⚠️ Bias Warning"
    #         elif bias_score < -5:
    #             political_label = "Slightly Opposition"
    #             reliability_icon = "⚠️ Minor Bias"
    #         elif bias_score > 30:
    #             political_label = "Strong Pro-Government"
    #             reliability_icon = "🔴 High Bias Risk"
    #         elif bias_score > 15:
    #             political_label = "Pro-Government"
    #             reliability_icon = "⚠️ Bias Warning"
    #         elif bias_score > 5:
    #             political_label = "Slightly Pro-Government"
    #             reliability_icon = "⚠️ Minor Bias"
    #         else:
    #             political_label = "Politically Neutral"
    #             reliability_icon = "✓ Balanced"
                
    #         source_info['bias_label'] = political_label
    #         source_info['reliability_icon'] = reliability_icon
    #         source_info['bias_score'] = bias_score
    #         source_info['bias_details']['political'] = {
    #             'stance': political_label,
    #             'score': bias_score,
    #             'confidence': profile.get('confidence', 0)
    #         }
            
    #     else:
    #         # Check source type for reliability
    #         if any(official in source_name.lower() for official in ['gov.lk', 'statistics', 'parliament', 'cbsl']):
    #             source_info['bias_label'] = "Official Source"
    #             source_info['reliability_icon'] = "🏛️ Government"
    #         elif any(academic in source_name.lower() for academic in ['researchgate', 'fao', 'worldbank']):
    #             source_info['bias_label'] = "Academic/Research"
    #             source_info['reliability_icon'] = "📚 Reliable"
    #         else:
    #             source_info['bias_label'] = "Unknown Political Stance"
    #             source_info['reliability_icon'] = "? Unknown"
        
    #     # Add emotional tone analysis
    #     if 'emotional_tone' in bias_analysis:
    #         emotional = bias_analysis['emotional_tone']
    #         tone = emotional.get('tone', 'neutral')
    #         tone_score = emotional.get('score', 0)
            
    #         source_info['bias_details']['emotional'] = {
    #             'tone': tone,
    #             'score': tone_score
    #         }
        
    #     # Add framing analysis
    #     if 'framing' in bias_analysis:
    #         framing = bias_analysis['framing']
    #         frame_type = framing.get('framing_type', 'neutral')
    #         frame_score = framing.get('score', 0)
            
    #         source_info['bias_details']['framing'] = {
    #             'type': frame_type,
    #             'score': frame_score
    #         }
        
    #     return source_info

    # def _generate_bias_warning(self, supporting_sources: List[Dict], 
    #                       refuting_sources: List[Dict], claim: str) -> List[str]:
    #     """Generate comprehensive bias warnings including emotional and framing bias"""
        
    #     warnings = []
        
    #     # Count political bias patterns
    #     support_gov = sum(1 for s in supporting_sources if 'pro-government' in s['bias_label'].lower())
    #     support_opp = sum(1 for s in supporting_sources if 'opposition' in s['bias_label'].lower())
    #     refute_gov = sum(1 for s in refuting_sources if 'pro-government' in s['bias_label'].lower())
    #     refute_opp = sum(1 for s in refuting_sources if 'opposition' in s['bias_label'].lower())
        
    #     # Political bias warnings
    #     if support_gov >= 2:
    #         warnings.append(f"• {support_gov} pro-government sources support this claim")
    #     if refute_opp >= 2:
    #         warnings.append(f"• {refute_opp} opposition sources refute this claim")
    #     if support_opp >= 2:
    #         warnings.append(f"• {support_opp} opposition sources support this claim")
    #     if refute_gov >= 2:
    #         warnings.append(f"• {refute_gov} pro-government sources refute this claim")
        
    #     # Emotional tone analysis
    #     all_sources = supporting_sources + refuting_sources
    #     negative_sources = []
    #     positive_sources = []
        
    #     for source in all_sources:
    #         if 'emotional' in source.get('bias_details', {}):
    #             emotional = source['bias_details']['emotional']
    #             if emotional['tone'] == 'negative' and emotional['score'] < -0.5:
    #                 negative_sources.append(source['name'])
    #             elif emotional['tone'] == 'positive' and emotional['score'] > 0.5:
    #                 positive_sources.append(source['name'])
        
    #     if len(negative_sources) >= 3:
    #         warnings.append(f"• High emotional negativity detected in {len(negative_sources)} sources")
    #     if len(positive_sources) >= 2:
    #         warnings.append(f"• Positive emotional framing in {len(positive_sources)} sources")
        
    #     # Framing analysis
    #     problematic_framing = []
    #     for source in all_sources:
    #         if 'framing' in source.get('bias_details', {}):
    #             framing = source['bias_details']['framing']
    #             if framing['type'] != 'neutral' and abs(framing['score']) > 0.3:
    #                 problematic_framing.append(f"{source['name']} ({framing['type']})")
        
    #     if problematic_framing:
    #         warnings.append(f"• Biased framing detected: {', '.join(problematic_framing)}")
        
    #     # Look for contradictory official sources
    #     official_support = [s for s in supporting_sources if 'official' in s['bias_label'].lower()]
    #     official_refute = [s for s in refuting_sources if 'official' in s['bias_label'].lower()]
        
    #     if official_support and official_refute:
    #         warnings.append("• Official sources disagree - conflicting government data")
        
    #     # Claim-specific contextual warnings
    #     if 'government' in claim.lower() or 'president' in claim.lower() or 'minister' in claim.lower():
    #         if support_gov > 0:
    #             warnings.append("• Government sources may overstate achievements on political claims")
    #         if refute_opp > 0:
    #             warnings.append("• Opposition sources may overstate criticism of government")
        
    #     if 'economic' in claim.lower() or 'imf' in claim.lower() or 'budget' in claim.lower():
    #         if support_gov > 0:
    #             warnings.append("• Government sources may emphasize positive economic indicators")
        
    #     # High bias risk warning
    #     high_risk_sources = [s for s in all_sources if '🔴' in s.get('reliability_icon', '')]
    #     if high_risk_sources:
    #         source_names = [s['name'] for s in high_risk_sources]
    #         warnings.append(f"• High bias risk sources: {', '.join(source_names)}")
        
    #     return warnings

    # def _generate_recommendation(self, claim: str, supporting_sources: List[Dict], 
    #                         refuting_sources: List[Dict], final_verdict: Dict) -> str:
    #     """Generate contextual recommendation"""
        
    #     verdict = final_verdict.get('verdict', 'UNCERTAIN').upper()
    #     confidence = final_verdict.get('confidence', 0) * 100
        
    #     # Count reliable vs biased sources
    #     reliable_support = sum(1 for s in supporting_sources if '✓' in s['reliability_icon'])
    #     reliable_refute = sum(1 for s in refuting_sources if '✓' in s['reliability_icon'])
    #     biased_support = len(supporting_sources) - reliable_support
    #     biased_refute = len(refuting_sources) - reliable_refute
        
    #     if verdict == 'SUPPORTED':
    #         if reliable_support > biased_support:
    #             return f"The claim appears well-supported by reliable sources. Confidence is high ({confidence:.0f}%)."
    #         else:
    #             return f"Claim has support, but mainly from potentially biased sources. Exercise caution."
                
    #     elif verdict == 'REFUTED':
    #         if reliable_refute > biased_refute:
    #             return f"The claim is contradicted by reliable sources. Evidence suggests it's false."
    #         else:
    #             return f"Claim is disputed, but mainly by potentially biased sources. Seek additional verification."
                
    #     else:  # UNCERTAIN
    #         if len(supporting_sources) == len(refuting_sources):
    #             return f"Evidence is split. Both sides have roughly equal support. More investigation needed."
    #         elif biased_support > 0 or biased_refute > 0:
    #             return f"Insufficient reliable evidence. Many sources show potential bias. Seek neutral sources."
    #         else:
    #             return f"Limited evidence available. Cannot verify with current sources ({confidence:.0f}% uncertainty)."
    
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

    # def _extract_source_info(self, evidence_item: Dict, bias_analysis: Dict) -> Dict:
    #     """Extract comprehensive source information with political and emotional bias"""
        
    #     source_name = evidence_item.get('source', 'Unknown Source')
    #     clean_name = source_name.replace('www.', '').replace('.lk', '').replace('.com', '')
        
    #     # Initialize source info
    #     source_info = {
    #         'name': clean_name.title(),
    #         'full_name': source_name,
    #         'bias_details': {},
    #         'bias_score': 0
    #     }
        
    #     # Get political bias from profile
    #     if "source_profile" in bias_analysis and bias_analysis["source_profile"]["has_profile"]:
    #         profile = bias_analysis["source_profile"]
    #         bias_score = profile.get("bias_score", 0)
            
    #         # Categorize political bias
    #         if bias_score < -30:
    #             political_label = "Strong Opposition"
    #             reliability_icon = "🔴 High Bias Risk"
    #         elif bias_score < -15:
    #             political_label = "Opposition-leaning"
    #             reliability_icon = "⚠️ Bias Warning"
    #         elif bias_score < -5:
    #             political_label = "Slightly Opposition"
    #             reliability_icon = "⚠️ Minor Bias"
    #         elif bias_score > 30:
    #             political_label = "Strong Pro-Government"
    #             reliability_icon = "🔴 High Bias Risk"
    #         elif bias_score > 15:
    #             political_label = "Pro-Government"
    #             reliability_icon = "⚠️ Bias Warning"
    #         elif bias_score > 5:
    #             political_label = "Slightly Pro-Government"
    #             reliability_icon = "⚠️ Minor Bias"
    #         else:
    #             political_label = "Politically Neutral"
    #             reliability_icon = "✓ Balanced"
                
    #         source_info['bias_label'] = political_label
    #         source_info['reliability_icon'] = reliability_icon
    #         source_info['bias_score'] = bias_score
    #         source_info['bias_details']['political'] = {
    #             'stance': political_label,
    #             'score': bias_score,
    #             'confidence': profile.get('confidence', 0)
    #         }
            
    #     else:
    #         # Check source type for reliability
    #         if any(official in source_name.lower() for official in ['gov.lk', 'statistics', 'parliament', 'cbsl']):
    #             source_info['bias_label'] = "Official Source"
    #             source_info['reliability_icon'] = "🏛️ Government"
    #         elif any(academic in source_name.lower() for academic in ['researchgate', 'fao', 'worldbank']):
    #             source_info['bias_label'] = "Academic/Research"
    #             source_info['reliability_icon'] = "📚 Reliable"
    #         else:
    #             source_info['bias_label'] = "Unknown Political Stance"
    #             source_info['reliability_icon'] = "? Unknown"
        
    #     # Add emotional tone analysis
    #     if 'emotional_tone' in bias_analysis:
    #         emotional = bias_analysis['emotional_tone']
    #         tone = emotional.get('tone', 'neutral')
    #         tone_score = emotional.get('score', 0)
            
    #         source_info['bias_details']['emotional'] = {
    #             'tone': tone,
    #             'score': tone_score
    #         }
        
    #     # Add framing analysis
    #     if 'framing' in bias_analysis:
    #         framing = bias_analysis['framing']
    #         frame_type = framing.get('framing_type', 'neutral')
    #         frame_score = framing.get('score', 0)
            
    #         source_info['bias_details']['framing'] = {
    #             'type': frame_type,
    #             'score': frame_score
    #         }
        
    #     return source_info

    def _extract_source_info(self, evidence_item: Dict, bias_analysis: Dict) -> Dict:
        """Extract source information, prioritizing sources with known political bias"""
        
        source_name = evidence_item.get('source', 'Unknown Source')
        clean_name = source_name.replace('www.', '').replace('.lk', '').replace('.com', '').title()
        
        # Initialize source info
        source_info = {
            'name': clean_name,
            'full_name': source_name,
            'bias_details': {},
            'bias_score': 0
        }
        
        # First Priority: Source profile (known Sri Lankan media bias)
        if "source_profile" in bias_analysis and bias_analysis["source_profile"]["has_profile"]:
            profile = bias_analysis["source_profile"]
            bias_score = profile.get("bias_score", 0)
            
            # Detailed political bias categorization
            if bias_score <= -40:
                political_label = "Strong Opposition"
                reliability_icon = "🔴 High Bias Risk"
            elif bias_score <= -20:
                political_label = "Opposition-leaning"
                reliability_icon = "⚠️ Bias Warning"
            elif bias_score <= -5:
                political_label = "Slightly Opposition"
                reliability_icon = "⚠️ Minor Bias"
            elif bias_score >= 40:
                political_label = "Strong Pro-Government"
                reliability_icon = "🔴 High Bias Risk"
            elif bias_score >= 20:
                political_label = "Pro-Government"
                reliability_icon = "⚠️ Bias Warning"
            elif bias_score >= 5:
                political_label = "Slightly Pro-Government"
                reliability_icon = "⚠️ Minor Bias"
            else:
                political_label = "Politically Neutral"
                reliability_icon = "✓ Balanced"
                
            source_info['bias_label'] = political_label
            source_info['reliability_icon'] = reliability_icon
            source_info['bias_score'] = bias_score
            
        # Second Priority: Official/Academic sources (high reliability)
        elif any(official in source_name.lower() for official in [
            'gov.lk', 'statistics', 'parliament', 'cbsl', 'treasury', 'president.gov'
        ]):
            source_info['bias_label'] = "Official Government"
            source_info['reliability_icon'] = "🏛️ Government Source"
            
        elif any(academic in source_name.lower() for academic in [
            'researchgate', 'fao', 'worldbank', 'imf', 'un.org', 'harti'
        ]):
            source_info['bias_label'] = "Academic/Research"
            source_info['reliability_icon'] = "📚 Academic Source"
            
        # Third Priority: International news/organizations
        elif any(intl in source_name.lower() for intl in [
            'bbc', 'cnn', 'reuters', 'ap.org', 'bloomberg'
        ]):
            source_info['bias_label'] = "International Media"
            source_info['reliability_icon'] = "🌍 International"
            
        # Skip unknown sources (don't include in bias analysis)
        else:
            source_info['bias_label'] = "Unknown Political Stance"
            source_info['reliability_icon'] = "? Unknown"
        
        # Add emotional and framing context
        if 'emotional_tone' in bias_analysis:
            emotional = bias_analysis['emotional_tone']
            source_info['bias_details']['emotional'] = {
                'tone': emotional.get('emotion', 'neutral'),
                'score': emotional.get('emotion_score', 0)
            }
        
        if 'framing_bias' in bias_analysis:
            framing = bias_analysis['framing_bias']
            source_info['bias_details']['framing'] = {
                'type': framing.get('framing_type', 'neutral'),
                'score': framing.get('framing_score', 0)
            }
        
        return source_info

    def _get_bias_context_summary(self, source: Dict) -> str:
        """Get brief bias context for display"""
        context_parts = []
        
        bias_details = source.get('bias_details', {})
        
        # Emotional context
        if 'emotional' in bias_details:
            emotional = bias_details['emotional']
            if emotional['tone'] != 'neutral' and abs(emotional['score']) > 0.5:
                context_parts.append(f"[{emotional['tone']} tone]")
        
        # Framing context
        if 'framing' in bias_details:
            framing = bias_details['framing']
            if framing['type'] != 'neutral' and abs(framing['score']) > 0.3:
                context_parts.append(f"[{framing['type']} framing]")
        
        return " ".join(context_parts) if context_parts else ""

    def _generate_bias_warning(self, supporting_sources: List[Dict], 
                            refuting_sources: List[Dict], claim: str) -> List[str]:
        """Generate comprehensive bias warnings including emotional and framing bias"""
        
        warnings = []
        
        # Count political bias patterns
        support_gov = sum(1 for s in supporting_sources if 'pro-government' in s.get('bias_label', '').lower())
        support_opp = sum(1 for s in supporting_sources if 'opposition' in s.get('bias_label', '').lower())
        refute_gov = sum(1 for s in refuting_sources if 'pro-government' in s.get('bias_label', '').lower())
        refute_opp = sum(1 for s in refuting_sources if 'opposition' in s.get('bias_label', '').lower())
        
        # Add all sources for emotional analysis
        all_sources = supporting_sources + refuting_sources
        
        # Political bias warnings
        if support_gov >= 1:
            warnings.append(f"• {support_gov} pro-government sources support this claim")
        if refute_opp >= 1:
            warnings.append(f"• {refute_opp} opposition sources refute this claim")
        if support_opp >= 1:
            warnings.append(f"• {support_opp} opposition sources support this claim")
        if refute_gov >= 1:
            warnings.append(f"• {refute_gov} pro-government sources refute this claim")
        
        # Emotional tone analysis
        negative_sources = []
        positive_sources = []
        
        for source in all_sources:
            if 'emotional' in source.get('bias_details', {}):
                emotional = source['bias_details']['emotional']
                if emotional['tone'] == 'negative' and emotional['score'] < -0.5:
                    negative_sources.append(source['name'])
                elif emotional['tone'] == 'positive' and emotional['score'] > 0.5:
                    positive_sources.append(source['name'])
        
        if len(negative_sources) >= 3:
            warnings.append(f"• High emotional negativity detected in {len(negative_sources)} sources")
        if len(positive_sources) >= 2:
            warnings.append(f"• Positive emotional framing in {len(positive_sources)} sources")
        
        # Framing analysis
        problematic_framing = []
        for source in all_sources:
            if 'framing' in source.get('bias_details', {}):
                framing = source['bias_details']['framing']
                if framing['type'] != 'neutral' and abs(framing['score']) > 0.3:
                    problematic_framing.append(f"{source['name']} ({framing['type']})")
        
        if problematic_framing:
            warnings.append(f"• Biased framing detected: {', '.join(problematic_framing)}")
        
        # Look for contradictory official sources
        official_support = [s for s in supporting_sources if 'official' in s.get('bias_label', '').lower()]
        official_refute = [s for s in refuting_sources if 'official' in s.get('bias_label', '').lower()]
        
        if official_support and official_refute:
            warnings.append("• Official sources disagree - conflicting government data")
        
        # Claim-specific contextual warnings
        if any(word in claim.lower() for word in ['government', 'president', 'minister', 'parliament']):
            if support_gov > 0:
                warnings.append("• Government sources may overstate achievements on political claims")
            if refute_opp > 0:
                warnings.append("• Opposition sources may overstate criticism of government")
        
        if any(word in claim.lower() for word in ['economic', 'imf', 'budget', 'gdp']):
            if support_gov > 0:
                warnings.append("• Government sources may emphasize positive economic indicators")
        
        # High bias risk warning
        high_risk_sources = [s for s in all_sources if '🔴' in s.get('reliability_icon', '')]
        if high_risk_sources:
            source_names = [s['name'] for s in high_risk_sources]
            warnings.append(f"• High bias risk sources: {', '.join(source_names)}")
        
        return warnings

    def _generate_recommendation(self, claim: str, supporting_sources: List[Dict], 
                                refuting_sources: List[Dict], final_verdict: Dict) -> str:
        """Generate contextual recommendation"""
        
        verdict = final_verdict.get('verdict', 'UNCERTAIN').upper()
        confidence = final_verdict.get('confidence', 0) * 100
        
        # Count reliable vs biased sources
        reliable_support = sum(1 for s in supporting_sources if '✓' in s.get('reliability_icon', ''))
        reliable_refute = sum(1 for s in refuting_sources if '✓' in s.get('reliability_icon', ''))
        biased_support = len(supporting_sources) - reliable_support
        biased_refute = len(refuting_sources) - reliable_refute
        
        if verdict == 'SUPPORTED':
            if reliable_support > biased_support:
                return f"The claim appears well-supported by reliable sources. Confidence is high ({confidence:.0f}%)."
            else:
                return f"Claim has support, but mainly from potentially biased sources. Exercise caution."
                
        elif verdict == 'REFUTED':
            if reliable_refute > biased_refute:
                return f"The claim is contradicted by reliable sources. Evidence suggests it's false."
            else:
                return f"Claim is disputed, but mainly by potentially biased sources. Seek additional verification."
                
        else:  # UNCERTAIN
            if len(supporting_sources) == len(refuting_sources):
                return f"Evidence is split. Both sides have roughly equal support. More investigation needed."
            elif biased_support > 0 or biased_refute > 0:
                return f"Insufficient reliable evidence. Many sources show potential bias. Seek neutral sources."
            else:
                return f"Limited evidence available. Cannot verify with current sources ({confidence:.0f}% uncertainty)."

    def _generate_enhanced_bias_warning(self, supporting_sources: List[Dict], 
                                   refuting_sources: List[Dict], 
                                   neutral_sources: List[Dict], 
                                   claim: str) -> List[str]:
        """Generate comprehensive bias warnings for sources with identified bias only"""
        
        warnings = []
        
        # Count bias patterns by verdict category
        support_gov = sum(1 for s in supporting_sources if 'pro-government' in s.get('bias_label', '').lower())
        support_opp = sum(1 for s in supporting_sources if 'opposition' in s.get('bias_label', '').lower())
        refute_gov = sum(1 for s in refuting_sources if 'pro-government' in s.get('bias_label', '').lower())
        refute_opp = sum(1 for s in refuting_sources if 'opposition' in s.get('bias_label', '').lower())
        neutral_gov = sum(1 for s in neutral_sources if 'pro-government' in s.get('bias_label', '').lower())
        neutral_opp = sum(1 for s in neutral_sources if 'opposition' in s.get('bias_label', '').lower())
        
        # Count official/academic sources
        support_official = sum(1 for s in supporting_sources if 'official' in s.get('bias_label', '').lower() or 'academic' in s.get('bias_label', '').lower())
        refute_official = sum(1 for s in refuting_sources if 'official' in s.get('bias_label', '').lower() or 'academic' in s.get('bias_label', '').lower())
        neutral_official = sum(1 for s in neutral_sources if 'official' in s.get('bias_label', '').lower() or 'academic' in s.get('bias_label', '').lower())
        
        # Total bias counts
        total_gov = support_gov + refute_gov + neutral_gov
        total_opp = support_opp + refute_opp + neutral_opp
        total_official = support_official + refute_official + neutral_official
        
        # Verdict-specific warnings
        if support_gov >= 1:
            warnings.append(f"• {support_gov} pro-government sources support this claim")
        if refute_gov >= 1:
            warnings.append(f"• {refute_gov} pro-government sources refute this claim")
        if support_opp >= 1:
            warnings.append(f"• {support_opp} opposition sources support this claim")
        if refute_opp >= 1:
            warnings.append(f"• {refute_opp} opposition sources refute this claim")
        
        # Overall bias distribution with breakdown
        if total_gov >= 2:
            breakdown_parts = []
            if support_gov > 0: breakdown_parts.append(f"{support_gov} supporting")
            if refute_gov > 0: breakdown_parts.append(f"{refute_gov} refuting")  
            if neutral_gov > 0: breakdown_parts.append(f"{neutral_gov} neutral")
            warnings.append(f"• {total_gov} pro-government sources total ({', '.join(breakdown_parts)})")
            
        if total_opp >= 2:
            breakdown_parts = []
            if support_opp > 0: breakdown_parts.append(f"{support_opp} supporting")
            if refute_opp > 0: breakdown_parts.append(f"{refute_opp} refuting")
            if neutral_opp > 0: breakdown_parts.append(f"{neutral_opp} neutral")
            warnings.append(f"• {total_opp} opposition sources total ({', '.join(breakdown_parts)})")
        
        # Official source reliability note
        if total_official >= 1:
            warnings.append(f"• {total_official} official/academic sources provide institutional perspective")
        
        # Bias imbalance warnings
        if total_gov > 0 and total_opp == 0:
            warnings.append("• ⚠️ No opposition sources - potential pro-government bias")
        elif total_opp > 0 and total_gov == 0:
            warnings.append("• ⚠️ No pro-government sources - potential opposition bias")
        elif abs(total_gov - total_opp) >= 3:
            stronger = "pro-government" if total_gov > total_opp else "opposition"
            warnings.append(f"• ⚠️ Strong {stronger} source bias ({max(total_gov, total_opp)} vs {min(total_gov, total_opp)})")
        
        # Contextual warnings based on claim type
        if any(word in claim.lower() for word in ['government', 'president', 'minister', 'parliament']):
            if total_gov > total_opp:
                warnings.append("• Government sources may overstate achievements on political claims")
            elif total_opp > total_gov:
                warnings.append("• Opposition sources may overstate criticism of government")
        
        return warnings