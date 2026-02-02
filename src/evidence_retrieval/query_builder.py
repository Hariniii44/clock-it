# import google.generativeai as genai
# from typing import List
# import json

# class GeminiQueryBuilder:
#     def __init__(self, api_key: str):
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel('gemini-2.5-flash')

#         # Load bias profiles
#         self.bias_profiles = self._load_bias_profiles()
    
#     def _load_bias_profiles(self) -> dict:
#         """Load bias profiles for source credibility assessment"""
#         try:
#             with open("data/bias_profiles.json", 'r', encoding='utf-8') as f:
#                 return json.load(f)
#         except FileNotFoundError:
#             print("⚠️ Bias profiles not found. Run bias detection first.")
#             return {}
        
#     def get_source_bias_info(self, url: str) -> dict:
#         """Get bias information for a source URL"""
#         # Extract domain from URL
#         domain = self._extract_domain(url)
        
#         if domain in self.bias_profiles:
#             profile = self.bias_profiles[domain]
#             return {
#                 "source": domain,
#                 "bias_score": profile["bias_score"],
#                 "bias_interpretation": profile["interpretation"],
#                 "confidence": profile["confidence"],
#                 "articles_analyzed": profile["articles_analyzed"]
#             }
#         else:
#             return {
#                 "source": domain,
#                 "bias_score": 0.0,
#                 "bias_interpretation": "Unknown bias",
#                 "confidence": 0.0,
#                 "articles_analyzed": 0
#             }
        
#     def _extract_domain(self, url: str) -> str:
#         """Extract domain from URL"""
#         try:
#             if url.startswith('http'):
#                 url = url.split('://', 1)[1]
#             domain = url.split('/')[0]
#             if domain.startswith('www.'):
#                 domain = domain[4:]
#             return domain
#         except:
#             return ""

    
#     def generate_search_queries(self, claim: str, num_queries: int = 5) -> List[str]:
#         """Generate targeted search queries using Gemini"""
        
#         prompt = f"""
#         You are an expert fact-checker designing search queries for Sri Lankan news sources.
        
#         CLAIM TO VERIFY: "{claim}"
        
#         Generate {num_queries} diverse, targeted search queries that would help find evidence to verify this claim about Sri Lanka.
        
#         Guidelines:
#         1. Include geographical specificity (provinces, cities if relevant)
#         2. Use relevant Sri Lankan terminology and context
#         3. Target different types of sources (government, media, international)
#         4. Include temporal aspects if the claim has time elements
#         5. Consider both supporting AND contradicting evidence
#         6. Use synonyms and alternative phrasings
        
#         Examples for reference:
#         - For economic claims: include terms like "Central Bank", "IMF", "World Bank", "economic data"
#         - For political claims: include "Parliament", "government", "opposition", "policy"
#         - For regional claims: include specific provinces/districts
#         - For crime claims: include "police", "law enforcement", specific regions
        
#         Return ONLY a JSON array of search query strings, no explanations:
#         ["query1", "query2", "query3", ...]
#         """
        
#         try:
#             response = self.model.generate_content(prompt)
            
#             # Extract JSON from response
#             response_text = response.text.strip()
            
#             # Handle potential markdown formatting
#             if "```json" in response_text:
#                 response_text = response_text.split("```json")[1].split("```")[0].strip()
#             elif "```" in response_text:
#                 response_text = response_text.split("```")[1].split("```")[0].strip()
            
#             queries = json.loads(response_text)
            
#             # Validate and clean
#             if isinstance(queries, list):
#                 return [q for q in queries if isinstance(q, str) and len(q.strip()) > 0][:num_queries]
#             else:
#                 return self._fallback_queries(claim, num_queries)
                
#         except Exception as e:
#             print(f"      ⚠️ Gemini query generation failed: {e}")
#             return self._fallback_queries(claim, num_queries)
    
#     def _fallback_queries(self, claim: str, num_queries: int) -> List[str]:
#         """Fallback to simple queries if Gemini fails"""
#         base_queries = [
#             f'"{claim}" Sri Lanka',
#             f'{claim} Sri Lanka news',
#             f'{claim} official statement Sri Lanka',
#             f'{claim} verification Sri Lanka',
#             f'{claim} fact check Sri Lanka'
#         ]
#         return base_queries[:num_queries]
    
#     def assess_source_reliability(self, sources_found: List[str]) -> dict:
#         """Assess reliability of sources found during fact-checking"""
#         assessment = {
#             "total_sources": len(sources_found),
#             "bias_distribution": {
#                 "opposition_leaning": 0,
#                 "neutral": 0,
#                 "pro_government": 0,
#                 "unknown": 0
#             },
#             "source_details": []
#         }
        
#         for url in sources_found:
#             bias_info = self.get_source_bias_info(url)
#             assessment["source_details"].append(bias_info)
            
#             # Categorize bias
#             score = bias_info["bias_score"]
#             if bias_info["source"] in self.bias_profiles:
#                 if score < -10:
#                     assessment["bias_distribution"]["opposition_leaning"] += 1
#                 elif score > 10:
#                     assessment["bias_distribution"]["pro_government"] += 1
#                 else:
#                     assessment["bias_distribution"]["neutral"] += 1
#             else:
#                 assessment["bias_distribution"]["unknown"] += 1
        
#         return assessment

#     # In groq_query_builder.py - Add this new method:

# class GroqQueryBuilder:
#     def __init__(self, api_key: str):
#         # ... existing init code ...
    
#     def filter_relevant_evidence(self, claim: str, evidence_list: List[Dict]) -> List[Dict]:
#         """Use Groq to intelligently filter relevant evidence"""
        
#         if not evidence_list:
#             return []
        
#         # Prepare evidence summaries for Groq
#         evidence_summaries = []
#         for i, evidence in enumerate(evidence_list):
#             summary = f"Evidence {i+1}:\nTitle: {evidence['title']}\nSnippet: {evidence['snippet']}\nSource: {evidence['source']}"
#             evidence_summaries.append(summary)
        
#         evidence_text = "\n\n".join(evidence_summaries)
        
#         prompt = f"""
#         You are an expert fact-checker analyzing evidence relevance.
        
#         CLAIM TO VERIFY: "{claim}"
        
#         RETRIEVED EVIDENCE:
#         {evidence_text}
        
#         Your task: Determine which evidence pieces are DIRECTLY RELEVANT to verifying the specific claim.
        
#         Relevance criteria:
#         1. DIRECTLY addresses the claim's main assertion
#         2. Contains specific information that could support, refute, or provide context for the claim
#         3. Is about the same topic, person, event, or data mentioned in the claim
#         4. Is NOT just tangentially related or about a different aspect
        
#         For the claim "{claim}", mark each evidence as:
#         - HIGHLY_RELEVANT: Directly addresses the claim, contains specific verifying information
#         - MODERATELY_RELEVANT: Related to the claim but not directly verifying
#         - LOW_RELEVANCE: Tangentially related but not useful for verification
#         - IRRELEVANT: Not related to the claim or about different topic
        
#         Return ONLY a JSON array with relevance scores for each evidence:
#         [
#             {{"evidence_id": 1, "relevance": "HIGHLY_RELEVANT", "reason": "brief reason"}},
#             {{"evidence_id": 2, "relevance": "IRRELEVANT", "reason": "brief reason"}},
#             ...
#         ]
#         """
        
#         try:
#             response = self.client.chat.completions.create(
#                 model="llama-3.1-8b-instant",
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=1000,
#                 temperature=0.1
#             )
            
#             response_text = response.choices[0].message.content.strip()
            
#             # Parse JSON response
#             if "```json" in response_text:
#                 response_text = response_text.split("```json")[1].split("```")[0].strip()
#             elif "```" in response_text:
#                 response_text = response_text.split("```")[1].split("```")[0].strip()
            
#             relevance_scores = json.loads(response_text)
            
#             # Filter evidence based on Groq's analysis
#             filtered_evidence = []
#             for score_data in relevance_scores:
#                 evidence_id = score_data["evidence_id"] - 1  # Convert to 0-based index
#                 relevance = score_data["relevance"]
#                 reason = score_data.get("reason", "")
                
#                 if evidence_id < len(evidence_list):
#                     evidence = evidence_list[evidence_id].copy()
#                     evidence['ai_relevance'] = relevance
#                     evidence['ai_reason'] = reason
                    
#                     # Only include highly and moderately relevant evidence
#                     if relevance in ["HIGHLY_RELEVANT", "MODERATELY_RELEVANT"]:
#                         filtered_evidence.append(evidence)
#                         print(f"         ✅ Kept: {evidence['title'][:40]}... ({relevance})")
#                     else:
#                         print(f"         ❌ Filtered: {evidence['title'][:40]}... ({relevance}: {reason})")
            
#             return filtered_evidence
            
#         except Exception as e:
#             print(f"         ⚠️ AI relevance filtering failed: {e}")
#             # Fallback to original evidence
#             return evidence_list