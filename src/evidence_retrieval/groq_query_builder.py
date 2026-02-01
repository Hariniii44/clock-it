from groq import Groq
from typing import List, Dict  # Added missing Dict import
import json

class GroqQueryBuilder:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        # Use the correct model name for Groq
        self.model = "llama-3.1-8b-instant"  # Fixed model name
    
    def generate_search_queries(self, claim: str, num_queries: int = 5) -> List[str]:
        """Generate targeted search queries using Groq LLM"""
        
        prompt = f"""
        You are an expert fact-checker designing search queries for Sri Lankan news sources.
        
        CLAIM TO VERIFY: "{claim}"
        
        Generate {num_queries} diverse, targeted search queries that would help find evidence to verify this claim about Sri Lanka.
        
        Guidelines:
        1. Target Sri Lankan media sources (Daily Mirror, EconomyNext, Ada Derana, Island, etc.)
        2. Include specific names, dates, and locations mentioned in the claim
        3. Consider both supporting AND contradicting evidence
        4. Use relevant Sri Lankan political and economic terminology
        5. Include temporal aspects if the claim has time elements
        
        For political claims like "{claim}":
        - Target government announcements and policy statements
        - Look for opposition reactions and criticism  
        - Include specific ministry names and relevant terms
        - Search for both recent news and official sources
        
        Return ONLY a JSON array of search query strings:
        ["query1", "query2", "query3", ...]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            queries = json.loads(response_text)
            
            # Validate and return
            if isinstance(queries, list) and len(queries) > 0:
                return [q.strip() for q in queries if isinstance(q, str) and len(q.strip()) > 10][:num_queries]
            else:
                return self._fallback_queries(claim, num_queries)

        except Exception as e:
            print(f"      ⚠️ Groq query generation failed: {e}")
            return self._fallback_queries(claim, num_queries)

    def filter_relevant_evidence(self, claim: str, evidence_list: List[Dict]) -> List[Dict]:
        """Use Groq to intelligently filter relevant evidence"""
        
        if not evidence_list or len(evidence_list) == 0:
            return []
        
        print(f"   🤖 Analyzing {len(evidence_list)} pieces of evidence with Groq...")
        
        # Prepare evidence summaries for Groq (limit to prevent token overflow)
        max_evidence = min(len(evidence_list), 10)  # Limit to 10 pieces
        evidence_summaries = []
        
        for i in range(max_evidence):
            evidence = evidence_list[i]
            title = evidence.get('title', 'No title')[:100]  # Truncate long titles
            snippet = evidence.get('snippet', 'No snippet')[:200]  # Truncate long snippets
            source = evidence.get('source', 'Unknown source')[:50]
            
            summary = f"Evidence {i+1}:\nTitle: {title}\nSnippet: {snippet}\nSource: {source}"
            evidence_summaries.append(summary)
        
        evidence_text = "\n\n".join(evidence_summaries)
        
        prompt = f"""
        You are an expert fact-checker analyzing evidence relevance.
        
        CLAIM TO VERIFY: "{claim}"

        RETRIEVED EVIDENCE:
        {evidence_text}
        
        Your task: Determine which evidence pieces are DIRECTLY RELEVANT to verifying this specific claim.
        
        Relevance criteria:
        1. DIRECTLY addresses the claim's main assertion
        2. Contains specific information that could support, refute, or provide context
        3. Is about the same topic, person, event, or policy mentioned in the claim
        4. Is NOT just tangentially related or about a different aspect
        
        For the claim "{claim}", mark each evidence as:
        - HIGHLY_RELEVANT: Directly addresses the claim with specific information
        - MODERATELY_RELEVANT: Related but not directly verifying
        - IRRELEVANT: Not related to the claim or about different topic
        
        Return ONLY a JSON array:
        [
            {{"evidence_id": 1, "relevance": "HIGHLY_RELEVANT", "reason": "Contains specific announcement details"}},
            {{"evidence_id": 2, "relevance": "IRRELEVANT", "reason": "About different topic"}},
            ...
        ]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model, 
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            relevance_scores = json.loads(response_text)
            
            # Filter evidence based on Groq's analysis
            filtered_evidence = []

            for score_data in relevance_scores:
                try:
                    evidence_id = int(score_data["evidence_id"]) - 1  # Convert to 0-based
                    relevance = score_data["relevance"]
                    reason = score_data.get("reason", "")
                    
                    if 0 <= evidence_id < len(evidence_list):
                        evidence = evidence_list[evidence_id].copy()
                        evidence['ai_relevance'] = relevance
                        evidence['ai_reason'] = reason
                        
                        # Only include highly and moderately relevant evidence
                        if relevance in ["HIGHLY_RELEVANT", "MODERATELY_RELEVANT"]:
                            filtered_evidence.append(evidence)
                            print(f"         ✅ Kept: {evidence.get('title', 'No title')[:40]}... ({relevance})")
                        else:
                            print(f"         ❌ Filtered: {evidence.get('title', 'No title')[:40]}... ({relevance})")
                except (KeyError, ValueError, IndexError) as e:
                    print(f"         ⚠️ Error processing evidence {score_data}: {e}")
                    continue
            
            return filtered_evidence

        except Exception as e:
            print(f"         ⚠️ AI relevance filtering failed: {e}")
            print(f"         📝 Raw response: {response_text if 'response_text' in locals() else 'No response'}")
            # Fallback to original evidence
            return evidence_list

    def _fallback_queries(self, claim: str, num_queries: int) -> List[str]:
        """Fallback to simple queries if Groq fails"""
        words = claim.lower().split()
        
        queries = [
            f'"{claim}" Sri Lanka',
            f'{" ".join(words[:5])} Sri Lanka news',
            f'{" ".join(words[:3])} announcement Sri Lanka',
            f'Sri Lanka {" ".join(words[2:5])} policy',
            f'corruption measures Sri Lanka government' if 'corruption' in claim.lower() else f'{" ".join(words)} Sri Lanka'
        ]
        
        return queries[:num_queries]