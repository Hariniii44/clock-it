from groq import Groq
from typing import List
import json

class GroqQueryBuilder:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "groq/compound"  # Fast and capable model
    
    def generate_search_queries(self, claim: str, num_queries: int = 5) -> List[str]:
        """Generate targeted search queries using Groq/Llama"""
        
        prompt = f"""You are an expert fact-checker designing search queries for Sri Lankan news sources.

CLAIM TO VERIFY: "{claim}"

Generate {num_queries} diverse, targeted search queries that would help find evidence to verify this claim about Sri Lanka.

Guidelines:
1. Include geographical specificity (provinces, cities if relevant)
2. Use relevant Sri Lankan terminology and context
3. Target different types of sources (government, media, international)
4. Include temporal aspects if the claim has time elements
5. Consider both supporting AND contradicting evidence
6. Use synonyms and alternative phrasings

Examples for reference:
- For economic claims: include terms like "Central Bank", "IMF", "World Bank", "economic data"
- For political claims: include "Parliament", "government", "opposition", "policy"
- For regional claims: include specific provinces/districts
- For crime claims: include "police", "law enforcement", specific regions

Return ONLY a JSON array of search query strings, no explanations:
["query1", "query2", "query3", ...]"""

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=1000,
                top_p=1,
                stop=None,
                stream=False,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Handle potential markdown formatting
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            queries = json.loads(response_text)
            
            # Validate and clean
            if isinstance(queries, list):
                return [q for q in queries if isinstance(q, str) and len(q.strip()) > 0][:num_queries]
            else:
                return self._fallback_queries(claim, num_queries)
                
        except Exception as e:
            print(f"      ⚠️ Groq query generation failed: {e}")
            return self._fallback_queries(claim, num_queries)
    
    def _fallback_queries(self, claim: str, num_queries: int) -> List[str]:
        """Fallback to simple queries if Groq fails"""
        base_queries = [
            f'"{claim}" Sri Lanka',
            f'{claim} Sri Lanka news',
            f'{claim} official statement Sri Lanka',
            f'{claim} verification Sri Lanka',
            f'{claim} fact check Sri Lanka'
        ]
        return base_queries[:num_queries]