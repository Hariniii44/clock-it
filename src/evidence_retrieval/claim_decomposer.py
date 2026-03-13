from groq import Groq
from typing import List, Dict
import json

# Sri Lankan political entities for geographic anchoring
SL_POLITICAL_ENTITIES = {
    # Parties
    "npp", "sjb", "unp", "slpp", "jvp", "itak", "tna", "upfa",
    # Key roles / titles
    "president", "prime minister", "cabinet", "parliament", "minister",
    # Institutions
    "central bank", "supreme court", "police", "treasury",
    # People (common names - extend as needed)
    "dissanayake", "wickremesinghe", "rajapaksa", "premadasa",
    "harini", "amarasuriya", "dullas", "sajith",
}


class ClaimDecomposer:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.1-8b-instant"

    # ------------------------------------------------------------------ #
    #  claim decomposition
    # ------------------------------------------------------------------ #
    def decompose_claim(self, claim: str, num_questions: int = 5) -> List[str]:
        """Decompose claim into verification questions following LiveFC approach"""

        prompt = f"""
        Transform the given claim into questions to verify the veracity of the given claim and find the potential correct facts from search engines.
        The questions must cover all aspects of the claim and must be generated only in English with no translation in brackets. Generate exactly {num_questions} questions for the claim and prefix the questions with Question number: without making any references to the claim.

        Here are some examples:

        Claim: "Kelvin Hopins was suspended from the Labor Party due to his membership in the Conservative Party."
        Question 1: Was Kelvin Hopins suspended from Labor Party?
        Question 2: Why was Kelvin Hopins suspended from Labor Party?
        Question 3: Was Kelvin Hopins a member of the Conservative Party?
        Question 4: What are the rules about dual party membership in UK politics?
        Question 5: When did the suspension of Kelvin Hopins occur?

        Claim: "The Sri Lankan economy grew by 15% in the last quarter."
        Question 1: What was the economic growth rate of Sri Lanka in the last quarter?
        Question 2: Which quarter is being referred to as the last quarter?
        Question 3: What are the official sources for Sri Lankan economic data?
        Question 4: How does this growth rate compare to previous quarters?
        Question 5: What factors contributed to Sri Lankan economic growth recently?

        Claim: "{claim}"
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )

            response_text = response.choices[0].message.content.strip()

            questions = []
            lines = response_text.split("\n")

            for line in lines:
                line = line.strip()
                if line.startswith("Question") and ":" in line:
                    question = line.split(":", 1)[1].strip()
                    questions.append(question)

            if not questions:
                questions = self._fallback_questions(claim, num_questions)

            return questions[:num_questions]

        except Exception as e:
            print(f"Claim decomposition failed: {e}")
            return self._fallback_questions(claim, num_questions)

    def _fallback_questions(self, claim: str, num_questions: int) -> List[str]:
        """Simple fallback question generation"""
        words = claim.split()
        questions = []

        questions.append("Is the main assertion in this claim true?")

        if "said" in claim.lower():
            speaker = words[0] if words else "the person"
            questions.append(f"Did {speaker} actually make this statement?")
            questions.append(f"When was this statement made?")
            questions.append(f"What was the context of this statement?")

        questions.append("What evidence supports or refutes this claim?")

        return questions[:num_questions]

    # ------------------------------------------------------------------ #
    #  search query generation                                #
    # ------------------------------------------------------------------ #
    def generate_search_queries(
        self, claim: str, num_queries: int = 5
    ) -> List[str]:
        """
        Generate search-engine-optimised query strings from a claim.

        Unlike decompose_claim() which produces human-readable verification
        questions for NLI reasoning, this method produces short, keyword-dense
        strings designed to maximise Serper/SerpAPI retrieval quality.

        Key behaviours
        --------------
        - Automatically anchors Sri Lankan political claims with 'Sri Lanka'
          to prevent entity disambiguation errors (e.g. NPP → Ghana vs. Sri Lanka)
        - Corrects obvious typos in the claim before query generation
        - Produces queries at varying specificity: one broad, rest narrower
        - Strips question marks and conversational language
        - Falls back gracefully if Groq is unavailable
        """

        geo_anchor = self._needs_sri_lanka_anchor(claim)
        anchor_instruction = (
            'Each query MUST include "Sri Lanka" as a geographic anchor '
            "because this system is focused on Sri Lankan political discourse."
            if geo_anchor
            else (
                "If the claim appears to be about Sri Lanka or Sri Lankan politics, "
                'include "Sri Lanka" in the query.'
            )
        )

        prompt = f"""
You are a search query optimiser for a fact-checking system focused on Sri Lankan political news.

Your task: convert the claim below into {num_queries} search engine queries that will retrieve
the most relevant news articles, official statements, and reports from Serper/Google.

RULES:
1. Output ONLY the queries — no explanations, no numbering labels, no question marks.
2. Each query must be on its own line, prefixed with "Query N:" (e.g. "Query 1:").
3. Queries must be keyword-dense, NOT natural language questions.
   BAD:  "Was the NPP opposition accused of misleading the public?"
   GOOD: "NPP Sri Lanka opposition misleading public rescue operation 2025"
4. {anchor_instruction}
5. Fix any obvious spelling errors in the claim before forming queries.
6. Vary specificity: first query broad, remaining queries progressively narrower
   with named entities, dates, or specific aspects of the claim.
7. Do NOT include social media sites, YouTube, or Facebook in queries.

EXAMPLES of good queries for a Sri Lankan political claim:
   Query 1: NPP Sri Lanka opposition rescue operation controversy 2025
   Query 2: National People's Power opposition misleading public statement Sri Lanka
   Query 3: Sri Lanka government opposition accusation sensitive operation news
   Query 4: NPP JVP opposition criticism rescue operation parliament Sri Lanka
   Query 5: Sri Lanka political controversy opposition misleading rescue 2025

Now generate {num_queries} queries for this claim:
Claim: "{claim}"
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.2,  # Lower temp = more consistent, focused queries
            )

            response_text = response.choices[0].message.content.strip()
            queries = self._parse_queries(response_text, num_queries)

            if not queries:
                print("Search query parsing failed — using fallback queries")
                queries = self._fallback_queries(claim, num_queries, geo_anchor)

            print(f"Generated {len(queries)} search queries:")
            for i, q in enumerate(queries, 1):
                print(f"   Query {i}: {q}")

            return queries

        except Exception as e:
            print(f"Search query generation failed: {e}")
            return self._fallback_queries(claim, num_queries, geo_anchor)

    # ------------------------------------------------------------------ #
    #  PRIVATE HELPERS for generate_search_queries                         #
    # ------------------------------------------------------------------ #
    def _needs_sri_lanka_anchor(self, claim: str) -> bool:
        """
        Return True if the claim mentions a known Sri Lankan political entity,
        which means we must anchor every search query with 'Sri Lanka' to
        prevent the model retrieving results for identically-named entities
        in other countries (e.g. Ghana's NPP vs. Sri Lanka's NPP).
        """
        claim_lower = claim.lower()
        return any(entity in claim_lower for entity in SL_POLITICAL_ENTITIES)

    def _parse_queries(self, response_text: str, num_queries: int) -> List[str]:
        """Extract query strings from the LLM response."""
        queries = []
        for line in response_text.split("\n"):
            line = line.strip()
            if line.lower().startswith("query") and ":" in line:
                query = line.split(":", 1)[1].strip()
                # Strip any trailing question marks the model might add
                query = query.rstrip("?")
                if query:
                    queries.append(query)
        return queries[:num_queries]

    def _fallback_queries(
        self, claim: str, num_queries: int, geo_anchor: bool
    ) -> List[str]:
        """
        Rule-based fallback when Groq is unavailable.
        Appends 'Sri Lanka' if the claim mentions known SL political entities.
        """
        anchor = "Sri Lanka" if geo_anchor else ""
        base = f"{claim} {anchor}".strip()

        queries = [base]

        # Extract first few meaningful words as a broad query
        words = [w for w in claim.split() if len(w) > 3]
        if len(words) >= 4:
            short = " ".join(words[:5])
            queries.append(f"{short} {anchor}".strip())

        # Add year-anchored variant
        queries.append(f"{base} 2025")

        # Add news-specific variant
        queries.append(f"{base} news statement")

        return queries[:num_queries]