from groq import Groq
from typing import List, Dict
import json

class ClaimDecomposer:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.1-8b-instant"
    
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
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse questions from response
            questions = []
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('Question') and ':' in line:
                    # Extract question after "Question X:"
                    question = line.split(':', 1)[1].strip()
                    questions.append(question)
            
            # Fallback if parsing fails
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
        
        # Basic who/what/when/where/why questions
        questions.append(f"Is the main assertion in this claim true?")
        
        if 'said' in claim.lower():
            speaker = words[0] if words else 'the person'
            questions.append(f"Did {speaker} actually make this statement?")
            questions.append(f"When was this statement made?")
            questions.append(f"What was the context of this statement?")
        
        # Add verification question
        questions.append(f"What evidence supports or refutes this claim?")
        
        return questions[:num_questions]