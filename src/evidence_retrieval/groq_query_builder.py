from groq import Groq
from typing import List, Dict  # Added missing Dict import
import json

class GroqQueryBuilder:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        # Use the correct model name for Groq
        self.model = "llama-3.1-8b-instant"  # Fixed model name
    
    # def generate_search_queries(self, claim: str, num_queries: int = 5) -> List[str]:
    #     """Enhanced search query generation with quote-specific and targeted searches"""
        
    #     # Analyze claim type for specialized strategies
    #     claim_analysis = self._analyze_claim_type(claim)
        
    #     prompt = f"""
    #     You are an expert fact-checker designing search queries for Sri Lankan sources.
        
    #     CLAIM TO VERIFY: "{claim}"
        
    #     CLAIM ANALYSIS: {claim_analysis['description']}
        
    #     Generate {num_queries} diverse, targeted search queries using these specialized strategies:
        
    #     STRATEGY 1 - EXACT QUOTE SEARCHES (for statements/announcements):
    #     - Use exact quotes in search: "specific phrase from claim"
    #     - Target presidential statements, ministerial announcements
    #     - Include speaker attribution: "President said" + key phrases
        
    #     STRATEGY 2 - OFFICIAL SOURCE TARGETING:
    #     - Government press releases: site:news.lk, site:president.gov.lk
    #     - Ministry statements: site:agrimin.gov.lk (for agricultural claims)
    #     - Parliamentary records: site:parliament.lk
        
    #     STRATEGY 3 - OPPOSITION/CRITICISM SEARCHES:
    #     - Opposition responses to government claims
    #     - Critical analysis of statements: "criticism", "response", "refutes"
    #     - Independent verification attempts
        
    #     STRATEGY 4 - CONTEXTUAL/BACKGROUND:
    #     - Related policy discussions
    #     - Expert opinions and analysis
    #     - Historical context for claims
        
    #     STRATEGY 5 - FACT-CHECKING SPECIFIC:
    #     - Verification attempts by media
    #     - Follow-up reporting on controversial statements
    #     - Data/evidence supporting or refuting claims
        
    #     For this specific claim type ({claim_analysis['type']}):
    #     {claim_analysis['specific_strategies']}
        
    #     Return ONLY a JSON array of search query strings:
    #     ["query1", "query2", "query3", ...]
    #     """

    #     try:
    #         response = self.client.chat.completions.create(
    #             model=self.model,
    #             messages=[{"role": "user", "content": prompt}],
    #             max_tokens=600,  # Increased for more detailed queries
    #             temperature=0.3
    #         )
            
    #         response_text = response.choices[0].message.content.strip()
            
    #         # Parse JSON response
    #         if "```json" in response_text:
    #             response_text = response_text.split("```json")[1].split("```")[0].strip()
    #         elif "```" in response_text:
    #             response_text = response_text.split("```")[1].split("```")[0].strip()
            
    #         queries = json.loads(response_text)
            
    #         # Validate and enhance queries
    #         if isinstance(queries, list) and len(queries) > 0:
    #             enhanced_queries = self._enhance_queries_with_strategies(claim, queries)
    #             return enhanced_queries[:num_queries]
    #         else:
    #             return self._strategic_fallback_queries(claim, num_queries)

    #     except Exception as e:
    #         print(f"      ⚠️ Enhanced query generation failed: {e}")
    #         return self._strategic_fallback_queries(claim, num_queries)

    def generate_search_queries(self, claim: str, num_queries: int = 5) -> List[str]:
        """Generate diverse query variations of the same claim for maximum coverage"""
        
        prompt = f"""
        You are a search expert. Generate {num_queries} different search queries to find evidence about this claim.
        
        CLAIM: "{claim}"
        
        Your task: Create diverse search query variations that would find relevant articles, news reports, or statements about this claim.
        
        Strategies:
        1. Use the exact claim as-is
        2. Extract key phrases and search for them in quotes
        3. Rephrase using different words but same meaning
        4. Search for related topics and context
        5. Look for verification attempts or fact-checking
        
        Guidelines:
        - NO site: restrictions 
        - Focus on keywords and phrases that would appear in relevant articles
        - Include quoted phrases for exact matches
        - Vary the word order and phrasing
        - Include synonyms and related terms
        
        Return ONLY a JSON array of search query strings:
        ["query1", "query2", "query3", ...]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.4  # Slightly more creative for variations
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Debug: Show what LLM generated
            print(f"🔍 DEBUG: LLM Query Response: {response_text[:150]}...")
            
            # Parse JSON response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            queries = json.loads(response_text)
            
            # Validate queries
            if isinstance(queries, list) and len(queries) > 0:
                # Clean up queries - remove any site: restrictions that might have snuck in
                cleaned_queries = []
                for query in queries:
                    if isinstance(query, str):
                        # Remove site: restrictions
                        cleaned_query = ' '.join([word for word in query.split() if not word.startswith('site:')])
                        cleaned_queries.append(cleaned_query)
                
                print(f"🔍 DEBUG: Cleaned queries: {cleaned_queries}")
                return cleaned_queries[:num_queries]
            else:
                return self._simple_fallback_queries(claim, num_queries)

        except Exception as e:
            print(f"      ⚠️ Query generation failed: {e}")
            return self._simple_fallback_queries(claim, num_queries)

    def _simple_fallback_queries(self, claim: str, num_queries: int) -> List[str]:
        """Simple fallback - just variations of the original claim"""
        words = claim.split()
        queries = []
        
        # Exact claim
        queries.append(f'"{claim}"')
        
        # Key phrases from the claim
        if 'said' in claim.lower():
            parts = claim.split('said', 1)
            if len(parts) > 1:
                statement = parts[1].strip()
                queries.append(f'"{statement[:50]}"')  # First part of what was said
        
        # Main subject + key action
        if len(words) > 3:
            main_subject = ' '.join(words[:2])  # e.g., "President Dissanayake"
            key_terms = ' '.join(words[3:6])    # e.g., "giving rice animals"
            queries.append(f'{main_subject} {key_terms}')
        
        # Key topics
        key_words = [word for word in words if len(word) > 4 and word.lower() not in ['president', 'said', 'that']]
        if len(key_words) >= 2:
            queries.append(f'{key_words[0]} {key_words[1]}')
        
        # Context search
        queries.append(f'{" ".join(words[:4])} Sri Lanka')
        
        return queries[:num_queries]

    def _analyze_claim_type(self, claim: str) -> Dict[str, str]:
        """Analyze claim to determine search strategy"""
        claim_lower = claim.lower()
        
        # Statement/Quote claims
        if any(word in claim_lower for word in ['said', 'announced', 'stated', 'declared', 'claimed']):
            return {
                'type': 'statement',
                'description': 'Political statement or announcement requiring quote verification',
                'specific_strategies': '''
                - Use exact quote searches: Extract key phrases and search with quotes
                - Target official channels: Government press releases, ministry statements
                - Look for opposition responses: Critical analysis, fact-checking attempts
                - Search for context: Related policy discussions, background information
                '''
            }
        
        # Statistical/Data claims  
        elif any(word in claim_lower for word in ['percent', '%', 'increased', 'decreased', 'growth', 'statistics']):
            return {
                'type': 'statistical',
                'description': 'Data-driven claim requiring official statistics verification',
                'specific_strategies': '''
                - Target official data sources: Central Bank, Department of Statistics
                - Look for government reports and ministerial statements with data
                - Search for independent verification of statistics
                - Find expert analysis of the claimed data
                '''
            }
        
        # Policy/Governance claims
        elif any(word in claim_lower for word in ['ministry', 'government', 'policy', 'measures', 'cabinet']):
            return {
                'type': 'policy',
                'description': 'Government policy or administrative claim',
                'specific_strategies': '''
                - Search cabinet decisions and ministry announcements
                - Look for parliamentary discussions and debates
                - Find opposition reactions and criticism
                - Search for implementation updates and progress reports
                '''
            }
        
        # Default
        else:
            return {
                'type': 'general',
                'description': 'General factual claim requiring broad verification',
                'specific_strategies': '''
                - Use comprehensive keyword searches
                - Target multiple source types for verification
                - Look for expert opinions and analysis
                - Search for related news coverage and reports
                '''
            }

    def _enhance_queries_with_strategies(self, claim: str, base_queries: List[str]) -> List[str]:
        """Add strategic enhancements to base queries"""
        enhanced = base_queries.copy()
        
        # Add quote-specific searches for statement claims
        if any(word in claim.lower() for word in ['said', 'announced', 'stated']):
            quote_queries = self._generate_quote_specific_queries(claim)
            enhanced.extend(quote_queries)
        
        # Add official source targeting
        official_queries = self._generate_official_source_queries(claim)
        enhanced.extend(official_queries)
        
        # Add opposition response searches
        opposition_queries = self._generate_opposition_queries(claim)
        enhanced.extend(opposition_queries)
        
        return enhanced

    def _generate_quote_specific_queries(self, claim: str) -> List[str]:
        """Generate generic quote-specific queries that work for ANY claim"""
        queries = []
        
        # Extract potential quotes from ANY statement claim
        if 'said' in claim.lower():
            parts = claim.lower().split('said')
            if len(parts) > 1:
                statement_part = parts[1].strip()
                # Get key words from what was allegedly said
                key_words = statement_part.split()[:5]  # First 5 words of the statement
                if key_words:
                    quote_phrase = ' '.join(key_words)
                    # PRIORITIZE NEWS SITES for quote verification
                    queries.append(f'"{quote_phrase}" site:dailymirror.lk OR site:adaderana.lk')
        
        # Generic pattern: Extract speaker and main subject for quote searches
        words = claim.split()
        speaker = None
        main_subjects = []
        
        # Find speaker (any person mentioned)
        for i, word in enumerate(words):
            if word in ['President', 'Minister', 'MP', 'Chairman', 'Director']:
                if i + 1 < len(words):
                    speaker = ' '.join(words[i:i+2])  # e.g., "President Dissanayake"
                break
        
        # Extract main nouns/subjects from the claim
        for word in words:
            if len(word) > 3 and word.lower() not in ['said', 'that', 'the', 'and', 'are', 'is', 'was', 'were']:
                main_subjects.append(word.lower())
        
        # Generate generic quote verification queries
        if speaker and main_subjects:
            # Quote with speaker and main topic
            queries.append(f'"{speaker}" "{main_subjects[0]}" quote statement')
            
            # News coverage of controversial statement
            queries.append(f'{speaker} {main_subjects[0]} controversy news site:adaderana.lk')
        
        # If we have key subjects but no clear speaker
        elif main_subjects and len(main_subjects) >= 2:
            # Search for the main topics in quotes
            queries.append(f'"{main_subjects[0]}" "{main_subjects[1]}" statement news')
        
        return queries[:3]  # Limit to 3 generic quote queries

    def _generate_official_source_queries(self, claim: str) -> List[str]:
        """Generate queries targeting official government sources"""
        queries = []
        
        # Government press releases
        queries.append(f'site:news.lk OR site:president.gov.lk "{claim.split()[1:4]}"')  # Skip 'President'
        
        # Ministry-specific targeting based on claim content
        if 'rice' in claim.lower() or 'agriculture' in claim.lower():
            queries.append(f'site:agrimin.gov.lk rice policy announcement')
        
        if 'economic' in claim.lower() or 'financial' in claim.lower():
            queries.append(f'site:treasury.gov.lk OR site:cbsl.lk economic policy')
        
        return queries[:2]

    def _generate_opposition_queries(self, claim: str) -> List[str]:
        """Generate queries to find opposition responses and criticism"""
        queries = []
        
        # Opposition responses
        speaker = self._extract_speaker(claim)
        if speaker:
            queries.extend([
                f'opposition response {speaker} criticism',
                f'SJB UNP response {speaker} statement',
                f'fact check verify {speaker} claim'
            ])
        
        return queries[:2]

    def _extract_speaker(self, claim: str) -> str:
        """Extract the main speaker from the claim"""
        if 'President' in claim and 'Dissanayake' in claim:
            return 'Dissanayake'
        elif 'Minister' in claim:
            return 'Minister'
        else:
            return 'government'

    def _strategic_fallback_queries(self, claim: str, num_queries: int) -> List[str]:
        """Enhanced fallback with strategic elements"""
        words = claim.lower().split()
        queries = []
        
        # Basic query
        queries.append(f'"{claim}" Sri Lanka')
        
        # Quote-specific if statement claim
        if 'said' in claim.lower():
            main_subject = words[0] if words else 'President'
            key_content = ' '.join(words[3:8]) if len(words) > 3 else 'statement'
            queries.append(f'"{main_subject}" "{key_content}" quote')
        
        # Official sources
        queries.append(f'site:news.lk {" ".join(words[:3])}')
        
        # Opposition perspective
        queries.append(f'opposition criticism {" ".join(words[:2])}')
        
        # Context search
        queries.append(f'{" ".join(words[:4])} analysis Sri Lanka')
        
        return queries[:num_queries]

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