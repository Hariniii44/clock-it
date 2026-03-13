"""
FramingAnalyzer
---------------
Detects and explains article-level political framing bias using two signals:

  1. Entity-level framing
     For each source, find sentences that mention key political entities from
     the claim and score whether those sentences use pro-government, pro-
     opposition, or neutral language (lexical word-list approach, no extra
     model needed).

  2. LLM cross-source analysis
     Sends all source snippets + entity framing scores to Groq in one prompt.
     Groq identifies:
       - Specific loaded phrases in each article
       - Framing direction per source (pro_government / pro_opposition / neutral)
       - Whether the framing is consistent with the source's known political profile
       - One-sentence evidence-backed explanation per source
       - Cross-source divergence summary

The entity framing scores are passed as structured context to the LLM so it
is doing *explanation*, not raw detection — the part LLMs are reliable at.

Output of analyze() feeds into EvidenceWeighter.weight_all_evidence() via the
optional framing_analyses parameter, where consistency_with_profile adjusts
the bias_penalty multiplier.
"""

import json
import re
from typing import Dict, List

from groq import Groq


# ---------------------------------------------------------------------------
# Lexical direction word-lists (Sri Lankan political context)
# Words associated with positive/negative framing of the ruling party / govt
# ---------------------------------------------------------------------------
_PRO_GOVT_WORDS = frozenset({
    'achieved', 'accomplished', 'addressed', 'announced', 'approved',
    'committed', 'completed', 'delivering', 'implemented', 'launched',
    'progress', 'resolved', 'secured', 'succeeded', 'tackled', 'welcomed',
    'development', 'growth', 'improved', 'strengthened', 'stabilised',
})

_ANTI_GOVT_WORDS = frozenset({
    'collapsed', 'crisis', 'failed', 'failure', 'incompetent',
    'mismanaged', 'neglected', 'rejected', 'struggling', 'unable',
    'worsened', 'corrupt', 'broken', 'denied', 'ignored', 'alarming',
    'devastating', 'disastrous', 'scandal', 'outrage', 'protest',
})


class FramingAnalyzer:
    """
    Detects political framing bias in each evidence source and explains it
    with specific phrases from the article text.
    """

    def __init__(self, groq_key: str, bias_profiles: Dict = None):
        self.client = Groq(api_key=groq_key)
        self.model = "llama-3.1-8b-instant"
        self.bias_profiles = bias_profiles or {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_domain(self, url: str) -> str:
        try:
            if url.startswith('http'):
                url = url.split('://', 1)[1]
            return url.split('/')[0].lstrip('www.')
        except Exception:
            return ''

    def _extract_claim_entities(self, claim: str) -> List[str]:
        """
        Extract key political entities from the claim so we know which
        nouns to track across sources.
        """
        sl_keywords = {
            'government', 'president', 'prime minister', 'parliament',
            'cabinet', 'opposition', 'minister', 'npp', 'sjb', 'unp',
            'slpp', 'jvp', 'central bank', 'treasury', 'police', 'court',
        }
        claim_lower = claim.lower()
        found = [kw for kw in sl_keywords if kw in claim_lower]

        # Also grab capitalised words (proper nouns like person names)
        proper_nouns = re.findall(r'\b[A-Z][a-z]{2,}\b', claim)
        found.extend(w.lower() for w in proper_nouns)

        return list(set(found)) if found else ['government']

    # ------------------------------------------------------------------
    # Option 4: entity-level framing (fast, no API)
    # ------------------------------------------------------------------

    def _score_entity_framing(self, text: str, entities: List[str]) -> Dict:
        """
        For each entity, collect the sentences that mention it and count
        pro-government vs anti-government words to derive a directional score.

        Returns: { entity: { direction, score, example_sentence } }
        """
        sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 15]
        entity_framing = {}

        for entity in entities:
            relevant = [s for s in sentences if entity in s.lower()]
            if not relevant:
                continue

            pro = sum(1 for s in relevant for w in _PRO_GOVT_WORDS if w in s.lower())
            anti = sum(1 for s in relevant for w in _ANTI_GOVT_WORDS if w in s.lower())

            if pro + anti == 0:
                direction, score = 'neutral', 0.0
            else:
                score = round((pro - anti) / (pro + anti), 3)
                direction = 'positive' if score > 0.1 else ('negative' if score < -0.1 else 'neutral')

            entity_framing[entity] = {
                'direction': direction,
                'score': score,
                'example_sentence': relevant[0][:150],
            }

        return entity_framing

    # ------------------------------------------------------------------
    # Option 2: single LLM call for cross-source analysis
    # ------------------------------------------------------------------

    def _build_llm_prompt(
        self,
        claim: str,
        evidence_list: List[Dict],
        entity_framing_list: List[Dict],
    ) -> str:
        sources_block = ""
        for i, (ev, ef) in enumerate(zip(evidence_list, entity_framing_list)):
            domain = self._extract_domain(ev.get('link', ''))
            profile = self.bias_profiles.get(domain, {})
            profile_note = (
                f"Known profile: {profile.get('interpretation', 'unknown')} "
                f"(score {profile.get('bias_score', 0):+.1f})"
                if profile else "Known profile: none"
            )

            entity_notes = "".join(
                f"\n    - '{ent}': {info['direction']} framing (score {info['score']}), "
                f"e.g. \"{info['example_sentence']}\""
                for ent, info in ef.items()
            ) or "\n    - none detected"

            snippet = ev.get('snippet', '')[:300]
            sources_block += (
                f"\n--- Source {i} | {domain} | {profile_note} ---"
                f"\nEntity framing:{entity_notes}"
                f"\nSnippet: {snippet}\n"
            )

        return f"""You are a media bias analyst specialising in Sri Lankan political news.

Claim being fact-checked: "{claim}"

Below are {len(evidence_list)} sources that covered this claim.
{sources_block}

For EACH source, identify:
1. framing_direction: pro_government / pro_opposition / neutral / unclear
2. loaded_phrases: up to 3 exact phrases from the snippet that reveal the framing
3. consistency_with_profile: consistent / inconsistent / unknown
   (consistent = framing direction matches the source's known political lean)
4. explanation: ONE sentence describing what the source said and why it reveals bias.
   Reference specific phrases. If the source is genuinely neutral, say so clearly.

Then provide an overall summary:
- cross_source_divergence: how sources differ in framing the same claim
- contested_entities: which entities are described differently across sources
- divergence_level: low / medium / high

RULES:
- Only cite phrases that literally appear in the snippets provided.
- Do not invent phrases or paraphrase as if quoting.
- If a source has no detectable bias, set framing_direction to "neutral" and loaded_phrases to [].

Respond ONLY with valid JSON in exactly this structure (no markdown, no extra text):
{{
  "sources": [
    {{
      "index": 0,
      "source": "domain.com",
      "framing_direction": "pro_government|pro_opposition|neutral|unclear",
      "loaded_phrases": ["phrase1", "phrase2"],
      "consistency_with_profile": "consistent|inconsistent|unknown",
      "explanation": "One sentence with specific evidence."
    }}
  ],
  "cross_source_divergence": "Summary of how sources differ.",
  "contested_entities": ["entity1"],
  "divergence_level": "low|medium|high"
}}"""

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(self, claim: str, evidence_list: List[Dict]) -> Dict:
        """
        Run entity-level framing (Option 4) then a single LLM call
        (Option 2) for cross-source analysis and per-source explanations.

        Returns a dict with keys:
          sources             — list of per-source framing dicts
          cross_source_divergence — plain-text summary
          contested_entities  — list of entity names
          divergence_level    — 'low' | 'medium' | 'high'
        """
        if not evidence_list:
            return {
                'sources': [],
                'cross_source_divergence': 'No sources to analyse.',
                'contested_entities': [],
                'divergence_level': 'low',
            }

        entities = self._extract_claim_entities(claim)

        # Step 1: fast entity-level framing per source
        entity_framing_list = [
            self._score_entity_framing(ev.get('snippet', ''), entities)
            for ev in evidence_list
        ]

        # Step 2: single LLM call
        prompt = self._build_llm_prompt(claim, evidence_list, entity_framing_list)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1800,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences the model sometimes adds
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'\s*```\s*$', '', raw, flags=re.MULTILINE)

            result = json.loads(raw)

            # Attach entity framing to each source entry for downstream use
            for entry in result.get('sources', []):
                idx = entry.get('index', 0)
                if idx < len(entity_framing_list):
                    entry['entity_framing'] = entity_framing_list[idx]

            return result

        except json.JSONDecodeError as exc:
            print(f"  Framing analysis: JSON parse failed ({exc}) — using entity framing only")
            return self._fallback_result(evidence_list, entity_framing_list)
        except Exception as exc:
            print(f"  Framing analysis failed: {exc}")
            return self._fallback_result(evidence_list, entity_framing_list)

    # ------------------------------------------------------------------
    # Fallback when LLM call fails
    # ------------------------------------------------------------------

    def _fallback_result(
        self,
        evidence_list: List[Dict],
        entity_framing_list: List[Dict],
    ) -> Dict:
        """Return entity-level framing only when the LLM call fails."""
        sources = []
        for i, (ev, ef) in enumerate(zip(evidence_list, entity_framing_list)):
            domain = self._extract_domain(ev.get('link', ''))
            scores = [v['score'] for v in ef.values()]
            avg = sum(scores) / len(scores) if scores else 0.0
            direction = (
                'pro_government' if avg > 0.1
                else ('pro_opposition' if avg < -0.1 else 'neutral')
            )
            sources.append({
                'index': i,
                'source': domain,
                'framing_direction': direction,
                'loaded_phrases': [],
                'consistency_with_profile': 'unknown',
                'explanation': (
                    f"LLM unavailable — entity-level analysis only. "
                    f"Average framing score: {avg:+.2f} ({direction})."
                ),
                'entity_framing': ef,
            })

        return {
            'sources': sources,
            'cross_source_divergence': 'Cross-source LLM analysis unavailable.',
            'contested_entities': [],
            'divergence_level': 'unknown',
        }
