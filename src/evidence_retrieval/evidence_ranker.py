from sentence_transformers import CrossEncoder
from typing import List, Dict
import torch

class EvidenceRanker:
    def __init__(self):
        """Initialize cross-encoder for evidence ranking following LiveFC approach"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        
        try:
            self.ranker = CrossEncoder(model_name)
            print(f"   ✅ Cross-encoder ranker loaded: {model_name}")
        except Exception as e:
            print(f"   ⚠️ Cross-encoder loading failed: {e}")
            self.ranker = None
    
    def rank_evidence(self, claim: str, evidence_list: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rank evidence using cross-encoder model"""
        
        if not self.ranker or not evidence_list:
            return evidence_list[:top_k]
        
        print(f"         📊 Ranking {len(evidence_list)} pieces of evidence...")
        
        # Prepare claim-evidence pairs for ranking
        pairs = []
        for evidence in evidence_list:
            # Combine title and snippet for ranking
            evidence_text = f"{evidence.get('title', '')} {evidence.get('snippet', '')}"
            pairs.append([claim, evidence_text])
        
        try:
            # Get relevance scores from cross-encoder
            scores = self.ranker.predict(pairs)
            
            # Add scores to evidence
            for i, evidence in enumerate(evidence_list):
                evidence['relevance_score'] = float(scores[i])
            
            # Sort by relevance score
            ranked_evidence = sorted(evidence_list, key=lambda x: x['relevance_score'], reverse=True)
            
            print(f"         📈 Evidence ranked by relevance scores")
            for i, evidence in enumerate(ranked_evidence[:3]):
                print(f"            {i+1}. {evidence.get('title', 'No title')[:50]}... (score: {evidence['relevance_score']:.3f})")
            
            return ranked_evidence[:top_k]
            
        except Exception as e:
            print(f"         ⚠️ Evidence ranking failed: {e}")
            return evidence_list[:top_k]
        
    def rank_evidence_with_priority(self, claim: str, evidence_list: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rank evidence with priority source weighting"""
        
        if not self.ranker or not evidence_list:
            return evidence_list[:top_k]
        
        print(f"         📊 Ranking {len(evidence_list)} pieces with priority weighting...")
        
        # Priority weights
        priority_weights = {
            'official_primary': 3.0,
            'fact_checkers': 2.5, 
            'sri_lankan_news': 2.0,
            'government': 1.5,
            'international': 1.0,
            'unknown': 0.5
        }
        
        # Prepare claim-evidence pairs for ranking
        pairs = []
        for evidence in evidence_list:
            evidence_text = f"{evidence.get('title', '')} {evidence.get('snippet', '')}"
            pairs.append([claim, evidence_text])
        
        try:
            # Get relevance scores from cross-encoder
            relevance_scores = self.ranker.predict(pairs)
            
            # Calculate final scores with priority weighting
            for i, evidence in enumerate(evidence_list):
                base_relevance = float(relevance_scores[i])
                priority_tier = evidence.get('priority_tier', 'unknown')
                priority_weight = priority_weights.get(priority_tier, 0.5)
                
                # Final score = base relevance + priority bonus
                final_score = base_relevance + (priority_weight * 0.5)  # 0.5 priority bonus multiplier
                
                evidence['relevance_score'] = base_relevance
                evidence['priority_weight'] = priority_weight  
                evidence['final_score'] = final_score
            
            # Sort by final score (relevance + priority)
            ranked_evidence = sorted(evidence_list, key=lambda x: x['final_score'], reverse=True)
            
            print(f"         📈 Evidence ranked with priority weighting:")
            for i, evidence in enumerate(ranked_evidence[:3]):
                tier = evidence.get('priority_tier', 'unknown')
                print(f"            {i+1}. [{tier.upper()}] {evidence.get('title', 'No title')[:40]}... (final: {evidence['final_score']:.3f})")
            
            return ranked_evidence[:top_k]
            
        except Exception as e:
            print(f"         ⚠️ Priority ranking failed: {e}")
            return evidence_list[:top_k]