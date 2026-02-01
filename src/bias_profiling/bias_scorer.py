import numpy as np
import networkx as nx
import json
from typing import Dict, List, Tuple
import pandas as pd

class BiasScorer:
    def __init__(self):
        self.bias_matrix = None
        self.sources = None
        self.bias_scores = {}
        
    def load_bias_matrix(self, filepath: str) -> bool:
        """Load bias matrix from Step 4"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.bias_matrix = np.array(data['bias_matrix'])
            self.sources = data['sources']
            
            print(f"📂 Loaded bias matrix for {len(self.sources)} sources")
            return True
            
        except FileNotFoundError:
            print(f"❌ Please run Step 4 first to create bias matrix")
            return False
    
    def build_bias_graph(self) -> nx.DiGraph:
        """
        Build directed graph from bias matrix
        Following Algorithm 3 from the paper
        """
        print("🔗 Building bias graph from matrix...")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes (sources)
        G.add_nodes_from(self.sources)
        
        # Add edges based on bias matrix
        n_sources = len(self.sources)
        edges_added = 0
        
        for i in range(n_sources):
            for j in range(n_sources):
                if i != j:  # No self-edges
                    weight = self.bias_matrix[i][j]
                    if abs(weight) > 0.001:  # Only add meaningful edges
                        # Edge direction: if weight > 0, source_i is more pro-gov than source_j
                        # So edge goes from source_j to source_i (less pro-gov to more pro-gov)
                        if weight > 0:
                            G.add_edge(self.sources[j], self.sources[i], weight=abs(weight))
                        else:
                            G.add_edge(self.sources[i], self.sources[j], weight=abs(weight))
                        edges_added += 1
        
        print(f"   ✅ Graph created: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G
    
    def remove_cycles(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Remove cycles by eliminating edges with maximum weights
        Following Algorithm 3 from the paper
        """
        print("🔄 Removing cycles from graph...")
        
        G_copy = G.copy()
        cycles_removed = 0
        
        while True:
            try:
                # Find a cycle
                cycle = nx.find_cycle(G_copy, orientation='original')
                cycles_removed += 1
                
                # Find edge with maximum weight in cycle
                max_weight = 0
                max_edge = None
                
                for edge in cycle:
                    u, v = edge[0], edge[1]
                    if G_copy.has_edge(u, v):
                        weight = G_copy[u][v]['weight']
                        if weight > max_weight:
                            max_weight = weight
                            max_edge = (u, v)
                
                # Remove the edge with maximum weight
                if max_edge:
                    G_copy.remove_edge(max_edge[0], max_edge[1])
                    print(f"   🗑️ Removed edge: {max_edge[0]} → {max_edge[1]} (weight: {max_weight:.3f})")
                
            except nx.NetworkXNoCycle:
                # No more cycles found
                break
        
        print(f"   ✅ Removed {cycles_removed} cycles")
        print(f"   📊 Final graph: {len(G_copy.nodes)} nodes, {len(G_copy.edges)} edges")
        
        return G_copy
    
    def calculate_bias_scores(self, G: nx.DiGraph) -> Dict[str, float]:
        """
        Calculate final bias scores using topological sort
        Following Algorithm 3 from the paper
        """
        print("🎯 Calculating final bias scores...")
        
        try:
            # Topological sort to get ordering
            topo_order = list(nx.topological_sort(G))
            print(f"   📊 Topological order: {' → '.join(topo_order)}")
            
            # Initialize bias scores
            bias_scores = {}
            
            # Set rightmost (most pro-government) node to 0
            bias_scores[topo_order[-1]] = 0.0
            print(f"   🎯 Baseline (most pro-gov): {topo_order[-1]} = 0.0")
            
            # Calculate scores for remaining nodes (right to left)
            for i in range(len(topo_order) - 2, -1, -1):
                current_node = topo_order[i]
                
                # Find all successors (nodes this node points to)
                successors = list(G.successors(current_node))
                
                if successors:
                    # Calculate score based on successors
                    scores = []
                    for successor in successors:
                        if successor in bias_scores:
                            edge_weight = G[current_node][successor]['weight']
                            successor_score = bias_scores[successor]
                            # Current node is more opposition-leaning, so subtract weight
                            calculated_score = successor_score - edge_weight
                            scores.append(calculated_score)
                    
                    if scores:
                        bias_scores[current_node] = np.mean(scores)
                    else:
                        bias_scores[current_node] = 0.0
                else:
                    # No successors, set to 0
                    bias_scores[current_node] = 0.0
            
            # Normalize scores to a nice range (-50 to +50)
            scores_list = list(bias_scores.values())
            min_score = min(scores_list)
            max_score = max(scores_list)
            
            if max_score != min_score:
                for source in bias_scores:
                    # Normalize to -50 to +50 range
                    normalized = ((bias_scores[source] - min_score) / (max_score - min_score)) * 100 - 50
                    bias_scores[source] = normalized
            
            print(f"   ✅ Calculated bias scores for {len(bias_scores)} sources")
            return bias_scores
            
        except Exception as e:
            print(f"   ❌ Error in score calculation: {e}")
            # Fallback: use simple ranking
            return self.fallback_scoring()
    
    def fallback_scoring(self) -> Dict[str, float]:
        """Fallback scoring method if graph approach fails"""
        print("⚠️ Using fallback scoring method...")
        
        # Calculate average bias for each source
        bias_scores = {}
        n_sources = len(self.sources)
        
        for i, source in enumerate(self.sources):
            # Average outgoing bias (how this source compares to others)
            outgoing_bias = np.mean([self.bias_matrix[i][j] for j in range(n_sources) if i != j])
            bias_scores[source] = outgoing_bias * 100  # Scale up
        
        return bias_scores
    
    def interpret_bias_scores(self, bias_scores: Dict[str, float]) -> Dict[str, Dict]:
        """Add human-readable interpretations to bias scores"""
        interpreted = {}
        
        for source, score in bias_scores.items():
            if score < -30:
                interpretation = "Strong Opposition-leaning"
                color = "🔴"
            elif score < -10:
                interpretation = "Opposition-leaning"
                color = "🟠"
            elif score < 10:
                interpretation = "Independent/Neutral"
                color = "🟡"
            elif score < 30:
                interpretation = "Pro-government"
                color = "🟢"
            else:
                interpretation = "Strong Pro-government"
                color = "🔵"
            
            interpreted[source] = {
                "bias_score": round(score, 2),
                "interpretation": interpretation,
                "color": color
            }
        
        return interpreted
    
    def save_bias_profiles(self, interpreted_scores: Dict, filepath: str):
        """Save final bias profiles"""
        # Load article counts for confidence calculation
        try:
            df = pd.read_csv("data/political_articles.csv")
            article_counts = df['source'].value_counts().to_dict()
        except:
            article_counts = {}
        
        profiles = {}
        for source, data in interpreted_scores.items():
            article_count = article_counts.get(source, 0)
            confidence = min(0.95, max(0.5, article_count / 1000))  # Simple confidence metric
            
            profiles[source] = {
                "bias_score": data["bias_score"],
                "interpretation": data["interpretation"],
                "articles_analyzed": article_count,
                "confidence": round(confidence, 2)
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved bias profiles to {filepath}")
    
    def print_results(self, interpreted_scores: Dict):
        """Print final bias analysis results"""
        print("\n" + "="*60)
        print("🎯 FINAL BIAS ANALYSIS RESULTS")
        print("="*60)
        
        # Sort by bias score (most opposition to most pro-government)
        sorted_sources = sorted(interpreted_scores.items(), 
                               key=lambda x: x[1]['bias_score'])
        
        print(f"\n📊 Source Bias Rankings (Opposition ← → Pro-Government):")
        print(f"{'Rank':<4} {'Source':<20} {'Score':<8} {'Interpretation'}")
        print("-" * 60)
        
        for rank, (source, data) in enumerate(sorted_sources, 1):
            score = data['bias_score']
            interpretation = data['interpretation']
            color = data['color']
            
            print(f"{rank:<4} {source:<20} {score:>+7.1f} {color} {interpretation}")
        
        print("\n" + "="*60)
        print("📈 INTERPRETATION GUIDE:")
        print("• Negative scores (-): Opposition-leaning coverage")
        print("• Positive scores (+): Pro-government coverage") 
        print("• Scores near 0: Independent/Neutral coverage")
        print("="*60)

if __name__ == "__main__":
    # Initialize bias scorer
    scorer = BiasScorer()
    
    # Load bias matrix from Step 4
    if not scorer.load_bias_matrix("data/bias_matrix.json"):
        exit(1)
    
    # Build bias graph
    bias_graph = scorer.build_bias_graph()
    
    # Remove cycles to make graph acyclic
    acyclic_graph = scorer.remove_cycles(bias_graph)
    
    # Calculate final bias scores
    bias_scores = scorer.calculate_bias_scores(acyclic_graph)
    
    # Add interpretations
    interpreted_scores = scorer.interpret_bias_scores(bias_scores)
    
    # Save bias profiles for integration
    scorer.save_bias_profiles(interpreted_scores, "data/bias_profiles.json")
    
    # Display results
    scorer.print_results(interpreted_scores)
    
    print(f"\n🎯 Bias profiling complete! Results saved to data/bias_profiles.json")
    print("✅ READY FOR INTEGRATION WITH FACT-CHECKING PIPELINE!")