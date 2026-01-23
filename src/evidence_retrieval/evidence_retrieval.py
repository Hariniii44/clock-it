# evidence_retrieval.py
import requests
from typing import List, Dict
import os

class EvidenceRetriever:
    def __init__(self, serpapi_key: str):
        self.serpapi_key = serpapi_key
        self.base_url = "https://serpapi.com/search"
    
    def retrieve_evidence(self, claim: str, num_results: int = 5) -> List[Dict]:
        """
        FR02: Retrieve relevant evidence from news sources
        """
        params = {
            "q": claim,
            "api_key": self.serpapi_key,
            "num": num_results,
            "gl": "lk",  # Sri Lanka
            "tbm": "nws"  # News search
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            results = response.json()
            
            evidence = []
            for result in results.get("news_results", [])[:num_results]:
                evidence.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "source": result.get("source", ""),
                    "link": result.get("link", ""),
                    "date": result.get("date", "")
                })
            
            return evidence
        except Exception as e:
            print(f"Error retrieving evidence: {e}")
            return []

