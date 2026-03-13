import sys
sys.path.append('src')
from evidence_retrieval.hybrid_retrieval_system import HybridRetrievalSystem
from verification.gemini_verification import GeminiClaimVerifier

def test_gemini_with_real_evidence():
    """Test Gemini with real evidence from your hybrid system"""
    
    # Initialize systems
    hybrid_retriever = HybridRetrievalSystem()  # Your existing system
    gemini_verifier = GeminiClaimVerifier()
    
    # Test with a claim that should have evidence in your datasets
    claim = "President made statements about economic development"
    
    # Get REAL evidence from your hybrid retrieval
    print(f"Retrieving evidence for: {claim}")
    evidence = hybrid_retriever.search_datasets(
        query=claim,
        top_k=10
    )
    
    print(f"Found {len(evidence)} pieces of evidence")
    
    # Verify with Gemini using REAL evidence
    result = gemini_verifier.verify_claim(claim, evidence)
    
    # Display results
    print("=" * 80)
    print("GEMINI VERIFICATION WITH REAL EVIDENCE")
    print("=" * 80)
    print(f"Claim: {claim}")
    print("\n" + gemini_verifier.get_verification_summary(result))

if __name__ == "__main__":
    test_gemini_with_real_evidence()