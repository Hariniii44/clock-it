import json

# Extended dataset with more Sri Lankan claims across different categories
extended_claims = [
    # FACTUAL CLAIMS - Government & Politics
    {
        "id": 11,
        "claim": "The Parliament of Sri Lanka has 225 members",
        "true_label": "TRUE",
        "category": "factual",
        "subcategory": "government",
        "difficulty": "medium"
    },
    {
        "id": 12,
        "claim": "Ranil Wickremesinghe served as Prime Minister of Sri Lanka multiple times",
        "true_label": "TRUE",
        "category": "factual",
        "subcategory": "government", 
        "difficulty": "medium"
    },
    {
        "id": 13,
        "claim": "Sri Lanka became independent from Britain in 1948",
        "true_label": "TRUE",
        "category": "historical",
        "subcategory": "independence",
        "difficulty": "easy"
    },
    
    # FACTUAL CLAIMS - Geography
    {
        "id": 14,
        "claim": "Sri Lanka is an island nation in the Indian Ocean",
        "true_label": "TRUE",
        "category": "factual",
        "subcategory": "geography",
        "difficulty": "easy"
    },
    {
        "id": 15,
        "claim": "The highest mountain in Sri Lanka is Pidurutalagala",
        "true_label": "TRUE",
        "category": "factual",
        "subcategory": "geography",
        "difficulty": "hard"
    },
    {
        "id": 16,
        "claim": "Sri Lanka shares a land border with India",
        "true_label": "FALSE",
        "category": "factual",
        "subcategory": "geography",
        "difficulty": "easy"
    },
    
    # FACTUAL CLAIMS - Economy
    {
        "id": 17,
        "claim": "The Sri Lankan Rupee is the official currency of Sri Lanka",
        "true_label": "TRUE",
        "category": "factual",
        "subcategory": "economy",
        "difficulty": "easy"
    },
    {
        "id": 18,
        "claim": "Sri Lanka defaulted on its foreign debt in 2022",
        "true_label": "TRUE",
        "category": "factual",
        "subcategory": "economy",
        "difficulty": "medium"
    },
    {
        "id": 19,
        "claim": "Tea is one of Sri Lanka's major export commodities",
        "true_label": "TRUE",
        "category": "factual",
        "subcategory": "economy",
        "difficulty": "easy"
    },
    
    # HISTORICAL CLAIMS
    {
        "id": 20,
        "claim": "The LTTE was defeated in 2009 ending the civil war",
        "true_label": "TRUE",
        "category": "historical",
        "subcategory": "conflict",
        "difficulty": "medium"
    },
    {
        "id": 21,
        "claim": "The 2004 Indian Ocean tsunami affected Sri Lanka",
        "true_label": "TRUE",
        "category": "historical",
        "subcategory": "disaster",
        "difficulty": "easy"
    },
    {
        "id": 22,
        "claim": "Ceylon was the former name of Sri Lanka",
        "true_label": "TRUE",
        "category": "historical",
        "subcategory": "naming",
        "difficulty": "medium"
    },
    
    # POLICY/CONTROVERSIAL CLAIMS
    {
        "id": 23,
        "claim": "The current government's economic reforms are successfully stabilizing the economy",
        "true_label": "CONTROVERSIAL",
        "category": "policy",
        "subcategory": "economy",
        "difficulty": "hard"
    },
    {
        "id": 24,
        "claim": "The 13th Amendment should be fully implemented in Sri Lanka",
        "true_label": "CONTROVERSIAL",
        "category": "policy", 
        "subcategory": "constitutional",
        "difficulty": "hard"
    },
    {
        "id": 25,
        "claim": "Sri Lanka's education system is among the best in South Asia",
        "true_label": "CONTROVERSIAL",
        "category": "policy",
        "subcategory": "education",
        "difficulty": "hard"
    },
    
    # CULTURAL/SOCIAL CLAIMS
    {
        "id": 26,
        "claim": "Buddhism is the majority religion in Sri Lanka",
        "true_label": "TRUE",
        "category": "factual",
        "subcategory": "culture",
        "difficulty": "easy"
    },
    {
        "id": 27,
        "claim": "Cricket is the most popular sport in Sri Lanka",
        "true_label": "TRUE",
        "category": "factual",
        "subcategory": "culture",
        "difficulty": "easy"
    },
    {
        "id": 28,
        "claim": "Sinhala and Tamil are the only languages spoken in Sri Lanka",
        "true_label": "FALSE",
        "category": "factual",
        "subcategory": "language",
        "difficulty": "medium"
    },
    
    # FALSE CLAIMS
    {
        "id": 29,
        "claim": "Sri Lanka has never experienced a military coup",
        "true_label": "TRUE",
        "category": "historical",
        "subcategory": "government",
        "difficulty": "hard"
    },
    {
        "id": 30,
        "claim": "The population of Sri Lanka is over 50 million",
        "true_label": "FALSE",
        "category": "factual",
        "subcategory": "demographics",
        "difficulty": "medium"
    }
]

def create_full_dataset():
    """Combine initial and extended claims into full dataset"""
    # Load initial dataset
    try:
        with open('ground_truth_dataset.json', 'r', encoding='utf-8') as f:
            initial_claims = json.load(f)
    except FileNotFoundError:
        initial_claims = []
    
    # Combine datasets
    full_dataset = initial_claims + extended_claims
    
    # Add reasoning for extended claims
    for claim in extended_claims:
        if claim['id'] == 11:
            claim['reasoning'] = "Constitutional fact about parliamentary composition"
        elif claim['id'] == 12:
            claim['reasoning'] = "Historical fact about political leadership"
        elif claim['id'] == 13:
            claim['reasoning'] = "Well-documented independence date"
        elif claim['id'] == 14:
            claim['reasoning'] = "Basic geographical fact"
        elif claim['id'] == 15:
            claim['reasoning'] = "Specific geographical knowledge"
        elif claim['id'] == 16:
            claim['reasoning'] = "Sri Lanka is an island, separated from India by water"
        elif claim['id'] == 17:
            claim['reasoning'] = "Official currency information"
        elif claim['id'] == 18:
            claim['reasoning'] = "Recent economic crisis event"
        elif claim['id'] == 19:
            claim['reasoning'] = "Major export commodity"
        elif claim['id'] == 20:
            claim['reasoning'] = "End of civil war is well documented"
        elif claim['id'] == 21:
            claim['reasoning'] = "Major natural disaster impact"
        elif claim['id'] == 22:
            claim['reasoning'] = "Historical naming convention"
        elif claim['id'] == 23:
            claim['reasoning'] = "Subjective assessment requiring current analysis"
        elif claim['id'] == 24:
            claim['reasoning'] = "Political opinion on constitutional implementation"
        elif claim['id'] == 25:
            claim['reasoning'] = "Comparative assessment requiring ranking data"
        elif claim['id'] == 26:
            claim['reasoning'] = "Demographic religious composition"
        elif claim['id'] == 27:
            claim['reasoning'] = "Cultural popularity assessment"
        elif claim['id'] == 28:
            claim['reasoning'] = "English is also widely spoken"
        elif claim['id'] == 29:
            claim['reasoning'] = "Sri Lanka has maintained civilian government"
        elif claim['id'] == 30:
            claim['reasoning'] = "Population is approximately 22 million"
    
    # Save full dataset
    with open('ground_truth_dataset_full.json', 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Created full dataset with {len(full_dataset)} claims")
    
    # Print category breakdown
    categories = {}
    for claim in full_dataset:
        cat = claim['category']
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    print("Category breakdown:")
    for cat, count in categories.items():
        print(f"  {cat}: {count} claims")

if __name__ == "__main__":
    create_full_dataset()