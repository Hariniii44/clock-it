import json
import pandas as pd
from typing import Dict, List, Set

class SriLankanPoliticians:
    def __init__(self):
        self.politicians = {}
        self._build_politician_list()
    
    def _build_politician_list(self):
        """Build comprehensive list of Sri Lankan politicians with party affiliations"""
        
        # Current Government (NPP/JVP) - Pro-government
        npn_politicians = [
            "Anura Kumara Dissanayake", "AKD", "Anura Dissanayake",
            "Harini Amarasuriya", "Harini", 
            "Vijitha Herath", "Vijitha",
            "Tilvin Silva", "Tilvin",
            "Bimal Ratnayake", "Bimal",
            "Sunil Handunnetti", "Sunil Handunnetti",
            "Nalaka Godahewa", "Nalaka",
            "Wasantha Mudalige", "Wasantha",
            "Nihal Abeysinghe", "Nihal Abeysinghe",
            "Chamindra Weerawardhana", "Chamindra"
        ]
        
        # Main Opposition Parties - Opposition
        opposition_politicians = [
            # SJB (Samagi Jana Balawegaya)
            "Sajith Premadasa", "Sajith", "Premadasa",
            "Eran Wickramaratne", "Eran",
            "Harsha de Silva", "Harsha",
            "Patali Champika Ranawaka", "Champika Ranawaka", "Patali Champika",
            "Rohini Kaviratne", "Rohini",
            "Mujibur Rahman", "Mujibur",
            "Nalin Bandara", "Nalin Bandara",
            "Diana Gamage", "Diana",
            "Kabir Hashim", "Kabir",
            "Ajith Mannapperuma", "Ajith Mannapperuma",
            
            # SLPP (Sri Lanka Podujana Peramuna) - Former ruling party
            "Mahinda Rajapaksa", "Mahinda", "MR",
            "Gotabaya Rajapaksa", "Gotabaya", "Gota", "GR",
            "Basil Rajapaksa", "Basil",
            "Namal Rajapaksa", "Namal",
            "Chamal Rajapaksa", "Chamal",
            "Dinesh Gunawardena", "Dinesh Gunawardena",
            "G.L. Peiris", "GL Peiris", "Professor Peiris",
            "Dullas Alahapperuma", "Dullas",
            "Johnston Fernando", "Johnston",
            "Wimal Weerawansa", "Wimal",
            "Udaya Gammanpila", "Udaya Gammanpila",
            
            # UNP (United National Party)
            "Ranil Wickremesinghe", "Ranil", "RW",
            "Vajira Abeywardena", "Vajira",
            "Ruwan Wijewardene", "Ruwan Wijewardene",
            "Malik Samarawickrama", "Malik",
            
            # Other Opposition
            "Maithripala Sirisena", "Maithripala", "MS",
            "Nimal Siripala de Silva", "Nimal Siripala",
            "Sarath Fonseka", "Sarath", "Field Marshal Fonseka",
            "Chandrika Bandaranaike Kumaratunga", "Chandrika", "CBK",
            "Karu Jayasuriya", "Karu",
            "Mangala Samaraweera", "Mangala"
        ]
        
        # Assign party affiliations
        for politician in npn_politicians:
            self.politicians[politician.lower()] = "government"
            
        for politician in opposition_politicians:
            self.politicians[politician.lower()] = "opposition"
            
        print(f"📝 Built politician database with {len(self.politicians)} names")
        print(f"   Government: {sum(1 for p in self.politicians.values() if p == 'government')} politicians")
        print(f"   Opposition: {sum(1 for p in self.politicians.values() if p == 'opposition')} politicians")
    
    def get_all_politicians(self) -> Dict[str, str]:
        """Return all politicians with their affiliations"""
        return self.politicians
    
    def get_government_politicians(self) -> List[str]:
        """Return list of government politicians"""
        return [name for name, party in self.politicians.items() if party == "government"]
    
    def get_opposition_politicians(self) -> List[str]:
        """Return list of opposition politicians"""
        return [name for name, party in self.politicians.items() if party == "opposition"]
    
    def save_politicians_list(self, filepath: str):
        """Save politicians list to JSON file"""
        data = {
            "politicians": self.politicians,
            "government": self.get_government_politicians(),
            "opposition": self.get_opposition_politicians(),
            "total_count": len(self.politicians)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved politician names to {filepath}")
    
    def filter_political_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter articles containing politician names"""
        print(f"🔍 Filtering political articles from {len(df)} total articles...")
        
        # Create pattern for politician names
        politician_names = list(self.politicians.keys())
        
        def contains_politician(text):
            """Check if text contains any politician name"""
            if pd.isna(text):
                return False
            text_lower = str(text).lower()
            return any(politician in text_lower for politician in politician_names)
        
        # Filter articles containing politicians in title OR description
        political_mask = (
            df['description'].apply(contains_politician) |
            df.get('title', pd.Series([False] * len(df))).apply(contains_politician)
        )
        
        df_political = df[political_mask].copy()
        
        print(f"   ✅ Found {len(df_political)} political articles")
        print(f"   📊 Political coverage: {len(df_political)/len(df)*100:.1f}% of all articles")
        
        # Show distribution by source
        if not df_political.empty:
            print(f"\n📊 Political articles per source:")
            source_counts = df_political['source'].value_counts()
            for source, count in source_counts.items():
                percentage = count / len(df_political) * 100
                print(f"   {source}: {count:,} articles ({percentage:.1f}%)")
        
        return df_political

if __name__ == "__main__":
    # Create politician database
    pol_db = SriLankanPoliticians()
    
    # Save politician list
    pol_db.save_politicians_list("data/sri_lankan_politicians.json")
    
    # Load the filtered articles from Step 1
    try:
        df = pd.read_csv("data/filtered_english_articles.csv")
        print(f"\n📂 Loaded {len(df)} articles from the LK news dataset")
        
        # Filter for political articles
        df_political = pol_db.filter_political_articles(df)
        
        # Save political articles
        if not df_political.empty:
            df_political.to_csv("data/political_articles.csv", index=False)
            print(f"\n💾 Saved {len(df_political)} political articles to data/political_articles.csv")
            print(f"\n🎯 Ready for Step 3 with {len(df_political)} political articles from {df_political['source'].nunique()} sources")
            print("\n✅ STEP 2 COMPLETE: Political articles identified and saved!")
        else:
            print("\n❌ No political articles found!")
            
    except FileNotFoundError:
        print("❌ Please run Step 1 first to create filtered_english_articles.csv")