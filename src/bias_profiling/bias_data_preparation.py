import pandas as pd
from datasets import load_dataset
from typing import List, Dict
import json

class BiasDataPreparation:
    def __init__(self):
        self.target_sources = [
            'dailymirror.lk',
            'island.lk', 
            'adaderana.lk',
            'newsfirst.lk',
            'ft.lk',
            'economynext.com',
            'lankaweb.com',
            'colombopage.com',
            'sundayobserver.lk'
        ]
        
    def load_nuwan_dataset(self) -> pd.DataFrame:
        """Load and filter Nuwan's dataset for English articles from target sources"""
        print("📥 Loading Nuwan's dataset from Hugging Face...")
        
        try:
            # Load dataset
            dataset = load_dataset("nuuuwan/lk-news-docs", split="train")
            df = dataset.to_pandas()
            print(f"   ✅ Loaded {len(df)} total articles")
            
            # Filter for English articles only
            df_english = df[df['lang'] == 'en'].copy()
            print(f"   ✅ Found {len(df_english)} English articles")
            
            # Extract source from url_metadata column
            if 'url_metadata' in df_english.columns:
                df_english['source'] = df_english['url_metadata'].apply(self.extract_domain)
                print(f"   Extracted sources from URLs")
                
                # Show unique sources found
                unique_sources = df_english['source'].value_counts()
                print(f"\n📊 All sources found in dataset:")
                for source, count in unique_sources.head(15).items():
                    print(f"   {source}: {count:,} articles")
                    
            else:
                print(f"   ❌ No url_metadata column found")
                return pd.DataFrame()
            
            # Filter for target news sources
            df_filtered = df_english[df_english['source'].isin(self.target_sources)].copy()
            print(f"\n   ✅ Found {len(df_filtered)} articles from target sources")
            
            # Print target source distribution
            if not df_filtered.empty:
                print("\n📊 Articles per TARGET source:")
                source_counts = df_filtered['source'].value_counts()
                for source, count in source_counts.items():
                    print(f"   {source}: {count:,} articles")
            else:
                print("   ❌ No articles found for target sources!")
                print("   💡 Checking if target sources exist with different formats...")
                
                # Check for partial matches
                all_sources = df_english['source'].unique()
                for target in self.target_sources:
                    matches = [s for s in all_sources if target.replace('.lk', '').replace('.com', '') in s]
                    if matches:
                        print(f"   📍 Found similar to {target}: {matches}")
            
            return df_filtered
            
        except Exception as e:
            print(f"   ❌ Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            if pd.isna(url) or not url:
                return ""
            
            # Remove protocol
            if url.startswith('http'):
                url = url.split('://', 1)[1]
            
            # Get domain part
            domain = url.split('/')[0]
            
            # Remove www. if present
            if domain.startswith('www.'):
                domain = domain[4:]
                
            return domain
        except:
            return ""
    
    def save_filtered_data(self, df: pd.DataFrame, filepath: str):
        """Save filtered dataset for future use"""
        df.to_csv(filepath, index=False)
        print(f"💾 Saved filtered data to {filepath}")
        
    def load_saved_data(self, filepath: str) -> pd.DataFrame:
        """Load previously saved filtered data"""
        try:
            df = pd.read_csv(filepath)
            print(f"📂 Loaded saved data: {len(df)} articles")
            return df
        except FileNotFoundError:
            print(f"❌ No saved data found at {filepath}")
            return pd.DataFrame()

if __name__ == "__main__":
    prep = BiasDataPreparation()
    
    # Load and filter the data
    saved_path = "data/filtered_english_articles.csv"
    
    df = prep.load_saved_data(saved_path)
    if df.empty:
        df = prep.load_nuwan_dataset()
        if not df.empty:
            prep.save_filtered_data(df, saved_path)
    
    if not df.empty:
        print(f"\n🎯 Ready for next step with {len(df)} articles from {df['source'].nunique()} sources")
        print("\n✅ STEP 1 COMPLETE: Data loaded and filtered successfully!")
    else:
        print("\n❌ No data loaded. Need to check source names.")