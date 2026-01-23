# scripts/test_dailynews_scraper.py

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evidence_retrieval.dailynews_scraper import DailyNewsScraper

def test_scraper():
    """Test Daily News scraper with 20 articles"""
    
    print("=" * 60)
    print("TESTING DAILY NEWS SCRAPER")
    print("=" * 60)
    print()
    
    # Initialize scraper
    scraper = DailyNewsScraper()
    
    # Test with 20 articles
    print("Scraping 20 articles from Daily News politics section...")
    print()
    
    scraper.scrape(
        section_url="https://dailynews.lk/category/politics/",
        max_articles=20
    )
    
    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print()
    print("Check the following:")
    print(f"1. CSV file: data/scraped_articles/dailynews_articles.csv")
    print(f"2. Log file: logs/dailynews_scraper.log")
    print()
    print("If you see articles in the CSV, the scraper works! ✓")

if __name__ == "__main__":
    test_scraper()