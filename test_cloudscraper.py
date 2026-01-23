#!/usr/bin/env python3
"""
Test cloudscraper vs regular requests
"""

import requests
import cloudscraper
from bs4 import BeautifulSoup
import socket

def force_ipv4():
    old_getaddrinfo = socket.getaddrinfo
    def new_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return old_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
    socket.getaddrinfo = new_getaddrinfo

def test_cloudscraper():
    """Test cloudscraper"""
    print("Testing CloudScraper...")
    
    force_ipv4()
    
    # Try main page first (might be less protected)
    url = "https://dailynews.lk/"
    
    try:
        scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'mobile': False
            },
            delay=2
        )
        
        response = scraper.get(url, timeout=15)
        
        print(f"Status: {response.status_code}")
        print(f"Content length: {len(response.content)} bytes")
        print(f"Text length: {len(response.text)} chars")
        print(f"Content starts with: {response.text[:100]}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            print(f"Parsed tags: {len(soup.find_all())}")
            
            title = soup.find('title')
            if title:
                print(f"Title: {title.get_text()}")
            
            # Try politics category now
            print("\nTrying politics category...")
            politics_url = "https://dailynews.lk/category/politics/"
            politics_response = scraper.get(politics_url, timeout=15)
            print(f"Politics status: {politics_response.status_code}")
            
            if politics_response.status_code == 200:
                politics_soup = BeautifulSoup(politics_response.text, 'html.parser')
                print(f"Politics tags: {len(politics_soup.find_all())}")
                
                # Save for debug
                with open('debug_politics_page.html', 'w', encoding='utf-8') as f:
                    f.write(politics_response.text)
                print("Saved politics page to debug_politics_page.html")
                
                # Look for article elements
                articles = politics_soup.find_all('article')
                print(f"Found {len(articles)} article tags")
                
                # Also check for common article selectors
                links = politics_soup.find_all('a', href=True)
                article_links = [a for a in links if '/2026/' in str(a.get('href', '')) or '/2025/' in str(a.get('href', ''))]
                print(f"Found {len(article_links)} potential article links")
                
                return True
            else:
                print(f"Failed to get politics page: {politics_response.status_code}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"CloudScraper error: {e}")
        return False

if __name__ == "__main__":
    test_cloudscraper()