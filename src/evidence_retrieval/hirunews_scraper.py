# src/evidence_retrieval/hirunews_scraper.py

from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime
import re

class HirunewsScraper(BaseScraper):
    """
    Scraper for Hiru News (hirunews.lk)
    Popular TV/web news portal
    """
    
    def __init__(self, output_dir: str = "data/scraped_articles"):
        super().__init__(source_name="hirunews", output_dir=output_dir)
        self.base_url = "https://www.hirunews.lk"
    
    def extract_article_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract article links from Hiru News pages.
        """
        article_links = []

        # DEBUG: Save the HTML we're getting
        debug_file = "debug_hirunews_sees.html"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        self.logger.info(f"DEBUG: Saved HTML to {debug_file}")
        
        # Method 1: Look for news/article containers
        article_containers = soup.find_all(['div', 'article'], class_=re.compile(r'(news|post|article|item)', re.I))
        
        self.logger.info(f"Found {len(article_containers)} potential article containers")
        
        if article_containers:
            for container in article_containers:
                links = container.find_all('a', href=True)
                
                for link_tag in links:
                    href = link_tag['href']
                    
                    # Make absolute URL
                    if not href.startswith('http'):
                        href = self.base_url + href if href.startswith('/') else self.base_url + '/' + href
                    
                    # Only include if it's from hirunews.lk and looks like a news article
                    if 'hirunews.lk' in href and not any(skip in href.lower() for skip in ['category', 'tag']):
                        if all(domain not in href for domain in ['facebook.com', 'twitter.com', 'youtube.com']):
                            article_links.append(href)
                            break
        
        # Method 2: Look for all meaningful links
        if not article_links:
            self.logger.info("Method 1 failed, trying fallback method...")
            all_links = soup.find_all('a', href=True)
            
            for link_tag in all_links:
                href = link_tag['href']
                
                # Make absolute URL
                if not href.startswith('http'):
                    href = self.base_url + href if href.startswith('/') else self.base_url + '/' + href
                
                # Look for article-like URLs
                if 'hirunews.lk' in href and not any(skip in href.lower() for skip in ['category', 'tag', 'page']):
                    # Has meaningful content in path
                    path = href.replace(self.base_url, '').strip('/')
                    if len(path) > 5 and '/' in path:  # Likely an article with path structure
                        if all(domain not in href for domain in ['facebook.com', 'twitter.com', 'youtube.com']):
                            article_links.append(href)
        
        # Remove duplicates
        seen = set()
        unique_links = []
        for link in article_links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        self.logger.info(f"Extracted {len(unique_links)} unique article links")
        for i, link in enumerate(unique_links[:5]):
            self.logger.info(f"  {i+1}: {link}")
            
        return unique_links
    
    def extract_article_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict]:
        """
        Extract article content from Hiru News article page.
        """
        try:
            # Extract title
            title = None
            title_selectors = [
                soup.find('h1'),
                soup.find('meta', property='og:title'),
                soup.find('title'),
                soup.find('h2', class_=re.compile(r'title', re.I))
            ]
            
            for selector in title_selectors:
                if selector:
                    if selector.name == 'meta':
                        title = selector.get('content', '').strip()
                    else:
                        title = selector.get_text().strip()
                        if selector.name == 'title':
                            # Clean up title
                            if ' | ' in title:
                                title = title.split(' | ')[0].strip()
                            if 'hiru news' in title.lower():
                                title = title.replace('Hiru News', '').replace('hiru news', '').strip(' -|')
                    if title:
                        break
            
            if not title:
                self.logger.warning(f"Could not extract title from {url}")
                return None
            
            # Extract content
            content_text = ""
            content_selectors = [
                soup.find('div', class_=re.compile(r'(content|article|post)', re.I)),
                soup.find('article'),
                soup.find('main'),
                soup.find('div', id=re.compile(r'content', re.I))
            ]
            
            for selector in content_selectors:
                if selector:
                    # Remove unwanted elements
                    for tag in selector.find_all(['script', 'style', 'iframe', 'nav', 'header', 'footer']):
                        tag.decompose()
                    
                    # Get text from paragraphs
                    paragraphs = selector.find_all(['p', 'div'])
                    content_parts = []
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if text and len(text) > 15:  # Filter out very short content
                            content_parts.append(text)
                    
                    content_text = '\n\n'.join(content_parts)
                    if content_text and len(content_text) > 100:
                        break
            
            if not content_text:
                self.logger.warning(f"Could not extract content from {url}")
                return None
            
            # Extract date
            pub_date = None
            date_selectors = [
                soup.find('time'),
                soup.find('span', class_=re.compile(r'date', re.I)),
                soup.find('div', class_=re.compile(r'date', re.I)),
                soup.find('meta', property='article:published_time')
            ]
            
            for selector in date_selectors:
                if selector:
                    if selector.name == 'meta':
                        pub_date = selector.get('content', '')
                    else:
                        pub_date = selector.get_text().strip()
                    if pub_date:
                        break
            
            word_count = len(content_text.split())
            
            return {
                'url': url,
                'title': title,
                'date': pub_date,
                'category': 'news',
                'content': content_text,
                'author': None,
                'word_count': word_count
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {e}")
            return None