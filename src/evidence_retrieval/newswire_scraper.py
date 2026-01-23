# src/evidence_retrieval/newswire_scraper.py

from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime
import re

class NewswireScraper(BaseScraper):
    """
    Scraper for News Wire (newswire.lk)
    Independent news portal
    """
    
    def __init__(self, output_dir: str = "data/scraped_articles"):
        super().__init__(source_name="newswire", output_dir=output_dir)
        self.base_url = "https://www.newswire.lk"
    
    def extract_article_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract article links from News Wire pages.
        """
        article_links = []

        # DEBUG: Save the HTML we're getting
        debug_file = "debug_newswire_sees.html"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        self.logger.info(f"DEBUG: Saved HTML to {debug_file}")
        
        # Method 1: Look for post/article containers
        article_containers = soup.find_all(['article', 'div'], class_=re.compile(r'(post|article|entry)', re.I))
        
        self.logger.info(f"Found {len(article_containers)} article containers")
        
        if article_containers:
            for container in article_containers:
                links = container.find_all('a', href=True)
                
                for link_tag in links:
                    href = link_tag['href']
                    
                    # Make absolute URL
                    if not href.startswith('http'):
                        href = self.base_url + href if href.startswith('/') else self.base_url + '/' + href
                    
                    # Only include if it's from newswire.lk and looks like an article
                    if 'newswire.lk' in href and not any(skip in href.lower() for skip in ['category', 'tag', 'author', 'page']):
                        if all(domain not in href for domain in ['facebook.com', 'twitter.com', 'youtube.com']):
                            article_links.append(href)
                            break
        
        # Method 2: Fallback - look for all article-like links
        if not article_links:
            self.logger.info("Method 1 failed, trying fallback method...")
            all_links = soup.find_all('a', href=True)
            
            for link_tag in all_links:
                href = link_tag['href']
                
                # Make absolute URL
                if not href.startswith('http'):
                    href = self.base_url + href if href.startswith('/') else self.base_url + '/' + href
                
                # Must be from newswire.lk and look like an article
                if 'newswire.lk' in href and not any(skip in href.lower() for skip in ['category', 'tag', 'author', 'page']):
                    # Has meaningful content in path
                    if len(href.replace(self.base_url, '').strip('/')) > 5:
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
        Extract article content from News Wire article page.
        """
        try:
            # Extract title
            title = None
            title_selectors = [
                soup.find('h1', class_=re.compile(r'(title|headline)', re.I)),
                soup.find('h1'),
                soup.find('meta', property='og:title'),
                soup.find('title')
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
                            if ' - ' in title and 'newswire' in title.lower():
                                title = title.split(' - ')[0].strip()
                    if title:
                        break
            
            if not title:
                self.logger.warning(f"Could not extract title from {url}")
                return None
            
            # Extract content
            content_text = ""
            content_selectors = [
                soup.find('div', class_=re.compile(r'(post-content|entry-content|article-content)', re.I)),
                soup.find('article'),
                soup.find('main'),
                soup.find('div', class_=re.compile(r'content', re.I))
            ]
            
            for selector in content_selectors:
                if selector:
                    # Remove unwanted elements
                    for tag in selector.find_all(['script', 'style', 'nav', 'header', 'footer']):
                        tag.decompose()
                    
                    # Get paragraphs
                    paragraphs = selector.find_all(['p'])
                    content_parts = []
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if text and len(text) > 20:
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
            
            # Extract author
            author = None
            author_selectors = [
                soup.find('span', class_=re.compile(r'author', re.I)),
                soup.find('div', class_=re.compile(r'author', re.I)),
                soup.find('meta', attrs={'name': 'author'})
            ]
            
            for selector in author_selectors:
                if selector:
                    if selector.name == 'meta':
                        author = selector.get('content', '').strip()
                    else:
                        author = selector.get_text().strip()
                    if author:
                        break
            
            word_count = len(content_text.split())
            
            return {
                'url': url,
                'title': title,
                'date': pub_date,
                'category': 'news',
                'content': content_text,
                'author': author,
                'word_count': word_count
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {e}")
            return None