# src/evidence_retrieval/island_scraper.py

from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime
import re

class IslandScraper(BaseScraper):
    """
    Scraper for The Island (island.lk)
    Independent newspaper - balanced reporting
    """
    
    def __init__(self, output_dir: str = "data/scraped_articles"):
        super().__init__(source_name="island", output_dir=output_dir)
        self.base_url = "https://island.lk"
    
    def extract_article_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract article links from The Island category pages.
        
        The Island structure:
        - Articles are typically in article tags or div containers
        - URLs follow pattern: /article-title/
        """
        article_links = []

        # DEBUG: Save the HTML we're getting
        debug_file = "debug_island_sees.html"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        self.logger.info(f"DEBUG: Saved HTML to {debug_file}")
        
        # Method 1: Find article containers - try common patterns
        article_containers = soup.find_all(['article', 'div'], class_=re.compile(r'(post|article|item|entry)', re.I))
        
        self.logger.info(f"Found {len(article_containers)} article containers")
        
        if article_containers:
            for container in article_containers:
                # Find links in this container
                links = container.find_all('a', href=True)
                
                for link_tag in links:
                    href = link_tag['href']
                    
                    # Make absolute URL
                    if not href.startswith('http'):
                        href = self.base_url + href if href.startswith('/') else self.base_url + '/' + href
                    
                    # Only include if it's from island.lk and looks like an article
                    if 'island.lk' in href and not any(skip in href.lower() for skip in ['category', 'tag', 'author', 'page']):
                        # Exclude social media and admin links
                        if all(domain not in href for domain in ['facebook.com', 'twitter.com', 'x.com', 'pinterest.com', 'linkedin.com', 'wp-admin']):
                            article_links.append(href)
                            break  # Only take first valid link from each container
        
        # Method 2: Fallback - find all links that look like articles
        if not article_links:
            self.logger.info("Method 1 failed, trying fallback method...")
            all_links = soup.find_all('a', href=True)
            
            for link_tag in all_links:
                href = link_tag['href']
                
                # Make absolute URL
                if not href.startswith('http'):
                    href = self.base_url + href if href.startswith('/') else self.base_url + '/' + href
                
                # Must be from island.lk and look like an article
                if 'island.lk' in href and not any(skip in href.lower() for skip in ['category', 'tag', 'author', 'page', 'wp-admin']):
                    # Has content in URL path (not just domain)
                    if len(href.replace(self.base_url, '').strip('/')) > 3:
                        if all(domain not in href for domain in ['facebook.com', 'twitter.com', 'x.com']):
                            article_links.append(href)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in article_links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        self.logger.info(f"Extracted {len(unique_links)} unique article links")
        for i, link in enumerate(unique_links[:5]):  # Log first 5 for debugging
            self.logger.info(f"  {i+1}: {link}")
            
        return unique_links
    
    def extract_article_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict]:
        """
        Extract article content from The Island article page.
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
                    elif selector.name == 'title':
                        title = selector.get_text().strip()
                        # Clean up title (remove site name)
                        if ' | ' in title:
                            title = title.split(' | ')[0].strip()
                        if ' - ' in title:
                            title = title.split(' - ')[0].strip()
                    else:
                        title = selector.get_text().strip()
                    if title:
                        break
            
            if not title:
                self.logger.warning(f"Could not extract title from {url}")
                return None
                
            # Extract date
            pub_date = None
            date_selectors = [
                soup.find('time'),
                soup.find(attrs={'datetime': True}),
                soup.find('meta', property='article:published_time'),
                soup.find('span', class_=re.compile(r'date', re.I)),
                soup.find('div', class_=re.compile(r'date', re.I))
            ]
            
            for selector in date_selectors:
                if selector:
                    if selector.name == 'meta':
                        pub_date = selector.get('content', '')
                    elif selector.get('datetime'):
                        pub_date = selector.get('datetime')
                    else:
                        pub_date = selector.get_text().strip()
                    
                    if pub_date:
                        try:
                            # Try to parse various date formats
                            if 'T' in pub_date:
                                pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                            else:
                                # Try common formats
                                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%B %d, %Y']:
                                    try:
                                        pub_date = datetime.strptime(pub_date, fmt)
                                        break
                                    except:
                                        continue
                            break
                        except:
                            continue
            
            # Extract content
            content_text = ""
            content_selectors = [
                soup.find('div', class_=re.compile(r'(content|article|post-content|entry-content)', re.I)),
                soup.find('article'),
                soup.find('main')
            ]
            
            for selector in content_selectors:
                if selector:
                    # Remove unwanted elements
                    for tag in selector.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                        tag.decompose()
                    
                    # Get text content
                    paragraphs = selector.find_all(['p', 'div'])
                    content_parts = []
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if text and len(text) > 20:  # Filter out short/empty content
                            content_parts.append(text)
                    
                    content_text = '\n\n'.join(content_parts)
                    if content_text and len(content_text) > 100:
                        break
            
            if not content_text:
                self.logger.warning(f"Could not extract content from {url}")
                return None
            
            # Extract author
            author = None
            author_selectors = [
                soup.find('span', class_=re.compile(r'author', re.I)),
                soup.find('div', class_=re.compile(r'author', re.I)),
                soup.find('meta', attrs={'name': 'author'}),
                soup.find(attrs={'rel': 'author'})
            ]
            
            for selector in author_selectors:
                if selector:
                    if selector.name == 'meta':
                        author = selector.get('content', '').strip()
                    else:
                        author = selector.get_text().strip()
                    if author:
                        break
            
            # Calculate word count
            word_count = len(content_text.split())
            
            return {
                'url': url,
                'title': title,
                'date': pub_date.isoformat() if isinstance(pub_date, datetime) else pub_date,
                'category': 'general',  # Default category
                'content': content_text,
                'author': author,
                'word_count': word_count
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {e}")
            return None