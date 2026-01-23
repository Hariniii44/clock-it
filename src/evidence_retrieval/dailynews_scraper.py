# src/evidence_retrieval/dailynews_scraper.py

from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from datetime import datetime
import re

class DailyNewsScraper(BaseScraper):
    """
    Scraper for Daily News (dailynews.lk)
    Pro-government aligned news source
    """
    
    def __init__(self, output_dir: str = "data/scraped_articles"):
        super().__init__(source_name="dailynews", output_dir=output_dir)
        self.base_url = "https://dailynews.lk"
    
    def extract_article_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract article links from Daily News category page.
        
        Daily News structure:
        - Articles are in <article> tags with classes like "item hentry"
        - Main link is in <a> tag inside the article
        - URLs follow pattern: /YYYY/MM/DD/category/article-id/title/
        """
        article_links = []

        # DEBUG: Save the HTML we're actually getting
        debug_file = "debug_scraper_sees.html"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        self.logger.info(f"DEBUG: Saved HTML to {debug_file}")
        
        # Also check what soup type we have
        self.logger.info(f"DEBUG: soup type = {type(soup)}")
        self.logger.info(f"DEBUG: soup has {len(soup.find_all())} total tags")
        
        
        # Method 1: Find all <article> tags with class "item hentry"
        articles = soup.find_all('article', class_='item hentry')
        
        self.logger.info(f"Found {len(articles)} article tags")
        
        if articles:
            for article in articles:
                # Find the main article link - look for h2 with entry-title class
                title_heading = article.find('h2', class_='penci-entry-title entry-title')
                if title_heading:
                    link_tag = title_heading.find('a', href=True)
                    if link_tag:
                        href = link_tag['href']
                        
                        # Only include if it's a politics article URL (has /politics/ in path)
                        if '/politics/' in href and 'dailynews.lk' in href:
                            # Exclude social media share links
                            if all(domain not in href for domain in ['facebook.com', 'twitter.com', 'x.com', 'pinterest.com', 'linkedin.com']):
                                article_links.append(href)
        
        # Method 2: Fallback - find ALL links with politics pattern if Method 1 fails
        if not article_links:
            self.logger.info("Method 1 failed, trying fallback method...")
            all_links = soup.find_all('a', href=True)
            
            for link_tag in all_links:
                href = link_tag['href']
                
                # Must have date pattern and be politics from dailynews.lk
                if re.search(r'/\d{4}/\d{2}/\d{2}/politics/', href) and 'dailynews.lk' in href:
                    # Exclude social media
                    if all(domain not in href for domain in ['facebook.com', 'twitter.com', 'x.com', 'pinterest.com', 'linkedin.com']):
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
        Extract article content from Daily News article page.
        
        Returns dictionary with:
        - url: Article URL
        - title: Article title
        - date: Publication date
        - category: Article category
        - content: Full article text
        - author: Author name (if available)
        """
        try:
            # Extract title
            title = None
            title_selectors = [
                soup.find('h1', class_=lambda x: x and 'title' in x.lower()),
                soup.find('h1', class_='entry-title'),
                soup.find('h1'),
                soup.find('meta', property='og:title')
            ]
            
            for selector in title_selectors:
                if selector:
                    if selector.name == 'meta':
                        title = selector.get('content', '').strip()
                    else:
                        title = selector.get_text(strip=True)
                    if title:
                        break
            
            if not title:
                self.logger.warning(f"No title found for {url}")
                return None
            
            # Extract date
            date = None
            date_selectors = [
                soup.find('time', datetime=True),
                soup.find('span', class_=lambda x: x and 'date' in x.lower()),
                soup.find('meta', property='article:published_time')
            ]
            
            for selector in date_selectors:
                if selector:
                    if selector.name == 'time':
                        date = selector.get('datetime', '')
                    elif selector.name == 'meta':
                        date = selector.get('content', '')
                    else:
                        date_text = selector.get_text(strip=True)
                        date = self._parse_date(date_text)
                    if date:
                        break
            
            # Extract from URL if date not found
            if not date:
                date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
                if date_match:
                    year, month, day = date_match.groups()
                    date = f"{year}-{month}-{day}"
            
            # Extract category from URL
            category = "politics"  # Default
            category_match = re.search(r'dailynews\.lk/\d{4}/\d{2}/\d{2}/([^/]+)/', url)
            if category_match:
                category = category_match.group(1)
            
            # Extract content
            content = None
            content_selectors = [
                soup.find('div', class_=lambda x: x and 'content' in x.lower()),
                soup.find('div', class_='entry-content'),
                soup.find('article'),
                soup.find('div', itemprop='articleBody')
            ]
            
            for selector in content_selectors:
                if selector:
                    # Remove script, style, and ad elements
                    for tag in selector.find_all(['script', 'style', 'iframe', 'ins']):
                        tag.decompose()
                    
                    # Get text
                    paragraphs = selector.find_all('p')
                    if paragraphs:
                        content = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                    
                    if content and len(content) > 100:  # Minimum length check
                        break
            
            if not content:
                self.logger.warning(f"No content found for {url}")
                return None
            
            # Extract author
            author = None
            author_selectors = [
                soup.find('span', class_=lambda x: x and 'author' in x.lower()),
                soup.find('a', rel='author'),
                soup.find('meta', attrs={'name': 'author'})
            ]
            
            for selector in author_selectors:
                if selector:
                    if selector.name == 'meta':
                        author = selector.get('content', '').strip()
                    else:
                        author = selector.get_text(strip=True)
                    if author:
                        break
            
            return {
                'url': url,
                'title': title,
                'date': date,
                'category': category,
                'content': content,
                'author': author if author else 'Unknown',
                'word_count': len(content.split())
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def _parse_date(self, date_text: str) -> Optional[str]:
        """
        Parse date from various text formats to YYYY-MM-DD.
        
        Examples:
        - "January 16, 2026" -> "2026-01-16"
        - "16 Jan 2026" -> "2026-01-16"
        """
        try:
            # Try common formats
            formats = [
                '%B %d, %Y',      # January 16, 2026
                '%d %B %Y',       # 16 January 2026
                '%b %d, %Y',      # Jan 16, 2026
                '%d %b %Y',       # 16 Jan 2026
                '%Y-%m-%d',       # 2026-01-16
                '%d/%m/%Y',       # 16/01/2026
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_text, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            return None
        except:
            return None


# Convenience function
def scrape_dailynews(max_articles: int = 50):
    """
    Convenience function to scrape Daily News politics section.
    
    Args:
        max_articles: Maximum number of articles to scrape
    """
    scraper = DailyNewsScraper()
    scraper.scrape(
        section_url="https://dailynews.lk/category/politics/",
        max_articles=max_articles
    )

