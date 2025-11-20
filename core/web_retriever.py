import requests
from typing import List, Dict, Any, Optional
import logging
from urllib.parse import quote
import time

from config.settings import settings

logger = logging.getLogger(__name__)

class WebRetriever:
    """Retrieve information from Google Search and Wikipedia"""
    
    def __init__(self):
        self.google_api_key = settings.GOOGLE_SEARCH_API_KEY
        self.google_engine_id = settings.GOOGLE_SEARCH_ENGINE_ID
        self.enable_web_search = settings.ENABLE_WEB_SEARCH
        
        # Rate limiting
        self.last_request_time = 0
        self.request_delay = 1  # seconds between requests
    
    def _rate_limit(self):
        """Implement simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def google_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform Google search using Custom Search JSON API
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        if not self.enable_web_search or not self.google_api_key or not self.google_engine_id:
            logger.warning("Google search is disabled or not configured")
            return []
        
        try:
            self._rate_limit()
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_engine_id,
                'q': query,
                'num': min(num_results, 10)  # API max is 10
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'google'
                })
            
            logger.info(f"Google search returned {len(results)} results for: {query}")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Google search request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return []
    
    def wikipedia_search(self, query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search Wikipedia for information
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of Wikipedia results
        """
        try:
            self._rate_limit()
            
            # Search Wikipedia API
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'format': 'json',
                'srlimit': num_results
            }
            
            response = requests.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            
            search_data = response.json()
            results = []
            
            for item in search_data.get('query', {}).get('search', []):
                # Get page content for the first result
                page_content = self._get_wikipedia_page_content(item['title'])
                
                results.append({
                    'title': item['title'],
                    'snippet': item['snippet'],
                    'content': page_content,
                    'source': 'wikipedia'
                })
            
            logger.info(f"Wikipedia search returned {len(results)} results for: {query}")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Wikipedia search request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []
    
    def _get_wikipedia_page_content(self, title: str) -> str:
        """
        Get content from a Wikipedia page
        
        Args:
            title: Page title
            
        Returns:
            Page content as string
        """
        try:
            self._rate_limit()
            
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'prop': 'extracts',
                'titles': title,
                'explaintext': True,
                'format': 'json',
                'exintro': True,  # Only get introduction
                'exchars': 500   # Limit characters
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            for page_id, page_data in pages.items():
                if page_id != '-1':  # Skip missing pages
                    return page_data.get('extract', 'No content available')
            
            return 'No content available'
            
        except Exception as e:
            logger.error(f"Error getting Wikipedia page content: {e}")
            return 'Error retrieving content'
    
    def search_web(self, query: str, use_google: bool = True, use_wikipedia: bool = True) -> List[Dict[str, Any]]:
        """
        Comprehensive web search combining Google and Wikipedia
        
        Args:
            query: Search query
            use_google: Whether to use Google search
            use_wikipedia: Whether to use Wikipedia
            
        Returns:
            Combined list of search results
        """
        all_results = []
        
        if use_google:
            google_results = self.google_search(query)
            all_results.extend(google_results)
        
        if use_wikipedia:
            wikipedia_results = self.wikipedia_search(query)
            all_results.extend(wikipedia_results)
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_results = []
        
        for result in all_results:
            title = result.get('title', '').lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_results.append(result)
        
        logger.info(f"Web search returned {len(unique_results)} unique results for: {query}")
        return unique_results
    
    def is_configured(self) -> bool:
        """Check if web search is properly configured"""
        return (self.enable_web_search and 
                self.google_api_key and 
                self.google_engine_id)