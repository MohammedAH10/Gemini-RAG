import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

from config.settings import settings

logger = logging.getLogger(__name__)


class SearchService:
    """Service for web search operations (Google and Wikipedia)"""

    def __init__(self):
        self.google_api_key = settings.GOOGLE_SEARCH_API_KEY
        self.google_engine_id = settings.GOOGLE_SEARCH_ENGINE_ID
        self.enable_web_search = settings.ENABLE_WEB_SEARCH

        # Rate limiting
        self.last_request_time = 0
        self.request_delay = 1  # seconds between requests

        # Cache for frequent queries (simple in-memory cache)
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

    def _rate_limit(self):
        """Implement simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()

    def _check_cache(self, query: str, search_type: str) -> Optional[List[Dict]]:
        """Check if results are cached"""
        cache_key = f"{search_type}:{query.lower()}"
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if time.time() - cached_data["timestamp"] < self._cache_ttl:
                return cached_data["results"]
        return None

    def _set_cache(self, query: str, search_type: str, results: List[Dict]):
        """Cache search results"""
        cache_key = f"{search_type}:{query.lower()}"
        self._cache[cache_key] = {"results": results, "timestamp": time.time()}

    def google_search(
        self, query: str, num_results: int = 5, use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Perform Google search using Custom Search JSON API

        Args:
            query: Search query
            num_results: Number of results to return
            use_cache: Whether to use cached results

        Returns:
            Search results
        """
        # Check cache first
        if use_cache:
            cached_results = self._check_cache(query, "google")
            if cached_results is not None:
                logger.info(f"Returning cached Google results for: {query}")
                return {
                    "success": True,
                    "results": cached_results[:num_results],
                    "source": "google",
                    "cached": True,
                    "result_count": len(cached_results[:num_results]),
                }

        if (
            not self.enable_web_search
            or not self.google_api_key
            or not self.google_engine_id
        ):
            return {
                "success": False,
                "error": "Google search is disabled or not configured",
                "results": [],
            }

        try:
            self._rate_limit()

            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_engine_id,
                "q": query,
                "num": min(num_results, 10),  # API max is 10
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get("items", []):
                result = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "google",
                    "display_link": item.get("displayLink", ""),
                    "cache_id": item.get("cacheId", ""),
                }

                # Try to get more content if snippet is too short
                if len(result["snippet"]) < 50:
                    result["snippet"] = (
                        self._get_page_preview(result["link"]) or result["snippet"]
                    )

                results.append(result)

            # Cache the results
            if use_cache and results:
                self._set_cache(query, "google", results)

            logger.info(f"Google search returned {len(results)} results for: {query}")

            return {
                "success": True,
                "results": results,
                "source": "google",
                "cached": False,
                "result_count": len(results),
                "query": query,
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Google search request failed: {e}")
            return {
                "success": False,
                "error": f"Google search request failed: {str(e)}",
                "results": [],
            }
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return {"success": False, "error": str(e), "results": []}

    def wikipedia_search(
        self, query: str, num_results: int = 3, use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Search Wikipedia for information

        Args:
            query: Search query
            num_results: Number of results to return
            use_cache: Whether to use cached results

        Returns:
            Wikipedia search results
        """
        # Check cache first
        if use_cache:
            cached_results = self._check_cache(query, "wikipedia")
            if cached_results is not None:
                logger.info(f"Returning cached Wikipedia results for: {query}")
                return {
                    "success": True,
                    "results": cached_results[:num_results],
                    "source": "wikipedia",
                    "cached": True,
                    "result_count": len(cached_results[:num_results]),
                }

        try:
            self._rate_limit()

            # Search Wikipedia API
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": num_results,
            }

            response = requests.get(search_url, params=search_params, timeout=15)
            response.raise_for_status()

            search_data = response.json()
            results = []

            for item in search_data.get("query", {}).get("search", []):
                # Get page content
                page_content = self._get_wikipedia_page_content(item["title"])

                result = {
                    "title": item["title"],
                    "snippet": item["snippet"],
                    "content": page_content,
                    "source": "wikipedia",
                    "page_id": item.get("pageid"),
                    "word_count": item.get("wordcount", 0),
                }
                results.append(result)

            # Cache the results
            if use_cache and results:
                self._set_cache(query, "wikipedia", results)

            logger.info(
                f"Wikipedia search returned {len(results)} results for: {query}"
            )

            return {
                "success": True,
                "results": results,
                "source": "wikipedia",
                "cached": False,
                "result_count": len(results),
                "query": query,
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Wikipedia search request failed: {e}")
            return {
                "success": False,
                "error": f"Wikipedia search request failed: {str(e)}",
                "results": [],
            }
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return {"success": False, "error": str(e), "results": []}

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
                "action": "query",
                "prop": "extracts",
                "titles": title,
                "explaintext": True,
                "format": "json",
                "exintro": True,  # Only get introduction
                "exchars": 800,  # Limit characters
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            pages = data.get("query", {}).get("pages", {})

            for page_id, page_data in pages.items():
                if page_id != "-1":  # Skip missing pages
                    extract = page_data.get("extract", "")
                    if extract:
                        return extract

            return "No content available"

        except Exception as e:
            logger.error(f"Error getting Wikipedia page content for '{title}': {e}")
            return "Error retrieving content"

    def _get_page_preview(self, url: str) -> Optional[str]:
        """
        Get a preview of a web page content

        Args:
            url: Page URL

        Returns:
            Page preview text or None
        """
        try:
            self._rate_limit()

            response = requests.get(
                url,
                timeout=10,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
            response.raise_for_status()

            # Simple text extraction (in production, use BeautifulSoup for better extraction)
            content = response.text[:500]  # First 500 characters
            return content

        except Exception:
            return None

    def hybrid_search(
        self,
        query: str,
        google_results: int = 3,
        wikipedia_results: int = 2,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining Google and Wikipedia

        Args:
            query: Search query
            google_results: Number of Google results
            wikipedia_results: Number of Wikipedia results
            use_cache: Whether to use cached results

        Returns:
            Combined search results
        """
        all_results = []
        sources_used = []

        # Get Google results
        google_response = self.google_search(query, google_results, use_cache)
        if google_response["success"]:
            all_results.extend(google_response["results"])
            sources_used.append("google")

        # Get Wikipedia results
        wikipedia_response = self.wikipedia_search(query, wikipedia_results, use_cache)
        if wikipedia_response["success"]:
            all_results.extend(wikipedia_response["results"])
            sources_used.append("wikipedia")

        # Remove duplicates based on title/content
        seen_content = set()
        unique_results = []

        for result in all_results:
            # Create a unique identifier for the result
            content_key = result.get("title", "") + result.get("snippet", "")[:100]
            content_hash = hash(content_key)

            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)

        # Sort by relevance (simple heuristic)
        unique_results.sort(
            key=lambda x: len(x.get("snippet", "") + x.get("content", "")), reverse=True
        )

        return {
            "success": len(unique_results) > 0,
            "results": unique_results,
            "sources": sources_used,
            "result_count": len(unique_results),
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def clear_cache(self) -> Dict[str, Any]:
        """Clear the search cache"""
        cache_size = len(self._cache)
        self._cache.clear()

        logger.info(f"Cleared search cache with {cache_size} entries")

        return {
            "success": True,
            "cleared_entries": cache_size,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            "cache_size": len(self._cache),
            "cache_ttl": self._cache_ttl,
            "cached_queries": list(self._cache.keys()),
        }

    def health_check(self) -> Dict[str, Any]:
        """Check search service health"""
        health_status = {
            "google_search": "disabled",
            "wikipedia_search": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Check Google search
        if self.enable_web_search and self.google_api_key and self.google_engine_id:
            # Simple health check by making a small query
            test_response = self.google_search("test", 1, use_cache=False)
            health_status["google_search"] = (
                "healthy" if test_response["success"] else "unhealthy"
            )

        # Check Wikipedia search
        test_response = self.wikipedia_search("test", 1, use_cache=False)
        health_status["wikipedia_search"] = (
            "healthy" if test_response["success"] else "unhealthy"
        )

        return health_status
