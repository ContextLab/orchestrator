"""Web-related tools for browser automation and search."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from .base import Tool

# Optional import for playwright - will be installed on demand
try:
    from playwright.async_api import async_playwright
except ImportError:
    async_playwright = None


class WebSearchBackend:
    """Base class for web search backends."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform a web search."""
        raise NotImplementedError


class DuckDuckGoSearchBackend(WebSearchBackend):
    """DuckDuckGo search backend using duckduckgo-search."""

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform DuckDuckGo search."""
        try:
            # Run DuckDuckGo search in a thread to avoid blocking
            loop = asyncio.get_event_loop()

            def _search():
                with DDGS() as ddgs:
                    results = list(
                        ddgs.text(
                            query,
                            region=self.config.get("region", "us-en"),
                            safesearch=self.config.get("safe_search", "moderate"),
                            timelimit=self.config.get("time_range", None),
                            max_results=max_results,
                        )
                    )
                return results

            ddg_results = await loop.run_in_executor(None, _search)

            # Convert DuckDuckGo results to our format
            results = []
            for idx, result in enumerate(ddg_results):
                results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                        "source": "duckduckgo",
                        "relevance": 1.0
                        - (idx * 0.1),  # Decreasing relevance by position
                        "date": datetime.now().isoformat(),
                        "rank": idx + 1,
                    }
                )

            return results

        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {e}")
            return []


class BingSearchBackend(WebSearchBackend):
    """Bing search backend using Bing Search API."""

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform Bing search."""
        api_key = self.config.get("api_key")
        if not api_key:
            self.logger.error("Bing API key not configured")
            return []

        endpoint = self.config.get(
            "endpoint", "https://api.bing.microsoft.com/v7.0/search"
        )

        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

        params = {
            "q": query,
            "count": min(max_results, 50),  # Bing max is 50
            "offset": 0,
            "mkt": "en-US",
            "safesearch": "Moderate",
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(endpoint, headers=headers, params=params)
                response.raise_for_status()

                data = response.json()
                web_pages = data.get("webPages", {}).get("value", [])

                results = []
                for idx, page in enumerate(web_pages):
                    results.append(
                        {
                            "title": page.get("name", ""),
                            "url": page.get("url", ""),
                            "snippet": page.get("snippet", ""),
                            "source": "bing",
                            "relevance": 1.0 - (idx * 0.1),
                            "date": page.get(
                                "dateLastCrawled", datetime.now().isoformat()
                            ),
                            "rank": idx + 1,
                        }
                    )

                return results

        except Exception as e:
            self.logger.error(f"Bing search failed: {e}")
            return []


class GoogleSearchBackend(WebSearchBackend):
    """Google search backend using Google Custom Search API."""

    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform Google search."""
        api_key = self.config.get("api_key")
        search_engine_id = self.config.get("search_engine_id")

        if not api_key or not search_engine_id:
            self.logger.error("Google API key or search engine ID not configured")
            return []

        endpoint = "https://www.googleapis.com/customsearch/v1"

        params = {
            "key": api_key,
            "cx": search_engine_id,
            "q": query,
            "num": min(max_results, 10),  # Google max is 10 per request
            "start": 1,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(endpoint, params=params)
                response.raise_for_status()

                data = response.json()
                items = data.get("items", [])

                results = []
                for idx, item in enumerate(items):
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                            "source": "google",
                            "relevance": 1.0 - (idx * 0.1),
                            "date": datetime.now().isoformat(),
                            "rank": idx + 1,
                        }
                    )

                return results

        except Exception as e:
            self.logger.error(f"Google search failed: {e}")
            return []


class WebScraper:
    """Web scraper using requests and BeautifulSoup."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Setup session with common headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": config.get(
                    "user_agent",
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                )
            }
        )

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape content from a URL."""
        try:
            # Run requests in a thread to avoid blocking
            loop = asyncio.get_event_loop()

            def _get_content():
                response = self.session.get(
                    url,
                    timeout=self.config.get("timeout", 30),
                    allow_redirects=self.config.get("follow_redirects", True),
                )
                response.raise_for_status()
                return response

            response = await loop.run_in_executor(None, _get_content)

            # Check content length
            max_length = self.config.get("max_content_length", 1048576)  # 1MB
            if len(response.content) > max_length:
                self.logger.warning(
                    f"Content too large ({len(response.content)} bytes), truncating"
                )
                content = response.content[:max_length]
            else:
                content = response.content

            # Parse HTML
            soup = BeautifulSoup(content, "lxml")

            # Extract content based on configuration
            result = {
                "url": url,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", "text/html"),
                "content_length": len(content),
                "scrape_time": datetime.now().isoformat(),
            }

            if self.config.get("extract_text", True):
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Extract text
                text = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                text = " ".join(chunk for chunk in chunks if chunk)

                result["text"] = text
                result["word_count"] = len(text.split())

            if self.config.get("extract_metadata", True):
                title = soup.find("title")
                result["title"] = title.string if title else ""

                # Extract meta tags
                meta_tags = {}
                for meta in soup.find_all("meta"):
                    name = meta.get("name") or meta.get("property")
                    content = meta.get("content")
                    if name and content:
                        meta_tags[name] = content

                result["meta_tags"] = meta_tags

                # Extract description
                description = meta_tags.get("description") or meta_tags.get(
                    "og:description"
                )
                if description:
                    result["description"] = description

            if self.config.get("extract_links", True):
                links = []
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    # Convert relative URLs to absolute
                    if not href.startswith(("http://", "https://")):
                        href = urljoin(url, href)

                    links.append(
                        {
                            "url": href,
                            "text": link.get_text().strip(),
                            "title": link.get("title", ""),
                        }
                    )

                result["links"] = links

            if self.config.get("extract_images", True):
                images = []
                for img in soup.find_all("img", src=True):
                    src = img["src"]
                    # Convert relative URLs to absolute
                    if not src.startswith(("http://", "https://")):
                        src = urljoin(url, src)

                    images.append(
                        {
                            "url": src,
                            "alt": img.get("alt", ""),
                            "title": img.get("title", ""),
                        }
                    )

                result["images"] = images

            return result

        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "scrape_time": datetime.now().isoformat(),
            }

    async def verify_url(self, url: str) -> Dict[str, Any]:
        """Verify if a URL is accessible."""
        try:
            loop = asyncio.get_event_loop()

            def _head_request():
                response = self.session.head(
                    url,
                    timeout=self.config.get("timeout", 30),
                    allow_redirects=self.config.get("follow_redirects", True),
                )
                return response

            response = await loop.run_in_executor(None, _head_request)

            return {
                "url": url,
                "accessible": True,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "content_length": response.headers.get("content-length", 0),
                "last_modified": response.headers.get("last-modified", ""),
                "verification_time": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to verify {url}: {e}")
            return {
                "url": url,
                "accessible": False,
                "error": str(e),
                "verification_time": datetime.now().isoformat(),
            }


class BrowserAutomation:
    """Browser automation using Playwright."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.playwright = None
        self.browser = None
        self.context = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self):
        """Start browser automation."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright not available. The tool should have installed it automatically."
            )

        self.playwright = await async_playwright().start()

        browser_type = self.config.get("browser_type", "chromium")
        if browser_type == "chromium":
            browser_class = self.playwright.chromium
        elif browser_type == "firefox":
            browser_class = self.playwright.firefox
        elif browser_type == "webkit":
            browser_class = self.playwright.webkit
        else:
            raise ValueError(f"Unsupported browser type: {browser_type}")

        try:
            self.browser = await browser_class.launch(
                headless=self.config.get("headless", True)
            )
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "Looks like Playwright" in str(e):
                # Browser binaries not installed, install them
                self.logger.info(f"Browser binaries not found. Installing {browser_type}...")
                import subprocess
                import sys
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "playwright", "install", browser_type]
                    )
                    # Try launching again
                    self.browser = await browser_class.launch(
                        headless=self.config.get("headless", True)
                    )
                except Exception as install_error:
                    raise RuntimeError(
                        f"Failed to install browser binaries: {install_error}. "
                        f"Please run: playwright install {browser_type}"
                    )
            else:
                raise

        viewport = self.config.get("viewport", {"width": 1920, "height": 1080})
        self.context = await self.browser.new_context(
            viewport=viewport,
            user_agent=self.config.get(
                "user_agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            ),
        )

    async def stop(self):
        """Stop browser automation."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def scrape_with_js(self, url: str) -> Dict[str, Any]:
        """Scrape content from a URL with JavaScript support."""
        if not self.browser:
            await self.start()

        try:
            page = await self.context.new_page()

            # Navigate to URL
            await page.goto(url, timeout=self.config.get("timeout", 30) * 1000)

            # Wait for page to load
            wait_time = self.config.get("wait_for_load", 2000)
            await page.wait_for_timeout(wait_time)

            # Extract content
            title = await page.title()
            content = await page.content()

            # Parse with BeautifulSoup
            soup = BeautifulSoup(content, "lxml")

            # Extract text
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            # Extract links
            links = []
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if not href.startswith(("http://", "https://")):
                    href = urljoin(url, href)
                links.append({"url": href, "text": link.get_text().strip()})

            # Extract images
            images = []
            for img in soup.find_all("img", src=True):
                src = img["src"]
                if not src.startswith(("http://", "https://")):
                    src = urljoin(url, src)
                images.append({"url": src, "alt": img.get("alt", "")})

            await page.close()

            return {
                "url": url,
                "title": title,
                "content": text,
                "links": links,
                "images": images,
                "word_count": len(text.split()),
                "scrape_time": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to scrape {url} with browser: {e}")
            return {
                "url": url,
                "error": str(e),
                "scrape_time": datetime.now().isoformat(),
            }


class RateLimiter:
    """Simple rate limiter for web requests."""

    def __init__(self, requests_per_minute: int = 30, burst_size: int = 5):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.requests = []
        self.last_request_time = 0

    async def acquire(self):
        """Acquire permission to make a request."""
        now = time.time()

        # Remove old requests (older than 1 minute)
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]

        # Check if we've exceeded the rate limit
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        # Check burst limit
        if len(self.requests) >= self.burst_size:
            sleep_time = 60 / self.requests_per_minute
            if now - self.last_request_time < sleep_time:
                await asyncio.sleep(sleep_time - (now - self.last_request_time))

        self.requests.append(now)
        self.last_request_time = now


class WebCache:
    """Simple in-memory cache for web requests."""

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}

    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry is expired."""
        if key not in self.cache:
            return True

        return time.time() - self.cache[key]["timestamp"] > self.ttl

    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        now = time.time()
        expired_keys = [
            key
            for key, value in self.cache.items()
            if now - value["timestamp"] > self.ttl
        ]

        for key in expired_keys:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]

    def _evict_lru(self):
        """Evict least recently used entries."""
        if len(self.cache) >= self.max_size:
            # Find LRU key
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[lru_key]
            del self.access_times[lru_key]

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self._cleanup_expired()

        if key in self.cache and not self._is_expired(key):
            self.access_times[key] = time.time()
            return self.cache[key]["value"]

        return None

    def set(self, key: str, value: Any):
        """Set value in cache."""
        self._cleanup_expired()
        self._evict_lru()

        self.cache[key] = {"value": value, "timestamp": time.time()}
        self.access_times[key] = time.time()


class HeadlessBrowserTool(Tool):
    """Tool for headless browser operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="headless-browser",
            description="Perform web browsing, search, and verification tasks using a headless browser",
        )
        self.add_parameter(
            "action",
            "string",
            "Action to perform: 'search', 'verify', 'scrape', 'scrape_js'",
        )
        self.add_parameter(
            "url", "string", "URL to visit (for verify/scrape)", required=False
        )
        self.add_parameter(
            "query", "string", "Search query (for search)", required=False
        )
        self.add_parameter(
            "sources",
            "array",
            "List of sources to search",
            required=False,
            default=["web"],
        )
        self.add_parameter(
            "max_results",
            "integer",
            "Maximum number of search results",
            required=False,
            default=10,
        )
        self.add_parameter(
            "backend",
            "string",
            "Search backend to use (duckduckgo, bing, google)",
            required=False,
        )

        # Initialize configuration
        self.config = config or {}
        self.web_config = self.config.get("web_tools", {})

        # Initialize search backends
        self.search_backends = {}
        self._init_search_backends()

        # Initialize scraper and browser automation
        self.scraper = WebScraper(self.web_config.get("scraping", {}))

        # Initialize rate limiter and cache
        rate_config = self.web_config.get("rate_limiting", {})
        self.rate_limiter = RateLimiter(
            requests_per_minute=rate_config.get("requests_per_minute", 30),
            burst_size=rate_config.get("burst_size", 5),
        )

        cache_config = self.web_config.get("caching", {})
        self.cache = (
            WebCache(
                max_size=cache_config.get("max_cache_size", 100),
                ttl=cache_config.get("ttl", 3600),
            )
            if cache_config.get("enabled", True)
            else None
        )

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _init_search_backends(self):
        """Initialize search backends based on configuration."""
        search_config = self.web_config.get("search", {})
        backends_config = search_config.get("backends", {})

        # Initialize DuckDuckGo backend
        if backends_config.get("duckduckgo", {}).get("enabled", True):
            self.search_backends["duckduckgo"] = DuckDuckGoSearchBackend(
                backends_config.get("duckduckgo", {})
            )

        # Initialize Bing backend
        if backends_config.get("bing", {}).get("enabled", False):
            self.search_backends["bing"] = BingSearchBackend(
                backends_config.get("bing", {})
            )

        # Initialize Google backend
        if backends_config.get("google", {}).get("enabled", False):
            self.search_backends["google"] = GoogleSearchBackend(
                backends_config.get("google", {})
            )

    def _get_search_backend(
        self, backend_name: Optional[str] = None
    ) -> WebSearchBackend:
        """Get search backend by name or default."""
        if backend_name and backend_name in self.search_backends:
            return self.search_backends[backend_name]

        # Use default backend
        default_backend = self.web_config.get("search", {}).get(
            "default_backend", "duckduckgo"
        )
        if default_backend in self.search_backends:
            return self.search_backends[default_backend]

        # Fallback to first available backend
        if self.search_backends:
            return list(self.search_backends.values())[0]

        raise ValueError("No search backends available")

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute browser action."""
        action = kwargs.get("action", "search")

        # Apply rate limiting
        if self.web_config.get("rate_limiting", {}).get("enabled", True):
            await self.rate_limiter.acquire()

        try:
            if action == "search":
                return await self._web_search(kwargs)
            elif action == "verify":
                return await self._verify_url(kwargs)
            elif action == "scrape":
                return await self._scrape_page(kwargs)
            elif action == "scrape_js":
                return await self._scrape_page_with_js(kwargs)
            else:
                raise ValueError(f"Unknown browser action: {action}")
        except Exception as e:
            self.logger.error(f"Error executing action {action}: {e}")
            return {
                "error": str(e),
                "action": action,
                "timestamp": datetime.now().isoformat(),
            }

    async def _web_search(self, kwargs) -> Dict[str, Any]:
        """Perform web search."""
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 10)
        backend_name = kwargs.get("backend")

        if not query:
            return {
                "error": "No query provided",
                "query": query,
                "results": [],
                "total_results": 0,
                "timestamp": datetime.now().isoformat(),
            }

        # Check cache first
        cache_key = f"search:{backend_name or 'default'}:{query}:{max_results}"
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.logger.info(f"Returning cached search results for: {query}")
                return cached_result

        # Get search backend
        try:
            backend = self._get_search_backend(backend_name)
        except ValueError as e:
            return {
                "error": str(e),
                "query": query,
                "results": [],
                "total_results": 0,
                "timestamp": datetime.now().isoformat(),
            }

        # Perform search
        start_time = time.time()
        results = await backend.search(query, max_results)
        search_time = time.time() - start_time

        response = {
            "query": query,
            "results": results,
            "total_results": len(results),
            "search_time": search_time,
            "backend": backend.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
        }

        # Cache results
        if self.cache:
            self.cache.set(cache_key, response)

        return response

    async def _verify_url(self, kwargs) -> Dict[str, Any]:
        """Verify URL authenticity and accessibility."""
        url = kwargs.get("url", "")

        if not url:
            return {
                "error": "No URL provided",
                "url": url,
                "accessible": False,
                "timestamp": datetime.now().isoformat(),
            }

        # Check cache first
        cache_key = f"verify:{url}"
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.logger.info(f"Returning cached verification for: {url}")
                return cached_result

        # Verify URL
        result = await self.scraper.verify_url(url)

        # Cache results
        if self.cache:
            self.cache.set(cache_key, result)

        return result

    async def _scrape_page(self, kwargs) -> Dict[str, Any]:
        """Scrape content from a web page."""
        url = kwargs.get("url", "")

        if not url:
            return {
                "error": "No URL provided",
                "url": url,
                "timestamp": datetime.now().isoformat(),
            }

        # Check cache first
        cache_key = f"scrape:{url}"
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.logger.info(f"Returning cached scraping results for: {url}")
                return cached_result

        # Scrape page
        result = await self.scraper.scrape_url(url)

        # Cache results
        if self.cache:
            self.cache.set(cache_key, result)

        return result

    async def _scrape_page_with_js(self, kwargs) -> Dict[str, Any]:
        """Scrape content from a web page with JavaScript support."""
        url = kwargs.get("url", "")

        if not url:
            return {
                "error": "No URL provided",
                "url": url,
                "timestamp": datetime.now().isoformat(),
            }

        # Check cache first
        cache_key = f"scrape_js:{url}"
        if self.cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.logger.info(f"Returning cached JS scraping results for: {url}")
                return cached_result

        # Try to import playwright and install if needed
        try:
            import importlib.util

            if importlib.util.find_spec("playwright") is None:
                raise ImportError("Playwright not available")
        except ImportError:
            self.logger.info("Playwright not installed. Installing now...")
            try:
                import subprocess
                import sys

                # Install playwright package
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "playwright"]
                )

                # Install browser binaries
                subprocess.check_call(
                    [sys.executable, "-m", "playwright", "install", "chromium"]
                )

                # Try importing again
                self.logger.info("Playwright installed successfully")
            except Exception as e:
                return {
                    "url": url,
                    "error": f"Failed to install Playwright: {str(e)}. Please install manually with: pip install playwright && playwright install chromium",
                    "timestamp": datetime.now().isoformat(),
                }

        # Scrape page with browser automation
        browser_config = self.web_config.get("browser", {})
        try:
            async with BrowserAutomation(browser_config) as browser:
                result = await browser.scrape_with_js(url)
        except Exception as e:
            # If browser automation fails, provide helpful error
            return {
                "url": url,
                "error": f"Browser automation failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
            }

        # Cache results
        if self.cache:
            self.cache.set(cache_key, result)

        return result


class WebSearchTool(Tool):
    """Simplified web search tool."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="web-search", description="Perform web search and return results"
        )
        self.add_parameter("query", "string", "Search query")
        self.add_parameter(
            "max_results",
            "integer",
            "Maximum number of results",
            required=False,
            default=10,
        )
        self.add_parameter(
            "backend",
            "string",
            "Search backend to use (duckduckgo, bing, google)",
            required=False,
        )

        # Initialize with headless browser tool
        self.browser_tool = HeadlessBrowserTool(config)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute web search."""
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 10)
        backend = kwargs.get("backend")

        # Use the headless browser tool for search
        return await self.browser_tool._web_search(
            {"query": query, "max_results": max_results, "backend": backend}
        )
