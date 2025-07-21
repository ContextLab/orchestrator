"""Tests for real web tools functionality - uses actual web requests."""

import pytest
import asyncio
import yaml
import os
from src.orchestrator.tools.web_tools import (
    HeadlessBrowserTool,
    WebSearchTool,
    DuckDuckGoSearchBackend,
    WebScraper,
    BrowserAutomation,
    RateLimiter,
    WebCache
)


@pytest.fixture
def web_config():
    """Load web tools configuration."""
    return {
        'web_tools': {
            'search': {
                'default_backend': 'duckduckgo',
                'max_results': 5,
                'backends': {
                    'duckduckgo': {
                        'enabled': True,
                        'region': 'us-en',
                        'safe_search': 'moderate'
                    },
                    'bing': {
                        'enabled': False,
                        'api_key': ''
                    },
                    'google': {
                        'enabled': False,
                        'api_key': '',
                        'search_engine_id': ''
                    }
                }
            },
            'scraping': {
                'timeout': 10,
                'max_content_length': 1000000,
                'extract_text': True,
                'extract_links': True,
                'extract_images': True,
                'extract_metadata': True
            },
            'browser': {
                'headless': True,
                'timeout': 10,
                'browser_type': 'chromium'
            },
            'rate_limiting': {
                'enabled': True,
                'requests_per_minute': 60,
                'burst_size': 10
            },
            'caching': {
                'enabled': True,
                'ttl': 3600,
                'max_cache_size': 50
            }
        }
    }


class TestDuckDuckGoSearchBackend:
    """Test DuckDuckGo search backend."""

    @pytest.mark.asyncio
    async def test_duckduckgo_search_success(self, web_config):
        """Test successful DuckDuckGo search with real API."""
        backend_config = web_config['web_tools']['search']['backends']['duckduckgo']
        backend = DuckDuckGoSearchBackend(backend_config)
        
        # Perform real search
        results = await backend.search('Python programming language', 3)
        
        # Verify we got real results
        assert isinstance(results, list)
        assert len(results) > 0  # Should get at least some results
        
        # Check the structure of results
        for result in results:
            assert 'title' in result
            assert 'url' in result
            assert 'snippet' in result
            assert 'source' in result
            assert result['source'] == 'duckduckgo'
            
            # Verify these are real URLs
            assert result['url'].startswith('http')
            
            # Title and snippet should not be empty
            assert len(result['title']) > 0
            assert len(result['snippet']) > 0

    @pytest.mark.asyncio
    async def test_duckduckgo_search_empty_query(self, web_config):
        """Test DuckDuckGo search with empty query."""
        backend_config = web_config['web_tools']['search']['backends']['duckduckgo']
        backend = DuckDuckGoSearchBackend(backend_config)
        
        # Search with empty query should return no results or handle gracefully
        results = await backend.search('', 5)
        
        # Should either return empty list or handle the empty query gracefully
        assert isinstance(results, list)
        # Empty query typically returns no results
        assert len(results) == 0


class TestWebScraper:
    """Test web scraper functionality."""

    @pytest.mark.asyncio
    async def test_scrape_url_success(self, web_config):
        """Test successful URL scraping with real website."""
        scraper_config = web_config['web_tools']['scraping']
        scraper = WebScraper(scraper_config)
        
        # Use httpbin.org for testing - it's designed for HTTP testing
        result = await scraper.scrape_url('https://httpbin.org/html')
        
        # Verify the result structure
        assert result['url'] == 'https://httpbin.org/html'
        assert result['status_code'] == 200
        assert 'text' in result
        assert len(result['text']) > 0
        
        # httpbin.org/html returns a sample HTML page
        # Check that we got actual HTML content
        assert 'Herman Melville' in result['text']  # This page contains Moby Dick text
        
        # If title extraction is supported
        if 'title' in result:
            assert isinstance(result['title'], str)

    @pytest.mark.asyncio
    async def test_scrape_url_error(self, web_config):
        """Test URL scraping error handling with invalid URL."""
        scraper_config = web_config['web_tools']['scraping']
        scraper = WebScraper(scraper_config)
        
        # Use an invalid URL that will cause a real error
        result = await scraper.scrape_url('https://this-domain-definitely-does-not-exist-12345.com')
        
        # Should handle the error gracefully
        assert result['url'] == 'https://this-domain-definitely-does-not-exist-12345.com'
        assert 'error' in result
        # The error message will vary but should indicate connection failure
        assert len(result['error']) > 0

    @pytest.mark.asyncio
    async def test_verify_url_success(self, web_config):
        """Test successful URL verification with real URL."""
        scraper_config = web_config['web_tools']['scraping']
        scraper = WebScraper(scraper_config)
        
        # Use a reliable URL for testing
        result = await scraper.verify_url('https://httpbin.org/')
        
        # Verify the result
        assert result['url'] == 'https://httpbin.org/'
        assert result['accessible'] is True
        assert result['status_code'] == 200
        
        # Check content type if provided
        if 'content_type' in result:
            assert isinstance(result['content_type'], str)
            # httpbin.org typically returns HTML
            assert 'text/html' in result['content_type'] or 'application/json' in result['content_type']


class TestHeadlessBrowserTool:
    """Test headless browser tool."""

    @pytest.mark.asyncio
    async def test_web_search_duckduckgo(self, web_config):
        """Test web search using DuckDuckGo with real search."""
        tool = HeadlessBrowserTool(web_config)
        
        # Perform real search
        result = await tool.execute(action='search', query='OpenAI GPT', max_results=3)
        
        # Verify real results
        assert result['query'] == 'OpenAI GPT'
        assert result['total_results'] > 0
        assert len(result['results']) > 0
        
        # Check result structure
        for res in result['results']:
            assert 'title' in res
            assert 'url' in res
            assert 'snippet' in res
            assert res['url'].startswith('http')
            assert len(res['title']) > 0
            assert len(res['snippet']) > 0

    @pytest.mark.asyncio
    async def test_verify_url(self, web_config):
        """Test URL verification with real URL."""
        tool = HeadlessBrowserTool(web_config)
        
        # Verify a real, reliable URL
        result = await tool.execute(action='verify', url='https://www.google.com')
        
        # Check results
        assert result['url'] == 'https://www.google.com'
        assert result['accessible'] is True
        assert result['status_code'] == 200

    @pytest.mark.asyncio
    async def test_scrape_page(self, web_config):
        """Test page scraping with real URL."""
        tool = HeadlessBrowserTool(web_config)
        
        # Scrape a real page
        result = await tool.execute(action='scrape', url='https://httpbin.org/html')
        
        # Verify results
        assert result['url'] == 'https://httpbin.org/html'
        assert 'text' in result
        assert len(result['text']) > 0
        
        # httpbin.org/html contains Herman Melville's Moby Dick text
        assert 'Herman Melville' in result['text']
        
        # Check other fields if present
        if 'title' in result:
            assert isinstance(result['title'], str)
        if 'links' in result:
            assert isinstance(result['links'], list)
        if 'images' in result:
            assert isinstance(result['images'], list)

    @pytest.mark.asyncio
    async def test_error_handling(self, web_config):
        """Test error handling in tool execution."""
        tool = HeadlessBrowserTool(web_config)
        
        # Test invalid action
        result = await tool.execute(action='invalid_action')
        
        assert 'error' in result
        assert result['action'] == 'invalid_action'

    @pytest.mark.asyncio
    async def test_empty_query_search(self, web_config):
        """Test search with empty query."""
        tool = HeadlessBrowserTool(web_config)
        
        result = await tool.execute(action='search', query='')
        
        assert 'error' in result
        assert result['error'] == 'No query provided'
        assert result['total_results'] == 0

    @pytest.mark.asyncio
    async def test_empty_url_verify(self, web_config):
        """Test verify with empty URL."""
        tool = HeadlessBrowserTool(web_config)
        
        result = await tool.execute(action='verify', url='')
        
        assert 'error' in result
        assert result['error'] == 'No URL provided'
        assert result['accessible'] is False

    @pytest.mark.asyncio
    async def test_empty_url_scrape(self, web_config):
        """Test scrape with empty URL."""
        tool = HeadlessBrowserTool(web_config)
        
        result = await tool.execute(action='scrape', url='')
        
        assert 'error' in result
        assert result['error'] == 'No URL provided'


class TestWebSearchTool:
    """Test web search tool."""

    @pytest.mark.asyncio
    async def test_web_search_tool(self, web_config):
        """Test web search tool with real search."""
        tool = WebSearchTool(web_config)
        
        # Perform real search
        result = await tool.execute(query='artificial intelligence', max_results=2)
        
        # Verify real results
        assert result['query'] == 'artificial intelligence'
        assert result['total_results'] > 0
        assert len(result['results']) > 0
        
        # Check each result
        for res in result['results']:
            assert 'title' in res
            assert 'url' in res
            assert res['url'].startswith('http')
            assert len(res['title']) > 0


class TestRateLimiter:
    """Test rate limiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_normal_operation(self):
        """Test normal operation of rate limiter."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)
        
        # Should allow immediate requests
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        end_time = asyncio.get_event_loop().time()
        
        # Should not block on first request
        assert end_time - start_time < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_burst_limit(self):
        """Test burst limit enforcement."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=2)
        
        # Make burst_size requests rapidly
        for _ in range(2):
            await limiter.acquire()
        
        # Next request should be delayed
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        end_time = asyncio.get_event_loop().time()
        
        # Should be delayed by at least the minimum interval
        assert end_time - start_time >= 0.5  # 60/60 = 1 second, but some tolerance


class TestWebCache:
    """Test web cache."""

    def test_cache_set_get(self):
        """Test cache set and get operations."""
        cache = WebCache(max_size=10, ttl=60)
        
        # Set and get a value
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
        
        # Get non-existent key
        assert cache.get('nonexistent') is None

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache = WebCache(max_size=10, ttl=0.1)  # 0.1 second TTL
        
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
        
        # Wait for expiration
        import time
        time.sleep(0.2)
        
        assert cache.get('key1') is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when max size is reached."""
        cache = WebCache(max_size=2, ttl=60)
        
        # Fill cache to capacity
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        # Access key1 to make it more recently used
        cache.get('key1')
        
        # Add another key, should evict key2 (least recently used)
        cache.set('key3', 'value3')
        
        assert cache.get('key1') == 'value1'
        assert cache.get('key2') is None
        assert cache.get('key3') == 'value3'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])