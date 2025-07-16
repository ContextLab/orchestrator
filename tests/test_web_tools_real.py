"""Tests for real web tools functionality."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
import yaml
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
        """Test successful DuckDuckGo search."""
        backend_config = web_config['web_tools']['search']['backends']['duckduckgo']
        backend = DuckDuckGoSearchBackend(backend_config)
        
        # Mock the DDGS search to avoid real API calls
        with patch('src.orchestrator.tools.web_tools.DDGS') as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.return_value = [
                {
                    'title': 'Test Result',
                    'href': 'https://example.com',
                    'body': 'Test snippet content'
                }
            ]
            
            results = await backend.search('test query', 5)
            
            assert len(results) == 1
            assert results[0]['title'] == 'Test Result'
            assert results[0]['url'] == 'https://example.com'
            assert results[0]['snippet'] == 'Test snippet content'
            assert results[0]['source'] == 'duckduckgo'

    @pytest.mark.asyncio
    async def test_duckduckgo_search_error(self, web_config):
        """Test DuckDuckGo search error handling."""
        backend_config = web_config['web_tools']['search']['backends']['duckduckgo']
        backend = DuckDuckGoSearchBackend(backend_config)
        
        # Mock the DDGS search to raise an exception
        with patch('src.orchestrator.tools.web_tools.DDGS') as mock_ddgs:
            mock_ddgs.side_effect = Exception("Search failed")
            
            results = await backend.search('test query', 5)
            
            assert results == []


class TestWebScraper:
    """Test web scraper functionality."""

    @pytest.mark.asyncio
    async def test_scrape_url_success(self, web_config):
        """Test successful URL scraping."""
        scraper_config = web_config['web_tools']['scraping']
        scraper = WebScraper(scraper_config)
        
        # Mock the requests session
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content = b'<html><head><title>Test Page</title></head><body><h1>Test Content</h1><p>This is a test page.</p></body></html>'
        
        with patch.object(scraper.session, 'get', return_value=mock_response):
            result = await scraper.scrape_url('https://example.com')
            
            assert result['url'] == 'https://example.com'
            assert result['status_code'] == 200
            assert result['title'] == 'Test Page'
            assert 'Test Content' in result['text']
            assert 'This is a test page.' in result['text']

    @pytest.mark.asyncio
    async def test_scrape_url_error(self, web_config):
        """Test URL scraping error handling."""
        scraper_config = web_config['web_tools']['scraping']
        scraper = WebScraper(scraper_config)
        
        # Mock the requests session to raise an exception
        with patch.object(scraper.session, 'get', side_effect=Exception("Network error")):
            result = await scraper.scrape_url('https://example.com')
            
            assert result['url'] == 'https://example.com'
            assert 'error' in result
            assert result['error'] == 'Network error'

    @pytest.mark.asyncio
    async def test_verify_url_success(self, web_config):
        """Test successful URL verification."""
        scraper_config = web_config['web_tools']['scraping']
        scraper = WebScraper(scraper_config)
        
        # Mock the requests session
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html', 'content-length': '1000'}
        
        with patch.object(scraper.session, 'head', return_value=mock_response):
            result = await scraper.verify_url('https://example.com')
            
            assert result['url'] == 'https://example.com'
            assert result['accessible'] is True
            assert result['status_code'] == 200
            assert result['content_type'] == 'text/html'


class TestHeadlessBrowserTool:
    """Test headless browser tool."""

    @pytest.mark.asyncio
    async def test_web_search_duckduckgo(self, web_config):
        """Test web search using DuckDuckGo."""
        tool = HeadlessBrowserTool(web_config)
        
        # Mock the DuckDuckGo search
        with patch('src.orchestrator.tools.web_tools.DDGS') as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.return_value = [
                {
                    'title': 'Test Result',
                    'href': 'https://example.com',
                    'body': 'Test snippet content'
                }
            ]
            
            result = await tool.execute(action='search', query='test query', max_results=5)
            
            assert result['query'] == 'test query'
            assert result['total_results'] == 1
            assert len(result['results']) == 1
            assert result['results'][0]['title'] == 'Test Result'

    @pytest.mark.asyncio
    async def test_verify_url(self, web_config):
        """Test URL verification."""
        tool = HeadlessBrowserTool(web_config)
        
        # Mock the scraper's verify_url method
        mock_result = {
            'url': 'https://example.com',
            'accessible': True,
            'status_code': 200
        }
        
        with patch.object(tool.scraper, 'verify_url', return_value=mock_result):
            result = await tool.execute(action='verify', url='https://example.com')
            
            assert result['url'] == 'https://example.com'
            assert result['accessible'] is True
            assert result['status_code'] == 200

    @pytest.mark.asyncio
    async def test_scrape_page(self, web_config):
        """Test page scraping."""
        tool = HeadlessBrowserTool(web_config)
        
        # Mock the scraper's scrape_url method
        mock_result = {
            'url': 'https://example.com',
            'title': 'Test Page',
            'text': 'Test content',
            'links': [],
            'images': []
        }
        
        with patch.object(tool.scraper, 'scrape_url', return_value=mock_result):
            result = await tool.execute(action='scrape', url='https://example.com')
            
            assert result['url'] == 'https://example.com'
            assert result['title'] == 'Test Page'
            assert result['text'] == 'Test content'

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
        """Test web search tool."""
        tool = WebSearchTool(web_config)
        
        # Mock the browser tool's _web_search method
        mock_result = {
            'query': 'test query',
            'results': [{'title': 'Test Result', 'url': 'https://example.com'}],
            'total_results': 1
        }
        
        with patch.object(tool.browser_tool, '_web_search', return_value=mock_result):
            result = await tool.execute(query='test query', max_results=5)
            
            assert result['query'] == 'test query'
            assert result['total_results'] == 1
            assert len(result['results']) == 1


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