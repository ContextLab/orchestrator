#!/usr/bin/env python3
"""
Example script demonstrating real web tools functionality.

This script shows how to use the updated web tools with real web search and content extraction.
"""

import asyncio

# Add the src directory to the path
from src.orchestrator.tools.web_tools import HeadlessBrowserTool, WebSearchTool


async def load_config():
    """Load configuration or use defaults."""
    # Use default config for testing
    config = {"web_search": {"enabled": True, "max_results": 5, "timeout": 30}}
    return config


async def test_web_search():
    """Test web search functionality."""
    print("🔍 Testing Web Search Functionality")
    print("=" * 50)

    config = await load_config()

    # Initialize web search tool
    search_tool = WebSearchTool(config)

    # Test DuckDuckGo search
    print("\n1. Testing DuckDuckGo Search:")
    try:
        result = await search_tool.execute(query="Python web scraping", max_results=3)

        print(f"   Query: {result['query']}")
        print(f"   Total results: {result['total_results']}")
        print(f"   Search time: {result.get('search_time', 0):.2f}s")
        print(f"   Backend: {result.get('backend', 'Unknown')}")

        for i, res in enumerate(result["results"][:2], 1):
            print(f"   Result {i}: {res['title']}")
            print(f"   URL: {res['url']}")
            print(f"   Snippet: {res['snippet'][:100]}...")
            print()

    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 50)


async def test_web_scraping():
    """Test web scraping functionality."""
    print("🌐 Testing Web Scraping Functionality")
    print("=" * 50)

    config = await load_config()

    # Initialize headless browser tool
    browser_tool = HeadlessBrowserTool(config)

    # Test URL verification
    print("\n1. Testing URL Verification:")
    test_url = "https://httpbin.org/html"

    try:
        result = await browser_tool.execute(action="verify", url=test_url)

        print(f"   URL: {result['url']}")
        print(f"   Accessible: {result['accessible']}")
        print(f"   Status Code: {result.get('status_code', 'Unknown')}")
        print(f"   Content Type: {result.get('content_type', 'Unknown')}")

    except Exception as e:
        print(f"   Error: {e}")

    # Test web scraping
    print("\n2. Testing Web Scraping:")

    try:
        result = await browser_tool.execute(action="scrape", url=test_url)

        print(f"   URL: {result['url']}")
        print(f"   Title: {result.get('title', 'No title')}")
        print(f"   Content length: {len(result.get('text', ''))}")
        print(f"   Word count: {result.get('word_count', 0)}")
        print(f"   Links found: {len(result.get('links', []))}")
        print(f"   Images found: {len(result.get('images', []))}")

        # Show first few lines of content
        content = result.get("text", "")
        if content:
            lines = content.split("\n")[:3]
            print("   Content preview:")
            for line in lines:
                if line.strip():
                    print(f"     {line.strip()[:80]}...")

    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 50)


async def test_browser_search():
    """Test browser-based search functionality."""
    print("🤖 Testing Browser Search Functionality")
    print("=" * 50)

    config = await load_config()

    # Initialize headless browser tool
    browser_tool = HeadlessBrowserTool(config)

    print("\n1. Testing Browser Search:")

    try:
        result = await browser_tool.execute(
            action="search", query="OpenAI ChatGPT", max_results=3
        )

        print(f"   Query: {result['query']}")
        print(f"   Total results: {result['total_results']}")
        print(f"   Search time: {result.get('search_time', 0):.2f}s")
        print(f"   Backend: {result.get('backend', 'Unknown')}")

        for i, res in enumerate(result["results"][:2], 1):
            print(f"   Result {i}: {res['title']}")
            print(f"   URL: {res['url']}")
            print(f"   Relevance: {res.get('relevance', 'Unknown')}")
            print(f"   Snippet: {res['snippet'][:100]}...")
            print()

    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 50)


async def test_caching():
    """Test caching functionality."""
    print("🗄️ Testing Caching Functionality")
    print("=" * 50)

    config = await load_config()

    # Initialize web search tool
    search_tool = WebSearchTool(config)

    print("\n1. Testing Search Caching:")

    query = "machine learning"

    try:
        # First search (should hit the web)
        print("   First search (should hit backend):")
        import time

        start_time = time.time()
        result1 = await search_tool.execute(query=query, max_results=2)
        first_time = time.time() - start_time

        print(f"   Time: {first_time:.2f}s")
        print(f"   Results: {result1['total_results']}")

        # Second search (should hit cache)
        print("   Second search (should hit cache):")
        start_time = time.time()
        result2 = await search_tool.execute(query=query, max_results=2)
        second_time = time.time() - start_time

        print(f"   Time: {second_time:.2f}s")
        print(f"   Results: {result2['total_results']}")

        # Check if caching improved performance
        if second_time < first_time:
            print("   ✓ Caching appears to be working (faster second request)")
        else:
            print("   ⚠️ Caching may not be working (similar response times)")

    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 50)


async def test_error_handling():
    """Test error handling."""
    print("⚠️ Testing Error Handling")
    print("=" * 50)

    config = await load_config()

    # Initialize tools
    browser_tool = HeadlessBrowserTool(config)
    search_tool = WebSearchTool(config)

    print("\n1. Testing Invalid URL:")

    try:
        result = await browser_tool.execute(action="verify", url="not-a-valid-url")

        if "error" in result:
            print(f"   ✓ Error handled correctly: {result['error']}")
        else:
            print("   ⚠️ No error reported for invalid URL")

    except Exception as e:
        print(f"   ✓ Exception handled: {e}")

    print("\n2. Testing Empty Query:")

    try:
        result = await search_tool.execute(query="")

        if "error" in result:
            print(f"   ✓ Error handled correctly: {result['error']}")
        else:
            print("   ⚠️ No error reported for empty query")

    except Exception as e:
        print(f"   ✓ Exception handled: {e}")

    print("\n3. Testing Invalid Action:")

    try:
        result = await browser_tool.execute(action="invalid_action")

        if "error" in result:
            print(f"   ✓ Error handled correctly: {result['error']}")
        else:
            print("   ⚠️ No error reported for invalid action")

    except Exception as e:
        print(f"   ✓ Exception handled: {e}")

    print("\n" + "=" * 50)


async def main():
    """Run all tests."""
    print("🚀 Real Web Tools Functionality Test")
    print("=" * 50)
    print("This script demonstrates the updated web tools with real functionality.")
    print("The tools now use actual web search APIs and content extraction.")
    print()

    # Run all tests
    await test_web_search()
    await test_web_scraping()
    await test_browser_search()
    await test_caching()
    await test_error_handling()

    print("\n✅ All tests completed!")
    print("\nThe web tools now provide real functionality:")
    print("- DuckDuckGo search integration")
    print("- Real web content scraping with BeautifulSoup")
    print("- Headless browser automation with Playwright")
    print("- Rate limiting and caching")
    print("- Comprehensive error handling")
    print("- Support for multiple search backends")


if __name__ == "__main__":
    asyncio.run(main())
