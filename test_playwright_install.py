#!/usr/bin/env python3
"""Test Playwright auto-installation in HeadlessBrowserTool."""

import asyncio
from src.orchestrator.tools.web_tools import HeadlessBrowserTool

async def test_scrape_js():
    """Test JavaScript scraping with auto-install."""
    print("Testing HeadlessBrowserTool with JavaScript scraping...")
    
    tool = HeadlessBrowserTool()
    
    # This should trigger Playwright installation if needed
    result = await tool.execute(
        url="https://example.com",
        action="scrape_js"
    )
    
    print(f"Result: {result}")
    
    if "error" in result:
        print(f"Error occurred: {result['error']}")
        if "Failed to install Playwright" in result.get("error", ""):
            print("\nPlaywright installation failed. Installing manually...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
            print("Playwright installed manually. Please run the test again.")
    else:
        print("Success! JavaScript scraping is working.")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Content length: {len(result.get('content', ''))}")

if __name__ == "__main__":
    asyncio.run(test_scrape_js())