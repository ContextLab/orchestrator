"""Test browser timeout issue."""

import asyncio
from orchestrator.tools.web_tools import HeadlessBrowserTool

async def test_browser():
    """Test browser tool with example.com."""
    tool = HeadlessBrowserTool()
    
    print("Testing scrape_js on example.com...")
    try:
        result = await tool.execute(url="https://example.com", action="scrape_js")
        print(f"Success: {result.keys()}")
        if "error" in result:
            print(f"Error: {result['error']}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")
    finally:
        await tool.stop()

asyncio.run(test_browser())