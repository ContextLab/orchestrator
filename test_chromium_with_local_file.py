#!/usr/bin/env python3
"""Test HeadlessBrowserTool with local HTML file to avoid network issues."""

import asyncio
import os
import tempfile
from pathlib import Path

async def test_js_scraping_local():
    """Test JavaScript scraping with a local HTML file."""
    # Create a test HTML file with JavaScript
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Original Content</h1>
        <div id="dynamic-content">Loading...</div>
        
        <script>
            // Simulate dynamic content loading
            setTimeout(() => {
                document.getElementById('dynamic-content').innerHTML = 
                    '<p>This content was added by JavaScript!</p>' +
                    '<ul>' +
                    '<li>Item 1</li>' +
                    '<li>Item 2</li>' +
                    '<li>Item 3</li>' +
                    '</ul>';
                document.title = 'JavaScript Loaded Successfully';
            }, 100);
        </script>
    </body>
    </html>
    """
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(html_content)
        test_file = f.name
    
    try:
        print("=== Testing HeadlessBrowserTool with Local File ===\n")
        
        from src.orchestrator.tools.web_tools import HeadlessBrowserTool
        
        tool = HeadlessBrowserTool()
        
        # Test with file:// URL
        file_url = f"file://{os.path.abspath(test_file)}"
        print(f"Testing with local file: {file_url}")
        
        result = await tool.execute(
            url=file_url,
            action="scrape_js"
        )
        
        print("\n=== Results ===")
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ JavaScript scraping successful!")
            print(f"  Title: {result.get('title', 'N/A')}")
            print(f"  URL: {result.get('url', 'N/A')}")
            print(f"  Word count: {result.get('word_count', 0)}")
            
            content = result.get('content', '')
            print(f"\n  Content preview (first 200 chars):")
            print(f"    {content[:200]}...")
            
            # Check if JavaScript was executed
            if "This content was added by JavaScript" in content:
                print("\n✅ JavaScript execution confirmed!")
                print("  - Dynamic content was successfully loaded")
            else:
                print("\n⚠️  JavaScript may not have executed properly")
            
            if result.get('title') == 'JavaScript Loaded Successfully':
                print("  - Document title was updated by JavaScript")
            
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)
            print(f"\nCleaned up test file: {test_file}")

if __name__ == "__main__":
    asyncio.run(test_js_scraping_local())