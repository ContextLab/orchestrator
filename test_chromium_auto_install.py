#!/usr/bin/env python3
"""Test automatic chromium installation in HeadlessBrowserTool.

This script tests that chromium is automatically installed when needed.
To test properly, first uninstall playwright if it exists:
  pip uninstall playwright -y
"""

import asyncio
import subprocess
import sys
import os

def check_playwright_installed():
    """Check if playwright is installed."""
    try:
        import playwright
        return True
    except ImportError:
        return False

def check_chromium_installed():
    """Check if chromium browser is installed for playwright."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "list"],
            capture_output=True,
            text=True
        )
        return "chromium" in result.stdout and "executable" in result.stdout
    except:
        return False

async def test_auto_install():
    """Test automatic installation of playwright and chromium."""
    print("=== Testing Automatic Chromium Installation ===\n")
    
    # Check initial state
    has_playwright = check_playwright_installed()
    has_chromium = check_chromium_installed() if has_playwright else False
    
    print(f"Initial state:")
    print(f"  Playwright installed: {'✓' if has_playwright else '✗'}")
    print(f"  Chromium installed: {'✓' if has_chromium else '✗'}")
    
    if has_playwright and has_chromium:
        print("\nWARNING: Playwright and chromium are already installed.")
        print("To properly test auto-installation, run:")
        print("  pip uninstall playwright -y")
        print("Then run this test again.\n")
    
    # Import and use HeadlessBrowserTool
    print("\nImporting HeadlessBrowserTool...")
    from src.orchestrator.tools.web_tools import HeadlessBrowserTool
    
    print("Creating HeadlessBrowserTool instance...")
    tool = HeadlessBrowserTool()
    
    print("\nTesting JavaScript scraping (will trigger auto-install if needed)...")
    result = await tool.execute(
        url="https://example.com",
        action="scrape_js"
    )
    
    # Check results
    print("\n=== Results ===")
    if "error" in result:
        print(f"❌ Error occurred: {result['error']}")
        if "Failed to install Playwright" in result.get("error", ""):
            print("\nPlaywright installation failed. This might be due to:")
            print("  - Network issues")
            print("  - Permission issues")
            print("  - Platform-specific requirements")
    else:
        print("✅ JavaScript scraping successful!")
        print(f"  Title: {result.get('title', 'N/A')}")
        print(f"  Content length: {len(result.get('content', ''))}")
        print(f"  Word count: {result.get('word_count', 0)}")
        
        # Verify installation
        print("\nVerifying installation:")
        has_playwright_after = check_playwright_installed()
        has_chromium_after = check_chromium_installed() if has_playwright_after else False
        
        print(f"  Playwright installed: {'✓' if has_playwright_after else '✗'}")
        print(f"  Chromium installed: {'✓' if has_chromium_after else '✗'}")
        
        if has_playwright_after and has_chromium_after:
            print("\n✅ Both Playwright and Chromium were successfully installed!")
        else:
            print("\n⚠️  Installation may be incomplete")

if __name__ == "__main__":
    asyncio.run(test_auto_install())