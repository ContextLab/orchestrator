#!/usr/bin/env python3
"""Test fresh Playwright installation simulation."""

import subprocess
import sys
import os

def test_playwright_import():
    """Test if playwright can be imported and chromium is available."""
    try:
        # Test basic import
        import playwright
        print("✅ Playwright package is installed")
        
        # Test chromium availability
        from playwright.sync_api import sync_playwright
        p = sync_playwright().start()
        try:
            browser = p.chromium.launch(headless=True)
            browser.close()
            print("✅ Chromium browser is installed and working")
            return True
        except Exception as e:
            print(f"❌ Chromium not working: {e}")
            return False
        finally:
            p.stop()
            
    except ImportError:
        print("❌ Playwright is not installed")
        return False
    except Exception as e:
        print(f"❌ Error testing playwright: {e}")
        return False

def main():
    print("=== Playwright Installation Status ===\n")
    
    # Check current status
    if test_playwright_import():
        print("\nPlaywright and chromium are properly installed!")
        
        # Show installation details
        print("\nInstallation details:")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "playwright"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith(('Name:', 'Version:', 'Location:')):
                        print(f"  {line}")
        except:
            pass
            
        # Show browser path
        try:
            from playwright.sync_api import sync_playwright
            p = sync_playwright().start()
            print(f"\nChromium executable path:")
            print(f"  {p.chromium.executable_path}")
            p.stop()
        except:
            pass
    else:
        print("\nPlaywright or chromium is not properly installed.")
        print("The HeadlessBrowserTool will automatically install them when needed.")

if __name__ == "__main__":
    main()