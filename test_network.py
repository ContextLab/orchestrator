#!/usr/bin/env python3
"""Test network connectivity issues."""

import asyncio
import requests
import httpx
import socket
from urllib.request import urlopen

def test_basic_connectivity():
    """Test basic network connectivity."""
    print("Testing basic network connectivity...")
    
    # Test DNS resolution
    try:
        ip = socket.gethostbyname("example.com")
        print(f"✓ DNS resolution working: example.com -> {ip}")
    except Exception as e:
        print(f"✗ DNS resolution failed: {e}")
    
    # Test with urllib
    try:
        with urlopen("https://example.com", timeout=5) as response:
            print(f"✓ urllib connection working: {response.status}")
    except Exception as e:
        print(f"✗ urllib connection failed: {e}")
    
    # Test with requests
    try:
        response = requests.get("https://example.com", timeout=5)
        print(f"✓ requests library working: {response.status_code}")
    except Exception as e:
        print(f"✗ requests library failed: {e}")
    
    # Test with httpx sync
    try:
        with httpx.Client() as client:
            response = client.get("https://example.com", timeout=5)
            print(f"✓ httpx sync working: {response.status_code}")
    except Exception as e:
        print(f"✗ httpx sync failed: {e}")

async def test_async_connectivity():
    """Test async network connectivity."""
    print("\nTesting async network connectivity...")
    
    # Test with httpx async
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://example.com", timeout=5)
            print(f"✓ httpx async working: {response.status_code}")
    except Exception as e:
        print(f"✗ httpx async failed: {e}")
    
    # Test requests in executor (like the tool does)
    try:
        loop = asyncio.get_event_loop()
        
        def _get_content():
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            return session.get("https://example.com", timeout=5)
        
        response = await loop.run_in_executor(None, _get_content)
        print(f"✓ requests in executor working: {response.status_code}")
    except Exception as e:
        print(f"✗ requests in executor failed: {e}")

def test_proxy_settings():
    """Check for proxy settings."""
    print("\nChecking proxy settings...")
    import os
    
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'NO_PROXY', 'no_proxy']
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"  {var}={value}")
    
    # Check requests proxy settings
    session = requests.Session()
    print(f"  Requests proxies: {session.proxies}")

if __name__ == "__main__":
    test_basic_connectivity()
    test_proxy_settings()
    asyncio.run(test_async_connectivity())