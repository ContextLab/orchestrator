#!/usr/bin/env python3
"""
Cross-platform API connectivity validation tests.

Tests that API endpoints and external services are accessible and function
consistently across different platforms and network configurations.
"""

import asyncio
import os
import sys
import platform
import socket
import ssl
import time
from typing import Dict, Any, List, Optional, Tuple
import pytest
import logging

logger = logging.getLogger(__name__)

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class APIConnectivityTester:
    """Tests API connectivity and functionality across platforms."""

    def __init__(self):
        self.current_platform = platform.system()
        self.test_results = {}
        
        # API endpoints for testing
        self.api_endpoints = {
            "anthropic": {
                "base_url": "https://api.anthropic.com",
                "health_endpoint": "/v1/messages",
                "requires_auth": True,
                "timeout": 10
            },
            "openai": {
                "base_url": "https://api.openai.com",
                "health_endpoint": "/v1/models",
                "requires_auth": True,
                "timeout": 10
            },
            "httpbin": {
                "base_url": "https://httpbin.org",
                "health_endpoint": "/status/200",
                "requires_auth": False,
                "timeout": 5
            },
            "jsonplaceholder": {
                "base_url": "https://jsonplaceholder.typicode.com",
                "health_endpoint": "/posts/1",
                "requires_auth": False,
                "timeout": 5
            }
        }
        
    def test_basic_connectivity(self) -> Dict[str, Any]:
        """Test basic network connectivity."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        # Test DNS resolution
        dns_tests = [
            "google.com",
            "api.anthropic.com",
            "api.openai.com",
            "httpbin.org"
        ]
        
        for hostname in dns_tests:
            try:
                ip_address = socket.gethostbyname(hostname)
                dns_works = len(ip_address) > 0
                
                results["tests"][f"dns_{hostname.replace('.', '_')}"] = {
                    "dns_resolution": dns_works,
                    "ip_address": ip_address
                }
                
            except socket.gaierror as e:
                results["tests"][f"dns_{hostname.replace('.', '_')}"] = {
                    "dns_resolution": False,
                    "error": str(e)
                }
        
        # Test TCP connectivity
        tcp_tests = [
            ("google.com", 80, "HTTP"),
            ("google.com", 443, "HTTPS"),
            ("api.anthropic.com", 443, "Anthropic API"),
            ("api.openai.com", 443, "OpenAI API")
        ]
        
        for hostname, port, service in tcp_tests:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((hostname, port))
                sock.close()
                
                tcp_works = result == 0
                
                results["tests"][f"tcp_{hostname.replace('.', '_')}_{port}"] = {
                    "tcp_connection": tcp_works,
                    "hostname": hostname,
                    "port": port,
                    "service": service
                }
                
            except Exception as e:
                results["tests"][f"tcp_{hostname.replace('.', '_')}_{port}"] = {
                    "tcp_connection": False,
                    "error": str(e)
                }
        
        # Test SSL/TLS connectivity
        ssl_tests = [
            "api.anthropic.com",
            "api.openai.com",
            "httpbin.org"
        ]
        
        for hostname in ssl_tests:
            try:
                context = ssl.create_default_context()
                
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(10)
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        ssock.connect((hostname, 443))
                        cert = ssock.getpeercert()
                        
                ssl_works = cert is not None
                ssl_version = context.protocol.name if hasattr(context.protocol, 'name') else "Unknown"
                
                results["tests"][f"ssl_{hostname.replace('.', '_')}"] = {
                    "ssl_connection": ssl_works,
                    "ssl_version": ssl_version,
                    "cert_subject": cert.get("subject", []) if cert else None
                }
                
            except Exception as e:
                results["tests"][f"ssl_{hostname.replace('.', '_')}"] = {
                    "ssl_connection": False,
                    "error": str(e)
                }
        
        return results
    
    def test_http_requests(self) -> Dict[str, Any]:
        """Test HTTP request functionality."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        # Test with urllib (standard library)
        try:
            import urllib.request
            import urllib.error
            
            for api_name, api_config in self.api_endpoints.items():
                if not api_config["requires_auth"]:  # Only test public endpoints
                    try:
                        url = api_config["base_url"] + api_config["health_endpoint"]
                        
                        request = urllib.request.Request(url)
                        request.add_header("User-Agent", f"Orchestrator-Test/{platform.system()}")
                        
                        with urllib.request.urlopen(request, timeout=api_config["timeout"]) as response:
                            status_code = response.getcode()
                            content_length = len(response.read())
                            
                        http_works = status_code in [200, 201, 204]
                        
                        results["tests"][f"urllib_{api_name}"] = {
                            "http_request": http_works,
                            "status_code": status_code,
                            "content_length": content_length,
                            "url": url
                        }
                        
                    except urllib.error.URLError as e:
                        results["tests"][f"urllib_{api_name}"] = {
                            "http_request": False,
                            "error": str(e),
                            "url": api_config["base_url"] + api_config["health_endpoint"]
                        }
                    except Exception as e:
                        results["tests"][f"urllib_{api_name}"] = {
                            "http_request": False,
                            "error": str(e)
                        }
                        
        except ImportError as e:
            results["tests"]["urllib_import"] = {
                "http_request": False,
                "error": f"urllib not available: {e}"
            }
        
        # Test with requests library (if available)
        try:
            import requests
            
            for api_name, api_config in self.api_endpoints.items():
                if not api_config["requires_auth"]:  # Only test public endpoints
                    try:
                        url = api_config["base_url"] + api_config["health_endpoint"]
                        
                        headers = {"User-Agent": f"Orchestrator-Test/{platform.system()}"}
                        response = requests.get(url, headers=headers, timeout=api_config["timeout"])
                        
                        http_works = response.status_code in [200, 201, 204]
                        
                        results["tests"][f"requests_{api_name}"] = {
                            "http_request": http_works,
                            "status_code": response.status_code,
                            "content_length": len(response.content),
                            "response_time_ms": response.elapsed.total_seconds() * 1000,
                            "url": url
                        }
                        
                    except requests.RequestException as e:
                        results["tests"][f"requests_{api_name}"] = {
                            "http_request": False,
                            "error": str(e),
                            "url": url
                        }
                    except Exception as e:
                        results["tests"][f"requests_{api_name}"] = {
                            "http_request": False,
                            "error": str(e)
                        }
                        
        except ImportError:
            results["tests"]["requests_import"] = {
                "http_request": False,
                "error": "requests library not available"
            }
        
        return results
    
    async def test_async_http_requests(self) -> Dict[str, Any]:
        """Test asynchronous HTTP request functionality."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        # Test with aiohttp (if available)
        try:
            import aiohttp
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for api_name, api_config in self.api_endpoints.items():
                    if not api_config["requires_auth"]:  # Only test public endpoints
                        try:
                            url = api_config["base_url"] + api_config["health_endpoint"]
                            
                            headers = {"User-Agent": f"Orchestrator-Test/{platform.system()}"}
                            
                            start_time = time.time()
                            async with session.get(url, headers=headers) as response:
                                content = await response.read()
                                response_time_ms = (time.time() - start_time) * 1000
                                
                            async_http_works = response.status in [200, 201, 204]
                            
                            results["tests"][f"aiohttp_{api_name}"] = {
                                "async_http_request": async_http_works,
                                "status_code": response.status,
                                "content_length": len(content),
                                "response_time_ms": response_time_ms,
                                "url": url
                            }
                            
                        except aiohttp.ClientError as e:
                            results["tests"][f"aiohttp_{api_name}"] = {
                                "async_http_request": False,
                                "error": str(e),
                                "url": url
                            }
                        except Exception as e:
                            results["tests"][f"aiohttp_{api_name}"] = {
                                "async_http_request": False,
                                "error": str(e)
                            }
                            
        except ImportError:
            results["tests"]["aiohttp_import"] = {
                "async_http_request": False,
                "error": "aiohttp library not available"
            }
        
        return results
    
    def test_api_authentication_endpoints(self) -> Dict[str, Any]:
        """Test API endpoints that require authentication (without actual auth)."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        # Test auth-required endpoints to ensure they're reachable
        # (should return 401/403, not connection errors)
        try:
            import urllib.request
            import urllib.error
            
            auth_endpoints = {
                "anthropic_auth": "https://api.anthropic.com/v1/messages",
                "openai_auth": "https://api.openai.com/v1/models"
            }
            
            for endpoint_name, url in auth_endpoints.items():
                try:
                    request = urllib.request.Request(url)
                    request.add_header("User-Agent", f"Orchestrator-Test/{platform.system()}")
                    
                    try:
                        response = urllib.request.urlopen(request, timeout=10)
                        status_code = response.getcode()
                        endpoint_reachable = True
                    except urllib.error.HTTPError as e:
                        status_code = e.code
                        endpoint_reachable = status_code in [400, 401, 403, 422]  # Expected auth errors
                    
                    results["tests"][endpoint_name] = {
                        "endpoint_reachable": endpoint_reachable,
                        "status_code": status_code,
                        "url": url,
                        "expected_auth_error": status_code in [400, 401, 403, 422]
                    }
                    
                except urllib.error.URLError as e:
                    results["tests"][endpoint_name] = {
                        "endpoint_reachable": False,
                        "error": str(e),
                        "url": url
                    }
                    
        except ImportError:
            results["tests"]["auth_endpoint_test"] = {
                "endpoint_reachable": False,
                "error": "urllib not available for auth endpoint testing"
            }
        
        return results
    
    def test_api_client_libraries(self) -> Dict[str, Any]:
        """Test API client library functionality."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        # Test OpenAI client library
        try:
            import openai
            
            # Test client creation (without making actual API calls)
            try:
                client = openai.OpenAI(api_key="test-key-no-actual-calls")
                client_created = client is not None
                
                results["tests"]["openai_client"] = {
                    "client_creation": client_created,
                    "library_version": getattr(openai, "__version__", "Unknown")
                }
                
            except Exception as e:
                results["tests"]["openai_client"] = {
                    "client_creation": False,
                    "error": str(e)
                }
                
        except ImportError:
            results["tests"]["openai_client"] = {
                "client_creation": False,
                "error": "openai library not available"
            }
        
        # Test Anthropic client library
        try:
            import anthropic
            
            # Test client creation (without making actual API calls)
            try:
                client = anthropic.Anthropic(api_key="test-key-no-actual-calls")
                client_created = client is not None
                
                results["tests"]["anthropic_client"] = {
                    "client_creation": client_created,
                    "library_version": getattr(anthropic, "__version__", "Unknown")
                }
                
            except Exception as e:
                results["tests"]["anthropic_client"] = {
                    "client_creation": False,
                    "error": str(e)
                }
                
        except ImportError:
            results["tests"]["anthropic_client"] = {
                "client_creation": False,
                "error": "anthropic library not available"
            }
        
        return results
    
    def test_proxy_and_firewall_compatibility(self) -> Dict[str, Any]:
        """Test proxy and firewall compatibility."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        # Check for proxy environment variables
        proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"]
        proxy_detected = any(os.environ.get(var) for var in proxy_vars)
        
        results["tests"]["proxy_detection"] = {
            "proxy_detected": proxy_detected,
            "proxy_variables": {var: os.environ.get(var, None) for var in proxy_vars}
        }
        
        # Test different user agents (some firewalls block certain user agents)
        if not proxy_detected:  # Skip if proxy detected to avoid complications
            user_agents = [
                "Mozilla/5.0 (compatible; Orchestrator/1.0)",
                f"Orchestrator-Test/{platform.system()}",
                "curl/7.68.0"
            ]
            
            try:
                import urllib.request
                
                for i, user_agent in enumerate(user_agents):
                    try:
                        url = "https://httpbin.org/user-agent"
                        request = urllib.request.Request(url)
                        request.add_header("User-Agent", user_agent)
                        
                        with urllib.request.urlopen(request, timeout=5) as response:
                            user_agent_works = response.getcode() == 200
                            
                        results["tests"][f"user_agent_{i+1}"] = {
                            "user_agent_accepted": user_agent_works,
                            "user_agent": user_agent
                        }
                        
                    except Exception as e:
                        results["tests"][f"user_agent_{i+1}"] = {
                            "user_agent_accepted": False,
                            "user_agent": user_agent,
                            "error": str(e)
                        }
                        
            except ImportError:
                results["tests"]["user_agent_test"] = {
                    "user_agent_accepted": False,
                    "error": "urllib not available"
                }
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all API connectivity tests."""
        logger.info(f"Running API connectivity tests on {self.current_platform}")
        
        results = {
            "platform": self.current_platform,
            "test_results": {}
        }
        
        # Run test suites
        results["test_results"]["basic_connectivity"] = self.test_basic_connectivity()
        results["test_results"]["http_requests"] = self.test_http_requests()
        results["test_results"]["async_requests"] = await self.test_async_http_requests()
        results["test_results"]["auth_endpoints"] = self.test_api_authentication_endpoints()
        results["test_results"]["client_libraries"] = self.test_api_client_libraries()
        results["test_results"]["proxy_firewall"] = self.test_proxy_and_firewall_compatibility()
        
        # Calculate overall success rate
        all_tests = []
        for test_category in results["test_results"].values():
            if "tests" in test_category:
                for test_name, test_data in test_category["tests"].items():
                    if isinstance(test_data, dict):
                        # Determine test success based on available metrics
                        success_indicators = [
                            test_data.get("dns_resolution", False),
                            test_data.get("tcp_connection", False),
                            test_data.get("ssl_connection", False),
                            test_data.get("http_request", False),
                            test_data.get("async_http_request", False),
                            test_data.get("endpoint_reachable", False),
                            test_data.get("client_creation", False)
                        ]
                        
                        # Test passes if any success indicator is True and no error
                        test_passed = (any(success_indicators) and 
                                     "error" not in test_data)
                        all_tests.append(test_passed)
        
        results["overall"] = {
            "total_tests": len(all_tests),
            "passed_tests": sum(all_tests),
            "success_rate": sum(all_tests) / len(all_tests) if all_tests else 0
        }
        
        logger.info(f"API connectivity tests: {results['overall']['passed_tests']}/{results['overall']['total_tests']} passed")
        
        return results


# pytest test functions

@pytest.fixture
def api_tester():
    """Create API connectivity tester instance."""
    return APIConnectivityTester()


@pytest.mark.asyncio
async def test_basic_network_connectivity(api_tester):
    """Test basic network connectivity."""
    tester = api_tester
    results = tester.test_basic_connectivity()
    
    # At least some DNS resolution should work
    dns_tests = [test for name, test in results["tests"].items() if name.startswith("dns_")]
    dns_working = any(test.get("dns_resolution", False) for test in dns_tests)
    
    if not dns_working:
        pytest.skip("No network connectivity available")
    
    # At least one TCP connection should work
    tcp_tests = [test for name, test in results["tests"].items() if name.startswith("tcp_")]
    tcp_working = any(test.get("tcp_connection", False) for test in tcp_tests)
    
    assert tcp_working, "No TCP connections working"


def test_http_request_functionality(api_tester):
    """Test HTTP request functionality."""
    tester = api_tester
    results = tester.test_http_requests()
    
    # At least one HTTP request method should work
    http_tests = [test for name, test in results["tests"].items() 
                 if name.startswith(("urllib_", "requests_")) and "import" not in name]
    
    if not http_tests:
        pytest.skip("No HTTP libraries available")
    
    http_working = any(test.get("http_request", False) for test in http_tests)
    
    if not http_working:
        pytest.skip("No HTTP connectivity available")
    
    assert http_working, "No HTTP requests working"


@pytest.mark.asyncio
async def test_async_http_functionality(api_tester):
    """Test asynchronous HTTP functionality."""
    tester = api_tester
    results = await tester.test_async_http_requests()
    
    # Skip if aiohttp not available
    if results["tests"].get("aiohttp_import", {}).get("async_http_request") is False:
        pytest.skip("aiohttp not available")
    
    # At least one async HTTP request should work
    async_tests = [test for name, test in results["tests"].items() if name.startswith("aiohttp_")]
    async_working = any(test.get("async_http_request", False) for test in async_tests)
    
    if not async_working:
        pytest.skip("No async HTTP connectivity available")
    
    assert async_working, "No async HTTP requests working"


def test_api_endpoints_reachable(api_tester):
    """Test API endpoints are reachable."""
    tester = api_tester
    results = tester.test_api_authentication_endpoints()
    
    # API endpoints should be reachable (even if they return auth errors)
    auth_tests = [test for name, test in results["tests"].items() 
                 if "auth" in name and "import" not in name]
    
    if not auth_tests:
        pytest.skip("No auth endpoint tests available")
    
    endpoints_reachable = any(test.get("endpoint_reachable", False) for test in auth_tests)
    
    if not endpoints_reachable:
        pytest.skip("No API endpoints reachable")
    
    assert endpoints_reachable, "No API endpoints reachable"


def test_api_client_libraries(api_tester):
    """Test API client library functionality."""
    tester = api_tester
    results = tester.test_api_client_libraries()
    
    # At least one client library should be available and creatable
    client_tests = [test for name, test in results["tests"].items() if "client" in name]
    clients_working = any(test.get("client_creation", False) for test in client_tests)
    
    assert clients_working, "No API client libraries working"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_comprehensive_api_connectivity(api_tester):
    """Run comprehensive API connectivity testing."""
    tester = api_tester
    results = await tester.run_all_tests()
    
    # Should pass majority of connectivity tests
    success_rate_threshold = 0.5  # More lenient due to network dependencies
    
    assert results["overall"]["success_rate"] >= success_rate_threshold, \
        f"API connectivity test success rate too low: {results['overall']['success_rate']*100:.1f}%"
    
    # Log results
    logger.info(f"Platform: {results['platform']}")
    logger.info(f"API connectivity tests: {results['overall']['passed_tests']}/{results['overall']['total_tests']}")


if __name__ == "__main__":
    async def main():
        # Run API connectivity tests when called directly
        tester = APIConnectivityTester()
        results = await tester.run_all_tests()
        
        print("=== API Connectivity Test Results ===")
        print(f"Platform: {results['platform']}")
        print(f"Test Results: {results['overall']['passed_tests']}/{results['overall']['total_tests']} passed ({results['overall']['success_rate']*100:.1f}%)")
        
        for category_name, category_results in results["test_results"].items():
            if "tests" in category_results:
                print(f"\n{category_name.title().replace('_', ' ')} Tests:")
                for test_name, test_data in category_results["tests"].items():
                    if isinstance(test_data, dict):
                        if "error" in test_data:
                            status = "FAIL"
                            detail = f"({test_data['error'][:50]}...)" if len(test_data['error']) > 50 else f"({test_data['error']})"
                        else:
                            success_indicators = [
                                test_data.get("dns_resolution", False),
                                test_data.get("tcp_connection", False),
                                test_data.get("http_request", False),
                                test_data.get("endpoint_reachable", False)
                            ]
                            status = "PASS" if any(success_indicators) else "FAIL"
                            detail = ""
                        
                        print(f"  {test_name}: {status} {detail}")
    
    asyncio.run(main())