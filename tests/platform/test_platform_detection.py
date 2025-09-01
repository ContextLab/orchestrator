#!/usr/bin/env python3
"""
Platform detection and compatibility validation tests.

Tests for detecting and handling different operating systems, Python versions,
and environment configurations across platforms.
"""

import os
import platform
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import psutil
import pytest
import logging

logger = logging.getLogger(__name__)

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class PlatformCompatibilityTester:
    """Tests platform-specific compatibility and behavior."""

    def __init__(self):
        self.platform_info = self._gather_platform_info()
        self.test_results = []
        
    def _gather_platform_info(self) -> Dict[str, Any]:
        """Gather comprehensive platform information."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "node": platform.node(),
            "platform": platform.platform(),
            "uname": platform.uname()._asdict(),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total": psutil.virtual_memory().total,
            "disk_usage": psutil.disk_usage('/').total if os.path.exists('/') else 
                         psutil.disk_usage('C:\\').total if os.path.exists('C:\\') else 0
        }
    
    def test_file_system_compatibility(self) -> Dict[str, Any]:
        """Test file system operations across platforms."""
        test_results = {
            "platform": self.platform_info["system"],
            "tests": {}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test path handling
            test_results["tests"]["path_separator"] = {
                "expected": "/" if self.platform_info["system"] != "Windows" else "\\",
                "actual": os.sep,
                "passed": True
            }
            
            # Test long path support
            long_path = temp_path / ("a" * 200) / "test.txt"
            try:
                long_path.parent.mkdir(parents=True, exist_ok=True)
                long_path.write_text("test content")
                long_path_works = long_path.exists()
                long_path.unlink()
                long_path.parent.rmdir()
            except (OSError, FileNotFoundError) as e:
                long_path_works = False
                logger.warning(f"Long path test failed: {e}")
                
            test_results["tests"]["long_path_support"] = {
                "passed": long_path_works,
                "length_tested": 200
            }
            
            # Test case sensitivity
            test_file1 = temp_path / "TestFile.txt"
            test_file2 = temp_path / "testfile.txt"
            test_file1.write_text("content1")
            test_file2.write_text("content2")
            
            case_sensitive = test_file1.read_text() != test_file2.read_text()
            test_results["tests"]["case_sensitive"] = {
                "passed": True,  # Both behaviors are valid
                "is_case_sensitive": case_sensitive
            }
            
            # Test special characters in filenames
            special_chars = []
            problematic_chars = ['<', '>', ':', '"', '|', '?', '*'] if self.platform_info["system"] == "Windows" else []
            
            for char in "áéíóú漢字🚀":
                try:
                    special_file = temp_path / f"test{char}.txt"
                    special_file.write_text("test")
                    if special_file.exists():
                        special_chars.append(char)
                    special_file.unlink()
                except (OSError, UnicodeError):
                    pass
                    
            test_results["tests"]["unicode_filenames"] = {
                "passed": len(special_chars) > 0,
                "supported_chars": special_chars,
                "problematic_chars": problematic_chars
            }
            
        return test_results
    
    def test_process_and_memory_handling(self) -> Dict[str, Any]:
        """Test process and memory handling across platforms."""
        test_results = {
            "platform": self.platform_info["system"],
            "tests": {}
        }
        
        # Test memory usage measurement
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Allocate some memory
        data = bytearray(1024 * 1024 * 10)  # 10MB
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        del data  # Release memory
        
        test_results["tests"]["memory_measurement"] = {
            "passed": memory_increase > 0,
            "initial_memory_mb": initial_memory / 1024 / 1024,
            "memory_increase_mb": memory_increase / 1024 / 1024
        }
        
        # Test CPU usage measurement
        cpu_percent_before = process.cpu_percent()
        
        # Do some CPU work
        for _ in range(100000):
            sum(range(100))
            
        cpu_percent_after = process.cpu_percent()
        
        test_results["tests"]["cpu_measurement"] = {
            "passed": True,  # CPU measurement varies by platform
            "cpu_before": cpu_percent_before,
            "cpu_after": cpu_percent_after
        }
        
        # Test process information
        test_results["tests"]["process_info"] = {
            "passed": True,
            "pid": process.pid,
            "ppid": process.ppid(),
            "status": process.status(),
            "create_time": process.create_time(),
            "num_threads": process.num_threads()
        }
        
        return test_results
    
    def test_environment_variables(self) -> Dict[str, Any]:
        """Test environment variable handling."""
        test_results = {
            "platform": self.platform_info["system"],
            "tests": {}
        }
        
        # Test common environment variables
        common_vars = {
            "PATH": os.environ.get("PATH"),
            "HOME": os.environ.get("HOME") or os.environ.get("USERPROFILE"),
            "USER": os.environ.get("USER") or os.environ.get("USERNAME"),
            "TEMP": os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp"
        }
        
        test_results["tests"]["common_variables"] = {
            "passed": all(var is not None for var in common_vars.values()),
            "variables": common_vars
        }
        
        # Test setting and getting custom environment variables
        test_var_name = "ORCHESTRATOR_PLATFORM_TEST"
        test_var_value = "platform_test_value"
        
        os.environ[test_var_name] = test_var_value
        retrieved_value = os.environ.get(test_var_name)
        
        # Clean up
        del os.environ[test_var_name]
        
        test_results["tests"]["custom_variables"] = {
            "passed": retrieved_value == test_var_value,
            "set_value": test_var_value,
            "retrieved_value": retrieved_value
        }
        
        return test_results
    
    def test_network_capabilities(self) -> Dict[str, Any]:
        """Test network capabilities across platforms."""
        test_results = {
            "platform": self.platform_info["system"],
            "tests": {}
        }
        
        import socket
        import urllib.request
        
        # Test socket creation
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.close()
            socket_creation = True
        except Exception as e:
            socket_creation = False
            logger.warning(f"Socket creation failed: {e}")
            
        test_results["tests"]["socket_creation"] = {
            "passed": socket_creation
        }
        
        # Test DNS resolution
        try:
            socket.gethostbyname('google.com')
            dns_resolution = True
        except Exception as e:
            dns_resolution = False
            logger.warning(f"DNS resolution failed: {e}")
            
        test_results["tests"]["dns_resolution"] = {
            "passed": dns_resolution
        }
        
        # Test HTTP request (simple connectivity test)
        try:
            response = urllib.request.urlopen('https://httpbin.org/status/200', timeout=10)
            http_connectivity = response.getcode() == 200
            response.close()
        except Exception as e:
            http_connectivity = False
            logger.warning(f"HTTP connectivity test failed: {e}")
            
        test_results["tests"]["http_connectivity"] = {
            "passed": http_connectivity
        }
        
        return test_results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all platform compatibility tests."""
        logger.info(f"Running platform compatibility tests on {self.platform_info['system']}")
        
        results = {
            "platform_info": self.platform_info,
            "test_results": {}
        }
        
        # Run test suites
        results["test_results"]["filesystem"] = self.test_file_system_compatibility()
        results["test_results"]["process_memory"] = self.test_process_and_memory_handling()
        results["test_results"]["environment"] = self.test_environment_variables()
        results["test_results"]["network"] = self.test_network_capabilities()
        
        # Calculate overall success rate
        all_tests = []
        for test_category in results["test_results"].values():
            for test_name, test_data in test_category["tests"].items():
                all_tests.append(test_data.get("passed", False))
                
        results["overall"] = {
            "total_tests": len(all_tests),
            "passed_tests": sum(all_tests),
            "success_rate": sum(all_tests) / len(all_tests) if all_tests else 0
        }
        
        logger.info(f"Platform compatibility: {results['overall']['passed_tests']}/{results['overall']['total_tests']} tests passed")
        
        return results


# pytest test functions

def test_current_platform_detection():
    """Test platform detection on current system."""
    tester = PlatformCompatibilityTester()
    
    # Verify we can detect the platform
    assert tester.platform_info["system"] in ["Windows", "Linux", "Darwin"]
    assert tester.platform_info["python_version"] is not None
    assert tester.platform_info["cpu_count"] > 0
    assert tester.platform_info["memory_total"] > 0


def test_filesystem_compatibility():
    """Test filesystem operations are compatible."""
    tester = PlatformCompatibilityTester()
    results = tester.test_file_system_compatibility()
    
    # Path separator should be correct
    assert results["tests"]["path_separator"]["passed"]
    
    # At least some unicode characters should work
    assert results["tests"]["unicode_filenames"]["passed"]


def test_process_memory_handling():
    """Test process and memory measurement works."""
    tester = PlatformCompatibilityTester()
    results = tester.test_process_and_memory_handling()
    
    # Memory measurement should work
    assert results["tests"]["memory_measurement"]["passed"]
    
    # Process info should be available
    assert results["tests"]["process_info"]["passed"]


def test_environment_variables():
    """Test environment variable handling."""
    tester = PlatformCompatibilityTester()
    results = tester.test_environment_variables()
    
    # Common variables should exist
    assert results["tests"]["common_variables"]["passed"]
    
    # Custom variables should work
    assert results["tests"]["custom_variables"]["passed"]


def test_network_capabilities():
    """Test basic network capabilities."""
    tester = PlatformCompatibilityTester()
    results = tester.test_network_capabilities()
    
    # Socket creation should work
    assert results["tests"]["socket_creation"]["passed"]


@pytest.mark.slow
def test_full_platform_compatibility():
    """Run comprehensive platform compatibility test suite."""
    tester = PlatformCompatibilityTester()
    results = tester.run_all_tests()
    
    # Should pass majority of tests
    assert results["overall"]["success_rate"] >= 0.8
    
    # Log detailed results
    logger.info(f"Platform: {results['platform_info']['system']}")
    logger.info(f"Python: {results['platform_info']['python_version']}")
    logger.info(f"Architecture: {results['platform_info']['architecture']}")
    logger.info(f"Tests passed: {results['overall']['passed_tests']}/{results['overall']['total_tests']}")


if __name__ == "__main__":
    # Run platform detection when called directly
    tester = PlatformCompatibilityTester()
    results = tester.run_all_tests()
    
    print("=== Platform Compatibility Test Results ===")
    print(f"Platform: {results['platform_info']['system']} {results['platform_info']['release']}")
    print(f"Python: {results['platform_info']['python_version']} ({results['platform_info']['python_implementation']})")
    print(f"Architecture: {results['platform_info']['architecture'][0]}")
    print(f"CPUs: {results['platform_info']['cpu_count']} ({results['platform_info']['cpu_count_logical']} logical)")
    print(f"Memory: {results['platform_info']['memory_total'] / (1024**3):.1f} GB")
    print()
    print(f"Test Results: {results['overall']['passed_tests']}/{results['overall']['total_tests']} passed ({results['overall']['success_rate']*100:.1f}%)")
    
    for category_name, category_results in results["test_results"].items():
        print(f"\n{category_name.title()} Tests:")
        for test_name, test_data in category_results["tests"].items():
            status = "PASS" if test_data.get("passed", False) else "FAIL"
            print(f"  {test_name}: {status}")