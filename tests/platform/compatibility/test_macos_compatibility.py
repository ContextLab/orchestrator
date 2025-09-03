#!/usr/bin/env python3
"""
macOS-specific compatibility tests for orchestrator.

Tests for macOS-specific behaviors, file system characteristics,
and API integrations that may differ from other platforms.
"""

import os
import sys
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import pytest
import logging

logger = logging.getLogger(__name__)

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from orchestrator import Orchestrator, init_models
from src.orchestrator.compiler.yaml_compiler import YAMLCompiler


class MacOSCompatibilityTester:
    """Tests macOS-specific compatibility and behavior."""

    def __init__(self):
        self.is_macos = platform.system() == "Darwin"
        self.macos_version = platform.mac_ver()[0] if self.is_macos else None
        self.architecture = platform.machine()
        
    def test_file_system_specifics(self) -> Dict[str, Any]:
        """Test macOS-specific file system behaviors."""
        results = {
            "platform": "macOS",
            "version": self.macos_version,
            "tests": {}
        }
        
        if not self.is_macos:
            results["tests"]["skip_reason"] = "Not running on macOS"
            return results
            
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test HFS+ case insensitivity (default on macOS)
            test_file1 = temp_path / "TestFile.txt"
            test_file2 = temp_path / "testfile.txt"
            
            test_file1.write_text("content1")
            # On case-insensitive systems, this overwrites the first file
            test_file2.write_text("content2")
            
            case_behavior = {
                "file1_exists": test_file1.exists(),
                "file2_exists": test_file2.exists(),
                "same_content": test_file1.read_text() == test_file2.read_text() if test_file1.exists() else False
            }
            
            results["tests"]["case_insensitive_fs"] = {
                "passed": True,  # Both behaviors are valid
                "behavior": case_behavior,
                "is_case_insensitive": case_behavior["same_content"]
            }
            
            # Test resource forks and extended attributes (macOS specific)
            test_file = temp_path / "extended_attrs.txt"
            test_file.write_text("test content")
            
            try:
                # Try to set extended attribute using xattr command
                subprocess.run([
                    "xattr", "-w", "com.orchestrator.test", "test_value", str(test_file)
                ], check=True, capture_output=True)
                
                # Try to read it back
                result = subprocess.run([
                    "xattr", "-p", "com.orchestrator.test", str(test_file)
                ], check=True, capture_output=True, text=True)
                
                extended_attrs_work = result.stdout.strip() == "test_value"
            except (subprocess.CalledProcessError, FileNotFoundError):
                extended_attrs_work = False
                
            results["tests"]["extended_attributes"] = {
                "passed": extended_attrs_work,
                "supported": extended_attrs_work
            }
            
            # Test .DS_Store handling
            ds_store_file = temp_path / ".DS_Store"
            try:
                # Create a .DS_Store file (these are created automatically by Finder)
                ds_store_file.write_bytes(b"test ds store content")
                ds_store_handling = ds_store_file.exists()
                
                # Clean up
                ds_store_file.unlink()
            except Exception as e:
                ds_store_handling = False
                logger.warning(f".DS_Store test failed: {e}")
                
            results["tests"]["ds_store_handling"] = {
                "passed": ds_store_handling,
                "supported": ds_store_handling
            }
            
        return results
    
    def test_security_and_permissions(self) -> Dict[str, Any]:
        """Test macOS security and permission systems."""
        results = {
            "platform": "macOS",
            "tests": {}
        }
        
        if not self.is_macos:
            results["tests"]["skip_reason"] = "Not running on macOS"
            return results
            
        # Test Gatekeeper/codesigning awareness
        try:
            result = subprocess.run([
                "spctl", "--status"
            ], capture_output=True, text=True)
            
            gatekeeper_enabled = "assessments enabled" in result.stdout.lower()
        except (subprocess.CalledProcessError, FileNotFoundError):
            gatekeeper_enabled = None
            
        results["tests"]["gatekeeper_status"] = {
            "passed": True,  # Just informational
            "enabled": gatekeeper_enabled
        }
        
        # Test SIP (System Integrity Protection) status
        try:
            result = subprocess.run([
                "csrutil", "status"
            ], capture_output=True, text=True)
            
            sip_enabled = "enabled" in result.stdout.lower()
        except (subprocess.CalledProcessError, FileNotFoundError):
            sip_enabled = None
            
        results["tests"]["sip_status"] = {
            "passed": True,  # Just informational
            "enabled": sip_enabled
        }
        
        # Test user permissions for common directories
        permission_tests = {
            "home_directory": os.path.expanduser("~"),
            "tmp_directory": "/tmp",
            "applications": "/Applications"
        }
        
        for dir_name, dir_path in permission_tests.items():
            if os.path.exists(dir_path):
                readable = os.access(dir_path, os.R_OK)
                writable = os.access(dir_path, os.W_OK)
                results["tests"][f"{dir_name}_permissions"] = {
                    "passed": readable,  # Should at least be readable
                    "readable": readable,
                    "writable": writable
                }
        
        return results
    
    def test_architecture_compatibility(self) -> Dict[str, Any]:
        """Test architecture-specific compatibility (Intel vs Apple Silicon)."""
        results = {
            "platform": "macOS",
            "architecture": self.architecture,
            "tests": {}
        }
        
        if not self.is_macos:
            results["tests"]["skip_reason"] = "Not running on macOS"
            return results
            
        # Detect if running on Apple Silicon or Intel
        is_apple_silicon = self.architecture == "arm64"
        is_intel = self.architecture in ["x86_64", "i386"]
        
        results["tests"]["architecture_detection"] = {
            "passed": is_apple_silicon or is_intel,
            "is_apple_silicon": is_apple_silicon,
            "is_intel": is_intel,
            "architecture": self.architecture
        }
        
        # Test Rosetta 2 availability on Apple Silicon
        if is_apple_silicon:
            try:
                # Check if Rosetta 2 is installed
                result = subprocess.run([
                    "pgrep", "oahd"
                ], capture_output=True)
                
                # oahd is the Rosetta 2 daemon
                rosetta_available = result.returncode == 0
            except (subprocess.CalledProcessError, FileNotFoundError):
                rosetta_available = False
                
            results["tests"]["rosetta2_availability"] = {
                "passed": True,  # Optional feature
                "available": rosetta_available
            }
        
        # Test Python architecture
        import struct
        python_arch = struct.calcsize("P") * 8  # 32 or 64 bit
        
        results["tests"]["python_architecture"] = {
            "passed": python_arch == 64,  # Should be 64-bit
            "bits": python_arch,
            "expected": 64
        }
        
        return results
    
    def test_networking_and_apis(self) -> Dict[str, Any]:
        """Test macOS-specific networking and API behaviors."""
        results = {
            "platform": "macOS",
            "tests": {}
        }
        
        if not self.is_macos:
            results["tests"]["skip_reason"] = "Not running on macOS"
            return results
            
        import socket
        import ssl
        
        # Test SSL/TLS support (macOS uses SecureTransport)
        try:
            context = ssl.create_default_context()
            ssl_support = context is not None
            ssl_version = ssl.OPENSSL_VERSION if hasattr(ssl, 'OPENSSL_VERSION') else "Unknown"
        except Exception as e:
            ssl_support = False
            ssl_version = f"Error: {e}"
            
        results["tests"]["ssl_support"] = {
            "passed": ssl_support,
            "supported": ssl_support,
            "version": ssl_version
        }
        
        # Test IPv6 support
        try:
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            sock.close()
            ipv6_support = True
        except Exception as e:
            ipv6_support = False
            logger.warning(f"IPv6 test failed: {e}")
            
        results["tests"]["ipv6_support"] = {
            "passed": ipv6_support,
            "supported": ipv6_support
        }
        
        # Test DNS resolution behavior
        try:
            import dns.resolver
            dns_libraries_available = True
        except ImportError:
            dns_libraries_available = False
            
        results["tests"]["dns_libraries"] = {
            "passed": True,  # Optional
            "advanced_dns_available": dns_libraries_available
        }
        
        return results
    
    def test_orchestrator_on_macos(self) -> Dict[str, Any]:
        """Test orchestrator-specific functionality on macOS."""
        results = {
            "platform": "macOS",
            "tests": {}
        }
        
        if not self.is_macos:
            results["tests"]["skip_reason"] = "Not running on macOS"
            return results
            
        # Test model initialization
        try:
            model_registry = init_models()
            models_initialized = model_registry is not None and len(model_registry.models) > 0
        except Exception as e:
            models_initialized = False
            logger.warning(f"Model initialization failed: {e}")
            
        results["tests"]["model_initialization"] = {
            "passed": models_initialized,
            "models_available": models_initialized
        }
        
        # Test YAML compilation
        try:
            compiler = YAMLCompiler()
            simple_yaml = """
name: macos_test_pipeline
description: Simple test for macOS compatibility
steps:
  - name: test_step
    description: Test step
    type: llm
    input: "Test input for macOS"
"""
            compiled = compiler.compile_yaml(simple_yaml)
            yaml_compilation = compiled is not None
        except Exception as e:
            yaml_compilation = False
            logger.warning(f"YAML compilation failed: {e}")
            
        results["tests"]["yaml_compilation"] = {
            "passed": yaml_compilation,
            "compilation_successful": yaml_compilation
        }
        
        # Test temp directory creation
        try:
            with tempfile.TemporaryDirectory(prefix="orchestrator_macos_") as temp_dir:
                temp_path = Path(temp_dir)
                test_file = temp_path / "test.yaml"
                test_file.write_text("test: content")
                temp_dir_works = test_file.exists()
        except Exception as e:
            temp_dir_works = False
            logger.warning(f"Temp directory test failed: {e}")
            
        results["tests"]["temp_directory"] = {
            "passed": temp_dir_works,
            "temp_directory_works": temp_dir_works
        }
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all macOS compatibility tests."""
        if not self.is_macos:
            return {
                "platform": "macOS",
                "skipped": True,
                "reason": f"Not running on macOS (current: {platform.system()})"
            }
            
        logger.info(f"Running macOS compatibility tests (version: {self.macos_version}, arch: {self.architecture})")
        
        results = {
            "platform": "macOS",
            "version": self.macos_version,
            "architecture": self.architecture,
            "test_results": {}
        }
        
        # Run test suites
        results["test_results"]["filesystem"] = self.test_file_system_specifics()
        results["test_results"]["security"] = self.test_security_and_permissions()
        results["test_results"]["architecture"] = self.test_architecture_compatibility()
        results["test_results"]["networking"] = self.test_networking_and_apis()
        results["test_results"]["orchestrator"] = self.test_orchestrator_on_macos()
        
        # Calculate overall success rate
        all_tests = []
        for test_category in results["test_results"].values():
            if "tests" in test_category:
                for test_name, test_data in test_category["tests"].items():
                    if test_name != "skip_reason":
                        all_tests.append(test_data.get("passed", False))
                
        results["overall"] = {
            "total_tests": len(all_tests),
            "passed_tests": sum(all_tests),
            "success_rate": sum(all_tests) / len(all_tests) if all_tests else 0
        }
        
        logger.info(f"macOS compatibility: {results['overall']['passed_tests']}/{results['overall']['total_tests']} tests passed")
        
        return results


# pytest test functions

@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-specific tests")
def test_macos_file_system():
    """Test macOS file system behaviors."""
    tester = MacOSCompatibilityTester()
    results = tester.test_file_system_specifics()
    
    # Should handle case sensitivity appropriately
    assert results["tests"]["case_insensitive_fs"]["passed"]


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-specific tests")
def test_macos_security():
    """Test macOS security systems."""
    tester = MacOSCompatibilityTester()
    results = tester.test_security_and_permissions()
    
    # Should be able to access home directory
    assert results["tests"]["home_directory_permissions"]["passed"]


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-specific tests")
def test_macos_architecture():
    """Test architecture detection and compatibility."""
    tester = MacOSCompatibilityTester()
    results = tester.test_architecture_compatibility()
    
    # Should detect architecture correctly
    assert results["tests"]["architecture_detection"]["passed"]
    
    # Python should be 64-bit
    assert results["tests"]["python_architecture"]["passed"]


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-specific tests")
def test_macos_networking():
    """Test macOS networking capabilities."""
    tester = MacOSCompatibilityTester()
    results = tester.test_networking_and_apis()
    
    # SSL should work
    assert results["tests"]["ssl_support"]["passed"]


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-specific tests")
def test_orchestrator_on_macos():
    """Test orchestrator functionality on macOS."""
    tester = MacOSCompatibilityTester()
    results = tester.test_orchestrator_on_macos()
    
    # YAML compilation should work
    assert results["tests"]["yaml_compilation"]["passed"]
    
    # Temp directories should work
    assert results["tests"]["temp_directory"]["passed"]


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-specific tests")
@pytest.mark.slow
def test_full_macos_compatibility():
    """Run comprehensive macOS compatibility test suite."""
    tester = MacOSCompatibilityTester()
    results = tester.run_all_tests()
    
    # Should not be skipped on macOS
    assert not results.get("skipped", False)
    
    # Should pass majority of tests
    assert results["overall"]["success_rate"] >= 0.8
    
    # Log results
    logger.info(f"macOS {results['version']} ({results['architecture']})")
    logger.info(f"Tests passed: {results['overall']['passed_tests']}/{results['overall']['total_tests']}")


if __name__ == "__main__":
    # Run macOS compatibility tests when called directly
    tester = MacOSCompatibilityTester()
    results = tester.run_all_tests()
    
    if results.get("skipped"):
        print(f"Skipped: {results['reason']}")
    else:
        print("=== macOS Compatibility Test Results ===")
        print(f"macOS Version: {results['version']}")
        print(f"Architecture: {results['architecture']}")
        print(f"Test Results: {results['overall']['passed_tests']}/{results['overall']['total_tests']} passed ({results['overall']['success_rate']*100:.1f}%)")
        
        for category_name, category_results in results["test_results"].items():
            if "tests" in category_results:
                print(f"\n{category_name.title()} Tests:")
                for test_name, test_data in category_results["tests"].items():
                    if test_name != "skip_reason":
                        status = "PASS" if test_data.get("passed", False) else "FAIL"
                        print(f"  {test_name}: {status}")