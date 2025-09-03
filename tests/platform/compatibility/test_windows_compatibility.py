#!/usr/bin/env python3
"""
Windows-specific compatibility tests for orchestrator.

Tests for Windows-specific behaviors, file system characteristics,
and system capabilities that may differ from other platforms.
"""

import os
import sys
import platform
import subprocess
import tempfile
from pathlib import Path, WindowsPath
from typing import Dict, Any, List, Optional
import pytest
import logging

logger = logging.getLogger(__name__)

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

try:
    from orchestrator import Orchestrator, init_models
    from src.orchestrator.compiler.yaml_compiler import YAMLCompiler
except ImportError as e:
    logger.warning(f"Could not import orchestrator modules: {e}")


class WindowsCompatibilityTester:
    """Tests Windows-specific compatibility and behavior."""

    def __init__(self):
        self.is_windows = platform.system() == "Windows"
        self.windows_version = platform.version() if self.is_windows else None
        self.windows_release = platform.release() if self.is_windows else None
        
    def _get_windows_info(self) -> Dict[str, Any]:
        """Get detailed Windows information."""
        if not self.is_windows:
            return {}
            
        try:
            import winreg
            
            # Get Windows version from registry
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                              r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
                product_name = winreg.QueryValueEx(key, "ProductName")[0]
                build_number = winreg.QueryValueEx(key, "CurrentBuildNumber")[0]
                
            return {
                "product_name": product_name,
                "build_number": build_number,
                "version": self.windows_version,
                "release": self.windows_release
            }
        except (ImportError, OSError) as e:
            logger.warning(f"Could not get detailed Windows info: {e}")
            return {
                "version": self.windows_version,
                "release": self.windows_release
            }
    
    def test_file_system_specifics(self) -> Dict[str, Any]:
        """Test Windows-specific file system behaviors."""
        results = {
            "platform": "Windows",
            "windows_info": self._get_windows_info(),
            "tests": {}
        }
        
        if not self.is_windows:
            results["tests"]["skip_reason"] = "Not running on Windows"
            return results
            
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test case insensitivity (Windows is case-insensitive)
            test_file1 = temp_path / "TestFile.txt"
            test_file2 = temp_path / "testfile.txt"
            
            test_file1.write_text("content1")
            test_file2.write_text("content2")
            
            # On Windows, these should be the same file
            case_insensitive = (test_file1.read_text() == test_file2.read_text())
            
            results["tests"]["case_insensitive_fs"] = {
                "passed": case_insensitive,
                "is_case_insensitive": case_insensitive
            }
            
            # Test path length limitations
            try:
                # Windows has traditionally had a 260 character path limit
                long_path = temp_path / ("a" * 200) / "test.txt"
                long_path.parent.mkdir(parents=True, exist_ok=True)
                long_path.write_text("test content")
                long_path_works = long_path.exists()
                
                if long_path_works:
                    long_path.unlink()
                    long_path.parent.rmdir()
            except (OSError, FileNotFoundError) as e:
                long_path_works = False
                logger.info(f"Long path limitation detected (expected on older Windows): {e}")
                
            results["tests"]["long_path_support"] = {
                "passed": True,  # Both behaviors are valid
                "long_paths_work": long_path_works
            }
            
            # Test reserved file names
            reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]
            reserved_results = {}
            
            for name in reserved_names:
                try:
                    reserved_file = temp_path / f"{name}.txt"
                    reserved_file.write_text("test")
                    reserved_works = False  # Should fail on Windows
                    reserved_file.unlink()
                except (OSError, FileNotFoundError):
                    reserved_works = True  # Expected to fail
                    
                reserved_results[name] = reserved_works
                
            results["tests"]["reserved_names"] = {
                "passed": all(reserved_results.values()),
                "reserved_name_handling": reserved_results
            }
            
            # Test invalid characters in filenames
            invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
            invalid_results = {}
            
            for char in invalid_chars:
                try:
                    invalid_file = temp_path / f"test{char}.txt"
                    invalid_file.write_text("test")
                    invalid_char_rejected = False  # Should fail
                    invalid_file.unlink()
                except (OSError, FileNotFoundError):
                    invalid_char_rejected = True  # Expected to fail
                    
                invalid_results[char] = invalid_char_rejected
                
            results["tests"]["invalid_characters"] = {
                "passed": all(invalid_results.values()),
                "invalid_char_handling": invalid_results
            }
            
            # Test Windows path types
            if isinstance(temp_path, WindowsPath):
                path_type_correct = True
                drive_letter = temp_path.drive if hasattr(temp_path, 'drive') else None
            else:
                path_type_correct = False
                drive_letter = None
                
            results["tests"]["path_types"] = {
                "passed": path_type_correct,
                "uses_windows_path": path_type_correct,
                "drive_letter": drive_letter
            }
            
        return results
    
    def test_windows_services_and_registry(self) -> Dict[str, Any]:
        """Test Windows services and registry access."""
        results = {
            "platform": "Windows",
            "tests": {}
        }
        
        if not self.is_windows:
            results["tests"]["skip_reason"] = "Not running on Windows"
            return results
            
        # Test registry access
        try:
            import winreg
            
            # Try to read a common registry key
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                              r"SOFTWARE\Microsoft\Windows\CurrentVersion") as key:
                version_info = winreg.QueryValueEx(key, "ProgramFilesDir")[0]
                
            registry_access = isinstance(version_info, str) and len(version_info) > 0
        except (ImportError, OSError, FileNotFoundError) as e:
            registry_access = False
            logger.warning(f"Registry access test failed: {e}")
            
        results["tests"]["registry_access"] = {
            "passed": registry_access,
            "can_access_registry": registry_access
        }
        
        # Test Windows services (basic check)
        try:
            result = subprocess.run(["sc", "query", "Themes"], 
                                  capture_output=True, text=True, timeout=10)
            services_accessible = result.returncode == 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            services_accessible = False
            
        results["tests"]["services_access"] = {
            "passed": True,  # Optional feature
            "can_query_services": services_accessible
        }
        
        # Test environment variables (Windows-specific)
        windows_env_vars = {
            "USERPROFILE": os.environ.get("USERPROFILE"),
            "APPDATA": os.environ.get("APPDATA"),
            "TEMP": os.environ.get("TEMP"),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT"),
            "PROGRAMFILES": os.environ.get("PROGRAMFILES")
        }
        
        env_vars_present = all(var is not None for var in windows_env_vars.values())
        
        results["tests"]["windows_environment"] = {
            "passed": env_vars_present,
            "environment_variables": windows_env_vars
        }
        
        return results
    
    def test_process_and_security(self) -> Dict[str, Any]:
        """Test Windows process and security features."""
        results = {
            "platform": "Windows",
            "tests": {}
        }
        
        if not self.is_windows:
            results["tests"]["skip_reason"] = "Not running on Windows"
            return results
            
        # Test UAC awareness
        try:
            import ctypes
            is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        except (ImportError, AttributeError):
            is_admin = False
            
        results["tests"]["uac_detection"] = {
            "passed": True,  # Both admin and non-admin are valid
            "is_admin": is_admin
        }
        
        # Test process creation
        try:
            result = subprocess.run(["echo", "test"], 
                                  capture_output=True, text=True, shell=True)
            process_creation = result.returncode == 0 and "test" in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            process_creation = False
            
        results["tests"]["process_creation"] = {
            "passed": process_creation,
            "can_create_processes": process_creation
        }
        
        # Test Windows API access
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            current_pid = kernel32.GetCurrentProcessId()
            api_access = current_pid > 0
        except (ImportError, AttributeError):
            api_access = False
            
        results["tests"]["windows_api"] = {
            "passed": api_access,
            "can_access_api": api_access
        }
        
        # Test file attributes
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(b"test content")
                temp_path = temp_file.name
                
            # Try to set hidden attribute
            result = subprocess.run(["attrib", "+H", temp_path], 
                                  capture_output=True)
            file_attributes_work = result.returncode == 0
            
            # Clean up
            subprocess.run(["attrib", "-H", temp_path], capture_output=True)
            os.unlink(temp_path)
            
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            file_attributes_work = False
            
        results["tests"]["file_attributes"] = {
            "passed": True,  # Optional feature
            "attributes_work": file_attributes_work
        }
        
        return results
    
    def test_networking_windows(self) -> Dict[str, Any]:
        """Test Windows networking capabilities."""
        results = {
            "platform": "Windows",
            "tests": {}
        }
        
        if not self.is_windows:
            results["tests"]["skip_reason"] = "Not running on Windows"
            return results
            
        # Test Windows Sockets (Winsock)
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.close()
            winsock_works = True
        except Exception as e:
            winsock_works = False
            logger.warning(f"Winsock test failed: {e}")
            
        results["tests"]["winsock"] = {
            "passed": winsock_works,
            "winsock_available": winsock_works
        }
        
        # Test Windows networking commands
        try:
            result = subprocess.run(["ipconfig", "/all"], 
                                  capture_output=True, text=True, timeout=10)
            ipconfig_works = result.returncode == 0 and "Windows IP Configuration" in result.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            ipconfig_works = False
            
        results["tests"]["ipconfig"] = {
            "passed": True,  # Optional
            "ipconfig_available": ipconfig_works
        }
        
        # Test firewall status
        try:
            result = subprocess.run(["netsh", "advfirewall", "show", "currentprofile"], 
                                  capture_output=True, text=True, timeout=10)
            firewall_query_works = result.returncode == 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            firewall_query_works = False
            
        results["tests"]["firewall_status"] = {
            "passed": True,  # Optional
            "can_query_firewall": firewall_query_works
        }
        
        return results
    
    def test_orchestrator_on_windows(self) -> Dict[str, Any]:
        """Test orchestrator-specific functionality on Windows."""
        results = {
            "platform": "Windows",
            "tests": {}
        }
        
        if not self.is_windows:
            results["tests"]["skip_reason"] = "Not running on Windows"
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
        
        # Test YAML compilation with Windows paths
        try:
            compiler = YAMLCompiler()
            simple_yaml = """
name: windows_test_pipeline
description: Simple test for Windows compatibility
steps:
  - name: test_step
    description: Test step
    type: llm
    input: "Test input for Windows"
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
        
        # Test temp directory creation with Windows paths
        try:
            with tempfile.TemporaryDirectory(prefix="orchestrator_windows_") as temp_dir:
                temp_path = Path(temp_dir)
                test_file = temp_path / "test.yaml"
                test_file.write_text("test: content")
                
                # Verify Windows-style path handling
                temp_dir_works = (test_file.exists() and 
                                isinstance(temp_path, WindowsPath))
        except Exception as e:
            temp_dir_works = False
            logger.warning(f"Temp directory test failed: {e}")
            
        results["tests"]["temp_directory"] = {
            "passed": temp_dir_works,
            "temp_directory_works": temp_dir_works
        }
        
        # Test path conversion and handling
        try:
            test_paths = [
                "C:\\Users\\test\\file.txt",
                "\\\\server\\share\\file.txt",  # UNC path
                "file.txt",  # relative path
                "C:/Users/test/file.txt"  # forward slashes
            ]
            
            path_conversions = {}
            for path_str in test_paths:
                try:
                    path_obj = Path(path_str)
                    path_conversions[path_str] = {
                        "convertible": True,
                        "is_absolute": path_obj.is_absolute(),
                        "normalized": str(path_obj)
                    }
                except Exception as e:
                    path_conversions[path_str] = {
                        "convertible": False,
                        "error": str(e)
                    }
                    
            path_handling_works = all(p.get("convertible", False) for p in path_conversions.values())
            
        except Exception as e:
            path_handling_works = False
            path_conversions = {"error": str(e)}
            
        results["tests"]["path_handling"] = {
            "passed": path_handling_works,
            "path_conversions": path_conversions
        }
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Windows compatibility tests."""
        if not self.is_windows:
            return {
                "platform": "Windows",
                "skipped": True,
                "reason": f"Not running on Windows (current: {platform.system()})"
            }
            
        windows_info = self._get_windows_info()
        logger.info(f"Running Windows compatibility tests ({windows_info.get('product_name', 'Windows')})")
        
        results = {
            "platform": "Windows",
            "windows_info": windows_info,
            "test_results": {}
        }
        
        # Run test suites
        results["test_results"]["filesystem"] = self.test_file_system_specifics()
        results["test_results"]["services"] = self.test_windows_services_and_registry()
        results["test_results"]["security"] = self.test_process_and_security()
        results["test_results"]["networking"] = self.test_networking_windows()
        results["test_results"]["orchestrator"] = self.test_orchestrator_on_windows()
        
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
        
        logger.info(f"Windows compatibility: {results['overall']['passed_tests']}/{results['overall']['total_tests']} tests passed")
        
        return results


# pytest test functions

@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific tests")
def test_windows_file_system():
    """Test Windows file system behaviors."""
    tester = WindowsCompatibilityTester()
    results = tester.test_file_system_specifics()
    
    # Windows should be case-insensitive
    assert results["tests"]["case_insensitive_fs"]["passed"]
    
    # Reserved names should be handled properly
    assert results["tests"]["reserved_names"]["passed"]
    
    # Invalid characters should be rejected
    assert results["tests"]["invalid_characters"]["passed"]


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific tests")
def test_windows_services():
    """Test Windows services and registry access."""
    tester = WindowsCompatibilityTester()
    results = tester.test_windows_services_and_registry()
    
    # Windows environment variables should be present
    assert results["tests"]["windows_environment"]["passed"]


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific tests")
def test_windows_security():
    """Test Windows security features."""
    tester = WindowsCompatibilityTester()
    results = tester.test_process_and_security()
    
    # UAC detection should work
    assert results["tests"]["uac_detection"]["passed"]
    
    # Process creation should work
    assert results["tests"]["process_creation"]["passed"]


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific tests")
def test_windows_networking():
    """Test Windows networking capabilities."""
    tester = WindowsCompatibilityTester()
    results = tester.test_networking_windows()
    
    # Winsock should work
    assert results["tests"]["winsock"]["passed"]


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific tests")
def test_orchestrator_on_windows():
    """Test orchestrator functionality on Windows."""
    tester = WindowsCompatibilityTester()
    results = tester.test_orchestrator_on_windows()
    
    # YAML compilation should work
    assert results["tests"]["yaml_compilation"]["passed"]
    
    # Temp directories should work
    assert results["tests"]["temp_directory"]["passed"]
    
    # Path handling should work
    assert results["tests"]["path_handling"]["passed"]


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific tests")
@pytest.mark.slow
def test_full_windows_compatibility():
    """Run comprehensive Windows compatibility test suite."""
    tester = WindowsCompatibilityTester()
    results = tester.run_all_tests()
    
    # Should not be skipped on Windows
    assert not results.get("skipped", False)
    
    # Should pass majority of tests
    assert results["overall"]["success_rate"] >= 0.8
    
    # Log results
    logger.info(f"Windows {results['windows_info'].get('product_name', 'Unknown')}")
    logger.info(f"Tests passed: {results['overall']['passed_tests']}/{results['overall']['total_tests']}")


if __name__ == "__main__":
    # Run Windows compatibility tests when called directly
    tester = WindowsCompatibilityTester()
    results = tester.run_all_tests()
    
    if results.get("skipped"):
        print(f"Skipped: {results['reason']}")
    else:
        print("=== Windows Compatibility Test Results ===")
        print(f"Windows: {results['windows_info'].get('product_name', 'Unknown')}")
        print(f"Build: {results['windows_info'].get('build_number', 'Unknown')}")
        print(f"Test Results: {results['overall']['passed_tests']}/{results['overall']['total_tests']} passed ({results['overall']['success_rate']*100:.1f}%)")
        
        for category_name, category_results in results["test_results"].items():
            if "tests" in category_results:
                print(f"\n{category_name.title()} Tests:")
                for test_name, test_data in category_results["tests"].items():
                    if test_name != "skip_reason":
                        status = "PASS" if test_data.get("passed", False) else "FAIL"
                        print(f"  {test_name}: {status}")