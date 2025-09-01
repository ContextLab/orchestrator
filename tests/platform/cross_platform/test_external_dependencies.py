#!/usr/bin/env python3
"""
Cross-platform external dependency testing.

Tests that external dependencies (Python packages, system libraries,
network services) load and function consistently across platforms.
"""

import os
import sys
import platform
import subprocess
import importlib
import socket
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pytest
import logging

logger = logging.getLogger(__name__)

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class ExternalDependencyTester:
    """Tests external dependency loading and functionality across platforms."""

    def __init__(self):
        self.current_platform = platform.system()
        self.python_version = platform.python_version()
        self.dependency_results = {}
        
    def test_core_python_dependencies(self) -> Dict[str, Any]:
        """Test core Python package dependencies."""
        results = {
            "platform": self.current_platform,
            "python_version": self.python_version,
            "tests": {}
        }
        
        # Core dependencies used by orchestrator
        core_dependencies = [
            "json", "os", "sys", "pathlib", "asyncio", "logging",
            "datetime", "tempfile", "subprocess", "socket", "time"
        ]
        
        for dep in core_dependencies:
            try:
                module = importlib.import_module(dep)
                import_successful = module is not None
                
                # Basic functionality test
                functionality_test = True
                if dep == "json":
                    test_data = {"test": "data"}
                    serialized = module.dumps(test_data)
                    deserialized = module.loads(serialized)
                    functionality_test = deserialized == test_data
                elif dep == "pathlib":
                    test_path = module.Path("test")
                    functionality_test = isinstance(test_path, module.Path)
                elif dep == "asyncio":
                    # Test basic asyncio functionality
                    async def test_coroutine():
                        return "success"
                    functionality_test = hasattr(module, "run")
                elif dep == "socket":
                    # Test socket creation
                    try:
                        test_socket = module.socket(module.AF_INET, module.SOCK_STREAM)
                        test_socket.close()
                        functionality_test = True
                    except Exception:
                        functionality_test = False
                
                results["tests"][dep] = {
                    "import_successful": import_successful,
                    "functionality_test": functionality_test,
                    "version": getattr(module, "__version__", "N/A")
                }
                
            except ImportError as e:
                results["tests"][dep] = {
                    "import_successful": False,
                    "functionality_test": False,
                    "error": str(e)
                }
        
        return results
    
    def test_third_party_dependencies(self) -> Dict[str, Any]:
        """Test third-party package dependencies."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        # Third-party dependencies used by orchestrator
        third_party_dependencies = [
            ("pydantic", "BaseModel"),
            ("pyyaml", "safe_load"),
            ("aiohttp", "ClientSession"),
            ("jinja2", "Template"),
            ("psutil", "Process"),
            ("requests", "get"),
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic"),
            ("pytest", "main")
        ]
        
        for dep_name, test_attr in third_party_dependencies:
            try:
                module = importlib.import_module(dep_name)
                import_successful = True
                
                # Test specific functionality
                if hasattr(module, test_attr):
                    attr_test = True
                    
                    # Additional functionality tests
                    if dep_name == "pydantic" and test_attr == "BaseModel":
                        try:
                            class TestModel(module.BaseModel):
                                name: str
                            test_model = TestModel(name="test")
                            functionality_test = test_model.name == "test"
                        except Exception:
                            functionality_test = False
                    elif dep_name == "pyyaml" and test_attr == "safe_load":
                        try:
                            test_yaml = "test: value"
                            loaded = getattr(module, test_attr)(test_yaml)
                            functionality_test = loaded == {"test": "value"}
                        except Exception:
                            functionality_test = False
                    elif dep_name == "psutil" and test_attr == "Process":
                        try:
                            process = getattr(module, test_attr)()
                            functionality_test = process.pid > 0
                        except Exception:
                            functionality_test = False
                    else:
                        functionality_test = True
                else:
                    attr_test = False
                    functionality_test = False
                
                results["tests"][dep_name] = {
                    "import_successful": import_successful,
                    "attribute_available": attr_test,
                    "functionality_test": functionality_test,
                    "version": getattr(module, "__version__", "N/A")
                }
                
            except ImportError as e:
                results["tests"][dep_name] = {
                    "import_successful": False,
                    "attribute_available": False,
                    "functionality_test": False,
                    "error": str(e)
                }
        
        return results
    
    def test_system_dependencies(self) -> Dict[str, Any]:
        """Test system-level dependencies and tools."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        # System dependencies that may be platform-specific
        system_commands = []
        
        if self.current_platform == "Windows":
            system_commands = [
                ("python", ["--version"]),
                ("pip", ["--version"]),
                ("cmd", ["/c", "echo", "test"]),
                ("powershell", ["-Command", "echo test"])
            ]
        else:  # Linux/macOS
            system_commands = [
                ("python3", ["--version"]),
                ("pip", ["--version"]),
                ("curl", ["--version"]),
                ("git", ["--version"]),
                ("which", ["python3"])
            ]
        
        for command, args in system_commands:
            try:
                result = subprocess.run([command] + args, 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=10)
                
                command_available = result.returncode == 0
                output_valid = len(result.stdout) > 0
                
                results["tests"][command] = {
                    "command_available": command_available,
                    "output_valid": output_valid,
                    "return_code": result.returncode,
                    "stdout": result.stdout.strip()[:100],  # First 100 chars
                    "stderr": result.stderr.strip()[:100] if result.stderr else ""
                }
                
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                results["tests"][command] = {
                    "command_available": False,
                    "output_valid": False,
                    "error": str(e)
                }
        
        return results
    
    def test_network_dependencies(self) -> Dict[str, Any]:
        """Test network connectivity and external services."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        # Test DNS resolution
        try:
            socket.gethostbyname("google.com")
            dns_resolution = True
        except socket.gaierror:
            dns_resolution = False
        
        results["tests"]["dns_resolution"] = {
            "dns_works": dns_resolution
        }
        
        # Test HTTP connectivity
        http_endpoints = [
            ("http_basic", "http://httpbin.org/status/200"),
            ("https_basic", "https://httpbin.org/status/200"),
            ("api_anthropic", "https://api.anthropic.com/"),
            ("api_openai", "https://api.openai.com/")
        ]
        
        for test_name, url in http_endpoints:
            try:
                import urllib.request
                response = urllib.request.urlopen(url, timeout=5)
                http_works = response.getcode() in [200, 404, 403]  # Various acceptable codes
                response.close()
                
                results["tests"][test_name] = {
                    "http_connectivity": http_works,
                    "status_code": response.getcode() if http_works else None
                }
                
            except Exception as e:
                results["tests"][test_name] = {
                    "http_connectivity": False,
                    "error": str(e)
                }
        
        # Test port connectivity
        common_ports = [
            (80, "HTTP"),
            (443, "HTTPS"),
            (53, "DNS")
        ]
        
        for port, service in common_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(("google.com", port))
                sock.close()
                
                port_accessible = result == 0
                
                results["tests"][f"port_{port}_{service}"] = {
                    "port_accessible": port_accessible,
                    "port": port,
                    "service": service
                }
                
            except Exception as e:
                results["tests"][f"port_{port}_{service}"] = {
                    "port_accessible": False,
                    "error": str(e)
                }
        
        return results
    
    def test_file_system_dependencies(self) -> Dict[str, Any]:
        """Test file system and temporary directory dependencies."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        # Test temporary directory access
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Test write permissions
                test_file = temp_path / "test.txt"
                test_file.write_text("test content")
                write_works = test_file.exists()
                
                # Test read permissions
                read_content = test_file.read_text()
                read_works = read_content == "test content"
                
                # Test subdirectory creation
                subdir = temp_path / "subdir"
                subdir.mkdir()
                subdir_works = subdir.is_dir()
                
                results["tests"]["temp_directory"] = {
                    "temp_dir_available": True,
                    "write_permissions": write_works,
                    "read_permissions": read_works,
                    "subdir_creation": subdir_works,
                    "temp_path": str(temp_path)
                }
                
        except Exception as e:
            results["tests"]["temp_directory"] = {
                "temp_dir_available": False,
                "error": str(e)
            }
        
        # Test specific directory access
        important_dirs = []
        
        if self.current_platform == "Windows":
            important_dirs = [
                (os.path.expanduser("~"), "user_home"),
                (os.environ.get("TEMP", ""), "temp_env"),
                (os.environ.get("APPDATA", ""), "app_data")
            ]
        else:  # Linux/macOS
            important_dirs = [
                (os.path.expanduser("~"), "user_home"),
                ("/tmp", "system_temp"),
                (os.environ.get("HOME", ""), "home_env")
            ]
        
        for dir_path, dir_name in important_dirs:
            if dir_path and os.path.exists(dir_path):
                try:
                    readable = os.access(dir_path, os.R_OK)
                    writable = os.access(dir_path, os.W_OK)
                    
                    results["tests"][f"dir_{dir_name}"] = {
                        "directory_exists": True,
                        "readable": readable,
                        "writable": writable,
                        "path": dir_path
                    }
                    
                except Exception as e:
                    results["tests"][f"dir_{dir_name}"] = {
                        "directory_exists": False,
                        "error": str(e)
                    }
            else:
                results["tests"][f"dir_{dir_name}"] = {
                    "directory_exists": False,
                    "reason": "path_not_found"
                }
        
        return results
    
    def test_orchestrator_specific_dependencies(self) -> Dict[str, Any]:
        """Test dependencies specific to orchestrator functionality."""
        results = {
            "platform": self.current_platform,
            "tests": {}
        }
        
        # Test orchestrator imports
        orchestrator_modules = [
            "orchestrator",
            "orchestrator.compiler.yaml_compiler",
            "orchestrator.models"
        ]
        
        for module_name in orchestrator_modules:
            try:
                module = importlib.import_module(module_name)
                import_successful = module is not None
                
                # Test specific functionality if available
                if "yaml_compiler" in module_name:
                    try:
                        compiler = module.YAMLCompiler()
                        functionality_test = compiler is not None
                    except Exception:
                        functionality_test = False
                elif module_name == "orchestrator.models":
                    try:
                        # Test model registry initialization
                        if hasattr(module, "get_model_registry"):
                            registry = module.get_model_registry()
                            functionality_test = registry is not None
                        else:
                            functionality_test = True
                    except Exception:
                        functionality_test = False
                else:
                    functionality_test = True
                
                results["tests"][module_name] = {
                    "import_successful": import_successful,
                    "functionality_test": functionality_test
                }
                
            except ImportError as e:
                results["tests"][module_name] = {
                    "import_successful": False,
                    "functionality_test": False,
                    "error": str(e)
                }
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all external dependency tests."""
        logger.info(f"Running external dependency tests on {self.current_platform}")
        
        results = {
            "platform": self.current_platform,
            "python_version": self.python_version,
            "test_results": {}
        }
        
        # Run test suites
        results["test_results"]["core_python"] = self.test_core_python_dependencies()
        results["test_results"]["third_party"] = self.test_third_party_dependencies()
        results["test_results"]["system"] = self.test_system_dependencies()
        results["test_results"]["network"] = self.test_network_dependencies()
        results["test_results"]["filesystem"] = self.test_file_system_dependencies()
        results["test_results"]["orchestrator"] = self.test_orchestrator_specific_dependencies()
        
        # Calculate overall success rate
        all_tests = []
        for test_category in results["test_results"].values():
            if "tests" in test_category:
                for test_name, test_data in test_category["tests"].items():
                    if isinstance(test_data, dict):
                        # Determine test success based on available metrics
                        success_indicators = [
                            test_data.get("import_successful", False),
                            test_data.get("functionality_test", False),
                            test_data.get("command_available", False),
                            test_data.get("http_connectivity", False),
                            test_data.get("dns_works", False),
                            test_data.get("temp_dir_available", False),
                            test_data.get("directory_exists", False)
                        ]
                        
                        # Test passes if any success indicator is True and no error
                        test_passed = (any(success_indicators) and 
                                     "error" not in test_data and 
                                     test_data.get("import_successful", True) is not False)
                        all_tests.append(test_passed)
        
        results["overall"] = {
            "total_tests": len(all_tests),
            "passed_tests": sum(all_tests),
            "success_rate": sum(all_tests) / len(all_tests) if all_tests else 0
        }
        
        logger.info(f"External dependency tests: {results['overall']['passed_tests']}/{results['overall']['total_tests']} passed")
        
        return results


# pytest test functions

@pytest.fixture
def dependency_tester():
    """Create external dependency tester instance."""
    return ExternalDependencyTester()


def test_core_python_dependencies(dependency_tester):
    """Test core Python dependencies are available."""
    tester = dependency_tester
    results = tester.test_core_python_dependencies()
    
    # Core modules should be importable
    core_modules = ["json", "os", "sys", "pathlib"]
    for module in core_modules:
        assert results["tests"][module]["import_successful"], f"Failed to import {module}"


def test_critical_third_party_dependencies(dependency_tester):
    """Test critical third-party dependencies."""
    tester = dependency_tester
    results = tester.test_third_party_dependencies()
    
    # At least some critical dependencies should be available
    critical_deps = ["pydantic", "pyyaml", "pytest"]
    available_critical = sum(1 for dep in critical_deps 
                           if results["tests"].get(dep, {}).get("import_successful", False))
    
    assert available_critical >= 2, f"Too few critical dependencies available: {available_critical}"


def test_system_commands(dependency_tester):
    """Test system command availability."""
    tester = dependency_tester
    results = tester.test_system_dependencies()
    
    # Python should be available
    python_available = (results["tests"].get("python", {}).get("command_available", False) or
                       results["tests"].get("python3", {}).get("command_available", False))
    assert python_available, "Python command not available"


def test_network_connectivity(dependency_tester):
    """Test basic network connectivity."""
    tester = dependency_tester
    results = tester.test_network_dependencies()
    
    # DNS should work (if network is available)
    dns_works = results["tests"]["dns_resolution"]["dns_works"]
    if not dns_works:
        pytest.skip("Network connectivity not available")
    
    # At least basic HTTP should work
    http_tests = [test for name, test in results["tests"].items() 
                 if name.startswith("http")]
    http_working = any(test.get("http_connectivity", False) for test in http_tests)
    
    assert http_working, "No HTTP connectivity working"


def test_file_system_access(dependency_tester):
    """Test file system access and permissions."""
    tester = dependency_tester
    results = tester.test_file_system_dependencies()
    
    # Temp directory should be available
    temp_test = results["tests"]["temp_directory"]
    assert temp_test["temp_dir_available"], "Temporary directory not available"
    assert temp_test["write_permissions"], "No write permissions in temp directory"
    assert temp_test["read_permissions"], "No read permissions in temp directory"


@pytest.mark.slow
def test_comprehensive_dependencies(dependency_tester):
    """Run comprehensive dependency testing."""
    tester = dependency_tester
    results = tester.run_all_tests()
    
    # Should pass majority of dependency tests
    assert results["overall"]["success_rate"] >= 0.6, \
        f"Dependency test success rate too low: {results['overall']['success_rate']*100:.1f}%"
    
    # Log results
    logger.info(f"Platform: {results['platform']}")
    logger.info(f"Python: {results['python_version']}")
    logger.info(f"Dependencies tested: {results['overall']['passed_tests']}/{results['overall']['total_tests']}")


if __name__ == "__main__":
    # Run external dependency tests when called directly
    tester = ExternalDependencyTester()
    results = tester.run_all_tests()
    
    print("=== External Dependency Test Results ===")
    print(f"Platform: {results['platform']}")
    print(f"Python Version: {results['python_version']}")
    print(f"Test Results: {results['overall']['passed_tests']}/{results['overall']['total_tests']} passed ({results['overall']['success_rate']*100:.1f}%)")
    
    for category_name, category_results in results["test_results"].items():
        if "tests" in category_results:
            print(f"\n{category_name.title().replace('_', ' ')} Dependencies:")
            for test_name, test_data in category_results["tests"].items():
                if isinstance(test_data, dict):
                    if "error" in test_data:
                        status = "FAIL"
                        detail = f"({test_data['error'][:50]}...)" if len(test_data['error']) > 50 else f"({test_data['error']})"
                    else:
                        success_indicators = [
                            test_data.get("import_successful", False),
                            test_data.get("functionality_test", False),
                            test_data.get("command_available", False),
                            test_data.get("http_connectivity", False)
                        ]
                        status = "PASS" if any(success_indicators) else "FAIL"
                        detail = ""
                    
                    print(f"  {test_name}: {status} {detail}")