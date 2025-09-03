#!/usr/bin/env python3
"""
Linux-specific compatibility tests for orchestrator.

Tests for Linux-specific behaviors, file system characteristics,
and system capabilities that may differ from other platforms.
"""

import os
import sys
import platform
import subprocess
import tempfile
import pwd
import grp
from pathlib import Path
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


class LinuxCompatibilityTester:
    """Tests Linux-specific compatibility and behavior."""

    def __init__(self):
        self.is_linux = platform.system() == "Linux"
        self.distribution = self._get_distribution_info() if self.is_linux else None
        self.kernel_version = platform.release() if self.is_linux else None
        
    def _get_distribution_info(self) -> Dict[str, str]:
        """Get Linux distribution information."""
        try:
            # Try modern approach first
            with open('/etc/os-release') as f:
                lines = f.readlines()
            
            info = {}
            for line in lines:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    info[key] = value.strip('"')
            
            return {
                "name": info.get("NAME", "Unknown"),
                "version": info.get("VERSION", "Unknown"),
                "id": info.get("ID", "unknown")
            }
        except (FileNotFoundError, PermissionError):
            # Fallback methods
            try:
                result = subprocess.run(['lsb_release', '-a'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    info = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            info[key.strip()] = value.strip()
                    
                    return {
                        "name": info.get("Description", "Unknown"),
                        "version": info.get("Release", "Unknown"),
                        "id": info.get("Distributor ID", "unknown").lower()
                    }
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
                
            return {"name": "Unknown Linux", "version": "Unknown", "id": "unknown"}
    
    def test_file_system_specifics(self) -> Dict[str, Any]:
        """Test Linux-specific file system behaviors."""
        results = {
            "platform": "Linux",
            "distribution": self.distribution,
            "tests": {}
        }
        
        if not self.is_linux:
            results["tests"]["skip_reason"] = "Not running on Linux"
            return results
            
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test case sensitivity (Linux is case-sensitive)
            test_file1 = temp_path / "TestFile.txt"
            test_file2 = temp_path / "testfile.txt"
            
            test_file1.write_text("content1")
            test_file2.write_text("content2")
            
            case_sensitive = (test_file1.read_text() != test_file2.read_text() and
                            test_file1.exists() and test_file2.exists())
            
            results["tests"]["case_sensitive_fs"] = {
                "passed": case_sensitive,
                "is_case_sensitive": case_sensitive
            }
            
            # Test long path support
            long_path = temp_path / ("a" * 200) / "test.txt"
            try:
                long_path.parent.mkdir(parents=True, exist_ok=True)
                long_path.write_text("test content")
                long_path_works = long_path.exists()
                long_path.unlink()
                long_path.parent.rmdir()
            except OSError as e:
                long_path_works = False
                logger.warning(f"Long path test failed: {e}")
                
            results["tests"]["long_path_support"] = {
                "passed": long_path_works,
                "length_tested": 200
            }
            
            # Test symbolic links
            try:
                original_file = temp_path / "original.txt"
                link_file = temp_path / "link.txt"
                
                original_file.write_text("original content")
                link_file.symlink_to(original_file)
                
                symlink_works = (link_file.is_symlink() and 
                               link_file.read_text() == "original content")
                               
                link_file.unlink()
                original_file.unlink()
            except (OSError, NotImplementedError) as e:
                symlink_works = False
                logger.warning(f"Symlink test failed: {e}")
                
            results["tests"]["symbolic_links"] = {
                "passed": symlink_works,
                "supported": symlink_works
            }
            
            # Test file permissions and ownership
            try:
                test_file = temp_path / "permissions.txt"
                test_file.write_text("test content")
                
                # Test chmod
                os.chmod(test_file, 0o644)
                mode = test_file.stat().st_mode & 0o777
                chmod_works = mode == 0o644
                
                # Test ownership info
                stat_info = test_file.stat()
                current_uid = os.getuid()
                current_gid = os.getgid()
                
                ownership_info = {
                    "uid": stat_info.st_uid,
                    "gid": stat_info.st_gid,
                    "current_uid": current_uid,
                    "current_gid": current_gid,
                    "matches": stat_info.st_uid == current_uid
                }
                
            except (OSError, AttributeError) as e:
                chmod_works = False
                ownership_info = {"error": str(e)}
                logger.warning(f"Permissions test failed: {e}")
                
            results["tests"]["file_permissions"] = {
                "passed": chmod_works,
                "chmod_works": chmod_works,
                "ownership_info": ownership_info
            }
            
        return results
    
    def test_system_capabilities(self) -> Dict[str, Any]:
        """Test Linux system capabilities and features."""
        results = {
            "platform": "Linux",
            "tests": {}
        }
        
        if not self.is_linux:
            results["tests"]["skip_reason"] = "Not running on Linux"
            return results
            
        # Test process capabilities
        try:
            current_user = pwd.getpwuid(os.getuid()).pw_name
            current_groups = [grp.getgrgid(gid).gr_name for gid in os.getgroups()]
            
            user_info = {
                "username": current_user,
                "uid": os.getuid(),
                "gid": os.getgid(),
                "groups": current_groups,
                "is_root": os.getuid() == 0
            }
        except (KeyError, OSError) as e:
            user_info = {"error": str(e)}
            logger.warning(f"User info test failed: {e}")
            
        results["tests"]["user_info"] = {
            "passed": isinstance(user_info.get("username"), str),
            "user_info": user_info
        }
        
        # Test /proc filesystem
        proc_tests = {
            "version": "/proc/version",
            "cpuinfo": "/proc/cpuinfo",
            "meminfo": "/proc/meminfo",
            "mounts": "/proc/mounts"
        }
        
        proc_results = {}
        for name, path in proc_tests.items():
            try:
                if os.path.exists(path) and os.access(path, os.R_OK):
                    with open(path, 'r') as f:
                        content = f.read(100)  # Read first 100 chars
                    proc_results[name] = {"accessible": True, "has_content": len(content) > 0}
                else:
                    proc_results[name] = {"accessible": False, "has_content": False}
            except (OSError, PermissionError) as e:
                proc_results[name] = {"accessible": False, "error": str(e)}
                
        results["tests"]["proc_filesystem"] = {
            "passed": all(r.get("accessible", False) for r in proc_results.values()),
            "proc_files": proc_results
        }
        
        # Test systemd presence (common init system)
        try:
            systemd_available = (os.path.exists("/run/systemd/system") or 
                               os.path.exists("/sys/fs/cgroup/systemd"))
        except OSError:
            systemd_available = False
            
        results["tests"]["systemd_presence"] = {
            "passed": True,  # Optional feature
            "systemd_available": systemd_available
        }
        
        # Test cgroups (container awareness)
        try:
            with open("/proc/1/cgroup", "r") as f:
                cgroup_content = f.read()
            
            in_container = ("docker" in cgroup_content.lower() or 
                          "lxc" in cgroup_content.lower() or
                          "kubepods" in cgroup_content.lower())
        except (FileNotFoundError, PermissionError):
            in_container = False
            cgroup_content = ""
            
        results["tests"]["container_detection"] = {
            "passed": True,  # Just informational
            "in_container": in_container,
            "cgroup_info": cgroup_content[:100] if cgroup_content else ""
        }
        
        return results
    
    def test_networking_and_security(self) -> Dict[str, Any]:
        """Test Linux networking and security features."""
        results = {
            "platform": "Linux",
            "tests": {}
        }
        
        if not self.is_linux:
            results["tests"]["skip_reason"] = "Not running on Linux"
            return results
            
        # Test iptables/netfilter presence
        try:
            result = subprocess.run(["which", "iptables"], 
                                  capture_output=True, text=True)
            iptables_available = result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            iptables_available = False
            
        results["tests"]["iptables_availability"] = {
            "passed": True,  # Optional
            "iptables_available": iptables_available
        }
        
        # Test SELinux/AppArmor presence
        selinux_enabled = os.path.exists("/sys/fs/selinux")
        apparmor_enabled = os.path.exists("/sys/kernel/security/apparmor")
        
        results["tests"]["security_modules"] = {
            "passed": True,  # Optional
            "selinux_present": selinux_enabled,
            "apparmor_present": apparmor_enabled
        }
        
        # Test capabilities
        try:
            import socket
            
            # Test raw socket creation (requires CAP_NET_RAW)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
                sock.close()
                raw_socket_cap = True
            except (OSError, PermissionError):
                raw_socket_cap = False
                
            # Test binding to privileged ports (requires CAP_NET_BIND_SERVICE or root)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('127.0.0.1', 80))
                sock.close()
                privileged_bind = True
            except (OSError, PermissionError):
                privileged_bind = False
                
        except ImportError:
            raw_socket_cap = False
            privileged_bind = False
            
        results["tests"]["network_capabilities"] = {
            "passed": True,  # Capabilities vary by user
            "raw_socket_capability": raw_socket_cap,
            "privileged_bind_capability": privileged_bind
        }
        
        return results
    
    def test_orchestrator_on_linux(self) -> Dict[str, Any]:
        """Test orchestrator-specific functionality on Linux."""
        results = {
            "platform": "Linux",
            "tests": {}
        }
        
        if not self.is_linux:
            results["tests"]["skip_reason"] = "Not running on Linux"
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
name: linux_test_pipeline
description: Simple test for Linux compatibility
steps:
  - name: test_step
    description: Test step
    type: llm
    input: "Test input for Linux"
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
        
        # Test temp directory creation with proper permissions
        try:
            with tempfile.TemporaryDirectory(prefix="orchestrator_linux_") as temp_dir:
                temp_path = Path(temp_dir)
                test_file = temp_path / "test.yaml"
                test_file.write_text("test: content")
                
                # Verify permissions
                dir_mode = temp_path.stat().st_mode & 0o777
                file_mode = test_file.stat().st_mode & 0o777
                
                temp_dir_works = (test_file.exists() and 
                                dir_mode >= 0o700 and 
                                file_mode >= 0o600)
        except Exception as e:
            temp_dir_works = False
            logger.warning(f"Temp directory test failed: {e}")
            
        results["tests"]["temp_directory"] = {
            "passed": temp_dir_works,
            "temp_directory_works": temp_dir_works
        }
        
        # Test signal handling
        import signal
        try:
            old_handler = signal.signal(signal.SIGUSR1, signal.SIG_DFL)
            signal_handling = True
            signal.signal(signal.SIGUSR1, old_handler)
        except (OSError, ValueError) as e:
            signal_handling = False
            logger.warning(f"Signal handling test failed: {e}")
            
        results["tests"]["signal_handling"] = {
            "passed": signal_handling,
            "signal_handling_works": signal_handling
        }
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Linux compatibility tests."""
        if not self.is_linux:
            return {
                "platform": "Linux",
                "skipped": True,
                "reason": f"Not running on Linux (current: {platform.system()})"
            }
            
        logger.info(f"Running Linux compatibility tests ({self.distribution['name']} {self.distribution['version']})")
        
        results = {
            "platform": "Linux",
            "distribution": self.distribution,
            "kernel_version": self.kernel_version,
            "test_results": {}
        }
        
        # Run test suites
        results["test_results"]["filesystem"] = self.test_file_system_specifics()
        results["test_results"]["system"] = self.test_system_capabilities()
        results["test_results"]["networking"] = self.test_networking_and_security()
        results["test_results"]["orchestrator"] = self.test_orchestrator_on_linux()
        
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
        
        logger.info(f"Linux compatibility: {results['overall']['passed_tests']}/{results['overall']['total_tests']} tests passed")
        
        return results


# pytest test functions

@pytest.mark.skipif(platform.system() != "Linux", reason="Linux-specific tests")
def test_linux_file_system():
    """Test Linux file system behaviors."""
    tester = LinuxCompatibilityTester()
    results = tester.test_file_system_specifics()
    
    # Linux should be case-sensitive
    assert results["tests"]["case_sensitive_fs"]["passed"]
    
    # Symbolic links should work
    assert results["tests"]["symbolic_links"]["passed"]


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux-specific tests")
def test_linux_system_capabilities():
    """Test Linux system capabilities."""
    tester = LinuxCompatibilityTester()
    results = tester.test_system_capabilities()
    
    # Should be able to get user info
    assert results["tests"]["user_info"]["passed"]
    
    # /proc filesystem should be accessible
    assert results["tests"]["proc_filesystem"]["passed"]


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux-specific tests")
def test_linux_networking():
    """Test Linux networking capabilities."""
    tester = LinuxCompatibilityTester()
    results = tester.test_networking_and_security()
    
    # Network capabilities test should complete
    assert results["tests"]["network_capabilities"]["passed"]


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux-specific tests")
def test_orchestrator_on_linux():
    """Test orchestrator functionality on Linux."""
    tester = LinuxCompatibilityTester()
    results = tester.test_orchestrator_on_linux()
    
    # YAML compilation should work
    assert results["tests"]["yaml_compilation"]["passed"]
    
    # Temp directories should work
    assert results["tests"]["temp_directory"]["passed"]
    
    # Signal handling should work
    assert results["tests"]["signal_handling"]["passed"]


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux-specific tests")
@pytest.mark.slow
def test_full_linux_compatibility():
    """Run comprehensive Linux compatibility test suite."""
    tester = LinuxCompatibilityTester()
    results = tester.run_all_tests()
    
    # Should not be skipped on Linux
    assert not results.get("skipped", False)
    
    # Should pass majority of tests
    assert results["overall"]["success_rate"] >= 0.8
    
    # Log results
    logger.info(f"Linux {results['distribution']['name']} {results['distribution']['version']}")
    logger.info(f"Tests passed: {results['overall']['passed_tests']}/{results['overall']['total_tests']}")


if __name__ == "__main__":
    # Run Linux compatibility tests when called directly
    tester = LinuxCompatibilityTester()
    results = tester.run_all_tests()
    
    if results.get("skipped"):
        print(f"Skipped: {results['reason']}")
    else:
        print("=== Linux Compatibility Test Results ===")
        print(f"Distribution: {results['distribution']['name']} {results['distribution']['version']}")
        print(f"Kernel: {results['kernel_version']}")
        print(f"Test Results: {results['overall']['passed_tests']}/{results['overall']['total_tests']} passed ({results['overall']['success_rate']*100:.1f}%)")
        
        for category_name, category_results in results["test_results"].items():
            if "tests" in category_results:
                print(f"\n{category_name.title()} Tests:")
                for test_name, test_data in category_results["tests"].items():
                    if test_name != "skip_reason":
                        status = "PASS" if test_data.get("passed", False) else "FAIL"
                        print(f"  {test_name}: {status}")