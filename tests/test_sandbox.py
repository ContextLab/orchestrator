"""Tests for sandboxed execution functionality."""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import patch

from src.orchestrator.executor.sandboxed_executor import (
    SandboxExecutor, DockerSandboxExecutor, SandboxedExecutor,
    SecurityManager, ResourceManager, ResourceError
)


class TestSandboxedExecutor:
    """Test cases for SandboxedExecutor class."""
    
    def test_executor_creation(self):
        """Test basic executor creation."""
        executor = SandboxedExecutor()
        
        assert executor.docker is not None
        assert executor.containers == {}
        assert executor.resource_limits["memory"] == "1g"
        assert executor.resource_limits["cpu_quota"] == 50000
        assert executor.resource_limits["pids_limit"] == 100
    
    def test_executor_with_custom_limits(self):
        """Test executor with custom resource limits."""
        custom_limits = {
            "memory": "2g",
            "cpu_quota": 100000,
            "pids_limit": 200
        }
        
        executor = SandboxedExecutor(resource_limits=custom_limits)
        
        assert executor.resource_limits["memory"] == "2g"
        assert executor.resource_limits["cpu_quota"] == 100000
        assert executor.resource_limits["pids_limit"] == 200
    
    @pytest.mark.asyncio
    async def test_execute_python_code(self):
        """Test executing Python code in sandbox."""
        executor = SandboxedExecutor()
        
        code = """
print("Hello, World!")
result = 2 + 2
print(f"Result: {result}")
"""
        
        result = await executor.execute_code(
            code=code,
            language="python",
            environment={"PYTHONPATH": "/usr/local/lib/python3.11/site-packages"},
            timeout=30
        )
        
        assert result["success"] is True
        assert "Hello, World!" in result["output"]
        assert "Result: 4" in result["output"]
        assert result["execution_time"] > 0
    
    @pytest.mark.asyncio
    async def test_execute_javascript_code(self):
        """Test executing JavaScript code in sandbox."""
        executor = SandboxedExecutor()
        
        code = """
console.log("Hello from Node.js!");
const result = 3 * 7;
console.log(`Result: ${result}`);
"""
        
        result = await executor.execute_code(
            code=code,
            language="javascript",
            environment={"NODE_ENV": "sandbox"},
            timeout=30
        )
        
        assert result["success"] is True
        assert "Hello from Node.js!" in result["output"]
        assert "Result: 21" in result["output"]
    
    @pytest.mark.asyncio
    async def test_execute_code_with_error(self):
        """Test executing code that produces errors."""
        executor = SandboxedExecutor()
        
        code = """
print("Before error")
x = 1 / 0  # Division by zero
print("After error")
"""
        
        result = await executor.execute_code(
            code=code,
            language="python",
            environment={},
            timeout=30
        )
        
        assert result["success"] is False
        assert "Before error" in result["output"]
        assert "ZeroDivisionError" in result["errors"]
        assert "After error" not in result["output"]
    
    @pytest.mark.asyncio
    async def test_execute_code_timeout(self):
        """Test code execution timeout."""
        executor = SandboxedExecutor()
        
        code = """
import time
time.sleep(10)  # Sleep longer than timeout
print("This should not execute")
"""
        
        result = await executor.execute_code(
            code=code,
            language="python",
            environment={},
            timeout=2  # 2 second timeout
        )
        
        assert result["success"] is False
        assert "timeout" in result["error"].lower()
        assert result["timeout"] == 2
    
    @pytest.mark.asyncio
    async def test_execute_malicious_code(self):
        """Test executing potentially malicious code."""
        executor = SandboxedExecutor()
        
        # Try to access file system
        code = """
import os
try:
    os.system("rm -rf /")
    print("SECURITY BREACH!")
except Exception as e:
    print(f"Security prevented: {e}")
"""
        
        result = await executor.execute_code(
            code=code,
            language="python",
            environment={},
            timeout=30
        )
        
        # Should be prevented by security measures
        assert "SECURITY BREACH!" not in result["output"]
        assert "Security prevented" in result["output"] or result["success"] is False
    
    @pytest.mark.asyncio
    async def test_execute_network_access_blocked(self):
        """Test that network access is blocked."""
        executor = SandboxedExecutor()
        
        code = """
import urllib.request
try:
    response = urllib.request.urlopen("https://google.com")
    print("NETWORK ACCESS ALLOWED!")
except Exception as e:
    print(f"Network blocked: {e}")
"""
        
        result = await executor.execute_code(
            code=code,
            language="python",
            environment={},
            timeout=30
        )
        
        assert "NETWORK ACCESS ALLOWED!" not in result["output"]
        assert "Network blocked" in result["output"] or result["success"] is False
    
    @pytest.mark.asyncio
    async def test_resource_limits_memory(self):
        """Test memory resource limits."""
        executor = SandboxedExecutor(resource_limits={"memory": "100m"})
        
        # Try to allocate large amount of memory
        code = """
try:
    # Try to allocate 1GB of memory
    large_list = [0] * (1024 * 1024 * 1024)
    print("MEMORY LIMIT EXCEEDED!")
except MemoryError:
    print("Memory limit enforced")
except Exception as e:
    print(f"Resource limit enforced: {e}")
"""
        
        result = await executor.execute_code(
            code=code,
            language="python",
            environment={},
            timeout=30
        )
        
        assert "MEMORY LIMIT EXCEEDED!" not in result["output"]
        # Should hit memory limit
        assert result["success"] is False or "limit enforced" in result["output"]
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_executions(self):
        """Test multiple concurrent code executions."""
        executor = SandboxedExecutor()
        
        async def execute_code_task(task_id):
            code = f"""
import time
time.sleep(0.1)  # Small delay
print("Task {task_id} completed")
"""
            return await executor.execute_code(
                code=code,
                language="python",
                environment={},
                timeout=30
            )
        
        # Execute 5 tasks concurrently
        tasks = [execute_code_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(result["success"] for result in results)
        assert all(f"Task {i} completed" in results[i]["output"] for i in range(5))
    
    @pytest.mark.asyncio
    async def test_container_cleanup(self):
        """Test proper container cleanup."""
        executor = SandboxedExecutor()
        
        initial_containers = len(executor.containers)
        
        code = "print('Hello')"
        await executor.execute_code(code, "python", {}, timeout=30)
        
        # Should clean up containers after execution
        assert len(executor.containers) == initial_containers
    
    def test_get_image_for_language(self):
        """Test getting appropriate Docker image for language."""
        executor = SandboxedExecutor()
        
        assert executor._get_image_for_language("python") == "orchestrator/python:3.11-slim"
        assert executor._get_image_for_language("javascript") == "orchestrator/node:18-slim"
        assert executor._get_image_for_language("unknown") == "orchestrator/ubuntu:22.04"
    
    @pytest.mark.asyncio
    async def test_cleanup_orphaned_containers(self):
        """Test cleanup of orphaned containers."""
        executor = SandboxedExecutor()
        
        # Simulate orphaned containers
        executor.containers["orphan1"] = "fake_container_1"
        executor.containers["orphan2"] = "fake_container_2"
        
        await executor.cleanup_orphaned_containers()
        
        # Should clean up orphaned containers
        assert len(executor.containers) == 0


class TestSecurityManager:
    """Test cases for SecurityManager class."""
    
    def test_security_manager_creation(self):
        """Test basic security manager creation."""
        manager = SecurityManager()
        
        assert manager.forbidden_modules == ["os", "subprocess", "eval", "exec"]
        assert manager.allowed_actions == ["generate", "transform", "analyze", "search"]
        assert manager.max_execution_time == 300
    
    def test_validate_code_safe(self):
        """Test validating safe code."""
        manager = SecurityManager()
        
        safe_code = """
def calculate_sum(a, b):
    return a + b

result = calculate_sum(5, 3)
print(result)
"""
        
        assert manager.validate_code(safe_code, "python") is True
    
    def test_validate_code_unsafe_imports(self):
        """Test validating code with unsafe imports."""
        manager = SecurityManager()
        
        unsafe_code = """
import os
import subprocess
os.system("rm -rf /")
"""
        
        assert manager.validate_code(unsafe_code, "python") is False
    
    def test_validate_code_unsafe_functions(self):
        """Test validating code with unsafe functions."""
        manager = SecurityManager()
        
        unsafe_code = """
eval("print('Hello')")
exec("x = 1 + 1")
"""
        
        assert manager.validate_code(unsafe_code, "python") is False
    
    def test_sanitize_environment(self):
        """Test sanitizing environment variables."""
        manager = SecurityManager()
        
        env = {
            "PYTHONPATH": "/safe/path",
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "SSH_PRIVATE_KEY": "secret",
            "API_KEY": "secret"
        }
        
        sanitized = manager.sanitize_environment(env)
        
        assert "PYTHONPATH" in sanitized
        assert "PATH" in sanitized
        assert "SSH_PRIVATE_KEY" not in sanitized
        assert "API_KEY" not in sanitized
    
    def test_check_resource_limits(self):
        """Test checking resource limits."""
        manager = SecurityManager()
        
        # Safe limits
        safe_limits = {"memory": "1g", "cpu": "50%", "processes": 10}
        assert manager.check_resource_limits(safe_limits) is True
        
        # Unsafe limits
        unsafe_limits = {"memory": "100g", "cpu": "500%", "processes": 10000}
        assert manager.check_resource_limits(unsafe_limits) is False


class TestResourceManager:
    """Test cases for ResourceManager class."""
    
    def test_resource_manager_creation(self):
        """Test basic resource manager creation."""
        manager = ResourceManager()
        
        assert manager.max_memory_gb == 4.0
        assert manager.max_cpu_cores == 2
        assert manager.max_processes == 100
        assert manager.max_disk_gb == 1.0
    
    def test_allocate_resources(self):
        """Test allocating resources for execution."""
        manager = ResourceManager()
        
        request = {
            "memory_gb": 1.0,
            "cpu_cores": 1,
            "processes": 10,
            "disk_gb": 0.5
        }
        
        allocation = manager.allocate_resources(request)
        
        assert allocation["memory_gb"] == 1.0
        assert allocation["cpu_cores"] == 1
        assert allocation["processes"] == 10
        assert allocation["disk_gb"] == 0.5
        assert allocation["allocated"] is True
    
    def test_allocate_resources_exceed_limits(self):
        """Test allocating resources that exceed limits."""
        manager = ResourceManager()
        
        request = {
            "memory_gb": 10.0,  # Exceeds limit
            "cpu_cores": 8,     # Exceeds limit
            "processes": 1000,  # Exceeds limit
            "disk_gb": 5.0      # Exceeds limit
        }
        
        allocation = manager.allocate_resources(request)
        
        assert allocation["allocated"] is False
        assert "memory_gb" in allocation["rejected_resources"]
        assert "cpu_cores" in allocation["rejected_resources"]
        assert "processes" in allocation["rejected_resources"]
        assert "disk_gb" in allocation["rejected_resources"]
    
    def test_deallocate_resources(self):
        """Test deallocating resources."""
        manager = ResourceManager()
        
        allocation_id = "test_allocation_123"
        request = {"memory_gb": 1.0, "cpu_cores": 1}
        
        # Allocate first
        manager.allocate_resources(request, allocation_id)
        
        # Then deallocate
        result = manager.deallocate_resources(allocation_id)
        
        assert result is True
        assert allocation_id not in manager.active_allocations
    
    def test_get_system_resources(self):
        """Test getting current system resources."""
        manager = ResourceManager()
        
        resources = manager.get_system_resources()
        
        assert "memory_available_gb" in resources
        assert "cpu_available_cores" in resources
        assert "disk_available_gb" in resources
        assert "processes_running" in resources
        assert all(isinstance(v, (int, float)) for v in resources.values())
    
    def test_resource_monitoring(self):
        """Test resource monitoring functionality."""
        manager = ResourceManager()
        
        # Start monitoring
        manager.start_monitoring()
        
        # Check monitoring status
        assert manager.monitoring_active is True
        
        # Get monitoring data
        data = manager.get_monitoring_data()
        assert "timestamp" in data
        assert "cpu_usage" in data
        assert "memory_usage" in data
        assert "disk_usage" in data
        
        # Stop monitoring
        manager.stop_monitoring()
        assert manager.monitoring_active is False