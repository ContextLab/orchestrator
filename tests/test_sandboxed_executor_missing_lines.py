"""Tests to cover specific missing lines in Sandboxed Executor."""

import asyncio
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import pytest

from src.orchestrator.executor.sandboxed_executor import (
    DockerSandboxExecutor,
    ProcessSandboxExecutor,
    ResourceManager,
    SandboxConfig,
    SandboxManager,
    ExecutionResult,
)


class TestableDockerClient:
    """Testable Docker client that simulates failures."""
    
    def __init__(self):
        self.should_fail = False
        self.failure_message = "Docker execution failed"
        self.call_history = []
        
    def ping(self):
        """Simulate ping."""
        self.call_history.append('ping')
        return True
        
    class containers:
        def __init__(self, parent):
            self.parent = parent
            
        def run(self, *args, **kwargs):
            self.parent.call_history.append(('run', args, kwargs))
            if self.parent.should_fail:
                raise Exception(self.parent.failure_message)
            # Return mock container with logs
            class MockContainer:
                def logs(self):
                    return b"Test output"
            return MockContainer()


class TestableDockerExecutor(DockerSandboxExecutor):
    """Testable Docker executor with injectable client."""
    
    def __init__(self, config, docker_client=None):
        super().__init__(config)
        self._test_docker_client = docker_client
        self._docker_available = docker_client is not None
        
    def _check_docker_availability(self):
        """Override to use test client."""
        return self._docker_available
        
    def _get_docker_client(self):
        """Return test client instead of real Docker client."""
        return self._test_docker_client


class TestableProcessExecutor(ProcessSandboxExecutor):
    """Testable process executor with controllable behavior."""
    
    def __init__(self, config):
        super().__init__(config)
        self._test_process = None
        self._test_exception = None
        
    def set_test_process(self, stdout: bytes, stderr: bytes, returncode: int):
        """Set test process output."""
        self._test_process = (stdout, stderr, returncode)
        
    def set_test_exception(self, exception: Exception):
        """Set exception to raise."""
        self._test_exception = exception
        
    async def _create_subprocess(self, *args, **kwargs):
        """Override to return test process."""
        if self._test_exception:
            raise self._test_exception
            
        class TestProcess:
            def __init__(self, stdout, stderr, returncode):
                self._stdout = stdout
                self._stderr = stderr
                self.returncode = returncode
                
            async def communicate(self):
                return (self._stdout, self._stderr)
                
            async def wait(self):
                pass
                
        if self._test_process:
            return TestProcess(*self._test_process)
        else:
            # Default successful execution
            return TestProcess(b"test output", b"", 0)


class TestSandboxedExecutorMissingLines:
    """Tests to cover specific missing lines in sandboxed executor."""

    @pytest.mark.asyncio
    async def test_docker_executor_exception_handling_lines_175_177(self):
        """Test lines 175-177: exception handling in Docker execute."""
        config = SandboxConfig()
        
        # Create testable Docker client
        docker_client = TestableDockerClient()
        docker_client.should_fail = True
        docker_client.failure_message = "Docker execution failed"
        
        # Create executor with test client
        executor = TestableDockerExecutor(config, docker_client)
        
        # Replace the actual docker module import
        import sys
        if 'docker' in sys.modules:
            original_docker = sys.modules['docker']
        else:
            original_docker = None
            
        # Create fake docker module
        class FakeDockerModule:
            @staticmethod
            def from_env():
                return docker_client
                
        sys.modules['docker'] = FakeDockerModule
        
        try:
            result = await executor.execute("print('test')", "python")

            # Should handle exception and return failed result (lines 175-177)
            assert result.success is False
            assert "Docker execution failed" in result.error
            assert result.execution_time > 0  # Should still record execution time
        finally:
            # Restore original docker module
            if original_docker:
                sys.modules['docker'] = original_docker
            else:
                sys.modules.pop('docker', None)

    @pytest.mark.asyncio
    async def test_docker_executor_file_cleanup_exception_lines_189_190(self):
        """Test lines 189-190: file cleanup exception handling in Docker."""
        config = SandboxConfig()
        
        # Create working Docker client
        docker_client = TestableDockerClient()
        executor = TestableDockerExecutor(config, docker_client)
        
        # Replace os.unlink temporarily
        original_unlink = os.unlink
        unlink_calls = []
        
        def failing_unlink(path):
            unlink_calls.append(path)
            raise OSError("Permission denied")
            
        os.unlink = failing_unlink
        
        # Replace the actual docker module import
        import sys
        if 'docker' in sys.modules:
            original_docker = sys.modules['docker']
        else:
            original_docker = None
            
        class FakeDockerModule:
            @staticmethod
            def from_env():
                return docker_client
                
        sys.modules['docker'] = FakeDockerModule
        
        try:
            result = await executor.execute("print('test')", "python")

            # Should handle file cleanup exception gracefully (line 190: pass)
            # The execution should complete despite cleanup failure
            assert isinstance(result.success, bool)  # Should complete without raising exception
            # Verify unlink was attempted
            assert len(unlink_calls) > 0
        finally:
            # Restore originals
            os.unlink = original_unlink
            if original_docker:
                sys.modules['docker'] = original_docker
            else:
                sys.modules.pop('docker', None)

    @pytest.mark.asyncio
    async def test_process_executor_file_cleanup_exception_lines_303_304(self):
        """Test lines 303-304: file cleanup exception handling in Process executor."""
        config = SandboxConfig()
        executor = TestableProcessExecutor(config)
        
        # Set successful process execution
        executor.set_test_process(b"test output", b"", 0)
        
        # Replace asyncio.create_subprocess_exec temporarily
        original_create = asyncio.create_subprocess_exec
        
        async def test_create(*args, **kwargs):
            return await executor._create_subprocess(*args, **kwargs)
            
        asyncio.create_subprocess_exec = test_create
        
        # Replace os.unlink temporarily
        original_unlink = os.unlink
        unlink_calls = []
        
        def failing_unlink(path):
            unlink_calls.append(path)
            raise OSError("File not found")
            
        os.unlink = failing_unlink
        
        try:
            result = await executor.execute("print('test')", "python")

            # Should handle file cleanup exception gracefully (line 304: pass)
            assert result.success is True  # Execution should succeed despite cleanup failure
            assert "test output" in result.output
            # Verify unlink was attempted
            assert len(unlink_calls) > 0
        finally:
            # Restore originals
            asyncio.create_subprocess_exec = original_create
            os.unlink = original_unlink

    def test_sandbox_manager_fallback_to_process_executor_line_342(self):
        """Test line 342: fallback to ProcessSandboxExecutor."""
        config = SandboxConfig()
        manager = SandboxManager(config)

        # Make all executors unavailable to force fallback
        for executor in manager.executors:
            if hasattr(executor, "_docker_available"):
                executor._docker_available = False

        # Replace is_available method for all executors
        for executor in manager.executors:
            original_is_available = executor.is_available
            executor.is_available = lambda: False

        available_executor = manager.get_available_executor()

        # Should fallback to ProcessSandboxExecutor (line 342)
        assert isinstance(available_executor, ProcessSandboxExecutor)
        assert available_executor.config == config

    def test_resource_manager_deallocate_nonexistent_allocation_line_647(self):
        """Test line 647: return False for nonexistent allocation."""
        manager = ResourceManager()

        # Try to deallocate allocation that doesn't exist
        result = manager.deallocate_resources("nonexistent_allocation_id")

        # Should return False (line 647)
        assert result is False
        assert "nonexistent_allocation_id" not in manager.active_allocations

    def test_resource_manager_deallocate_existing_allocation(self):
        """Test successful deallocation to contrast with line 647."""
        manager = ResourceManager()

        # First allocate resources
        allocation_id = "test_allocation_123"
        request = {"memory_gb": 1.0, "cpu_cores": 1}

        allocation = manager.allocate_resources(request, allocation_id)
        assert allocation["allocated"] is True
        assert allocation_id in manager.active_allocations

        # Then deallocate
        result = manager.deallocate_resources(allocation_id)

        # Should return True and remove from active allocations
        assert result is True
        assert allocation_id not in manager.active_allocations


class TestSandboxedExecutorImplementationGuidance:
    """Tests to verify sandboxed executor follows design patterns."""

    @pytest.mark.asyncio
    async def test_security_validation_follows_design(self):
        """Test that security validation follows design document patterns."""
        from src.orchestrator.executor.sandboxed_executor import SecurityManager

        manager = SecurityManager()

        # Test forbidden imports as per design (security patterns)
        malicious_code = "import os; os.system('rm -rf /')"
        assert manager.validate_code(malicious_code, "python") is False

        # Test safe code passes validation
        safe_code = "print('Hello World')"
        assert manager.validate_code(safe_code, "python") is True

        # Test non-Python languages are allowed (design decision)
        js_code = "console.log('test')"
        assert manager.validate_code(js_code, "javascript") is True

    def test_resource_limits_follow_design(self):
        """Test resource limits follow design document specifications."""
        from src.orchestrator.executor.sandboxed_executor import SecurityManager

        manager = SecurityManager()

        # Test design-compliant resource limits
        safe_limits = {"memory": "1g", "cpu": "50%", "processes": 10}
        assert manager.check_resource_limits(safe_limits) is True

        # Test limits that exceed design specifications
        unsafe_limits = {"memory": "100g", "cpu": "500%", "processes": 10000}
        assert manager.check_resource_limits(unsafe_limits) is False

    def test_executor_hierarchy_follows_design(self):
        """Test executor hierarchy follows design patterns."""
        # Test that Docker executor is preferred (design decision)
        config = SandboxConfig()
        manager = SandboxManager(config)

        # Should have Docker executor first in list (preferred)
        assert isinstance(manager.executors[0], DockerSandboxExecutor)
        # Should have Process executor as fallback
        assert isinstance(manager.executors[1], ProcessSandboxExecutor)

        # Test graceful degradation pattern
        available = manager.get_available_executor()
        assert available is not None  # Should always have at least process executor

    @pytest.mark.asyncio
    async def test_execution_result_abstraction_design(self):
        """Test ExecutionResult abstraction follows design."""
        from src.orchestrator.executor.sandboxed_executor import ExecutionResult

        # Test comprehensive result structure (design requirement)
        result = ExecutionResult(
            success=True,
            output="test output",
            execution_time=1.5,
            exit_code=0,
            resource_usage={"memory": 100, "cpu": 50},
        )

        assert result.success is True
        assert result.output == "test output"
        assert result.execution_time == 1.5
        assert result.exit_code == 0
        assert result.resource_usage == {"memory": 100, "cpu": 50}
        assert result.error is None  # Default value

    def test_sandbox_config_abstraction_design(self):
        """Test SandboxConfig follows design patterns."""
        # Test default configuration follows design specifications
        config = SandboxConfig()

        assert config.memory_limit == "128m"  # Conservative default
        assert config.cpu_quota == 50000  # 50% CPU limit
        assert config.time_limit == 30  # 30 second timeout
        assert config.network_disabled is True  # Security by default
        assert config.read_only_filesystem is True  # Security by default
        assert "json" in config.allowed_packages  # Safe packages allowed

        # Test customization capability (design requirement)
        custom_config = SandboxConfig(
            memory_limit="256m", cpu_quota=100000, time_limit=60, network_disabled=False
        )

        assert custom_config.memory_limit == "256m"
        assert custom_config.cpu_quota == 100000
        assert custom_config.time_limit == 60
        assert custom_config.network_disabled is False
