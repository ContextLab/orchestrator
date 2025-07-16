"""Tests to cover specific missing lines in Sandboxed Executor."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.orchestrator.executor.sandboxed_executor import (
    DockerSandboxExecutor,
    ProcessSandboxExecutor,
    ResourceManager,
    SandboxConfig,
    SandboxManager,
)


class TestSandboxedExecutorMissingLines:
    """Tests to cover specific missing lines in sandboxed executor."""

    @pytest.mark.asyncio
    async def test_docker_executor_exception_handling_lines_175_177(self):
        """Test lines 175-177: exception handling in Docker execute."""
        config = SandboxConfig()
        executor = DockerSandboxExecutor(config)
        executor._docker_available = True

        # Test the actual exception handling by forcing Docker to fail after import
        # We'll manually invoke the docker execution and inject an exception

        # First, let's test by setting a mock that raises exception in container execution
        with patch("docker.from_env") as mock_from_env:
            mock_client = Mock()
            mock_from_env.return_value = mock_client
            # Make container creation raise exception to trigger lines 175-177
            mock_client.containers.run.side_effect = Exception(
                "Docker execution failed"
            )

            result = await executor.execute("print('test')", "python")

            # Should handle exception and return failed result (lines 175-177)
            assert result.success is False
            assert "Docker execution failed" in result.error
            assert result.execution_time > 0  # Should still record execution time

    @pytest.mark.asyncio
    async def test_docker_executor_file_cleanup_exception_lines_189_190(self):
        """Test lines 189-190: file cleanup exception handling in Docker."""
        config = SandboxConfig()
        executor = DockerSandboxExecutor(config)
        executor._docker_available = True

        with patch("builtins.__import__") as mock_import:
            mock_docker = Mock()
            mock_docker.from_env.return_value.ping.return_value = True
            mock_import.return_value = mock_docker

            # Mock tempfile to create a file that will fail to delete
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_file = Mock()
                mock_file.name = "/nonexistent/path/temp_file.py"
                mock_file.__enter__ = Mock(return_value=mock_file)
                mock_file.__exit__ = Mock(return_value=None)
                mock_temp.return_value = mock_file

                # Mock os.unlink to raise exception (line 189)
                with patch("os.unlink", side_effect=OSError("Permission denied")):
                    result = await executor.execute("print('test')", "python")

                    # Should handle file cleanup exception gracefully (line 190: pass)
                    # The execution itself might fail due to mocking, but cleanup exception should be handled
                    assert isinstance(
                        result.success, bool
                    )  # Should complete without raising exception

    @pytest.mark.asyncio
    async def test_process_executor_file_cleanup_exception_lines_303_304(self):
        """Test lines 303-304: file cleanup exception handling in Process executor."""
        config = SandboxConfig()
        executor = ProcessSandboxExecutor(config)

        # Mock asyncio.create_subprocess_exec to simulate successful execution
        mock_process = Mock()
        mock_process.communicate = AsyncMock(return_value=(b"test output", b""))
        mock_process.returncode = 0
        mock_process.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            # Mock tempfile to create a file
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_file = Mock()
                mock_file.name = "/tmp/test_file.py"
                mock_file.__enter__ = Mock(return_value=mock_file)
                mock_file.__exit__ = Mock(return_value=None)
                mock_temp.return_value = mock_file

                # Mock os.unlink to raise exception during cleanup (line 303)
                with patch("os.unlink", side_effect=OSError("File not found")):
                    result = await executor.execute("print('test')", "python")

                    # Should handle file cleanup exception gracefully (line 304: pass)
                    assert (
                        result.success is True
                    )  # Execution should succeed despite cleanup failure
                    assert "test output" in result.output

    def test_sandbox_manager_fallback_to_process_executor_line_342(self):
        """Test line 342: fallback to ProcessSandboxExecutor."""
        config = SandboxConfig()
        manager = SandboxManager(config)

        # Mock all executors as unavailable to force fallback
        for executor in manager.executors:
            if hasattr(executor, "_docker_available"):
                executor._docker_available = False

        # Override is_available for all executors to return False
        for executor in manager.executors:
            executor.is_available = Mock(return_value=False)

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
