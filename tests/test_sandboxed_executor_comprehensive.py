"""Comprehensive tests for sandboxed execution functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestrator.core.task import Task
from src.orchestrator.executor.sandboxed_executor import (
    DockerSandboxExecutor,
    ExecutionResult,
    ProcessSandboxExecutor,
    ResourceError,
    ResourceManager,
    SandboxConfig,
    SandboxedExecutor,
    SandboxExecutor,
    SandboxManager,
    SecurityManager,
)


class TestExecutionResult:
    """Test cases for ExecutionResult class."""

    def test_execution_result_creation_success(self):
        """Test creating successful execution result."""
        result = ExecutionResult(
            success=True, output="Hello World", execution_time=1.5, exit_code=0
        )

        assert result.success is True
        assert result.output == "Hello World"
        assert result.error is None
        assert result.execution_time == 1.5
        assert result.exit_code == 0
        assert result.resource_usage is None

    def test_execution_result_creation_failure(self):
        """Test creating failed execution result."""
        resource_usage = {"memory": 100, "cpu": 50}
        result = ExecutionResult(
            success=False,
            output="",
            error="Test error",
            execution_time=0.5,
            exit_code=1,
            resource_usage=resource_usage,
        )

        assert result.success is False
        assert result.output == ""
        assert result.error == "Test error"
        assert result.execution_time == 0.5
        assert result.exit_code == 1
        assert result.resource_usage == resource_usage


class TestSandboxConfig:
    """Test cases for SandboxConfig class."""

    def test_sandbox_config_default_creation(self):
        """Test default sandbox config creation."""
        config = SandboxConfig()

        assert config.memory_limit == "128m"
        assert config.cpu_quota == 50000
        assert config.time_limit == 30
        assert config.network_disabled is True
        assert config.read_only_filesystem is True
        assert config.allowed_packages == ["json", "math", "datetime", "re"]

    def test_sandbox_config_custom_creation(self):
        """Test custom sandbox config creation."""
        custom_packages = ["json", "math", "numpy"]
        config = SandboxConfig(
            memory_limit="256m",
            cpu_quota=100000,
            time_limit=60,
            network_disabled=False,
            read_only_filesystem=False,
            allowed_packages=custom_packages,
        )

        assert config.memory_limit == "256m"
        assert config.cpu_quota == 100000
        assert config.time_limit == 60
        assert config.network_disabled is False
        assert config.read_only_filesystem is False
        assert config.allowed_packages == custom_packages

    def test_sandbox_config_post_init_none_packages(self):
        """Test post_init behavior when allowed_packages is None."""
        config = SandboxConfig(allowed_packages=None)

        assert config.allowed_packages == ["json", "math", "datetime", "re"]

    def test_sandbox_config_post_init_existing_packages(self):
        """Test post_init behavior when allowed_packages is already set."""
        custom_packages = ["custom", "packages"]
        config = SandboxConfig(allowed_packages=custom_packages)

        assert config.allowed_packages == custom_packages


class TestDockerSandboxExecutor:
    """Test cases for DockerSandboxExecutor class."""

    def test_docker_executor_creation(self):
        """Test Docker executor creation."""
        config = SandboxConfig(memory_limit="512m")
        executor = DockerSandboxExecutor(config)

        assert executor.config == config
        assert hasattr(executor, "_docker_available")
        assert hasattr(executor, "docker")

    def test_docker_executor_initialize_docker_import_error(self):
        """Test Docker initialization when docker module unavailable."""
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'docker'")
        ):
            executor = DockerSandboxExecutor()

            assert executor._docker_available is False
            assert executor.docker is None

    def test_docker_executor_initialize_docker_connection_error(self):
        """Test Docker initialization when Docker daemon unavailable."""
        mock_docker = MagicMock()
        mock_docker.from_env.return_value.ping.side_effect = Exception(
            "Connection failed"
        )

        with patch("builtins.__import__", return_value=mock_docker):
            executor = DockerSandboxExecutor()

            assert executor._docker_available is False
            assert executor.docker is None

    def test_docker_executor_is_available_true(self):
        """Test is_available when Docker is available."""
        executor = DockerSandboxExecutor()
        executor._docker_available = True

        assert executor.is_available() is True

    def test_docker_executor_is_available_false(self):
        """Test is_available when Docker is not available."""
        executor = DockerSandboxExecutor()
        executor._docker_available = False

        assert executor.is_available() is False

    @pytest.mark.asyncio
    async def test_docker_executor_execute_docker_unavailable(self):
        """Test execute when Docker is not available."""
        executor = DockerSandboxExecutor()
        executor._docker_available = False

        result = await executor.execute("print('hello')", "python")

        assert result.success is False
        assert result.output == ""
        assert result.error == "Docker not available"
        assert result.execution_time == 0.0

    @pytest.mark.asyncio
    async def test_docker_executor_execute_import_error(self):
        """Test execute when docker module import fails."""
        executor = DockerSandboxExecutor()
        executor._docker_available = True

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'docker'")
        ):
            result = await executor.execute("print('hello')", "python")

            assert result.success is False
            assert result.error == "Docker library not installed"

    def test_docker_executor_get_docker_image(self):
        """Test getting Docker images for different languages."""
        executor = DockerSandboxExecutor()

        assert executor._get_docker_image("python") == "python:3.11-slim"
        assert executor._get_docker_image("javascript") == "node:18-slim"
        assert executor._get_docker_image("bash") == "bash:5.1"
        assert executor._get_docker_image("unknown") == "ubuntu:22.04"

    def test_docker_executor_get_file_extension(self):
        """Test getting file extensions for different languages."""
        executor = DockerSandboxExecutor()

        assert executor._get_file_extension("python") == "py"
        assert executor._get_file_extension("javascript") == "js"
        assert executor._get_file_extension("bash") == "sh"
        assert executor._get_file_extension("unknown") == "txt"

    def test_docker_executor_get_execution_command(self):
        """Test getting execution commands for different languages."""
        executor = DockerSandboxExecutor()

        assert executor._get_execution_command("python", "test.py") == [
            "python",
            "test.py",
        ]
        assert executor._get_execution_command("javascript", "test.js") == [
            "node",
            "test.js",
        ]
        assert executor._get_execution_command("bash", "test.sh") == ["bash", "test.sh"]
        assert executor._get_execution_command("unknown", "test.txt") == [
            "cat",
            "test.txt",
        ]


class TestProcessSandboxExecutor:
    """Test cases for ProcessSandboxExecutor class."""

    def test_process_executor_creation(self):
        """Test process executor creation."""
        config = SandboxConfig(time_limit=60)
        executor = ProcessSandboxExecutor(config)

        assert executor.config == config

    def test_process_executor_is_available(self):
        """Test that process executor is always available."""
        executor = ProcessSandboxExecutor()

        assert executor.is_available() is True

    @pytest.mark.asyncio
    async def test_process_executor_execute_python_success(self):
        """Test successful Python execution in process."""
        executor = ProcessSandboxExecutor()

        code = "print('Hello Process')"
        result = await executor.execute(code, "python")

        assert result.success is True
        assert "Hello Process" in result.output
        assert result.error is None or result.error == ""
        assert result.execution_time > 0
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_process_executor_execute_python_error(self):
        """Test Python execution with error in process."""
        executor = ProcessSandboxExecutor()

        code = "raise ValueError('Test error')"
        result = await executor.execute(code, "python")

        assert result.success is False
        assert "ValueError" in result.error
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_process_executor_execute_timeout(self):
        """Test process execution timeout."""
        config = SandboxConfig(time_limit=1)
        executor = ProcessSandboxExecutor(config)

        code = "import time; time.sleep(2)"
        result = await executor.execute(code, "python")

        assert result.success is False
        assert result.error == "Execution timeout"
        assert result.exit_code == -1

    @pytest.mark.asyncio
    async def test_process_executor_execute_exception_handling(self):
        """Test process executor exception handling."""
        executor = ProcessSandboxExecutor()

        # Mock asyncio.create_subprocess_exec to raise an exception
        with patch(
            "asyncio.create_subprocess_exec", side_effect=Exception("Process error")
        ):
            result = await executor.execute("print('test')", "python")

            assert result.success is False
            assert "Process error" in result.error
            assert result.exit_code == 1

    def test_process_executor_get_file_extension(self):
        """Test getting file extensions for different languages."""
        executor = ProcessSandboxExecutor()

        assert executor._get_file_extension("python") == "py"
        assert executor._get_file_extension("javascript") == "js"
        assert executor._get_file_extension("bash") == "sh"
        assert executor._get_file_extension("unknown") == "txt"

    def test_process_executor_get_execution_command(self):
        """Test getting execution commands for different languages."""
        executor = ProcessSandboxExecutor()

        assert executor._get_execution_command("python", "/tmp/test.py") == [
            "python",
            "/tmp/test.py",
        ]
        assert executor._get_execution_command("javascript", "/tmp/test.js") == [
            "node",
            "/tmp/test.js",
        ]
        assert executor._get_execution_command("bash", "/tmp/test.sh") == [
            "bash",
            "/tmp/test.sh",
        ]
        assert executor._get_execution_command("unknown", "/tmp/test.txt") == [
            "cat",
            "/tmp/test.txt",
        ]


class TestSandboxManager:
    """Test cases for SandboxManager class."""

    def test_sandbox_manager_creation(self):
        """Test sandbox manager creation."""
        config = SandboxConfig(memory_limit="256m")
        manager = SandboxManager(config)

        assert manager.config == config
        assert len(manager.executors) == 2
        assert isinstance(manager.executors[0], DockerSandboxExecutor)
        assert isinstance(manager.executors[1], ProcessSandboxExecutor)

    def test_sandbox_manager_creation_default_config(self):
        """Test sandbox manager creation with default config."""
        manager = SandboxManager()

        assert isinstance(manager.config, SandboxConfig)
        assert len(manager.executors) == 2

    def test_sandbox_manager_get_available_executor_docker_available(self):
        """Test getting executor when Docker is available."""
        manager = SandboxManager()

        # Mock Docker as available
        manager.executors[0]._docker_available = True

        executor = manager.get_available_executor()

        assert isinstance(executor, DockerSandboxExecutor)

    def test_sandbox_manager_get_available_executor_docker_unavailable(self):
        """Test getting executor when Docker is unavailable."""
        manager = SandboxManager()

        # Mock Docker as unavailable
        manager.executors[0]._docker_available = False

        executor = manager.get_available_executor()

        assert isinstance(executor, ProcessSandboxExecutor)

    def test_sandbox_manager_get_available_executor_fallback(self):
        """Test fallback to process executor."""
        manager = SandboxManager()

        # Mock all executors as unavailable
        for executor in manager.executors:
            if hasattr(executor, "_docker_available"):
                executor._docker_available = False

        executor = manager.get_available_executor()

        assert isinstance(executor, ProcessSandboxExecutor)

    @pytest.mark.asyncio
    async def test_sandbox_manager_execute_task_success(self):
        """Test successful task execution."""
        manager = SandboxManager()

        task = Task(
            id="test_task",
            name="Test Task",
            action="execute",
            parameters={"code": "print('Task executed')", "language": "python"},
        )

        result = await manager.execute_task(task)

        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert "Task executed" in result.output

    @pytest.mark.asyncio
    async def test_sandbox_manager_execute_task_no_code(self):
        """Test task execution with missing code."""
        manager = SandboxManager()

        task = Task(
            id="test_task",
            name="Test Task",
            action="execute",
            parameters={},  # No code parameter
        )

        result = await manager.execute_task(task)

        assert result.success is False
        assert result.error == "No code provided in task parameters"
        assert result.execution_time == 0.0

    @pytest.mark.asyncio
    async def test_sandbox_manager_execute_task_custom_language(self):
        """Test task execution with custom language."""
        manager = SandboxManager()

        task = Task(
            id="test_task",
            name="Test Task",
            action="execute",
            parameters={
                "code": "console.log('JavaScript task')",
                "language": "javascript",
            },
        )

        result = await manager.execute_task(task)

        assert isinstance(result, ExecutionResult)
        # Note: May fail if Node.js not installed, but structure should be correct

    def test_sandbox_manager_get_capabilities(self):
        """Test getting manager capabilities."""
        manager = SandboxManager()

        capabilities = manager.get_capabilities()

        assert "available_executors" in capabilities
        assert "primary_executor" in capabilities
        assert "supported_languages" in capabilities
        assert capabilities["supported_languages"] == ["python", "javascript", "bash"]
        assert (
            len(capabilities["available_executors"]) >= 1
        )  # At least process executor

        # Check executor details
        for executor_info in capabilities["available_executors"]:
            assert "type" in executor_info
            assert "config" in executor_info
            assert "memory_limit" in executor_info["config"]
            assert "cpu_quota" in executor_info["config"]
            assert "time_limit" in executor_info["config"]
            assert "network_disabled" in executor_info["config"]


class TestSandboxedExecutorClass:
    """Test cases for SandboxedExecutor class (test compatibility wrapper)."""

    def test_sandboxed_executor_creation_default(self):
        """Test SandboxedExecutor creation with defaults."""
        executor = SandboxedExecutor()

        assert executor.resource_limits["memory"] == "1g"
        assert executor.resource_limits["cpu_quota"] == 50000
        assert executor.resource_limits["pids_limit"] == 100
        assert executor.containers == {}

    def test_sandboxed_executor_creation_custom_limits(self):
        """Test SandboxedExecutor creation with custom limits."""
        custom_limits = {"memory": "2g", "cpu_quota": 100000, "pids_limit": 200}

        executor = SandboxedExecutor(resource_limits=custom_limits)

        assert executor.resource_limits["memory"] == "2g"
        assert executor.resource_limits["cpu_quota"] == 100000
        assert executor.resource_limits["pids_limit"] == 200

    @pytest.mark.asyncio
    async def test_sandboxed_executor_execute_code_security_validation_failure(self):
        """Test execute_code with security validation failure."""
        executor = SandboxedExecutor()

        # Code that should fail security validation
        malicious_code = "import os; os.system('rm -rf /')"

        result = await executor.execute_code(malicious_code, "python", {}, 30)

        assert result["success"] is False
        assert "Security validation failed" in result["error"]
        assert "forbidden imports" in result["errors"]
        assert result["execution_time"] == 0.0

    @pytest.mark.asyncio
    async def test_sandboxed_executor_execute_code_docker_unavailable(self):
        """Test execute_code when Docker is unavailable."""
        executor = SandboxedExecutor()
        executor._docker_available = False

        # Mock the process executor execute method directly
        mock_result = ExecutionResult(
            success=True, output="Process result", execution_time=1.0
        )

        with patch(
            "src.orchestrator.executor.sandboxed_executor.ProcessSandboxExecutor"
        ) as mock_process_class:
            mock_process_instance = AsyncMock()
            mock_process_instance.execute.return_value = mock_result
            mock_process_class.return_value = mock_process_instance

            result = await executor.execute_code("print('test')", "python", {}, 30)

            assert result["success"] is True
            assert result["output"] == "Process result"

    @pytest.mark.asyncio
    async def test_sandboxed_executor_execute_code_success(self):
        """Test successful execute_code."""
        executor = SandboxedExecutor()

        with patch.object(executor, "execute") as mock_execute:
            mock_execute.return_value = ExecutionResult(
                success=True, output="Success output", execution_time=1.5
            )

            result = await executor.execute_code("print('test')", "python", {}, 30)

            assert result["success"] is True
            assert result["output"] == "Success output"
            assert "execution_time" in result

    @pytest.mark.asyncio
    async def test_sandboxed_executor_execute_code_failure(self):
        """Test failed execute_code."""
        executor = SandboxedExecutor()

        with patch.object(executor, "execute") as mock_execute:
            mock_execute.return_value = ExecutionResult(
                success=False,
                output="Error output",
                error="Test error",
                execution_time=0.5,
            )

            result = await executor.execute_code("invalid code", "python", {}, 30)

            assert result["success"] is False
            assert result["error"] == "Test error"
            assert result["errors"] == "Test error"

    @pytest.mark.asyncio
    async def test_sandboxed_executor_execute_code_exception(self):
        """Test execute_code with unexpected exception."""
        executor = SandboxedExecutor()

        with patch.object(
            executor, "execute", side_effect=Exception("Unexpected error")
        ):
            result = await executor.execute_code("print('test')", "python", {}, 30)

            assert result["success"] is False
            assert "Unexpected error" in result["error"]
            assert "Unexpected error" in result["errors"]

    def test_sandboxed_executor_get_image_for_language(self):
        """Test getting Docker images for languages."""
        executor = SandboxedExecutor()

        assert (
            executor._get_image_for_language("python")
            == "orchestrator/python:3.11-slim"
        )
        assert (
            executor._get_image_for_language("javascript")
            == "orchestrator/node:18-slim"
        )
        assert executor._get_image_for_language("bash") == "orchestrator/bash:5.1"
        assert (
            executor._get_image_for_language("unknown") == "orchestrator/ubuntu:22.04"
        )

    @pytest.mark.asyncio
    async def test_sandboxed_executor_cleanup_orphaned_containers(self):
        """Test cleaning up orphaned containers."""
        executor = SandboxedExecutor()

        # Add some fake containers
        executor.containers["container1"] = "fake_container_1"
        executor.containers["container2"] = "fake_container_2"

        await executor.cleanup_orphaned_containers()

        assert len(executor.containers) == 0


class TestSecurityManagerAdvanced:
    """Advanced test cases for SecurityManager class."""

    def test_security_manager_validate_code_non_python(self):
        """Test code validation for non-Python languages."""
        manager = SecurityManager()

        # JavaScript code with potentially dangerous content
        js_code = "const fs = require('fs'); fs.unlinkSync('/important/file');"

        # Should return True for non-Python languages (validation skipped)
        assert manager.validate_code(js_code, "javascript") is True

    def test_security_manager_validate_code_edge_cases(self):
        """Test code validation edge cases."""
        manager = SecurityManager()

        # Test partial matches that should not trigger
        safe_code_1 = "import json  # This is safe"
        assert manager.validate_code(safe_code_1, "python") is True

        # Test substring matches that should not trigger
        safe_code_2 = "my_import_function = 'os'"  # Contains 'os' but not 'import os'
        assert manager.validate_code(safe_code_2, "python") is True

        # Test word boundary matching for functions
        safe_code_3 = "result = execute_task()"  # Contains 'exec' but not 'exec('
        assert manager.validate_code(safe_code_3, "python") is True

        # Test actual dangerous patterns
        unsafe_code_1 = "import os\nos.system('rm -rf /')"
        assert manager.validate_code(unsafe_code_1, "python") is False

        unsafe_code_2 = "from subprocess import call"
        assert manager.validate_code(unsafe_code_2, "python") is False

    def test_security_manager_sanitize_environment_comprehensive(self):
        """Test comprehensive environment sanitization."""
        manager = SecurityManager()

        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/user",
            "USER": "testuser",
            "PWD": "/current/dir",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "PYTHONPATH": "/usr/lib/python3",
            "SSH_PRIVATE_KEY": "-----BEGIN PRIVATE KEY-----",
            "API_KEY": "secret123",
            "SECRET_KEY": "supersecret",
            "PASSWORD": "mypassword",
            "AWS_SECRET_ACCESS_KEY": "awssecret",
            "CUSTOM_VAR": "should_be_removed",
        }

        sanitized = manager.sanitize_environment(env)

        # Safe variables should be kept
        assert "PATH" in sanitized
        assert "HOME" in sanitized
        assert "USER" in sanitized
        assert "PWD" in sanitized
        assert "LANG" in sanitized
        assert "LC_ALL" in sanitized
        assert "PYTHONPATH" in sanitized

        # Dangerous variables should be removed
        assert "SSH_PRIVATE_KEY" not in sanitized
        assert "API_KEY" not in sanitized
        assert "SECRET_KEY" not in sanitized
        assert "PASSWORD" not in sanitized
        assert "AWS_SECRET_ACCESS_KEY" not in sanitized
        assert "CUSTOM_VAR" not in sanitized

    def test_security_manager_check_resource_limits_edge_cases(self):
        """Test resource limits checking edge cases."""
        manager = SecurityManager()

        # Test various memory formats
        assert manager.check_resource_limits({"memory": "1g"}) is True
        assert manager.check_resource_limits({"memory": "10g"}) is True
        assert manager.check_resource_limits({"memory": "11g"}) is False
        assert (
            manager.check_resource_limits({"memory": "1000m"}) is True
        )  # No 'm' parsing implemented

        # Test various CPU formats
        assert manager.check_resource_limits({"cpu": "50%"}) is True
        assert manager.check_resource_limits({"cpu": "100%"}) is True
        assert manager.check_resource_limits({"cpu": "101%"}) is False

        # Test process limits
        assert manager.check_resource_limits({"processes": 100}) is True
        assert manager.check_resource_limits({"processes": 1000}) is True
        assert manager.check_resource_limits({"processes": 1001}) is False

        # Test combined limits
        good_limits = {"memory": "5g", "cpu": "80%", "processes": 500}
        assert manager.check_resource_limits(good_limits) is True

        bad_limits = {"memory": "15g", "cpu": "150%", "processes": 2000}
        assert manager.check_resource_limits(bad_limits) is False

    def test_security_manager_attributes(self):
        """Test SecurityManager attribute initialization."""
        manager = SecurityManager()

        assert "os" in manager.forbidden_modules
        assert "subprocess" in manager.forbidden_modules
        assert "eval" in manager.forbidden_modules
        assert "exec" in manager.forbidden_modules

        assert "generate" in manager.allowed_actions
        assert "transform" in manager.allowed_actions
        assert "analyze" in manager.allowed_actions
        assert "search" in manager.allowed_actions

        assert manager.max_execution_time == 300

        # Test unsafe imports set
        assert "os" in manager.unsafe_imports
        assert "subprocess" in manager.unsafe_imports
        assert "socket" in manager.unsafe_imports

        # Test safe imports set
        assert "json" in manager.safe_imports
        assert "math" in manager.safe_imports
        assert "datetime" in manager.safe_imports

        # Test unsafe functions set
        assert "eval" in manager.unsafe_functions
        assert "exec" in manager.unsafe_functions
        assert "open" in manager.unsafe_functions

        # Test resource limits
        assert manager.resource_limits["max_memory"] == 128 * 1024 * 1024
        assert manager.resource_limits["max_cpu_time"] == 30
        assert manager.resource_limits["max_file_size"] == 1024 * 1024


class TestResourceManagerAdvanced:
    """Advanced test cases for ResourceManager class."""

    def test_resource_manager_allocate_resources_with_allocation_id(self):
        """Test resource allocation with specific allocation ID."""
        manager = ResourceManager()

        request = {"memory_gb": 2.0, "cpu_cores": 1, "processes": 50, "disk_gb": 0.8}
        allocation_id = "test_allocation_123"

        allocation = manager.allocate_resources(request, allocation_id)

        assert allocation["allocated"] is True
        assert allocation["memory_gb"] == 2.0
        assert allocation["cpu_cores"] == 1
        assert allocation["processes"] == 50
        assert allocation["disk_gb"] == 0.8
        assert "allocated_at" in allocation
        assert allocation_id in manager.active_allocations

    def test_resource_manager_can_allocate_internal(self):
        """Test internal _can_allocate method."""
        manager = ResourceManager()

        # Valid request
        valid_request = {
            "memory_gb": 2.0,
            "cpu_cores": 1,
            "processes": 50,
            "disk_gb": 0.5,
        }
        assert manager._can_allocate(valid_request) is True

        # Invalid request - exceeds memory
        invalid_memory = {
            "memory_gb": 10.0,  # Exceeds max_memory_gb (4.0)
            "cpu_cores": 1,
            "processes": 50,
            "disk_gb": 0.5,
        }
        assert manager._can_allocate(invalid_memory) is False

        # Invalid request - exceeds CPU
        invalid_cpu = {
            "memory_gb": 2.0,
            "cpu_cores": 5,  # Exceeds max_cpu_cores (2)
            "processes": 50,
            "disk_gb": 0.5,
        }
        assert manager._can_allocate(invalid_cpu) is False

        # Invalid request - exceeds processes
        invalid_processes = {
            "memory_gb": 2.0,
            "cpu_cores": 1,
            "processes": 500,  # Exceeds max_processes (100)
            "disk_gb": 0.5,
        }
        assert manager._can_allocate(invalid_processes) is False

        # Invalid request - exceeds disk
        invalid_disk = {
            "memory_gb": 2.0,
            "cpu_cores": 1,
            "processes": 50,
            "disk_gb": 2.0,  # Exceeds max_disk_gb (1.0)
        }
        assert manager._can_allocate(invalid_disk) is False

    def test_resource_manager_get_system_resources_with_psutil(self):
        """Test getting system resources with psutil available."""
        manager = ResourceManager()

        # Mock psutil
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value.available = 8 * (1024**3)  # 8GB
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.disk_usage.return_value.free = 500 * (1024**3)  # 500GB
        mock_psutil.pids.return_value = list(range(200))  # 200 processes

        with patch("builtins.__import__", return_value=mock_psutil):
            resources = manager.get_system_resources()

            assert resources["memory_available_gb"] == 8.0
            assert resources["cpu_available_cores"] == 8
            assert resources["disk_available_gb"] == 500.0
            assert resources["processes_running"] == 200

    def test_resource_manager_get_system_resources_without_psutil(self):
        """Test getting system resources without psutil."""
        manager = ResourceManager()

        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'psutil'")
        ):
            resources = manager.get_system_resources()

            # Should return default values
            assert resources["memory_available_gb"] == 8.0
            assert resources["cpu_available_cores"] == 4
            assert resources["disk_available_gb"] == 100.0
            assert resources["processes_running"] == 50

    def test_resource_manager_monitoring_lifecycle(self):
        """Test complete monitoring lifecycle."""
        manager = ResourceManager()

        # Initially not monitoring
        assert manager.monitoring_active is False

        # Start monitoring
        manager.start_monitoring()
        assert manager.monitoring_active is True

        # Get monitoring data
        data = manager.get_monitoring_data()
        assert "timestamp" in data
        assert "cpu_usage" in data
        assert "memory_usage" in data
        assert "disk_usage" in data
        assert isinstance(data["timestamp"], float)
        assert isinstance(data["cpu_usage"], float)
        assert isinstance(data["memory_usage"], float)
        assert isinstance(data["disk_usage"], float)

        # Stop monitoring
        manager.stop_monitoring()
        assert manager.monitoring_active is False

    def test_resource_manager_monitor_resources(self):
        """Test resource monitoring functionality."""
        manager = ResourceManager()

        # Add some allocated resources
        manager.allocated_resources["task1"] = {"memory": 1024, "cpu_cores": 2}
        manager.allocated_resources["task2"] = {"memory": 512, "cpu_cores": 1}

        monitoring_data = manager.monitor_resources()

        assert monitoring_data["total_allocated"] == 2
        assert monitoring_data["memory_used"] == 1536  # 1024 + 512
        assert monitoring_data["cpu_used"] == 3  # 2 + 1
        assert "task1" in monitoring_data["tasks"]
        assert "task2" in monitoring_data["tasks"]

    def test_resource_manager_initialization_attributes(self):
        """Test ResourceManager initialization attributes."""
        manager = ResourceManager()

        assert manager.max_memory_gb == 4.0
        assert manager.max_cpu_cores == 2
        assert manager.max_processes == 100
        assert manager.max_disk_gb == 1.0

        assert manager.allocated_resources == {}
        assert manager.active_allocations == {}
        assert manager.monitoring_active is False

        assert manager.resource_limits["max_memory"] == 1024 * 1024 * 1024
        assert manager.resource_limits["max_cpu_cores"] == 2
        assert manager.resource_limits["max_processes"] == 10

        assert hasattr(manager, "logger")


class TestResourceError:
    """Test cases for ResourceError exception."""

    def test_resource_error_creation(self):
        """Test ResourceError exception creation."""
        error = ResourceError("Resource allocation failed")

        assert str(error) == "Resource allocation failed"
        assert isinstance(error, Exception)

    def test_resource_error_raise(self):
        """Test raising ResourceError exception."""
        with pytest.raises(ResourceError) as exc_info:
            raise ResourceError("Test resource error")

        assert str(exc_info.value) == "Test resource error"


class TestSandboxExecutorAbstract:
    """Test cases for abstract SandboxExecutor class."""

    def test_sandbox_executor_abstract_instantiation(self):
        """Test that SandboxExecutor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SandboxExecutor()

    def test_sandbox_executor_subclass_missing_methods(self):
        """Test that subclasses must implement abstract methods."""

        class IncompleteExecutor(SandboxExecutor):
            # Missing execute and is_available methods
            pass

        with pytest.raises(TypeError):
            IncompleteExecutor()

    def test_sandbox_executor_subclass_complete(self):
        """Test complete subclass implementation."""

        class CompleteExecutor(SandboxExecutor):
            async def execute(
                self, code: str, language: str = "python"
            ) -> ExecutionResult:
                return ExecutionResult(success=True, output="test", execution_time=1.0)

            def is_available(self) -> bool:
                return True

        # Should be able to instantiate
        executor = CompleteExecutor()
        assert executor.is_available() is True
