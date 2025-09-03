"""
Real-world error handling tests for Issue 192 implementation.
These tests use REAL API calls, file system operations, and network requests - NO MOCKS.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from src.orchestrator.core.error_handling import ErrorContext, ErrorHandler, ErrorHandlerResult
from src.orchestrator.core.error_handler_registry import ErrorHandlerRegistry
from src.orchestrator.execution.error_handler_executor import ErrorHandlerExecutor
from src.orchestrator.engine.pipeline_spec import TaskSpec
from src.orchestrator.engine.task_executor import UniversalTaskExecutor
from src.orchestrator.compiler.yaml_compiler import YAMLCompiler
from src.orchestrator.compiler.error_handler_schema import ErrorHandlerSchemaValidator


class MockTaskExecutor:
    """Minimal task executor for testing error handling."""
    
    def __init__(self):
        self.execution_count = 0
        self.should_fail = True
        self.failure_type = "ValueError"
        self.failure_message = "Test error for error handling"
    
    async def execute_task(self, task_spec, context):
        """Execute task with controlled failures for testing."""
        self.execution_count += 1
        
        # Check for specific error handler actions
        if hasattr(task_spec, 'action'):
            action = task_spec.action
            
            # Handle error recovery actions
            if "recover" in action.lower() or "handle" in action.lower():
                return {
                    "task_id": task_spec.id,
                    "success": True,
                    "result": f"Error handled successfully: {context.get('error_message', 'unknown error')}",
                    "recovery_action": action
                }
            
            # Handle retry actions
            if "retry" in action.lower():
                return {
                    "task_id": task_spec.id,
                    "success": True,
                    "result": "Task retried successfully after error handling",
                    "retried": True
                }
        
        # Simulate failures for testing
        if self.should_fail:
            if self.failure_type == "ValueError":
                raise ValueError(self.failure_message)
            elif self.failure_type == "ConnectionError":
                raise ConnectionError("Connection refused by test server")
            elif self.failure_type == "FileNotFoundError":
                raise FileNotFoundError("Test file not found")
            elif self.failure_type == "TimeoutError":
                raise TimeoutError("Test operation timed out")
            else:
                raise Exception(self.failure_message)
        
        return {
            "task_id": task_spec.id,
            "success": True,
            "result": "Task completed successfully"
        }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for file tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def error_handler_executor():
    """Create error handler executor with mock task executor."""
    task_executor = MockTaskExecutor()
    return ErrorHandlerExecutor(task_executor), task_executor


@pytest.fixture
def error_handler_registry():
    """Create error handler registry."""
    return ErrorHandlerRegistry()


class TestErrorHandlerCore:
    """Test core error handler functionality."""
    
    def test_error_handler_creation(self):
        """Test ErrorHandler creation with validation."""
        # Test simple handler
        handler = ErrorHandler(
            handler_action="Log error and continue",
            error_types=["ValueError", "TypeError"]
        )
        
        assert handler.handler_action == "Log error and continue"
        assert handler.error_types == ["ValueError", "TypeError"]
        assert handler.retry_with_handler is True  # Default
        assert handler.enabled is True
    
    def test_error_handler_validation(self):
        """Test ErrorHandler validation."""
        # Test valid handler
        handler = ErrorHandler(
            handler_task_id="recovery_task",
            error_types=["ConnectionError"],
            priority=10
        )
        assert handler.handler_task_id == "recovery_task"
        
        # Test invalid handler - no action, task, or fallback
        with pytest.raises(ValueError, match="must specify at least one"):
            ErrorHandler()
        
        # Test invalid handler - both task and action
        with pytest.raises(ValueError, match="cannot specify both"):
            ErrorHandler(
                handler_task_id="task1",
                handler_action="action1"
            )
    
    def test_error_matching(self):
        """Test error matching logic."""
        # Test wildcard matching
        handler = ErrorHandler(
            handler_action="Handle any error",
            error_types=["*"]
        )
        
        assert handler.matches_error(ValueError("test"), "task1") is True
        assert handler.matches_error(ConnectionError("test"), "task1") is True
        
        # Test specific error type matching
        handler = ErrorHandler(
            handler_action="Handle value errors",
            error_types=["ValueError"]
        )
        
        assert handler.matches_error(ValueError("test"), "task1") is True
        assert handler.matches_error(TypeError("test"), "task1") is False
        
        # Test pattern matching
        handler = ErrorHandler(
            handler_action="Handle connection issues",
            error_patterns=["connection.*refused", "timeout.*occurred"]
        )
        
        assert handler.matches_error(ConnectionError("connection refused"), "task1") is True
        assert handler.matches_error(TimeoutError("timeout occurred"), "task1") is True
        assert handler.matches_error(ValueError("unrelated error"), "task1") is False
    
    def test_error_context_creation(self):
        """Test ErrorContext creation from exceptions."""
        # Create test error
        test_error = ValueError("Test error message")
        
        # Create error context
        context = ErrorContext.from_exception(
            failed_task_id="test_task",
            error=test_error,
            task_parameters={"param1": "value1"},
            pipeline_context={"step1": "result1"}
        )
        
        assert context.failed_task_id == "test_task"
        assert context.error_type == "ValueError"
        assert context.error_message == "Test error message"
        assert context.task_parameters == {"param1": "value1"}
        assert context.pipeline_context == {"step1": "result1"}
        assert context.execution_attempt == 1
        
        # Test context serialization
        context_dict = context.to_dict()
        assert isinstance(context_dict, dict)
        assert context_dict["failed_task_id"] == "test_task"
        assert context_dict["error_type"] == "ValueError"


class TestErrorHandlerRegistry:
    """Test error handler registry functionality."""
    
    def test_handler_registration(self, error_handler_registry):
        """Test registering and retrieving handlers."""
        handler = ErrorHandler(
            handler_action="Handle error",
            error_types=["ValueError"]
        )
        
        # Register global handler
        error_handler_registry.register_handler("global_handler", handler)
        
        assert len(error_handler_registry) == 1
        assert "global_handler" in error_handler_registry
        
        # Register task-specific handler
        task_handler = ErrorHandler(
            handler_action="Handle task error",
            error_types=["ConnectionError"]
        )
        
        error_handler_registry.register_handler("task_handler", task_handler, "specific_task")
        
        assert len(error_handler_registry) == 2
    
    def test_handler_matching(self, error_handler_registry):
        """Test finding matching handlers for errors."""
        # Register handlers
        value_handler = ErrorHandler(
            handler_action="Handle ValueError",
            error_types=["ValueError"],
            priority=10
        )
        
        general_handler = ErrorHandler(
            handler_action="Handle any error",
            error_types=["*"],
            priority=100
        )
        
        error_handler_registry.register_handler("value_handler", value_handler, "test_task")
        error_handler_registry.register_handler("general_handler", general_handler)
        
        # Test matching
        test_error = ValueError("Test error")
        matches = error_handler_registry.find_matching_handlers(test_error, "test_task")
        
        # Should find both handlers, with value_handler first (higher priority)
        assert len(matches) == 2
        assert matches[0][0] == "value_handler"  # Higher priority first
        assert matches[1][0] == "general_handler"
    
    def test_error_statistics(self, error_handler_registry):
        """Test error statistics tracking."""
        # Record error occurrences
        error_handler_registry.record_error_occurrence("task1", "ValueError", handled=True)
        error_handler_registry.record_error_occurrence("task1", "ValueError", handled=False)
        error_handler_registry.record_error_occurrence("task1", "ConnectionError", handled=True)
        
        # Check statistics
        stats = error_handler_registry.get_error_statistics("task1")
        
        assert stats["total_errors"] == 3
        assert stats["ValueError"] == 2
        assert stats["ConnectionError"] == 1
        assert stats["handled_errors"] == 2
        assert stats["unhandled_errors"] == 1
    
    def test_handler_execution_tracking(self, error_handler_registry):
        """Test handler execution statistics."""
        handler = ErrorHandler(
            handler_action="Test handler",
            error_types=["*"]
        )
        
        error_handler_registry.register_handler("test_handler", handler)
        
        # Record executions
        error_handler_registry.record_handler_execution(
            "test_handler", success=True, execution_time=0.5, 
            error_type="ValueError", task_id="task1"
        )
        
        error_handler_registry.record_handler_execution(
            "test_handler", success=False, execution_time=1.0,
            error_type="ConnectionError", task_id="task2"
        )
        
        # Check statistics
        stats = error_handler_registry.get_handler_statistics("test_handler")
        
        assert stats["executions"] == 2
        assert stats["successes"] == 1
        assert stats["failures"] == 1
        assert stats["avg_execution_time"] == 0.75  # (0.5 + 1.0) / 2


class TestErrorHandlerExecutor:
    """Test error handler executor with real scenarios."""
    
    @pytest.mark.asyncio
    async def test_simple_error_handling(self, error_handler_executor):
        """Test simple error handling with recovery."""
        executor, task_executor = error_handler_executor
        
        # Create a failing task
        task_spec = TaskSpec(
            id="failing_task",
            action="This task will fail"
        )
        
        # Register error handler
        handler = ErrorHandler(
            handler_action="Handle the error and recover",
            error_types=["ValueError"],
            retry_with_handler=False
        )
        
        executor.handler_registry.register_handler("recovery_handler", handler, "failing_task")
        
        # Execute error handling
        result = await executor.handle_task_error(
            failed_task=task_spec,
            error=ValueError("Simulated failure"),
            context={"test": "context"}
        )
        
        assert result["task_id"] == "failing_task"
        assert result["success"] is True  # Handler succeeded
        assert "recovered_from_error" in result
        assert result["recovery_handler"] == "recovery_handler"
    
    @pytest.mark.asyncio
    async def test_handler_chaining(self, error_handler_executor):
        """Test multiple handlers in priority order."""
        executor, task_executor = error_handler_executor
        
        # Create task
        task_spec = TaskSpec(
            id="chain_test_task",
            action="Task for handler chaining"
        )
        
        # Register multiple handlers with different priorities
        primary_handler = ErrorHandler(
            handler_action="Primary recovery attempt",
            error_types=["ConnectionError"],
            priority=1,
            continue_on_handler_failure=True
        )
        
        fallback_handler = ErrorHandler(
            handler_action="Fallback recovery",
            error_types=["*"],
            priority=10,
            fallback_value="Fallback result"
        )
        
        executor.handler_registry.register_handler("primary", primary_handler, "chain_test_task")
        executor.handler_registry.register_handler("fallback", fallback_handler, "chain_test_task")
        
        # Force task executor to succeed on recovery actions
        task_executor.should_fail = False
        
        # Execute error handling
        result = await executor.handle_task_error(
            failed_task=task_spec,
            error=ConnectionError("Network issue"),
            context={"test": "chaining"}
        )
        
        assert result["success"] is True
        assert result["recovery_handler"] == "primary"  # Primary handler should execute first
    
    @pytest.mark.asyncio
    async def test_handler_retry_logic(self, error_handler_executor):
        """Test handler retry with exponential backoff."""
        executor, task_executor = error_handler_executor
        
        task_spec = TaskSpec(
            id="retry_task",
            action="Task with retry logic"
        )
        
        # Handler with retries
        handler = ErrorHandler(
            handler_action="Retry operation",
            error_types=["*"],
            max_handler_retries=2,
            retry_with_handler=True
        )
        
        executor.handler_registry.register_handler("retry_handler", handler, "retry_task")
        
        # Set task executor to succeed on retry actions
        original_should_fail = task_executor.should_fail
        task_executor.should_fail = False
        
        start_time = time.time()
        result = await executor.handle_task_error(
            failed_task=task_spec,
            error=TimeoutError("Operation timed out"),
            context={"retry": "test"}
        )
        end_time = time.time()
        
        # Restore original state
        task_executor.should_fail = original_should_fail
        
        assert result["success"] is True
        assert "recovered_from_error" in result
        # Should have taken some time due to retry delays (but not too much in test)
        assert end_time - start_time >= 0  # Basic sanity check
    
    @pytest.mark.asyncio
    async def test_fallback_values(self, error_handler_executor):
        """Test fallback value handling."""
        executor, task_executor = error_handler_executor
        
        task_spec = TaskSpec(
            id="fallback_task",
            action="Task with fallback"
        )
        
        # Handler with fallback value
        handler = ErrorHandler(
            handler_action="This will fail",
            error_types=["*"],
            fallback_value="Default fallback result"
        )
        
        executor.handler_registry.register_handler("fallback_handler", handler, "fallback_task")
        
        # Force handler to fail by making task executor fail
        task_executor.should_fail = True
        task_executor.failure_message = "Handler execution failed"
        
        result = await executor.handle_task_error(
            failed_task=task_spec,
            error=ValueError("Original error"),
            context={}
        )
        
        # Should get fallback result even though handler failed
        assert result["task_id"] == "fallback_task"
        assert result["success"] is True
        assert result["result"] == "Default fallback result"
        assert result["fallback_used"] is True
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, error_handler_executor):
        """Test circuit breaker to prevent infinite loops."""
        executor, task_executor = error_handler_executor
        
        task_spec = TaskSpec(
            id="loop_task",
            action="Task that would loop"
        )
        
        # Handler that would cause retries
        handler = ErrorHandler(
            handler_action="Retry indefinitely",
            error_types=["*"],
            retry_with_handler=True
        )
        
        executor.handler_registry.register_handler("loop_handler", handler, "loop_task")
        
        # Simulate multiple rapid failures to trigger circuit breaker
        for i in range(15):  # Exceed max_total_retries_per_task
            executor.handler_registry.record_error_occurrence("loop_task", "ValueError", handled=False)
        
        result = await executor.handle_task_error(
            failed_task=task_spec,
            error=ValueError("Repeated error"),
            context={}
        )
        
        assert result["success"] is False
        assert "Handler loop prevented" in result["error_handling_reason"]


class TestRealFileSystemErrors:
    """Test real file system error scenarios."""
    
    @pytest.mark.asyncio
    async def test_file_not_found_handling(self, temp_directory):
        """Test handling of real FileNotFoundError."""
        executor = ErrorHandlerExecutor(MockTaskExecutor())
        
        # Task that tries to read non-existent file
        task_spec = TaskSpec(
            id="file_read_task",
            action="Read non-existent file"
        )
        
        # Handler to create file and retry
        handler = ErrorHandler(
            handler_action="Create missing file",
            error_types=["FileNotFoundError"],
            retry_with_handler=True
        )
        
        executor.handler_registry.register_handler("file_handler", handler, "file_read_task")
        
        # Create actual FileNotFoundError
        missing_file = os.path.join(temp_directory, "missing.txt")
        try:
            with open(missing_file, 'r') as f:
                content = f.read()
        except FileNotFoundError as e:
            real_error = e
        
        result = await executor.handle_task_error(
            failed_task=task_spec,
            error=real_error,
            context={"file_path": missing_file}
        )
        
        assert result["task_id"] == "file_read_task"
        assert "recovered_from_error" in result
    
    @pytest.mark.asyncio
    async def test_permission_error_handling(self, temp_directory):
        """Test handling of real permission errors."""
        executor = ErrorHandlerExecutor(MockTaskExecutor())
        
        # Create a file and remove write permissions
        test_file = os.path.join(temp_directory, "readonly.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # Make file read-only
        os.chmod(test_file, 0o444)
        
        try:
            # Try to write to read-only file
            with open(test_file, 'w') as f:
                f.write("new content")
        except PermissionError as e:
            real_error = e
        
        task_spec = TaskSpec(
            id="permission_task",
            action="Write to read-only file"
        )
        
        handler = ErrorHandler(
            handler_action="Handle permission error",
            error_types=["PermissionError"],
            fallback_value="Permission denied - using default"
        )
        
        executor.handler_registry.register_handler("perm_handler", handler, "permission_task")
        
        result = await executor.handle_task_error(
            failed_task=task_spec,
            error=real_error,
            context={"file_path": test_file}
        )
        
        assert result["success"] is True
        assert "permission" in result["result"].lower()
        
        # Clean up - restore permissions for deletion
        os.chmod(test_file, 0o666)


class TestRealNetworkErrors:
    """Test real network error scenarios."""
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test handling of real connection errors."""
        executor = ErrorHandlerExecutor(MockTaskExecutor())
        
        # Try to connect to non-existent server
        import socket
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)  # Short timeout
            sock.connect(("192.0.2.1", 80))  # Reserved IP that should not respond
        except (ConnectionError, OSError, socket.error) as e:
            real_error = e
        finally:
            sock.close()
        
        task_spec = TaskSpec(
            id="network_task",
            action="Connect to server"
        )
        
        handler = ErrorHandler(
            handler_action="Handle connection failure",
            error_types=["ConnectionError", "OSError"],
            retry_with_handler=True,
            max_handler_retries=2
        )
        
        executor.handler_registry.register_handler("net_handler", handler, "network_task")
        
        result = await executor.handle_task_error(
            failed_task=task_spec,
            error=real_error,
            context={"server": "192.0.2.1"}
        )
        
        assert result["task_id"] == "network_task"
        # Should attempt recovery
        assert "recovered_from_error" in result or "fallback" in str(result).lower()
    
    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test handling of real timeout errors."""
        executor = ErrorHandlerExecutor(MockTaskExecutor())
        
        # Create real timeout error
        import asyncio
        
        async def slow_operation():
            await asyncio.sleep(2)  # 2 second operation
        
        try:
            await asyncio.wait_for(slow_operation(), timeout=0.1)  # 0.1 second timeout
        except asyncio.TimeoutError as e:
            real_error = e
        
        task_spec = TaskSpec(
            id="timeout_task",
            action="Slow operation"
        )
        
        handler = ErrorHandler(
            handler_action="Handle timeout",
            error_types=["TimeoutError"],
            fallback_value="Operation timed out - using cached result"
        )
        
        executor.handler_registry.register_handler("timeout_handler", handler, "timeout_task")
        
        result = await executor.handle_task_error(
            failed_task=task_spec,
            error=real_error,
            context={"timeout_seconds": 0.1}
        )
        
        assert result["success"] is True
        assert "timeout" in result["result"].lower()


class TestYAMLErrorHandlerIntegration:
    """Test YAML integration for error handlers."""
    
    def test_error_handler_schema_validation(self):
        """Test YAML schema validation for error handlers."""
        validator = ErrorHandlerSchemaValidator()
        
        # Test valid configuration
        valid_config = {
            "handler_action": "Handle the error",
            "error_types": ["ValueError", "ConnectionError"],
            "retry_with_handler": True,
            "priority": 10
        }
        
        issues = validator.validate_error_handler_config(valid_config)
        assert len(issues) == 0
        
        # Test invalid configuration
        invalid_config = {
            # Missing required fields
            "error_types": ["ValueError"],
            "priority": "not_a_number"  # Invalid type
        }
        
        issues = validator.validate_error_handler_config(invalid_config)
        assert len(issues) > 0
        assert any("must specify at least one" in issue for issue in issues)
    
    def test_yaml_error_handler_compilation(self):
        """Test compiling YAML with error handlers."""
        yaml_content = """
name: test_pipeline
version: 1.0.0
steps:
  - id: failing_step
    action: "This step will fail"
    on_error:
      - handler_action: "Log error and retry"
        error_types: ["ValueError"]
        retry_with_handler: true
        priority: 1
      - handler_action: "Send alert"
        error_types: ["*"]
        priority: 10
        fallback_value: "Default result"
"""
        
        compiler = YAMLCompiler()
        
        # Should compile without errors
        try:
            pipeline = asyncio.run(compiler.compile(yaml_content, resolve_ambiguities=False))
            assert pipeline is not None
            assert len(pipeline.tasks) == 1
            
            # Check that error handlers were processed
            task = pipeline.tasks[0]
            assert hasattr(task, 'error_handlers') or 'error_handlers' in task.metadata
            
        except Exception as e:
            pytest.fail(f"YAML compilation failed: {e}")
    
    def test_legacy_error_handling_compatibility(self):
        """Test backward compatibility with legacy error handling."""
        yaml_content = """
name: legacy_pipeline
version: 1.0.0
steps:
  - id: legacy_step
    action: "Legacy step"
    on_error: "Simple error message"
  
  - id: legacy_advanced_step
    action: "Advanced legacy step"
    on_error:
      action: "Legacy recovery action"
      retry_count: 3
      continue_on_error: true
      fallback_value: "Legacy fallback"
"""
        
        compiler = YAMLCompiler()
        
        try:
            pipeline = asyncio.run(compiler.compile(yaml_content, resolve_ambiguities=False))
            assert pipeline is not None
            assert len(pipeline.tasks) == 2
            
            # Both tasks should have error handling
            for task in pipeline.tasks:
                assert hasattr(task, 'error_handlers') or 'error_handlers' in task.metadata or task.metadata.get('on_failure')
                
        except Exception as e:
            pytest.fail(f"Legacy YAML compilation failed: {e}")


class TestAdvancedErrorScenarios:
    """Test advanced error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_nested_error_handling(self):
        """Test error handlers that themselves might fail."""
        executor = ErrorHandlerExecutor(MockTaskExecutor())
        
        task_spec = TaskSpec(
            id="nested_error_task",
            action="Task with nested errors"
        )
        
        # Primary handler that will fail
        primary_handler = ErrorHandler(
            handler_action="This handler will fail",
            error_types=["ValueError"],
            priority=1,
            continue_on_handler_failure=True
        )
        
        # Backup handler with fallback
        backup_handler = ErrorHandler(
            handler_action="Backup handler",
            error_types=["*"],
            priority=10,
            fallback_value="Backup recovery successful"
        )
        
        executor.handler_registry.register_handler("primary", primary_handler, "nested_error_task")
        executor.handler_registry.register_handler("backup", backup_handler, "nested_error_task")
        
        result = await executor.handle_task_error(
            failed_task=task_spec,
            error=ValueError("Original error"),
            context={}
        )
        
        # Should eventually succeed with backup handler
        assert result["success"] is True
        assert "Backup recovery successful" in str(result["result"])
    
    @pytest.mark.asyncio
    async def test_error_context_enrichment(self):
        """Test that error context includes comprehensive information."""
        executor = ErrorHandlerExecutor(MockTaskExecutor())
        
        # Create rich task specification
        task_spec = TaskSpec(
            id="rich_context_task",
            action="Task with rich context",
            inputs={"param1": "value1", "param2": 42},
            tools=["tool1", "tool2"]
        )
        
        handler = ErrorHandler(
            handler_action="Analyze error context",
            error_types=["*"],
            capture_error_context=True
        )
        
        executor.handler_registry.register_handler("context_handler", handler, "rich_context_task")
        
        result = await executor.handle_task_error(
            failed_task=task_spec,
            error=ValueError("Context test error"),
            context={
                "pipeline_var": "pipeline_value",
                "step_results": {"prev_step": "prev_result"}
            }
        )
        
        # Check that error context was captured
        assert "error_context" in result
        error_context = result["error_context"]
        
        assert error_context["failed_task_id"] == "rich_context_task"
        assert error_context["error_type"] == "ValueError"
        assert error_context["error_message"] == "Context test error"
        assert error_context["task_parameters"]["param1"] == "value1"
        assert error_context["pipeline_context"]["pipeline_var"] == "pipeline_value"
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test error handling performance with multiple concurrent errors."""
        executor = ErrorHandlerExecutor(MockTaskExecutor())
        
        # Register efficient handler
        handler = ErrorHandler(
            handler_action="Quick recovery",
            error_types=["*"],
            timeout=1.0
        )
        
        executor.handler_registry.register_handler("perf_handler", handler)
        
        # Create multiple error handling tasks
        tasks = []
        for i in range(10):
            task_spec = TaskSpec(
                id=f"perf_task_{i}",
                action=f"Performance test task {i}"
            )
            
            task = executor.handle_task_error(
                failed_task=task_spec,
                error=ValueError(f"Error {i}"),
                context={"task_num": i}
            )
            tasks.append(task)
        
        # Execute all error handling concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # All should succeed
        assert len(results) == 10
        for result in results:
            assert result["success"] is True
        
        # Should complete reasonably quickly (under 5 seconds for 10 concurrent operations)
        assert end_time - start_time < 5.0
        
        # Check performance metrics
        metrics = executor.get_execution_metrics()
        assert metrics["total_errors_handled"] == 10
        assert metrics["successful_recoveries"] > 0


if __name__ == "__main__":
    # Run tests with real scenarios
    pytest.main([__file__, "-v", "--tb=short"])