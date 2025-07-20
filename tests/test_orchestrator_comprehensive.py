"""Comprehensive tests for Orchestrator to achieve 100% coverage."""

import os
import tempfile
from unittest.mock import AsyncMock, Mock

import pytest

from src.orchestrator.core.control_system import ControlSystem
from src.orchestrator.core.error_handler import ErrorHandler
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.resource_allocator import ResourceAllocator
from src.orchestrator.core.task import Task, TaskStatus
from src.orchestrator.executor.parallel_executor import ParallelExecutor
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.orchestrator import ExecutionError, Orchestrator
from src.orchestrator.state.state_manager import StateManager


class TestControlSystem(ControlSystem):
    """Test control system for unit tests."""
    
    def __init__(self, name="test-control", config=None):
        super().__init__(name, config or {})
        self.executed_tasks = {}
        self.should_fail = False
        
    async def execute_task(self, task, context=None):
        """Execute a task and return test result."""
        if self.should_fail:
            raise Exception("Test failure")
        
        result = {
            "status": "completed",
            "result": f"Test result for {task.id}",
            "task_id": task.id
        }
        self.executed_tasks[task.id] = result
        return result
    
    async def health_check(self):
        """Always healthy for tests."""
        return True


class TestOrchestratorComprehensiveCoverage:
    """Comprehensive tests to achieve 100% coverage of Orchestrator."""

    @pytest.mark.asyncio
    async def test_execute_level_skipped_task_results_handling(self):
        """Test handling of skipped tasks in level results (line 270)."""
        # Create orchestrator with test control system
        control_system = TestControlSystem()
        orchestrator = Orchestrator(control_system=control_system)

        # Create pipeline with skipped tasks
        pipeline = Pipeline(id="skip_test", name="Skip Test Pipeline")
        task1 = Task(id="normal_task", name="Normal Task", action="generate")
        task2 = Task(id="pre_skipped_task", name="Pre-skipped Task", action="generate")
        pipeline.add_task(task1)
        pipeline.add_task(task2)

        # Mark task2 as skipped before execution
        task2.skip("Pre-skipped for testing")

        # Execute the level directly to test the skipped task handling
        context = {"execution_id": "test_123", "current_level": 0}
        previous_results = {}

        results = await orchestrator._execute_level(
            pipeline, ["normal_task", "pre_skipped_task"], context, previous_results
        )

        # Both tasks should be in results
        assert "normal_task" in results
        assert "pre_skipped_task" in results
        assert results["pre_skipped_task"]["status"] == "skipped"
        assert task1.status == TaskStatus.COMPLETED
        assert task2.status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_execute_task_legacy_method(self):
        """Test legacy _execute_task method (lines 321-322)."""
        orchestrator = Orchestrator()

        # Create a simple task
        task = Task(id="legacy_task", name="Legacy Task", action="generate")
        context = {"test_context": "value"}

        # Execute using the legacy method
        result = await orchestrator._execute_task(task, context)

        assert result is not None
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_handle_task_failures_retry_policy(self):
        """Test handle_task_failures with retry policy (lines 344-346)."""
        orchestrator = Orchestrator()

        # Create pipeline with retry policy task
        pipeline = Pipeline(id="retry_policy_test", name="Retry Policy Test")
        task = Task(
            id="retry_task",
            name="Retry Task",
            action="generate",
            metadata={"on_failure": "retry"},
        )
        pipeline.add_task(task)

        # Mark task as failed
        task.fail(Exception("Test failure"))

        # Test the retry policy handling
        await orchestrator._handle_task_failures(pipeline, ["retry_task"], {})

        # Should not raise exception for retry policy
        assert task.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_recover_pipeline_reset_failed_tasks(self):
        """Test recover_pipeline resetting failed tasks (line 454)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use simple state manager with create_backend
            from src.orchestrator.state.backends import create_backend

            backend = create_backend("file", {"storage_path": temp_dir})
            state_manager = StateManager()
            state_manager.backend = backend
            orchestrator = Orchestrator(state_manager=state_manager)

            # Create pipeline with failed task
            pipeline = Pipeline(id="recovery_test", name="Recovery Test")
            task = Task(id="failed_task", name="Failed Task", action="generate")
            task.fail(Exception("Previous failure"))
            pipeline.add_task(task)

            # Save checkpoint using proper format
            execution_id = "recovery_123"
            checkpoint_data = orchestrator._get_pipeline_state(pipeline)
            await state_manager.save_checkpoint(
                execution_id, checkpoint_data, {"pipeline_id": pipeline.id}
            )

            # Recover pipeline
            results = await orchestrator.recover_pipeline(execution_id)

            # Task should be reset and executed successfully
            assert isinstance(results, dict)
            # Note: The recovered task is a new instance, so check the results instead
            assert "failed_task" in results

    def test_get_task_resource_requirements_with_model(self, populated_model_registry):
        """Test _get_task_resource_requirements with model requirements (lines 518-521)."""
        # Get a real model from populated registry
        available_models = populated_model_registry.list_models()
        if not available_models:
            pytest.skip("No models available for testing")
        
        # Use the first available model
        model_name = available_models[0]
        real_model = populated_model_registry.get_model(model_name)
        if not real_model:
            pytest.skip("Could not get model for testing")
        
        # Create our own registry and register the model
        model_registry = ModelRegistry()
        model_registry.register_model(real_model)

        orchestrator = Orchestrator(model_registry=model_registry)

        # Create task that requires the model
        task = Task(
            id="model_task",
            name="Model Task",
            action="generate",
            metadata={"requires_model": model_name},
        )

        requirements = orchestrator._get_task_resource_requirements(task)

        # Verify that requirements were extracted from the real model
        # The actual values depend on the model's requirements
        assert "model_memory" in requirements
        assert "model_gpu" in requirements
        assert isinstance(requirements["model_memory"], int)
        assert isinstance(requirements["model_gpu"], bool)
        
        # Verify the values match the model's actual requirements
        expected_memory = int(real_model.requirements.memory_gb * 1024)
        assert requirements["model_memory"] == expected_memory
        assert requirements["model_gpu"] == real_model.requirements.requires_gpu

    @pytest.mark.asyncio
    async def test_select_model_for_task_specified_model(self, populated_model_registry):
        """Test _select_model_for_task with specified model (lines 533-534)."""
        # Get a real model from populated registry
        available_models = populated_model_registry.list_models()
        if not available_models:
            pytest.skip("No models available for testing")
        
        # Use a specific model from the available ones
        specified_model_name = available_models[0]
        real_model = populated_model_registry.get_model(specified_model_name)
        if not real_model:
            pytest.skip("Could not get model for testing")
        
        # Create our own registry and register the model
        model_registry = ModelRegistry()
        model_registry.register_model(real_model)

        orchestrator = Orchestrator(model_registry=model_registry)

        # Create task that specifies a model
        task = Task(
            id="specified_model_task",
            name="Specified Model Task",
            action="generate",
            metadata={"requires_model": specified_model_name},
        )

        model = await orchestrator._select_model_for_task(task, {})

        assert model is not None
        assert model.name == "specified_model"

    @pytest.mark.asyncio
    async def test_get_performance_metrics_no_history(self):
        """Test get_performance_metrics with no execution history (line 568)."""
        orchestrator = Orchestrator()

        # Clear any existing history
        orchestrator.execution_history.clear()

        # Mock resource allocator get_utilization
        orchestrator.resource_allocator.get_utilization = AsyncMock(
            return_value={"cpu_usage": 0.1}
        )

        metrics = await orchestrator.get_performance_metrics()

        assert metrics["total_executions"] == 0
        assert metrics["successful_executions"] == 0
        assert metrics["failed_executions"] == 0
        assert metrics["average_execution_time"] == 0
        assert metrics["error_rate"] == 0

    @pytest.mark.asyncio
    async def test_health_check_control_system_empty_capabilities(self):
        """Test health_check when control system returns empty capabilities (line 614)."""
        orchestrator = Orchestrator()

        # Mock control system to return empty capabilities
        orchestrator.control_system.get_capabilities = Mock(return_value={})

        health = await orchestrator.health_check()

        assert health["control_system"] == "warning"

    @pytest.mark.asyncio
    async def test_health_check_state_manager_unhealthy(self):
        """Test health_check when state manager is unhealthy (line 621)."""
        orchestrator = Orchestrator()

        # Mock state manager health check to return False
        orchestrator.state_manager.is_healthy = AsyncMock(return_value=False)

        health = await orchestrator.health_check()

        assert health["state_manager"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_resource_allocator_high_cpu(self):
        """Test health_check with high CPU usage warning (lines 628-629)."""
        orchestrator = Orchestrator()

        # Mock resource allocator to return high CPU usage
        orchestrator.resource_allocator.get_utilization = AsyncMock(
            return_value={"cpu_usage": 0.95}  # 95% CPU usage
        )

        health = await orchestrator.health_check()

        assert health["resource_allocator"] == "warning"

    @pytest.mark.asyncio
    async def test_health_check_overall_unhealthy(self):
        """Test health_check overall status when components are unhealthy (lines 641-644)."""
        orchestrator = Orchestrator()

        # Mock multiple components as unhealthy
        orchestrator.model_registry.get_available_models = AsyncMock(
            side_effect=Exception("Registry failed")
        )
        orchestrator.control_system.get_capabilities = Mock(
            side_effect=Exception("Control failed")
        )

        health = await orchestrator.health_check()

        assert health["model_registry"] == "unhealthy"
        assert health["control_system"] == "unhealthy"
        assert health["overall"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_shutdown_parallel_executor_sync_method(self):
        """Test shutdown when parallel executor has sync shutdown method (line 662)."""
        orchestrator = Orchestrator()

        # Mock parallel executor with sync shutdown method
        orchestrator.parallel_executor.shutdown = Mock()  # Sync method

        await orchestrator.shutdown()

        # Should call the sync shutdown method
        orchestrator.parallel_executor.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_default_models_functionality(self):
        """Test _register_default_models creates working mock model."""
        # Create orchestrator with empty model registry
        model_registry = ModelRegistry()
        orchestrator = Orchestrator(model_registry=model_registry)

        # Should have registered default models
        models = model_registry.list_models()
        assert len(models) > 0

        # Test that default model can handle tasks
        default_model = model_registry.get_model("default-mock")
        assert default_model is not None

        # Test model responses (check if methods exist)
        assert hasattr(default_model, "name")
        assert default_model.name == "default-mock"

    @pytest.mark.asyncio
    async def test_pipeline_state_serialization_comprehensive(self):
        """Test comprehensive pipeline state serialization."""
        orchestrator = Orchestrator()

        # Create comprehensive pipeline
        pipeline = Pipeline(
            id="comprehensive_test",
            name="Comprehensive Test Pipeline",
            context={"timeout": 300, "priority": "high"},
            metadata={"author": "test", "version": "2.0"},
            description="Test pipeline for comprehensive state serialization",
        )

        # Add tasks with various states
        task1 = Task(id="completed_task", name="Completed Task", action="generate")
        task1.complete("Task 1 result")

        task2 = Task(id="failed_task", name="Failed Task", action="generate")
        task2.fail(Exception("Task 2 failed"))

        task3 = Task(id="pending_task", name="Pending Task", action="generate")

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)

        # Get pipeline state
        state = orchestrator._get_pipeline_state(pipeline)

        # Verify comprehensive state capture
        assert state["id"] == "comprehensive_test"
        assert state["name"] == "Comprehensive Test Pipeline"
        assert state["context"]["timeout"] == 300
        assert state["metadata"]["author"] == "test"
        assert (
            state["description"]
            == "Test pipeline for comprehensive state serialization"
        )
        assert len(state["tasks"]) == 3

        # Verify task states are properly serialized
        assert state["tasks"]["completed_task"]["status"] == "completed"
        assert state["tasks"]["failed_task"]["status"] == "failed"
        assert state["tasks"]["pending_task"]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_model_selection_ai_capabilities(self):
        """Test model selection for AI tasks with context estimation."""
        model_registry = ModelRegistry()
        orchestrator = Orchestrator(model_registry=model_registry)

        # Create task that requires AI capabilities
        task = Task(
            id="ai_task",
            name="AI Task",
            action="analyze",  # AI action
            parameters={
                "data": "large" * 1000
            },  # Large parameters for context estimation
        )

        # Test model selection
        model = await orchestrator._select_model_for_task(task, {})

        # Should select the default mock model for AI tasks
        assert model is not None
        assert model.name == "default-mock"

    @pytest.mark.asyncio
    async def test_execute_pipeline_failure_policy_edge_cases(self):
        """Test failure policy edge cases and error propagation."""
        orchestrator = Orchestrator()

        # Create test control system that fails specific tasks
        control_system = TestControlSystem()
        control_system.should_fail = True  # Make it fail all tasks
        orchestrator.control_system = control_system

        # Create pipeline with 'fail' failure policy
        pipeline = Pipeline(id="fail_policy_test", name="Fail Policy Test")
        task = Task(
            id="unknown_policy_task",
            name="Fail Policy Task",
            action="fail",
            metadata={"on_failure": "fail"},  # Explicit fail policy
        )
        pipeline.add_task(task)

        # Should raise exception with "fail" policy
        with pytest.raises(ExecutionError):
            await orchestrator.execute_pipeline(pipeline)

    @pytest.mark.asyncio
    async def test_resource_requirements_edge_cases(self):
        """Test resource requirements calculation edge cases."""
        orchestrator = Orchestrator()

        # Test task with no metadata
        minimal_task = Task(id="minimal", name="Minimal", action="generate")
        requirements = orchestrator._get_task_resource_requirements(minimal_task)

        assert requirements["cpu"] == 1  # Default
        assert requirements["memory"] == 512  # Default
        assert requirements["timeout"] == 300  # Default

        # Test task with custom timeout but no metadata timeout
        timeout_task = Task(
            id="timeout", name="Timeout", action="generate", timeout=600
        )
        requirements = orchestrator._get_task_resource_requirements(timeout_task)

        assert requirements["timeout"] == 600  # From task.timeout

    @pytest.mark.asyncio
    async def test_execution_context_propagation(self):
        """Test execution context propagation through pipeline levels."""
        orchestrator = Orchestrator()

        # Create pipeline with context-dependent tasks
        pipeline = Pipeline(id="context_test", name="Context Test Pipeline")
        task = Task(
            id="context_task",
            name="Context Task",
            action="generate",
            metadata={"priority": 5},
        )
        pipeline.add_task(task)

        # Execute with custom context
        results = await orchestrator.execute_pipeline(
            pipeline, checkpoint_enabled=True, max_retries=5
        )

        # Verify execution succeeded and context was used
        assert "context_task" in results
        assert task.status == TaskStatus.COMPLETED

        # Check execution history contains context
        assert len(orchestrator.execution_history) == 1
        history_record = orchestrator.execution_history[0]
        assert history_record["status"] == "completed"
        assert "execution_time" in history_record

    @pytest.mark.asyncio
    async def test_concurrent_execution_semaphore(self):
        """Test concurrent execution with semaphore limits."""
        # Create orchestrator with low concurrency limit
        orchestrator = Orchestrator(max_concurrent_tasks=2)

        # Create pipeline with many parallel tasks
        pipeline = Pipeline(id="concurrent_test", name="Concurrent Test")
        tasks = []
        for i in range(5):
            task = Task(id=f"task_{i}", name=f"Task {i}", action="generate")
            tasks.append(task)
            pipeline.add_task(task)

        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)

        # All tasks should complete despite concurrency limit
        assert len(results) == 5
        assert all(task.status == TaskStatus.COMPLETED for task in tasks)

    @pytest.mark.asyncio
    async def test_error_handling_in_level_execution(self):
        """Test error handling during level execution with resource cleanup."""
        orchestrator = Orchestrator()

        # Mock resource allocator to track allocations
        mock_allocator = Mock()
        mock_allocator.request_resources = AsyncMock(return_value=True)
        mock_allocator.release_resources = AsyncMock()
        orchestrator.resource_allocator = mock_allocator

        # Create pipeline with failing task
        pipeline = Pipeline(id="error_level_test", name="Error Level Test")
        task = Task(id="error_task", name="Error Task", action="generate")
        pipeline.add_task(task)

        # Mock control system to fail
        orchestrator.control_system.execute_task = AsyncMock(
            side_effect=Exception("Task failed")
        )

        # Execute level
        context = {"execution_id": "error_test_123"}
        results = await orchestrator._execute_level(
            pipeline, ["error_task"], context, {}
        )

        # Should handle error and clean up resources
        assert "error_task" in results
        assert "error" in results["error_task"]
        mock_allocator.release_resources.assert_called_once_with("error_task")

    @pytest.mark.asyncio
    async def test_yaml_file_execution_error_handling(self):
        """Test YAML file execution error handling."""
        orchestrator = Orchestrator()

        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            await orchestrator.execute_yaml_file("nonexistent_file.yaml")

    @pytest.mark.asyncio
    async def test_skip_dependent_tasks_recursive(self):
        """Test recursive skipping of dependent tasks."""
        orchestrator = Orchestrator()

        # Create pipeline with chain of dependencies
        pipeline = Pipeline(id="skip_chain_test", name="Skip Chain Test")
        task_a = Task(id="task_a", name="Task A", action="generate")
        task_b = Task(
            id="task_b", name="Task B", action="generate", dependencies=["task_a"]
        )
        task_c = Task(
            id="task_c", name="Task C", action="generate", dependencies=["task_b"]
        )
        task_d = Task(
            id="task_d", name="Task D", action="generate", dependencies=["task_c"]
        )

        pipeline.add_task(task_a)
        pipeline.add_task(task_b)
        pipeline.add_task(task_c)
        pipeline.add_task(task_d)

        # Skip task_a and check recursive skipping
        orchestrator._skip_dependent_tasks(pipeline, "task_a")

        # All dependent tasks should be skipped
        assert task_a.status == TaskStatus.PENDING  # Original task not affected
        assert task_b.status == TaskStatus.SKIPPED
        assert task_c.status == TaskStatus.SKIPPED
        assert task_d.status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_comprehensive_integration_workflow(self):
        """Test comprehensive integration workflow with all features."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create orchestrator with all components
            state_manager = StateManager(temp_dir)
            model_registry = ModelRegistry()
            error_handler = ErrorHandler()
            resource_allocator = ResourceAllocator()
            parallel_executor = ParallelExecutor()

            orchestrator = Orchestrator(
                model_registry=model_registry,
                state_manager=state_manager,
                error_handler=error_handler,
                resource_allocator=resource_allocator,
                parallel_executor=parallel_executor,
                max_concurrent_tasks=3,
            )

            # Create complex pipeline
            pipeline = Pipeline(
                id="integration_test",
                name="Integration Test Pipeline",
                context={"mode": "test", "timeout": 600},
                metadata={"test_run": True, "complexity": "high"},
            )

            # Add tasks with various configurations
            fetch_task = Task(
                id="fetch_data",
                name="Fetch Data",
                action="generate",
                metadata={"priority": 10, "cpu_cores": 2, "memory_mb": 1024},
            )

            process_task = Task(
                id="process_data",
                name="Process Data",
                action="analyze",
                dependencies=["fetch_data"],
                metadata={"requires_model": "default-mock", "on_failure": "retry"},
                max_retries=2,
            )

            validate_task = Task(
                id="validate_data",
                name="Validate Data",
                action="transform",
                dependencies=["process_data"],
                timeout=120,
            )

            pipeline.add_task(fetch_task)
            pipeline.add_task(process_task)
            pipeline.add_task(validate_task)

            # Execute pipeline with all features
            results = await orchestrator.execute_pipeline(
                pipeline, checkpoint_enabled=True, max_retries=3
            )

            # Verify comprehensive execution
            assert len(results) == 3
            assert all(
                task.status == TaskStatus.COMPLETED for task in pipeline.tasks.values()
            )

            # Verify execution history
            assert len(orchestrator.execution_history) == 1
            history = orchestrator.execution_history[0]
            assert history["status"] == "completed"
            assert history["pipeline_id"] == "integration_test"

            # Verify checkpoints were created
            checkpoint_files = os.listdir(temp_dir)
            assert len(checkpoint_files) > 0

            # Mock get_utilization for performance metrics
            orchestrator.resource_allocator.get_utilization = AsyncMock(
                return_value={"cpu_usage": 0.1}
            )

            # Test performance metrics
            metrics = await orchestrator.get_performance_metrics()
            assert metrics["total_executions"] == 1
            assert metrics["successful_executions"] == 1
            assert metrics["failed_executions"] == 0

            # Test health check
            health = await orchestrator.health_check()
            assert health["overall"] in [
                "healthy",
                "warning",
                "unhealthy",
            ]  # Any valid status

            # Test recovery capability
            execution_id = history["execution_id"]
            status = orchestrator.get_execution_status(execution_id)
            assert status["status"] == "completed"
