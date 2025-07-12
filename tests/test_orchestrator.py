"""Tests for Orchestrator class."""

import pytest
import tempfile
import os

from orchestrator.orchestrator import Orchestrator, ExecutionError
from orchestrator.core.pipeline import Pipeline
from orchestrator.core.task import Task, TaskStatus
from orchestrator.core.control_system import MockControlSystem
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.core.model import MockModel
from orchestrator.state.state_manager import StateManager
from orchestrator.compiler.yaml_compiler import YAMLCompiler


class TestOrchestrator:
    """Test cases for Orchestrator class."""
    
    def test_orchestrator_creation(self):
        """Test basic orchestrator creation."""
        orchestrator = Orchestrator()
        
        assert orchestrator.model_registry is not None
        assert orchestrator.control_system is not None
        assert orchestrator.state_manager is not None
        assert orchestrator.yaml_compiler is not None
        assert orchestrator.max_concurrent_tasks == 10
        assert orchestrator.running_pipelines == {}
        assert orchestrator.execution_history == []
    
    def test_orchestrator_with_custom_components(self):
        """Test orchestrator with custom components."""
        registry = ModelRegistry()
        control_system = MockControlSystem()
        state_manager = StateManager()
        compiler = YAMLCompiler()
        
        orchestrator = Orchestrator(
            model_registry=registry,
            control_system=control_system,
            state_manager=state_manager,
            yaml_compiler=compiler,
            max_concurrent_tasks=5
        )
        
        assert orchestrator.model_registry is registry
        assert orchestrator.control_system is control_system
        assert orchestrator.state_manager is state_manager
        assert orchestrator.yaml_compiler is compiler
        assert orchestrator.max_concurrent_tasks == 5
    
    @pytest.mark.asyncio
    async def test_execute_simple_pipeline(self):
        """Test executing a simple pipeline."""
        orchestrator = Orchestrator()
        
        # Create a simple pipeline
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task = Task(id="test_task", name="Test Task", action="generate")
        pipeline.add_task(task)
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)
        
        # Check results
        assert isinstance(results, dict)
        assert "test_task" in results
        assert task.status == TaskStatus.COMPLETED
        assert len(orchestrator.execution_history) == 1
        assert orchestrator.execution_history[0]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_with_dependencies(self):
        """Test executing pipeline with task dependencies."""
        orchestrator = Orchestrator()
        
        # Create pipeline with dependencies
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="generate")
        task2 = Task(id="task2", name="Task 2", action="analyze", dependencies=["task1"])
        task3 = Task(id="task3", name="Task 3", action="transform", dependencies=["task2"])
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)
        
        # Check results
        assert len(results) == 3
        assert "task1" in results
        assert "task2" in results
        assert "task3" in results
        
        # Check that all tasks completed
        assert task1.status == TaskStatus.COMPLETED
        assert task2.status == TaskStatus.COMPLETED
        assert task3.status == TaskStatus.COMPLETED
        
        # Check execution order was respected
        assert task1.completed_at < task2.completed_at < task3.completed_at
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_parallel_tasks(self):
        """Test executing pipeline with parallel tasks."""
        orchestrator = Orchestrator()
        
        # Create pipeline with parallel tasks
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="generate")
        task2 = Task(id="task2", name="Task 2", action="generate")  # Parallel to task1
        task3 = Task(id="task3", name="Task 3", action="analyze", dependencies=["task1", "task2"])
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)
        
        # Check results
        assert len(results) == 3
        assert all(task.status == TaskStatus.COMPLETED for task in pipeline.tasks.values())
        
        # Check that task3 ran after task1 and task2
        assert task3.completed_at > task1.completed_at
        assert task3.completed_at > task2.completed_at
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_with_failure(self):
        """Test executing pipeline with task failure."""
        orchestrator = Orchestrator()
        
        # Create control system that fails specific tasks
        control_system = MockControlSystem()
        control_system.set_task_result("failing_task", Exception("Test failure"))
        orchestrator.control_system = control_system
        
        # Create pipeline
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="generate")
        task2 = Task(id="failing_task", name="Failing Task", action="fail")
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        
        # Execute pipeline - should fail
        with pytest.raises(ExecutionError):
            await orchestrator.execute_pipeline(pipeline)
        
        # Check execution history
        assert len(orchestrator.execution_history) == 1
        assert orchestrator.execution_history[0]["status"] == "failed"
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_with_retry(self):
        """Test executing pipeline with task retry."""
        orchestrator = Orchestrator()
        
        # Create pipeline with retryable task
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task = Task(id="test_task", name="Test Task", action="generate", max_retries=2)
        pipeline.add_task(task)
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)
        
        # Check results
        assert "test_task" in results
        assert task.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_with_checkpointing(self):
        """Test executing pipeline with checkpointing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_manager = StateManager(temp_dir)
            orchestrator = Orchestrator(state_manager=state_manager)
            
            # Create pipeline
            pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
            task = Task(id="test_task", name="Test Task", action="generate")
            pipeline.add_task(task)
            
            # Execute pipeline with checkpointing
            results = await orchestrator.execute_pipeline(pipeline, checkpoint_enabled=True)
            
            # Check results
            assert "test_task" in results
            assert task.status == TaskStatus.COMPLETED
            
            # Check that checkpoint was created
            checkpoint_files = os.listdir(temp_dir)
            assert len(checkpoint_files) > 0
    
    @pytest.mark.asyncio
    async def test_execute_yaml_simple(self):
        """Test executing YAML pipeline."""
        orchestrator = Orchestrator()
        
        yaml_content = """
        name: test_pipeline
        version: 1.0.0
        steps:
          - id: step1
            name: Step 1
            action: generate
            parameters:
              prompt: Hello World
        """
        
        results = await orchestrator.execute_yaml(yaml_content)
        
        assert isinstance(results, dict)
        assert "step1" in results
    
    @pytest.mark.asyncio
    async def test_execute_yaml_with_context(self):
        """Test executing YAML pipeline with context."""
        orchestrator = Orchestrator()
        
        yaml_content = """
        name: "{{ pipeline_name }}"
        version: 1.0.0
        steps:
          - id: step1
            name: Step 1
            action: generate
            parameters:
              prompt: "{{ greeting }}"
        """
        
        context = {"pipeline_name": "Templated Pipeline", "greeting": "Hello World"}
        
        results = await orchestrator.execute_yaml(yaml_content, context)
        
        assert isinstance(results, dict)
        assert "step1" in results
    
    @pytest.mark.asyncio
    async def test_execute_yaml_file(self):
        """Test executing YAML from file."""
        orchestrator = Orchestrator()
        
        yaml_content = """
        name: test_pipeline
        version: 1.0.0
        steps:
          - id: step1
            name: Step 1
            action: generate
            parameters:
              prompt: Hello World
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_file = f.name
        
        try:
            results = await orchestrator.execute_yaml_file(yaml_file)
            
            assert isinstance(results, dict)
            assert "step1" in results
        finally:
            os.unlink(yaml_file)
    
    @pytest.mark.asyncio
    async def test_recover_pipeline(self):
        """Test recovering a failed pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_manager = StateManager(temp_dir)
            orchestrator = Orchestrator(state_manager=state_manager)
            
            # Create pipeline
            pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
            task = Task(id="test_task", name="Test Task", action="generate")
            pipeline.add_task(task)
            
            # Save a checkpoint
            execution_id = "test_execution_123"
            await state_manager.save_checkpoint(
                execution_id,
                orchestrator._get_pipeline_state(pipeline),
                {"pipeline_id": pipeline.id}
            )
            
            # Recover pipeline
            results = await orchestrator.recover_pipeline(execution_id)
            
            assert isinstance(results, dict)
    
    @pytest.mark.asyncio
    async def test_recover_pipeline_not_found(self):
        """Test recovering non-existent pipeline."""
        orchestrator = Orchestrator()
        
        with pytest.raises(ExecutionError, match="No checkpoint found"):
            await orchestrator.recover_pipeline("nonexistent_execution")
    
    def test_get_execution_status_running(self):
        """Test getting status of running pipeline."""
        orchestrator = Orchestrator()
        
        # Create pipeline
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task = Task(id="test_task", name="Test Task", action="generate")
        pipeline.add_task(task)
        
        execution_id = "test_execution_123"
        orchestrator.running_pipelines[execution_id] = pipeline
        
        status = orchestrator.get_execution_status(execution_id)
        
        assert status["execution_id"] == execution_id
        assert status["status"] == "running"
        assert "progress" in status
        assert "pipeline" in status
    
    def test_get_execution_status_completed(self):
        """Test getting status of completed pipeline."""
        orchestrator = Orchestrator()
        
        # Add to execution history
        execution_record = {
            "execution_id": "test_execution_123",
            "pipeline_id": "test_pipeline",
            "status": "completed",
            "results": {"task1": "result1"},
            "execution_time": 1.5,
            "completed_at": 1234567890.0
        }
        orchestrator.execution_history.append(execution_record)
        
        status = orchestrator.get_execution_status("test_execution_123")
        
        assert status["execution_id"] == "test_execution_123"
        assert status["status"] == "completed"
        assert status["results"] == {"task1": "result1"}
    
    def test_get_execution_status_not_found(self):
        """Test getting status of non-existent execution."""
        orchestrator = Orchestrator()
        
        status = orchestrator.get_execution_status("nonexistent")
        
        assert status["execution_id"] == "nonexistent"
        assert status["status"] == "not_found"
    
    def test_list_running_pipelines(self):
        """Test listing running pipelines."""
        orchestrator = Orchestrator()
        
        # Add running pipelines
        pipeline1 = Pipeline(id="pipeline1", name="Pipeline 1")
        pipeline2 = Pipeline(id="pipeline2", name="Pipeline 2")
        
        orchestrator.running_pipelines["exec1"] = pipeline1
        orchestrator.running_pipelines["exec2"] = pipeline2
        
        running = orchestrator.list_running_pipelines()
        
        assert len(running) == 2
        assert "exec1" in running
        assert "exec2" in running
    
    def test_get_execution_history(self):
        """Test getting execution history."""
        orchestrator = Orchestrator()
        
        # Add execution records
        records = [
            {
                "execution_id": "exec1",
                "status": "completed",
                "execution_time": 1.0
            },
            {
                "execution_id": "exec2", 
                "status": "failed",
                "execution_time": 0.5
            }
        ]
        orchestrator.execution_history.extend(records)
        
        history = orchestrator.get_execution_history()
        
        assert len(history) == 2
        assert history[0]["execution_id"] == "exec1"
        assert history[1]["execution_id"] == "exec2"
    
    def test_get_execution_history_with_limit(self):
        """Test getting execution history with limit."""
        orchestrator = Orchestrator()
        
        # Add many execution records
        for i in range(10):
            orchestrator.execution_history.append({
                "execution_id": f"exec{i}",
                "status": "completed"
            })
        
        history = orchestrator.get_execution_history(limit=5)
        
        assert len(history) == 5
        # Should get the last 5 records
        assert history[0]["execution_id"] == "exec5"
        assert history[4]["execution_id"] == "exec9"
    
    def test_clear_execution_history(self):
        """Test clearing execution history."""
        orchestrator = Orchestrator()
        
        # Add some records
        orchestrator.execution_history.append({
            "execution_id": "exec1",
            "status": "completed"
        })
        
        orchestrator.clear_execution_history()
        
        assert len(orchestrator.execution_history) == 0
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test shutting down orchestrator."""
        orchestrator = Orchestrator()
        
        # Add some running pipelines
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        orchestrator.running_pipelines["exec1"] = pipeline
        
        # Add execution history
        orchestrator.execution_history.append({
            "execution_id": "exec1",
            "status": "completed"
        })
        
        await orchestrator.shutdown()
        
        # Check cleanup
        assert len(orchestrator.running_pipelines) == 0
        assert len(orchestrator.execution_history) == 0
    
    def test_get_pipeline_state(self):
        """Test getting pipeline state."""
        orchestrator = Orchestrator()
        
        # Create pipeline
        pipeline = Pipeline(
            id="test_pipeline",
            name="Test Pipeline",
            context={"key": "value"},
            metadata={"author": "test"}
        )
        task = Task(id="test_task", name="Test Task", action="generate")
        pipeline.add_task(task)
        
        state = orchestrator._get_pipeline_state(pipeline)
        
        assert state["id"] == "test_pipeline"
        assert "tasks" in state
        assert "test_task" in state["tasks"]
        assert state["context"] == {"key": "value"}
        assert state["metadata"] == {"author": "test"}
    
    def test_repr(self):
        """Test string representation of orchestrator."""
        orchestrator = Orchestrator()
        
        # Add a running pipeline
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        orchestrator.running_pipelines["exec1"] = pipeline
        
        repr_str = repr(orchestrator)
        
        assert "Orchestrator" in repr_str
        assert "running_pipelines=1" in repr_str
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_task_failure_policies(self):
        """Test different task failure policies."""
        orchestrator = Orchestrator()
        
        # Test "continue" policy
        control_system = MockControlSystem()
        control_system.set_task_result("failing_task", Exception("Test failure"))
        orchestrator.control_system = control_system
        
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="generate")
        task2 = Task(
            id="failing_task",
            name="Failing Task", 
            action="fail",
            metadata={"on_failure": "continue"}
        )
        task3 = Task(id="task3", name="Task 3", action="generate")
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        
        # This should not raise an exception with continue policy
        results = await orchestrator.execute_pipeline(pipeline, checkpoint_enabled=True)
        
        # Task 1 and 3 should complete, task 2 should fail
        assert task1.status == TaskStatus.COMPLETED
        assert task2.status == TaskStatus.FAILED
        assert task3.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_skip_dependent_tasks(self):
        """Test skipping dependent tasks when task fails."""
        orchestrator = Orchestrator()
        
        # Create control system that fails specific tasks
        control_system = MockControlSystem()
        control_system.set_task_result("failing_task", Exception("Test failure"))
        orchestrator.control_system = control_system
        
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(
            id="failing_task",
            name="Failing Task",
            action="fail",
            metadata={"on_failure": "skip"}
        )
        task2 = Task(id="task2", name="Task 2", action="generate", dependencies=["failing_task"])
        task3 = Task(id="task3", name="Task 3", action="generate", dependencies=["task2"])
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline, checkpoint_enabled=True)
        
        # Task 1 should fail, tasks 2 and 3 should be skipped
        assert task1.status == TaskStatus.FAILED
        assert task2.status == TaskStatus.SKIPPED
        assert task3.status == TaskStatus.SKIPPED
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_complex_workflow(self):
        """Test executing complex workflow with multiple patterns."""
        orchestrator = Orchestrator()
        
        # Create complex pipeline
        pipeline = Pipeline(
            id="complex_pipeline",
            name="Complex Pipeline",
            context={"timeout": 300},
            metadata={"author": "test"}
        )
        
        # Initial parallel tasks
        task1 = Task(id="data_fetch", name="Data Fetch", action="fetch")
        task2 = Task(id="config_load", name="Config Load", action="load")
        
        # Processing tasks
        task3 = Task(
            id="data_process",
            name="Data Process",
            action="process",
            dependencies=["data_fetch", "config_load"]
        )
        task4 = Task(
            id="data_validate",
            name="Data Validate",
            action="validate",
            dependencies=["data_process"]
        )
        
        # Parallel analysis tasks
        task5 = Task(
            id="analysis_a",
            name="Analysis A",
            action="analyze",
            dependencies=["data_validate"]
        )
        task6 = Task(
            id="analysis_b",
            name="Analysis B",
            action="analyze",
            dependencies=["data_validate"]
        )
        
        # Final aggregation
        task7 = Task(
            id="aggregate",
            name="Aggregate Results",
            action="aggregate",
            dependencies=["analysis_a", "analysis_b"]
        )
        
        for task in [task1, task2, task3, task4, task5, task6, task7]:
            pipeline.add_task(task)
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)
        
        # Check all tasks completed
        assert len(results) == 7
        assert all(task.status == TaskStatus.COMPLETED for task in pipeline.tasks.values())
        
        # Check execution order constraints
        assert task1.completed_at < task3.completed_at
        assert task2.completed_at < task3.completed_at
        assert task3.completed_at < task4.completed_at
        assert task4.completed_at < task5.completed_at
        assert task4.completed_at < task6.completed_at
        assert task5.completed_at < task7.completed_at
        assert task6.completed_at < task7.completed_at
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_empty(self):
        """Test executing empty pipeline."""
        orchestrator = Orchestrator()
        
        pipeline = Pipeline(id="empty_pipeline", name="Empty Pipeline")
        
        results = await orchestrator.execute_pipeline(pipeline)
        
        assert isinstance(results, dict)
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_single_task(self):
        """Test executing pipeline with single task."""
        orchestrator = Orchestrator()
        
        pipeline = Pipeline(id="single_pipeline", name="Single Pipeline")
        task = Task(id="single_task", name="Single Task", action="generate")
        pipeline.add_task(task)
        
        results = await orchestrator.execute_pipeline(pipeline)
        
        assert len(results) == 1
        assert "single_task" in results
        assert task.status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_with_timeout(self):
        """Test executing pipeline with task timeout."""
        orchestrator = Orchestrator()
        
        pipeline = Pipeline(id="timeout_pipeline", name="Timeout Pipeline")
        task = Task(
            id="timeout_task",
            name="Timeout Task",
            action="generate",
            timeout=1  # 1 second timeout
        )
        pipeline.add_task(task)
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)
        
        # Task should complete normally (MockControlSystem is fast)
        assert task.status == TaskStatus.COMPLETED
        assert "timeout_task" in results
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_max_retries(self):
        """Test pipeline execution with max retries."""
        orchestrator = Orchestrator()
        
        pipeline = Pipeline(id="retry_pipeline", name="Retry Pipeline")
        task = Task(
            id="retry_task",
            name="Retry Task",
            action="generate",
            max_retries=2
        )
        pipeline.add_task(task)
        
        # Execute pipeline
        results = await orchestrator.execute_pipeline(pipeline)
        
        # Task should complete
        assert task.status == TaskStatus.COMPLETED
        assert "retry_task" in results
        assert task.retry_count == 0  # No retries needed with MockControlSystem