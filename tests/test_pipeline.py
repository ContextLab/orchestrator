"""Tests for Pipeline class."""

import time

import pytest

from src.orchestrator.core.pipeline import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    CircularDependencyError,
    InvalidDependencyError,
    Pipeline)
from src.orchestrator.core.task import Task, TaskStatus


class TestPipeline:
    """Test cases for Pipeline class."""

    def test_pipeline_creation(self):
        """Test basic pipeline creation."""
        pipeline = Pipeline(
            id="test_pipeline",
            name="Test Pipeline",
            context={"key": "value"},
            metadata={"meta": "data"},
            description="Test description",
            version="1.0.0")

        assert pipeline.id == "test_pipeline"
        assert pipeline.name == "Test Pipeline"
        assert pipeline.context == {"key": "value"}
        assert pipeline.metadata == {"meta": "data"}
        assert pipeline.description == "Test description"
        assert pipeline.version == "1.0.0"
        assert len(pipeline.tasks) == 0
        assert pipeline.created_at > 0

    def test_pipeline_validation_empty_id(self):
        """Test pipeline validation with empty ID."""
        with pytest.raises(ValueError, match="Pipeline ID cannot be empty"):
            Pipeline(id="", name="Test")

    def test_pipeline_validation_empty_name(self):
        """Test pipeline validation with empty name."""
        with pytest.raises(ValueError, match="Pipeline name cannot be empty"):
            Pipeline(id="test", name="")

    def test_add_task(self):
        """Test adding a task to pipeline."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task = Task(id="task1", name="Task 1", action="action1")

        pipeline.add_task(task)

        assert len(pipeline.tasks) == 1
        assert "task1" in pipeline.tasks
        assert pipeline.tasks["task1"] == task

    def test_add_task_duplicate_id(self):
        """Test adding task with duplicate ID."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task1", name="Task 2", action="action2")

        pipeline.add_task(task1)

        with pytest.raises(ValueError, match="already exists"):
            pipeline.add_task(task2)

    def test_remove_task(self):
        """Test removing a task from pipeline."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task = Task(id="task1", name="Task 1", action="action1")

        pipeline.add_task(task)
        pipeline.remove_task("task1")

        assert len(pipeline.tasks) == 0
        assert "task1" not in pipeline.tasks

    def test_remove_task_not_exists(self):
        """Test removing non-existent task."""
        pipeline = Pipeline(id="test", name="Test Pipeline")

        with pytest.raises(ValueError, match="does not exist"):
            pipeline.remove_task_strict("nonexistent")

    def test_remove_task_with_dependents(self):
        """Test removing task that other tasks depend on."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(
            id="task2", name="Task 2", action="action2", dependencies=["task1"]
        )

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        with pytest.raises(ValueError, match="tasks .* depend on it"):
            pipeline.remove_task_strict("task1")

    def test_get_task(self):
        """Test getting a task by ID."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task = Task(id="task1", name="Task 1", action="action1")

        pipeline.add_task(task)
        retrieved_task = pipeline.get_task("task1")

        assert retrieved_task == task

    def test_get_task_not_exists(self):
        """Test getting non-existent task."""
        pipeline = Pipeline(id="test", name="Test Pipeline")

        with pytest.raises(ValueError, match="does not exist"):
            pipeline.get_task_strict("nonexistent")

    def test_invalid_dependency(self):
        """Test validation of invalid dependencies."""
        pipeline = Pipeline(id="test", name="Test Pipeline")

        with pytest.raises(InvalidDependencyError):
            task = Task(
                id="task1",
                name="Task 1",
                action="action1",
                dependencies=["nonexistent"])
            pipeline.add_task(task)

    def test_circular_dependency_simple(self):
        """Test detection of simple circular dependency."""
        pipeline = Pipeline(id="test", name="Test Pipeline")

        # Add tasks without circular dependencies first
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2")

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        # Now create circular dependencies
        task1.dependencies = ["task2"]
        task2.dependencies = ["task1"]

        with pytest.raises(CircularDependencyError):
            pipeline._validate_dependencies()

    def test_circular_dependency_complex(self):
        """Test detection of complex circular dependency."""
        pipeline = Pipeline(id="test", name="Test Pipeline")

        # Add tasks without circular dependencies first
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2")
        task3 = Task(id="task3", name="Task 3", action="action3")

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)

        # Now create complex circular dependencies: task1 -> task3 -> task2 -> task1
        task1.dependencies = ["task3"]
        task2.dependencies = ["task1"]
        task3.dependencies = ["task2"]

        with pytest.raises(CircularDependencyError):
            pipeline._validate_dependencies()

    def test_get_execution_order_simple(self):
        """Test execution order for simple pipeline."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(
            id="task2", name="Task 2", action="action2", dependencies=["task1"]
        )
        task3 = Task(
            id="task3", name="Task 3", action="action3", dependencies=["task1"]
        )

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)

        execution_order = pipeline.get_execution_levels()

        assert len(execution_order) == 2
        assert execution_order[0] == ["task1"]
        assert set(execution_order[1]) == {"task2", "task3"}

    def test_get_execution_order_complex(self):
        """Test execution order for complex pipeline."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(
            id="task2", name="Task 2", action="action2", dependencies=["task1"]
        )
        task3 = Task(
            id="task3", name="Task 3", action="action3", dependencies=["task1"]
        )
        task4 = Task(
            id="task4", name="Task 4", action="action4", dependencies=["task2", "task3"]
        )

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        pipeline.add_task(task4)

        execution_order = pipeline.get_execution_levels()

        assert len(execution_order) == 3
        assert execution_order[0] == ["task1"]
        assert set(execution_order[1]) == {"task2", "task3"}
        assert execution_order[2] == ["task4"]

    def test_get_ready_tasks(self):
        """Test getting ready tasks."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(
            id="task2", name="Task 2", action="action2", dependencies=["task1"]
        )
        task3 = Task(id="task3", name="Task 3", action="action3")

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)

        # Initially, tasks with no dependencies are ready
        ready_tasks = pipeline.get_ready_task_ids(set())
        assert set(ready_tasks) == {"task1", "task3"}

        # Complete task1 and mark task3 as running to test different statuses
        task1.complete("result")
        task3.start()

        # After task1 completes, task2 becomes ready (task3 is running, so not returned)
        ready_tasks = pipeline.get_ready_task_ids({"task1"})
        assert set(ready_tasks) == {"task2"}

    def test_get_failed_tasks(self):
        """Test getting failed tasks."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2")

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        # Initially no failed tasks
        assert pipeline.get_failed_tasks() == []

        # After task1 fails
        task1.fail(Exception("Test error"))
        assert pipeline.get_failed_tasks() == ["task1"]

    def test_get_completed_tasks(self):
        """Test getting completed tasks."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2")

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        # Initially no completed tasks
        assert pipeline.get_completed_tasks() == []

        # After task1 completes
        task1.complete("result")
        assert pipeline.get_completed_tasks() == ["task1"]

    def test_get_running_tasks(self):
        """Test getting running tasks."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2")

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        # Initially no running tasks
        assert pipeline.get_running_tasks() == []

        # After task1 starts
        task1.start()
        assert pipeline.get_running_tasks() == ["task1"]

    def test_reset(self):
        """Test resetting pipeline."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2")

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        # Complete tasks
        task1.complete("result1")
        task2.fail(Exception("error"))

        # Reset pipeline
        pipeline.reset()

        # All tasks should be pending
        assert all(
            task.status == TaskStatus.PENDING for task in pipeline.tasks.values()
        )

    def test_is_complete(self):
        """Test checking if pipeline is complete."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2")

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        # Initially not complete
        assert not pipeline.is_complete()

        # After one task completes
        task1.complete("result")
        assert not pipeline.is_complete()

        # After all tasks complete
        task2.complete("result")
        assert pipeline.is_complete()

    def test_is_failed(self):
        """Test checking if pipeline has failed."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2")

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        # Initially not failed
        assert not pipeline.is_failed()

        # After one task fails
        task1.fail(Exception("error"))
        assert pipeline.is_failed()

    def test_get_progress(self):
        """Test getting pipeline progress."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2")
        task3 = Task(id="task3", name="Task 3", action="action3")

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)

        # Initial progress
        progress = pipeline.get_progress()
        assert progress["total"] == 3
        assert progress["pending"] == 3
        assert progress["running"] == 0
        assert progress["completed"] == 0
        assert progress["failed"] == 0
        assert progress["skipped"] == 0

        # After some tasks change status
        task1.complete("result")
        task2.start()
        task3.fail(Exception("error"))

        progress = pipeline.get_progress()
        assert progress["total"] == 3
        assert progress["pending"] == 0
        assert progress["running"] == 1
        assert progress["completed"] == 1
        assert progress["failed"] == 1
        assert progress["skipped"] == 0

    def test_get_critical_path(self):
        """Test getting critical path."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(
            id="task2", name="Task 2", action="action2", dependencies=["task1"]
        )
        task3 = Task(
            id="task3", name="Task 3", action="action3", dependencies=["task2"]
        )
        task4 = Task(
            id="task4", name="Task 4", action="action4", dependencies=["task1"]
        )

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        pipeline.add_task(task4)

        critical_path = pipeline.get_critical_path()

        # The critical path should be the longest: task1 -> task2 -> task3
        assert len(critical_path) == 3
        assert critical_path == ["task1", "task2", "task3"]

    def test_to_dict(self):
        """Test pipeline serialization to dict."""
        pipeline = Pipeline(
            id="test",
            name="Test Pipeline",
            context={"key": "value"},
            metadata={"meta": "data"},
            description="Test description",
            version="1.0.0")

        task = Task(id="task1", name="Task 1", action="action1")
        pipeline.add_task(task)

        pipeline_dict = pipeline.to_dict()

        assert pipeline_dict["id"] == "test"
        assert pipeline_dict["name"] == "Test Pipeline"
        assert pipeline_dict["context"] == {"key": "value"}
        assert pipeline_dict["metadata"] == {"meta": "data"}
        assert pipeline_dict["description"] == "Test description"
        assert pipeline_dict["version"] == "1.0.0"
        assert "tasks" in pipeline_dict
        assert "task1" in pipeline_dict["tasks"]
        assert "created_at" in pipeline_dict

    def test_from_dict(self):
        """Test pipeline deserialization from dict."""
        pipeline_dict = {
            "id": "test",
            "name": "Test Pipeline",
            "context": {"key": "value"},
            "metadata": {"meta": "data"},
            "description": "Test description",
            "version": "1.0.0",
            "created_at": time.time(),
            "tasks": {
                "task1": {
                    "id": "task1",
                    "name": "Task 1",
                    "action": "action1",
                    "parameters": {},
                    "dependencies": [],
                    "status": "pending",
                    "result": None,
                    "error": None,
                    "metadata": {},
                    "timeout": None,
                    "max_retries": 3,
                    "retry_count": 0,
                    "created_at": time.time(),
                    "started_at": None,
                    "completed_at": None,
                }
            },
        }

        pipeline = Pipeline.from_dict(pipeline_dict)

        assert pipeline.id == "test"
        assert pipeline.name == "Test Pipeline"
        assert pipeline.context == {"key": "value"}
        assert pipeline.metadata == {"meta": "data"}
        assert pipeline.description == "Test description"
        assert pipeline.version == "1.0.0"
        assert len(pipeline.tasks) == 1
        assert "task1" in pipeline.tasks
        assert pipeline.tasks["task1"].id == "task1"

    def test_repr(self):
        """Test pipeline string representation."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task = Task(id="task1", name="Task 1", action="action1")
        pipeline.add_task(task)

        repr_str = repr(pipeline)

        assert "test" in repr_str
        assert "Test Pipeline" in repr_str
        assert "tasks=1" in repr_str

    def test_len(self):
        """Test pipeline length."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        assert len(pipeline) == 0

        task = Task(id="task1", name="Task 1", action="action1")
        pipeline.add_task(task)
        assert len(pipeline) == 1

    def test_contains(self):
        """Test pipeline contains operator."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task = Task(id="task1", name="Task 1", action="action1")
        pipeline.add_task(task)

        assert "task1" in pipeline
        assert "task2" not in pipeline

    def test_iter(self):
        """Test pipeline iteration."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2")

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        task_ids = list(pipeline)
        assert set(task_ids) == {"task1", "task2"}

    def test_empty_pipeline_execution_order(self):
        """Test execution order for empty pipeline."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        execution_order = pipeline.get_execution_order()
        assert execution_order == []

    def test_single_task_execution_order(self):
        """Test execution order for single task."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task = Task(id="task1", name="Task 1", action="action1")
        pipeline.add_task(task)

        execution_order = pipeline.get_execution_levels()
        assert execution_order == [["task1"]]

    def test_no_dependencies_execution_order(self):
        """Test execution order for tasks with no dependencies."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2")
        task3 = Task(id="task3", name="Task 3", action="action3")

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)

        execution_order = pipeline.get_execution_levels()
        assert len(execution_order) == 1
        assert set(execution_order[0]) == {"task1", "task2", "task3"}
