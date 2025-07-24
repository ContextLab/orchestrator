"""Direct import tests for pipeline module to achieve coverage measurement."""

import time

import pytest

# Direct import to ensure coverage measurement
from src.orchestrator.core.pipeline import (
    CircularDependencyError,
    InvalidDependencyError,
    Pipeline,
)
from src.orchestrator.core.task import Task


class TestCircularDependencyError:
    """Test CircularDependencyError exception."""

    def test_circular_dependency_error_creation(self):
        """Test CircularDependencyError creation."""
        error = CircularDependencyError("Circular dependency detected")
        assert str(error) == "Circular dependency detected"
        assert isinstance(error, Exception)


class TestInvalidDependencyError:
    """Test InvalidDependencyError exception."""

    def test_invalid_dependency_error_creation(self):
        """Test InvalidDependencyError creation."""
        error = InvalidDependencyError("Invalid dependency")
        assert str(error) == "Invalid dependency"
        assert isinstance(error, Exception)


class TestPipeline:
    """Test Pipeline class."""

    def test_pipeline_creation_minimal(self):
        """Test minimal Pipeline creation."""
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")

        assert pipeline.id == "test_pipeline"
        assert pipeline.name == "Test Pipeline"
        assert pipeline.tasks == {}
        assert pipeline.context == {}
        assert pipeline.metadata == {}
        assert isinstance(pipeline.created_at, float)
        assert pipeline.version == "1.0.0"
        assert pipeline.description is None

    def test_pipeline_creation_full(self):
        """Test full Pipeline creation."""
        tasks = {
            "task1": Task("task1", "Task 1", "action1"),
            "task2": Task("task2", "Task 2", "action2"),
        }
        context = {"env": "test", "debug": True}
        metadata = {"author": "test", "version": "1.0"}

        pipeline = Pipeline(
            id="full_pipeline",
            name="Full Test Pipeline",
            tasks=tasks,
            context=context,
            metadata=metadata,
            version="2.0.0",
            description="A comprehensive test pipeline",
        )

        assert pipeline.id == "full_pipeline"
        assert pipeline.name == "Full Test Pipeline"
        assert pipeline.tasks == tasks
        assert pipeline.context == context
        assert pipeline.metadata == metadata
        assert pipeline.version == "2.0.0"
        assert pipeline.description == "A comprehensive test pipeline"

    def test_pipeline_validation_empty_id(self):
        """Test validation with empty ID."""
        with pytest.raises(ValueError, match="Pipeline ID cannot be empty"):
            Pipeline(id="", name="Test Pipeline")

    def test_pipeline_validation_empty_name(self):
        """Test validation with empty name."""
        with pytest.raises(ValueError, match="Pipeline name cannot be empty"):
            Pipeline(id="test_pipeline", name="")

    def test_pipeline_validation_dependencies_valid(self):
        """Test validation with valid dependencies."""
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2", dependencies=["task1"])
        task3 = Task("task3", "Task 3", "action3", dependencies=["task1", "task2"])

        tasks = {"task1": task1, "task2": task2, "task3": task3}

        # Should not raise any exception
        pipeline = Pipeline(id="valid_pipeline", name="Valid Pipeline", tasks=tasks)

        assert len(pipeline.tasks) == 3

    def test_pipeline_validation_invalid_dependency(self):
        """Test validation with invalid dependency."""
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2", dependencies=["nonexistent_task"])

        tasks = {"task1": task1, "task2": task2}

        with pytest.raises(InvalidDependencyError):
            Pipeline(id="invalid_pipeline", name="Invalid Pipeline", tasks=tasks)

    def test_pipeline_validation_circular_dependency_simple(self):
        """Test validation with simple circular dependency."""
        task1 = Task("task1", "Task 1", "action1", dependencies=["task2"])
        task2 = Task("task2", "Task 2", "action2", dependencies=["task1"])

        tasks = {"task1": task1, "task2": task2}

        with pytest.raises(CircularDependencyError):
            Pipeline(id="circular_pipeline", name="Circular Pipeline", tasks=tasks)

    def test_pipeline_validation_circular_dependency_complex(self):
        """Test validation with complex circular dependency."""
        task1 = Task("task1", "Task 1", "action1", dependencies=["task3"])
        task2 = Task("task2", "Task 2", "action2", dependencies=["task1"])
        task3 = Task("task3", "Task 3", "action3", dependencies=["task2"])

        tasks = {"task1": task1, "task2": task2, "task3": task3}

        with pytest.raises(CircularDependencyError):
            Pipeline(
                id="complex_circular_pipeline",
                name="Complex Circular Pipeline",
                tasks=tasks,
            )

    def test_pipeline_add_task(self):
        """Test adding tasks to pipeline."""
        pipeline = Pipeline(id="test", name="Test")

        task1 = Task("task1", "Task 1", "action1")
        pipeline.add_task(task1)

        assert "task1" in pipeline.tasks
        assert pipeline.tasks["task1"] == task1

    def test_pipeline_add_task_with_dependency(self):
        """Test adding task with valid dependency."""
        pipeline = Pipeline(id="test", name="Test")

        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2", dependencies=["task1"])

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        assert len(pipeline.tasks) == 2

    def test_pipeline_add_task_invalid_dependency(self):
        """Test adding task with invalid dependency."""
        pipeline = Pipeline(id="test", name="Test")

        task_with_invalid_dep = Task(
            "task1", "Task 1", "action1", dependencies=["nonexistent"]
        )

        with pytest.raises(InvalidDependencyError):
            pipeline.add_task(task_with_invalid_dep)

    def test_pipeline_add_task_creates_circular_dependency(self):
        """Test adding task that creates circular dependency."""
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2", dependencies=["task1"])

        pipeline = Pipeline(
            id="test", name="Test", tasks={"task1": task1, "task2": task2}
        )

        # Try to add task that creates circular dependency
        task3 = Task("task3", "Task 3", "action3", dependencies=["task2"])
        task1.dependencies.append("task3")  # This will create a circle

        with pytest.raises(CircularDependencyError):
            pipeline.add_task(task3)

    def test_pipeline_remove_task(self):
        """Test removing tasks from pipeline."""
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2")

        pipeline = Pipeline(
            id="test", name="Test", tasks={"task1": task1, "task2": task2}
        )

        removed_task = pipeline.remove_task("task1")

        assert removed_task == task1
        assert "task1" not in pipeline.tasks
        assert "task2" in pipeline.tasks

    def test_pipeline_remove_nonexistent_task(self):
        """Test removing non-existent task."""
        pipeline = Pipeline(id="test", name="Test")

        result = pipeline.remove_task("nonexistent")

        assert result is None

    def test_pipeline_remove_task_with_dependents(self):
        """Test removing task that other tasks depend on."""
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2", dependencies=["task1"])

        pipeline = Pipeline(
            id="test", name="Test", tasks={"task1": task1, "task2": task2}
        )

        with pytest.raises(
            ValueError, match="Cannot remove task .* because it has dependents"
        ):
            pipeline.remove_task("task1")

    def test_pipeline_get_task(self):
        """Test getting task from pipeline."""
        task1 = Task("task1", "Task 1", "action1")

        pipeline = Pipeline(id="test", name="Test", tasks={"task1": task1})

        retrieved_task = pipeline.get_task("task1")

        assert retrieved_task == task1

    def test_pipeline_get_nonexistent_task(self):
        """Test getting non-existent task."""
        pipeline = Pipeline(id="test", name="Test")

        result = pipeline.get_task("nonexistent")

        assert result is None

    def test_pipeline_has_task(self):
        """Test checking if pipeline has task."""
        task1 = Task("task1", "Task 1", "action1")

        pipeline = Pipeline(id="test", name="Test", tasks={"task1": task1})

        assert pipeline.has_task("task1") is True
        assert pipeline.has_task("nonexistent") is False

    def test_pipeline_get_execution_order(self):
        """Test getting execution order of tasks."""
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2", dependencies=["task1"])
        task3 = Task("task3", "Task 3", "action3", dependencies=["task1"])
        task4 = Task("task4", "Task 4", "action4", dependencies=["task2", "task3"])

        pipeline = Pipeline(
            id="test",
            name="Test",
            tasks={"task1": task1, "task2": task2, "task3": task3, "task4": task4},
        )

        execution_order = pipeline.get_execution_order()

        # task1 should be first
        assert execution_order.index("task1") == 0

        # task2 and task3 should come after task1
        assert execution_order.index("task2") > execution_order.index("task1")
        assert execution_order.index("task3") > execution_order.index("task1")

        # task4 should come after both task2 and task3
        assert execution_order.index("task4") > execution_order.index("task2")
        assert execution_order.index("task4") > execution_order.index("task3")

    def test_pipeline_get_execution_order_no_tasks(self):
        """Test getting execution order with no tasks."""
        pipeline = Pipeline(id="test", name="Test")

        execution_order = pipeline.get_execution_order()

        assert execution_order == []

    def test_pipeline_get_ready_tasks(self):
        """Test getting ready tasks."""
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2", dependencies=["task1"])
        task3 = Task("task3", "Task 3", "action3")

        pipeline = Pipeline(
            id="test",
            name="Test",
            tasks={"task1": task1, "task2": task2, "task3": task3},
        )

        # Initially, only tasks without dependencies should be ready
        ready_tasks = pipeline.get_ready_tasks(set())
        ready_task_ids = [task.id for task in ready_tasks]

        assert "task1" in ready_task_ids
        assert "task3" in ready_task_ids
        assert "task2" not in ready_task_ids

        # After task1 completes, task2 should be ready
        ready_tasks = pipeline.get_ready_tasks({"task1"})
        ready_task_ids = [task.id for task in ready_tasks]

        assert "task2" in ready_task_ids
        assert "task3" in ready_task_ids

    def test_pipeline_get_dependencies(self):
        """Test getting task dependencies."""
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2", dependencies=["task1"])

        pipeline = Pipeline(
            id="test", name="Test", tasks={"task1": task1, "task2": task2}
        )

        deps = pipeline.get_dependencies("task2")
        assert deps == ["task1"]

        deps = pipeline.get_dependencies("task1")
        assert deps == []

        deps = pipeline.get_dependencies("nonexistent")
        assert deps == []

    def test_pipeline_get_dependents(self):
        """Test getting task dependents."""
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2", dependencies=["task1"])
        task3 = Task("task3", "Task 3", "action3", dependencies=["task1"])

        pipeline = Pipeline(
            id="test",
            name="Test",
            tasks={"task1": task1, "task2": task2, "task3": task3},
        )

        dependents = pipeline.get_dependents("task1")
        assert set(dependents) == {"task2", "task3"}

        dependents = pipeline.get_dependents("task2")
        assert dependents == []

        dependents = pipeline.get_dependents("nonexistent")
        assert dependents == []

    def test_pipeline_is_valid(self):
        """Test pipeline validation check."""
        # Valid pipeline
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2", dependencies=["task1"])

        pipeline = Pipeline(
            id="test", name="Test", tasks={"task1": task1, "task2": task2}
        )

        assert pipeline.is_valid() is True

    def test_pipeline_is_valid_with_issues(self):
        """Test pipeline validation with issues."""
        # Create pipeline first, then modify to create issues
        pipeline = Pipeline(id="test", name="Test")

        # Add task with invalid dependency directly to bypass validation
        task_with_invalid_dep = Task(
            "task1", "Task 1", "action1", dependencies=["nonexistent"]
        )
        pipeline.tasks["task1"] = task_with_invalid_dep

        assert pipeline.is_valid() is False

    def test_pipeline_get_status(self):
        """Test getting pipeline status."""
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2")
        task1.start()
        task1.complete()
        task2.start()

        pipeline = Pipeline(
            id="test", name="Test", tasks={"task1": task1, "task2": task2}
        )

        status = pipeline.get_status()

        assert status["total_tasks"] == 2
        assert status["completed_tasks"] == 1
        assert status["running_tasks"] == 1
        assert status["failed_tasks"] == 0
        assert status["pending_tasks"] == 0
        assert status["skipped_tasks"] == 0

    def test_pipeline_to_dict(self):
        """Test pipeline to dictionary conversion."""
        task1 = Task("task1", "Task 1", "action1")

        pipeline = Pipeline(
            id="test_pipeline",
            name="Test Pipeline",
            tasks={"task1": task1},
            context={"env": "test"},
            metadata={"version": "1.0"},
            version="2.0.0",
            description="Test description",
        )

        pipeline_dict = pipeline.to_dict()

        assert pipeline_dict["id"] == "test_pipeline"
        assert pipeline_dict["name"] == "Test Pipeline"
        assert pipeline_dict["context"] == {"env": "test"}
        assert pipeline_dict["metadata"] == {"version": "1.0"}
        assert pipeline_dict["version"] == "2.0.0"
        assert pipeline_dict["description"] == "Test description"
        assert "tasks" in pipeline_dict
        assert "task1" in pipeline_dict["tasks"]
        assert isinstance(pipeline_dict["created_at"], float)

    def test_pipeline_from_dict(self):
        """Test pipeline creation from dictionary."""
        pipeline_dict = {
            "id": "from_dict_pipeline",
            "name": "From Dict Pipeline",
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
            "context": {"env": "prod"},
            "metadata": {"author": "test"},
            "version": "1.5.0",
            "description": "Restored pipeline",
            "created_at": time.time(),
        }

        pipeline = Pipeline.from_dict(pipeline_dict)

        assert pipeline.id == "from_dict_pipeline"
        assert pipeline.name == "From Dict Pipeline"
        assert len(pipeline.tasks) == 1
        assert "task1" in pipeline.tasks
        assert pipeline.context == {"env": "prod"}
        assert pipeline.metadata == {"author": "test"}
        assert pipeline.version == "1.5.0"
        assert pipeline.description == "Restored pipeline"

    def test_pipeline_repr(self):
        """Test pipeline string representation."""
        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")

        repr_str = repr(pipeline)

        assert "Pipeline(" in repr_str
        assert "id='test_pipeline'" in repr_str
        assert "name='Test Pipeline'" in repr_str

    def test_pipeline_equality(self):
        """Test pipeline equality comparison."""
        pipeline1 = Pipeline(id="same_id", name="Pipeline 1")
        pipeline2 = Pipeline(id="same_id", name="Pipeline 2")  # Different name
        pipeline3 = Pipeline(id="different_id", name="Pipeline 3")

        # Pipelines with same ID are equal
        assert pipeline1 == pipeline2

        # Pipelines with different IDs are not equal
        assert pipeline1 != pipeline3

        # Should not be equal to non-Pipeline objects
        assert pipeline1 != "not a pipeline"
        assert pipeline1 != 123

    def test_pipeline_hash(self):
        """Test pipeline hash function."""
        pipeline1 = Pipeline(id="same_id", name="Pipeline 1")
        pipeline2 = Pipeline(id="same_id", name="Pipeline 2")
        pipeline3 = Pipeline(id="different_id", name="Pipeline 3")

        # Pipelines with same ID have same hash
        assert hash(pipeline1) == hash(pipeline2)

        # Pipelines with different IDs have different hashes
        assert hash(pipeline1) != hash(pipeline3)

        # Can be used in sets
        pipeline_set = {pipeline1, pipeline2, pipeline3}
        assert len(pipeline_set) == 2  # pipeline1 and pipeline2 are considered same

    def test_pipeline_clear_tasks(self):
        """Test clearing all tasks from pipeline."""
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2")

        pipeline = Pipeline(
            id="test", name="Test", tasks={"task1": task1, "task2": task2}
        )

        pipeline.clear_tasks()

        assert len(pipeline.tasks) == 0

    def test_pipeline_task_count(self):
        """Test getting task count."""
        pipeline = Pipeline(id="test", name="Test")

        assert pipeline.task_count == 0

        task1 = Task("task1", "Task 1", "action1")
        pipeline.add_task(task1)

        assert pipeline.task_count == 1

    def test_pipeline_execution_order_with_parallel_tasks(self):
        """Test execution order with tasks that can run in parallel."""
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2")  # Independent of task1
        task3 = Task("task3", "Task 3", "action3", dependencies=["task1", "task2"])

        pipeline = Pipeline(
            id="test",
            name="Test",
            tasks={"task1": task1, "task2": task2, "task3": task3},
        )

        execution_order = pipeline.get_execution_order()

        # task3 should be last
        assert execution_order.index("task3") == 2

        # task1 and task2 can be in any order but should come before task3
        task1_index = execution_order.index("task1")
        task2_index = execution_order.index("task2")
        task3_index = execution_order.index("task3")

        assert task1_index < task3_index
        assert task2_index < task3_index

    def test_pipeline_remove_task_strict_lines_117_119(self):
        """Test remove_task_strict method (lines 117-119)."""
        task = Task("task1", "Task 1", "action1")

        pipeline = Pipeline(id="test", name="Test", tasks={"task1": task})

        # Test successful removal using strict method
        removed_task = pipeline.remove_task_strict("task1")

        assert removed_task == task
        assert "task1" not in pipeline.tasks
        assert len(pipeline.tasks) == 0

    def test_pipeline_get_task_safe_line_143(self):
        """Test get_task_safe method (line 143)."""
        task = Task("task1", "Task 1", "action1")

        pipeline = Pipeline(id="test", name="Test", tasks={"task1": task})

        # Test getting existing task safely
        retrieved_task = pipeline.get_task_safe("task1")
        assert retrieved_task == task

        # Test getting non-existent task safely
        missing_task = pipeline.get_task_safe("nonexistent")
        assert missing_task is None

    def test_pipeline_get_task_strict_line_160(self):
        """Test get_task_strict method (line 160)."""
        task = Task("task1", "Task 1", "action1")

        pipeline = Pipeline(id="test", name="Test", tasks={"task1": task})

        # Test getting existing task with strict method
        retrieved_task = pipeline.get_task_strict("task1")
        assert retrieved_task == task

    def test_pipeline_get_execution_order_flat_lines_412_413(self):
        """Test get_execution_order_flat method (lines 412-413)."""
        task1 = Task("task1", "Task 1", "action1")
        task2 = Task("task2", "Task 2", "action2", dependencies=["task1"])
        task3 = Task("task3", "Task 3", "action3", dependencies=["task1"])

        pipeline = Pipeline(
            id="test",
            name="Test",
            tasks={"task1": task1, "task2": task2, "task3": task3},
        )

        # Test flat execution order
        flat_order = pipeline.get_execution_order_flat()

        # Should be a flat list, not nested
        assert isinstance(flat_order, list)
        assert len(flat_order) == 3
        assert "task1" in flat_order
        assert "task2" in flat_order
        assert "task3" in flat_order

        # task1 should come before task2 and task3
        task1_index = flat_order.index("task1")
        task2_index = flat_order.index("task2")
        task3_index = flat_order.index("task3")

        assert task1_index < task2_index
        assert task1_index < task3_index
