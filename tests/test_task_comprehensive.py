"""Comprehensive unit tests for task module to achieve 100% coverage."""

import time

import pytest

# Import the task module directly to ensure coverage measurement
from src.orchestrator.core.task import Task, TaskStatus


class TestTaskStatus:
    """Test TaskStatus enum."""

    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.SKIPPED.value == "skipped"

    def test_task_status_iteration(self):
        """Test TaskStatus enum iteration."""
        statuses = list(TaskStatus)
        assert len(statuses) == 5
        assert TaskStatus.PENDING in statuses
        assert TaskStatus.COMPLETED in statuses


class TestTask:
    """Test Task class."""

    def test_task_creation_minimal(self):
        """Test minimal Task creation."""
        task = Task(id="test_task", name="Test Task", action="test_action")

        assert task.id == "test_task"
        assert task.name == "Test Task"
        assert task.action == "test_action"
        assert task.status == TaskStatus.PENDING
        assert task.parameters == {}
        assert task.dependencies == []
        assert task.result is None
        assert task.error is None
        assert task.metadata == {}
        assert task.timeout is None
        assert task.max_retries == 3
        assert task.retry_count == 0
        assert isinstance(task.created_at, float)
        assert task.started_at is None
        assert task.completed_at is None

    def test_task_creation_full(self):
        """Test full Task creation with all parameters."""
        task = Task(
            id="full_task",
            name="Full Test Task",
            action="complex_action",
            parameters={"param1": "value1", "param2": 42},
            dependencies=["dep1", "dep2"],
            timeout=300,
            max_retries=5,
            metadata={"group": "test_group", "version": "1.0"},
        )

        assert task.id == "full_task"
        assert task.name == "Full Test Task"
        assert task.action == "complex_action"
        assert task.parameters == {"param1": "value1", "param2": 42}
        assert task.dependencies == ["dep1", "dep2"]
        assert task.timeout == 300
        assert task.max_retries == 5
        assert task.metadata == {"group": "test_group", "version": "1.0"}

    def test_task_validation_empty_id(self):
        """Test validation with empty ID."""
        with pytest.raises(ValueError, match="Task ID cannot be empty"):
            Task(id="", name="Test", action="test")

    def test_task_validation_empty_name(self):
        """Test validation with empty name."""
        with pytest.raises(ValueError, match="Task name cannot be empty"):
            Task(id="test", name="", action="test")

    def test_task_validation_empty_action(self):
        """Test validation with empty action."""
        with pytest.raises(ValueError, match="Task action cannot be empty"):
            Task(id="test", name="Test", action="")

    def test_task_validation_negative_retries(self):
        """Test validation with negative max_retries."""
        with pytest.raises(ValueError, match="Max retries must be non-negative"):
            Task(id="test", name="Test", action="test", max_retries=-1)

    def test_task_validation_self_dependency(self):
        """Test validation with self-dependency."""
        with pytest.raises(ValueError, match="Task test cannot depend on itself"):
            Task(id="test", name="Test", action="test", dependencies=["test"])

    def test_task_is_ready_no_dependencies(self):
        """Test task readiness with no dependencies."""
        task = Task("ready_test", "Test", "test")
        assert task.is_ready(set()) is True
        assert task.is_ready({"other_task"}) is True

    def test_task_is_ready_with_dependencies_satisfied(self):
        """Test task readiness with satisfied dependencies."""
        task = Task("dep_test", "Test", "test", dependencies=["dep1", "dep2"])

        # Not ready when dependencies not completed
        assert task.is_ready(set()) is False
        assert task.is_ready({"dep1"}) is False

        # Ready when all dependencies completed
        assert task.is_ready({"dep1", "dep2"}) is True
        assert task.is_ready({"dep1", "dep2", "extra"}) is True

    def test_task_can_retry_initial_state(self):
        """Test retry capability in initial state."""
        task = Task("retry_test", "Test", "test", max_retries=2)
        assert task.can_retry() is False  # Not failed yet

    def test_task_can_retry_after_failure(self):
        """Test retry capability after failure."""
        task = Task("retry_test", "Test", "test", max_retries=2)
        task.fail(Exception("Test error"))

        assert task.can_retry() is True
        assert task.retry_count == 1

        # Fail again
        task.fail(Exception("Second error"))
        assert task.can_retry() is False  # Exhausted retries
        assert task.retry_count == 2

    def test_task_start(self):
        """Test task start."""
        task = Task("start_test", "Test", "test")

        start_time = time.time()
        task.start()

        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None
        assert task.started_at >= start_time

    def test_task_complete(self):
        """Test task completion."""
        task = Task("complete_test", "Test", "test")
        task.start()

        result = {"output": "success"}
        complete_time = time.time()
        task.complete(result)

        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
        assert task.error is None
        assert task.completed_at is not None
        assert task.completed_at >= complete_time

    def test_task_complete_no_result(self):
        """Test task completion without result."""
        task = Task("complete_no_result", "Test", "test")
        task.start()

        task.complete()

        assert task.status == TaskStatus.COMPLETED
        assert task.result is None
        assert task.error is None

    def test_task_fail(self):
        """Test task failure."""
        task = Task("fail_test", "Test", "test")
        task.start()

        error = Exception("Test failure")
        fail_time = time.time()
        task.fail(error)

        assert task.status == TaskStatus.FAILED
        assert task.error == error
        assert task.completed_at is not None
        assert task.completed_at >= fail_time
        assert task.retry_count == 1

    def test_task_skip(self):
        """Test task skip."""
        task = Task("skip_test", "Test", "test")

        skip_time = time.time()
        task.skip()

        assert task.status == TaskStatus.SKIPPED
        assert task.completed_at is not None
        assert task.completed_at >= skip_time

    def test_task_skip_with_reason(self):
        """Test task skip with reason."""
        task = Task("skip_reason_test", "Test", "test")

        task.skip("Dependency failed")

        assert task.status == TaskStatus.SKIPPED
        assert task.metadata["skip_reason"] == "Dependency failed"

    def test_task_reset(self):
        """Test task reset."""
        task = Task("reset_test", "Test", "test")
        task.start()
        task.complete("done")

        # Task is completed
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "done"
        assert task.started_at is not None
        assert task.completed_at is not None

        # Reset task
        task.reset()

        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None
        assert task.started_at is None
        assert task.completed_at is None

    def test_task_execution_time_not_started(self):
        """Test execution time for task that hasn't started."""
        task = Task("not_started", "Test", "test")
        assert task.execution_time is None

    def test_task_execution_time_running(self):
        """Test execution time for running task."""
        task = Task("running", "Test", "test")
        task.start()

        # Sleep briefly to ensure measurable time
        time.sleep(0.01)

        execution_time = task.execution_time
        assert execution_time is not None
        assert execution_time > 0

    def test_task_execution_time_completed(self):
        """Test execution time for completed task."""
        task = Task("completed", "Test", "test")
        task.start()

        time.sleep(0.01)
        task.complete()

        execution_time = task.execution_time
        assert execution_time is not None
        assert execution_time > 0

    def test_task_is_terminal_pending(self):
        """Test terminal state for pending task."""
        task = Task("pending", "Test", "test")
        assert task.is_terminal is False

    def test_task_is_terminal_running(self):
        """Test terminal state for running task."""
        task = Task("running", "Test", "test")
        task.start()
        assert task.is_terminal is False

    def test_task_is_terminal_completed(self):
        """Test terminal state for completed task."""
        task = Task("completed", "Test", "test")
        task.complete()
        assert task.is_terminal is True

    def test_task_is_terminal_failed(self):
        """Test terminal state for failed task."""
        task = Task("failed", "Test", "test")
        task.fail(Exception("error"))
        assert task.is_terminal is True

    def test_task_is_terminal_skipped(self):
        """Test terminal state for skipped task."""
        task = Task("skipped", "Test", "test")
        task.skip()
        assert task.is_terminal is True

    def test_task_to_dict(self):
        """Test task to dictionary conversion."""
        task = Task(
            id="dict_test",
            name="Dict Test",
            action="convert",
            parameters={"key": "value"},
            dependencies=["dep1"],
            timeout=60,
            max_retries=2,
            metadata={"group": "test"},
        )

        task.start()
        task.complete("result")

        task_dict = task.to_dict()

        assert task_dict["id"] == "dict_test"
        assert task_dict["name"] == "Dict Test"
        assert task_dict["action"] == "convert"
        assert task_dict["parameters"] == {"key": "value"}
        assert task_dict["dependencies"] == ["dep1"]
        assert task_dict["status"] == "completed"
        assert task_dict["result"] == "result"
        assert task_dict["error"] is None
        assert task_dict["metadata"] == {"group": "test"}
        assert task_dict["timeout"] == 60
        assert task_dict["max_retries"] == 2
        assert task_dict["retry_count"] == 0
        assert isinstance(task_dict["created_at"], float)
        assert isinstance(task_dict["started_at"], float)
        assert isinstance(task_dict["completed_at"], float)
        assert isinstance(task_dict["execution_time"], float)

    def test_task_to_dict_with_error(self):
        """Test task to dictionary conversion with error."""
        task = Task("error_test", "Error Test", "test")
        task.start()
        task.fail(Exception("Test error"))

        task_dict = task.to_dict()

        assert task_dict["status"] == "failed"
        assert task_dict["error"] == "Test error"
        assert task_dict["result"] is None

    def test_task_from_dict_basic(self):
        """Test task creation from dictionary."""
        task_dict = {
            "id": "from_dict_test",
            "name": "From Dict Test",
            "action": "create",
            "parameters": {"mode": "test"},
            "dependencies": ["dep1", "dep2"],
            "timeout": 120,
            "max_retries": 5,
            "retry_count": 1,
            "metadata": {"version": "1.0"},
        }

        task = Task.from_dict(task_dict)

        assert task.id == "from_dict_test"
        assert task.name == "From Dict Test"
        assert task.action == "create"
        assert task.parameters == {"mode": "test"}
        assert task.dependencies == ["dep1", "dep2"]
        assert task.timeout == 120
        assert task.max_retries == 5
        assert task.retry_count == 1
        assert task.metadata == {"version": "1.0"}

    def test_task_from_dict_with_status(self):
        """Test task creation from dictionary with status."""
        task_dict = {
            "id": "status_test",
            "name": "Status Test",
            "action": "test",
            "status": "running",
        }

        task = Task.from_dict(task_dict)

        assert task.status == TaskStatus.RUNNING

    def test_task_from_dict_with_error(self):
        """Test task creation from dictionary with error."""
        task_dict = {
            "id": "error_test",
            "name": "Error Test",
            "action": "test",
            "error": "Previous error message",
        }

        task = Task.from_dict(task_dict)

        assert isinstance(task.error, Exception)
        assert str(task.error) == "Previous error message"

    def test_task_from_dict_execution_time_removed(self):
        """Test that execution_time is removed from dict during creation."""
        task_dict = {
            "id": "time_test",
            "name": "Time Test",
            "action": "test",
            "execution_time": 1.5,  # Should be ignored
        }

        task = Task.from_dict(task_dict)

        # execution_time should be computed, not set from dict
        assert task.execution_time is None  # Task hasn't started

    def test_task_repr(self):
        """Test task string representation."""
        task = Task("repr_test", "Repr Test", "test")

        repr_str = repr(task)

        assert "Task(" in repr_str
        assert "id='repr_test'" in repr_str
        assert "name='Repr Test'" in repr_str
        assert "status=pending" in repr_str

    def test_task_repr_different_status(self):
        """Test task string representation with different status."""
        task = Task("repr_test2", "Repr Test 2", "test")
        task.start()

        repr_str = repr(task)
        assert "status=running" in repr_str

    def test_task_equality(self):
        """Test task equality comparison."""
        task1 = Task("same_id", "Task 1", "action1")
        task2 = Task("same_id", "Task 2", "action2")  # Different name/action
        task3 = Task("different_id", "Task 3", "action3")

        # Tasks with same ID are equal
        assert task1 == task2

        # Tasks with different IDs are not equal
        assert task1 != task3
        assert task2 != task3

    def test_task_equality_with_non_task(self):
        """Test task equality with non-task objects."""
        task = Task("test", "Test", "test")

        assert task != "not_a_task"
        assert task != 123
        assert task is not None

    def test_task_hash(self):
        """Test task hash function."""
        task1 = Task("hash_test", "Hash Test 1", "action1")
        task2 = Task("hash_test", "Hash Test 2", "action2")  # Same ID
        task3 = Task("different", "Different", "action3")

        # Tasks with same ID have same hash
        assert hash(task1) == hash(task2)

        # Tasks with different IDs have different hashes
        assert hash(task1) != hash(task3)

        # Can be used in sets
        task_set = {task1, task2, task3}
        assert len(task_set) == 2  # task1 and task2 are considered same

    def test_task_multiple_failures(self):
        """Test task behavior with multiple failures."""
        task = Task("multi_fail", "Multi Fail", "test", max_retries=3)

        # First failure
        task.fail(Exception("First error"))
        assert task.retry_count == 1
        assert task.can_retry() is True

        # Second failure
        task.fail(Exception("Second error"))
        assert task.retry_count == 2
        assert task.can_retry() is True

        # Third failure
        task.fail(Exception("Third error"))
        assert task.retry_count == 3
        assert task.can_retry() is False

    def test_task_zero_max_retries(self):
        """Test task with zero max retries."""
        task = Task("no_retry", "No Retry", "test", max_retries=0)

        task.fail(Exception("Error"))
        assert task.retry_count == 1
        assert task.can_retry() is False

    def test_task_time_properties_consistency(self):
        """Test consistency of time-related properties."""
        task = Task("time_test", "Time Test", "test")

        # Before start
        assert task.started_at is None
        assert task.completed_at is None
        assert task.execution_time is None

        # After start
        start_time = time.time()
        task.start()
        assert task.started_at >= start_time
        assert task.completed_at is None
        assert task.execution_time is not None

        # After completion
        time.sleep(0.01)  # Ensure measurable time difference
        complete_time = time.time()
        task.complete()
        assert task.completed_at >= complete_time
        assert task.completed_at > task.started_at
        assert task.execution_time == task.completed_at - task.started_at

    def test_task_metadata_modification(self):
        """Test task metadata can be modified."""
        task = Task("meta_test", "Meta Test", "test")

        # Initial metadata
        assert task.metadata == {}

        # Add metadata
        task.metadata["stage"] = "preprocessing"
        task.metadata["priority"] = "high"

        assert task.metadata["stage"] == "preprocessing"
        assert task.metadata["priority"] == "high"

        # Modify existing metadata
        task.metadata["stage"] = "processing"
        assert task.metadata["stage"] == "processing"

    def test_task_parameters_modification(self):
        """Test task parameters can be modified."""
        task = Task("param_test", "Param Test", "test")

        # Initial parameters
        assert task.parameters == {}

        # Add parameters
        task.parameters["input_file"] = "data.csv"
        task.parameters["output_format"] = "json"

        assert task.parameters["input_file"] == "data.csv"
        assert task.parameters["output_format"] == "json"

        # Modify existing parameters
        task.parameters["output_format"] = "xml"
        assert task.parameters["output_format"] == "xml"

    def test_task_dependencies_modification(self):
        """Test task dependencies can be modified."""
        task = Task("dep_test", "Dep Test", "test", dependencies=["dep1"])

        # Initial dependencies
        assert task.dependencies == ["dep1"]

        # Add dependencies
        task.dependencies.append("dep2")
        task.dependencies.append("dep3")

        assert "dep2" in task.dependencies
        assert "dep3" in task.dependencies
        assert len(task.dependencies) == 3

        # Remove dependency
        task.dependencies.remove("dep1")
        assert "dep1" not in task.dependencies
        assert len(task.dependencies) == 2
