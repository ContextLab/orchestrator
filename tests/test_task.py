"""Tests for Task class."""

import time

import pytest

from orchestrator.core.task import Task, TaskStatus


class TestTask:
    """Test cases for Task class."""

    def test_task_creation(self):
        """Test basic task creation."""
        task = Task(
            id="test_task",
            name="Test Task",
            action="test_action",
            parameters={"param1": "value1"},
            dependencies=["dep1", "dep2"],
        )

        assert task.id == "test_task"
        assert task.name == "Test Task"
        assert task.action == "test_action"
        assert task.parameters == {"param1": "value1"}
        assert task.dependencies == ["dep1", "dep2"]
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None
        assert task.retry_count == 0
        assert task.max_retries == 3
        assert task.created_at > 0
        assert task.started_at is None
        assert task.completed_at is None

    def test_task_validation_empty_id(self):
        """Test task validation with empty ID."""
        with pytest.raises(ValueError, match="Task ID cannot be empty"):
            Task(id="", name="Test", action="test")

    def test_task_validation_empty_name(self):
        """Test task validation with empty name."""
        with pytest.raises(ValueError, match="Task name cannot be empty"):
            Task(id="test", name="", action="test")

    def test_task_validation_empty_action(self):
        """Test task validation with empty action."""
        with pytest.raises(ValueError, match="Task action cannot be empty"):
            Task(id="test", name="Test", action="")

    def test_task_validation_negative_retries(self):
        """Test task validation with negative max retries."""
        with pytest.raises(ValueError, match="Max retries must be non-negative"):
            Task(id="test", name="Test", action="test", max_retries=-1)

    def test_task_validation_self_dependency(self):
        """Test task validation with self dependency."""
        with pytest.raises(ValueError, match="cannot depend on itself"):
            Task(id="test", name="Test", action="test", dependencies=["test"])

    def test_is_ready_no_dependencies(self):
        """Test is_ready with no dependencies."""
        task = Task(id="test", name="Test", action="test")
        assert task.is_ready(set())
        assert task.is_ready({"other_task"})

    def test_is_ready_with_dependencies(self):
        """Test is_ready with dependencies."""
        task = Task(id="test", name="Test", action="test", dependencies=["dep1", "dep2"])

        assert not task.is_ready(set())
        assert not task.is_ready({"dep1"})
        assert task.is_ready({"dep1", "dep2"})
        assert task.is_ready({"dep1", "dep2", "extra"})

    def test_can_retry_initial(self):
        """Test can_retry on initial task."""
        task = Task(id="test", name="Test", action="test")
        assert not task.can_retry()  # Not failed yet

    def test_can_retry_after_failure(self):
        """Test can_retry after failure."""
        task = Task(id="test", name="Test", action="test", max_retries=2)

        # First failure
        task.fail(Exception("Test error"))
        assert task.can_retry()
        assert task.retry_count == 1

        # Second failure
        task.fail(Exception("Test error"))
        assert not task.can_retry()
        assert task.retry_count == 2

    def test_start_task(self):
        """Test starting a task."""
        task = Task(id="test", name="Test", action="test")
        start_time = time.time()

        task.start()

        assert task.status == TaskStatus.RUNNING
        assert task.started_at >= start_time
        assert task.completed_at is None

    def test_complete_task(self):
        """Test completing a task."""
        task = Task(id="test", name="Test", action="test")
        task.start()
        complete_time = time.time()

        result = {"output": "test result"}
        task.complete(result)

        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
        assert task.completed_at >= complete_time
        assert task.error is None

    def test_fail_task(self):
        """Test failing a task."""
        task = Task(id="test", name="Test", action="test")
        task.start()
        fail_time = time.time()

        error = Exception("Test error")
        task.fail(error)

        assert task.status == TaskStatus.FAILED
        assert task.error == error
        assert task.completed_at >= fail_time
        assert task.retry_count == 1

    def test_skip_task(self):
        """Test skipping a task."""
        task = Task(id="test", name="Test", action="test")
        skip_time = time.time()

        task.skip("Not needed")

        assert task.status == TaskStatus.SKIPPED
        assert task.completed_at >= skip_time
        assert task.metadata["skip_reason"] == "Not needed"

    def test_reset_task(self):
        """Test resetting a task."""
        task = Task(id="test", name="Test", action="test")
        task.start()
        task.complete("result")

        task.reset()

        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None
        assert task.started_at is None
        assert task.completed_at is None

    def test_execution_time_not_started(self):
        """Test execution time when task not started."""
        task = Task(id="test", name="Test", action="test")
        assert task.execution_time is None

    def test_execution_time_running(self):
        """Test execution time when task is running."""
        task = Task(id="test", name="Test", action="test")
        task.start()
        time.sleep(0.01)  # Small delay

        execution_time = task.execution_time
        assert execution_time is not None
        assert execution_time > 0

    def test_execution_time_completed(self):
        """Test execution time when task is completed."""
        task = Task(id="test", name="Test", action="test")
        task.start()
        time.sleep(0.01)  # Small delay
        task.complete("result")

        execution_time = task.execution_time
        assert execution_time is not None
        assert execution_time > 0

    def test_is_terminal_pending(self):
        """Test is_terminal for pending task."""
        task = Task(id="test", name="Test", action="test")
        assert not task.is_terminal

    def test_is_terminal_running(self):
        """Test is_terminal for running task."""
        task = Task(id="test", name="Test", action="test")
        task.start()
        assert not task.is_terminal

    def test_is_terminal_completed(self):
        """Test is_terminal for completed task."""
        task = Task(id="test", name="Test", action="test")
        task.complete("result")
        assert task.is_terminal

    def test_is_terminal_failed(self):
        """Test is_terminal for failed task."""
        task = Task(id="test", name="Test", action="test")
        task.fail(Exception("error"))
        assert task.is_terminal

    def test_is_terminal_skipped(self):
        """Test is_terminal for skipped task."""
        task = Task(id="test", name="Test", action="test")
        task.skip("reason")
        assert task.is_terminal

    def test_to_dict(self):
        """Test task serialization to dict."""
        task = Task(
            id="test",
            name="Test Task",
            action="test_action",
            parameters={"key": "value"},
            dependencies=["dep1"],
            timeout=60,
            max_retries=2,
            metadata={"meta": "data"},
        )

        task_dict = task.to_dict()

        assert task_dict["id"] == "test"
        assert task_dict["name"] == "Test Task"
        assert task_dict["action"] == "test_action"
        assert task_dict["parameters"] == {"key": "value"}
        assert task_dict["dependencies"] == ["dep1"]
        assert task_dict["status"] == "pending"
        assert task_dict["result"] is None
        assert task_dict["error"] is None
        assert task_dict["timeout"] == 60
        assert task_dict["max_retries"] == 2
        assert task_dict["retry_count"] == 0
        assert task_dict["metadata"] == {"meta": "data"}
        assert "created_at" in task_dict
        assert "started_at" in task_dict
        assert "completed_at" in task_dict
        assert "execution_time" in task_dict

    def test_from_dict(self):
        """Test task deserialization from dict."""
        task_dict = {
            "id": "test",
            "name": "Test Task",
            "action": "test_action",
            "parameters": {"key": "value"},
            "dependencies": ["dep1"],
            "status": "completed",
            "result": "test_result",
            "error": None,
            "timeout": 60,
            "max_retries": 2,
            "retry_count": 0,
            "metadata": {"meta": "data"},
            "created_at": time.time(),
            "started_at": time.time(),
            "completed_at": time.time(),
        }

        task = Task.from_dict(task_dict)

        assert task.id == "test"
        assert task.name == "Test Task"
        assert task.action == "test_action"
        assert task.parameters == {"key": "value"}
        assert task.dependencies == ["dep1"]
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "test_result"
        assert task.error is None
        assert task.timeout == 60
        assert task.max_retries == 2
        assert task.retry_count == 0
        assert task.metadata == {"meta": "data"}

    def test_from_dict_with_error(self):
        """Test task deserialization with error."""
        task_dict = {
            "id": "test",
            "name": "Test Task",
            "action": "test_action",
            "status": "failed",
            "error": "Test error message",
        }

        task = Task.from_dict(task_dict)

        assert task.status == TaskStatus.FAILED
        assert isinstance(task.error, Exception)
        assert str(task.error) == "Test error message"

    def test_repr(self):
        """Test task string representation."""
        task = Task(id="test", name="Test Task", action="test_action")
        repr_str = repr(task)

        assert "test" in repr_str
        assert "Test Task" in repr_str
        assert "pending" in repr_str

    def test_equality(self):
        """Test task equality comparison."""
        task1 = Task(id="test", name="Task 1", action="action1")
        task2 = Task(id="test", name="Task 2", action="action2")
        task3 = Task(id="other", name="Task 3", action="action3")

        assert task1 == task2  # Same ID
        assert task1 != task3  # Different ID
        assert task1 != "not_a_task"  # Different type

    def test_hash(self):
        """Test task hashing."""
        task1 = Task(id="test", name="Task 1", action="action1")
        task2 = Task(id="test", name="Task 2", action="action2")
        task3 = Task(id="other", name="Task 3", action="action3")

        assert hash(task1) == hash(task2)  # Same ID
        assert hash(task1) != hash(task3)  # Different ID

        # Test use in set
        task_set = {task1, task2, task3}
        assert len(task_set) == 2  # task1 and task2 are the same


class TestTaskStatus:
    """Test cases for TaskStatus enum."""

    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.SKIPPED.value == "skipped"

    def test_task_status_enum_members(self):
        """Test TaskStatus enum members."""
        assert len(TaskStatus) == 5
        assert TaskStatus.PENDING in TaskStatus
        assert TaskStatus.RUNNING in TaskStatus
        assert TaskStatus.COMPLETED in TaskStatus
        assert TaskStatus.FAILED in TaskStatus
        assert TaskStatus.SKIPPED in TaskStatus
