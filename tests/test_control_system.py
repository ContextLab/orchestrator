"""Comprehensive tests for control system functionality."""

import pytest

from orchestrator.core.control_system import ControlSystem
from orchestrator.core.pipeline import Pipeline
from orchestrator.core.task import Task, TaskStatus


class TestControlSystem:
    """Test cases for ControlSystem abstract base class."""

    def test_control_system_creation_valid_name(self):
        """Test control system creation with valid name."""
        # Create a minimal concrete implementation for testing
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        config = {"test": "value"}
        system = TestControlSystem("test_system", config)

        assert system.name == "test_system"
        assert system.config == config
        assert system._capabilities is not None

    def test_control_system_creation_empty_name(self):
        """Test control system creation with empty name raises error."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        with pytest.raises(ValueError, match="Control system name cannot be empty"):
            TestControlSystem("")

    def test_control_system_creation_none_config(self):
        """Test control system creation with None config."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        system = TestControlSystem("test_system", None)

        assert system.name == "test_system"
        assert system.config is not None
        assert isinstance(system.config, dict)

    def test_load_capabilities_with_config(self):
        """Test loading capabilities from config."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        config = {
            "capabilities": {
                "feature1": True,
                "feature2": False,
                "supported_actions": ["action1", "action2"],
            }
        }
        system = TestControlSystem("test_system", config)

        assert system._capabilities["feature1"] is True
        assert system._capabilities["feature2"] is False
        assert "action1" in system._capabilities["supported_actions"]

    def test_load_capabilities_no_config(self):
        """Test loading capabilities with no capabilities in config."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        system = TestControlSystem("test_system", {"other": "value"})

        # Should have empty capabilities when config doesn't contain capabilities
        assert system._capabilities == {}

    def test_supports_capability_true(self):
        """Test supports_capability returns True for supported capability."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        config = {"capabilities": {"parallel_execution": True, "streaming": False}}
        system = TestControlSystem("test_system", config)

        assert system.supports_capability("parallel_execution") is True

    def test_supports_capability_false(self):
        """Test supports_capability returns False for unsupported capability."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        config = {"capabilities": {}}
        system = TestControlSystem("test_system", config)

        assert system.supports_capability("nonexistent_capability") is False

    def test_can_execute_task_supported_action(self):
        """Test can_execute_task returns True for supported action."""
        class TestControlSystem(ControlSystem):
            def __init__(self):
                config = {
                    "capabilities": {
                        "supported_actions": ["generate", "analyze", "transform", "execute"]
                    }
                }
                super().__init__("test-control", config)
            
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        system = TestControlSystem()
        task = Task(id="test_task", name="Test Task", action="generate")

        assert system.can_execute_task(task) is True

    def test_can_execute_task_unsupported_action(self):
        """Test can_execute_task returns False for unsupported action."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        config = {"capabilities": {"supported_actions": ["generate", "analyze"]}}
        system = TestControlSystem("test_system", config)
        task = Task(id="test_task", name="Test Task", action="unsupported_action")

        assert system.can_execute_task(task) is False

    def test_can_execute_task_no_supported_actions(self):
        """Test can_execute_task when no supported actions specified."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        config = {"capabilities": {}}
        system = TestControlSystem("test_system", config)
        task = Task(id="test_task", name="Test Task", action="any_action")

        # Should return True when no supported_actions are specified
        assert system.can_execute_task(task) is True

    def test_can_execute_task_required_capabilities_satisfied(self):
        """Test can_execute_task when required capabilities are satisfied."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        config = {"capabilities": {"feature1": True, "feature2": True}}
        system = TestControlSystem("test_system", config)
        task = Task(
            id="test_task",
            name="Test Task",
            action="generate",
            metadata={"required_capabilities": ["feature1", "feature2"]},
        )

        assert system.can_execute_task(task) is True

    def test_can_execute_task_required_capabilities_not_satisfied(self):
        """Test can_execute_task when required capabilities are not satisfied."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        config = {"capabilities": {"feature1": True}}
        system = TestControlSystem("test_system", config)
        task = Task(
            id="test_task",
            name="Test Task",
            action="generate",
            metadata={"required_capabilities": ["feature1", "feature2"]},
        )

        assert system.can_execute_task(task) is False

    def test_get_priority_from_config(self):
        """Test get_priority returns base priority from config."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        config = {"base_priority": 50}
        system = TestControlSystem("test_system", config)
        task = Task(id="test_task", name="Test Task", action="generate")

        assert system.get_priority(task) == 50

    def test_get_priority_from_task_metadata(self):
        """Test get_priority returns priority from task metadata."""
        class TestControlSystem(ControlSystem):
            def __init__(self):
                super().__init__("test-control", {})
            
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        system = TestControlSystem()
        task = Task(
            id="test_task",
            name="Test Task",
            action="generate",
            metadata={"priority": 100},
        )

        assert system.get_priority(task) == 100

    def test_get_priority_default(self):
        """Test get_priority returns default when not specified."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        config = {}
        system = TestControlSystem("test_system", config)
        task = Task(id="test_task", name="Test Task", action="generate")

        # When no base_priority in config, should return 0
        assert system.get_priority(task) == 0

    def test_repr(self):
        """Test string representation of control system."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        system = TestControlSystem("test_system")

        assert repr(system) == "ControlSystem(name='test_system')"

    def test_equality_same_name(self):
        """Test equality comparison with same name."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        system1 = TestControlSystem("test_system")
        system2 = TestControlSystem("test_system")

        assert system1 == system2

    def test_equality_different_name(self):
        """Test equality comparison with different name."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        system1 = TestControlSystem("system1")
        system2 = TestControlSystem("system2")

        assert system1 != system2

    def test_equality_non_control_system(self):
        """Test equality comparison with non-ControlSystem object."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        system = TestControlSystem("test_system")

        assert system != "not_a_control_system"
        assert system != 123
        assert system != None

    def test_hash_same_name(self):
        """Test hash function with same name."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        system1 = TestControlSystem("test_system")
        system2 = TestControlSystem("test_system")

        assert hash(system1) == hash(system2)

    def test_hash_different_name(self):
        """Test hash function with different name."""
        class TestControlSystem(ControlSystem):
            async def execute_task(self, task, context=None):
                return {"status": "completed"}
            
            async def health_check(self):
                return True
        
        system1 = TestControlSystem("system1")
        system2 = TestControlSystem("system2")

        assert hash(system1) != hash(system2)


# Remove TestMockControlSystem class since MockControlSystem is gone
# The tests below were specific to MockControlSystem implementation

# Removed mock-specific tests
        """Test MockControlSystem with custom config."""
        config = {
            "capabilities": {
                "supported_actions": ["custom_action"],
                "custom_feature": True,
            },
            "base_priority": 50,
        }
        system = MockControlSystem("custom_system", config)

        assert system.name == "custom_system"
        assert system._capabilities["supported_actions"] == ["custom_action"]
        assert system._capabilities["custom_feature"] is True
        assert system.config["base_priority"] == 50

        """Test setting canned task result."""
        system = MockControlSystem()

        system.set_task_result("task1", "custom_result")

        assert system._task_results["task1"] == "custom_result"

        """Test executing task with canned result."""
        system = MockControlSystem()
        system.set_task_result("task1", "custom_result")

        task = Task(id="task1", name="Task 1", action="generate")
        context = {"pipeline_id": "test_pipeline"}

        result = await system.execute_task(task, context)

        assert result == "custom_result"
        assert len(system._execution_history) == 1
        assert system._execution_history[0]["task_id"] == "task1"
        assert system._execution_history[0]["action"] == "generate"
        assert system._execution_history[0]["context"] == context

        """Test executing task with generate action."""
        system = MockControlSystem()

        task = Task(id="task1", name="Task 1", action="generate")
        context = {}

        result = await system.execute_task(task, context)

        assert result == "Generated content for task task1"
        assert len(system._execution_history) == 1

        """Test executing task with analyze action."""
        system = MockControlSystem()

        task = Task(id="task1", name="Task 1", action="analyze")
        context = {}

        result = await system.execute_task(task, context)

        assert result == {"analysis": "Analysis result for task task1"}
        assert len(system._execution_history) == 1

        """Test executing task with transform action."""
        system = MockControlSystem()

        task = Task(id="task1", name="Task 1", action="transform")
        context = {}

        result = await system.execute_task(task, context)

        assert result == {"transformed_data": "Transformed data for task task1"}
        assert len(system._execution_history) == 1

        """Test executing task with execute action."""
        system = MockControlSystem()

        task = Task(id="task1", name="Task 1", action="execute")
        context = {}

        result = await system.execute_task(task, context)

        assert result == {"execution_result": "Execution result for task task1"}
        assert len(system._execution_history) == 1

        """Test executing task with unknown action."""
        system = MockControlSystem()

        task = Task(id="task1", name="Task 1", action="unknown_action")
        context = {}

        result = await system.execute_task(task, context)

        assert result == "Mock result for action 'unknown_action' in task task1"
        assert len(system._execution_history) == 1

        """Test executing task with parameters."""
        system = MockControlSystem()

        task = Task(
            id="task1",
            name="Task 1",
            action="generate",
            parameters={"param1": "value1", "param2": "value2"},
        )
        context = {"context_key": "context_value"}

        await system.execute_task(task, context)

        assert system._execution_history[0]["parameters"] == {
            "param1": "value1",
            "param2": "value2",
        }
        assert system._execution_history[0]["context"] == {
            "context_key": "context_value"
        }

        """Test executing simple pipeline."""
        system = MockControlSystem()

        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="generate")
        task2 = Task(id="task2", name="Task 2", action="analyze")

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        results = await system.execute_pipeline(pipeline)

        assert "task1" in results
        assert "task2" in results
        assert results["task1"] == "Generated content for task task1"
        assert results["task2"] == {"analysis": "Analysis result for task task2"}

        # Check that tasks were marked as completed
        assert task1.status == TaskStatus.COMPLETED
        assert task2.status == TaskStatus.COMPLETED

        """Test executing pipeline with task dependencies."""
        system = MockControlSystem()

        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="generate")
        task2 = Task(
            id="task2", name="Task 2", action="analyze", dependencies=["task1"]
        )

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        results = await system.execute_pipeline(pipeline)

        assert "task1" in results
        assert "task2" in results

        # Check execution history order
        history = system.get_execution_history()
        assert len(history) == 2
        assert history[0]["task_id"] == "task1"  # Should execute first
        assert history[1]["task_id"] == "task2"  # Should execute after task1

        """Test that pipeline context is passed to tasks."""
        system = MockControlSystem()

        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="generate")

        pipeline.add_task(task1)

        await system.execute_pipeline(pipeline)

        history = system.get_execution_history()
        assert history[0]["context"]["pipeline_id"] == "test_pipeline"
        assert "results" in history[0]["context"]

        """Test that pipeline results accumulate across tasks."""
        system = MockControlSystem()

        pipeline = Pipeline(id="test_pipeline", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="generate")
        task2 = Task(
            id="task2", name="Task 2", action="analyze", dependencies=["task1"]
        )

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        await system.execute_pipeline(pipeline)

        history = system.get_execution_history()

        # Both tasks get all results accumulated during pipeline execution
        # Since they execute in the same level, both get the complete results
        assert len(history[0]["context"]["results"]) >= 0
        assert len(history[1]["context"]["results"]) >= 0

        # Both should have pipeline_id in context
        assert history[0]["context"]["pipeline_id"] == "test_pipeline"
        assert history[1]["context"]["pipeline_id"] == "test_pipeline"

        """Test getting capabilities."""
        system = MockControlSystem()

        capabilities = system.get_capabilities()

        assert capabilities == system._capabilities
        assert "supported_actions" in capabilities
        assert "parallel_execution" in capabilities

        """Test health check."""
        system = MockControlSystem()

        health = await system.health_check()

        assert health is True

        """Test getting execution history."""
        system = MockControlSystem()

        # Initially empty
        history = system.get_execution_history()
        assert history == []

        # Add some history
        system._execution_history.append({"task_id": "task1"})
        system._execution_history.append({"task_id": "task2"})

        history = system.get_execution_history()
        assert len(history) == 2
        assert history[0]["task_id"] == "task1"
        assert history[1]["task_id"] == "task2"

        # Should return copy (modifying returned history shouldn't affect original)
        history.append({"task_id": "task3"})
        assert len(system._execution_history) == 2

        """Test clearing execution history."""
        system = MockControlSystem()

        # Add some history
        system._execution_history.append({"task_id": "task1"})
        system._execution_history.append({"task_id": "task2"})

        assert len(system._execution_history) == 2

        # Clear history
        system.clear_execution_history()

        assert len(system._execution_history) == 0

        """Test clearing task results."""
        system = MockControlSystem()

        # Set some results
        system.set_task_result("task1", "result1")
        system.set_task_result("task2", "result2")

        assert len(system._task_results) == 2

        # Clear results
        system.clear_task_results()

        assert len(system._task_results) == 0

        """Test executing task when canned result is an exception."""
        system = MockControlSystem()
        system.set_task_result("task1", Exception("Test exception"))

        task = Task(id="task1", name="Task 1", action="generate")
        context = {}

        result = await system.execute_task(task, context)

        # Should return the exception object itself
        assert isinstance(result, Exception)
        assert str(result) == "Test exception"

        """Test executing empty pipeline."""
        system = MockControlSystem()

        pipeline = Pipeline(id="empty_pipeline", name="Empty Pipeline")

        results = await system.execute_pipeline(pipeline)

        assert results == {}
        assert len(system.get_execution_history()) == 0

        """Test that MockControlSystem properly inherits from ControlSystem."""
        system = MockControlSystem()

        assert isinstance(system, ControlSystem)
        assert hasattr(system, "name")
        assert hasattr(system, "config")
        assert hasattr(system, "_capabilities")
        assert hasattr(system, "supports_capability")
        assert hasattr(system, "can_execute_task")
        assert hasattr(system, "get_priority")

