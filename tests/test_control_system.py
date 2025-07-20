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


# TestMockControlSystem class removed since MockControlSystem is gone
# All mock-specific tests have been removed