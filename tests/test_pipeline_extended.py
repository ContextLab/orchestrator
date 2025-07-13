"""Extended tests for Pipeline class to improve coverage."""

import pytest
import time

from orchestrator.core.pipeline import Pipeline, CircularDependencyError, InvalidDependencyError
from orchestrator.core.task import Task, TaskStatus


class TestPipelineExtended:
    """Extended test cases for Pipeline class to improve coverage."""
    
    def test_circular_dependency_detection_proper_order(self):
        """Test circular dependency detection when tasks are added in proper order."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        
        # Add tasks one by one to test circular dependency detection at each step
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2", dependencies=["task1"])
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        
        # Now try to add a task that creates a circular dependency
        task3 = Task(id="task3", name="Task 3", action="action3", dependencies=["task2"])
        pipeline.add_task(task3)
        
        # This should create a circular dependency
        task_circular = Task(id="task1_circular", name="Task 1 Circular", action="action1", 
                           dependencies=["task3"])
        
        # Temporarily modify task1 to depend on task3 to create a cycle
        task1.dependencies = ["task3"]
        
        with pytest.raises(CircularDependencyError):
            pipeline._validate_dependencies()
    
    def test_circular_dependency_complex_cycle(self):
        """Test complex circular dependency detection."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        
        # Build a complex pipeline first
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2", dependencies=["task1"])
        task3 = Task(id="task3", name="Task 3", action="action3", dependencies=["task2"])
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        
        # Now modify task1 to depend on task3, creating a cycle
        task1.dependencies = ["task3"]
        
        with pytest.raises(CircularDependencyError):
            pipeline._validate_dependencies()
    
    def test_cycle_detection_with_multiple_cycles(self):
        """Test cycle detection when multiple cycles exist."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        
        # Create a pipeline with potential for multiple cycles
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2", dependencies=["task1"])
        task3 = Task(id="task3", name="Task 3", action="action3", dependencies=["task2"])
        task4 = Task(id="task4", name="Task 4", action="action4")
        task5 = Task(id="task5", name="Task 5", action="action5", dependencies=["task4"])
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        pipeline.add_task(task4)
        pipeline.add_task(task5)
        
        # Create cycles
        task1.dependencies = ["task3"]  # Cycle 1: task1 -> task3 -> task2 -> task1
        task4.dependencies = ["task5"]  # Cycle 2: task4 -> task5 -> task4
        
        cycles = pipeline._detect_cycles()
        assert len(cycles) >= 1  # Should detect at least one cycle
    
    def test_get_ready_tasks_with_different_statuses(self):
        """Test get_ready_tasks with tasks in different statuses."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2", dependencies=["task1"])
        task3 = Task(id="task3", name="Task 3", action="action3")
        task4 = Task(id="task4", name="Task 4", action="action4", dependencies=["task1"])
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        pipeline.add_task(task4)
        
        # Set different statuses
        task1.complete("result1")  # Completed
        task3.start()              # Running
        # task2 and task4 should be ready now that task1 is complete
        # task3 is running so not ready
        
        ready_tasks = pipeline.get_ready_task_ids({"task1"})
        # Only task2 and task4 should be ready (pending tasks with satisfied dependencies)
        expected_ready = {"task2", "task4"}
        assert set(ready_tasks) == expected_ready
    
    def test_execution_order_with_circular_dependency(self):
        """Test execution order when circular dependency is detected."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2", dependencies=["task1"])
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        
        # Create a circular dependency
        task1.dependencies = ["task2"]
        
        # This should raise an error when trying to get execution order
        with pytest.raises(CircularDependencyError, match="Cannot determine execution order"):
            pipeline.get_execution_order()
    
    def test_critical_path_empty_pipeline(self):
        """Test critical path for empty pipeline."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        critical_path = pipeline.get_critical_path()
        assert critical_path == []
    
    def test_critical_path_single_task(self):
        """Test critical path for single task."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        task1 = Task(id="task1", name="Task 1", action="action1")
        pipeline.add_task(task1)
        
        critical_path = pipeline.get_critical_path()
        assert critical_path == ["task1"]
    
    def test_critical_path_multiple_branches(self):
        """Test critical path with multiple branches of different lengths."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        
        # Create a diamond-shaped dependency graph
        task1 = Task(id="task1", name="Task 1", action="action1")  # Start
        task2 = Task(id="task2", name="Task 2", action="action2", dependencies=["task1"])  # Left branch
        task3 = Task(id="task3", name="Task 3", action="action3", dependencies=["task1"])  # Right branch (shorter)
        task4 = Task(id="task4", name="Task 4", action="action4", dependencies=["task2"])  # Left continues
        task5 = Task(id="task5", name="Task 5", action="action5", dependencies=["task4"])  # Left continues (longest)
        task6 = Task(id="task6", name="Task 6", action="action6", dependencies=["task3", "task5"])  # End
        
        for task in [task1, task2, task3, task4, task5, task6]:
            pipeline.add_task(task)
        
        critical_path = pipeline.get_critical_path()
        # The longest path should be task1 -> task2 -> task4 -> task5 -> task6
        expected_path = ["task1", "task2", "task4", "task5", "task6"]
        assert critical_path == expected_path
    
    def test_from_dict_with_empty_tasks(self):
        """Test Pipeline.from_dict with empty tasks."""
        pipeline_dict = {
            "id": "test",
            "name": "Test Pipeline",
            "context": {},
            "metadata": {},
            "created_at": time.time(),
            "version": "1.0.0",
            "description": None
        }
        
        pipeline = Pipeline.from_dict(pipeline_dict)
        assert len(pipeline.tasks) == 0
        assert pipeline.id == "test"
        assert pipeline.name == "Test Pipeline"
    
    def test_from_dict_without_tasks_key(self):
        """Test Pipeline.from_dict without tasks key."""
        pipeline_dict = {
            "id": "test",
            "name": "Test Pipeline",
            "context": {},
            "metadata": {},
            "created_at": time.time(),
            "version": "1.0.0",
            "description": None
        }
        
        pipeline = Pipeline.from_dict(pipeline_dict)
        assert len(pipeline.tasks) == 0
    
    def test_skipped_tasks_in_progress(self):
        """Test progress reporting with skipped tasks."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2")
        task3 = Task(id="task3", name="Task 3", action="action3")
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        
        # Set one task to skipped status
        task1.status = TaskStatus.SKIPPED
        task2.complete("result")
        # task3 remains pending
        
        progress = pipeline.get_progress()
        assert progress["total"] == 3
        assert progress["skipped"] == 1
        assert progress["completed"] == 1
        assert progress["pending"] == 1
    
    def test_validate_dependencies_edge_cases(self):
        """Test dependency validation edge cases."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        
        # Task with self-dependency should be caught by Task validation
        with pytest.raises(ValueError, match="cannot depend on itself"):
            task1 = Task(id="task1", name="Task 1", action="action1", dependencies=["task1"])
    
    def test_dependency_validation_with_valid_dependencies(self):
        """Test that valid dependencies don't raise errors."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        
        task1 = Task(id="task1", name="Task 1", action="action1")
        pipeline.add_task(task1)
        
        # This should work fine
        task2 = Task(id="task2", name="Task 2", action="action2", dependencies=["task1"])
        pipeline.add_task(task2)
        
        # Validate dependencies manually
        pipeline._validate_dependencies()  # Should not raise
    
    def test_get_dependents_functionality(self):
        """Test _get_dependents method."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2", dependencies=["task1"])
        task3 = Task(id="task3", name="Task 3", action="action3", dependencies=["task1"])
        task4 = Task(id="task4", name="Task 4", action="action4")
        
        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        pipeline.add_task(task4)
        
        # task1 should have task2 and task3 as dependents
        dependents = pipeline._get_dependents("task1")
        assert set(dependents) == {"task2", "task3"}
        
        # task4 should have no dependents
        dependents = pipeline._get_dependents("task4")
        assert dependents == []
    
    def test_execution_order_levels_complex(self):
        """Test execution order with complex dependency levels."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        
        # Create a more complex pipeline with multiple levels
        # Level 0: task1, task2 (no dependencies)
        # Level 1: task3 (depends on task1), task4 (depends on task2)
        # Level 2: task5 (depends on task3 and task4)
        
        task1 = Task(id="task1", name="Task 1", action="action1")
        task2 = Task(id="task2", name="Task 2", action="action2")
        task3 = Task(id="task3", name="Task 3", action="action3", dependencies=["task1"])
        task4 = Task(id="task4", name="Task 4", action="action4", dependencies=["task2"])
        task5 = Task(id="task5", name="Task 5", action="action5", dependencies=["task3", "task4"])
        
        for task in [task1, task2, task3, task4, task5]:
            pipeline.add_task(task)
        
        execution_order = pipeline.get_execution_levels()
        
        assert len(execution_order) == 3
        assert set(execution_order[0]) == {"task1", "task2"}
        assert set(execution_order[1]) == {"task3", "task4"}
        assert execution_order[2] == ["task5"]
    
    def test_empty_pipeline_methods(self):
        """Test various methods on empty pipeline."""
        pipeline = Pipeline(id="test", name="Test Pipeline")
        
        assert pipeline.get_ready_task_ids(set()) == []
        assert pipeline.get_failed_tasks() == []
        assert pipeline.get_completed_tasks() == []
        assert pipeline.get_running_tasks() == []
        assert pipeline.is_complete() == True  # Empty pipeline is considered complete
        assert pipeline.is_failed() == False
        assert pipeline.get_critical_path() == []
        
        progress = pipeline.get_progress()
        assert progress["total"] == 0
        assert all(count == 0 for key, count in progress.items() if key != "total")