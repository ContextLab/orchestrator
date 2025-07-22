"""Tests for control flow features."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from src.orchestrator.control_flow.auto_resolver import ControlFlowAutoResolver
from src.orchestrator.compiler.ambiguity_resolver import AmbiguityResolver
from src.orchestrator.control_flow.conditional import ConditionalHandler, ConditionalTask
from src.orchestrator.control_flow.loops import ForLoopHandler, WhileLoopHandler
from src.orchestrator.control_flow.dynamic_flow import DynamicFlowHandler
from src.orchestrator.compiler.control_flow_compiler import ControlFlowCompiler
from src.orchestrator.engine.control_flow_engine import ControlFlowEngine
from src.orchestrator.core.task import Task, TaskStatus
from src.orchestrator.models.model_registry import ModelRegistry


class TestControlFlowAutoResolver:
    """Test AUTO tag resolution for control flow."""
    
    @pytest.mark.asyncio
    async def test_resolve_condition_boolean(self):
        """Test resolving boolean conditions."""
        # Mock model for resolver
        mock_model = Mock()
        mock_model.generate = AsyncMock(return_value="true")
        
        # Mock ambiguity resolver
        with patch.object(AmbiguityResolver, '__init__', return_value=None):
            resolver = ControlFlowAutoResolver()
            resolver.ambiguity_resolver = Mock()
            resolver.ambiguity_resolver.resolve = AsyncMock(return_value="true")
        
        # Test simple boolean strings
        assert await resolver.resolve_condition("true", {}, {}) is True
        assert await resolver.resolve_condition("false", {}, {}) is False
        assert await resolver.resolve_condition("yes", {}, {}) is True
        assert await resolver.resolve_condition("no", {}, {}) is False
        
    @pytest.mark.asyncio
    async def test_resolve_condition_with_auto(self):
        """Test resolving conditions with AUTO tags."""
        # Mock model registry
        mock_registry = Mock(spec=ModelRegistry)
        mock_model = Mock()
        mock_model.generate = AsyncMock(return_value="true")
        mock_registry.select_model = AsyncMock(return_value=mock_model)
        
        resolver = ControlFlowAutoResolver(mock_registry)
        
        condition = "<AUTO>Should we process advanced features?</AUTO>"
        result = await resolver.resolve_condition(condition, {}, {})
        assert result is True
        
    @pytest.mark.asyncio
    async def test_resolve_iterator(self):
        """Test resolving iterator expressions."""
        resolver = ControlFlowAutoResolver()
        
        # Test list
        items = await resolver.resolve_iterator("[1, 2, 3]", {}, {})
        assert items == [1, 2, 3]
        
        # Test comma-separated
        items = await resolver.resolve_iterator("a, b, c", {}, {})
        assert items == ["a", "b", "c"]
        
        # Test single item
        items = await resolver.resolve_iterator("single", {}, {})
        assert items == ["single"]
        
    @pytest.mark.asyncio
    async def test_resolve_count(self):
        """Test resolving count expressions."""
        resolver = ControlFlowAutoResolver()
        
        assert await resolver.resolve_count("5", {}, {}) == 5
        assert await resolver.resolve_count("10", {}, {}) == 10
        assert await resolver.resolve_count(3, {}, {}) == 3
        
    @pytest.mark.asyncio
    async def test_resolve_target(self):
        """Test resolving jump targets."""
        resolver = ControlFlowAutoResolver()
        
        valid_targets = ["step1", "step2", "step3"]
        
        # Valid target
        target = await resolver.resolve_target("step2", {}, {}, valid_targets)
        assert target == "step2"
        
        # Invalid target
        with pytest.raises(ValueError):
            await resolver.resolve_target("invalid", {}, {}, valid_targets)


class TestConditionalHandler:
    """Test conditional execution handler."""
    
    @pytest.mark.asyncio
    async def test_evaluate_condition(self):
        """Test condition evaluation."""
        handler = ConditionalHandler()
        
        # Task without condition
        task = Task(id="t1", name="Test", action="test")
        assert await handler.evaluate_condition(task, {}, {}) is True
        
        # Task with condition in metadata
        task.metadata = {"condition": "true"}
        assert await handler.evaluate_condition(task, {}, {}) is True
        
        task.metadata = {"condition": "false"}
        assert await handler.evaluate_condition(task, {}, {}) is False
        
    def test_create_conditional_task(self):
        """Test creating conditional tasks."""
        handler = ConditionalHandler()
        
        task_def = {
            "id": "check_data",
            "action": "validate",
            "if": "{{ data_exists }}",
            "else": "skip_validation",
            "parameters": {"data": "{{ input_data }}"}
        }
        
        task = handler.create_conditional_task(task_def)
        
        assert isinstance(task, ConditionalTask)
        assert task.id == "check_data"
        assert task.metadata["condition"] == "{{ data_exists }}"
        assert task.metadata["else_task_id"] == "skip_validation"


class TestForLoopHandler:
    """Test for-each loop handler."""
    
    @pytest.mark.asyncio
    async def test_expand_for_loop(self):
        """Test expanding for-each loops."""
        handler = ForLoopHandler()
        
        loop_def = {
            "id": "process_items",
            "for_each": "[1, 2, 3]",
            "action": "process",
            "parameters": {
                "item": "{{$item}}",
                "index": "{{$index}}"
            }
        }
        
        tasks = await handler.expand_for_loop(loop_def, {}, {})
        
        assert len(tasks) == 3
        assert tasks[0].id == "process_items_0_process_items_item"
        assert tasks[0].parameters["item"] == "1"
        assert tasks[0].parameters["index"] == "0"
        
    @pytest.mark.asyncio
    async def test_expand_for_loop_with_steps(self):
        """Test expanding for-each loops with multiple steps."""
        handler = ForLoopHandler()
        
        loop_def = {
            "id": "multi_step_loop",
            "for_each": "['a', 'b']",
            "steps": [
                {
                    "id": "step1",
                    "action": "action1",
                    "parameters": {"value": "{{$item}}"}
                },
                {
                    "id": "step2", 
                    "action": "action2",
                    "dependencies": ["step1"]
                }
            ]
        }
        
        tasks = await handler.expand_for_loop(loop_def, {}, {})
        
        assert len(tasks) == 4  # 2 items × 2 steps
        
        # Check dependencies
        assert tasks[2].dependencies == ["multi_step_loop_1_step1"]  # step2 of item 1
        assert tasks[3].dependencies == ["multi_step_loop_1_step1", "multi_step_loop_0_step2"]  # Sequential


class TestWhileLoopHandler:
    """Test while loop handler."""
    
    @pytest.mark.asyncio
    async def test_should_continue(self):
        """Test while loop continuation logic."""
        handler = WhileLoopHandler()
        
        # Should stop at max iterations
        should_continue = await handler.should_continue(
            "loop1", "true", {}, {}, 10, 10
        )
        assert should_continue is False
        
        # Should continue if under max
        should_continue = await handler.should_continue(
            "loop2", "true", {}, {}, 5, 10
        )
        assert should_continue is True
        
    @pytest.mark.asyncio
    async def test_create_iteration_tasks(self):
        """Test creating tasks for while loop iteration."""
        handler = WhileLoopHandler()
        
        loop_def = {
            "id": "improve_loop",
            "while": "{{ quality < threshold }}",
            "steps": [
                {
                    "id": "improve",
                    "action": "refine",
                    "parameters": {"iteration": "{{$iteration}}"}
                }
            ]
        }
        
        tasks = await handler.create_iteration_tasks(loop_def, 0, {}, {})
        
        assert len(tasks) == 2  # improve task + result capture
        assert tasks[0].id == "improve_loop_0_improve"
        assert tasks[0].parameters["iteration"] == "0"
        assert tasks[1].id == "improve_loop_0_result"


class TestDynamicFlowHandler:
    """Test dynamic flow control handler."""
    
    @pytest.mark.asyncio
    async def test_process_goto(self):
        """Test processing goto directives."""
        handler = DynamicFlowHandler()
        
        task = Task(
            id="check_error",
            name="Check Error",
            action="check",
            metadata={"goto": "error_handler"}
        )
        
        all_tasks = {
            "check_error": task,
            "error_handler": Task(id="error_handler", name="Handle Error", action="handle"),
            "success_handler": Task(id="success_handler", name="Success", action="success")
        }
        
        target = await handler.process_goto(task, {}, {}, all_tasks)
        assert target == "error_handler"
        
    @pytest.mark.asyncio
    async def test_resolve_dynamic_dependencies(self):
        """Test resolving dynamic dependencies."""
        handler = DynamicFlowHandler()
        
        task = Task(
            id="final_step",
            name="Final",
            action="finalize",
            dependencies=["static_dep"],
            metadata={"dynamic_dependencies": "['dynamic_dep1', 'dynamic_dep2']"}
        )
        
        all_tasks = {
            "static_dep": Task(id="static_dep", name="Static", action="static"),
            "dynamic_dep1": Task(id="dynamic_dep1", name="Dynamic1", action="dyn1"),
            "dynamic_dep2": Task(id="dynamic_dep2", name="Dynamic2", action="dyn2"),
            "final_step": task
        }
        
        deps = await handler.resolve_dynamic_dependencies(task, {}, {}, all_tasks)
        
        assert "static_dep" in deps
        assert "dynamic_dep1" in deps
        assert "dynamic_dep2" in deps


class TestControlFlowEngine:
    """Test control flow execution engine."""
    
    @pytest.mark.asyncio
    async def test_execute_conditional_pipeline(self):
        """Test executing pipeline with conditional logic."""
        yaml_content = """
name: Conditional Pipeline
steps:
  - id: check_condition
    action: check
    parameters:
      value: 10
      
  - id: process_high
    action: process
    if: "true"
    parameters:
      type: high
      
  - id: process_low
    action: process  
    if: "false"
    parameters:
      type: low
"""
        
        # Mock tools
        mock_check = Mock()
        mock_check.execute = AsyncMock(return_value={"result": 15})
        
        mock_process = Mock()
        mock_process.execute = AsyncMock(return_value={"processed": True})
        
        mock_registry = Mock()
        mock_registry.get_tool = Mock(side_effect=lambda name: {
            "check": mock_check,
            "process": mock_process
        }.get(name))
        
        engine = ControlFlowEngine(tool_registry=mock_registry)
        
        result = await engine.execute_yaml(yaml_content, {})
        
        assert result["success"] is True
        assert "check_condition" in result["completed_tasks"]
        assert "process_high" in result["completed_tasks"]
        assert "process_low" in result["skipped_tasks"]
        
    @pytest.mark.asyncio
    async def test_execute_for_loop_pipeline(self):
        """Test executing pipeline with for-each loop."""
        yaml_content = """
name: Loop Pipeline
steps:
  - id: get_items
    action: list
    
  - id: process_items
    for_each: "[1, 2, 3]"
    action: process
    parameters:
      item: "{{$item}}"
      index: "{{$index}}"
    depends_on: [get_items]
"""
        
        # Mock tools
        mock_list = Mock()
        mock_list.execute = AsyncMock(return_value={"items": [1, 2, 3]})
        
        mock_process = Mock()
        mock_process.execute = AsyncMock(return_value={"result": "ok"})
        
        mock_registry = Mock()
        mock_registry.get_tool = Mock(side_effect=lambda name: {
            "list": mock_list,
            "process": mock_process
        }.get(name))
        
        engine = ControlFlowEngine(tool_registry=mock_registry)
        
        result = await engine.execute_yaml(yaml_content, {})
        
        assert result["success"] is True
        assert "get_items" in result["completed_tasks"]
        
        # Check that all loop iterations were executed
        assert mock_process.execute.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])