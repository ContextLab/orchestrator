"""Real tests for control flow features with actual functionality."""

import pytest
import asyncio
import os
import json
from src.orchestrator.engine.control_flow_engine import ControlFlowEngine
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.tools.base import Tool, default_registry
from src.orchestrator.compiler.control_flow_compiler import ControlFlowCompiler
from src.orchestrator.control_flow.conditional import ConditionalHandler
from src.orchestrator.core.task import Task, TaskStatus


class RealTestTool(Tool):
    """Real test tool that actually executes."""
    
    def __init__(self):
        super().__init__("test_tool", "Tool for testing control flow")
        
    async def execute(self, **kwargs) -> dict:
        """Execute test action."""
        action = kwargs.get('action', 'default')
        
        if action == 'check':
            value = kwargs.get('value', 0)
            return {"result": value > 5, "value": value}
        elif action == 'process':
            return {"status": "processed", "type": kwargs.get('type', 'unknown')}
        elif action == 'analyze':
            return {"score": 0.75, "analysis": "complete"}
        elif action == 'list':
            return {"items": ["item1", "item2", "item3"]}
        else:
            return {"result": f"Executed {action}"}


@pytest.mark.asyncio
async def test_conditional_execution_real():
    """Test conditional execution with real tools."""
    yaml_content = """
name: Conditional Test
version: "1.0.0"
steps:
  - id: check_value
    action: test_tool
    parameters:
      action: check
      value: 10
      
  - id: high_branch
    action: test_tool
    if: "true"
    parameters:
      action: process
      type: high
    depends_on: [check_value]
      
  - id: low_branch
    action: test_tool
    if: "false"
    parameters:
      action: process
      type: low
    depends_on: [check_value]
      
  - id: final_step
    action: test_tool
    parameters:
      action: analyze
    depends_on: [high_branch, low_branch]
"""
    
    # Register test tool
    default_registry.register(RealTestTool())
    
    # Create engine without model requirements
    engine = ControlFlowEngine(tool_registry=default_registry)
    
    # Execute
    result = await engine.execute_yaml(yaml_content, {})
    
    # Debug output
    print(f"Completed tasks: {result['completed_tasks']}")
    print(f"Skipped tasks: {result['skipped_tasks']}")
    print(f"Results: {json.dumps(result['results'], indent=2)}")
    
    # Verify results
    assert result['success'] is True
    assert 'check_value' in result['completed_tasks']
    assert 'high_branch' in result['completed_tasks']
    assert 'low_branch' in result['skipped_tasks']
    assert 'final_step' in result['completed_tasks']
    
    # Check specific results
    assert result['results']['check_value']['result'] is True
    assert result['results']['high_branch']['type'] == 'high'
    assert result['results']['final_step']['analysis'] == 'complete'


@pytest.mark.asyncio
async def test_for_loop_execution_real():
    """Test for-each loop with real execution."""
    yaml_content = """
name: Loop Test
version: "1.0.0"
steps:
  - id: get_items
    action: test_tool
    parameters:
      action: list
      
  - id: process_each
    for_each: '["apple", "banana", "cherry"]'
    action: test_tool
    parameters:
      action: process
      type: "fruit_{{$index}}"
    depends_on: [get_items]
"""
    
    # Register test tool
    default_registry.register(RealTestTool())
    
    # Create engine
    engine = ControlFlowEngine(tool_registry=default_registry)
    
    # Execute
    result = await engine.execute_yaml(yaml_content, {})
    
    # Verify results
    assert result['success'] is True
    assert 'get_items' in result['completed_tasks']
    
    # Check that all loop iterations executed
    completed = result['completed_tasks']
    assert any('process_each_0' in task for task in completed)
    assert any('process_each_1' in task for task in completed)
    assert any('process_each_2' in task for task in completed)
    
    # Verify iteration results
    for i in range(3):
        task_id = f"process_each_{i}_process_each_item"
        assert result['results'][task_id]['type'] == f'fruit_{i}'


@pytest.mark.asyncio
async def test_dynamic_flow_real():
    """Test dynamic flow control with goto."""
    yaml_content = """
name: Dynamic Flow Test
version: "1.0.0"
steps:
  - id: start
    action: test_tool
    parameters:
      action: check
      value: 3
      
  - id: router
    action: test_tool
    parameters:
      action: default
    goto: "{{ start.result == true ? 'success_path' : 'failure_path' }}"
    depends_on: [start]
    
  - id: skipped_step
    action: test_tool
    parameters:
      action: process
      note: "This should be skipped"
    depends_on: [router]
    
  - id: failure_path
    action: test_tool
    parameters:
      action: process
      type: failure
      
  - id: success_path
    action: test_tool
    parameters:
      action: process
      type: success
      
  - id: end
    action: test_tool
    parameters:
      action: analyze
    depends_on: [failure_path, success_path]
"""
    
    # Register test tool
    default_registry.register(RealTestTool())
    
    # Create engine
    engine = ControlFlowEngine(tool_registry=default_registry)
    
    # Execute
    result = await engine.execute_yaml(yaml_content, {})
    
    # Verify results
    assert result['success'] is True
    assert 'start' in result['completed_tasks']
    assert 'router' in result['completed_tasks']
    assert 'failure_path' in result['completed_tasks']  # Since value 3 <= 5
    assert 'end' in result['completed_tasks']
    
    # Verify skipped tasks
    assert 'skipped_step' in result['skipped_tasks']
    assert 'success_path' not in result['completed_tasks']


def test_conditional_task_creation():
    """Test creating conditional tasks."""
    handler = ConditionalHandler()
    
    # Create a simple conditional task without needing model
    task_def = {
        'id': 'test_task',
        'action': 'test',
        'if': '{{ value > 10 }}',
        'else': 'alternative_task'
    }
    
    task = handler.create_conditional_task(task_def)
    
    assert task.id == 'test_task'
    assert task.metadata['condition'] == '{{ value > 10 }}'
    assert task.metadata['else_task_id'] == 'alternative_task'


def test_task_dependency_resolution():
    """Test dynamic dependency resolution without model."""
    task1 = Task(id='task1', name='Task 1', action='test')
    task2 = Task(id='task2', name='Task 2', action='test', dependencies=['task1'])
    task3 = Task(id='task3', name='Task 3', action='test', dependencies=['task1', 'task2'])
    
    # Complete task1
    task1.complete({'result': 'done'})
    
    # Check dependencies
    assert task2.is_ready({'task1'})
    assert not task3.is_ready({'task1'})
    assert task3.is_ready({'task1', 'task2'})


@pytest.mark.asyncio
async def test_compiler_integration():
    """Test control flow compiler integration."""
    yaml_content = """
name: Compiler Test
version: "1.0.0"
parameters:
  threshold:
    type: number
    default: 5
steps:
  - id: check
    action: test_tool
    parameters:
      action: check
      value: "{{ threshold }}"
      
  - id: result
    action: test_tool
    if: "{{ check.result }}"
    parameters:
      action: process
      type: "above_threshold"
"""
    
    # Create compiler without model dependency
    compiler = ControlFlowCompiler()
    # Mock the ambiguity resolver to avoid model requirements
    compiler.ambiguity_resolver.model = None
    compiler.ambiguity_resolver.resolve = lambda x, y: x  # Just return the content
    
    # Compile pipeline
    pipeline = await compiler.compile(yaml_content, {'threshold': 7}, resolve_ambiguities=False)
    
    # Verify pipeline structure
    assert pipeline.id == 'Compiler Test'
    assert len(pipeline.tasks) == 2
    assert 'check' in pipeline.tasks
    assert 'result' in pipeline.tasks
    
    # Verify conditional metadata
    result_task = pipeline.tasks['result']
    assert 'condition' in result_task.metadata
    assert result_task.metadata['condition'] == '{{ check.result }}'


if __name__ == "__main__":
    # Run the real tests
    asyncio.run(test_conditional_execution_real())
    asyncio.run(test_for_loop_execution_real())
    asyncio.run(test_dynamic_flow_real())
    test_conditional_task_creation()
    test_task_dependency_resolution()
    asyncio.run(test_compiler_integration())
    
    print("All real tests passed!")