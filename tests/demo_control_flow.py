#!/usr/bin/env python3
"""Demo script for control flow features."""

import asyncio
import json
from typing import Dict, Any

from src.orchestrator.engine.control_flow_engine import ControlFlowEngine
from src.orchestrator.tools.base import Tool, ToolRegistry


# Mock tools for demonstration
class DemoTool(Tool):
    """Generic demo tool for testing."""
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        action = kwargs.get('action', 'default')
        
        if action == 'analyze':
            return {
                "summary": "Analysis complete",
                "score": 0.75,
                "findings": ["finding1", "finding2", "finding3"]
            }
        elif action == 'process':
            method = kwargs.get('method', 'basic')
            return {
                "processed": True,
                "method": method,
                "result": f"Processed with {method} method"
            }
        elif action == 'validate':
            return {
                "valid": True,
                "score": kwargs.get('threshold', 0.8) + 0.1,
                "data": kwargs.get('data', {})
            }
        else:
            return {"result": f"Action {action} completed"}


class DemoDecisionTool(Tool):
    """Demo tool for decision making."""
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        task = kwargs.get('task', '')
        
        # Simple decision logic for demo
        if 'advanced' in task.lower() or 'true' in task.lower():
            return {"result": True}
        elif 'simple' in task.lower() or 'false' in task.lower():
            return {"result": False}
        elif 'how many' in task.lower():
            return {"result": 3}
        elif 'generate' in task.lower() and 'array' in task.lower():
            return {"result": ["Branch A: Performance", "Branch B: Security", "Branch C: Usability"]}
        else:
            return {"result": "decision made"}


class DemoSearchTool(Tool):
    """Demo web search tool."""
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        query = kwargs.get('query', '')
        max_results = kwargs.get('max_results', 5)
        
        # Generate fake search results
        results = []
        for i in range(max_results):
            results.append({
                "title": f"Result {i+1} for '{query}'",
                "url": f"https://example.com/{i+1}",
                "snippet": f"This is a snippet about {query}..."
            })
            
        return {"results": results}


class DemoReportTool(Tool):
    """Demo report generator tool."""
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        title = kwargs.get('title', 'Report')
        content = kwargs.get('content', '')
        kwargs.get('format', 'markdown')
        
        # Simple template processing
        if '{{' in content:
            # Just return the template for demo
            report = f"# {title}\n\n{content}"
        else:
            report = f"# {title}\n\n{content}"
            
        return {"report": report}


async def demo_conditional_execution():
    """Demonstrate conditional execution."""
    print("\n=== Conditional Execution Demo ===")
    
    yaml_content = """
name: Conditional Demo
steps:
  - id: check_value
    action: demo
    parameters:
      action: analyze
      
  - id: high_path
    action: demo
    if: "{{ check_value.score > 0.7 }}"
    parameters:
      action: process
      method: advanced
      
  - id: low_path
    action: demo
    if: "{{ check_value.score <= 0.7 }}"
    parameters:
      action: process
      method: simple
"""
    
    # Setup registry
    registry = ToolRegistry()
    registry.register(DemoTool("demo", "Demo tool"))
    
    # Create engine
    engine = ControlFlowEngine(tool_registry=registry)
    
    # Execute
    result = await engine.execute_yaml(yaml_content, {})
    
    print(f"Success: {result['success']}")
    print(f"Completed: {result['completed_tasks']}")
    print(f"Skipped: {result['skipped_tasks']}")
    print(f"Results: {json.dumps(result['results'], indent=2)}")


async def demo_for_loop():
    """Demonstrate for-each loops."""
    print("\n=== For Loop Demo ===")
    
    yaml_content = """
name: For Loop Demo
steps:
  - id: get_items
    action: decision
    parameters:
      task: "Generate 3 items array"
      
  - id: process_items
    for_each: '["Item A", "Item B", "Item C"]'
    action: demo
    parameters:
      action: process
      item: "{{$item}}"
      index: "{{$index}}"
"""
    
    # Setup registry
    registry = ToolRegistry()
    registry.register(DemoTool("demo", "Demo tool"))
    registry.register(DemoDecisionTool("decision", "Decision tool"))
    
    # Create engine
    engine = ControlFlowEngine(tool_registry=registry)
    
    # Execute
    result = await engine.execute_yaml(yaml_content, {})
    
    print(f"Success: {result['success']}")
    print(f"Completed tasks: {len(result['completed_tasks'])}")
    
    # Show loop results
    for task_id in sorted(result['completed_tasks']):
        if 'process_items' in task_id:
            print(f"  {task_id}: {result['results'][task_id]}")


async def demo_while_loop():
    """Demonstrate while loops."""
    print("\n=== While Loop Demo ===")
    
    yaml_content = """
name: While Loop Demo
steps:
  - id: initialize
    action: demo
    parameters:
      action: initialize
      
  - id: improve_loop
    while: "{{ current_result.score | default(0.5) < 0.9 }}"
    max_iterations: 3
    steps:
      - id: improve
        action: demo
        parameters:
          action: validate
          threshold: "{{ current_result.score | default(0.5) }}"
          
      - id: capture_result
        action: demo
        parameters:
          action: capture
          score: "{{ improve.score }}"
"""
    
    # Setup registry  
    registry = ToolRegistry()
    registry.register(DemoTool("demo", "Demo tool"))
    
    # Create engine
    engine = ControlFlowEngine(tool_registry=registry)
    
    # Execute
    result = await engine.execute_yaml(yaml_content, {"current_result": {"score": 0.5}})
    
    print(f"Success: {result['success']}")
    
    # Count iterations
    iterations = sum(1 for task in result['completed_tasks'] if '_result' in task)
    print(f"While loop iterations: {iterations}")
    
    # Show progression
    for i in range(iterations):
        task_id = f"improve_loop_{i}_improve"
        if task_id in result['results']:
            print(f"  Iteration {i}: score = {result['results'][task_id]['score']}")


async def demo_dynamic_flow():
    """Demonstrate dynamic flow control."""
    print("\n=== Dynamic Flow Control Demo ===")
    
    yaml_content = """
name: Dynamic Flow Demo
steps:
  - id: start
    action: demo
    parameters:
      action: start
      
  - id: check_condition
    action: decision
    parameters:
      task: "Should we go to advanced path?"
      
  - id: router
    action: demo
    goto: "{{ check_condition.result ? 'advanced_path' : 'simple_path' }}"
    depends_on: [check_condition]
    
  - id: simple_path
    action: demo
    parameters:
      action: simple
    goto: "finish"
    
  - id: advanced_path
    action: demo
    parameters:
      action: advanced
    goto: "finish"
    
  - id: skipped_task
    action: demo
    parameters:
      note: "This should be skipped"
      
  - id: finish
    action: demo
    parameters:
      action: complete
"""
    
    # Setup registry
    registry = ToolRegistry()
    registry.register(DemoTool("demo", "Demo tool"))
    registry.register(DemoDecisionTool("decision", "Decision tool"))
    
    # Create engine
    engine = ControlFlowEngine(tool_registry=registry)
    
    # Execute
    result = await engine.execute_yaml(yaml_content, {})
    
    print(f"Success: {result['success']}")
    print(f"Execution path: {' -> '.join(result['completed_tasks'])}")
    print(f"Skipped: {result['skipped_tasks']}")


async def main():
    """Run all demos."""
    print("Control Flow Feature Demonstration")
    print("=" * 50)
    
    await demo_conditional_execution()
    await demo_for_loop()
    await demo_while_loop()
    await demo_dynamic_flow()
    
    print("\n" + "=" * 50)
    print("All demos completed!")


if __name__ == "__main__":
    asyncio.run(main())