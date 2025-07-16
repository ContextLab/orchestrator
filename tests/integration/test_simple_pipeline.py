#!/usr/bin/env python3
"""Test a simple pipeline without model requirements."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.orchestrator import Orchestrator
from orchestrator.core.control_system import MockControlSystem
from orchestrator.core.task import Task


# Mock actions for testing
async def mock_search(task):
    """Mock search action."""
    query = task.parameters.get("query", "")
    print(f"[Search] Searching for: {query}")
    return {
        "results": [
            {"title": "Result 1", "url": "https://example.com/1"},
            {"title": "Result 2", "url": "https://example.com/2"},
            {"title": "Result 3", "url": "https://example.com/3"},
        ]
    }


async def mock_analyze(task):
    """Mock analyze action."""
    data = task.parameters.get("data", {})
    print(f"[Analyze] Analyzing {len(data.get('results', []))} results")
    return {
        "findings": [
            "Key finding 1",
            "Key finding 2",
            "Key finding 3"
        ],
        "insights": "Analysis complete"
    }


async def mock_summarize(task):
    """Mock summarize action."""
    content = task.parameters.get("content", {})
    print("[Summarize] Creating summary")
    return {
        "summary": "# Research Summary\n\n- Key finding 1\n- Key finding 2\n- Key finding 3\n\nAnalysis complete."
    }


class TestControlSystem(MockControlSystem):
    """Test control system with custom actions."""
    
    def __init__(self):
        super().__init__(
            name="test-control-system",
            config={
                "capabilities": {
                    "supported_actions": ["search", "analyze", "summarize"],
                    "parallel_execution": True,
                    "streaming": False,
                    "checkpoint_support": True,
                },
                "base_priority": 10,
            }
        )
        self.actions = {
            "search": mock_search,
            "analyze": mock_analyze,
            "summarize": mock_summarize
        }
        self._results = {}
    
    async def execute_task(self, task: Task, context: dict = None):
        """Execute a task using mock actions."""
        # Handle $results references
        for key, value in task.parameters.items():
            if isinstance(value, str) and value.startswith("$results."):
                parts = value.split(".")
                if len(parts) >= 2:
                    task_id = parts[1]
                    if task_id in self._results:
                        task.parameters[key] = self._results[task_id]
        
        # Execute the action
        handler = self.actions.get(task.action)
        if handler:
            result = await handler(task)
            self._results[task.id] = result
            return result
        else:
            return {"status": "completed", "result": f"Mock result for {task.action}"}


async def test_simple_pipeline():
    """Test a simple pipeline."""
    print("Testing Simple Pipeline")
    print("=" * 50)
    
    # Load the pipeline
    with open("pipelines/simple_research.yaml", "r") as f:
        pipeline_yaml = f.read()
    
    # Initialize orchestrator with test control system
    control_system = TestControlSystem()
    orchestrator = Orchestrator(control_system=control_system)
    
    # Compile and execute using execute_yaml
    try:
        # Execute the pipeline
        results = await orchestrator.execute_yaml(
            pipeline_yaml,
            context={"topic": "Python async programming"}
        )
        
        print("\nPipeline execution completed!")
        print("\nResults:")
        for task_id, result in results.items():
            print(f"\n{task_id}:")
            if isinstance(result, dict) and 'summary' in result:
                print(result['summary'][:100] + "...")
            else:
                print(f"  {result}")
        
        return results
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = asyncio.run(test_simple_pipeline())
    
    if results:
        print("\n✅ Pipeline executed successfully!")
    else:
        print("\n❌ Pipeline execution failed!")
        sys.exit(1)