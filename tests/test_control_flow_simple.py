#!/usr/bin/env python3
"""Simple test for control flow debugging."""

import asyncio
from src.orchestrator.compiler.control_flow_compiler import ControlFlowCompiler
from src.orchestrator.core.task import Task
from src.orchestrator import init_models

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider


async def test_simple_condition():
    """Test simple conditional compilation."""

    yaml_content = """
name: Simple Condition Test
version: "1.0.0"
steps:
  - id: task1
    action: test
    if: "true"
    parameters:
      note: "Should execute"

  - id: task2
    action: test
    if: "false"
    parameters:
      note: "Should not execute"
"""

    # Initialize models and create compiler
    model_registry = init_models()
    compiler = ControlFlowCompiler(model_registry=model_registry)

    # Compile without resolving AUTO tags
    pipeline = await compiler.compile(yaml_content, {}, resolve_ambiguities=False)

    print(f"Pipeline tasks: {list(pipeline.tasks.keys())}")

    # Check task metadata
    for task_id, task in pipeline.tasks.items():
        print(f"\nTask {task_id}:")
        print(f"  Type: {type(task)}")
        print(f"  Metadata: {task.metadata}")
        if hasattr(task, "condition"):
            print(f"  Condition attr: {task.condition}")

    return pipeline


async def test_condition_evaluation():
    """Test condition evaluation in handler."""
    from src.orchestrator.control_flow.conditional import ConditionalHandler

    handler = ConditionalHandler()

    # Create a task with condition
    task = Task(
        id="test_task", name="Test", action="test", metadata={"condition": "true"}
    )

    # Evaluate condition
    result = await handler.evaluate_condition(task, {}, {})
    print(f"Condition 'true' evaluates to: {result}")

    # Test false condition
    task.metadata["condition"] = "false"
    result = await handler.evaluate_condition(task, {}, {})
    print(f"Condition 'false' evaluates to: {result}")

    # Test template condition
    task.metadata["condition"] = "{{ value > 5 }}"
    result = await handler.evaluate_condition(task, {"value": 10}, {})
    print(f"Condition '{{{{ value > 5 }}}}' with value=10 evaluates to: {result}")


if __name__ == "__main__":
    print("Testing simple condition compilation...")
    pipeline = asyncio.run(test_simple_condition())

    print("\n\nTesting condition evaluation...")
    asyncio.run(test_condition_evaluation())
