#!/usr/bin/env python3
"""Debug control flow execution."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator import Orchestrator, init_models
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.models.registry_singleton import get_model_registry

async def debug_pipeline():
    """Debug pipeline compilation."""
    # Initialize models
    print("Initializing models...")
    init_models()
    
    registry = get_model_registry()
    compiler = YAMLCompiler(model_registry=registry)
    
    # Simple test pipeline with control flow
    yaml_content = """
id: test-control-flow
name: Test Control Flow
description: Test control flow with loop

parameters:
  items:
    type: array
    default: ["apple", "banana", "cherry"]

steps:
  - id: process_items
    for_each: "{{ items }}"
    steps:
      - id: process_item
        action: echo
        parameters:
          message: "Processing {{ $item }}"

outputs:
  items: "{{ items }}"
"""
    
    # Compile pipeline
    inputs = {"items": ["rose", "moon", "ocean"]}
    pipeline = await compiler.compile(yaml_content, inputs)
    
    # Check tasks
    print(f"\nPipeline has {len(pipeline.tasks)} tasks:")
    for task_id, task in pipeline.tasks.items():
        print(f"  - {task_id}: action={task.action}, metadata={task.metadata}")
        if "for_each" in task.metadata:
            print(f"    for_each: {task.metadata['for_each']}")
        if "steps" in task.metadata:
            print(f"    steps: {task.metadata['steps']}")


if __name__ == "__main__":
    asyncio.run(debug_pipeline())