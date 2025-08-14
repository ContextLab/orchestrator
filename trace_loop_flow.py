#!/usr/bin/env python3
"""
Trace the flow of loop variables through the system to understand where they get lost.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, '/Users/jmanning/orchestrator/src')

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

async def trace_yaml_compilation():
    """Trace how YAML with for_each loops gets compiled."""
    from orchestrator.compiler.yaml_compiler import YAMLCompiler
    
    yaml_content = """
id: trace-test
name: Trace Test
parameters:
  items: ["A", "B"]
steps:
  - id: loop
    for_each: "{{ items }}"
    steps:
      - id: task
        action: generate_text
        parameters:
          prompt: "Item: {{ item }}"
"""
    
    compiler = YAMLCompiler()
    context = {"items": ["A", "B"]}
    
    print("=" * 80)
    print("TRACING YAML COMPILATION")
    print("=" * 80)
    
    # Compile the YAML
    pipeline = await compiler.compile(yaml_content, context)
    
    print(f"\nPipeline ID: {pipeline.id}")
    print(f"Pipeline context: {pipeline.context}")
    print(f"Number of tasks: {len(pipeline.tasks)}")
    
    # Examine each task
    for task_id, task in pipeline.tasks.items():
        print(f"\n--- Task: {task_id} ---")
        print(f"  Type: {type(task).__name__}")
        print(f"  Action: {task.action}")
        print(f"  Parameters: {task.parameters}")
        print(f"  Metadata keys: {list(task.metadata.keys())}")
        
        if "loop_context" in task.metadata:
            lc = task.metadata["loop_context"]
            print(f"  Loop context type: {type(lc)}")
            if hasattr(lc, '__dict__'):
                print(f"  Loop context attributes: {vars(lc)}")
            elif isinstance(lc, dict):
                print(f"  Loop context dict: {lc}")
        
        if "parent_for_each" in task.metadata:
            print(f"  Parent ForEach: {task.metadata['parent_for_each']}")
        
        if "loop_id" in task.metadata:
            print(f"  Loop ID: {task.metadata['loop_id']}")
    
    return pipeline


def trace_orchestrator_execution():
    """Trace how the orchestrator executes loop tasks."""
    import asyncio
    from orchestrator.orchestrator import Orchestrator
    from orchestrator import init_models
    
    yaml_content = """
id: trace-exec
name: Trace Execution
parameters:
  items: ["X"]
steps:
  - id: loop
    for_each: "{{ items }}"
    steps:
      - id: show
        tool: filesystem
        action: write
        parameters:
          path: "/tmp/trace_{{ item }}.txt"
          content: "Item is: {{ item }}"
"""
    
    async def run_trace():
        model_registry = init_models()
        orchestrator = Orchestrator(model_registry=model_registry)
        
        # Add custom logging to trace execution
        original_execute = orchestrator._execute_task_with_resources
        
        async def traced_execute(task, context):
            print(f"\n>>> Executing task: {task.id}")
            print(f"    Task metadata: {task.metadata}")
            print(f"    Context keys: {list(context.keys())}")
            if 'item' in context:
                print(f"    Context['item'] = {context['item']}")
            if 'index' in context:
                print(f"    Context['index'] = {context['index']}")
            
            result = await original_execute(task, context)
            return result
        
        orchestrator._execute_task_with_resources = traced_execute
        
        context = {"items": ["X"]}
        result = await orchestrator.execute_yaml(yaml_content, context=context)
        
        # Check the output
        trace_file = Path("/tmp/trace_X.txt")
        if trace_file.exists():
            print(f"\nOutput file content: {trace_file.read_text()}")
        else:
            print(f"\nOutput file not created!")
        
        # Also check for wrong filename
        wrong_file = Path("/tmp/trace_{{item}}.txt")
        if wrong_file.exists():
            print(f"\nWrong file created: {wrong_file}")
            print(f"Content: {wrong_file.read_text()}")
        
        return result
    
    print("\n" + "=" * 80)
    print("TRACING ORCHESTRATOR EXECUTION")
    print("=" * 80)
    
    return asyncio.run(run_trace())


if __name__ == "__main__":
    import asyncio
    
    # Trace compilation
    pipeline = asyncio.run(trace_yaml_compilation())
    
    # Trace execution
    trace_orchestrator_execution()