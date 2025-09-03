#!/usr/bin/env python3
"""
Test the runtime resolution system with control_flow_advanced.yaml.

This script tests if the new runtime resolution system fixes Issue #159
by properly rendering templates in for_each loops.
"""

import asyncio
import sys
import os
import yaml
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestrator.runtime import RuntimeResolutionIntegration


def load_pipeline_config(yaml_path: str):
    """Load pipeline configuration from YAML."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def test_template_resolution():
    """Test template resolution with the control_flow_advanced example."""
    print("=" * 80)
    print("Testing Runtime Resolution System with control_flow_advanced.yaml")
    print("=" * 80)
    
    # Initialize runtime resolution
    runtime = RuntimeResolutionIntegration("control_flow_advanced")
    
    # Load pipeline config
    config = load_pipeline_config("examples/control_flow_advanced.yaml")
    
    # Register pipeline parameters
    print("\n1. Registering pipeline parameters...")
    pipeline_params = config.get('parameters', {})
    for key, value in pipeline_params.items():
        runtime.state.register_variable(key, value)
        print(f"   - {key}: {value}")
    
    # Simulate some task results from the pipeline
    print("\n2. Simulating task execution results...")
    
    # Generate text task result
    generate_text_result = "This is a test text for translation"
    runtime.register_task_result("generate_text_0", {
        "result": generate_text_result,
        "status": "success"
    })
    print(f"   - generate_text_0: {generate_text_result}")
    
    # Translate task results (simulating the for_each loop)
    languages = ["Spanish", "French", "German"]
    translations = {
        "Spanish": "Este es un texto de prueba para traducción",
        "French": "Ceci est un texte de test pour la traduction", 
        "German": "Dies ist ein Testtext für die Übersetzung"
    }
    
    for i, lang in enumerate(languages):
        task_id = f"translate_text_{i}"
        runtime.register_task_result(task_id, {
            "result": translations[lang],
            "status": "success",
            "language": lang
        })
        print(f"   - {task_id}: {translations[lang][:50]}...")
    
    # Test template resolution
    print("\n3. Testing template resolution...")
    
    # Template that was failing in Issue #159
    template = "Translation: {{ translate_text_0['result'] }}"
    print(f"   Template: {template}")
    
    try:
        resolved = runtime.resolve_template_with_context(template)
        print(f"   Resolved: {resolved}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test with different access patterns
    print("\n4. Testing different template access patterns...")
    
    test_templates = [
        ("Direct result access", "{{ translate_text_0 }}"),
        ("Result field access", "{{ translate_text_0_result }}"),
        ("Dict key access", "{{ translate_text_0['result'] }}"),
        ("Original text", "{{ generate_text_0_result }}"),
    ]
    
    for desc, template in test_templates:
        try:
            resolved = runtime.resolve_template_with_context(template)
            print(f"   {desc}:")
            print(f"      Template: {template}")
            print(f"      Resolved: {resolved[:100] if isinstance(resolved, str) else resolved}")
        except Exception as e:
            print(f"   {desc}: ERROR - {e}")
    
    # Test loop expansion
    print("\n5. Testing for_each loop expansion...")
    
    # Create a mock for_each task
    class MockForEachTask:
        def __init__(self):
            self.id = "translate_loop"
            self.for_each_expr = "{{ languages }}"
            self.loop_steps = [
                {
                    "id": "translate",
                    "action": "generate",
                    "parameters": {
                        "prompt": "Translate to {{ item }}: {{ generate_text_0_result }}",
                        "model": "gpt-3.5-turbo"
                    }
                },
                {
                    "id": "save_translation",
                    "action": "save_to_file",
                    "parameters": {
                        "path": "translations/{{ item }}.txt",
                        "content": "Translation: {{ translate }}"  # This should reference the translate step result
                    },
                    "dependencies": ["translate"]
                }
            ]
            self.dependencies = ["generate_text_0"]
            self.metadata = {}
    
    # Register languages for loop
    runtime.state.register_variable("languages", languages)
    
    # Convert and register the loop
    mock_task = MockForEachTask()
    loop_task = runtime.convert_for_each_task(mock_task)
    
    # Check if loop can expand
    can_expand = runtime.can_expand_loop(loop_task.id)
    print(f"   Can expand loop: {can_expand}")
    
    if can_expand:
        # Expand the loop
        expanded_tasks = runtime.expand_specific_loop(loop_task.id)
        print(f"   Expanded {len(expanded_tasks)} tasks:")
        
        for task in expanded_tasks[:3]:  # Show first 3
            print(f"      - {task['id']}")
            print(f"        Action: {task['action']}")
            if 'prompt' in task.get('parameters', {}):
                print(f"        Prompt: {task['parameters']['prompt'][:60]}...")
            if 'path' in task.get('parameters', {}):
                print(f"        Path: {task['parameters']['path']}")
    
    # Test while loop
    print("\n6. Testing while loop expansion...")
    
    # Initialize counter
    runtime.state.register_variable("retry_count", 0)
    runtime.state.register_variable("max_retries", 3)
    
    class MockWhileTask:
        def __init__(self):
            self.id = "retry_loop"
            self.dependencies = []
            self.metadata = {
                "condition": "retry_count < max_retries",
                "steps": [
                    {
                        "id": "attempt",
                        "action": "generate",
                        "parameters": {
                            "prompt": "Attempt {{ iteration }}"
                        }
                    }
                ],
                "max_iterations": 10
            }
    
    mock_while = MockWhileTask()
    while_loop = runtime.convert_while_task(mock_while)
    
    # Try to expand first iteration
    can_expand_while = runtime.can_expand_loop(while_loop.id)
    print(f"   Can expand while loop: {can_expand_while}")
    
    if can_expand_while:
        iteration_tasks = runtime.expand_specific_loop(while_loop.id)
        print(f"   Expanded iteration with {len(iteration_tasks)} tasks")
        if iteration_tasks:
            print(f"      First task: {iteration_tasks[0]['id']}")
    
    # Get execution summary
    print("\n7. Execution Summary:")
    summary = runtime.get_execution_summary()
    print(f"   Total items in context: {len(runtime.state.get_available_context())}")
    print(f"   Resolved items: {summary.get('resolved_items', 0)}")
    print(f"   Failed items: {summary.get('failed_tasks', 0)}")
    print(f"   Loops registered: {summary['loops']['registered']}")
    print(f"   Loops expandable: {summary['loops']['expandable']}")
    print(f"   Loops completed: {summary['loops']['completed']}")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_template_resolution()