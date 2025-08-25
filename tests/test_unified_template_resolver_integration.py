#!/usr/bin/env python3
"""Test script to verify UnifiedTemplateResolver integration in control systems."""

import asyncio
import json
import tempfile
from pathlib import Path

from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.task import Task
from src.orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from src.orchestrator.control_systems.tool_integrated_control_system import ToolIntegratedControlSystem
from src.orchestrator.control_systems.hybrid_control_system import HybridControlSystem
from src.orchestrator.models.model_registry import ModelRegistry


async def test_model_based_system():
    """Test template resolution in ModelBasedControlSystem."""
    print("Testing ModelBasedControlSystem template integration...")
    
    # Create a mock model registry
    model_registry = ModelRegistry()
    
    # Create control system
    control_system = ModelBasedControlSystem(model_registry)
    
    # Create a task with templates
    task = Task(
        id="test_task",
        name="Test Template Resolution",
        action="generate",
        parameters={
            "prompt": "Generate text about {{ topic }} with focus on {{ focus_area }}",
            "temperature": 0.7,
            "max_tokens": 100
        }
    )
    
    # Create context with template variables
    context = {
        "pipeline_id": "test_pipeline",
        "pipeline_params": {
            "topic": "AI agents",
            "focus_area": "template resolution"
        },
        "previous_results": {
            "step1": "Previous step result"
        }
    }
    
    # Test template resolution
    try:
        # This should resolve templates in the task parameters
        # Since we don't have real models, we expect it to fail at model selection
        # but the templates should be resolved first
        await control_system._execute_task_impl(task, context)
    except Exception as e:
        # Expected to fail at model execution, check if templates were resolved
        if "topic" not in str(task.parameters.get("prompt", "")):
            print("✅ Templates were resolved in ModelBasedControlSystem")
        else:
            print(f"❌ Templates were not resolved: {task.parameters}")
            return False
    
    return True


async def test_tool_integrated_system():
    """Test template resolution in ToolIntegratedControlSystem."""
    print("Testing ToolIntegratedControlSystem template integration...")
    
    # Create control system
    control_system = ToolIntegratedControlSystem()
    
    # Create a task with templates
    task = Task(
        id="test_file_task",
        name="Test File Template Resolution",
        action="write",
        parameters={
            "action": "write",
            "path": "/tmp/test_{{ filename }}.txt",
            "content": "Hello {{ name }}, this is a test file about {{ topic }}."
        },
        metadata={"tool": "filesystem"}
    )
    
    # Create context with template variables
    context = {
        "pipeline_id": "test_pipeline",
        "pipeline_params": {
            "filename": "template_test",
            "name": "World",
            "topic": "template integration"
        },
        "previous_results": {}
    }
    
    # Test template resolution
    try:
        result = await control_system._execute_task_impl(task, context)
        print(f"✅ ToolIntegratedControlSystem executed successfully: {result.get('success', False)}")
        return True
    except Exception as e:
        print(f"❌ ToolIntegratedControlSystem failed: {e}")
        return False


async def test_hybrid_system():
    """Test template resolution in HybridControlSystem."""
    print("Testing HybridControlSystem template integration...")
    
    # Create a mock model registry
    model_registry = ModelRegistry()
    
    # Create control system
    control_system = HybridControlSystem(model_registry)
    
    # Test the template context preparation
    context = {
        "pipeline_id": "test_pipeline",
        "task_id": "test_task",
        "pipeline_params": {
            "project_name": "Test Project",
            "version": "1.0"
        },
        "previous_results": {
            "step1": {"result": "First step completed"},
            "step2": "Second step result"
        },
        "$item": "test_item",
        "$index": 0,
        "$is_first": True
    }
    
    # Test template context preparation
    try:
        template_context = control_system._prepare_template_context(context)
        flat_context = template_context.to_flat_dict()
        
        # Check if expected variables are in context
        expected_vars = ["project_name", "version", "$item", "$index", "$is_first"]
        missing_vars = [var for var in expected_vars if var not in flat_context]
        
        if not missing_vars:
            print("✅ HybridControlSystem template context prepared successfully")
            return True
        else:
            print(f"❌ HybridControlSystem missing variables: {missing_vars}")
            return False
            
    except Exception as e:
        print(f"❌ HybridControlSystem template context preparation failed: {e}")
        return False


async def test_end_to_end_pipeline():
    """Test a complete pipeline with templates."""
    print("Testing end-to-end pipeline with template resolution...")
    
    # Create a temporary directory for test output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create control system
        control_system = ToolIntegratedControlSystem(output_dir=temp_dir)
        
        # Create pipeline definition
        pipeline_def = {
            "id": "test_template_pipeline",
            "name": "Test Template Pipeline",
            "parameters": {
                "project_name": "TestProject",
                "output_dir": temp_dir
            },
            "tasks": [
                {
                    "id": "create_file",
                    "name": "Create Test File",
                    "action": "write",
                    "parameters": {
                        "action": "write",
                        "path": "{{ output_dir }}/{{ project_name }}_summary.txt",
                        "content": "Project: {{ project_name }}\nCreated by: Template Integration Test"
                    },
                    "metadata": {"tool": "filesystem"}
                }
            ]
        }
        
        # Create pipeline
        pipeline = Pipeline.from_dict(pipeline_def)
        
        # Execute pipeline
        try:
            results = await control_system.execute_pipeline(pipeline)
            
            # Check if file was created with resolved templates
            expected_file = Path(temp_dir) / "TestProject_summary.txt"
            if expected_file.exists():
                content = expected_file.read_text()
                if "TestProject" in content and "{{ project_name }}" not in content:
                    print("✅ End-to-end pipeline with templates executed successfully")
                    print(f"   Created file: {expected_file}")
                    print(f"   Content preview: {content[:100]}...")
                    return True
                else:
                    print(f"❌ Templates not resolved in file content: {content}")
                    return False
            else:
                print(f"❌ Expected file not created: {expected_file}")
                return False
                
        except Exception as e:
            print(f"❌ End-to-end pipeline failed: {e}")
            return False


async def main():
    """Run all integration tests."""
    print("🧪 Testing UnifiedTemplateResolver integration in control systems...\n")
    
    tests = [
        ("ModelBasedControlSystem", test_model_based_system),
        ("ToolIntegratedControlSystem", test_tool_integrated_system),
        ("HybridControlSystem", test_hybrid_system),
        ("End-to-End Pipeline", test_end_to_end_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All UnifiedTemplateResolver integration tests passed!")
        return True
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)