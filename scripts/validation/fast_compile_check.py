#!/usr/bin/env python3
"""Fast compilation check for all pipelines - just checks if they compile without errors."""

import asyncio
import sys
import os
from pathlib import Path
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import init_models
from orchestrator.compiler.yaml_compiler import YAMLCompiler

async def check_compilation():
    """Check if all pipelines compile successfully."""
    
    # Initialize models once
    print("Initializing models...")
    model_registry = init_models()
    
    if not model_registry or not model_registry.models:
        print("❌ No models available")
        return
    
    # Priority pipelines to check
    priority_pipelines = [
        "auto_tags_demo.yaml",
        "control_flow_advanced.yaml",
        "control_flow_conditional.yaml",
        "control_flow_dynamic.yaml",
        "creative_image_pipeline.yaml",
        "data_processing.yaml",
        "data_processing_pipeline.yaml",
        "interactive_pipeline.yaml",
        "llm_routing_pipeline.yaml",
        "mcp_integration_pipeline.yaml",
        "mcp_memory_workflow.yaml",
        "model_routing_demo.yaml",
        "modular_analysis_pipeline.yaml",
        "multimodal_processing.yaml",
        "recursive_data_processing.yaml",
        "research_minimal.yaml",
        "simple_data_processing.yaml",
        "simple_timeout_test.yaml",
        "statistical_analysis.yaml",
        "terminal_automation.yaml",
        "test_timeout.yaml",
        "test_timeout_websearch.yaml",
        "test_validation_pipeline.yaml",
        "validation_pipeline.yaml",
        "web_research_pipeline.yaml",
        "working_web_search.yaml"
    ]
    
    examples_dir = Path("examples")
    results = {}
    
    print(f"\nChecking {len(priority_pipelines)} pipelines...")
    print("=" * 60)
    
    for pipeline_name in priority_pipelines:
        pipeline_path = examples_dir / pipeline_name
        
        if not pipeline_path.exists():
            results[pipeline_name] = "NOT_FOUND"
            print(f"❌ {pipeline_name}: File not found")
            continue
        
        try:
            start = time.time()
            
            # Read and compile
            with open(pipeline_path) as f:
                yaml_content = f.read()
            
            compiler = YAMLCompiler(development_mode=True)
            pipeline = await compiler.compile(yaml_content)
            
            elapsed = time.time() - start
            
            # Check for common issues in the compiled pipeline
            issues = []
            
            # Check if pipeline has tasks
            if not pipeline.tasks:
                issues.append("No tasks")
            
            # Check for undefined tools
            for task_id, task in pipeline.tasks.items():
                if hasattr(task, 'tool') and task.tool and '{{' in str(task.tool):
                    issues.append(f"Unrendered tool in {task_id}")
                if hasattr(task, 'parameters'):
                    params_str = str(task.parameters)
                    if '{{' in params_str and '}}' in params_str:
                        if '$item' in params_str or '$index' in params_str:
                            issues.append(f"Unrendered loop vars in {task_id}")
            
            if issues:
                results[pipeline_name] = f"COMPILED_WITH_ISSUES: {', '.join(issues[:2])}"
                print(f"⚠️  {pipeline_name}: Compiled in {elapsed:.1f}s but has issues: {', '.join(issues[:2])}")
            else:
                results[pipeline_name] = "SUCCESS"
                print(f"✅ {pipeline_name}: Compiled successfully in {elapsed:.1f}s")
                
        except Exception as e:
            results[pipeline_name] = f"COMPILE_ERROR: {str(e)[:50]}"
            print(f"❌ {pipeline_name}: Compilation failed - {str(e)[:50]}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    success = sum(1 for r in results.values() if r == "SUCCESS")
    with_issues = sum(1 for r in results.values() if "WITH_ISSUES" in str(r))
    failed = sum(1 for r in results.values() if "ERROR" in str(r) or r == "NOT_FOUND")
    
    print(f"Total: {len(results)}")
    print(f"✅ Success: {success}")
    print(f"⚠️  With Issues: {with_issues}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print("\nFailed pipelines:")
        for name, result in results.items():
            if "ERROR" in str(result) or result == "NOT_FOUND":
                print(f"  - {name}: {result}")

if __name__ == "__main__":
    asyncio.run(check_compilation())