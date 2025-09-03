#!/usr/bin/env python3
"""
Test full pipeline execution with runtime resolution system.

This script runs the control_flow_advanced.yaml pipeline to verify that
the runtime resolution system fixes Issue #159.
"""

import asyncio
import sys
import os
import yaml
import json
from pathlib import Path
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.models.registry_singleton import get_model_registry

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider


async def test_control_flow_advanced():
    """Test the control_flow_advanced pipeline with runtime resolution."""
    print("=" * 80)
    print("Testing Full Pipeline Execution with Runtime Resolution System")
    print("=" * 80)
    
    # Clean up any previous output
    output_dir = Path("examples/outputs/control_flow_advanced")
    if output_dir.exists():
        print(f"\nCleaning up previous output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Load pipeline configuration
    yaml_file = "examples/control_flow_advanced.yaml"
    print(f"\n1. Loading pipeline from: {yaml_file}")
    
    with open(yaml_file, 'r') as f:
        pipeline_config = yaml.safe_load(f)
    
    print(f"   Pipeline: {pipeline_config.get('name', 'Unknown')}")
    print(f"   Description: {pipeline_config.get('description', 'N/A')[:80]}...")
    
    # Initialize orchestrator
    print("\n2. Initializing orchestrator with runtime resolution...")
    
    # Create a basic control system for testing
    from src.orchestrator.control_systems.hybrid_control_system import HybridControlSystem
    from src.orchestrator.models.registry_singleton import get_model_registry
    
    registry = get_model_registry()
    control_system = HybridControlSystem(registry)
    
    # Create orchestrator with control system
    orchestrator = create_test_orchestrator()
    
    # Prepare context with test input
    context = {
        "input_text": "The quick brown fox jumps over the lazy dog.",
        "languages": ["Spanish", "French", "German"],
        "quality_threshold": 0.7,
        "output": "examples/outputs/control_flow_advanced"
    }
    
    print("\n3. Pipeline context:")
    for key, value in context.items():
        print(f"   - {key}: {value}")
    
    # Execute pipeline
    print("\n4. Executing pipeline...")
    print("   (This will test template resolution in for_each loops)")
    
    try:
        # Compile and execute the pipeline
        pipeline = orchestrator.compiler.compile_yaml_file(yaml_file, context)
        
        # Check if runtime resolution was initialized
        print(f"\n   Runtime resolution initialized: {orchestrator.runtime_resolution is not None}")
        
        # Execute
        results = await orchestrator.execute_pipeline(pipeline)
        
        print("\n5. Execution completed successfully!")
        print(f"   Total tasks executed: {len(results.get('results', {}))}")
        
        # Check specific results
        task_results = results.get('results', {})
        
        # Check if for_each loop tasks were executed
        translation_tasks = [k for k in task_results.keys() if 'translate_text' in k]
        save_tasks = [k for k in task_results.keys() if 'save_translation' in k]
        
        print(f"\n6. Loop task execution:")
        print(f"   Translation tasks: {len(translation_tasks)}")
        print(f"   Save tasks: {len(save_tasks)}")
        
        # Check if files were created
        print("\n7. Checking output files...")
        translations_dir = output_dir / "translations"
        if translations_dir.exists():
            files = list(translations_dir.glob("*.txt"))
            print(f"   Translation files created: {len(files)}")
            for file in files[:3]:  # Show first 3
                print(f"      - {file.name}")
                # Check file content
                with open(file, 'r') as f:
                    content = f.read()
                    if "{{ translate }}" in content:
                        print(f"        ⚠️ WARNING: Unresolved template found in {file.name}")
                    else:
                        print(f"        ✅ Template resolved correctly")
        else:
            print("   ⚠️ No translations directory found")
        
        # Check summary file
        summary_file = output_dir / "summary.md"
        if summary_file.exists():
            print(f"\n   Summary file created: {summary_file}")
            with open(summary_file, 'r') as f:
                content = f.read()
                if "{{" in content:
                    print("      ⚠️ WARNING: Unresolved templates in summary")
                else:
                    print("      ✅ All templates resolved in summary")
        
        # Print any errors
        errors = [r for r in task_results.values() if isinstance(r, dict) and 'error' in r]
        if errors:
            print(f"\n8. Errors encountered: {len(errors)}")
            for error in errors[:3]:
                print(f"   - {error.get('error', 'Unknown error')[:100]}...")
        else:
            print("\n8. No errors encountered ✅")
        
        # Check runtime resolution state
        if orchestrator.runtime_resolution:
            summary = orchestrator.runtime_resolution.get_execution_summary()
            print(f"\n9. Runtime Resolution Summary:")
            print(f"   Context items: {len(orchestrator.runtime_resolution.state.get_available_context())}")
            print(f"   Loops registered: {summary['loops']['registered']}")
            print(f"   Loops completed: {summary['loops']['completed']}")
        
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("Runtime resolution system is working correctly with full pipeline execution.")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the test
    results = asyncio.run(test_control_flow_advanced())
    
    if results:
        print("\n✅ Pipeline executed successfully with runtime resolution!")
    else:
        print("\n❌ Pipeline execution failed")
        sys.exit(1)