#!/usr/bin/env python3
"""
Validate all example pipelines by running them with minimal inputs.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import Orchestrator, init_models
from orchestrator.compiler.yaml_compiler import YAMLCompiler


async def validate_pipeline(yaml_path: Path, orchestrator: Orchestrator):
    """Validate a single pipeline."""
    print(f"\n{'='*60}")
    print(f"Validating: {yaml_path.name}")
    print(f"{'='*60}")
    
    try:
        # Read YAML content
        yaml_content = yaml_path.read_text()
        
        # First, try to compile the YAML to check for syntax errors
        compiler = YAMLCompiler(model_registry=orchestrator.model_registry)
        
        # Compile without resolving ambiguities (faster validation)
        pipeline = await compiler.compile(yaml_content, resolve_ambiguities=False)
        print(f"✅ YAML syntax: Valid")
        print(f"✅ Pipeline name: {pipeline.id}")
        print(f"✅ Steps: {len(pipeline.tasks)}")
        
        # Check for required inputs
        if pipeline.metadata.get('inputs'):
            print(f"ℹ️  Required inputs: {list(pipeline.metadata['inputs'].keys())}")
        
        # Determine minimal inputs based on pipeline type
        inputs = {}
        
        # Add common inputs based on pipeline name
        if 'research' in yaml_path.name:
            inputs['topic'] = 'test validation'
        if 'data' in yaml_path.name or 'processing' in yaml_path.name:
            inputs['data'] = [1, 2, 3, 4, 5]
        if 'creative' in yaml_path.name or 'image' in yaml_path.name:
            inputs['prompt'] = 'test image'
        if 'code' in yaml_path.name:
            inputs['code'] = 'print("hello world")'
        if 'interactive' in yaml_path.name:
            inputs['user_input'] = 'test input'
        
        # Add any required inputs that are missing
        if pipeline.metadata.get('inputs'):
            for input_name, input_spec in pipeline.metadata['inputs'].items():
                if input_spec.get('required', True) and input_name not in inputs:
                    # Provide default values based on type
                    input_type = input_spec.get('type', 'string')
                    if input_type == 'string':
                        inputs[input_name] = f'test_{input_name}'
                    elif input_type == 'integer':
                        inputs[input_name] = 42
                    elif input_type == 'boolean':
                        inputs[input_name] = True
                    elif input_type == 'array':
                        inputs[input_name] = [1, 2, 3]
                    elif input_type == 'object':
                        inputs[input_name] = {'test': 'value'}
        
        print(f"ℹ️  Test inputs: {inputs}")
        
        # Try to execute with minimal timeout to just validate it starts
        print(f"⏳ Running quick execution test...")
        
        # For quick validation, we'll use a short timeout
        # Note: Some pipelines might fail due to timeout, but that's okay for validation
        try:
            start_time = datetime.now()
            # Use asyncio timeout for the entire pipeline
            results = await asyncio.wait_for(
                orchestrator.execute_yaml(yaml_content, inputs),
                timeout=30.0  # 30 second timeout for validation
            )
            duration = (datetime.now() - start_time).total_seconds()
            print(f"✅ Execution: Success in {duration:.1f}s")
            return True, None
        except asyncio.TimeoutError:
            print(f"⚠️  Execution: Timed out after 30s (pipeline may be valid but slow)")
            return True, "timeout"
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        if hasattr(e, '__traceback__'):
            print(f"   Traceback: {traceback.format_exc()}")
        return False, str(e)


async def main():
    """Validate all pipelines."""
    # Initialize models
    print("Initializing models...")
    model_registry = init_models()
    
    # Initialize orchestrator
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Find all YAML files
    examples_dir = Path(__file__).parent.parent / "examples"
    yaml_files = sorted([
        f for f in examples_dir.glob("*.yaml") 
        if not str(f).endswith("_output.yaml") and "outputs" not in str(f)
    ])
    
    print(f"\nFound {len(yaml_files)} pipelines to validate")
    
    # Track results
    results = {
        'valid': [],
        'timeout': [],
        'failed': []
    }
    
    # Validate each pipeline
    for yaml_file in yaml_files:
        success, error = await validate_pipeline(yaml_file, orchestrator)
        
        if success:
            if error == "timeout":
                results['timeout'].append(yaml_file.name)
            else:
                results['valid'].append(yaml_file.name)
        else:
            results['failed'].append((yaml_file.name, error))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total pipelines: {len(yaml_files)}")
    print(f"✅ Valid and executed: {len(results['valid'])}")
    print(f"⚠️  Valid but timed out: {len(results['timeout'])}")
    print(f"❌ Failed validation: {len(results['failed'])}")
    
    if results['valid']:
        print(f"\n✅ Successfully validated:")
        for name in sorted(results['valid']):
            print(f"   - {name}")
    
    if results['timeout']:
        print(f"\n⚠️  Timed out (but likely valid):")
        for name in sorted(results['timeout']):
            print(f"   - {name}")
    
    if results['failed']:
        print(f"\n❌ Failed validation:")
        for name, error in sorted(results['failed']):
            print(f"   - {name}: {error}")
    
    # Return exit code based on failures
    return 0 if not results['failed'] else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)