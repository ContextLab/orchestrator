#!/usr/bin/env python3
"""
Simple pipeline runner - the main script for executing pipelines.
"""

import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import Orchestrator, init_models


def check_output_path_usage(yaml_content: str, output_dir: str) -> bool:
    """
    Check if pipeline uses output_path parameter and will respect -o flag.
    
    Args:
        yaml_content: YAML pipeline content
        output_dir: Output directory specified with -o flag
        
    Returns:
        True if pipeline will respect -o flag, False if it uses hardcoded paths
    """
    if not output_dir:
        return True  # No -o flag used, so no issue
    
    # Check if YAML contains {{ output_path }} usage
    if '{{ output_path }}' in yaml_content or '{{output_path}}' in yaml_content:
        return True
    
    # Check for hardcoded output patterns that suggest -o flag will be ignored
    hardcoded_patterns = [
        'examples/outputs/',
        'outputs/',
        'data/outputs/',
        'results/'
    ]
    
    for pattern in hardcoded_patterns:
        if pattern in yaml_content:
            return False
    
    return True


async def run_pipeline(yaml_file: str, inputs: dict = None, output_dir: str = None):
    """Run a single pipeline."""
    # Read pipeline YAML first
    yaml_path = Path(yaml_file)
    if not yaml_path.exists():
        print(f"Error: Pipeline file not found: {yaml_file}")
        return 1
    
    yaml_content = yaml_path.read_text()
    
    # Check if pipeline will respect -o flag and warn if not
    if output_dir:
        will_respect = check_output_path_usage(yaml_content, output_dir)
        if not will_respect:
            print(f"⚠️  Warning: This pipeline may not respect the -o flag.", flush=True)
            print(f"   Pipeline uses hardcoded output paths instead of {{ output_path }} parameter.", flush=True)
            print(f"   Files may be saved to default locations instead of: {output_dir}", flush=True)
            print(flush=True)
    
    # Initialize models
    print("Initializing models...")
    model_registry = init_models()
    
    # Initialize orchestrator with models
    orchestrator = Orchestrator(model_registry=model_registry)
    
    # Set output directory if specified
    if output_dir:
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # Add output_path to inputs if not already specified
        if 'output_path' not in inputs:
            inputs['output_path'] = output_dir
    
    # Run pipeline
    print(f"Running pipeline: {yaml_path.name}")
    print(f"Inputs: {json.dumps(inputs or {}, indent=2)}")
    
    try:
        start_time = datetime.now()
        
        results = await orchestrator.execute_yaml(yaml_content, inputs or {})
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ Pipeline completed in {duration:.1f} seconds")
        
        # Print results summary
        if results:
            print("\nResults:")
            for task_id, result in results.items():
                if isinstance(result, dict) and 'filepath' in result:
                    print(f"  {task_id}: Saved to {result['filepath']}")
                elif isinstance(result, str) and len(result) > 100:
                    print(f"  {task_id}: {result[:100]}...")
                else:
                    print(f"  {task_id}: {result}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run an Orchestrator pipeline')
    parser.add_argument('pipeline', help='Path to YAML pipeline file')
    parser.add_argument('-i', '--input', action='append', help='Input key=value pairs')
    parser.add_argument('-f', '--input-file', help='JSON file with inputs')
    parser.add_argument('-o', '--output-dir', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Parse inputs
    inputs = {}
    
    # From command line key=value pairs
    if args.input:
        for item in args.input:
            if '=' in item:
                key, value = item.split('=', 1)
                # Try to parse as JSON, otherwise treat as string
                try:
                    inputs[key] = json.loads(value)
                except json.JSONDecodeError:
                    inputs[key] = value
    
    # From JSON file
    if args.input_file:
        with open(args.input_file, 'r') as f:
            file_inputs = json.load(f)
            inputs.update(file_inputs)
    
    # Run pipeline
    exit_code = asyncio.run(run_pipeline(args.pipeline, inputs, args.output_dir))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()