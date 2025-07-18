#!/usr/bin/env python3
"""Run pipeline with actual models and proper tool execution."""

import asyncio
import os
from pathlib import Path
from datetime import datetime
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import Orchestrator
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.integrations.anthropic_model import AnthropicModel
from orchestrator.integrations.google_model import GoogleModel
from scripts.run_pipelines_with_proper_rendering import ToolAwareControlSystem


def setup_models():
    """Set up high-quality models."""
    registry = ModelRegistry()
    
    # Register available models
    if os.getenv("ANTHROPIC_API_KEY"):
        registry.register_model(AnthropicModel(model_name="claude-sonnet-4-20250514"))
        print("✓ Anthropic Claude Sonnet 4 registered")
    
    if os.getenv("OPENAI_API_KEY"):
        registry.register_model(OpenAIModel(model_name="gpt-4.1"))
        print("✓ OpenAI GPT-4.1 registered")
    
    if os.getenv("GOOGLE_API_KEY"):
        registry.register_model(GoogleModel(model_name="gemini-2.5-flash"))
        print("✓ Google Gemini 2.5 Flash registered")
    
    if not registry.models:
        print("⚠️  No models registered. Please set API keys.")
        sys.exit(1)
    
    return registry


async def run_pipeline(yaml_file: str, inputs: dict = None, output_dir: str = None):
    """Run a single pipeline with actual models."""
    # Set up models
    registry = setup_models()
    
    # Create enhanced control system
    control_system = ToolAwareControlSystem(registry)
    
    # Initialize orchestrator with proper control system
    orchestrator = Orchestrator(
        model_registry=registry,
        control_system=control_system
    )
    
    # Ensure output directory exists
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read pipeline YAML
    yaml_path = Path(yaml_file)
    if not yaml_path.exists():
        print(f"Error: Pipeline file not found: {yaml_file}")
        return 1
    
    yaml_content = yaml_path.read_text()
    
    # Run pipeline
    print(f"\nRunning pipeline: {yaml_path.name}")
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Run pipeline with real models')
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