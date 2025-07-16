#!/usr/bin/env python3
"""Run a simple snippet test to debug issues."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up environment
os.environ.setdefault('ORCHESTRATOR_CONFIG', str(Path(__file__).parent / "config" / "orchestrator.yaml"))

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Test a simple YAML snippet
def test_yaml_snippet():
    import yaml
    
    yaml_content = """id: hello_world
name: Hello World Pipeline
description: A simple example pipeline

steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello to the world in a creative way!"

  - id: translate
    action: generate_text
    parameters:
      prompt: "Translate this greeting to Spanish: {{ greet.result }}"
    dependencies: [greet]

outputs:
  greeting: "{{ greet.result }}"
  spanish: "{{ translate.result }}"
"""
    
    try:
        data = yaml.safe_load(yaml_content)
        print("✓ YAML parsing successful")
        print(f"  Pipeline: {data['name']}")
        print(f"  Steps: {len(data['steps'])}")
        return True
    except Exception as e:
        print(f"✗ YAML parsing failed: {e}")
        return False

# Test pipeline compilation with real models
async def test_pipeline_compilation():
    from orchestrator.compiler import YAMLCompiler
    from orchestrator.models import ModelRegistry
    
    yaml_content = """id: test_pipeline
name: Test Pipeline
description: Testing compilation

steps:
  - id: test_step
    action: generate_text
    parameters:
      prompt: "Hello, world!"
"""
    
    try:
        # Parse YAML first
        import yaml
        data = yaml.safe_load(yaml_content)
        
        # Set up compiler
        compiler = YAMLCompiler()
        registry = ModelRegistry()
        
        # Initialize models
        print("Initializing models...")
        import orchestrator
        registry = orchestrator.init_models()
        compiler.set_model_registry(registry)
        
        print(f"Available models: {len(registry.models)}")
        for model_name in list(registry.models.keys())[:5]:
            print(f"  - {model_name}")
        
        # Compile pipeline
        print("\nCompiling pipeline...")
        pipeline = await compiler.compile(data)
        
        print("✓ Pipeline compilation successful")
        print(f"  Pipeline ID: {pipeline.id}")
        print(f"  Tasks: {len(pipeline.tasks)}")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Main test runner
async def main():
    print("=== Running Simple Snippet Tests ===\n")
    
    # Check environment
    print("Environment check:")
    api_keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "GOOGLE_AI_API_KEY": os.environ.get("GOOGLE_AI_API_KEY", ""),
    }
    
    for key, value in api_keys.items():
        if value:
            print(f"  {key}: {value[:10]}...{value[-4:]}")
        else:
            print(f"  {key}: NOT SET")
    
    print("\n1. Testing YAML parsing...")
    test_yaml_snippet()
    
    print("\n2. Testing pipeline compilation with real models...")
    await test_pipeline_compilation()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())