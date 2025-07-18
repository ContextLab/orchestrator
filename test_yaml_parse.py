#!/usr/bin/env python3
"""Test YAML parsing without execution."""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.compiler.auto_tag_yaml_parser import parse_yaml_with_auto_tags


async def test_yaml_parsing():
    """Test YAML parsing without execution."""
    
    # Test research assistant
    yaml_file = Path("examples") / "research_assistant.yaml"
    if not yaml_file.exists():
        print(f"File not found: {yaml_file}")
        return False
        
    print(f"Testing YAML parsing for {yaml_file.name}...")
    
    try:
        # Read YAML content
        with open(yaml_file, 'r') as f:
            yaml_content = f.read()
        
        # Parse with AUTO tags
        print("1. Parsing with AUTO tags...")
        parsed_yaml = parse_yaml_with_auto_tags(yaml_content)
        print(f"   ✓ Parsed {len(parsed_yaml.get('steps', []))} steps")
        
        # Compile to pipeline
        print("2. Compiling to pipeline...")
        compiler = YAMLCompiler()
        
        inputs = {
            "query": "What is machine learning?",
            "context": "Simple explanation for beginners",
            "max_sources": 3,
            "quality_threshold": 0.8
        }
        
        pipeline = await compiler.compile(yaml_content, inputs)
        print(f"   ✓ Compiled pipeline with {len(pipeline.tasks)} tasks")
        
        # Show task dependencies
        print("3. Task dependencies:")
        for task in pipeline.tasks.values():
            deps = task.dependencies or []
            print(f"   - {task.id}: depends on {deps}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    asyncio.run(test_yaml_parsing())