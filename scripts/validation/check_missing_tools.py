#!/usr/bin/env python3
"""Check which tools referenced in pipelines are not implemented."""

import os
import sys
import yaml
from pathlib import Path
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import and register tools
from src.orchestrator.tools.base import default_registry, register_default_tools

# Make sure tools are registered
register_default_tools()

# Get actual tool names
actual_tools = set(default_registry.list_tools())

def extract_tools_from_pipeline(filepath: Path) -> set:
    """Extract all tool names referenced in a pipeline."""
    tools = set()
    
    def extract_from_step(step):
        if 'tool' in step:
            tools.add(step['tool'])
        
        # Check nested steps
        if 'steps' in step:
            for substep in step['steps']:
                extract_from_step(substep)
        
        # Check for_each steps
        if 'for_each' in step and 'steps' in step:
            for substep in step['steps']:
                extract_from_step(substep)
        
        # Check while_loop steps
        if 'while' in step and 'steps' in step:
            for substep in step['steps']:
                extract_from_step(substep)
    
    try:
        with open(filepath, 'r') as f:
            pipeline = yaml.safe_load(f)
        
        if 'steps' in pipeline:
            for step in pipeline['steps']:
                extract_from_step(step)
    except:
        pass
    
    return tools

def main():
    """Check for missing tool implementations."""
    examples_dir = Path(__file__).parent.parent / 'examples'
    
    # Collect all tools used in pipelines
    pipeline_tools = set()
    tool_usage = defaultdict(list)
    
    for yaml_file in sorted(examples_dir.glob('*.yaml')):
        tools = extract_tools_from_pipeline(yaml_file)
        pipeline_tools.update(tools)
        for tool in tools:
            tool_usage[tool].append(yaml_file.name)
    
    # Find missing tools
    missing_tools = pipeline_tools - actual_tools
    
    print("Tool Implementation Status")
    print("=" * 80)
    print()
    
    print(f"Implemented tools ({len(actual_tools)}):")
    for tool in sorted(actual_tools):
        print(f"  ✓ {tool}")
    
    print()
    print(f"Missing tools ({len(missing_tools)}):")
    for tool in sorted(missing_tools):
        pipelines = tool_usage[tool]
        print(f"  ✗ {tool}")
        print(f"    Used in: {', '.join(pipelines[:3])}" + 
              (f" and {len(pipelines)-3} more" if len(pipelines) > 3 else ""))
    
    print()
    print("Summary:")
    print(f"  Total tools referenced: {len(pipeline_tools)}")
    print(f"  Implemented: {len(pipeline_tools & actual_tools)}")
    print(f"  Missing: {len(missing_tools)}")
    
    # Check for Python executor specifically
    if 'python-executor' in missing_tools:
        print()
        print("Note: python-executor is referenced but not implemented.")
        print("      This might be handled by code execution in model actions.")

if __name__ == '__main__':
    main()