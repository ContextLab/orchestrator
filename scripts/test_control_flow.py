#!/usr/bin/env python3
"""Test control flow with simple data."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator import Orchestrator, init_models

async def test_control_flow():
    """Test control flow loop."""
    # Initialize models
    print("Initializing models...")
    init_models()
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Simple test pipeline with control flow
    yaml_content = """
id: test-control-flow
name: Test Control Flow
description: Test control flow with loop

parameters:
  items:
    type: array
    default: ["apple", "banana", "cherry"]
  output_dir:
    type: string
    default: "test_output"

steps:
  - id: create_dir
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_dir }}/.gitkeep"
      content: "Directory created"
  
  - id: process_items
    for_each: "{{ items }}"
    steps:
      - id: process_item
        action: generate_text
        parameters:
          prompt: "Write a haiku about {{ $item }}"
          model: openai/gpt-4o-mini
          max_tokens: 50
      
      - id: save_item
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_dir }}/{{ $item }}.txt"
          content: |
            # {{ $item | upper }}
            
            Item index: {{ $index }}
            Is first: {{ $is_first }}
            Is last: {{ $is_last }}
            
            ## Haiku
            {{ process_item.result }}
        dependencies:
          - process_item
    dependencies:
      - create_dir
  
  - id: create_summary
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_dir }}/summary.md"
      content: |
        # Summary
        
        Total items: {{ items | length }}
        
        ## Items processed:
        {% for item in items %}
        - {{ item }}
        {% endfor %}
        
        Generated at: {{ execution.timestamp }}
    dependencies:
      - process_items

outputs:
  items_processed: "{{ items | length }}"
  output_dir: "{{ output_dir }}"
"""
    
    # Run pipeline
    inputs = {"items": ["rose", "moon", "ocean"], "output_dir": "test_results"}
    
    try:
        results = await orchestrator.execute_yaml(yaml_content, inputs)
        print("\n✅ Pipeline executed successfully!")
        print(f"Outputs: {results.get('outputs', {})}")
        
        # Check created files
        output_files = list(Path("test_results").glob("*.txt"))
        summary = Path("test_results/summary.md")
        
        print(f"\nFiles created: {len(output_files)} items + summary")
        
        if summary.exists():
            print(f"\nSummary content:\n{summary.read_text()}")
        
        # Show one item file
        if output_files:
            print(f"\nExample item ({output_files[0].name}):\n{output_files[0].read_text()}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_control_flow())