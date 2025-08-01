#!/usr/bin/env python3
"""Test how step results are structured."""

import asyncio
import logging
import sys
import json

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Set specific loggers to DEBUG
logging.getLogger('orchestrator.core.template_manager').setLevel(logging.DEBUG)

async def main():
    from src.orchestrator import Orchestrator
    from scripts.run_pipeline import init_models
    
    # Initialize models first
    await init_models()
    
    # Create a test pipeline that mimics research_advanced_tools structure
    test_yaml = """
name: Test Result Structure
description: Test how results are structured

parameters:
  topic: "test"

steps:
  - id: search_topic
    tool: web-search
    action: search
    parameters:
      query: "{{ topic }}"
      max_results: 3
      
  - id: analyze_findings
    action: generate_text
    parameters:
      prompt: "Analyze these search results"
      max_tokens: 50
    dependencies:
      - search_topic
      
  - id: save_report
    tool: filesystem  
    action: write
    parameters:
      path: "test_report.md"
      content: |
        # Test Report: {{ topic }}
        
        ## Search Results
        Total results: {{ search_topic.total_results }}
        
        ## Analysis
        {{ analyze_findings.result }}
        
        ## Results List
        {% for result in search_topic.results %}
        - {{ result.title }}
        {% endfor %}
    dependencies:
      - search_topic
      - analyze_findings
"""
    
    # Initialize orchestrator with debug templates
    orchestrator = Orchestrator(debug_templates=True)
    
    # Run the test pipeline
    result = await orchestrator.execute_yaml(
        test_yaml,
        context={"topic": "test"}
    )
    
    print("\n=== Step Results ===")
    if 'steps' in result:
        for step_id, step_result in result['steps'].items():
            print(f"\n{step_id}:")
            print(f"  Type: {type(step_result)}")
            if isinstance(step_result, dict):
                print(f"  Keys: {list(step_result.keys())}")
                if 'total_results' in step_result:
                    print(f"  total_results: {step_result['total_results']}")
                if 'results' in step_result and isinstance(step_result['results'], list):
                    print(f"  results length: {len(step_result['results'])}")
    
    # Check the output file
    from pathlib import Path
    output_file = Path("test_report.md")
    if output_file.exists():
        content = output_file.read_text()
        print(f"\n=== Report Content ===")
        print(content)
        if "{{" in content:
            print("\nERROR: Templates not rendered!")
        else:
            print("\nSUCCESS: Templates rendered correctly!")
        # Clean up
        output_file.unlink()

if __name__ == "__main__":
    asyncio.run(main())