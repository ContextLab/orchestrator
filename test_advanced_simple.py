#!/usr/bin/env python3
"""Test advanced pipeline with simple approach."""

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

async def main():
    from src.orchestrator import Orchestrator
    from scripts.run_pipeline import init_models
    
    # Initialize models first
    await init_models()
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # First test the minimal pipeline to confirm it works
    print("=== Testing research_minimal.yaml ===")
    result_minimal = await orchestrator.execute_yaml_file(
        "examples/research_minimal.yaml",
        context={"topic": "test"}
    )
    
    # Check minimal output
    minimal_files = list(Path("examples/outputs/research_minimal").glob("test_*.md"))
    if minimal_files:
        print(f"Minimal pipeline created: {minimal_files[0]}")
        content = minimal_files[0].read_text()
        print(f"Templates rendered: {'{{' not in content}")
    
    print("\n=== Testing research_advanced_tools.yaml ===")
    # Now test the advanced pipeline
    result_advanced = await orchestrator.execute_yaml_file(
        "examples/research_advanced_tools.yaml",
        context={"topic": "test"}
    )
    
    # Check advanced output
    advanced_files = list(Path("examples/outputs/research_advanced_tools").glob("research_test.md"))
    if advanced_files:
        print(f"Advanced pipeline created: {advanced_files[0]}")
        content = advanced_files[0].read_text()
        print(f"Templates rendered: {'{{' not in content}")
        
        # Show first few lines to see what's wrong
        lines = content.split('\n')[:10]
        print("\nFirst 10 lines:")
        for i, line in enumerate(lines):
            print(f"{i+1}: {line}")
    
    # Compare step results
    print("\n=== Step Results Comparison ===")
    if 'steps' in result_minimal:
        print("Minimal pipeline steps:", list(result_minimal['steps'].keys()))
    if 'steps' in result_advanced:
        print("Advanced pipeline steps:", list(result_advanced['steps'].keys()))
        
        # Check specific step types
        for step_id, step_result in result_advanced['steps'].items():
            if isinstance(step_result, dict):
                print(f"\n{step_id}:")
                print(f"  Type: {type(step_result)}")
                print(f"  Keys: {list(step_result.keys())[:5]}")
                if 'result' in step_result:
                    print(f"  Has .result: True")
                if 'total_results' in step_result:
                    print(f"  Has .total_results: True")

if __name__ == "__main__":
    asyncio.run(main())