#!/usr/bin/env python3
"""Debug conditional template evaluation."""

import asyncio
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    from src.orchestrator import Orchestrator
    from src.orchestrator.models import init_models
    from pathlib import Path
    
    # Initialize models first
    await init_models()
    
    # Load YAML file
    yaml_file = Path("examples/research_advanced_tools.yaml")
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Run with minimal input
    result = await orchestrator.execute_yaml_file(
        str(yaml_file),
        context={"topic": "test"}
    )
    
    # Check specific results
    print("\n=== Results ===")
    if "steps" in result:
        steps = result["steps"]
        print(f"search_topic results: {len(steps.get('search_topic', {}).get('results', []))}")
        print(f"deep_search results: {len(steps.get('deep_search', {}).get('results', []))}")
        print(f"extract_content status: {steps.get('extract_content', {})}")
        
        # Check if report has unrendered templates
        if "save_report" in steps:
            report_path = steps["save_report"].get("path", "")
            if report_path and Path(report_path).exists():
                content = Path(report_path).read_text()
                if "{{" in content:
                    print(f"\nWARNING: Report contains unrendered templates!")
                    print(f"First 500 chars:\n{content[:500]}")

if __name__ == "__main__":
    asyncio.run(main())