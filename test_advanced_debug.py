#!/usr/bin/env python3
"""Debug advanced pipeline template rendering."""

import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Ensure all loggers are at INFO level
logging.getLogger('orchestrator.tools.system_tools').setLevel(logging.INFO)
logging.getLogger('orchestrator.control_flow.conditional').setLevel(logging.INFO)
logging.getLogger('orchestrator.control_flow.auto_resolver').setLevel(logging.INFO)

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
        
        # Check save_report step
        if "save_report" in steps:
            print(f"\nsave_report result: {steps['save_report']}")
            
            # Check if report has unrendered templates
            report_path = steps["save_report"].get("path", "")
            if report_path and Path(report_path).exists():
                content = Path(report_path).read_text()
                if "{{" in content:
                    print(f"\nWARNING: Report contains unrendered templates!")
                    print(f"First 500 chars:\n{content[:500]}")
                else:
                    print(f"\nSUCCESS: All templates rendered correctly!")

if __name__ == "__main__":
    asyncio.run(main())