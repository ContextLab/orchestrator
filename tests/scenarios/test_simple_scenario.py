"""
Simple scenario test to verify our test framework works.
"""

import pytest
import tempfile
from pathlib import Path
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator import init_models


@pytest.mark.asyncio
async def test_simple_pipeline_execution():
    """Test a simple pipeline executes correctly."""
    
    # Create temporary directory
    test_dir = Path(tempfile.mkdtemp()) / "simple_test"
    test_dir.mkdir(exist_ok=True)
    
    # Simple test pipeline
    simple_pipeline = """
name: simple_test_pipeline
version: "1.0.0"
description: "Simple test to verify orchestrator works"

steps:
  - id: simple_step
    action: python_code
    parameters:
      code: |
        print("Hello from simple test!")
        result = "test_success"
        print(f"Result: {result}")
"""
    
    # Write pipeline to file
    pipeline_path = test_dir / "simple.yaml"
    pipeline_path.write_text(simple_pipeline)
    
    # Initialize models first
    model_registry = init_models()
    
    # Create orchestrator and execute
    orchestrator = Orchestrator(model_registry=model_registry)
    yaml_content = pipeline_path.read_text()
    result = await orchestrator.execute_yaml(yaml_content)
    
    # Basic assertions
    assert result is not None
    print(f"✓ Simple scenario test completed successfully")
    print(f"Result type: {type(result)}")
    print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_simple_pipeline_execution())