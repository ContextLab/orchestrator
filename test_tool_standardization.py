#!/usr/bin/env python3
"""Simple test to verify tool return format standardization."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator.tools.data_tools import DataProcessingTool
from orchestrator.tools.validation import ValidationTool

async def test_data_processing_tool():
    """Test DataProcessingTool with new standardized format."""
    tool = DataProcessingTool()
    
    # Test convert action
    result = await tool.execute(
        action="convert",
        data='{"name": "John", "age": 30}',
        format="json",
        output_format="json"
    )
    
    print("DataProcessingTool convert result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Error: {result.get('error')}")
    print(f"  Has result field: {'result' in result}")
    if 'result' in result:
        print(f"  Result keys: {list(result['result'].keys())}")
    print()

async def test_validation_tool():
    """Test ValidationTool with new standardized format."""
    tool = ValidationTool()
    
    # Test simple validation
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
    
    result = await tool.execute(
        action="validate",
        data={"name": "John", "age": 30},
        schema=schema
    )
    
    print("ValidationTool validate result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Error: {result.get('error')}")
    print(f"  Has result field: {'result' in result}")
    if 'result' in result:
        print(f"  Result keys: {list(result['result'].keys())}")
        print(f"  Valid: {result['result'].get('valid')}")
    print()

async def test_error_case():
    """Test error handling with new standardized format."""
    tool = DataProcessingTool()
    
    # Test with invalid action
    result = await tool.execute(action="invalid_action")
    
    print("DataProcessingTool error case:")
    print(f"  Success: {result.get('success')}")
    print(f"  Error: {result.get('error')}")
    print(f"  Result: {result.get('result')}")
    print()

async def main():
    print("Testing Tool Return Format Standardization")
    print("=" * 50)
    
    await test_data_processing_tool()
    await test_validation_tool()
    await test_error_case()
    
    print("All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())