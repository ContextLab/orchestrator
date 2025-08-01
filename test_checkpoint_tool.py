#!/usr/bin/env python3
"""Test the checkpoint extraction tool."""

import asyncio
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator.tools.checkpoint_tool import CheckpointTool

async def test_checkpoint_tool():
    """Test checkpoint tool functionality."""
    tool = CheckpointTool()
    
    print("=" * 60)
    print("Testing Checkpoint Tool")
    print("=" * 60)
    
    # Test listing checkpoints
    print("\n1. Listing checkpoints:")
    result = await tool.execute(action="list")
    if "error" not in result:
        print(f"   Total checkpoints: {result['total']}")
        print(f"   Showing: {result['showing']}")
        if result['checkpoints']:
            for cp in result['checkpoints'][:3]:
                print(f"   - {cp['file']} ({cp['pipeline']}, {cp['timestamp']})")
    else:
        print(f"   Error: {result['error']}")
    
    # Test inspecting a checkpoint
    print("\n2. Inspecting latest checkpoint:")
    result = await tool.execute(action="inspect")
    if "error" not in result:
        print(f"   Pipeline: {result['pipeline_id']}")
        print(f"   Execution ID: {result['execution_id']}")
        print(f"   Timestamp: {result['timestamp']}")
        print(f"   Total tasks: {result['total_tasks']}")
        print(f"   Task summary:")
        for task_id, summary in list(result['task_summary'].items())[:5]:
            print(f"     - {task_id}: {summary['status']}")
    else:
        print(f"   Error: {result['error']}")
    
    # Test extracting to markdown
    print("\n3. Extracting checkpoint to markdown:")
    result = await tool.execute(
        action="extract",
        output_format="markdown",
        output_file="checkpoint_report.md"
    )
    if result.get('success'):
        print(f"   ✓ Saved to: {result['output_file']}")
        print(f"   Format: {result['format']}")
        print(f"   Size: {result['size']} bytes")
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Test extracting to yaml
    print("\n4. Extracting checkpoint to yaml:")
    result = await tool.execute(
        action="extract",
        output_format="yaml",
        output_file="checkpoint_report.yaml"
    )
    if result.get('success'):
        print(f"   ✓ Saved to: {result['output_file']}")
        print(f"   Format: {result['format']}")
        print(f"   Size: {result['size']} bytes")
    else:
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Test filtering by pipeline name
    print("\n5. Listing checkpoints for 'Advanced Research Tools' pipeline:")
    result = await tool.execute(
        action="list",
        pipeline_name="Advanced Research Tools"
    )
    if "error" not in result:
        print(f"   Found {result['total']} checkpoints")
        if result['checkpoints']:
            for cp in result['checkpoints'][:3]:
                print(f"   - {cp['file']} ({cp['timestamp']})")
    else:
        print(f"   Error: {result['error']}")

if __name__ == "__main__":
    asyncio.run(test_checkpoint_tool())