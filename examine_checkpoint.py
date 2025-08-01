#!/usr/bin/env python3
"""Examine checkpoint files to understand pipeline execution."""

import json
import gzip
import base64
import sys
from pathlib import Path
from datetime import datetime

def decompress_checkpoint_data(checkpoint_file):
    """Load and decompress checkpoint data."""
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    if checkpoint.get('state', {}).get('compressed'):
        # Decode hex string and decompress
        compressed_data = checkpoint['state']['data']
        compressed_bytes = bytes.fromhex(compressed_data)
        decompressed_bytes = gzip.decompress(compressed_bytes)
        state_data = json.loads(decompressed_bytes.decode('utf-8'))
        checkpoint['state']['data'] = state_data
        checkpoint['state']['compressed'] = False
    
    return checkpoint

def analyze_checkpoint(checkpoint_file):
    """Analyze a checkpoint file and print useful information."""
    print(f"\n=== Analyzing: {checkpoint_file.name} ===")
    
    checkpoint = decompress_checkpoint_data(checkpoint_file)
    
    # Basic info
    print(f"Pipeline: {checkpoint['metadata']['pipeline_id']}")
    print(f"Execution ID: {checkpoint['execution_id']}")
    print(f"Timestamp: {datetime.fromtimestamp(checkpoint['timestamp'])}")
    
    # Pipeline context
    if 'pipeline_context' in checkpoint['metadata']:
        print(f"\nPipeline Context:")
        for key, value in checkpoint['metadata']['pipeline_context'].items():
            if key != 'all_step_ids':
                print(f"  {key}: {value}")
    
    # Pipeline state
    state = checkpoint['state']['data']
    if 'tasks' in state:
        print(f"\nTasks ({len(state['tasks'])} total):")
        for task_id, task_data in state['tasks'].items():
            status = task_data.get('status', 'unknown')
            print(f"  {task_id}: {status}")
            
            # Show results for completed tasks
            if status == 'completed' and 'result' in task_data:
                result = task_data['result']
                if isinstance(result, dict):
                    print(f"    Result keys: {list(result.keys())}")
                    if 'error' in result:
                        print(f"    Error: {result['error']}")
                elif isinstance(result, str) and len(result) > 100:
                    print(f"    Result: {result[:100]}...")
                else:
                    print(f"    Result: {result}")
    
    # Previous results
    if 'previous_results' in checkpoint['metadata']:
        print(f"\nPrevious Results:")
        for step_id in list(checkpoint['metadata']['previous_results'].keys())[:5]:
            print(f"  {step_id}: [data available]")
        if len(checkpoint['metadata']['previous_results']) > 5:
            print(f"  ... and {len(checkpoint['metadata']['previous_results']) - 5} more")

def find_latest_checkpoint(pipeline_name=None):
    """Find the latest checkpoint file."""
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("No checkpoints directory found")
        return None
    
    checkpoints = list(checkpoint_dir.glob("*.json"))
    if pipeline_name:
        checkpoints = [c for c in checkpoints if pipeline_name in c.name]
    
    if not checkpoints:
        print(f"No checkpoints found{' for ' + pipeline_name if pipeline_name else ''}")
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoints[0]

def main():
    if len(sys.argv) > 1:
        checkpoint_file = Path(sys.argv[1])
    else:
        # Find latest checkpoint
        checkpoint_file = find_latest_checkpoint("Advanced Research Tools")
        if not checkpoint_file:
            checkpoint_file = find_latest_checkpoint()
    
    if checkpoint_file and checkpoint_file.exists():
        analyze_checkpoint(checkpoint_file)
    else:
        print("No checkpoint file found")

if __name__ == "__main__":
    main()