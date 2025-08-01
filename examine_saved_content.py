#!/usr/bin/env python3
"""Examine what content was actually saved in the filesystem write step."""

import json
import gzip
from pathlib import Path

def get_checkpoint_step_result(checkpoint_file, step_id):
    """Get the result of a specific step from checkpoint."""
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    # Decompress if needed
    if checkpoint.get('state', {}).get('compressed'):
        compressed_data = checkpoint['state']['data']
        compressed_bytes = bytes.fromhex(compressed_data)
        decompressed_bytes = gzip.decompress(compressed_bytes)
        state_data = json.loads(decompressed_bytes.decode('utf-8'))
    else:
        state_data = checkpoint['state']['data']
    
    # Get task data
    tasks = state_data.get('tasks', {})
    if step_id in tasks:
        return tasks[step_id].get('result')
    
    return None

def main():
    # Find latest checkpoint for Advanced Research Tools
    checkpoint_dir = Path("checkpoints")
    checkpoints = list(checkpoint_dir.glob("Advanced Research Tools Pipeline (Fixed)_*.json"))
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not checkpoints:
        print("No checkpoints found")
        return
    
    checkpoint_file = checkpoints[0]
    print(f"Using checkpoint: {checkpoint_file.name}")
    
    # Get the save_report step result
    save_report_result = get_checkpoint_step_result(checkpoint_file, 'save_report')
    if save_report_result:
        print(f"\nsave_report result:")
        print(f"  Path: {save_report_result.get('path')}")
        print(f"  Size: {save_report_result.get('size')} bytes")
        print(f"  Success: {save_report_result.get('success')}")
    
    # Get the read_report step result to see what was actually written
    read_report_result = get_checkpoint_step_result(checkpoint_file, 'read_report')
    if read_report_result and 'content' in read_report_result:
        content = read_report_result['content']
        print(f"\nActual file content (first 500 chars):")
        print("=" * 60)
        print(content[:500])
        print("=" * 60)
        
        # Check for unrendered templates
        if "{{" in content:
            print("\n⚠️  WARNING: File contains unrendered templates!")
            # Find all template expressions
            import re
            templates = re.findall(r'{{[^}]+}}', content)
            print(f"Found {len(templates)} unrendered templates:")
            for template in templates[:10]:
                print(f"  - {template}")
            if len(templates) > 10:
                print(f"  ... and {len(templates) - 10} more")
        else:
            print("\n✅ All templates were rendered successfully!")

if __name__ == "__main__":
    main()