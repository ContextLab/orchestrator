#!/usr/bin/env python3
"""Trace template rendering flow in the filesystem write."""

import json
import gzip
from pathlib import Path

def get_full_checkpoint_data(checkpoint_file):
    """Get full decompressed checkpoint data."""
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    # Decompress if needed
    if checkpoint.get('state', {}).get('compressed'):
        compressed_data = checkpoint['state']['data']
        compressed_bytes = bytes.fromhex(compressed_data)
        decompressed_bytes = gzip.decompress(compressed_bytes)
        state_data = json.loads(decompressed_bytes.decode('utf-8'))
        checkpoint['state']['data'] = state_data
    
    return checkpoint

def main():
    # Find latest checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoints = list(checkpoint_dir.glob("Advanced Research Tools Pipeline (Fixed)_*.json"))
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not checkpoints:
        print("No checkpoints found")
        return
    
    checkpoint_file = checkpoints[0]
    print(f"Using checkpoint: {checkpoint_file.name}")
    
    checkpoint = get_full_checkpoint_data(checkpoint_file)
    
    # Check template_manager in metadata
    metadata = checkpoint.get('metadata', {})
    print(f"\nCheckpoint metadata keys: {list(metadata.keys())}")
    
    # Check if template_manager was in context
    if 'template_manager' in metadata:
        print("✓ template_manager found in metadata")
    else:
        print("✗ template_manager NOT in metadata")
    
    # Check the save_report task details
    tasks = checkpoint['state']['data'].get('tasks', {})
    save_report_task = tasks.get('save_report', {})
    
    print(f"\nsave_report task details:")
    print(f"  Status: {save_report_task.get('status')}")
    print(f"  Action: {save_report_task.get('action')}")
    print(f"  Tool metadata: {save_report_task.get('metadata', {}).get('tool')}")
    
    # Check parameters
    params = save_report_task.get('parameters', {})
    if 'content' in params:
        content_preview = params['content'][:200] + "..."
        print(f"  Content parameter (first 200 chars): {content_preview}")
        print(f"  Content has templates: {('{{' in params['content'])}")
    
    # Check template_metadata
    template_metadata = save_report_task.get('template_metadata', {})
    if template_metadata:
        print(f"\n  Template metadata found:")
        for key, meta in template_metadata.items():
            if isinstance(meta, dict):
                print(f"    {key}: dependencies={meta.get('dependencies', [])}")
    else:
        print(f"\n  No template metadata found")
    
    # Check previous_results
    previous_results = metadata.get('previous_results', {})
    print(f"\nAvailable previous results: {list(previous_results.keys())}")
    
    # Check if results have the expected structure
    if 'search_topic' in previous_results:
        st = previous_results['search_topic']
        print(f"\nsearch_topic structure:")
        print(f"  Type: {type(st)}")
        if isinstance(st, dict):
            print(f"  Keys: {list(st.keys())[:10]}")
            if 'total_results' in st:
                print(f"  total_results: {st['total_results']}")
    
    if 'analyze_findings' in previous_results:
        af = previous_results['analyze_findings']
        print(f"\nanalyze_findings structure:")
        print(f"  Type: {type(af)}")
        if isinstance(af, str):
            print(f"  Length: {len(af)} chars")
            print(f"  Preview: {af[:100]}...")

if __name__ == "__main__":
    main()