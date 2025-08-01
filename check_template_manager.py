#!/usr/bin/env python3
"""Check if template_manager is in the context."""

import json
import gzip
from pathlib import Path

def get_checkpoint_data(checkpoint_file):
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
    # Get the latest checkpoint
    checkpoint_file = Path("checkpoints/research_advanced_tools_1754066420_1754066513.json")
    
    checkpoint = get_checkpoint_data(checkpoint_file)
    
    # Check metadata for template_manager
    metadata = checkpoint.get('metadata', {})
    print("Checkpoint metadata keys:")
    for key in sorted(metadata.keys()):
        if 'template' in key.lower() or 'manager' in key.lower():
            print(f"  * {key}")
        else:
            print(f"    {key}")
    
    # Check if any context-like field has template_manager
    print("\nSearching for template_manager in checkpoint...")
    
    def search_dict(d, path=""):
        found = []
        if isinstance(d, dict):
            for k, v in d.items():
                if 'template' in str(k).lower() and 'manager' in str(k).lower():
                    found.append(f"{path}.{k}")
                if isinstance(v, (dict, list)):
                    found.extend(search_dict(v, f"{path}.{k}"))
        elif isinstance(d, list):
            for i, item in enumerate(d):
                if isinstance(item, (dict, list)):
                    found.extend(search_dict(item, f"{path}[{i}]"))
        return found
    
    found_paths = search_dict(checkpoint)
    if found_paths:
        print("Found template_manager at:")
        for path in found_paths:
            print(f"  {path}")
    else:
        print("No template_manager found in checkpoint!")
    
    # Check the filesystem write call in logs
    print("\n=== FILESYSTEM TOOL INVOCATION ===")
    # Look for any debug info
    state_data = checkpoint.get('state', {}).get('data', {})
    if 'debug_info' in state_data:
        print("Debug info found:", state_data['debug_info'])

if __name__ == "__main__":
    main()