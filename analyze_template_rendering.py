#!/usr/bin/env python3
"""Analyze template rendering in the save_report step."""

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
    
    print(f"Analyzing checkpoint: {checkpoint_file.name}")
    checkpoint = get_checkpoint_data(checkpoint_file)
    
    # Get the save_report task details
    tasks = checkpoint['state']['data'].get('tasks', {})
    save_report = tasks.get('save_report', {})
    
    print("\n=== SAVE_REPORT TASK ANALYSIS ===")
    print(f"Status: {save_report.get('status')}")
    print(f"Action: {save_report.get('action')}")
    
    # Check parameters
    params = save_report.get('parameters', {})
    if 'content' in params:
        content = params['content']
        print(f"\nContent parameter length: {len(content)} chars")
        print("First 500 chars of content:")
        print("-" * 60)
        print(content[:500])
        print("-" * 60)
        
        # Count templates
        template_count = content.count('{{')
        print(f"\nUnrendered templates in content: {template_count}")
        
        # Find all template expressions
        import re
        templates = re.findall(r'{{[^}]+}}', content)
        print(f"\nFirst 10 template expressions found:")
        for i, template in enumerate(templates[:10]):
            print(f"  {i+1}. {template}")
    
    # Check if rendered_parameters exist
    rendered_params = save_report.get('rendered_parameters', {})
    if rendered_params:
        print(f"\nRendered parameters found: {list(rendered_params.keys())}")
        if 'content' in rendered_params:
            rendered_content = rendered_params['content']
            print(f"Rendered content length: {len(rendered_content)} chars")
            print("First 500 chars of rendered content:")
            print("-" * 60)
            print(rendered_content[:500])
            print("-" * 60)
    else:
        print("\nNo rendered_parameters found!")
    
    # Check template metadata
    template_metadata = save_report.get('template_metadata', {})
    if template_metadata:
        print(f"\nTemplate metadata found:")
        for key, meta in template_metadata.items():
            if isinstance(meta, dict):
                deps = meta.get('dependencies', [])
                print(f"  {key}: dependencies={deps}")
    
    # Check if previous results were available
    metadata = checkpoint.get('metadata', {})
    previous_results = metadata.get('previous_results', {})
    print(f"\nAvailable previous results at save_report time:")
    for step_id in ['search_topic', 'deep_search', 'analyze_findings', 'generate_recommendations']:
        if step_id in previous_results:
            result = previous_results[step_id]
            if isinstance(result, dict):
                print(f"  {step_id}: {list(result.keys())[:5]}...")
            else:
                print(f"  {step_id}: {type(result).__name__}")

if __name__ == "__main__":
    main()