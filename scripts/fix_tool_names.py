#!/usr/bin/env python3
"""Fix tool names in pipeline YAML files."""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Tool name mappings from pipelines to actual tool names
TOOL_NAME_MAPPING = {
    'filesystem': 'file_system',
    'web-search': 'web_search', 
    'headless-browser': 'headless_browser',
    'pdf-compiler': 'pdf_compiler',
    'report-generator': 'report_generator',
    'data-processing': 'data_processing',
    'multi-model-routing': 'multi_model_routing',
    'task-delegation': 'task_delegation',
    'prompt-optimization': 'prompt_optimization',
    'python-executor': 'python_executor',
    'image-generation': 'image_generation',
    'image-analysis': 'image_analysis',
    'audio-processing': 'audio_processing',
    'video-processing': 'video_processing',
    'user-prompt': 'user_prompt',
    'approval-gate': 'approval_gate',
    'feedback-collection': 'feedback_collection',
    'mcp-server': 'mcp_server',
    'mcp-memory': 'mcp_memory',
    'mcp-resource': 'mcp_resource',
    'pipeline-executor': 'pipeline_executor',
    'recursion-control': 'recursion_control',
}

def fix_tool_names(data: Any) -> Any:
    """Recursively fix tool names in a data structure."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key == 'tool' and isinstance(value, str) and value in TOOL_NAME_MAPPING:
                result[key] = TOOL_NAME_MAPPING[value]
                print(f"  Fixed tool name: {value} -> {TOOL_NAME_MAPPING[value]}")
            else:
                result[key] = fix_tool_names(value)
        return result
    elif isinstance(data, list):
        return [fix_tool_names(item) for item in data]
    else:
        return data

def process_pipeline(filepath: Path, dry_run: bool = False) -> bool:
    """Process a single pipeline file."""
    print(f"\nProcessing: {filepath.name}")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            pipeline = yaml.safe_load(content)
        
        # Fix tool names
        fixed_pipeline = fix_tool_names(pipeline)
        
        # Check if any changes were made
        if yaml.dump(pipeline) == yaml.dump(fixed_pipeline):
            print("  No changes needed")
            return False
        
        if not dry_run:
            # Write back to file
            with open(filepath, 'w') as f:
                yaml.dump(fixed_pipeline, f, default_flow_style=False, sort_keys=False)
            print("  File updated")
        else:
            print("  Changes detected (dry run - not saved)")
        
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    """Fix tool names in all pipeline files."""
    examples_dir = Path(__file__).parent.parent / 'examples'
    
    print("Tool Name Fix Report")
    print("=" * 80)
    
    # Add dry run option
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        print("DRY RUN MODE - No files will be modified")
        print()
    
    updated_count = 0
    error_count = 0
    
    for yaml_file in sorted(examples_dir.glob('*.yaml')):
        try:
            if process_pipeline(yaml_file, dry_run):
                updated_count += 1
        except Exception as e:
            print(f"  Error processing {yaml_file.name}: {e}")
            error_count += 1
    
    print("\n" + "=" * 80)
    print(f"Total files processed: {len(list(examples_dir.glob('*.yaml')))}")
    print(f"Files updated: {updated_count}")
    print(f"Errors: {error_count}")
    
    if dry_run:
        print("\nTo apply changes, run without --dry-run flag")

if __name__ == '__main__':
    main()