#!/usr/bin/env python3
"""Quick script to check input names for all YAML examples."""

import yaml
from pathlib import Path
from orchestrator.compiler.auto_tag_yaml_parser import parse_yaml_with_auto_tags

examples_dir = Path("examples")

for yaml_file in sorted(examples_dir.glob("*.yaml")):
    print(f"\n{yaml_file.name}:")
    print("-" * 40)
    
    with open(yaml_file, 'r') as f:
        content = f.read()
    
    # Parse with AUTO tag support
    data = parse_yaml_with_auto_tags(content)
    
    if 'inputs' in data:
        for input_name, input_config in data['inputs'].items():
            required = input_config.get('required', False)
            default = input_config.get('default', 'None')
            input_type = input_config.get('type', 'any')
            print(f"  {input_name}: {input_type} (required={required}, default={default})")
    else:
        print("  No inputs defined!")