#!/usr/bin/env python3
"""Debug the save_report step execution."""

import asyncio
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator.compiler.yaml_compiler import YAMLCompiler

def debug_save_report():
    """Debug how save_report step is compiled."""
    
    # Load the pipeline YAML
    with open("examples/research_advanced_tools.yaml", "r") as f:
        pipeline_yaml = yaml.safe_load(f)
    
    # Find save_report step
    save_report_def = None
    for step in pipeline_yaml.get("steps", []):
        if step.get("id") == "save_report":
            save_report_def = step
            break
    
    if not save_report_def:
        print("save_report step not found!")
        return
    
    print("=== save_report step definition ===")
    print(yaml.dump(save_report_def, default_flow_style=False))
    
    # Check what fields are in the definition
    print("\n=== Step fields ===")
    for key, value in save_report_def.items():
        if key == "parameters":
            print(f"{key}: {list(value.keys())}")
        elif isinstance(value, str) and len(value) > 100:
            print(f"{key}: <{len(value)} chars>")
        else:
            print(f"{key}: {value}")
    
    # Check if it would get metadata["tool"]
    print("\n=== Expected metadata ===")
    metadata = save_report_def.get("metadata", {})
    if "tool" in save_report_def:
        metadata["tool"] = save_report_def["tool"]
    print(f"metadata would be: {metadata}")
    
    # Check action mapping
    action = save_report_def.get("action")
    if not action and "tool" in save_report_def:
        action = save_report_def["tool"]
    print(f"action would be: {action}")

if __name__ == "__main__":
    debug_save_report()