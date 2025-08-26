#!/usr/bin/env python3
"""Verify that the tool names in pipelines match actual tool registrations."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import and register tools
from src.orchestrator.tools.base import default_registry, register_default_tools

# Make sure tools are registered
register_default_tools()

# Get actual tool names
actual_tools = sorted(default_registry.list_tools())

print("Actual registered tools:")
for tool in actual_tools:
    print(f"  - {tool}")

# Expected tool names based on our fix
expected_mapping = {
    'filesystem': 'file_system',
    'web-search': 'web_search', 
    'headless-browser': 'headless_browser',
    'pdf-compiler': 'pdf_compiler',
    'report-generator': 'report_generator',
    'data-processing': 'data_processing',
    'multi-model-routing': 'multi_model_routing',
    'task-delegation': 'task_delegation',
    'prompt-optimization': 'prompt_optimization',
}

print("\nTool name discrepancies:")
for old_name, new_name in expected_mapping.items():
    if old_name in actual_tools:
        print(f"  ✓ {old_name} is correct (not {new_name})")
    elif new_name in actual_tools:
        print(f"  ✗ {old_name} should be {new_name}")
    else:
        print(f"  ? Neither {old_name} nor {new_name} found")