#!/usr/bin/env python3
"""Debug nested AUTO tag handling."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser

# The problematic YAML
yaml_content = """
config:
  value: <AUTO>Choose <AUTO>inner value</AUTO> for outer</AUTO>
"""

print("Original YAML:")
print(yaml_content)
print("-" * 60)

parser = AutoTagYAMLParser()

# Let's trace through the replacement process
import re

content = yaml_content
auto_tag_pattern = re.compile(r'<AUTO>(.*?)</AUTO>', re.DOTALL)

# First, find all matches
matches = list(auto_tag_pattern.finditer(content))
print(f"Found {len(matches)} AUTO tags:")
for i, match in enumerate(matches):
    print(f"  Match {i}: {match.group(0)}")
    print(f"    Start: {match.start()}, End: {match.end()}")
    print(f"    Content: {match.group(1)}")
    print()

# Check for innermost
print("Checking for innermost tags:")
for i, match in enumerate(matches):
    inner_content = match.group(1)
    has_auto = '<AUTO>' in inner_content
    print(f"  Match {i}: {'has' if has_auto else 'no'} AUTO tags inside")

print("-" * 60)
print("Running parser...")

try:
    result = parser.parse(yaml_content)
    print("Parse successful!")
    print("Result:", result)
except Exception as e:
    print(f"Parse failed: {e}")
    
    # Show the processed YAML
    print("\nProcessed YAML before parsing:")
    processed = parser._replace_auto_tags(yaml_content)
    print(processed)
    
    print("\nTag registry:")
    for k, v in parser.tag_registry.items():
        print(f"  {k}: {v}")