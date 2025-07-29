#!/usr/bin/env python3
"""Debug template rendering in detail."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.compiler.template_renderer import TemplateRenderer

# Test the _get_nested_value function
context = {}
result = TemplateRenderer._get_nested_value("execution.timestamp", context)
print(f"_get_nested_value result: {result}")

# Test with actual execution object
context2 = {"execution": {"timestamp": "test"}}
result2 = TemplateRenderer._get_nested_value("execution.timestamp", context2)
print(f"_get_nested_value with context: {result2}")

# Let's trace what happens
template = "{{ execution.timestamp }}"
print(f"\nRendering template: {template}")
print(f"Context: {{}}")

# Manually test the regex
import re
matches = re.findall(r"\{\{([^}]+)\}\}", template)
print(f"Regex matches: {matches}")

# Test the full render
result = TemplateRenderer.render(template, {})
print(f"Final result: {result}")