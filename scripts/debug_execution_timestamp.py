#!/usr/bin/env python3
"""Debug execution.timestamp issue."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.compiler.template_renderer import TemplateRenderer

# Test template
template = "Generated at: {{ execution.timestamp }}"

# Test with empty context
result = TemplateRenderer.render(template, {})
print(f"Empty context: {result}")

# Test with execution in context
context = {"execution": {"timestamp": "2025-01-01 12:00:00"}}
result = TemplateRenderer.render(template, context)
print(f"With execution context: {result}")

# Test special handling
template2 = "Time: {{ execution.timestamp }}, Other: {{ missing }}"
result2 = TemplateRenderer.render(template2, {})
print(f"Special handling: {result2}")