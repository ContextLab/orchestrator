#!/usr/bin/env python3
"""Test execution.timestamp with empty context."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.compiler.template_renderer import TemplateRenderer

# Test the exact test case
template = "Generated at: {{ execution.timestamp }}"
context = {}

result = TemplateRenderer.render(template, context)
print(f"Result: '{result}'")
print(f"Contains '20': {'20' in result}")
print(f"Result length: {len(result)}")

# Try Jinja2 style first
template2 = "Generated at: {{ execution.timestamp }}"
text = TemplateRenderer._render_jinja2(template2, {})
print(f"\nAfter Jinja2: '{text}'")

# Then simple
text2 = TemplateRenderer._render_simple(text, {})
print(f"After simple: '{text2}'")