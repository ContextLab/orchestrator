#!/usr/bin/env python3
"""Debug nested for loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.compiler.template_renderer import TemplateRenderer

# Test nested for loop
template = """{% if items %}
Items ({{ items | length }}):
{% for item in items %}
{{ loop.index }}. {{ item.name }}
{% endfor %}
{% endif %}"""

context = {
    "items": [
        {"name": "Item 1"},
        {"name": "Item 2"}
    ]
}

result = TemplateRenderer.render(template, context)
print("Result:")
print(repr(result))
print(f"\nContains 'Items (2)': {'Items (2)' in result}")
print(f"Contains '1. Item 1': {'1. Item 1' in result}")

# Now test the exact failing template
template2 = """# {{ title }}

{% for section in sections %}
## {{ section.name }}

{% if section.items %}
Items ({{ section.items | length }}):
{% for item in section.items %}
{{ loop.index }}. {{ item.name }}: {{ item.value | default('N/A') }}
{% endfor %}
{% else %}
No items in this section.
{% endif %}

{% endfor %}

Summary: {{ stats.total }} items processed at {{ execution.timestamp }}"""

context2 = {
    "title": "Complex Report",
    "sections": [
        {
            "name": "Section A",
            "items": [
                {"name": "Item 1", "value": 100},
                {"name": "Item 2"}
            ]
        },
        {
            "name": "Section B",
            "items": []
        }
    ],
    "stats": {"total": 2}
}

print("\n\n--- Full template test ---")
result2 = TemplateRenderer.render(template2, context2)
print(result2)