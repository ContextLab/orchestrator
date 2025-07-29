#!/usr/bin/env python3
"""Debug nested template."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestrator.compiler.template_renderer import TemplateRenderer

template = """{% for section in sections %}
## {{ section.name }}

{% if section.items %}
Items ({{ section.items | length }}):
{% else %}
No items in this section.
{% endif %}

{% endfor %}"""

context = {
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
    ]
}

result = TemplateRenderer.render(template, context)
print("Result:")
print(result)
print("\n---")
print(f"Contains 'Items (2)': {'Items (2)' in result}")
print(f"Contains 'Section A': {'Section A' in result}")
print(f"Contains 'Section B': {'Section B' in result}")