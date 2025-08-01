#!/usr/bin/env python3
"""Simple test of template rendering."""

import asyncio
import sys

async def test_simple():
    # Add the src directory to Python path
    sys.path.insert(0, '/Users/jmanning/orchestrator/src')
    
    from orchestrator.core.template_manager import TemplateManager
    
    # Create template manager
    tm = TemplateManager(debug_mode=True)
    
    # Register some test data mimicking step results
    tm.register_context("topic", "test topic")
    tm.register_context("search_topic", {
        "total_results": 10,
        "results": [
            {"title": "Result 1", "url": "http://example1.com"},
            {"title": "Result 2", "url": "http://example2.com"}
        ]
    })
    tm.register_context("analyze_findings", {
        "result": "This is the analysis text"
    })
    
    # Test template rendering
    template = """
# Report: {{ topic }}

Total results: {{ search_topic.total_results }}

Analysis: {{ analyze_findings.result }}

Results:
{% for result in search_topic.results %}
- {{ result.title }}
{% endfor %}
"""
    
    rendered = tm.render(template)
    print("=== Rendered Output ===")
    print(rendered)
    
    if "{{" in rendered:
        print("\nERROR: Templates not rendered!")
    else:
        print("\nSUCCESS: Templates rendered correctly!")

if __name__ == "__main__":
    asyncio.run(test_simple())