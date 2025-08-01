#!/usr/bin/env python3
"""Test FileSystemTool template rendering directly."""

import asyncio
import logging
from pathlib import Path
import sys

# Configure logging to see debug messages
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator.tools.system_tools import FileSystemTool
from src.orchestrator.core.template_manager import TemplateManager

async def test_filesystem_rendering():
    """Test template rendering in FileSystemTool."""
    
    # Create FileSystemTool
    tool = FileSystemTool()
    
    # Create TemplateManager and register some context
    template_manager = TemplateManager()
    
    # Register some test data
    template_manager.register_context("search_topic", {
        "total_results": 10,
        "results": [
            {"title": "Result 1", "url": "http://example1.com"},
            {"title": "Result 2", "url": "http://example2.com"}
        ]
    })
    
    template_manager.register_context("analyze_findings", {
        "result": "This is the analysis result."
    })
    template_manager.register_context("topic", "test template rendering")
    template_manager.register_context("execution", {
        "timestamp": "2025-08-01 13:00:00",
        "date": "2025-08-01"
    })
    
    # Test content with templates
    content = """# Research Report: {{ topic }}

**Generated on:** {{ execution.timestamp }}
**Total Sources:** {{ search_topic.total_results }}

## Analysis

{{ analyze_findings.result }}

## Search Results

{% for result in search_topic.results %}
{{ loop.index }}. **{{ result.title }}** - {{ result.url }}
{% endfor %}
"""
    
    print("=== TEST 1: Write with template_manager ===")
    result = await tool.execute(
        action="write",
        path="test_output_with_templates.md",
        content=content,
        _template_manager=template_manager
    )
    print(f"Result: {result}")
    
    # Read back the file to check
    with open("test_output_with_templates.md", "r") as f:
        written_content = f.read()
    
    print("\nWritten content:")
    print("-" * 60)
    print(written_content)
    print("-" * 60)
    
    # Check if templates were rendered
    if "{{" in written_content:
        print("\n❌ FAIL: Templates were NOT rendered!")
    else:
        print("\n✅ SUCCESS: Templates were rendered!")
    
    # Test without template manager
    print("\n\n=== TEST 2: Write without template_manager ===")
    result2 = await tool.execute(
        action="write",
        path="test_output_no_templates.md",
        content=content
    )
    print(f"Result: {result2}")
    
    # Read back
    with open("test_output_no_templates.md", "r") as f:
        written_content2 = f.read()
    
    if "{{" in written_content2:
        print("✅ Expected: Templates preserved when no template_manager")
    else:
        print("❌ Unexpected: Templates rendered without template_manager")
    
    # Clean up
    Path("test_output_with_templates.md").unlink(missing_ok=True)
    Path("test_output_no_templates.md").unlink(missing_ok=True)

if __name__ == "__main__":
    asyncio.run(test_filesystem_rendering())