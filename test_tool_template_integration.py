#!/usr/bin/env python3
"""
Simple test to verify template integration with updated tools.
"""

import asyncio
from orchestrator.core.template_manager import TemplateManager
from orchestrator.tools.system_tools import FileSystemTool

async def test_filesystem_tool_with_templates():
    """Test that FileSystemTool works with template rendering."""
    print("Testing FileSystemTool with template rendering...")
    
    # Create template manager and register some context
    template_manager = TemplateManager()
    template_manager.register_context("report_title", "Test Report")
    template_manager.register_context("author", "Test Author")
    
    # Create filesystem tool
    fs_tool = FileSystemTool()
    
    # Test with template parameters
    params = {
        "action": "write",
        "path": "/tmp/{{report_title | slugify}}.md",
        "content": "# {{report_title}}\n\nAuthor: {{author}}",
        "template_manager": template_manager
    }
    
    try:
        result = await fs_tool.execute(**params)
        print(f"Result: {result}")
        
        # Verify the file was created with rendered content
        if result.get("success"):
            print("✅ FileSystemTool template integration works!")
            # Check file content
            filepath = result.get("path") or result.get("filepath")
            with open(filepath, "r") as f:
                content = f.read()
                print(f"File content: {content}")
                if "Test Report" in content and "Test Author" in content:
                    print("✅ Template variables were rendered correctly!")
                else:
                    print("❌ Template variables were not rendered correctly!")
        else:
            print(f"❌ FileSystemTool failed: {result}")
            
    except Exception as e:
        print(f"❌ Error testing FileSystemTool: {e}")

async def test_without_template_manager():
    """Test that tools still work without template manager."""
    print("\nTesting FileSystemTool without template manager...")
    
    # Create filesystem tool
    fs_tool = FileSystemTool()
    
    # Test without template_manager parameter
    params = {
        "action": "write",
        "path": "/tmp/no_template_test.md",
        "content": "# Simple Content\n\nNo templates here.",
    }
    
    try:
        result = await fs_tool.execute(**params)
        print(f"Result: {result}")
        
        if result.get("success"):
            print("✅ FileSystemTool works without template manager!")
        else:
            print(f"❌ FileSystemTool failed: {result}")
            
    except Exception as e:
        print(f"❌ Error testing FileSystemTool without template manager: {e}")

if __name__ == "__main__":
    asyncio.run(test_filesystem_tool_with_templates())
    asyncio.run(test_without_template_manager())