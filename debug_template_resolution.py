#!/usr/bin/env python3
"""Debug script to understand template resolution issues."""

import logging
import sys
import os

# Add the src directory to the path so we can import orchestrator modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator.core.unified_template_resolver import UnifiedTemplateResolver, TemplateResolutionContext
from orchestrator.core.template_manager import TemplateManager
from orchestrator.core.loop_context import GlobalLoopContextManager, LoopContextVariables

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_template_resolution():
    """Test template resolution with loop variables."""
    
    # Create the components
    template_manager = TemplateManager(debug_mode=True)
    loop_manager = GlobalLoopContextManager()
    resolver = UnifiedTemplateResolver(
        template_manager=template_manager,
        loop_context_manager=loop_manager,
        debug_mode=True
    )
    
    # Simulate a for_each loop with 3 items
    items = ["file1.txt", "file2.txt", "file3.txt"]
    
    # Test resolution for the first item (index 0)
    current_item = items[0]
    current_index = 0
    
    # Create loop context as would happen during pipeline execution
    loop_context = loop_manager.create_loop_context(
        step_id="process_files",
        item=current_item,
        index=current_index, 
        items=items
    )
    
    # Push the loop context
    loop_manager.push_loop(loop_context)
    
    # Collect context for template resolution
    template_context = resolver.collect_context(
        pipeline_id="control-flow-for-loop",
        task_id="save_file", 
        tool_name="filesystem",
        pipeline_inputs={"file_list": items, "output_dir": "test_output"},
        step_results={
            "read_file": {"content": "Sample file content", "size": 100},
            "analyze_content": {"result": "This is analysis result"},
            "transform_content": {"result": "This is transformed content"}
        }
    )
    
    # Test template resolution
    test_content = """# Processed: {{ $item }}

File index: {{ $index }}
Is first: {{ $is_first }}
Is last: {{ $is_last }}

## Original Size
{{ read_file.size }} bytes

## Analysis
{{ analyze_content.result }}

## Transformed Content
{{ transform_content.result }}
"""
    
    print("=== Template Resolution Test ===")
    print(f"Original content:\n{test_content}")
    print("\n=== Context Info ===")
    print(f"Loop variables from loop manager: {loop_manager.get_accessible_loop_variables()}")
    print(f"Template context flat dict keys: {list(template_context.to_flat_dict().keys())}")
    
    # Resolve the template
    resolved_content = resolver.resolve_templates(test_content, template_context)
    
    print(f"\n=== Resolved Content ===")
    print(resolved_content)
    
    # Check what's still unresolved
    unresolved = []
    for line in resolved_content.split('\n'):
        if '{{' in line and '}}' in line:
            unresolved.append(line.strip())
    
    if unresolved:
        print(f"\n=== Still Unresolved ===")
        for line in unresolved:
            print(f"  {line}")
    else:
        print(f"\n✅ All templates resolved successfully!")

if __name__ == "__main__":
    test_template_resolution()