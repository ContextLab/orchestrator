#!/usr/bin/env python3
"""Debug script to understand step results template resolution issues."""

import logging
import sys
import os

# Add the src directory to the path so we can import orchestrator modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator.core.unified_template_resolver import UnifiedTemplateResolver, TemplateResolutionContext
from orchestrator.core.template_manager import TemplateManager
from orchestrator.core.loop_context import GlobalLoopContextManager, LoopContextVariables

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_step_results_resolution():
    """Test step results template resolution."""
    
    # Create the components
    template_manager = TemplateManager(debug_mode=True)
    loop_manager = GlobalLoopContextManager()
    resolver = UnifiedTemplateResolver(
        template_manager=template_manager,
        loop_context_manager=loop_manager,
        debug_mode=True
    )
    
    # Create mock step results as they would appear from tools
    step_results = {
        "read_file": {"content": "Sample file content", "size": 100, "result": "file read successfully"},
        "analyze_content": {"result": "This is analysis result", "confidence": 0.95},
        "transform_content": {"result": "This is transformed content", "word_count": 50}
    }
    
    print("=== Step Results Test ===")
    print(f"Step results: {step_results}")
    
    # Register step results directly with template manager to see how wrapping works
    template_manager.clear_context()
    template_manager.register_context("read_file", step_results["read_file"])
    template_manager.register_context("analyze_content", step_results["analyze_content"])
    template_manager.register_context("transform_content", step_results["transform_content"])
    
    # Test simple template resolution
    simple_template = "{{ read_file.content }}"
    print(f"\\nTesting simple template: {simple_template}")
    simple_resolved = template_manager.render(simple_template)
    print(f"Simple resolved: {simple_resolved}")
    
    # Test with size
    size_template = "{{ read_file.size }}"
    print(f"\\nTesting size template: {size_template}")
    size_resolved = template_manager.render(size_template)
    print(f"Size resolved: {size_resolved}")
    
    # Test using the unified resolver
    print(f"\\n=== Using Unified Resolver ===")
    
    # Collect context for template resolution
    template_context = resolver.collect_context(
        pipeline_id="control-flow-for-loop",
        task_id="save_file", 
        tool_name="filesystem",
        step_results=step_results
    )
    
    context_dict = template_context.to_flat_dict()
    print(f"Context keys: {list(context_dict.keys())}")
    print(f"read_file in context: {'read_file' in context_dict}")
    
    if 'read_file' in context_dict:
        read_file_obj = context_dict['read_file']
        print(f"read_file type: {type(read_file_obj)}")
        print(f"read_file value: {read_file_obj}")
        
        # Check if it has the content attribute
        if hasattr(read_file_obj, 'content'):
            print(f"read_file.content (attr): {read_file_obj.content}")
        if isinstance(read_file_obj, dict) and 'content' in read_file_obj:
            print(f"read_file['content'] (dict): {read_file_obj['content']}")
    
    # Test resolution with the unified resolver
    test_template = "{{ read_file.content }} - Size: {{ read_file.size }}"
    resolved = resolver.resolve_templates(test_template, template_context)
    print(f"\\nTemplate: {test_template}")
    print(f"Resolved: {resolved}")
    
    # Check unresolved variables
    unresolved = resolver.get_unresolved_variables(test_template)
    print(f"Unresolved variables: {unresolved}")
    
    # Test more complex template
    complex_template = """Analysis: {{ analyze_content.result }}
Transform: {{ transform_content.result }}
Content: {{ read_file.content }}
Size: {{ read_file.size }} bytes"""
    
    print(f"\\n=== Complex Template Test ===")
    print(f"Template:\\n{complex_template}")
    complex_resolved = resolver.resolve_templates(complex_template, template_context)
    print(f"\\nResolved:\\n{complex_resolved}")

if __name__ == "__main__":
    test_step_results_resolution()