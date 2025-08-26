#!/usr/bin/env python3
"""Test enhanced context collection from template manager."""

import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from orchestrator.core.unified_template_resolver import UnifiedTemplateResolver, TemplateResolutionContext
from orchestrator.core.template_manager import TemplateManager
from orchestrator.core.loop_context import GlobalLoopContextManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_context():
    """Test enhanced context collection."""
    
    # Create the components
    template_manager = TemplateManager(debug_mode=True)
    loop_manager = GlobalLoopContextManager()
    resolver = UnifiedTemplateResolver(
        template_manager=template_manager,
        loop_context_manager=loop_manager,
        debug_mode=True
    )
    
    # Simulate the real-world scenario: template manager has results but step_results doesn't
    print("=== Simulating Real Pipeline Context ===")
    
    # Register results with template manager (as the orchestrator does)
    template_manager.register_context("read_file", {"content": "Real file content", "size": 150})
    template_manager.register_context("analyze_content", {"result": "Real analysis result"}) 
    template_manager.register_context("transform_content", {"result": "Real transform result"})
    
    print(f"Template manager context keys: {list(template_manager.context.keys())}")
    
    # Collect context with empty step_results (simulating the orchestrator bug)  
    template_context = resolver.collect_context(
        pipeline_id="test-pipeline",
        task_id="save_file",
        tool_name="filesystem",
        step_results={},  # Empty - simulating the orchestrator issue
        pipeline_inputs={"file_list": ["file1.txt"], "output_dir": "test"}
    )
    
    enhanced_results = template_context.step_results
    print(f"Enhanced step results keys: {list(enhanced_results.keys())}")
    print(f"read_file available: {'read_file' in enhanced_results}")
    
    # Test template resolution
    test_template = "Content: {{ read_file.content }}, Analysis: {{ analyze_content.result }}"
    resolved = resolver.resolve_templates(test_template, template_context)
    
    print(f"\nTemplate: {test_template}")
    print(f"Resolved: {resolved}")
    
    # Check unresolved
    unresolved = resolver.get_unresolved_variables(test_template)
    print(f"Unresolved variables: {unresolved}")
    
    if not unresolved:
        print("\n✅ Enhanced context collection working!")
    else:
        print(f"\n❌ Still have unresolved variables: {unresolved}")

if __name__ == "__main__":
    test_enhanced_context()