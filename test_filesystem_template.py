#!/usr/bin/env python3
"""Test script to debug filesystem template resolution."""

import asyncio
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator.tools.system_tools import FileSystemTool
from orchestrator.core.template_manager import TemplateManager

async def test_filesystem_template_resolution():
    """Test filesystem tool template resolution directly."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create filesystem tool
    fs_tool = FileSystemTool()
    
    # Create template manager with test context
    template_manager = TemplateManager()
    template_manager.register_context('output_path', 'test_output')
    template_manager.register_context('iteration', 1)
    template_manager.register_context('$iteration', 1)
    template_manager.register_context('parameters', {
        'input_document': 'test.md',
        'filename': 'example'
    })
    
    logger.info("Template manager context:")
    logger.info(f"Context keys: {list(template_manager.context.keys())}")
    
    # Test path template resolution
    test_path = "{{ output_path }}/iteration_{{ $iteration }}_document.md"
    logger.info(f"Testing path template: {test_path}")
    
    # Test content template resolution 
    test_content = """# Test Document

Input document: {{ parameters.input_document }}
Iteration: {{ $iteration }}
Filename: {{ parameters.filename }}
"""
    
    logger.info(f"Testing content template (first 100 chars): {test_content[:100]}...")
    
    # Test the tool execution
    try:
        result = await fs_tool.execute(
            action="write",
            path=test_path,
            content=test_content,
            template_manager=template_manager
        )
        logger.info(f"Tool execution result: {result}")
        
        # Check if file was created with resolved name
        expected_path = "test_output/iteration_1_document.md"
        if Path(expected_path).exists():
            logger.info(f"✅ File created with resolved path: {expected_path}")
            
            # Check content
            content = Path(expected_path).read_text()
            logger.info(f"File content:\n{content}")
            
            if "{{ " in content:
                logger.error("❌ Content still contains unresolved templates!")
            else:
                logger.info("✅ Content templates resolved successfully")
        else:
            logger.error(f"❌ File not found at expected path: {expected_path}")
            
            # Check if file was created with unresolved name
            literal_path = "{{ output_path }}/iteration_{{ $iteration }}_document.md"
            if Path(literal_path).exists():
                logger.error(f"❌ File created with literal template path: {literal_path}")
    
    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_filesystem_template_resolution())