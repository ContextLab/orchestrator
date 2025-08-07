#!/usr/bin/env python3
"""
Simple AutoDebugger test to debug initialization issues.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator.tools.auto_debugger import AutoDebuggerTool

async def test_simple_autodebugger():
    """Simple test to check AutoDebugger initialization and basic functionality."""
    
    print("Creating AutoDebugger instance...")
    debugger = AutoDebuggerTool()
    
    print("Testing simple debugging task...")
    
    # Very simple debugging task
    simple_broken_code = """
def hello():
    print("Hello"  # Missing closing parenthesis
"""
    
    try:
        result = await debugger._arun(
            task_description="Fix simple Python syntax error",
            content_to_debug=simple_broken_code,
            error_context="SyntaxError: '(' was never closed", 
            expected_outcome="Working Python function",
            available_tools=[]  # Use empty tools list to avoid tool registry issues
        )
        
        print("AutoDebugger result:")
        print(result[:500] + "..." if len(result) > 500 else result)
        
    except Exception as e:
        print(f"AutoDebugger failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_autodebugger())