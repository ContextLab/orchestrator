"""
Integration test for AutoDebugger Tool with orchestrator pipeline system.

This test verifies that the AutoDebugger tool properly integrates with the
orchestrator's Tool system and can be executed through the standard tool registry.
"""

import pytest
import asyncio
from src.orchestrator.tools.base import default_registry


class TestAutoDebuggerIntegration:
    """Integration tests for AutoDebugger with orchestrator."""
    
    @pytest.mark.asyncio
    async def test_autodebugger_tool_registration(self):
        """Test that AutoDebugger is properly registered in tool registry."""
        
        # Verify tool is registered
        tools = default_registry.list_tools()
        assert "auto_debugger" in tools, "AutoDebugger tool not found in registry"
        
        # Get the tool instance
        tool = default_registry.get_tool("auto_debugger")
        assert tool is not None, "AutoDebugger tool instance is None"
        
        # Verify tool properties
        assert tool.name == "auto_debugger"
        assert "Universal debugging tool" in tool.description
        assert len(tool.parameters) == 5  # Expected parameters
        
        # Verify required parameters
        param_names = [p.name for p in tool.parameters]
        assert "task_description" in param_names
        assert "content_to_debug" in param_names
        assert "error_context" in param_names
        assert "expected_outcome" in param_names
        assert "available_tools" in param_names
    
    @pytest.mark.asyncio
    async def test_autodebugger_basic_execution(self):
        """Test basic AutoDebugger execution through orchestrator tool system."""
        
        # Simple debugging task
        broken_python = """
def hello_world():
    print("Hello, World!"
    return "success"
"""
        
        # Execute through orchestrator tool registry
        tool = default_registry.get_tool("auto_debugger")
        result = await tool.execute(
            task_description="Fix Python syntax error in hello_world function",
            content_to_debug=broken_python,
            error_context="SyntaxError: '(' was never closed (missing closing parenthesis)",
            expected_outcome="Working Python function that prints Hello, World! and returns success"
        )
        
        # Validate result structure
        assert isinstance(result, dict)
        assert "success" in result
        assert "session_id" in result
        assert "task_description" in result
        assert "total_iterations" in result
        
        # The result should indicate whether debugging succeeded or provide error info
        if result["success"]:
            # If successful, check for expected fields
            assert "final_content" in result
            assert "validation" in result
            assert "debug_summary" in result
            assert result["total_iterations"] >= 1
        else:
            # If failed, check for error information
            assert "error_message" in result
            # This is acceptable - the test is checking integration, not debugging success
    
    @pytest.mark.asyncio
    async def test_autodebugger_parameter_validation(self):
        """Test AutoDebugger parameter validation."""
        
        tool = default_registry.get_tool("auto_debugger")
        
        # Test missing required parameter
        with pytest.raises(ValueError, match="task_description"):
            await tool.execute()
        
        # Test empty task description
        with pytest.raises(ValueError, match="task_description is required"):
            await tool.execute(task_description="")
    
    @pytest.mark.asyncio
    async def test_autodebugger_tool_discovery(self):
        """Test that tool discovery correctly identifies AutoDebugger for debugging tasks."""
        
        from src.orchestrator.tools.discovery import ToolDiscoveryEngine
        
        discovery = ToolDiscoveryEngine(default_registry)
        
        # Test various debugging-related actions
        debug_actions = [
            "debug python syntax errors",
            "fix javascript runtime issues", 
            "resolve api integration problems",
            "troubleshoot yaml configuration",
            "repair broken code"
        ]
        
        for action in debug_actions:
            matches = discovery.discover_tools_for_action(action)
            
            # Should find AutoDebugger in the matches
            auto_debugger_found = any(
                match.tool_name == "auto_debugger" 
                for match in matches
            )
            
            assert auto_debugger_found, f"AutoDebugger not suggested for action: {action}"
            
            # Should have reasonable confidence for debugging tasks
            auto_debugger_matches = [
                match for match in matches 
                if match.tool_name == "auto_debugger"
            ]
            
            assert len(auto_debugger_matches) > 0
            best_match = auto_debugger_matches[0]
            assert best_match.confidence >= 0.7, f"Low confidence for debugging action: {action}"
    
    def test_autodebugger_tool_schema(self):
        """Test AutoDebugger tool schema for MCP compatibility."""
        
        tool = default_registry.get_tool("auto_debugger")
        schema = tool.get_schema()
        
        # Verify schema structure
        assert "name" in schema
        assert "description" in schema
        assert "inputSchema" in schema
        
        assert schema["name"] == "auto_debugger"
        
        # Verify input schema
        input_schema = schema["inputSchema"]
        assert input_schema["type"] == "object"
        assert "properties" in input_schema
        assert "required" in input_schema
        
        # Verify required parameters
        required_params = input_schema["required"]
        assert "task_description" in required_params
        
        # Verify parameter properties
        properties = input_schema["properties"]
        assert "task_description" in properties
        assert "content_to_debug" in properties
        assert "error_context" in properties
        assert "expected_outcome" in properties
        assert "available_tools" in properties
    
    @pytest.mark.asyncio  
    async def test_autodebugger_error_handling(self):
        """Test AutoDebugger error handling and graceful failure modes."""
        
        tool = default_registry.get_tool("auto_debugger")
        
        # Test with potentially problematic input
        result = await tool.execute(
            task_description="Debug completely invalid and nonsensical content",
            content_to_debug="This is not valid code in any language: @@@ INVALID SYMBOLS @@@",
            error_context="Complete system failure with undefined behavior",
            expected_outcome="Should handle gracefully or provide meaningful error"
        )
        
        # Should return a structured result even in failure cases
        assert isinstance(result, dict)
        assert "success" in result
        assert "session_id" in result
        assert "task_description" in result
        
        # Should have either succeeded with some reasonable attempt or failed gracefully
        if not result["success"]:
            assert "error_message" in result
            assert isinstance(result["error_message"], str)
            assert len(result["error_message"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])