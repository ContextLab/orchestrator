"""Real LangChain Integration Tests - Issue #203 Phase 1

Tests the bidirectional LangChain-Orchestrator integration using REAL tools,
REAL API calls, and REAL execution. NO MOCKS as per user requirements.

Test Coverage:
- Bidirectional tool adapters (LangChain ↔ Orchestrator)
- Universal tool registry with LangChain support
- Tool compatibility enhancements
- Real LangChain tool execution
- Cross-ecosystem tool discovery and usage
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List

# Import LangChain with fallback
try:
    from langchain.tools import BaseTool, Tool as LangChainTool
    from langchain.tools import ShellTool
    from langchain.callbacks.manager import CallbackManagerForToolRun
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = None
    LangChainTool = None
    ShellTool = None

# Import orchestrator components
from src.orchestrator.tools.base import Tool as OrchestratorTool
from src.orchestrator.tools.langchain_adapter import (
    LangChainToolAdapter,
    OrchestratorToolAdapter,
    ToolAdapterFactory,
    make_langchain_tool,
    make_orchestrator_tool
)
from src.orchestrator.tools.universal_registry import (
    UniversalToolRegistry,
    ToolCategory,
    ToolSource,
    get_universal_registry
)
from src.orchestrator.tools.langchain_compatibility import (
    LangChainCompatibilityManager,
    enhance_tool_categories_with_langchain,
    create_langchain_tool_collection
)


pytestmark = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, 
    reason="LangChain not available - install with: pip install langchain"
)


# Real test tools - NO MOCKS

class RealCalculatorTool(OrchestratorTool):
    """Real calculator tool for testing orchestrator → LangChain adaptation."""
    
    def __init__(self):
        super().__init__(
            name="real-calculator",
            description="Performs real mathematical calculations"
        )
        self.add_parameter("expression", "string", "Mathematical expression to evaluate", required=True)
        self.add_parameter("precision", "integer", "Decimal precision", required=False, default=2)
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute real calculation."""
        expression = kwargs.get("expression", "")
        precision = kwargs.get("precision", 2)
        
        try:
            # Use eval for real calculation (safe in test environment)
            result = eval(expression)
            formatted_result = round(result, precision) if isinstance(result, float) else result
            
            return {
                "success": True,
                "expression": expression,
                "result": formatted_result,
                "precision": precision,
                "type": type(result).__name__
            }
        except Exception as e:
            return {
                "success": False,
                "expression": expression,
                "error": str(e)
            }


class RealFileManagerTool(OrchestratorTool):
    """Real file manager tool for testing."""
    
    def __init__(self):
        super().__init__(
            name="real-file-manager",
            description="Real file operations for testing"
        )
        self.add_parameter("action", "string", "Action: create, read, write, delete", required=True)
        self.add_parameter("path", "string", "File path", required=True)
        self.add_parameter("content", "string", "File content (for write)", required=False, default="")
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute real file operations."""
        action = kwargs.get("action", "")
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        
        try:
            if action == "create":
                Path(path).touch()
                return {"success": True, "action": "create", "path": path, "exists": Path(path).exists()}
            
            elif action == "write":
                Path(path).write_text(content)
                return {"success": True, "action": "write", "path": path, "size": len(content)}
            
            elif action == "read":
                content = Path(path).read_text()
                return {"success": True, "action": "read", "path": path, "content": content, "size": len(content)}
            
            elif action == "delete":
                existed = Path(path).exists()
                if existed:
                    Path(path).unlink()
                return {"success": True, "action": "delete", "path": path, "existed": existed}
            
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            return {"success": False, "action": action, "path": path, "error": str(e)}


def create_real_langchain_tool() -> BaseTool:
    """Create a real LangChain tool for testing."""
    from langchain.tools import StructuredTool
    from pydantic import BaseModel, Field
    
    class RealTextProcessorInput(BaseModel):
        text: str = Field(description="Text to process")
        operation: str = Field(description="Operation: upper, lower, reverse, length", default="upper")
    
    def process_text(text: str, operation: str = "upper") -> str:
        """Real text processing function."""
        try:
            if operation == "upper":
                return text.upper()
            elif operation == "lower":
                return text.lower()
            elif operation == "reverse":
                return text[::-1]
            elif operation == "length":
                return str(len(text))
            else:
                return f"Unknown operation: {operation}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    return StructuredTool.from_function(
        name="real-text-processor",
        description="Real text processing tool for testing LangChain → Orchestrator adaptation",
        func=process_text,
        args_schema=RealTextProcessorInput
    )


# Test Classes

@pytest.mark.asyncio
class TestLangChainAdapterReal:
    """Test bidirectional LangChain adapters with real tools."""
    
    async def test_orchestrator_to_langchain_real_calculation(self):
        """Test real orchestrator tool → LangChain adapter."""
        # Create real orchestrator tool
        calc_tool = RealCalculatorTool()
        
        # Create LangChain adapter
        langchain_adapter = ToolAdapterFactory.create_langchain_adapter(calc_tool)
        
        # Verify adapter properties
        assert langchain_adapter.name == "real-calculator"
        assert "mathematical calculations" in langchain_adapter.description
        
        # Test real execution via LangChain interface
        langchain_tool = langchain_adapter.get_langchain_tool()
        result = langchain_tool.run({"expression": "2 + 3 * 4"})
        
        # Verify real calculation result
        assert "14" in result  # 2 + (3 * 4) = 14
        
        # Test with parameters
        result2 = langchain_tool.run({"expression": "22 / 7", "precision": 3})
        assert "3.143" in result2  # pi approximation
    
    async def test_orchestrator_to_langchain_real_file_operations(self):
        """Test real file operations through LangChain adapter."""
        # Create temporary directory for real file operations
        with tempfile.TemporaryDirectory() as temp_dir:
            file_tool = RealFileManagerTool()
            langchain_adapter = ToolAdapterFactory.create_langchain_adapter(file_tool)
            
            test_file = os.path.join(temp_dir, "test_real_file.txt")
            test_content = "Hello, real world!"
            
            langchain_tool = langchain_adapter.get_langchain_tool()
            
            # Test real file creation
            result1 = langchain_tool.run({"action": "create", "path": test_file})
            assert "true" in result1.lower() or "success" in result1.lower()
            assert os.path.exists(test_file)
            
            # Test real file writing
            result2 = langchain_tool.run({"action": "write", "path": test_file, "content": test_content})
            assert "true" in result2.lower() or "success" in result2.lower()
            
            # Test real file reading
            result3 = langchain_tool.run({"action": "read", "path": test_file})
            assert test_content in result3
            
            # Test real file deletion
            result4 = langchain_tool.run({"action": "delete", "path": test_file})
            assert "true" in result4.lower() or "success" in result4.lower()
            assert not os.path.exists(test_file)
    
    async def test_langchain_to_orchestrator_real_text_processing(self):
        """Test real LangChain tool → orchestrator adapter."""
        # Create real LangChain tool
        text_processor = create_real_langchain_tool()
        
        # Create orchestrator adapter
        orchestrator_adapter = ToolAdapterFactory.create_orchestrator_adapter(text_processor)
        
        # Verify adapter properties
        assert orchestrator_adapter.name == "real-text-processor"
        assert "text processing" in orchestrator_adapter.description.lower()
        assert len(orchestrator_adapter.parameters) > 0
        
        # Test real execution via orchestrator interface
        result = await orchestrator_adapter.execute(text="Hello World", operation="upper")
        
        # Verify real processing result
        assert result["success"] is True
        assert result["output"] == "HELLO WORLD"
        
        # Test different operations
        result2 = await orchestrator_adapter.execute(text="Testing", operation="reverse")
        assert result2["output"] == "gnitseT"
        
        result3 = await orchestrator_adapter.execute(text="Count me", operation="length")
        assert result3["output"] == "8"
    
    async def test_async_execution_real(self):
        """Test real async execution through adapters."""
        # Test async orchestrator tool
        calc_tool = RealCalculatorTool()
        langchain_adapter = ToolAdapterFactory.create_langchain_adapter(calc_tool)
        
        # Test async execution
        langchain_tool = langchain_adapter.get_langchain_tool()
        result = await langchain_tool.arun({"expression": "10 ** 3"})
        assert "1000" in result
        
        # Test LangChain tool async adaptation
        text_processor = create_real_langchain_tool()
        orchestrator_adapter = ToolAdapterFactory.create_orchestrator_adapter(text_processor)
        
        # Test async execution
        result2 = await orchestrator_adapter.execute(text="async test", operation="upper")
        assert result2["success"] is True
        assert result2["output"] == "ASYNC TEST"


@pytest.mark.asyncio
class TestUniversalRegistryReal:
    """Test Universal Tool Registry with real tools."""
    
    def setup_method(self):
        """Set up fresh registry for each test."""
        self.registry = UniversalToolRegistry()
    
    async def test_register_real_orchestrator_tools(self):
        """Test registering real orchestrator tools."""
        calc_tool = RealCalculatorTool()
        
        # Register with enhanced metadata
        self.registry.register_orchestrator_tool(
            calc_tool,
            category=ToolCategory.CUSTOM,
            tags=["math", "calculator", "testing"],
            security_level="moderate"
        )
        
        # Verify registration
        assert "real-calculator" in self.registry.tool_metadata
        metadata = self.registry.tool_metadata["real-calculator"]
        assert metadata.source == ToolSource.ORCHESTRATOR
        assert metadata.category == ToolCategory.CUSTOM
        assert "math" in metadata.tags
        
        # Test discovery
        math_tools = self.registry.discover_tools(tags=["math"])
        assert "real-calculator" in math_tools
        
        # Test execution via registry
        result = await self.registry.execute_tool_enhanced("real-calculator", expression="5 * 6")
        assert result.success is True
        assert "30" in str(result.output)
    
    async def test_register_real_langchain_tools(self):
        """Test registering real LangChain tools."""
        if not LANGCHAIN_AVAILABLE:
            pytest.skip("LangChain not available")
        
        text_processor = create_real_langchain_tool()
        
        # Register LangChain tool
        adapter = self.registry.register_langchain_tool(
            text_processor,
            category=ToolCategory.CUSTOM,
            tags=["text", "processing", "langchain"]
        )
        
        # Verify registration
        assert adapter is not None
        assert "real-text-processor" in self.registry.tool_metadata
        metadata = self.registry.tool_metadata["real-text-processor"]
        assert metadata.source == ToolSource.HYBRID  # Registered as both
        assert metadata.langchain_compatible is True
        
        # Test discovery
        text_tools = self.registry.discover_tools(tags=["text"])
        assert "real-text-processor" in text_tools
        
        # Test execution via orchestrator interface
        result = await self.registry.execute_tool_enhanced(
            "real-text-processor", 
            text="registry test",
            operation="upper"
        )
        assert result.success is True
        assert "REGISTRY TEST" in str(result.output)
    
    def test_tool_discovery_real(self):
        """Test real tool discovery capabilities."""
        # Register multiple real tools
        calc_tool = RealCalculatorTool()
        file_tool = RealFileManagerTool()
        text_processor = create_real_langchain_tool()
        
        self.registry.register_orchestrator_tool(calc_tool, ToolCategory.CUSTOM, ["math", "calculator"])
        self.registry.register_orchestrator_tool(file_tool, ToolCategory.SYSTEM, ["file", "system"])
        
        if LANGCHAIN_AVAILABLE:
            self.registry.register_langchain_tool(text_processor, ToolCategory.CUSTOM, ["text", "processing"])
        
        # Test category-based discovery
        system_tools = self.registry.discover_tools(category=ToolCategory.SYSTEM)
        assert "real-file-manager" in system_tools
        
        custom_tools = self.registry.discover_tools(category=ToolCategory.CUSTOM)
        assert "real-calculator" in custom_tools
        
        # Test tag-based discovery
        math_tools = self.registry.discover_tools(tags=["math"])
        assert "real-calculator" in math_tools
        
        # Test source-based discovery
        orchestrator_tools = self.registry.discover_tools(source=ToolSource.ORCHESTRATOR)
        assert "real-calculator" in orchestrator_tools
        assert "real-file-manager" in orchestrator_tools
    
    def test_get_tool_info_real(self):
        """Test getting comprehensive tool information."""
        calc_tool = RealCalculatorTool()
        self.registry.register_orchestrator_tool(calc_tool, ToolCategory.CUSTOM, ["math"])
        
        # Get tool info
        info = self.registry.get_tool_info("real-calculator")
        
        # Verify comprehensive information
        assert info is not None
        assert info["name"] == "real-calculator"
        assert info["source"] == "orchestrator"
        assert info["category"] == "custom"
        assert "math" in info["tags"]
        assert info["langchain_compatible"] is True
        assert "parameters" in info
        assert len(info["parameters"]) == 2  # expression and precision


@pytest.mark.asyncio
class TestLangChainCompatibilityReal:
    """Test LangChain compatibility enhancements with real tools."""
    
    def setup_method(self):
        """Set up compatibility manager."""
        if not LANGCHAIN_AVAILABLE:
            pytest.skip("LangChain not available")
        
        self.compat_manager = LangChainCompatibilityManager()
    
    def test_make_tool_compatible_real(self):
        """Test making real orchestrator tool LangChain compatible."""
        calc_tool = RealCalculatorTool()
        
        # Make compatible
        result = self.compat_manager.make_tool_compatible(
            calc_tool,
            enhance_description=True,
            add_examples=True
        )
        
        # Verify compatibility result
        assert result.success is True
        assert result.tool_name == "real-calculator"
        assert result.langchain_tool is not None
        assert result.error is None
        
        # Verify enhanced LangChain tool
        langchain_tool = result.langchain_tool
        assert langchain_tool.name == "real-calculator"
        assert "Parameters:" in langchain_tool.description  # Enhanced description
        assert hasattr(langchain_tool, 'args_schema')
        
        # Test real execution via compatible tool
        sync_result = langchain_tool.func(expression="7 + 8")
        assert "15" in sync_result
    
    async def test_enhanced_description_real(self):
        """Test enhanced description generation for real tools."""
        file_tool = RealFileManagerTool()
        
        result = self.compat_manager.make_tool_compatible(
            file_tool,
            enhance_description=True,
            add_examples=True
        )
        
        assert result.success is True
        description = result.langchain_tool.description
        
        # Verify enhancement
        assert "Parameters:" in description
        assert "action (string)" in description
        assert "path (string)" in description
        assert file_tool.description in description
    
    async def test_parameter_mapping_real(self):
        """Test real parameter mapping and validation."""
        calc_tool = RealCalculatorTool()
        result = self.compat_manager.make_tool_compatible(calc_tool)
        
        assert result.success is True
        langchain_tool = result.langchain_tool
        
        # Verify parameter schema
        schema = langchain_tool.args_schema.schema()
        assert "properties" in schema
        assert "expression" in schema["properties"]
        assert "precision" in schema["properties"]
        assert "expression" in schema["required"]
        assert "precision" not in schema.get("required", [])  # Optional parameter


@pytest.mark.asyncio
class TestEndToEndIntegrationReal:
    """End-to-end tests with real tool execution."""
    
    async def test_full_integration_workflow_real(self):
        """Test complete integration workflow with real tools."""
        if not LANGCHAIN_AVAILABLE:
            pytest.skip("LangChain not available")
        
        # Step 1: Create real tools
        calc_tool = RealCalculatorTool()
        text_processor = create_real_langchain_tool()
        
        # Step 2: Set up universal registry
        registry = UniversalToolRegistry()
        
        # Step 3: Register tools with different sources
        registry.register_orchestrator_tool(calc_tool, ToolCategory.CUSTOM, ["math"])
        orchestrator_adapter = registry.register_langchain_tool(text_processor, ToolCategory.CUSTOM, ["text"])
        
        # Step 4: Create compatibility layer
        compat_manager = LangChainCompatibilityManager()
        compat_result = compat_manager.make_tool_compatible(calc_tool)
        
        # Step 5: Verify bidirectional functionality
        
        # Test orchestrator tool via registry
        calc_result = await registry.execute_tool_enhanced("real-calculator", expression="12 * 12")
        assert calc_result.success is True
        assert "144" in str(calc_result.output)
        
        # Test LangChain tool via registry  
        text_result = await registry.execute_tool_enhanced(
            "real-text-processor",
            text="integration test",
            operation="upper"
        )
        assert text_result.success is True
        assert "INTEGRATION TEST" in str(text_result.output)
        
        # Test tool discovery
        all_tools = registry.discover_tools()
        assert "real-calculator" in all_tools
        assert "real-text-processor" in all_tools
        
        # Test getting LangChain versions (this should return a StructuredTool)
        langchain_calc = registry.get_langchain_tool("real-calculator")
        assert langchain_calc is not None
        
        # Test direct LangChain execution
        lc_result = langchain_calc.run({"expression": "8 + 7"})
        assert "15" in lc_result
        
        # Step 6: Verify statistics
        stats = registry.get_statistics()
        assert stats["total_tools"] >= 2
        assert stats["langchain_available"] is True
        assert stats["by_source"]["orchestrator"] >= 1
        assert stats["by_source"]["hybrid"] >= 1


@pytest.mark.asyncio
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    async def test_mixed_tool_pipeline_real(self):
        """Test pipeline using both orchestrator and LangChain tools."""
        if not LANGCHAIN_AVAILABLE:
            pytest.skip("LangChain not available")
        
        # Create mixed pipeline with real tools
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Use file tool to create test data
            file_tool = RealFileManagerTool()
            test_file = os.path.join(temp_dir, "pipeline_test.txt")
            
            create_result = await file_tool.execute(action="create", path=test_file)
            assert create_result["success"] is True
            
            write_result = await file_tool.execute(
                action="write", 
                path=test_file, 
                content="Pipeline test data: 42"
            )
            assert write_result["success"] is True
            
            # Step 2: Read data back
            read_result = await file_tool.execute(action="read", path=test_file)
            assert read_result["success"] is True
            content = read_result["content"]
            
            # Step 3: Process text with LangChain tool
            text_processor = create_real_langchain_tool()
            orchestrator_adapter = OrchestratorToolAdapter(text_processor)
            
            upper_result = await orchestrator_adapter.execute(text=content, operation="upper")
            assert upper_result["success"] is True
            assert "PIPELINE TEST DATA: 42" in upper_result["output"]
            
            # Step 4: Extract number and calculate
            calc_tool = RealCalculatorTool()
            calc_result = await calc_tool.execute(expression="42 * 2")
            assert calc_result["success"] is True
            assert calc_result["result"] == 84
            
            # Step 5: Clean up with file tool
            delete_result = await file_tool.execute(action="delete", path=test_file)
            assert delete_result["success"] is True
    
    async def test_tool_chaining_real(self):
        """Test chaining real tool executions."""
        if not LANGCHAIN_AVAILABLE:
            pytest.skip("LangChain not available")
        
        # Chain: Calculate → Format → Process
        calc_tool = RealCalculatorTool()
        text_processor = create_real_langchain_tool()
        text_adapter = OrchestratorToolAdapter(text_processor)
        
        # Step 1: Calculate
        calc_result = await calc_tool.execute(expression="3.14159 * 2")
        assert calc_result["success"] is True
        pi_doubled = str(calc_result["result"])
        
        # Step 2: Format text
        format_text = f"Pi doubled is approximately {pi_doubled}"
        
        # Step 3: Process with LangChain tool
        upper_result = await text_adapter.execute(text=format_text, operation="upper")
        assert upper_result["success"] is True
        
        final_output = upper_result["output"]
        assert "PI DOUBLED IS APPROXIMATELY" in final_output
        assert "6.28" in final_output


# Integration test that verifies everything works together
@pytest.mark.asyncio
async def test_complete_langchain_integration_real():
    """Complete integration test using real tools and real execution."""
    if not LANGCHAIN_AVAILABLE:
        pytest.skip("LangChain not available")
    
    print("\\n=== Complete LangChain Integration Test ===")
    
    # Create real tools
    calc_tool = RealCalculatorTool()
    text_processor = create_real_langchain_tool()
    
    # Test bidirectional adapters
    langchain_calc = make_langchain_tool(calc_tool)
    orchestrator_text = make_orchestrator_tool(text_processor)
    
    # Execute real calculations
    result1 = langchain_calc.run({"expression": "123 + 456"})
    assert "579" in result1
    print(f"✓ LangChain adapter executed: {result1[:50]}...")
    
    result2 = await orchestrator_text.execute(text="Hello World", operation="reverse")
    assert result2["success"] is True
    assert result2["output"] == "dlroW olleH"
    print(f"✓ Orchestrator adapter executed: {result2['output']}")
    
    # Test universal registry
    registry = get_universal_registry()
    
    # Get statistics
    stats = registry.get_statistics()
    print(f"✓ Registry contains {stats['total_tools']} total tools")
    print(f"✓ LangChain available: {stats['langchain_available']}")
    
    # Test tool discovery
    all_tools = registry.discover_tools()
    print(f"✓ Discovered {len(all_tools)} tools via registry")
    
    print("=== Integration Test Complete ===\\n")
    
    return True


if __name__ == "__main__":
    # Run basic integration test
    asyncio.run(test_complete_langchain_integration_real())