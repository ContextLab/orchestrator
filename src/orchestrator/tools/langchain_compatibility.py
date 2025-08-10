"""LangChain Compatibility Layer - Issue #203 Phase 1

Enhances existing orchestrator tool categories with LangChain compatibility.
Provides seamless integration between orchestrator tools and LangChain ecosystem.
"""

import logging
from typing import Dict, List, Any, Optional, Type, Union, get_type_hints
from dataclasses import dataclass
import inspect
import asyncio

# Import LangChain components with fallback
try:
    from langchain.tools import BaseTool, StructuredTool
    from langchain.callbacks.manager import CallbackManagerForToolRun
    from pydantic import BaseModel, Field, create_model
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = None
    StructuredTool = None
    CallbackManagerForToolRun = None
    BaseModel = None
    Field = None
    create_model = None

from .base import Tool as OrchestratorTool, ToolParameter
from .langchain_adapter import ToolAdapterFactory
from .universal_registry import universal_registry, ToolCategory, ToolSource

logger = logging.getLogger(__name__)


@dataclass
class CompatibilityResult:
    """Result of making a tool LangChain compatible."""
    success: bool
    tool_name: str
    langchain_tool: Optional[BaseTool] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class LangChainCompatibilityManager:
    """Manager for making orchestrator tools LangChain compatible.
    
    Features:
    - Automatic schema generation for LangChain tools
    - Parameter mapping and validation
    - Async/sync execution bridging
    - Enhanced error handling
    - Tool discovery and registration
    """
    
    def __init__(self):
        self.compatible_tools: Dict[str, BaseTool] = {}
        self.compatibility_metadata: Dict[str, Dict[str, Any]] = {}
        
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available - compatibility features disabled")
    
    def make_tool_compatible(
        self,
        orchestrator_tool: OrchestratorTool,
        enhance_description: bool = True,
        add_examples: bool = True
    ) -> CompatibilityResult:
        """Make an orchestrator tool LangChain compatible."""
        
        if not LANGCHAIN_AVAILABLE:
            return CompatibilityResult(
                success=False,
                tool_name=orchestrator_tool.name,
                error="LangChain not available"
            )
        
        try:
            # Generate enhanced description
            description = orchestrator_tool.description
            if enhance_description:
                description = self._enhance_description(orchestrator_tool, add_examples)
            
            # Create Pydantic model for parameters
            input_model = self._create_input_model(orchestrator_tool)
            
            # Create LangChain tool using StructuredTool
            langchain_tool = StructuredTool(
                name=orchestrator_tool.name,
                description=description,
                args_schema=input_model,
                func=self._create_sync_wrapper(orchestrator_tool),
                coroutine=self._create_async_wrapper(orchestrator_tool),
                handle_tool_error=True
            )
            
            # Store compatibility info
            self.compatible_tools[orchestrator_tool.name] = langchain_tool
            self.compatibility_metadata[orchestrator_tool.name] = {
                "original_parameters": len(orchestrator_tool.parameters),
                "enhanced_description": enhance_description,
                "has_examples": add_examples,
                "input_model_name": input_model.__name__,
                "supports_async": True
            }
            
            return CompatibilityResult(
                success=True,
                tool_name=orchestrator_tool.name,
                langchain_tool=langchain_tool,
                metadata=self.compatibility_metadata[orchestrator_tool.name]
            )
            
        except Exception as e:
            logger.error(f"Failed to make tool {orchestrator_tool.name} compatible: {e}")
            return CompatibilityResult(
                success=False,
                tool_name=orchestrator_tool.name,
                error=str(e)
            )
    
    def _enhance_description(
        self,
        tool: OrchestratorTool,
        add_examples: bool = True
    ) -> str:
        """Create enhanced description with parameter details and examples."""
        
        base_description = tool.description
        
        # Add parameter details
        if tool.parameters:
            param_details = []
            for param in tool.parameters:
                detail = f"- {param.name} ({param.type}): {param.description}"
                if not param.required:
                    detail += f" (optional, default: {param.default})"
                param_details.append(detail)
            
            base_description += f"\n\nParameters:\n" + "\n".join(param_details)
        
        # Add usage examples based on tool category
        if add_examples:
            examples = self._generate_usage_examples(tool)
            if examples:
                base_description += f"\n\nExamples:\n{examples}"
        
        return base_description
    
    def _generate_usage_examples(self, tool: OrchestratorTool) -> Optional[str]:
        """Generate usage examples based on tool type and parameters."""
        
        examples_map = {
            "web-search": 'query="python machine learning tutorials"',
            "headless-browser": 'action="scrape", url="https://example.com"',
            "filesystem": 'action="read", path="/path/to/file.txt"',
            "terminal": 'command="ls -la", working_dir="/home/user"',
            "data-processing": 'action="convert", data={"key": "value"}, format="json"',
            "validation": 'data={"name": "John"}, schema={"type": "object"}',
            "python-executor": 'code="print(\'Hello World\')", timeout=30',
            "image-analysis": 'action="describe", image_path="/path/to/image.jpg"',
            "user-prompt": 'prompt="Please confirm the action", timeout=60'
        }
        
        return examples_map.get(tool.name)
    
    def _create_input_model(self, tool: OrchestratorTool) -> Type[BaseModel]:
        """Create Pydantic model for tool parameters."""
        
        if not tool.parameters:
            # Create minimal model with no required fields
            return create_model(
                f"{tool.name.replace('-', '_').title()}Input",
                __base__=BaseModel
            )
        
        # Build field definitions
        field_definitions = {}
        
        for param in tool.parameters:
            # Map orchestrator types to Python types
            python_type = self._map_parameter_type(param.type)
            
            # Create field with proper validation
            if param.required:
                field_definitions[param.name] = (
                    python_type,
                    Field(description=param.description)
                )
            else:
                field_definitions[param.name] = (
                    Optional[python_type],
                    Field(default=param.default, description=param.description)
                )
        
        # Create the model class
        model_name = f"{tool.name.replace('-', '_').title()}Input"
        return create_model(model_name, **field_definitions, __base__=BaseModel)
    
    def _map_parameter_type(self, orchestrator_type: str) -> Type:
        """Map orchestrator parameter type to Python type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "float": float,
            "boolean": bool,
            "array": List[Any],
            "object": Dict[str, Any],
            "any": Any
        }
        
        return type_mapping.get(orchestrator_type.lower(), str)
    
    def _create_sync_wrapper(self, tool: OrchestratorTool):
        """Create synchronous wrapper for orchestrator tool."""
        
        def sync_wrapper(**kwargs) -> str:
            """Synchronous wrapper for orchestrator tool execution."""
            try:
                # Run async method in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(tool.execute(**kwargs))
                finally:
                    loop.close()
                
                # Convert result to string format expected by LangChain
                if isinstance(result, dict):
                    if 'output' in result:
                        return str(result['output'])
                    elif 'result' in result:
                        return str(result['result'])
                    else:
                        import json
                        return json.dumps(result, indent=2, default=str)
                else:
                    return str(result)
                    
            except Exception as e:
                logger.error(f"Error in sync wrapper for {tool.name}: {e}")
                return f"Error: {str(e)}"
        
        return sync_wrapper
    
    def _create_async_wrapper(self, tool: OrchestratorTool):
        """Create asynchronous wrapper for orchestrator tool."""
        
        async def async_wrapper(**kwargs) -> str:
            """Asynchronous wrapper for orchestrator tool execution."""
            try:
                result = await tool.execute(**kwargs)
                
                # Convert result to string format expected by LangChain
                if isinstance(result, dict):
                    if 'output' in result:
                        return str(result['output'])
                    elif 'result' in result:
                        return str(result['result'])
                    else:
                        import json
                        return json.dumps(result, indent=2, default=str)
                else:
                    return str(result)
                    
            except Exception as e:
                logger.error(f"Error in async wrapper for {tool.name}: {e}")
                return f"Error: {str(e)}"
        
        return async_wrapper
    
    def get_compatible_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get LangChain compatible version of a tool."""
        return self.compatible_tools.get(tool_name)
    
    def list_compatible_tools(self) -> List[str]:
        """List all tools with LangChain compatibility."""
        return list(self.compatible_tools.keys())
    
    def get_compatibility_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get compatibility metadata for a tool."""
        return self.compatibility_metadata.get(tool_name)


def enhance_tool_categories_with_langchain():
    """Enhance existing tool categories with LangChain compatibility."""
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain not available - skipping compatibility enhancement")
        return
    
    logger.info("Enhancing tool categories with LangChain compatibility")
    
    # Get compatibility manager
    compat_manager = LangChainCompatibilityManager()
    
    # Import specific tool categories to enhance
    enhanced_categories = []
    
    try:
        # Web tools
        from .web_tools import HeadlessBrowserTool, WebSearchTool
        
        web_tools = [HeadlessBrowserTool(), WebSearchTool()]
        for tool in web_tools:
            result = compat_manager.make_tool_compatible(tool, enhance_description=True)
            if result.success:
                # Register with universal registry
                universal_registry.register_langchain_tool(
                    result.langchain_tool,
                    category=ToolCategory.WEB,
                    tags=["web", "browser", "search"],
                    register_as_orchestrator=False  # Already registered as orchestrator
                )
                enhanced_categories.append(f"web:{tool.name}")
        
    except ImportError as e:
        logger.warning(f"Could not enhance web tools: {e}")
    
    try:
        # System tools
        from .system_tools import FileSystemTool, TerminalTool
        
        system_tools = [FileSystemTool(), TerminalTool()]
        for tool in system_tools:
            result = compat_manager.make_tool_compatible(tool, enhance_description=True)
            if result.success:
                universal_registry.register_langchain_tool(
                    result.langchain_tool,
                    category=ToolCategory.SYSTEM,
                    tags=["system", "filesystem", "terminal"],
                    register_as_orchestrator=False
                )
                enhanced_categories.append(f"system:{tool.name}")
        
    except ImportError as e:
        logger.warning(f"Could not enhance system tools: {e}")
    
    try:
        # Data tools
        from .data_tools import DataProcessingTool
        
        data_tool = DataProcessingTool()
        result = compat_manager.make_tool_compatible(data_tool, enhance_description=True)
        if result.success:
            universal_registry.register_langchain_tool(
                result.langchain_tool,
                category=ToolCategory.DATA,
                tags=["data", "processing", "transform"],
                register_as_orchestrator=False
            )
            enhanced_categories.append(f"data:{data_tool.name}")
        
    except ImportError as e:
        logger.warning(f"Could not enhance data tools: {e}")
    
    try:
        # Code execution tools
        from .code_execution import PythonExecutorTool
        
        code_tool = PythonExecutorTool()
        result = compat_manager.make_tool_compatible(code_tool, enhance_description=True)
        if result.success:
            universal_registry.register_langchain_tool(
                result.langchain_tool,
                category=ToolCategory.CODE_EXECUTION,
                tags=["code", "python", "execution"],
                register_as_orchestrator=False
            )
            enhanced_categories.append(f"code:{code_tool.name}")
        
    except ImportError as e:
        logger.warning(f"Could not enhance code execution tools: {e}")
    
    logger.info(f"Enhanced {len(enhanced_categories)} tool categories with LangChain compatibility")
    return compat_manager, enhanced_categories


def create_langchain_tool_collection() -> Optional[Dict[str, BaseTool]]:
    """Create a collection of LangChain-compatible orchestrator tools."""
    if not LANGCHAIN_AVAILABLE:
        return None
    
    # Enhance tool categories
    compat_manager, enhanced_categories = enhance_tool_categories_with_langchain()
    
    if not compat_manager:
        return None
    
    # Return the collection of compatible tools
    return compat_manager.compatible_tools


def demonstrate_langchain_integration():
    """Demonstrate LangChain integration with orchestrator tools."""
    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain not available - cannot demonstrate integration")
        return
    
    logger.info("Demonstrating LangChain integration with orchestrator tools")
    
    # Create tool collection
    langchain_tools = create_langchain_tool_collection()
    
    if not langchain_tools:
        logger.warning("No LangChain tools available for demonstration")
        return
    
    # Show integration statistics
    stats = {
        "total_langchain_tools": len(langchain_tools),
        "tool_names": list(langchain_tools.keys()),
        "integration_success": True
    }
    
    logger.info(f"LangChain integration demonstration complete: {stats}")
    return stats


# Global compatibility manager instance
compatibility_manager = None

if LANGCHAIN_AVAILABLE:
    compatibility_manager = LangChainCompatibilityManager()
    
    # Auto-enhance tool categories
    try:
        enhance_tool_categories_with_langchain()
        logger.info("Auto-enhancement of tool categories completed")
    except Exception as e:
        logger.error(f"Auto-enhancement failed: {e}")


__all__ = [
    "LangChainCompatibilityManager",
    "CompatibilityResult",
    "enhance_tool_categories_with_langchain",
    "create_langchain_tool_collection",
    "demonstrate_langchain_integration",
    "compatibility_manager"
]