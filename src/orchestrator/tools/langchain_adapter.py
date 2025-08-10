"""
LangChain Tool Adapter System - Issue #203 Phase 1

Bidirectional adapter system that allows:
- Existing orchestrator tools to be used as LangChain tools
- LangChain tools to be registered in orchestrator registry

This creates a universal tool interface supporting both ecosystems.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
import inspect

from .base import Tool as OrchestratorTool, ToolParameter

logger = logging.getLogger(__name__)

# Import LangChain components with graceful fallback
try:
    from langchain.tools import BaseTool, StructuredTool
    from langchain.callbacks.manager import CallbackManagerForToolRun
    from pydantic import BaseModel, Field, create_model
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available - install with: pip install langchain")
    LANGCHAIN_AVAILABLE = False
    
    # Create mock classes for type hints when LangChain not available
    class BaseTool:
        pass
    
    class StructuredTool:
        pass
    
    class CallbackManagerForToolRun:
        pass


@dataclass
class ToolExecutionResult:
    """Standardized tool execution result format."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LangChainToolAdapter:
    """
    Adapter to use orchestrator tools as LangChain tools.
    
    This creates a StructuredTool wrapper around orchestrator tools.
    """
    
    def __init__(self, orchestrator_tool: OrchestratorTool):
        """
        Initialize adapter with orchestrator tool.
        
        Args:
            orchestrator_tool: The orchestrator tool to adapt
        """
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain not available - install with: pip install langchain")
        
        self.orchestrator_tool = orchestrator_tool
        self._langchain_tool = self._create_structured_tool()
    
    def _create_structured_tool(self) -> StructuredTool:
        """Create a LangChain StructuredTool from the orchestrator tool."""
        # Create input schema
        input_model = self._create_input_model()
        
        # Create the structured tool
        return StructuredTool.from_function(
            name=self.orchestrator_tool.name,
            description=self.orchestrator_tool.description,
            func=self._sync_wrapper,
            coroutine=self._async_wrapper,
            args_schema=input_model
        )
    
    def _create_input_model(self) -> BaseModel:
        """Create Pydantic model for orchestrator tool parameters."""
        if not self.orchestrator_tool.parameters:
            return create_model(f"{self.orchestrator_tool.name}Input")
        
        field_definitions = {}
        for param in self.orchestrator_tool.parameters:
            python_type = self._map_type(param.type)
            if param.required:
                field_definitions[param.name] = (python_type, Field(description=param.description))
            else:
                field_definitions[param.name] = (python_type, Field(default=param.default, description=param.description))
        
        return create_model(f"{self.orchestrator_tool.name}Input", **field_definitions)
    
    def _map_type(self, orchestrator_type: str):
        """Map orchestrator types to Python types."""
        type_map = {
            "string": str,
            "integer": int,
            "float": float,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        return type_map.get(orchestrator_type.lower(), str)
    
    def _sync_wrapper(self, **kwargs) -> str:
        """Synchronous wrapper."""
        import asyncio
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Use thread executor if event loop is already running
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self.orchestrator_tool.execute(**kwargs)
                    )
                    result = future.result()
            else:
                # Create new event loop
                result = asyncio.run(self.orchestrator_tool.execute(**kwargs))
        except RuntimeError:
            # Fallback: create new event loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    self.orchestrator_tool.execute(**kwargs)
                )
                result = future.result()
        
        return self._format_result(result)
    
    async def _async_wrapper(self, **kwargs) -> str:
        """Asynchronous wrapper."""
        result = await self.orchestrator_tool.execute(**kwargs)
        return self._format_result(result)
    
    def _format_result(self, result) -> str:
        """Format orchestrator result for LangChain."""
        if isinstance(result, dict):
            if 'output' in result:
                return str(result['output'])
            elif 'result' in result:
                return str(result['result'])
            else:
                import json
                return json.dumps(result, default=str)
        return str(result)
    
    @property
    def name(self) -> str:
        """Tool name."""
        return self._langchain_tool.name
    
    @property
    def description(self) -> str:
        """Tool description."""
        return self._langchain_tool.description
    
    def get_langchain_tool(self) -> StructuredTool:
        """Get the underlying LangChain StructuredTool."""
        return self._langchain_tool
    
    def _run(
        self,
        query: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run via the structured tool."""
        # Add query to kwargs if it's provided
        if query:
            kwargs['query'] = query
        return self._langchain_tool.run(kwargs, run_manager=run_manager)
    
    async def _arun(
        self,
        query: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run async via the structured tool."""
        # Add query to kwargs if it's provided
        if query:
            kwargs['query'] = query
        return await self._langchain_tool.arun(kwargs, run_manager=run_manager)
    
    def _convert_langchain_to_orchestrator_params(
        self, query: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Convert LangChain parameters to orchestrator format.
        
        Args:
            query: Main query parameter
            **kwargs: Additional LangChain parameters
            
        Returns:
            Parameters in orchestrator format
        """
        orchestrator_params = kwargs.copy()
        
        # Add query as primary parameter if tool expects it
        tool_params = {p.name for p in self.orchestrator_tool.parameters}
        
        # Common parameter mappings
        if query and not orchestrator_params:
            # If no specific params, try common parameter names
            if 'query' in tool_params:
                orchestrator_params['query'] = query
            elif 'input' in tool_params:
                orchestrator_params['input'] = query
            elif 'text' in tool_params:
                orchestrator_params['text'] = query
            elif 'data' in tool_params:
                orchestrator_params['data'] = query
            elif len(tool_params) == 1:
                # If tool has single parameter, use it
                param_name = next(iter(tool_params))
                orchestrator_params[param_name] = query
        
        return orchestrator_params
    
    def _convert_orchestrator_to_langchain_result(self, result: Dict[str, Any]) -> str:
        """
        Convert orchestrator tool result to LangChain string format.
        
        Args:
            result: Orchestrator tool result
            
        Returns:
            Result formatted for LangChain
        """
        if isinstance(result, dict):
            # Try to extract meaningful output
            if 'output' in result:
                output = result['output']
            elif 'result' in result:
                output = result['result']
            elif 'data' in result:
                output = result['data']
            elif 'content' in result:
                output = result['content']
            else:
                # Return full result as JSON
                output = result
            
            # Convert to string
            if isinstance(output, (dict, list)):
                return json.dumps(output, indent=2, default=str)
            else:
                return str(output)
        else:
            return str(result)


class OrchestratorToolAdapter(OrchestratorTool):
    """
    Adapter to use LangChain tools in orchestrator registry.
    
    This allows LangChain tools to be registered and used
    in orchestrator pipelines and workflows.
    """
    
    def __init__(self, langchain_tool: BaseTool):
        """
        Initialize adapter with LangChain tool.
        
        Args:
            langchain_tool: The LangChain tool to adapt
        """
        self.langchain_tool = langchain_tool
        
        # Initialize as orchestrator tool
        super().__init__(
            name=langchain_tool.name,
            description=langchain_tool.description
        )
        
        # Auto-detect parameters from LangChain tool
        self._detect_parameters()
    
    def _detect_parameters(self):
        """Auto-detect parameters from LangChain tool."""
        # Check if tool has schema
        if hasattr(self.langchain_tool, 'args_schema') and self.langchain_tool.args_schema:
            schema = self.langchain_tool.args_schema.schema()
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            for param_name, param_info in properties.items():
                param_type = param_info.get('type', 'string')
                param_desc = param_info.get('description', f'{param_name} parameter')
                is_required = param_name in required
                
                self.add_parameter(
                    name=param_name,
                    type=param_type,
                    description=param_desc,
                    required=is_required
                )
        else:
            # Default parameters for tools without schema
            self.add_parameter(
                name="query",
                type="string",
                description="Input query or data for the tool",
                required=False,
                default=""
            )
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the LangChain tool.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Structured result dictionary
        """
        try:
            # Convert orchestrator params to LangChain format
            langchain_params = self._convert_orchestrator_to_langchain_params(kwargs)
            
            # Execute LangChain tool
            if hasattr(self.langchain_tool, 'arun'):
                # Use async method if available
                result = await self.langchain_tool.arun(langchain_params)
            elif hasattr(self.langchain_tool, '_arun'):
                # Use private async method with empty config
                result = await self.langchain_tool._arun(**langchain_params)
            else:
                # Use sync method in thread
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: self.langchain_tool.run(langchain_params)
                )
            
            # Return structured result
            return {
                "success": True,
                "output": result,
                "tool_name": self.name,
                "metadata": {
                    "langchain_tool": True,
                    "original_type": type(self.langchain_tool).__name__
                }
            }
            
        except Exception as e:
            logger.error(f"Error executing LangChain tool {self.name}: {e}")
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "tool_name": self.name,
                "metadata": {
                    "langchain_tool": True,
                    "original_type": type(self.langchain_tool).__name__
                }
            }
    
    def _convert_orchestrator_to_langchain_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert orchestrator parameters to LangChain format.
        
        Args:
            kwargs: Orchestrator parameters
            
        Returns:
            Parameters in LangChain format
        """
        # For structured tools, pass parameters as is
        langchain_params = kwargs.copy()
        
        # Handle common parameter conversions only if needed
        # LangChain tools often expect 'query' as main parameter for simple tools
        if len(kwargs) == 1 and 'query' not in kwargs:
            # Single parameter conversion for simple tools
            param_value = next(iter(kwargs.values()))
            if 'input' in kwargs:
                langchain_params['query'] = kwargs['input']
            elif 'text' in kwargs:
                langchain_params['query'] = kwargs['text']
            elif 'data' in kwargs:
                langchain_params['query'] = str(kwargs['data'])
        
        return langchain_params


class ToolAdapterFactory:
    """Factory for creating tool adapters."""
    
    @staticmethod
    def create_langchain_adapter(orchestrator_tool: OrchestratorTool) -> LangChainToolAdapter:
        """
        Create LangChain adapter for orchestrator tool.
        
        Args:
            orchestrator_tool: Tool to adapt
            
        Returns:
            LangChain adapter
        """
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain not available - install with: pip install langchain")
        
        return LangChainToolAdapter(orchestrator_tool)
    
    @staticmethod
    def create_orchestrator_adapter(langchain_tool: BaseTool) -> OrchestratorToolAdapter:
        """
        Create orchestrator adapter for LangChain tool.
        
        Args:
            langchain_tool: LangChain tool to adapt
            
        Returns:
            Orchestrator adapter
        """
        return OrchestratorToolAdapter(langchain_tool)
    
    @staticmethod
    def detect_tool_type(tool: Any) -> str:
        """
        Detect tool type.
        
        Args:
            tool: Tool to analyze
            
        Returns:
            Tool type string
        """
        if isinstance(tool, OrchestratorTool):
            return "orchestrator"
        elif LANGCHAIN_AVAILABLE and isinstance(tool, BaseTool):
            return "langchain"
        else:
            return "unknown"


# Utility functions for backward compatibility
def make_langchain_tool(orchestrator_tool: OrchestratorTool) -> StructuredTool:
    """
    Convert orchestrator tool to LangChain tool.
    
    Args:
        orchestrator_tool: Tool to convert
        
    Returns:
        LangChain StructuredTool
    """
    adapter = ToolAdapterFactory.create_langchain_adapter(orchestrator_tool)
    return adapter.get_langchain_tool()


def make_orchestrator_tool(langchain_tool: BaseTool) -> OrchestratorToolAdapter:
    """
    Convert LangChain tool to orchestrator tool.
    
    Args:
        langchain_tool: Tool to convert
        
    Returns:
        Orchestrator-compatible tool
    """
    return ToolAdapterFactory.create_orchestrator_adapter(langchain_tool)


# Export main classes and functions
__all__ = [
    "LangChainToolAdapter",
    "OrchestratorToolAdapter", 
    "ToolAdapterFactory",
    "ToolExecutionResult",
    "make_langchain_tool",
    "make_orchestrator_tool",
    "LANGCHAIN_AVAILABLE"
]