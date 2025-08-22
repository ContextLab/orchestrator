"""Base tool classes and registry."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.template_manager import TemplateManager
    from ..core.unified_template_resolver import UnifiedTemplateResolver, TemplateResolutionContext


@dataclass
class ToolParameter:
    """Tool parameter definition."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """Base class for all tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters: List[ToolParameter] = []

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters (with automatic template rendering)."""
        # Extract template resolution components from kwargs
        template_manager = kwargs.pop('template_manager', None)
        unified_resolver = kwargs.pop('unified_template_resolver', None)
        resolution_context = kwargs.pop('template_resolution_context', None)
        
        # Use unified template resolver if available (preferred method)
        if unified_resolver and resolution_context:
            rendered_kwargs = unified_resolver.resolve_before_tool_execution(
                tool_name=self.name,
                tool_parameters=kwargs,
                context=resolution_context
            )
        # Fallback to legacy template manager method
        elif template_manager:
            rendered_kwargs = self._render_parameters(kwargs, template_manager)
        else:
            rendered_kwargs = kwargs
        
        # Validate rendered parameters
        self.validate_parameters(rendered_kwargs)
        
        # Pass template resolution components to implementation if it needs runtime rendering
        if template_manager and self.name == "filesystem":
            rendered_kwargs['_template_manager'] = template_manager
        if unified_resolver:
            rendered_kwargs['_unified_resolver'] = unified_resolver
        if resolution_context:
            rendered_kwargs['_resolution_context'] = resolution_context
        
        # Execute the actual implementation
        return await self._execute_impl(**rendered_kwargs)
    
    @abstractmethod
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool implementation (to be overridden by subclasses)."""
        pass
    
    def _render_parameters(self, kwargs: Dict[str, Any], template_manager: 'TemplateManager') -> Dict[str, Any]:
        """
        Automatically render all template strings in parameters using deep template rendering.
        
        This ensures that all string values containing Jinja2 templates are properly rendered
        before being passed to the tool implementation.
        
        Args:
            kwargs: Tool parameters that may contain template strings
            template_manager: Template manager with registered context
            
        Returns:
            Parameters with all templates rendered
        """
        # Use deep_render to handle nested structures and all template syntax
        return template_manager.deep_render(kwargs)

    def add_parameter(
        self,
        name: str,
        type: str,
        description: str,
        required: bool = True,
        default: Any = None,
    ):
        """Add a parameter to the tool."""
        param = ToolParameter(name, type, description, required, default)
        self.parameters.append(param)

    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for MCP."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    param.name: {"type": param.type, "description": param.description}
                    for param in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required],
            },
        }

    def validate_parameters(self, kwargs: Dict[str, Any]) -> None:
        """Validate that required parameters are provided."""
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(
                    f"Required parameter '{param.name}' not provided for tool '{self.name}'"
                )


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools (for MCP)."""
        return [tool.get_schema() for tool in self.tools.values()]

    async def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")

        tool.validate_parameters(kwargs)
        return await tool.execute(**kwargs)


# Global tool registry
default_registry = ToolRegistry()


def register_default_tools():
    """Register all default tools with the global registry."""
    # Import here to avoid circular imports
    from .data_tools import DataProcessingTool
    from .validation import ValidationTool
    from .report_tools import PDFCompilerTool, ReportGeneratorTool
    from .system_tools import FileSystemTool, TerminalTool
    from .web_tools import HeadlessBrowserTool, WebSearchTool
    from .llm_tools import (
        TaskDelegationTool,
        MultiModelRoutingTool,
        PromptOptimizationTool,
    )
    from .user_interaction_tools import (
        UserPromptTool,
        ApprovalGateTool,
        FeedbackCollectionTool,
    )
    from .pipeline_recursion_tools import PipelineExecutorTool, RecursionControlTool
    from .multimodal_tools import (
        ImageAnalysisTool,
        ImageGenerationTool,
        AudioProcessingTool,
        VideoProcessingTool,
    )
    from .mcp_tools import MCPServerTool, MCPMemoryTool, MCPResourceTool
    from .code_execution import PythonExecutorTool
    from .auto_debugger_wrapper import AutoDebuggerTool

    tools = [
        HeadlessBrowserTool(),
        WebSearchTool(),
        TerminalTool(),
        FileSystemTool(),
        DataProcessingTool(),
        ValidationTool(),
        ReportGeneratorTool(),
        PDFCompilerTool(),
        TaskDelegationTool(),
        MultiModelRoutingTool(),
        PromptOptimizationTool(),
        UserPromptTool(),
        ApprovalGateTool(),
        FeedbackCollectionTool(),
        PipelineExecutorTool(),
        RecursionControlTool(),
        ImageAnalysisTool(),
        ImageGenerationTool(),
        AudioProcessingTool(),
        VideoProcessingTool(),
        MCPServerTool(),
        MCPMemoryTool(),
        MCPResourceTool(),
        PythonExecutorTool(),
        AutoDebuggerTool(),
    ]

    for tool in tools:
        default_registry.register(tool)

    return len(tools)


# Auto-register default tools
_registered = register_default_tools()
