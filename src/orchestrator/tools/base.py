"""Base tool classes and registry."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass

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
    from .report_tools import PDFCompilerTool, ReportGeneratorTool
    from .system_tools import FileSystemTool, TerminalTool
    from .web_tools import HeadlessBrowserTool, WebSearchTool
    from .llm_tools import TaskDelegationTool, MultiModelRoutingTool, PromptOptimizationTool
    from .user_interaction_tools import UserPromptTool, ApprovalGateTool, FeedbackCollectionTool
    from .pipeline_recursion_tools import PipelineExecutorTool, RecursionControlTool
    from .multimodal_tools import (
        ImageAnalysisTool,
        ImageGenerationTool,
        AudioProcessingTool,
        VideoProcessingTool,
    )
    from .mcp_tools import MCPServerTool, MCPMemoryTool, MCPResourceTool

    tools = [
        HeadlessBrowserTool(),
        WebSearchTool(),
        TerminalTool(),
        FileSystemTool(),
        DataProcessingTool(),
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
    ]

    for tool in tools:
        default_registry.register(tool)

    return len(tools)


# Auto-register default tools
_registered = register_default_tools()
