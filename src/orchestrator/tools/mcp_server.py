"""MCP (Model Context Protocol) server integration for tools."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ToolRegistry, default_registry
from .data_tools import DataProcessingTool, ValidationTool
from .system_tools import FileSystemTool, TerminalTool
from .web_tools import HeadlessBrowserTool, WebSearchTool

logger = logging.getLogger(__name__)


class MCPToolServer:
    """MCP server for exposing tools to AI models."""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.registry = tool_registry or default_registry
        self.server_process: Optional[asyncio.subprocess.Process] = None
        self.config_path: Optional[Path] = None

    def register_default_tools(self):
        """Register all default tools."""
        tools = [
            HeadlessBrowserTool(),
            WebSearchTool(),
            TerminalTool(),
            FileSystemTool(),
            DataProcessingTool(),
            ValidationTool(),
        ]

        for tool in tools:
            self.registry.register(tool)

        logger.info(f"Registered {len(tools)} default tools")

    def get_mcp_config(self) -> Dict[str, Any]:
        """Generate MCP server configuration."""
        return {
            "mcpServers": {
                "orchestrator-tools": {
                    "command": "python",
                    "args": ["-m", "orchestrator.tools.mcp_server"],
                    "env": {"ORCHESTRATOR_TOOLS": "enabled"},
                }
            }
        }

    def get_tools_manifest(self) -> Dict[str, Any]:
        """Get tools manifest for MCP."""
        return {
            "tools": self.registry.get_schemas(),
            "version": "1.0.0",
            "capabilities": {"tools": {"listChanged": True}},
        }

    async def start_server(self, port: int = 8000) -> bool:
        """Start the MCP server."""
        try:
            # Create temporary config file
            config = self.get_tools_manifest()
            config_path = Path.cwd() / "mcp_tools_config.json"

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            self.config_path = config_path

            # In a real implementation, this would start an actual MCP server
            # For now, we'll simulate it
            logger.info(f"MCP tool server started (simulated) on port {port}")
            logger.info(f"Tools available: {', '.join(self.registry.list_tools())}")

            return True

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False

    async def stop_server(self):
        """Stop the MCP server."""
        if self.server_process:
            self.server_process.terminate()
            await self.server_process.wait()
            self.server_process = None

        if self.config_path and self.config_path.exists():
            self.config_path.unlink()
            self.config_path = None

        logger.info("MCP tool server stopped")

    async def handle_tool_call(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a tool call from an AI model."""
        try:
            result = await self.registry.execute_tool(tool_name, **arguments)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Tool call failed for {tool_name}: {e}")
            return {"success": False, "error": str(e)}


class ToolDetector:
    """Automatically detect and register tools needed by a pipeline."""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.registry = tool_registry or default_registry

    def detect_tools_from_yaml(self, pipeline_def: Dict[str, Any]) -> List[str]:
        """Detect tools needed from pipeline definition."""
        required_tools = set()

        # Check steps for tool references
        steps = pipeline_def.get("steps", [])
        for step in steps:
            # Direct tool specification
            if "tool" in step:
                tool_name = step["tool"]
                required_tools.add(tool_name)

            # Tool inference from action
            action = step.get("action", "")
            if action.startswith("!"):
                # Shell command
                required_tools.add("terminal")
            elif "search" in action.lower() or "web" in action.lower():
                required_tools.add("headless-browser")
            elif (
                "file" in action.lower()
                or "read" in action.lower()
                or "write" in action.lower()
            ):
                required_tools.add("filesystem")
            elif "validate" in action.lower() or "check" in action.lower():
                required_tools.add("validation")
            elif "process" in action.lower() or "transform" in action.lower():
                required_tools.add("data-processing")

        return list(required_tools)

    def ensure_tools_available(self, required_tools: List[str]) -> Dict[str, bool]:
        """Ensure required tools are available."""
        availability = {}

        for tool_name in required_tools:
            if self.registry.get_tool(tool_name):
                availability[tool_name] = True
            else:
                # Try to auto-register tool
                success = self._auto_register_tool(tool_name)
                availability[tool_name] = success

        return availability

    def _auto_register_tool(self, tool_name: str) -> bool:
        """Attempt to auto-register a tool by name."""
        tool_mapping = {
            "headless-browser": HeadlessBrowserTool,
            "web-search": WebSearchTool,
            "terminal": TerminalTool,
            "filesystem": FileSystemTool,
            "data-processing": DataProcessingTool,
            "validation": ValidationTool,
        }

        if tool_name in tool_mapping:
            tool_class = tool_mapping[tool_name]
            tool = tool_class()
            self.registry.register(tool)
            logger.info(f"Auto-registered tool: {tool_name}")
            return True

        logger.warning(f"Unknown tool: {tool_name}")
        return False


# Global instances
default_mcp_server = MCPToolServer()
default_tool_detector = ToolDetector()


async def main():
    """Main function for running MCP server."""
    server = MCPToolServer()
    server.register_default_tools()

    try:
        await server.start_server()

        # Keep server running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("Shutting down MCP server...")
    finally:
        await server.stop_server()


if __name__ == "__main__":
    asyncio.run(main())
