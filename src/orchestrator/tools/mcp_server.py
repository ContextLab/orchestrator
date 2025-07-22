"""MCP (Model Context Protocol) server integration for tools."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ToolRegistry, default_registry
from .data_tools import DataProcessingTool
from .system_tools import FileSystemTool, TerminalTool
from .web_tools import HeadlessBrowserTool, WebSearchTool

logger = logging.getLogger(__name__)


class MCPToolServer:
    """MCP server for exposing tools to AI models."""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.registry = tool_registry or default_registry
        self.server_process: Optional[asyncio.subprocess.Process] = None
        self.config_path: Optional[Path] = None
        self.server_runner = None  # For aiohttp server
        self.httpd = None  # For basic HTTP server

    def register_default_tools(self):
        """Register all default tools."""
        tools = [
            HeadlessBrowserTool(),
            WebSearchTool(),
            TerminalTool(),
            FileSystemTool(),
            DataProcessingTool(),
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

            # Start actual MCP server process
            try:
                # Import aiohttp for the actual server
                import aiohttp
                from aiohttp import web
                
                # Create the web application
                app = web.Application()
                
                # Add routes for MCP protocol
                async def handle_tools_list(request):
                    """Handle tools list request."""
                    return web.json_response(self.get_tools_manifest())
                
                async def handle_tool_call(request):
                    """Handle tool call request."""
                    data = await request.json()
                    tool_name = data.get("tool")
                    arguments = data.get("arguments", {})
                    result = await self.handle_tool_call(tool_name, arguments)
                    return web.json_response(result)
                
                app.router.add_get("/tools", handle_tools_list)
                app.router.add_post("/tools/call", handle_tool_call)
                
                # Create the server runner
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, 'localhost', port)
                await site.start()
                
                # Store the runner for cleanup
                self.server_runner = runner
                
                logger.info(f"MCP tool server started on http://localhost:{port}")
                logger.info(f"Tools available: {', '.join(self.registry.list_tools())}")
                
                return True
                
            except ImportError:
                # Fallback to a basic HTTP server if aiohttp is not available
                import http.server
                import socketserver
                import threading
                
                class MCPHandler(http.server.BaseHTTPRequestHandler):
                    def do_GET(self):
                        if self.path == "/tools":
                            self.send_response(200)
                            self.send_header("Content-type", "application/json")
                            self.end_headers()
                            manifest = json.dumps(config)
                            self.wfile.write(manifest.encode())
                    
                    def do_POST(self):
                        if self.path == "/tools/call":
                            content_length = int(self.headers['Content-Length'])
                            post_data = self.rfile.read(content_length)
                            json.loads(post_data.decode())
                            
                            # Simple synchronous handling for basic server
                            self.send_response(200)
                            self.send_header("Content-type", "application/json")
                            self.end_headers()
                            response = json.dumps({"success": True, "result": {"message": "Tool call received"}})
                            self.wfile.write(response.encode())
                    
                    def log_message(self, format, *args):
                        # Suppress default logging
                        pass
                
                # Start server in a thread
                handler = MCPHandler
                httpd = socketserver.TCPServer(("", port), handler)
                server_thread = threading.Thread(target=httpd.serve_forever)
                server_thread.daemon = True
                server_thread.start()
                
                self.httpd = httpd
                
                logger.info(f"MCP tool server started (basic HTTP) on port {port}")
                logger.info(f"Tools available: {', '.join(self.registry.list_tools())}")
                
                return True

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False

    async def stop_server(self):
        """Stop the MCP server."""
        # Stop aiohttp server if running
        if self.server_runner:
            await self.server_runner.cleanup()
            self.server_runner = None
        
        # Stop basic HTTP server if running
        if self.httpd:
            self.httpd.shutdown()
            self.httpd = None
        
        # Stop any subprocess if running
        if self.server_process:
            self.server_process.terminate()
            await self.server_process.wait()
            self.server_process = None

        # Clean up config file
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
