"""MCP (Model Context Protocol) tools for interacting with MCP servers."""

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Tool


@dataclass
class MCPConnection:
    """MCP server connection information."""

    server_name: str
    command: str
    args: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    process: Optional[subprocess.Popen] = None
    connected: bool = False
    capabilities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPToolInfo:
    """Information about an MCP tool."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    server: str


@dataclass
class MCPResourceInfo:
    """Information about an MCP resource."""

    uri: str
    name: str
    description: Optional[str] = None
    mime_type: Optional[str] = None
    server: str = ""


class MCPServerTool(Tool):
    """Connect to and interact with MCP servers."""

    def __init__(self):
        super().__init__(
            name="mcp-server", description="Connect to MCP servers and execute their tools"
        )
        self.add_parameter(
            "action", "string", "Action to perform: connect, list_tools, execute_tool, disconnect"
        )
        self.add_parameter(
            "server_config", "object", "Server configuration for connect action", required=False
        )
        self.add_parameter("server_name", "string", "Name of the server", required=False)
        self.add_parameter("tool_name", "string", "Name of the tool to execute", required=False)
        self.add_parameter("tool_params", "object", "Parameters for tool execution", required=False)

        self.logger = logging.getLogger(__name__)
        self.connections: Dict[str, MCPConnection] = {}

    async def _connect_server(self, server_name: str, config: Dict[str, Any]) -> bool:
        """Connect to an MCP server."""
        if server_name in self.connections and self.connections[server_name].connected:
            return True

        # Extract configuration
        command = config.get("command", "npx")
        args = config.get("args", [])
        env = os.environ.copy()
        env.update(config.get("env", {}))

        # Create connection
        connection = MCPConnection(server_name=server_name, command=command, args=args, env=env)

        try:
            # Start the MCP server process
            self.logger.info(f"Starting MCP server '{server_name}': {command} {' '.join(args)}")

            # For demonstration, we'll simulate a connection
            # In production, this would start the actual MCP server process
            connection.connected = True
            connection.capabilities = {"tools": True, "resources": True, "prompts": True}

            self.connections[server_name] = connection
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server '{server_name}': {e}")
            return False

    async def _list_server_tools(self, server_name: str) -> List[MCPToolInfo]:
        """List available tools from a server."""
        if server_name not in self.connections or not self.connections[server_name].connected:
            return []

        # In production, this would query the actual MCP server
        # For now, return example tools
        return [
            MCPToolInfo(
                name="search",
                description="Search for information",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
                server=server_name,
            ),
            MCPToolInfo(
                name="fetch_resource",
                description="Fetch a resource by URI",
                input_schema={
                    "type": "object",
                    "properties": {"uri": {"type": "string"}},
                    "required": ["uri"],
                },
                server=server_name,
            ),
        ]

    async def _execute_tool(
        self, server_name: str, tool_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool on an MCP server."""
        if server_name not in self.connections or not self.connections[server_name].connected:
            raise ValueError(f"Server '{server_name}' not connected")

        # In production, this would send the request to the MCP server
        # For demonstration, return simulated results
        self.logger.info(
            f"Executing tool '{tool_name}' on server '{server_name}' with params: {params}"
        )

        # Simulate tool execution
        if tool_name == "search":
            return {
                "results": [
                    {
                        "title": f"Result for {params.get('query', 'unknown')}",
                        "url": "http://example.com",
                    },
                    {"title": "Another result", "url": "http://example.org"},
                ],
                "total": 2,
            }
        elif tool_name == "fetch_resource":
            return {
                "content": f"Resource content for URI: {params.get('uri', 'unknown')}",
                "metadata": {"fetched_at": time.time()},
            }
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute MCP server operations."""
        action = kwargs["action"]

        # Validate action
        valid_actions = ["connect", "list_tools", "execute_tool", "disconnect"]
        if action not in valid_actions:
            return {
                "success": False,
                "error": f"Invalid action: {action}. Must be one of {valid_actions}",
            }

        try:
            if action == "connect":
                server_config = kwargs.get("server_config")
                if not server_config:
                    return {"success": False, "error": "server_config required for connect action"}

                server_name = kwargs.get("server_name", "default")
                connected = await self._connect_server(server_name, server_config)

                return {
                    "success": connected,
                    "server_name": server_name,
                    "connected": connected,
                    "capabilities": self.connections[server_name].capabilities if connected else {},
                }

            elif action == "list_tools":
                server_name = kwargs.get("server_name", "default")
                tools = await self._list_server_tools(server_name)

                return {
                    "success": True,
                    "server_name": server_name,
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.input_schema,
                        }
                        for tool in tools
                    ],
                }

            elif action == "execute_tool":
                server_name = kwargs.get("server_name", "default")
                tool_name = kwargs.get("tool_name")
                if not tool_name:
                    return {"success": False, "error": "tool_name required for execute_tool action"}

                tool_params = kwargs.get("tool_params", {})
                result = await self._execute_tool(server_name, tool_name, tool_params)

                return {
                    "success": True,
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "result": result,
                }

            elif action == "disconnect":
                server_name = kwargs.get("server_name", "default")
                if server_name in self.connections:
                    connection = self.connections[server_name]
                    if connection.process:
                        connection.process.terminate()
                    connection.connected = False
                    del self.connections[server_name]

                return {
                    "success": True,
                    "server_name": server_name,
                    "message": f"Disconnected from server '{server_name}'",
                }

        except Exception as e:
            self.logger.error(f"MCP server error: {e}")
            return {"success": False, "error": str(e)}


class MCPMemoryTool(Tool):
    """Manage memory and context for MCP interactions."""

    def __init__(self):
        super().__init__(
            name="mcp-memory", description="Store and retrieve context for MCP server interactions"
        )
        self.add_parameter("action", "string", "Action: store, retrieve, list, clear")
        self.add_parameter("key", "string", "Memory key", required=False)
        self.add_parameter("value", "any", "Value to store", required=False)
        self.add_parameter("namespace", "string", "Memory namespace", default="default")
        self.add_parameter("ttl", "integer", "Time to live in seconds (0 for permanent)", default=0)

        self.logger = logging.getLogger(__name__)
        self.memory_store: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def _get_namespace(self, namespace: str) -> Dict[str, Any]:
        """Get or create namespace."""
        if namespace not in self.memory_store:
            self.memory_store[namespace] = {}
            self.metadata[namespace] = {}
        return self.memory_store[namespace]

    def _is_expired(self, namespace: str, key: str) -> bool:
        """Check if a key has expired."""
        if namespace not in self.metadata or key not in self.metadata[namespace]:
            return False

        meta = self.metadata[namespace][key]
        if meta.get("ttl", 0) == 0:
            return False

        expires_at = meta.get("stored_at", 0) + meta.get("ttl", 0)
        return time.time() > expires_at

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute memory operations."""
        action = kwargs["action"]
        namespace = kwargs.get("namespace", "default")

        # Validate action
        valid_actions = ["store", "retrieve", "list", "clear"]
        if action not in valid_actions:
            return {
                "success": False,
                "error": f"Invalid action: {action}. Must be one of {valid_actions}",
            }

        try:
            if action == "store":
                key = kwargs.get("key")
                if not key:
                    return {"success": False, "error": "key required for store action"}

                value = kwargs.get("value")
                ttl = kwargs.get("ttl", 0)

                # Store value
                ns = self._get_namespace(namespace)
                ns[key] = value

                # Store metadata
                if namespace not in self.metadata:
                    self.metadata[namespace] = {}

                self.metadata[namespace][key] = {
                    "stored_at": time.time(),
                    "ttl": ttl,
                    "type": type(value).__name__,
                }

                return {
                    "success": True,
                    "namespace": namespace,
                    "key": key,
                    "stored": True,
                    "ttl": ttl,
                }

            elif action == "retrieve":
                key = kwargs.get("key")
                if not key:
                    return {"success": False, "error": "key required for retrieve action"}

                ns = self._get_namespace(namespace)

                # Check expiration
                if self._is_expired(namespace, key):
                    # Remove expired key
                    if key in ns:
                        del ns[key]
                    if key in self.metadata[namespace]:
                        del self.metadata[namespace][key]

                    return {
                        "success": True,
                        "namespace": namespace,
                        "key": key,
                        "found": False,
                        "expired": True,
                    }

                if key in ns:
                    return {
                        "success": True,
                        "namespace": namespace,
                        "key": key,
                        "value": ns[key],
                        "found": True,
                        "metadata": self.metadata[namespace].get(key, {}),
                    }
                else:
                    return {"success": True, "namespace": namespace, "key": key, "found": False}

            elif action == "list":
                ns = self._get_namespace(namespace)

                # Filter out expired keys
                valid_keys = []
                for key in list(ns.keys()):
                    if self._is_expired(namespace, key):
                        del ns[key]
                        if key in self.metadata[namespace]:
                            del self.metadata[namespace][key]
                    else:
                        valid_keys.append(key)

                return {
                    "success": True,
                    "namespace": namespace,
                    "keys": valid_keys,
                    "count": len(valid_keys),
                }

            elif action == "clear":
                if namespace in self.memory_store:
                    self.memory_store[namespace].clear()
                if namespace in self.metadata:
                    self.metadata[namespace].clear()

                return {"success": True, "namespace": namespace, "cleared": True}

        except Exception as e:
            self.logger.error(f"MCP memory error: {e}")
            return {"success": False, "error": str(e)}


class MCPResourceTool(Tool):
    """Access and manage MCP resources."""

    def __init__(self):
        super().__init__(name="mcp-resource", description="Access resources from MCP servers")
        self.add_parameter("action", "string", "Action: list, read, subscribe, unsubscribe")
        self.add_parameter("server_name", "string", "MCP server name", default="default")
        self.add_parameter("uri", "string", "Resource URI", required=False)
        self.add_parameter(
            "subscription_id", "string", "Subscription ID for unsubscribe", required=False
        )

        self.logger = logging.getLogger(__name__)
        self.resources: Dict[str, List[MCPResourceInfo]] = {}
        self.subscriptions: Dict[str, Dict[str, Any]] = {}

    def _get_server_resources(self, server_name: str) -> List[MCPResourceInfo]:
        """Get resources for a server."""
        # In production, this would query the MCP server
        # For demonstration, return example resources
        if server_name not in self.resources:
            self.resources[server_name] = [
                MCPResourceInfo(
                    uri="file:///data/config.json",
                    name="Configuration",
                    description="System configuration file",
                    mime_type="application/json",
                    server=server_name,
                ),
                MCPResourceInfo(
                    uri="memory://context/conversation",
                    name="Conversation Context",
                    description="Current conversation memory",
                    mime_type="application/json",
                    server=server_name,
                ),
                MCPResourceInfo(
                    uri="api://weather/current",
                    name="Current Weather",
                    description="Real-time weather data",
                    mime_type="application/json",
                    server=server_name,
                ),
            ]

        return self.resources[server_name]

    async def _read_resource(self, server_name: str, uri: str) -> Dict[str, Any]:
        """Read a resource from MCP server."""
        # In production, this would fetch from the MCP server
        # For demonstration, return simulated content

        if uri.startswith("file://"):
            return {
                "content": {"setting1": "value1", "setting2": "value2"},
                "metadata": {"size": 1024, "modified": time.time()},
            }
        elif uri.startswith("memory://"):
            return {
                "content": {"messages": ["Hello", "How can I help?"], "context": "greeting"},
                "metadata": {"entries": 2},
            }
        elif uri.startswith("api://"):
            return {
                "content": {"temperature": 72, "conditions": "sunny", "humidity": 45},
                "metadata": {"source": "weather-api", "timestamp": time.time()},
            }
        else:
            return {"error": f"Unknown URI scheme: {uri}"}

    async def _subscribe_resource(self, server_name: str, uri: str) -> str:
        """Subscribe to resource updates."""
        subscription_id = f"sub_{server_name}_{len(self.subscriptions)}"

        self.subscriptions[subscription_id] = {
            "server": server_name,
            "uri": uri,
            "created_at": time.time(),
            "active": True,
        }

        return subscription_id

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute resource operations."""
        action = kwargs["action"]
        server_name = kwargs.get("server_name", "default")

        # Validate action
        valid_actions = ["list", "read", "subscribe", "unsubscribe"]
        if action not in valid_actions:
            return {
                "success": False,
                "error": f"Invalid action: {action}. Must be one of {valid_actions}",
            }

        try:
            if action == "list":
                resources = self._get_server_resources(server_name)

                return {
                    "success": True,
                    "server_name": server_name,
                    "resources": [
                        {
                            "uri": r.uri,
                            "name": r.name,
                            "description": r.description,
                            "mime_type": r.mime_type,
                        }
                        for r in resources
                    ],
                    "count": len(resources),
                }

            elif action == "read":
                uri = kwargs.get("uri")
                if not uri:
                    return {"success": False, "error": "uri required for read action"}

                content = await self._read_resource(server_name, uri)

                return {
                    "success": True,
                    "server_name": server_name,
                    "uri": uri,
                    "resource": content,
                }

            elif action == "subscribe":
                uri = kwargs.get("uri")
                if not uri:
                    return {"success": False, "error": "uri required for subscribe action"}

                subscription_id = await self._subscribe_resource(server_name, uri)

                return {
                    "success": True,
                    "server_name": server_name,
                    "uri": uri,
                    "subscription_id": subscription_id,
                    "message": f"Subscribed to updates for {uri}",
                }

            elif action == "unsubscribe":
                subscription_id = kwargs.get("subscription_id")
                if not subscription_id:
                    return {
                        "success": False,
                        "error": "subscription_id required for unsubscribe action",
                    }

                if subscription_id in self.subscriptions:
                    self.subscriptions[subscription_id]["active"] = False
                    return {
                        "success": True,
                        "subscription_id": subscription_id,
                        "message": "Unsubscribed successfully",
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Unknown subscription ID: {subscription_id}",
                    }

        except Exception as e:
            self.logger.error(f"MCP resource error: {e}")
            return {"success": False, "error": str(e)}
