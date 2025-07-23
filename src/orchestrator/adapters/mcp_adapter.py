"""MCP (Model Context Protocol) adapter for integrating with MCP servers."""

import json
import logging
import asyncio
import aiohttp
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

from ..core.control_system import ControlAction, ControlSystem
from ..core.model import Model
from ..core.pipeline import Pipeline
from ..core.task import Task
from ..models.model_registry import ModelRegistry
from ..models.registry_singleton import get_model_registry
from ..control_systems.model_based_control_system import ModelBasedControlSystem


@dataclass
class MCPResource:
    """Represents an MCP resource."""

    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPTool:
    """Represents an MCP tool."""

    name: str
    description: str
    inputSchema: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPPrompt:
    """Represents an MCP prompt template."""

    name: str
    description: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPMessage:
    """MCP protocol message."""

    def __init__(self, method: str, params: Dict[str, Any] = None, id: str = None):
        self.method = method
        self.params = params or {}
        self.id = id or self._generate_id()
        self.jsonrpc = "2.0"

    def _generate_id(self) -> str:
        """Generate unique message ID."""
        import uuid

        return str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        msg = {"jsonrpc": self.jsonrpc, "method": self.method, "params": self.params}
        if self.id:
            msg["id"] = self.id
        return msg

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())


class MCPClient:
    """Client for communicating with MCP servers."""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session_id = None
        self.capabilities = {}
        self.resources: List[MCPResource] = []
        self.tools: List[MCPTool] = []
        self.prompts: List[MCPPrompt] = []
        self.logger = logging.getLogger(f"mcp_client.{server_url}")
        self._connection = None

    async def connect(self) -> bool:
        """Connect to MCP server."""
        try:
            # Initialize connection (simplified - would use actual MCP transport)
            self._connection = {"connected": True, "url": self.server_url}

            # Send initialize request
            init_msg = MCPMessage(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "resources": {"subscribe": True},
                        "tools": {},
                        "prompts": {},
                    },
                    "clientInfo": {
                        "name": "orchestrator-mcp-adapter",
                        "version": "1.0.0",
                    },
                },
            )

            # Would send actual message to server
            response = await self._send_message(init_msg)

            if response and response.get("result"):
                result = response["result"]
                self.capabilities = result.get("capabilities", {})
                self.session_id = result.get("sessionId")

                # Discover available resources, tools, and prompts
                await self._discover_capabilities()

                self.logger.info(f"Connected to MCP server at {self.server_url}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server: {e}")
            return False

        return False

    async def disconnect(self):
        """Disconnect from MCP server."""
        if self._connection:
            await self._cleanup_transport()
            self._connection = None
            self.session_id = None
            self.logger.info(f"Disconnected from MCP server at {self.server_url}")

    async def _send_message(self, message: MCPMessage) -> Optional[Dict[str, Any]]:
        """Send message to MCP server using real transport."""
        
        # Check if using HTTP transport
        if self.server_url.startswith(("http://", "https://")):
            return await self._send_http_message(message)
        else:
            # For stdio-based servers, use subprocess
            return await self._send_stdio_message(message)
    
    async def _send_http_message(self, message: MCPMessage) -> Optional[Dict[str, Any]]:
        """Send message via HTTP transport."""
        if not hasattr(self, '_http_session'):
            self._http_session = aiohttp.ClientSession()
        
        # Prepare JSON-RPC message
        json_rpc_message = {
            "jsonrpc": "2.0",
            "method": message.method,
            "params": message.params,
            "id": message.id or str(uuid.uuid4())
        }
        
        try:
            async with self._http_session.post(
                self.server_url,
                json=json_rpc_message,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    self.logger.error(f"HTTP error {response.status}: {await response.text()}")
                    return None
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP transport error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in HTTP transport: {e}")
            return None
    
    async def _send_stdio_message(self, message: MCPMessage) -> Optional[Dict[str, Any]]:
        """Send message via stdio transport to a subprocess."""
        if not hasattr(self, '_process') or self._process is None:
            # Start the MCP server process
            try:
                self._process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", self.server_url,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except Exception as e:
                self.logger.error(f"Failed to start MCP server process: {e}")
                return None
        
        # Prepare JSON-RPC message
        json_rpc_message = {
            "jsonrpc": "2.0",
            "method": message.method,
            "params": message.params,
            "id": message.id or str(uuid.uuid4())
        }
        
        try:
            # Send message
            message_bytes = (json.dumps(json_rpc_message) + "\n").encode()
            self._process.stdin.write(message_bytes)
            await self._process.stdin.drain()
            
            # Read response
            response_line = await self._process.stdout.readline()
            if response_line:
                return json.loads(response_line.decode())
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Stdio transport error: {e}")
            return None
    
    async def _cleanup_transport(self):
        """Clean up transport resources."""
        if hasattr(self, '_http_session') and self._http_session:
            await self._http_session.close()
        
        if hasattr(self, '_process') and self._process:
            self._process.terminate()
            await self._process.wait()

    async def _discover_capabilities(self):
        """Discover server capabilities."""
        # List resources
        if "resources" in self.capabilities:
            resources_msg = MCPMessage("resources/list")
            response = await self._send_message(resources_msg)
            if response and "result" in response:
                for res_data in response["result"].get("resources", []):
                    resource = MCPResource(**res_data)
                    self.resources.append(resource)

        # List tools
        if "tools" in self.capabilities:
            tools_msg = MCPMessage("tools/list")
            response = await self._send_message(tools_msg)
            if response and "result" in response:
                for tool_data in response["result"].get("tools", []):
                    tool = MCPTool(**tool_data)
                    self.tools.append(tool)

        # List prompts
        if "prompts" in self.capabilities:
            prompts_msg = MCPMessage("prompts/list")
            response = await self._send_message(prompts_msg)
            if response and "result" in response:
                for prompt_data in response["result"].get("prompts", []):
                    prompt = MCPPrompt(**prompt_data)
                    self.prompts.append(prompt)

    async def read_resource(self, uri: str) -> Optional[Dict[str, Any]]:
        """Read a resource from the MCP server."""
        msg = MCPMessage("resources/read", {"uri": uri})
        response = await self._send_message(msg)

        if response and "result" in response:
            return response["result"]
        return None

    async def call_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Call a tool on the MCP server."""
        msg = MCPMessage("tools/call", {"name": name, "arguments": arguments})
        response = await self._send_message(msg)

        if response and "result" in response:
            return response["result"]
        return None

    async def get_prompt(
        self, name: str, arguments: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Get a prompt from the MCP server."""
        params = {"name": name}
        if arguments:
            params["arguments"] = arguments

        msg = MCPMessage("prompts/get", params)
        response = await self._send_message(msg)

        if response and "result" in response:
            return response["result"]
        return None


class MCPModel(Model):
    """Model that uses MCP for enhanced capabilities."""

    def __init__(self, name: str, mcp_client: MCPClient, base_model: Model):
        super().__init__(name)
        self.mcp_client = mcp_client
        self.base_model = base_model
        self.capabilities.update(base_model.capabilities)

        # Add MCP-specific capabilities
        self.capabilities.update(
            {
                "mcp_resources": len(mcp_client.resources),
                "mcp_tools": len(mcp_client.tools),
                "mcp_prompts": len(mcp_client.prompts),
            }
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using base model with MCP context."""
        # Check if we can enhance the prompt with MCP resources
        enhanced_prompt = await self._enhance_prompt_with_mcp(prompt, **kwargs)

        # Use base model for generation
        return await self.base_model.generate(enhanced_prompt, **kwargs)

    async def _enhance_prompt_with_mcp(self, prompt: str, **kwargs) -> str:
        """Enhance prompt with MCP resources and tools."""
        enhanced_prompt = prompt

        # Try to find relevant resources
        for resource in self.mcp_client.resources:
            if any(
                keyword in prompt.lower()
                for keyword in [
                    resource.name.lower(),
                    resource.description.lower() if resource.description else "",
                ]
            ):
                # Read the resource
                resource_data = await self.mcp_client.read_resource(resource.uri)
                if resource_data:
                    enhanced_prompt += f"\n\nRelevant context from {resource.name}:\n{resource_data.get('contents', '')}"

        # Check if any tools might be useful
        tool_suggestions = []
        for tool in self.mcp_client.tools:
            if any(
                keyword in prompt.lower()
                for keyword in ["search", "analyze", "calculate"]
            ):
                tool_suggestions.append(f"- {tool.name}: {tool.description}")

        if tool_suggestions:
            enhanced_prompt += "\n\nAvailable tools:\n" + "\n".join(tool_suggestions)

        return enhanced_prompt

    def can_execute(self, task: Task) -> bool:
        """Check if this model can execute the task."""
        # Check base model capability first
        if not self.base_model.can_execute(task):
            return False

        # Check if MCP resources might be helpful
        task_type = task.parameters.get("type", "")
        if task_type in ["research", "analysis", "data_processing"]:
            return len(self.mcp_client.resources) > 0 or len(self.mcp_client.tools) > 0

        return True


class MCPAdapter(ControlSystem):
    """Adapter for integrating Orchestrator with MCP servers."""

    def __init__(self, config: Dict[str, Any] = None, model_registry: ModelRegistry = None):
        if config is None:
            config = {"name": "mcp_adapter"}

        super().__init__(config.get("name", "mcp_adapter"))
        self.config = config
        self.clients: Dict[str, MCPClient] = {}
        self.models: Dict[str, MCPModel] = {}
        self.logger = logging.getLogger("mcp_adapter")
        
        # Initialize model registry and control system for AI-enhanced execution
        self.model_registry = model_registry or get_model_registry()
        self.ai_control = ModelBasedControlSystem(self.model_registry)

    async def add_server(self, server_name: str, server_url: str) -> bool:
        """Add and connect to an MCP server."""
        client = MCPClient(server_url)
        success = await client.connect()

        if success:
            self.clients[server_name] = client
            self.logger.info(f"Added MCP server: {server_name} at {server_url}")
            return True

        return False

    async def execute_task(self, task: Task, context: Dict[str, Any] = None) -> Any:
        """Execute a single task using MCP tools and AI models."""
        context = context or {}
        
        # Determine the best approach based on task action
        action = task.action.lower() if isinstance(task.action, str) else str(task.action).lower()
        
        # Check if this is an MCP tool invocation
        if action.startswith("mcp:") or action.startswith("tool:"):
            # Extract tool name and execute via MCP
            tool_name = action.split(":", 1)[1] if ":" in action else action
            return await self._execute_mcp_tool(task, tool_name, context)
        
        # Check if this is a resource read operation
        elif action in ["read", "fetch", "get"] and "resource" in task.parameters:
            return await self._execute_resource_read(task, context)
        
        # Check if this is a prompt-based operation
        elif action in ["prompt", "generate"] and any(client.prompts for client in self.clients.values()):
            return await self._execute_mcp_prompt(task, context)
        
        # Otherwise, use AI model with MCP context enhancement
        else:
            return await self._execute_with_ai_and_mcp(task, context)
    
    async def _execute_mcp_tool(self, task: Task, tool_name: str, context: Dict[str, Any]) -> Any:
        """Execute a task using an MCP tool."""
        # Find a client that has this tool
        for server_name, client in self.clients.items():
            tool = next((t for t in client.tools if t.name == tool_name), None)
            if tool:
                # Prepare arguments from task parameters
                arguments = task.parameters.copy()
                
                # Call the tool
                result = await client.call_tool(tool_name, arguments)
                if result:
                    return result
                else:
                    raise RuntimeError(f"MCP tool '{tool_name}' execution failed on server '{server_name}'")
        
        raise ValueError(f"MCP tool '{tool_name}' not found in any connected server")
    
    async def _execute_resource_read(self, task: Task, context: Dict[str, Any]) -> Any:
        """Execute a resource read operation."""
        resource_uri = task.parameters.get("resource") or task.parameters.get("uri")
        if not resource_uri:
            raise ValueError("Resource URI not specified in task parameters")
        
        # Try to read from any connected server
        for server_name, client in self.clients.items():
            result = await client.read_resource(resource_uri)
            if result:
                return result
        
        raise RuntimeError(f"Failed to read resource '{resource_uri}' from any MCP server")
    
    async def _execute_mcp_prompt(self, task: Task, context: Dict[str, Any]) -> Any:
        """Execute a task using an MCP prompt template."""
        prompt_name = task.parameters.get("prompt_name") or task.parameters.get("template")
        
        if prompt_name:
            # Find a client with this prompt
            for server_name, client in self.clients.items():
                prompt = next((p for p in client.prompts if p.name == prompt_name), None)
                if prompt:
                    # Get the prompt with arguments
                    prompt_result = await client.get_prompt(prompt_name, task.parameters)
                    if prompt_result:
                        # Use AI to process the prompt
                        enhanced_task = Task(
                            id=task.id,
                            name=task.name,
                            action="generate",
                            parameters={"prompt": prompt_result.get("prompt", str(prompt_result))}
                        )
                        return await self.ai_control.execute_task(enhanced_task, context)
        
        # Fallback to AI execution
        return await self._execute_with_ai_and_mcp(task, context)
    
    async def _execute_with_ai_and_mcp(self, task: Task, context: Dict[str, Any]) -> Any:
        """Execute task using AI model enhanced with MCP context."""
        # Gather relevant MCP context
        mcp_context = await self._gather_mcp_context(task, context)
        
        # Enhance the context with MCP information
        enhanced_context = context.copy()
        enhanced_context["mcp_resources"] = mcp_context.get("resources", [])
        enhanced_context["mcp_tools"] = mcp_context.get("tools", [])
        enhanced_context["mcp_capabilities"] = mcp_context.get("capabilities", {})
        
        # Execute using AI control system
        return await self.ai_control.execute_task(task, enhanced_context)
    
    async def _gather_mcp_context(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather relevant MCP context for task execution."""
        mcp_context = {
            "resources": [],
            "tools": [],
            "capabilities": {}
        }
        
        # Collect available resources and tools from all clients
        for server_name, client in self.clients.items():
            mcp_context["resources"].extend([
                {"server": server_name, "resource": r} for r in client.resources
            ])
            mcp_context["tools"].extend([
                {"server": server_name, "tool": t} for t in client.tools
            ])
            mcp_context["capabilities"][server_name] = client.capabilities
        
        return mcp_context

    async def execute_pipeline(self, pipeline: Pipeline) -> Dict[str, Any]:
        """Execute an entire pipeline."""
        results = {}
        for task_id in pipeline:
            task = pipeline.get_task(task_id)
            results[task_id] = await self.execute_task(task)
        return results

    def get_capabilities(self) -> Dict[str, Any]:
        """Return system capabilities."""
        # Get base AI capabilities
        ai_capabilities = self.ai_control.get_capabilities()
        
        # Add MCP-specific capabilities
        mcp_capabilities = {
            "supports_tools": True,
            "supports_resources": True,
            "supports_prompts": True,
            "supports_mcp_protocol": True,
            "mcp_servers": list(self.clients.keys()),
            "mcp_tools": [tool.name for client in self.clients.values() for tool in client.tools],
            "mcp_resources": [res.uri for client in self.clients.values() for res in client.resources],
        }
        
        # Merge capabilities
        combined = ai_capabilities.copy()
        combined.update(mcp_capabilities)
        
        # Add MCP actions to supported actions
        mcp_actions = ["mcp:*", "tool:*", "read", "fetch", "prompt"]
        if "supported_actions" in combined:
            combined["supported_actions"].extend(mcp_actions)
        else:
            combined["supported_actions"] = mcp_actions
        
        return combined

    async def health_check(self) -> bool:
        """Check if the system is healthy."""
        # Check if we have at least one connected MCP server
        any_connected = any(client._connection for client in self.clients.values())
        
        # Check if AI control system is healthy
        ai_healthy = await self.ai_control.health_check()
        
        return any_connected or ai_healthy  # Can work with either MCP or AI

    async def remove_server(self, server_name: str):
        """Remove and disconnect from an MCP server."""
        if server_name in self.clients:
            await self.clients[server_name].disconnect()
            del self.clients[server_name]

            # Remove associated models
            models_to_remove = [
                model_name
                for model_name, model in self.models.items()
                if model.mcp_client == self.clients.get(server_name)
            ]
            for model_name in models_to_remove:
                del self.models[model_name]

            self.logger.info(f"Removed MCP server: {server_name}")

    def create_mcp_model(
        self, name: str, server_name: str, base_model: Model
    ) -> Optional[MCPModel]:
        """Create an MCP-enhanced model."""
        if server_name not in self.clients:
            self.logger.error(f"MCP server '{server_name}' not found")
            return None

        client = self.clients[server_name]
        mcp_model = MCPModel(name, client, base_model)
        self.models[name] = mcp_model

        return mcp_model

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Call a tool on a specific MCP server."""
        if server_name not in self.clients:
            self.logger.error(f"MCP server '{server_name}' not found")
            return None

        client = self.clients[server_name]
        return await client.call_tool(tool_name, arguments)

    async def read_resource(
        self, server_name: str, resource_uri: str
    ) -> Optional[Dict[str, Any]]:
        """Read a resource from a specific MCP server."""
        if server_name not in self.clients:
            self.logger.error(f"MCP server '{server_name}' not found")
            return None

        client = self.clients[server_name]
        return await client.read_resource(resource_uri)

    async def decide_action(self, task: Task, context: Dict[str, Any]) -> ControlAction:
        """Decide control action based on MCP context."""
        # Check if task might benefit from MCP resources
        task_type = task.parameters.get("type", "")

        if task_type in ["research", "data_analysis", "web_search"]:
            # Check if we have relevant MCP servers
            for server_name, client in self.clients.items():
                # Look for relevant tools
                relevant_tools = [
                    tool
                    for tool in client.tools
                    if any(keyword in task_type for keyword in tool.name.split("_"))
                ]

                if relevant_tools:
                    # Suggest using MCP-enhanced execution
                    context["suggested_mcp_server"] = server_name
                    context["available_tools"] = [tool.name for tool in relevant_tools]
                    return ControlAction.EXECUTE

        return ControlAction.EXECUTE

    def get_server_status(self, server_name: str) -> Dict[str, Any]:
        """Get status of a specific MCP server."""
        if server_name not in self.clients:
            return {"error": "Server not found"}

        client = self.clients[server_name]
        return {
            "name": server_name,
            "url": client.server_url,
            "connected": client._connection is not None,
            "session_id": client.session_id,
            "capabilities": client.capabilities,
            "resources": len(client.resources),
            "tools": len(client.tools),
            "prompts": len(client.prompts),
        }

    def get_all_resources(self) -> Dict[str, List[MCPResource]]:
        """Get all resources from all connected servers."""
        all_resources = {}
        for server_name, client in self.clients.items():
            all_resources[server_name] = client.resources
        return all_resources

    def get_all_tools(self) -> Dict[str, List[MCPTool]]:
        """Get all tools from all connected servers."""
        all_tools = {}
        for server_name, client in self.clients.items():
            all_tools[server_name] = client.tools
        return all_tools

    def get_statistics(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        total_resources = sum(len(client.resources) for client in self.clients.values())
        total_tools = sum(len(client.tools) for client in self.clients.values())
        total_prompts = sum(len(client.prompts) for client in self.clients.values())

        return {
            "connected_servers": len(self.clients),
            "mcp_models": len(self.models),
            "total_resources": total_resources,
            "total_tools": total_tools,
            "total_prompts": total_prompts,
            "servers": {
                name: {
                    "resources": len(client.resources),
                    "tools": len(client.tools),
                    "prompts": len(client.prompts),
                    "connected": client._connection is not None,
                }
                for name, client in self.clients.items()
            },
        }

    async def cleanup(self):
        """Clean up all connections."""
        # Clean up HTTP sessions in all clients
        for client in self.clients.values():
            if hasattr(client, '_http_session'):
                await client._http_session.close()
        
        # Remove all servers
        for server_name in list(self.clients.keys()):
            await self.remove_server(server_name)
