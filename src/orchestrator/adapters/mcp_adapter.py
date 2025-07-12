"""MCP (Model Context Protocol) adapter for integrating with MCP servers."""

import asyncio
import json
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

from ..core.control_system import ControlSystem, ControlAction
from ..core.task import Task, TaskStatus
from ..core.model import Model
from ..core.pipeline import Pipeline


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
        msg = {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params
        }
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
            init_msg = MCPMessage("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "resources": {"subscribe": True},
                    "tools": {},
                    "prompts": {}
                },
                "clientInfo": {
                    "name": "orchestrator-mcp-adapter",
                    "version": "1.0.0"
                }
            })
            
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
            self._connection = None
            self.session_id = None
            self.logger.info(f"Disconnected from MCP server at {self.server_url}")
    
    async def _send_message(self, message: MCPMessage) -> Optional[Dict[str, Any]]:
        """Send message to MCP server."""
        # Simplified implementation - would use actual transport
        # For now, return mock responses based on method
        
        if message.method == "initialize":
            return {
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "resources": {"subscribe": True},
                        "tools": {},
                        "prompts": {}
                    },
                    "serverInfo": {
                        "name": "mock-mcp-server",
                        "version": "1.0.0"
                    },
                    "sessionId": "session_123"
                }
            }
        
        elif message.method == "resources/list":
            return {
                "result": {
                    "resources": [
                        {
                            "uri": "file:///data/documents.json",
                            "name": "Documents",
                            "description": "Company documents",
                            "mimeType": "application/json"
                        }
                    ]
                }
            }
        
        elif message.method == "tools/list":
            return {
                "result": {
                    "tools": [
                        {
                            "name": "web_search",
                            "description": "Search the web for information",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Search query"}
                                },
                                "required": ["query"]
                            }
                        }
                    ]
                }
            }
        
        elif message.method == "prompts/list":
            return {
                "result": {
                    "prompts": [
                        {
                            "name": "analyze_data",
                            "description": "Analyze data and provide insights",
                            "arguments": [
                                {"name": "data", "description": "Data to analyze", "required": True}
                            ]
                        }
                    ]
                }
            }
        
        return None
    
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
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a tool on the MCP server."""
        msg = MCPMessage("tools/call", {
            "name": name,
            "arguments": arguments
        })
        response = await self._send_message(msg)
        
        if response and "result" in response:
            return response["result"]
        return None
    
    async def get_prompt(self, name: str, arguments: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
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
        self.capabilities.update({
            "mcp_resources": len(mcp_client.resources),
            "mcp_tools": len(mcp_client.tools),
            "mcp_prompts": len(mcp_client.prompts)
        })
    
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
            if any(keyword in prompt.lower() for keyword in [resource.name.lower(), resource.description.lower() if resource.description else ""]):
                # Read the resource
                resource_data = await self.mcp_client.read_resource(resource.uri)
                if resource_data:
                    enhanced_prompt += f"\n\nRelevant context from {resource.name}:\n{resource_data.get('contents', '')}"
        
        # Check if any tools might be useful
        tool_suggestions = []
        for tool in self.mcp_client.tools:
            if any(keyword in prompt.lower() for keyword in ["search", "analyze", "calculate"]):
                tool_suggestions.append(f"- {tool.name}: {tool.description}")
        
        if tool_suggestions:
            enhanced_prompt += f"\n\nAvailable tools:\n" + "\n".join(tool_suggestions)
        
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
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {"name": "mcp_adapter"}
        
        super().__init__(config.get("name", "mcp_adapter"))
        self.config = config
        self.clients: Dict[str, MCPClient] = {}
        self.models: Dict[str, MCPModel] = {}
        self.logger = logging.getLogger("mcp_adapter")
    
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
        """Execute a single task."""
        # Mock execution for testing
        return f"Executed task {task.id} via MCP"
    
    async def execute_pipeline(self, pipeline: Pipeline) -> Dict[str, Any]:
        """Execute an entire pipeline."""
        results = {}
        for task_id in pipeline:
            task = pipeline.get_task(task_id)
            results[task_id] = await self.execute_task(task)
        return results
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return system capabilities."""
        return {
            "supports_tools": True,
            "supports_resources": True,
            "supports_prompts": True,
            "supported_actions": ["analyze", "query", "search"]
        }
    
    async def health_check(self) -> bool:
        """Check if the system is healthy."""
        return True  # Simplified for testing
    
    async def remove_server(self, server_name: str):
        """Remove and disconnect from an MCP server."""
        if server_name in self.clients:
            await self.clients[server_name].disconnect()
            del self.clients[server_name]
            
            # Remove associated models
            models_to_remove = [
                model_name for model_name, model in self.models.items()
                if model.mcp_client == self.clients.get(server_name)
            ]
            for model_name in models_to_remove:
                del self.models[model_name]
            
            self.logger.info(f"Removed MCP server: {server_name}")
    
    def create_mcp_model(self, name: str, server_name: str, base_model: Model) -> Optional[MCPModel]:
        """Create an MCP-enhanced model."""
        if server_name not in self.clients:
            self.logger.error(f"MCP server '{server_name}' not found")
            return None
        
        client = self.clients[server_name]
        mcp_model = MCPModel(name, client, base_model)
        self.models[name] = mcp_model
        
        return mcp_model
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a tool on a specific MCP server."""
        if server_name not in self.clients:
            self.logger.error(f"MCP server '{server_name}' not found")
            return None
        
        client = self.clients[server_name]
        return await client.call_tool(tool_name, arguments)
    
    async def read_resource(self, server_name: str, resource_uri: str) -> Optional[Dict[str, Any]]:
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
                    tool for tool in client.tools
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
            "prompts": len(client.prompts)
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
                    "connected": client._connection is not None
                }
                for name, client in self.clients.items()
            }
        }
    
    async def cleanup(self):
        """Clean up all connections."""
        for server_name in list(self.clients.keys()):
            await self.remove_server(server_name)