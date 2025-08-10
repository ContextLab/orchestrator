"""Enhanced MCP Integration - Issue #203 Phase 2

Advanced MCP server management with:
- Auto-discovery of MCP servers and tools
- Enhanced resource management
- Server health monitoring  
- Dynamic tool registration
- Resource caching and optimization
- MCP server orchestration
"""

import asyncio
import logging
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Callable
from pathlib import Path
from enum import Enum
import aiofiles
import yaml

from .universal_registry import UniversalToolRegistry, ToolCategory, ToolSource, ToolMetadata
from ..adapters.mcp_adapter import MCPClient, MCPTool, MCPResource, MCPAdapter

logger = logging.getLogger(__name__)


class MCPServerStatus(Enum):
    """Status of MCP servers."""
    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class MCPServerConfig:
    """Configuration for MCP server."""
    name: str
    url: str
    enabled: bool = True
    auto_discovery: bool = True
    health_check_interval: int = 60  # seconds
    reconnect_attempts: int = 3
    timeout: int = 30
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    auth: Optional[Dict[str, str]] = None


@dataclass
class MCPServerInfo:
    """Runtime information about MCP server."""
    config: MCPServerConfig
    status: MCPServerStatus
    last_health_check: Optional[float] = None
    connection_attempts: int = 0
    error_message: Optional[str] = None
    tools_count: int = 0
    resources_count: int = 0
    prompts_count: int = 0
    capabilities: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResourceCache:
    """Cache for MCP resources."""
    uri: str
    content: Any
    timestamp: float
    ttl: int = 3600  # 1 hour default
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return time.time() - self.timestamp > self.ttl


class EnhancedMCPManager:
    """Enhanced MCP manager with auto-discovery and resource management."""
    
    def __init__(self, registry: UniversalToolRegistry):
        self.registry = registry
        self.servers: Dict[str, MCPServerInfo] = {}
        self.clients: Dict[str, MCPClient] = {}
        self.resource_cache: Dict[str, MCPResourceCache] = {}
        self.auto_discovery_enabled = True
        self.health_check_task: Optional[asyncio.Task] = None
        self.config_file: Optional[Path] = None
        
        # Event callbacks
        self.on_server_connected: List[Callable[[str, MCPServerInfo], None]] = []
        self.on_server_disconnected: List[Callable[[str, MCPServerInfo], None]] = []
        self.on_tool_discovered: List[Callable[[str, MCPTool], None]] = []
        self.on_resource_discovered: List[Callable[[str, MCPResource], None]] = []
        
    async def initialize(self, config_file: Optional[Path] = None):
        """Initialize the enhanced MCP manager."""
        self.config_file = config_file
        
        # Load configuration
        if config_file and config_file.exists():
            await self._load_config(config_file)
        
        # Start health check task
        if not self.health_check_task:
            self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Enhanced MCP Manager initialized")
    
    async def _load_config(self, config_file: Path):
        """Load MCP server configurations from file."""
        try:
            async with aiofiles.open(config_file, 'r') as f:
                content = await f.read()
                config_data = yaml.safe_load(content)
            
            servers_config = config_data.get('mcp_servers', [])
            for server_data in servers_config:
                config = MCPServerConfig(**server_data)
                await self.add_server(config)
                
            logger.info(f"Loaded {len(servers_config)} MCP server configurations")
            
        except Exception as e:
            logger.error(f"Failed to load MCP config from {config_file}: {e}")
    
    async def add_server(self, config: MCPServerConfig) -> bool:
        """Add and configure MCP server."""
        server_info = MCPServerInfo(
            config=config,
            status=MCPServerStatus.UNKNOWN
        )
        
        self.servers[config.name] = server_info
        
        if config.enabled:
            return await self.connect_server(config.name)
        
        return True
    
    async def connect_server(self, server_name: str) -> bool:
        """Connect to MCP server with enhanced error handling."""
        if server_name not in self.servers:
            logger.error(f"MCP server {server_name} not configured")
            return False
        
        server_info = self.servers[server_name]
        server_info.status = MCPServerStatus.CONNECTING
        
        try:
            # Create and connect client
            client = MCPClient(server_info.config.url)
            success = await client.connect()
            
            if success:
                self.clients[server_name] = client
                server_info.status = MCPServerStatus.CONNECTED
                server_info.connection_attempts = 0
                server_info.error_message = None
                server_info.last_health_check = time.time()
                
                # Update capabilities and counts
                server_info.capabilities = client.capabilities
                server_info.tools_count = len(client.tools)
                server_info.resources_count = len(client.resources)
                server_info.prompts_count = len(client.prompts)
                
                # Auto-discover and register tools if enabled
                if server_info.config.auto_discovery:
                    await self._auto_discover_tools(server_name, client)
                    await self._auto_discover_resources(server_name, client)
                
                # Trigger callbacks
                for callback in self.on_server_connected:
                    try:
                        callback(server_name, server_info)
                    except Exception as e:
                        logger.warning(f"Server connected callback error: {e}")
                
                logger.info(f"Connected to MCP server: {server_name}")
                return True
            
            else:
                server_info.status = MCPServerStatus.ERROR
                server_info.connection_attempts += 1
                server_info.error_message = "Connection failed"
                logger.error(f"Failed to connect to MCP server: {server_name}")
                return False
                
        except Exception as e:
            server_info.status = MCPServerStatus.ERROR
            server_info.connection_attempts += 1
            server_info.error_message = str(e)
            logger.error(f"Error connecting to MCP server {server_name}: {e}")
            return False
    
    async def _auto_discover_tools(self, server_name: str, client: MCPClient):
        """Auto-discover and register MCP tools."""
        for mcp_tool in client.tools:
            # Create enhanced metadata
            metadata = ToolMetadata(
                name=f"mcp:{server_name}:{mcp_tool.name}",
                source=ToolSource.MCP,
                category=self._categorize_mcp_tool(mcp_tool),
                description=mcp_tool.description,
                tags=self._generate_tool_tags(mcp_tool, server_name),
                mcp_server=server_name,
                execution_context="mcp_server",
                capabilities={
                    "mcp_tool": True,
                    "server": server_name,
                    "input_schema": mcp_tool.inputSchema,
                    "auto_discovered": True
                }
            )
            
            # Register with universal registry
            self.registry._register_metadata(metadata.name, metadata)
            
            # Store MCP tool reference
            if hasattr(self.registry, 'mcp_tools'):
                self.registry.mcp_tools[metadata.name] = mcp_tool
            
            # Trigger callbacks
            for callback in self.on_tool_discovered:
                try:
                    callback(server_name, mcp_tool)
                except Exception as e:
                    logger.warning(f"Tool discovered callback error: {e}")
            
            logger.debug(f"Auto-discovered MCP tool: {metadata.name}")
    
    async def _auto_discover_resources(self, server_name: str, client: MCPClient):
        """Auto-discover MCP resources."""
        for resource in client.resources:
            # Trigger callbacks
            for callback in self.on_resource_discovered:
                try:
                    callback(server_name, resource)
                except Exception as e:
                    logger.warning(f"Resource discovered callback error: {e}")
            
            logger.debug(f"Auto-discovered MCP resource: {resource.name}")
    
    def _categorize_mcp_tool(self, tool: MCPTool) -> ToolCategory:
        """Automatically categorize MCP tools based on name and description."""
        name_lower = tool.name.lower()
        desc_lower = tool.description.lower()
        
        # Web-related tools
        if any(keyword in name_lower or keyword in desc_lower for keyword in 
               ['web', 'http', 'url', 'scrape', 'browser', 'search']):
            return ToolCategory.WEB
        
        # Data tools
        elif any(keyword in name_lower or keyword in desc_lower for keyword in 
                ['data', 'json', 'csv', 'database', 'sql', 'query']):
            return ToolCategory.DATA
        
        # System tools
        elif any(keyword in name_lower or keyword in desc_lower for keyword in 
                ['system', 'file', 'directory', 'process', 'command']):
            return ToolCategory.SYSTEM
        
        # Code execution
        elif any(keyword in name_lower or keyword in desc_lower for keyword in 
                ['execute', 'run', 'code', 'python', 'script']):
            return ToolCategory.CODE_EXECUTION
        
        # LLM tools
        elif any(keyword in name_lower or keyword in desc_lower for keyword in 
                ['llm', 'model', 'generate', 'prompt', 'ai']):
            return ToolCategory.LLM
        
        else:
            return ToolCategory.MCP_INTEGRATION
    
    def _generate_tool_tags(self, tool: MCPTool, server_name: str) -> List[str]:
        """Generate tags for MCP tools."""
        tags = ['mcp', server_name]
        
        # Add tags based on tool name and description
        name_words = tool.name.lower().replace('-', '_').split('_')
        desc_words = tool.description.lower().split()
        
        # Common tag keywords
        tag_keywords = {
            'web', 'http', 'api', 'data', 'file', 'system', 'database',
            'search', 'process', 'execute', 'generate', 'analyze'
        }
        
        for word in name_words + desc_words:
            if word in tag_keywords and word not in tags:
                tags.append(word)
        
        return tags[:10]  # Limit to 10 tags
    
    async def disconnect_server(self, server_name: str) -> bool:
        """Disconnect from MCP server."""
        if server_name not in self.servers:
            return False
        
        server_info = self.servers[server_name]
        
        # Disconnect client
        if server_name in self.clients:
            try:
                await self.clients[server_name].disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting MCP server {server_name}: {e}")
            
            del self.clients[server_name]
        
        # Update status
        server_info.status = MCPServerStatus.DISCONNECTED
        server_info.error_message = None
        
        # Trigger callbacks
        for callback in self.on_server_disconnected:
            try:
                callback(server_name, server_info)
            except Exception as e:
                logger.warning(f"Server disconnected callback error: {e}")
        
        logger.info(f"Disconnected from MCP server: {server_name}")
        return True
    
    async def _health_check_loop(self):
        """Background task for health checking MCP servers."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                for server_name, server_info in self.servers.items():
                    if server_info.config.enabled and server_info.status == MCPServerStatus.CONNECTED:
                        await self._health_check_server(server_name)
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _health_check_server(self, server_name: str):
        """Perform health check on specific server."""
        if server_name not in self.clients:
            return
        
        server_info = self.servers[server_name]
        client = self.clients[server_name]
        
        try:
            # Simple health check - try to list tools
            current_time = time.time()
            if (server_info.last_health_check and 
                current_time - server_info.last_health_check < server_info.config.health_check_interval):
                return  # Too soon for next check
            
            # Attempt a simple operation
            # Could be enhanced with actual ping/health endpoint
            if hasattr(client, '_connection') and client._connection:
                server_info.last_health_check = current_time
                logger.debug(f"Health check passed for server: {server_name}")
            else:
                # Connection lost, try to reconnect
                logger.warning(f"Health check failed for server: {server_name}, attempting reconnect")
                await self.connect_server(server_name)
                
        except Exception as e:
            logger.error(f"Health check error for server {server_name}: {e}")
            server_info.status = MCPServerStatus.ERROR
            server_info.error_message = str(e)
    
    async def get_cached_resource(self, uri: str, server_name: Optional[str] = None) -> Optional[Any]:
        """Get resource from cache or fetch from MCP server."""
        # Check cache first
        if uri in self.resource_cache:
            cached = self.resource_cache[uri]
            if not cached.is_expired():
                logger.debug(f"Returning cached resource: {uri}")
                return cached.content
            else:
                # Remove expired entry
                del self.resource_cache[uri]
        
        # Fetch from server
        if server_name and server_name in self.clients:
            try:
                client = self.clients[server_name]
                resource_data = await client.read_resource(uri)
                
                if resource_data:
                    # Cache the resource
                    self.resource_cache[uri] = MCPResourceCache(
                        uri=uri,
                        content=resource_data,
                        timestamp=time.time()
                    )
                    
                    logger.debug(f"Fetched and cached resource: {uri}")
                    return resource_data
                    
            except Exception as e:
                logger.error(f"Failed to fetch resource {uri} from {server_name}: {e}")
        
        # Try all connected servers
        for name, client in self.clients.items():
            try:
                resource_data = await client.read_resource(uri)
                if resource_data:
                    # Cache the resource
                    self.resource_cache[uri] = MCPResourceCache(
                        uri=uri,
                        content=resource_data,
                        timestamp=time.time()
                    )
                    
                    logger.debug(f"Fetched and cached resource {uri} from {name}")
                    return resource_data
                    
            except Exception as e:
                logger.debug(f"Resource {uri} not found in {name}: {e}")
                continue
        
        logger.warning(f"Resource not found: {uri}")
        return None
    
    async def execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any], 
                              server_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Execute MCP tool with enhanced error handling and retry logic."""
        
        # Extract server and tool name if prefixed
        if tool_name.startswith("mcp:") and ":" in tool_name:
            parts = tool_name.split(":", 2)
            if len(parts) >= 3:
                server_name = parts[1]
                actual_tool_name = parts[2]
            else:
                actual_tool_name = tool_name
        else:
            actual_tool_name = tool_name
        
        # Try specific server first
        if server_name and server_name in self.clients:
            try:
                client = self.clients[server_name]
                result = await client.call_tool(actual_tool_name, arguments)
                if result:
                    logger.debug(f"Executed MCP tool {actual_tool_name} on {server_name}")
                    return result
            except Exception as e:
                logger.error(f"MCP tool execution failed on {server_name}: {e}")
        
        # Try all connected servers
        for name, client in self.clients.items():
            try:
                result = await client.call_tool(actual_tool_name, arguments)
                if result:
                    logger.debug(f"Executed MCP tool {actual_tool_name} on {name}")
                    return result
            except Exception as e:
                logger.debug(f"MCP tool {actual_tool_name} not found on {name}: {e}")
                continue
        
        logger.error(f"MCP tool {actual_tool_name} not found on any server")
        return None
    
    def get_server_statistics(self) -> Dict[str, Any]:
        """Get comprehensive MCP server statistics."""
        stats = {
            "total_servers": len(self.servers),
            "connected_servers": len([s for s in self.servers.values() 
                                    if s.status == MCPServerStatus.CONNECTED]),
            "total_tools": sum(s.tools_count for s in self.servers.values()),
            "total_resources": sum(s.resources_count for s in self.servers.values()),
            "total_prompts": sum(s.prompts_count for s in self.servers.values()),
            "cached_resources": len(self.resource_cache),
            "servers": {}
        }
        
        for name, info in self.servers.items():
            stats["servers"][name] = {
                "status": info.status.value,
                "url": info.config.url,
                "enabled": info.config.enabled,
                "tools_count": info.tools_count,
                "resources_count": info.resources_count,
                "prompts_count": info.prompts_count,
                "last_health_check": info.last_health_check,
                "connection_attempts": info.connection_attempts,
                "error_message": info.error_message
            }
        
        return stats
    
    async def discover_servers_by_pattern(self, base_urls: List[str], 
                                        ports: List[int] = None) -> List[MCPServerConfig]:
        """Auto-discover MCP servers by scanning URLs and ports."""
        discovered = []
        ports = ports or [8000, 8001, 8080, 9000]
        
        for base_url in base_urls:
            for port in ports:
                test_url = f"{base_url}:{port}"
                
                try:
                    # Try to connect briefly
                    test_client = MCPClient(test_url)
                    success = await test_client.connect()
                    
                    if success:
                        # Create configuration
                        server_name = f"auto_discovered_{len(discovered)}"
                        config = MCPServerConfig(
                            name=server_name,
                            url=test_url,
                            auto_discovery=True,
                            tags=["auto_discovered"]
                        )
                        
                        discovered.append(config)
                        await test_client.disconnect()
                        
                        logger.info(f"Auto-discovered MCP server: {test_url}")
                
                except Exception:
                    # Server not found, continue
                    pass
        
        return discovered
    
    async def cleanup(self):
        """Clean up MCP manager resources."""
        # Cancel health check task
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all servers
        for server_name in list(self.servers.keys()):
            await self.disconnect_server(server_name)
        
        # Clear caches
        self.resource_cache.clear()
        
        logger.info("Enhanced MCP Manager cleaned up")


# Integration with Universal Registry
async def integrate_mcp_with_registry(registry: UniversalToolRegistry, 
                                    config_file: Optional[Path] = None) -> EnhancedMCPManager:
    """Integrate enhanced MCP manager with universal registry."""
    
    mcp_manager = EnhancedMCPManager(registry)
    await mcp_manager.initialize(config_file)
    
    # Add registry integration to MCP manager
    registry.mcp_manager = mcp_manager
    
    logger.info("MCP integration with Universal Registry complete")
    return mcp_manager


__all__ = [
    "EnhancedMCPManager",
    "MCPServerConfig", 
    "MCPServerInfo",
    "MCPServerStatus",
    "MCPResourceCache",
    "integrate_mcp_with_registry"
]