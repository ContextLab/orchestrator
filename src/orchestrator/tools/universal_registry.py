"""Universal Tool Registry - Issue #203 Phase 1

Enhanced tool registry that supports:
- Native orchestrator tools
- LangChain tools via adapters
- MCP server tools
- Sandbox execution management
- Tool discovery and categorization
- Cross-ecosystem compatibility
"""

import logging
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from .base import Tool as OrchestratorTool, ToolRegistry, ToolParameter
from .langchain_adapter import (
    LangChainToolAdapter,
    OrchestratorToolAdapter,
    ToolAdapterFactory,
    make_langchain_tool,
    make_orchestrator_tool,
    LANGCHAIN_AVAILABLE
)

# Import MCP components
try:
    from ..adapters.mcp_adapter import MCPClient, MCPTool, MCPAdapter
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MCPClient = None
    MCPTool = None
    MCPAdapter = None

# Import sandbox manager
try:
    from ..security.langchain_sandbox import LangChainSandbox, SandboxConfig, SecurityPolicy
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    LangChainSandbox = None
    SandboxConfig = None
    SecurityPolicy = None

# Import LangChain tools if available
if LANGCHAIN_AVAILABLE:
    from langchain.tools import BaseTool
else:
    BaseTool = None

logger = logging.getLogger(__name__)


class ToolSource(Enum):
    """Source/origin of tools in the registry."""
    ORCHESTRATOR = "orchestrator"
    LANGCHAIN = "langchain"
    MCP = "mcp"
    HYBRID = "hybrid"


class ToolCategory(Enum):
    """Categories for tool organization."""
    WEB = "web"
    DATA = "data"
    SYSTEM = "system"
    LLM = "llm"
    MULTIMODAL = "multimodal"
    USER_INTERACTION = "user_interaction"
    PIPELINE = "pipeline"
    SECURITY = "security"
    VALIDATION = "validation"
    REPORT = "report"
    CODE_EXECUTION = "code_execution"
    MCP_INTEGRATION = "mcp_integration"
    CUSTOM = "custom"


@dataclass
class ToolMetadata:
    """Enhanced metadata for registered tools."""
    name: str
    source: ToolSource
    category: ToolCategory
    description: str
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    security_level: str = "moderate"  # strict, moderate, permissive
    execution_context: str = "default"  # default, sandboxed, mcp_server
    mcp_server: Optional[str] = None
    langchain_compatible: bool = False
    async_compatible: bool = True


@dataclass
class ToolExecutionResult:
    """Standardized result for tool execution across all sources."""
    success: bool
    output: Any
    source: ToolSource
    execution_time: float
    tool_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class UniversalToolRegistry(ToolRegistry):
    """Enhanced tool registry supporting orchestrator, LangChain, and MCP tools.
    
    Features:
    - Multi-source tool registration (orchestrator, LangChain, MCP)
    - Automatic bidirectional adapters
    - Tool categorization and discovery
    - Sandbox execution management
    - Cross-ecosystem compatibility
    - Advanced tool metadata and capabilities
    """
    
    def __init__(self):
        super().__init__()
        
        # Enhanced storage for different tool sources
        self.tool_metadata: Dict[str, ToolMetadata] = {}
        self.langchain_tools: Dict[str, BaseTool] = {} if LANGCHAIN_AVAILABLE else {}
        self.mcp_servers: Dict[str, MCPClient] = {} if MCP_AVAILABLE else {}
        self.mcp_tools: Dict[str, MCPTool] = {} if MCP_AVAILABLE else {}
        
        # Tool organization
        self.categories: Dict[ToolCategory, Set[str]] = {cat: set() for cat in ToolCategory}
        self.tags_index: Dict[str, Set[str]] = {}
        
        # Execution management
        self.sandbox_manager: Optional[LangChainSandbox] = None
        self.mcp_adapter: Optional[MCPAdapter] = None
        
        # Initialize components if available
        self._initialize_components()
        
        logger.info("Universal Tool Registry initialized")
    
    def _initialize_components(self):
        """Initialize optional components if available."""
        # Initialize sandbox manager
        if SANDBOX_AVAILABLE:
            try:
                self.sandbox_manager = LangChainSandbox()
                logger.info("LangChain Sandbox manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize sandbox manager: {e}")
        
        # Initialize MCP adapter
        if MCP_AVAILABLE:
            try:
                self.mcp_adapter = MCPAdapter()
                logger.info("MCP adapter initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MCP adapter: {e}")
    
    # Enhanced registration methods
    
    def register_orchestrator_tool(
        self,
        tool: OrchestratorTool,
        category: ToolCategory = ToolCategory.CUSTOM,
        tags: List[str] = None,
        security_level: str = "moderate"
    ) -> None:
        """Register a native orchestrator tool with enhanced metadata."""
        # Register with base registry
        super().register(tool)
        
        # Create enhanced metadata
        metadata = ToolMetadata(
            name=tool.name,
            source=ToolSource.ORCHESTRATOR,
            category=category,
            description=tool.description,
            tags=tags or [],
            security_level=security_level,
            langchain_compatible=True,  # All orchestrator tools can be adapted
            capabilities={
                "parameters": len(tool.parameters),
                "supports_templates": True,
                "mcp_schema_compatible": True
            }
        )
        
        self._register_metadata(tool.name, metadata)
        logger.debug(f"Registered orchestrator tool: {tool.name}")
    
    def register_langchain_tool(
        self,
        langchain_tool: BaseTool,
        category: ToolCategory = ToolCategory.CUSTOM,
        tags: List[str] = None,
        register_as_orchestrator: bool = True
    ) -> Optional[OrchestratorToolAdapter]:
        """Register a LangChain tool and optionally create orchestrator adapter."""
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not available - cannot register LangChain tool")
            return None
        
        tool_name = langchain_tool.name
        
        # Store original LangChain tool
        self.langchain_tools[tool_name] = langchain_tool
        
        orchestrator_adapter = None
        if register_as_orchestrator:
            # Create orchestrator adapter and register it
            orchestrator_adapter = ToolAdapterFactory.create_orchestrator_adapter(langchain_tool)
            super().register(orchestrator_adapter)
        
        # Create metadata
        metadata = ToolMetadata(
            name=tool_name,
            source=ToolSource.LANGCHAIN if not register_as_orchestrator else ToolSource.HYBRID,
            category=category,
            description=langchain_tool.description,
            tags=tags or [],
            langchain_compatible=True,
            capabilities={
                "original_type": type(langchain_tool).__name__,
                "has_async": hasattr(langchain_tool, '_arun'),
                "has_schema": hasattr(langchain_tool, 'args_schema') and langchain_tool.args_schema is not None
            }
        )
        
        self._register_metadata(tool_name, metadata)
        logger.debug(f"Registered LangChain tool: {tool_name}")
        
        return orchestrator_adapter
    
    async def register_mcp_server(
        self,
        server_name: str,
        server_url: str,
        auto_register_tools: bool = True
    ) -> bool:
        """Register an MCP server and optionally auto-register its tools."""
        if not MCP_AVAILABLE:
            logger.error("MCP not available - cannot register MCP server")
            return False
        
        if not self.mcp_adapter:
            logger.error("MCP adapter not initialized")
            return False
        
        # Connect to MCP server
        success = await self.mcp_adapter.add_server(server_name, server_url)
        if not success:
            logger.error(f"Failed to connect to MCP server: {server_name}")
            return False
        
        # Store server reference
        if server_name in self.mcp_adapter.clients:
            self.mcp_servers[server_name] = self.mcp_adapter.clients[server_name]
        
        if auto_register_tools:
            await self._register_mcp_tools_from_server(server_name)
        
        logger.info(f"Registered MCP server: {server_name}")
        return True
    
    async def _register_mcp_tools_from_server(self, server_name: str) -> None:
        """Register all tools from an MCP server."""
        if server_name not in self.mcp_servers:
            return
        
        client = self.mcp_servers[server_name]
        
        for mcp_tool in client.tools:
            tool_name = f"mcp:{server_name}:{mcp_tool.name}"
            
            # Store MCP tool reference
            self.mcp_tools[tool_name] = mcp_tool
            
            # Create metadata
            metadata = ToolMetadata(
                name=tool_name,
                source=ToolSource.MCP,
                category=ToolCategory.MCP_INTEGRATION,
                description=mcp_tool.description,
                mcp_server=server_name,
                execution_context="mcp_server",
                capabilities={
                    "mcp_tool": True,
                    "server": server_name,
                    "input_schema": mcp_tool.inputSchema
                }
            )
            
            self._register_metadata(tool_name, metadata)
            logger.debug(f"Registered MCP tool: {tool_name}")
    
    def _register_metadata(self, tool_name: str, metadata: ToolMetadata) -> None:
        """Register tool metadata and update indices."""
        self.tool_metadata[tool_name] = metadata
        
        # Update category index
        self.categories[metadata.category].add(tool_name)
        
        # Update tags index
        for tag in metadata.tags:
            if tag not in self.tags_index:
                self.tags_index[tag] = set()
            self.tags_index[tag].add(tool_name)
    
    # Enhanced discovery and query methods
    
    def discover_tools(
        self,
        category: Optional[ToolCategory] = None,
        tags: Optional[List[str]] = None,
        source: Optional[ToolSource] = None,
        capabilities: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Discover tools based on filters."""
        candidates = set(self.tool_metadata.keys())
        
        # Filter by category
        if category:
            candidates &= self.categories.get(category, set())
        
        # Filter by tags
        if tags:
            tag_matches = set()
            for tag in tags:
                tag_matches.update(self.tags_index.get(tag, set()))
            candidates &= tag_matches
        
        # Filter by source
        if source:
            source_matches = {
                name for name, meta in self.tool_metadata.items()
                if meta.source == source
            }
            candidates &= source_matches
        
        # Filter by capabilities
        if capabilities:
            capability_matches = set()
            for tool_name in candidates:
                metadata = self.tool_metadata[tool_name]
                if all(
                    metadata.capabilities.get(key) == value
                    for key, value in capabilities.items()
                ):
                    capability_matches.add(tool_name)
            candidates &= capability_matches
        
        return sorted(list(candidates))
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a tool."""
        if tool_name not in self.tool_metadata:
            return None
        
        metadata = self.tool_metadata[tool_name]
        
        info = {
            "name": metadata.name,
            "source": metadata.source.value,
            "category": metadata.category.value,
            "description": metadata.description,
            "version": metadata.version,
            "tags": metadata.tags,
            "capabilities": metadata.capabilities,
            "dependencies": metadata.dependencies,
            "security_level": metadata.security_level,
            "execution_context": metadata.execution_context,
            "langchain_compatible": metadata.langchain_compatible,
            "async_compatible": metadata.async_compatible
        }
        
        # Add source-specific information
        if metadata.source == ToolSource.ORCHESTRATOR:
            orchestrator_tool = self.get_tool(tool_name)
            if orchestrator_tool:
                info["parameters"] = [
                    {
                        "name": p.name,
                        "type": p.type,
                        "description": p.description,
                        "required": p.required,
                        "default": p.default
                    }
                    for p in orchestrator_tool.parameters
                ]
        
        elif metadata.source == ToolSource.LANGCHAIN:
            langchain_tool = self.langchain_tools.get(tool_name)
            if langchain_tool:
                info["langchain_type"] = type(langchain_tool).__name__
                if hasattr(langchain_tool, 'args_schema') and langchain_tool.args_schema:
                    schema = langchain_tool.args_schema.schema()
                    info["input_schema"] = schema
        
        elif metadata.source == ToolSource.MCP:
            mcp_tool = self.mcp_tools.get(tool_name)
            if mcp_tool:
                info["mcp_server"] = metadata.mcp_server
                info["input_schema"] = mcp_tool.inputSchema
        
        return info
    
    def get_langchain_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool as a LangChain tool (with automatic adaptation if needed)."""
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not available")
            return None
        
        # Check if it's already a LangChain tool
        if tool_name in self.langchain_tools:
            return self.langchain_tools[tool_name]
        
        # Check if it's an orchestrator tool that can be adapted
        orchestrator_tool = self.get_tool(tool_name)
        if orchestrator_tool:
            try:
                adapter = ToolAdapterFactory.create_langchain_adapter(orchestrator_tool)
                return adapter.get_langchain_tool()  # Return the StructuredTool
            except Exception as e:
                logger.error(f"Failed to create LangChain adapter for {tool_name}: {e}")
                return None
        
        return None
    
    # Enhanced execution methods
    
    async def execute_tool_enhanced(
        self,
        tool_name: str,
        execution_context: Optional[str] = None,
        **kwargs
    ) -> ToolExecutionResult:
        """Execute tool with enhanced context and result handling."""
        import time
        start_time = time.time()
        
        # Get tool metadata
        metadata = self.tool_metadata.get(tool_name)
        if not metadata:
            return ToolExecutionResult(
                success=False,
                output=None,
                source=ToolSource.ORCHESTRATOR,
                execution_time=0.0,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' not found in registry"
            )
        
        # Determine execution context
        context = execution_context or metadata.execution_context
        
        try:
            # Execute based on source and context
            if metadata.source == ToolSource.MCP:
                result = await self._execute_mcp_tool(tool_name, metadata, **kwargs)
            elif context == "sandboxed" and self.sandbox_manager:
                result = await self._execute_sandboxed_tool(tool_name, metadata, **kwargs)
            else:
                result = await self._execute_standard_tool(tool_name, metadata, **kwargs)
            
            execution_time = time.time() - start_time
            
            return ToolExecutionResult(
                success=True,
                output=result,
                source=metadata.source,
                execution_time=execution_time,
                tool_name=tool_name,
                metadata={
                    "context": context,
                    "security_level": metadata.security_level,
                    "category": metadata.category.value
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            
            return ToolExecutionResult(
                success=False,
                output=None,
                source=metadata.source,
                execution_time=execution_time,
                tool_name=tool_name,
                error=str(e)
            )
    
    async def _execute_standard_tool(
        self,
        tool_name: str,
        metadata: ToolMetadata,
        **kwargs
    ) -> Any:
        """Execute tool in standard context."""
        # Use base registry execution for orchestrator tools
        return await super().execute_tool(tool_name, **kwargs)
    
    async def _execute_mcp_tool(
        self,
        tool_name: str,
        metadata: ToolMetadata,
        **kwargs
    ) -> Any:
        """Execute MCP tool via MCP adapter."""
        if not self.mcp_adapter or not metadata.mcp_server:
            raise RuntimeError(f"MCP execution not available for {tool_name}")
        
        # Extract actual tool name (remove mcp:server: prefix)
        actual_tool_name = tool_name.split(':')[-1]
        
        # Execute via MCP adapter
        result = await self.mcp_adapter.call_tool(
            metadata.mcp_server,
            actual_tool_name,
            kwargs
        )
        
        return result
    
    async def _execute_sandboxed_tool(
        self,
        tool_name: str,
        metadata: ToolMetadata,
        **kwargs
    ) -> Any:
        """Execute tool in sandboxed environment."""
        if not self.sandbox_manager:
            raise RuntimeError(f"Sandbox execution not available for {tool_name}")
        
        # For now, fall back to standard execution
        # In future, could wrap execution in sandbox
        logger.warning(f"Sandboxed execution requested for {tool_name}, falling back to standard")
        return await self._execute_standard_tool(tool_name, metadata, **kwargs)
    
    # Utility methods
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        stats = {
            "total_tools": len(self.tool_metadata),
            "by_source": {},
            "by_category": {},
            "by_security_level": {},
            "langchain_available": LANGCHAIN_AVAILABLE,
            "mcp_available": MCP_AVAILABLE,
            "sandbox_available": SANDBOX_AVAILABLE
        }
        
        # Count by source
        for source in ToolSource:
            count = sum(
                1 for meta in self.tool_metadata.values()
                if meta.source == source
            )
            stats["by_source"][source.value] = count
        
        # Count by category
        for category in ToolCategory:
            count = len(self.categories[category])
            stats["by_category"][category.value] = count
        
        # Count by security level
        security_levels = {}
        for meta in self.tool_metadata.values():
            level = meta.security_level
            security_levels[level] = security_levels.get(level, 0) + 1
        stats["by_security_level"] = security_levels
        
        # Add component-specific stats
        if self.mcp_adapter:
            mcp_stats = self.mcp_adapter.get_statistics()
            stats["mcp_servers"] = mcp_stats
        
        return stats
    
    def export_catalog(self) -> Dict[str, Any]:
        """Export comprehensive tool catalog."""
        catalog = {
            "registry_type": "universal",
            "version": "1.0.0",
            "tools": {},
            "categories": {},
            "tags": {},
            "statistics": self.get_statistics()
        }
        
        # Export tool information
        for tool_name in self.tool_metadata:
            catalog["tools"][tool_name] = self.get_tool_info(tool_name)
        
        # Export category information
        for category, tools in self.categories.items():
            catalog["categories"][category.value] = {
                "count": len(tools),
                "tools": sorted(list(tools))
            }
        
        # Export tags information
        for tag, tools in self.tags_index.items():
            catalog["tags"][tag] = {
                "count": len(tools),
                "tools": sorted(list(tools))
            }
        
        return catalog


# Create global universal registry instance
universal_registry = UniversalToolRegistry()


def get_universal_registry() -> UniversalToolRegistry:
    """Get the global universal registry instance."""
    return universal_registry


def migrate_from_base_registry(base_registry: ToolRegistry) -> None:
    """Migrate tools from base registry to universal registry."""
    logger.info("Migrating tools from base registry to universal registry")
    
    # Auto-categorize existing tools based on their names and types
    category_mapping = {
        "headless-browser": ToolCategory.WEB,
        "web-search": ToolCategory.WEB,
        "terminal": ToolCategory.SYSTEM,
        "filesystem": ToolCategory.SYSTEM,
        "data-processing": ToolCategory.DATA,
        "validation": ToolCategory.VALIDATION,
        "report-generator": ToolCategory.REPORT,
        "pdf-compiler": ToolCategory.REPORT,
        "task-delegation": ToolCategory.LLM,
        "multi-model-routing": ToolCategory.LLM,
        "prompt-optimization": ToolCategory.LLM,
        "user-prompt": ToolCategory.USER_INTERACTION,
        "approval-gate": ToolCategory.USER_INTERACTION,
        "feedback-collection": ToolCategory.USER_INTERACTION,
        "pipeline-executor": ToolCategory.PIPELINE,
        "recursion-control": ToolCategory.PIPELINE,
        "image-analysis": ToolCategory.MULTIMODAL,
        "image-generation": ToolCategory.MULTIMODAL,
        "audio-processing": ToolCategory.MULTIMODAL,
        "video-processing": ToolCategory.MULTIMODAL,
        "mcp-server": ToolCategory.MCP_INTEGRATION,
        "mcp-memory": ToolCategory.MCP_INTEGRATION,
        "mcp-resource": ToolCategory.MCP_INTEGRATION,
        "python-executor": ToolCategory.CODE_EXECUTION,
        "auto-debugger": ToolCategory.CODE_EXECUTION
    }
    
    migrated_count = 0
    for tool_name, tool in base_registry.tools.items():
        category = category_mapping.get(tool_name, ToolCategory.CUSTOM)
        
        try:
            universal_registry.register_orchestrator_tool(
                tool,
                category=category,
                tags=[],
                security_level="moderate"
            )
            migrated_count += 1
        except Exception as e:
            logger.error(f"Failed to migrate tool {tool_name}: {e}")
    
    logger.info(f"Successfully migrated {migrated_count} tools to universal registry")


# Auto-migration from default registry
try:
    from .base import default_registry
    if len(default_registry.tools) > 0:
        migrate_from_base_registry(default_registry)
        logger.info("Auto-migration from default registry completed")
except ImportError:
    logger.warning("Could not auto-migrate from default registry")


__all__ = [
    "UniversalToolRegistry",
    "ToolSource",
    "ToolCategory", 
    "ToolMetadata",
    "ToolExecutionResult",
    "universal_registry",
    "get_universal_registry",
    "migrate_from_base_registry"
]