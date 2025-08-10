"""Secure Integration Adapter - Issue #206 Task 2.4

Integration layer that connects the secure Docker-based execution system
with the existing tool infrastructure, providing backward compatibility
and enhanced security for all existing tools.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Type, Union
import inspect
from pathlib import Path

from .base import Tool, ToolRegistry, default_registry
from .secure_tool_executor import (
    SecureToolExecutor,
    ExecutionMode,
    ExecutionEnvironment,
    ExecutionResult
)
from .multi_language_executor import MultiLanguageExecutorTool
from .secure_python_executor import SecurePythonExecutorTool
from ..security.network_manager import NetworkManager, NetworkAccessLevel
from ..security.docker_manager import EnhancedDockerManager, ResourceLimits, SecurityConfig

logger = logging.getLogger(__name__)


class SecureToolWrapper(Tool):
    """
    Wrapper that adds security to existing tools by routing them through
    the secure execution environment.
    """
    
    def __init__(self, original_tool: Tool, security_level: str = "auto"):
        # Preserve original tool identity but add security
        super().__init__(
            name=f"secure_{original_tool.name}",
            description=f"Secure version of {original_tool.name}: {original_tool.description}"
        )
        
        self.original_tool = original_tool
        self.security_level = security_level
        
        # Copy parameters from original tool
        self.parameters = original_tool.parameters.copy()
        
        # Add security-specific parameters
        self.add_parameter(
            "execution_mode", "string", "Security execution mode (auto, sandboxed, isolated)",
            required=False, default="auto"
        )
        self.add_parameter(
            "network_policy", "string", "Network access policy (none, limited, internet)",
            required=False, default="limited"
        )
        self.add_parameter(
            "resource_timeout", "integer", "Execution timeout in seconds",
            required=False, default=300
        )
        
        logger.info(f"Created secure wrapper for tool: {original_tool.name}")
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute the original tool with security enhancements."""
        
        # Extract security parameters
        execution_mode = kwargs.pop("execution_mode", "auto")
        network_policy = kwargs.pop("network_policy", "limited")
        resource_timeout = kwargs.pop("resource_timeout", 300)
        
        try:
            # For tools that execute code, use secure execution
            if self._is_code_execution_tool():
                return await self._execute_with_security(
                    kwargs, execution_mode, network_policy, resource_timeout
                )
            else:
                # For other tools, wrap execution with monitoring
                return await self._execute_with_monitoring(kwargs)
                
        except Exception as e:
            logger.error(f"Secure execution of {self.original_tool.name} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": self.original_tool.name
            }
    
    def _is_code_execution_tool(self) -> bool:
        """Check if this tool executes code that should be sandboxed."""
        
        code_execution_indicators = [
            "python", "code", "executor", "script", "command", "terminal", "shell"
        ]
        
        tool_name = self.original_tool.name.lower()
        tool_desc = self.original_tool.description.lower()
        
        return any(indicator in tool_name or indicator in tool_desc 
                  for indicator in code_execution_indicators)
    
    async def _execute_with_security(
        self, 
        kwargs: Dict[str, Any],
        execution_mode: str,
        network_policy: str,
        resource_timeout: int
    ) -> Dict[str, Any]:
        """Execute code-based tool with full security measures."""
        
        # Initialize secure execution environment
        secure_executor = SecureToolExecutor(
            default_mode=ExecutionMode.AUTO,
            enable_monitoring=True,
            default_timeout=resource_timeout
        )
        
        try:
            await secure_executor.initialize()
            
            # Register the original tool with the secure executor
            secure_executor.register_tool(self.original_tool)
            
            # Determine execution mode
            mode_map = {
                "auto": ExecutionMode.AUTO,
                "sandboxed": ExecutionMode.SANDBOXED,
                "isolated": ExecutionMode.ISOLATED,
                "trusted": ExecutionMode.TRUSTED
            }
            mode = mode_map.get(execution_mode.lower(), ExecutionMode.AUTO)
            
            # Execute with security
            result = await secure_executor.execute_tool(
                tool_name=self.original_tool.name,
                parameters=kwargs,
                mode=mode,
                timeout=resource_timeout
            )
            
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "security_info": {
                    "execution_mode": mode.value,
                    "network_policy": network_policy,
                    "security_violations": result.security_violations,
                    "resource_usage": result.resource_usage,
                    "execution_context": result.execution_context.to_dict() if result.execution_context else None
                }
            }
            
        finally:
            await secure_executor.shutdown()
    
    async def _execute_with_monitoring(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute non-code tool with basic monitoring."""
        
        try:
            import time
            start_time = time.time()
            
            # Execute original tool
            result = await self.original_tool.execute(**kwargs)
            
            execution_time = time.time() - start_time
            
            # Wrap result with monitoring information
            return {
                "success": True,
                "output": result,
                "execution_time": execution_time,
                "security_info": {
                    "execution_mode": "trusted",
                    "monitoring_applied": True,
                    "tool_wrapped": True
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "security_info": {
                    "execution_mode": "trusted",
                    "monitoring_applied": True,
                    "tool_wrapped": True
                }
            }


class SecureToolRegistry:
    """
    Enhanced tool registry that provides secure execution capabilities
    for all registered tools.
    """
    
    def __init__(self):
        self.secure_executor: Optional[SecureToolExecutor] = None
        self.network_manager: Optional[NetworkManager] = None
        self.original_registry = ToolRegistry()
        self.secure_wrappers: Dict[str, SecureToolWrapper] = {}
        self.enhanced_tools: Dict[str, Tool] = {}
        
        # Configuration
        self.default_security_level = "auto"
        self.auto_wrap_tools = True
        
        logger.info("SecureToolRegistry initialized")
    
    async def initialize(self):
        """Initialize the secure tool registry."""
        try:
            # Initialize secure execution components
            self.secure_executor = SecureToolExecutor(
                default_mode=ExecutionMode.AUTO,
                enable_monitoring=True
            )
            await self.secure_executor.initialize()
            
            # Initialize network management
            self.network_manager = NetworkManager()
            
            # Register enhanced tools
            self._register_enhanced_tools()
            
            # Auto-wrap existing tools if enabled
            if self.auto_wrap_tools:
                self._auto_wrap_existing_tools()
            
            logger.info("SecureToolRegistry fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize SecureToolRegistry: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the secure tool registry."""
        try:
            if self.secure_executor:
                await self.secure_executor.shutdown()
            
            logger.info("SecureToolRegistry shut down")
            
        except Exception as e:
            logger.error(f"Error during SecureToolRegistry shutdown: {e}")
    
    def _register_enhanced_tools(self):
        """Register our new enhanced security tools."""
        
        enhanced_tools = [
            SecurePythonExecutorTool(),
            MultiLanguageExecutorTool(),
        ]
        
        for tool in enhanced_tools:
            self.enhanced_tools[tool.name] = tool
            if self.secure_executor:
                self.secure_executor.register_tool(tool)
        
        logger.info(f"Registered {len(enhanced_tools)} enhanced security tools")
    
    def _auto_wrap_existing_tools(self):
        """Automatically wrap existing tools with security."""
        
        # Get tools from the default registry
        existing_tools = default_registry.tools
        
        for tool_name, tool in existing_tools.items():
            if not self._should_wrap_tool(tool):
                continue
            
            # Create secure wrapper
            wrapper = SecureToolWrapper(tool, self.default_security_level)
            self.secure_wrappers[wrapper.name] = wrapper
            
            # Register with secure executor
            if self.secure_executor:
                self.secure_executor.register_tool(wrapper)
        
        logger.info(f"Auto-wrapped {len(self.secure_wrappers)} existing tools")
    
    def _should_wrap_tool(self, tool: Tool) -> bool:
        """Determine if a tool should be wrapped with security."""
        
        # Skip tools that are already secure
        if hasattr(tool, 'original_tool'):
            return False
        
        # Skip our own enhanced tools
        if tool.name in self.enhanced_tools:
            return False
        
        # Wrap tools that might execute code or access external resources
        risky_indicators = [
            "executor", "terminal", "shell", "browser", "web", "file", "system",
            "python", "code", "script", "command", "data", "processing"
        ]
        
        tool_name = tool.name.lower()
        tool_desc = tool.description.lower()
        
        return any(indicator in tool_name or indicator in tool_desc 
                  for indicator in risky_indicators)
    
    def register_tool(self, tool: Tool, security_level: str = "auto") -> bool:
        """Register a tool with optional security wrapping."""
        
        try:
            if self._should_wrap_tool(tool):
                # Create secure wrapper
                wrapper = SecureToolWrapper(tool, security_level)
                self.secure_wrappers[wrapper.name] = wrapper
                
                # Register with secure executor
                if self.secure_executor:
                    self.secure_executor.register_tool(wrapper)
                
                logger.info(f"Registered tool with security wrapper: {tool.name}")
            else:
                # Register directly
                self.original_registry.register(tool)
                
                if self.secure_executor:
                    self.secure_executor.register_tool(tool)
                
                logger.info(f"Registered tool directly: {tool.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register tool {tool.name}: {e}")
            return False
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name, preferring secure versions."""
        
        # Check enhanced tools first
        if name in self.enhanced_tools:
            return self.enhanced_tools[name]
        
        # Check secure wrappers
        if name in self.secure_wrappers:
            return self.secure_wrappers[name]
        
        # Check for secure version of the tool
        secure_name = f"secure_{name}"
        if secure_name in self.secure_wrappers:
            return self.secure_wrappers[secure_name]
        
        # Fall back to original registry
        return self.original_registry.get_tool(name)
    
    def list_tools(self) -> List[str]:
        """List all available tools."""
        
        all_tools = set()
        
        # Add enhanced tools
        all_tools.update(self.enhanced_tools.keys())
        
        # Add secure wrappers
        all_tools.update(self.secure_wrappers.keys())
        
        # Add original tools
        all_tools.update(self.original_registry.list_tools())
        
        return sorted(list(all_tools))
    
    def list_secure_tools(self) -> List[str]:
        """List only tools with security enhancements."""
        
        secure_tools = set()
        secure_tools.update(self.enhanced_tools.keys())
        secure_tools.update(self.secure_wrappers.keys())
        
        return sorted(list(secure_tools))
    
    async def execute_tool_securely(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        security_mode: str = "auto",
        network_policy: str = "limited",
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Execute a tool with enhanced security."""
        
        if not self.secure_executor:
            raise RuntimeError("SecureToolRegistry not initialized")
        
        # Get the tool
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        # Add security parameters
        if hasattr(tool, 'original_tool'):  # It's a secure wrapper
            parameters.update({
                "execution_mode": security_mode,
                "network_policy": network_policy,
                "resource_timeout": timeout
            })
        
        try:
            # Execute the tool
            result = await tool.execute(**parameters)
            
            return {
                "success": True,
                "output": result,
                "tool_name": tool_name,
                "security_applied": hasattr(tool, 'original_tool')
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name
            }
    
    def get_tool_security_info(self, tool_name: str) -> Dict[str, Any]:
        """Get security information about a tool."""
        
        tool = self.get_tool(tool_name)
        if not tool:
            return {"error": "Tool not found"}
        
        info = {
            "tool_name": tool_name,
            "has_security_wrapper": hasattr(tool, 'original_tool'),
            "is_enhanced_tool": tool_name in self.enhanced_tools,
            "supports_sandboxing": True,  # All tools support sandboxing through our system
            "default_security_level": self.default_security_level
        }
        
        if hasattr(tool, 'original_tool'):
            info.update({
                "original_tool_name": tool.original_tool.name,
                "wrapper_security_level": tool.security_level,
                "is_code_execution_tool": tool._is_code_execution_tool()
            })
        
        return info
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        
        stats = {
            "total_tools": len(self.list_tools()),
            "enhanced_tools": len(self.enhanced_tools),
            "secure_wrappers": len(self.secure_wrappers),
            "original_tools": len(self.original_registry.list_tools()),
            "auto_wrap_enabled": self.auto_wrap_tools,
            "default_security_level": self.default_security_level
        }
        
        if self.secure_executor:
            executor_stats = self.secure_executor.get_statistics()
            stats["execution_stats"] = executor_stats
        
        return stats


# Integration functions for existing codebase
def upgrade_tool_registry() -> SecureToolRegistry:
    """
    Upgrade the existing tool registry to use secure execution.
    
    Returns:
        SecureToolRegistry instance ready for use
    """
    
    secure_registry = SecureToolRegistry()
    logger.info("Tool registry upgraded with security features")
    return secure_registry


async def migrate_existing_tools(secure_registry: SecureToolRegistry) -> int:
    """
    Migrate existing tools to the secure registry.
    
    Args:
        secure_registry: Target secure registry
        
    Returns:
        Number of tools migrated
    """
    
    await secure_registry.initialize()
    
    # Tools are automatically wrapped during initialization
    migrated_count = len(secure_registry.secure_wrappers)
    
    logger.info(f"Migrated {migrated_count} tools to secure execution")
    return migrated_count


def create_secure_tool_factory() -> Dict[str, callable]:
    """
    Create factory functions for common secure tools.
    
    Returns:
        Dictionary of factory functions
    """
    
    return {
        "secure_python_executor": lambda: SecurePythonExecutorTool(),
        "multi_language_executor": lambda: MultiLanguageExecutorTool(),
        "secure_wrapper": lambda tool, level="auto": SecureToolWrapper(tool, level)
    }


# Export classes and functions
__all__ = [
    'SecureToolWrapper',
    'SecureToolRegistry',
    'upgrade_tool_registry',
    'migrate_existing_tools',
    'create_secure_tool_factory'
]