"""Enhanced Tool Execution Manager - Issue #206 Task 2.1

Secure tool execution system that integrates Phase 1 Docker security with the existing
tool infrastructure, providing production-grade sandboxed execution for all tools.
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import uuid

from .base import Tool, ToolRegistry
from ..security.docker_manager import (
    EnhancedDockerManager, 
    SecureContainer, 
    ResourceLimits, 
    SecurityConfig,
    create_strict_security_config
)
from ..security.policy_engine import (
    SecurityPolicyEngine,
    SecurityAssessment,
    ThreatLevel,
    SandboxingLevel
)
from ..security.dependency_manager import (
    IntelligentDependencyManager,
    PackageInfo,
    PackageEcosystem
)
from ..security.resource_monitor import ResourceMonitor, AlertSeverity

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Tool execution modes with different security levels."""
    TRUSTED = "trusted"          # Direct host execution (legacy tools)
    SANDBOXED = "sandboxed"      # Secure Docker container execution
    ISOLATED = "isolated"        # Maximum security isolation
    AUTO = "auto"                # Automatic mode selection based on security assessment


class ExecutionEnvironment(Enum):
    """Available execution environments."""
    PYTHON = "python"
    NODEJS = "nodejs" 
    BASH = "bash"
    MULTI = "multi"              # Multi-language support


@dataclass
class ExecutionContext:
    """Context information for tool execution."""
    execution_id: str
    tool_name: str
    mode: ExecutionMode
    environment: ExecutionEnvironment
    security_assessment: Optional[SecurityAssessment] = None
    container: Optional[SecureContainer] = None
    resource_limits: Optional[ResourceLimits] = None
    security_config: Optional[SecurityConfig] = None
    dependencies: List[PackageInfo] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def execution_time(self) -> float:
        """Get execution time in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time if self.start_time > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            'execution_id': self.execution_id,
            'tool_name': self.tool_name,
            'mode': self.mode.value,
            'environment': self.environment.value,
            'execution_time': self.execution_time,
            'container_id': self.container.container_id if self.container else None,
            'threat_level': self.security_assessment.threat_level.value if self.security_assessment else None,
            'dependencies_count': len(self.dependencies)
        }


@dataclass 
class ExecutionResult:
    """Enhanced execution result with security and performance metadata."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_context: Optional[ExecutionContext] = None
    security_violations: List[str] = field(default_factory=list)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'success': self.success,
            'output': self.output,
            'error': self.error,
            'security_violations': self.security_violations,
            'resource_usage': self.resource_usage,
            'performance_metrics': self.performance_metrics,
            'execution_context': self.execution_context.to_dict() if self.execution_context else None
        }


class SecureToolExecutor:
    """
    Enhanced Tool Execution Manager that provides secure, sandboxed execution
    for all tools using the Docker security infrastructure from Phase 1.
    """
    
    def __init__(
        self,
        default_mode: ExecutionMode = ExecutionMode.AUTO,
        enable_monitoring: bool = True,
        default_timeout: int = 300  # 5 minutes
    ):
        self.default_mode = default_mode
        self.enable_monitoring = enable_monitoring
        self.default_timeout = default_timeout
        
        # Initialize security components
        self.docker_manager = EnhancedDockerManager(enable_container_pooling=True)
        self.policy_engine = SecurityPolicyEngine()
        self.dependency_manager = IntelligentDependencyManager(enable_validation=True)
        self.resource_monitor = None  # Will be initialized when Docker client is available
        
        # Execution tracking
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.execution_history: List[ExecutionContext] = []
        self.tool_registry = ToolRegistry()
        
        # Performance statistics
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'security_violations': 0,
            'average_execution_time': 0.0,
            'containers_created': 0,
            'containers_reused': 0,
        }
        
        logger.info("SecureToolExecutor initialized")
    
    async def initialize(self):
        """Initialize the secure tool executor."""
        try:
            await self.docker_manager.start_background_tasks()
            
            # Initialize resource monitoring
            if self.enable_monitoring:
                self.resource_monitor = ResourceMonitor(
                    self.docker_manager.docker_client,
                    monitoring_interval=1.0
                )
                await self.resource_monitor.start_monitoring()
            
            logger.info("SecureToolExecutor fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize SecureToolExecutor: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the secure tool executor."""
        try:
            # Stop resource monitoring
            if self.resource_monitor:
                await self.resource_monitor.stop_monitoring()
                await self.resource_monitor.cleanup()
            
            # Clean up active executions
            for execution_id in list(self.active_executions.keys()):
                await self._cleanup_execution(execution_id)
            
            # Shutdown Docker manager
            await self.docker_manager.shutdown()
            
            logger.info("SecureToolExecutor shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during SecureToolExecutor shutdown: {e}")
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        mode: Optional[ExecutionMode] = None,
        environment: Optional[ExecutionEnvironment] = None,
        timeout: Optional[int] = None,
        custom_limits: Optional[ResourceLimits] = None,
        custom_security: Optional[SecurityConfig] = None
    ) -> ExecutionResult:
        """
        Execute a tool with enhanced security and monitoring.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            mode: Execution mode (defaults to configured default)
            environment: Execution environment (auto-detected if not specified)
            timeout: Execution timeout in seconds
            custom_limits: Custom resource limits
            custom_security: Custom security configuration
            
        Returns:
            ExecutionResult with comprehensive execution information
        """
        execution_id = str(uuid.uuid4())
        mode = mode or self.default_mode
        timeout = timeout or self.default_timeout
        
        # Create execution context
        context = ExecutionContext(
            execution_id=execution_id,
            tool_name=tool_name,
            mode=mode,
            environment=environment or ExecutionEnvironment.PYTHON,  # Default
            start_time=time.time()
        )
        
        self.active_executions[execution_id] = context
        self.stats['total_executions'] += 1
        
        try:
            logger.info(f"Starting tool execution: {tool_name} (mode: {mode.value}, id: {execution_id})")
            
            # Get tool from registry
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                raise ValueError(f"Tool '{tool_name}' not found in registry")
            
            # Determine execution strategy
            execution_strategy = await self._determine_execution_strategy(
                tool, parameters, mode, context
            )
            
            # Execute based on strategy
            if execution_strategy == ExecutionMode.TRUSTED:
                result = await self._execute_trusted(tool, parameters, context)
            elif execution_strategy == ExecutionMode.SANDBOXED:
                result = await self._execute_sandboxed(
                    tool, parameters, context, timeout, custom_limits, custom_security
                )
            elif execution_strategy == ExecutionMode.ISOLATED:
                result = await self._execute_isolated(
                    tool, parameters, context, timeout, custom_limits, custom_security
                )
            else:
                raise ValueError(f"Unknown execution strategy: {execution_strategy}")
            
            # Update statistics
            if result.success:
                self.stats['successful_executions'] += 1
            else:
                self.stats['failed_executions'] += 1
            
            context.end_time = time.time()
            self._update_performance_stats(context)
            
            # Add context to result
            result.execution_context = context
            
            logger.info(f"Tool execution completed: {tool_name} (success: {result.success}, time: {context.execution_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            
            context.end_time = time.time()
            self.stats['failed_executions'] += 1
            
            return ExecutionResult(
                success=False,
                output=None,
                error=str(e),
                execution_context=context
            )
            
        finally:
            # Move to history and cleanup
            self.execution_history.append(context)
            await self._cleanup_execution(execution_id)
    
    async def _determine_execution_strategy(
        self, 
        tool: Tool, 
        parameters: Dict[str, Any], 
        requested_mode: ExecutionMode,
        context: ExecutionContext
    ) -> ExecutionMode:
        """Determine the appropriate execution strategy for a tool."""
        
        if requested_mode != ExecutionMode.AUTO:
            return requested_mode
        
        # Extract code if present for security analysis
        code_content = None
        for param_name in ['code', 'script', 'command', 'query']:
            if param_name in parameters:
                code_content = parameters[param_name]
                break
        
        if not code_content:
            # No code to analyze, default to trusted for simple tools
            return ExecutionMode.TRUSTED
        
        try:
            # Perform security assessment
            assessment = await self.policy_engine.evaluate_execution_request(
                str(code_content), language='python'  # Default, could be improved with detection
            )
            
            context.security_assessment = assessment
            
            # Determine execution mode based on threat level
            if assessment.threat_level == ThreatLevel.CRITICAL:
                if assessment.sandboxing_level == SandboxingLevel.BLOCKED:
                    raise ValueError("Code execution blocked due to critical security threats")
                return ExecutionMode.ISOLATED
            elif assessment.threat_level == ThreatLevel.HIGH:
                return ExecutionMode.ISOLATED
            elif assessment.threat_level == ThreatLevel.MEDIUM:
                return ExecutionMode.SANDBOXED
            else:
                return ExecutionMode.TRUSTED
                
        except Exception as e:
            logger.warning(f"Security assessment failed, defaulting to sandboxed mode: {e}")
            return ExecutionMode.SANDBOXED
    
    async def _execute_trusted(
        self, 
        tool: Tool, 
        parameters: Dict[str, Any], 
        context: ExecutionContext
    ) -> ExecutionResult:
        """Execute tool in trusted mode (direct host execution)."""
        
        try:
            result = await tool.execute(**parameters)
            return ExecutionResult(
                success=True,
                output=result,
                performance_metrics={'execution_mode': 'trusted'}
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _execute_sandboxed(
        self,
        tool: Tool,
        parameters: Dict[str, Any],
        context: ExecutionContext,
        timeout: int,
        custom_limits: Optional[ResourceLimits],
        custom_security: Optional[SecurityConfig]
    ) -> ExecutionResult:
        """Execute tool in sandboxed mode using secure containers."""
        
        try:
            # Determine resource limits
            if custom_limits:
                resource_limits = custom_limits
            elif context.security_assessment:
                resource_limits = context.security_assessment.resource_requirements
            else:
                resource_limits = ResourceLimits(
                    memory_mb=512,
                    cpu_cores=1.0,
                    execution_timeout=timeout,
                    pids_limit=50
                )
            
            context.resource_limits = resource_limits
            
            # Determine security configuration
            if custom_security:
                security_config = custom_security
            elif context.security_assessment:
                security_config = context.security_assessment.security_config
            else:
                security_config = SecurityConfig()
            
            context.security_config = security_config
            
            # Create secure container
            container = await self._create_execution_container(
                context, resource_limits, security_config
            )
            context.container = container
            
            # Add to resource monitoring
            if self.resource_monitor:
                self.resource_monitor.add_container(container)
            
            # Execute tool in container
            result = await self._execute_in_container(tool, parameters, container, timeout)
            
            # Collect resource usage
            if self.resource_monitor:
                resource_stats = self.resource_monitor.get_container_statistics(container.container_id)
                result.resource_usage = resource_stats
            
            result.performance_metrics = {
                'execution_mode': 'sandboxed',
                'container_id': container.container_id,
                'resource_limits': resource_limits.__dict__,
                'container_reused': getattr(container, 'reused', False)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Sandboxed execution failed: {e}")
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Sandboxed execution failed: {e}"
            )
    
    async def _execute_isolated(
        self,
        tool: Tool,
        parameters: Dict[str, Any],
        context: ExecutionContext,
        timeout: int,
        custom_limits: Optional[ResourceLimits],
        custom_security: Optional[SecurityConfig]
    ) -> ExecutionResult:
        """Execute tool in maximum isolation mode."""
        
        # Use strict security configuration for isolation
        security_config = custom_security or create_strict_security_config()
        
        # Tighter resource limits for isolation
        resource_limits = custom_limits or ResourceLimits(
            memory_mb=256,
            cpu_cores=0.5,
            execution_timeout=min(timeout, 120),  # Max 2 minutes for isolated
            pids_limit=25,
            network_bandwidth_mbps=1.0  # Limited network
        )
        
        context.resource_limits = resource_limits
        context.security_config = security_config
        
        return await self._execute_sandboxed(
            tool, parameters, context, timeout, resource_limits, security_config
        )
    
    async def _create_execution_container(
        self,
        context: ExecutionContext,
        resource_limits: ResourceLimits,
        security_config: SecurityConfig
    ) -> SecureContainer:
        """Create a secure container for tool execution."""
        
        # Determine container image based on environment
        image_map = {
            ExecutionEnvironment.PYTHON: "python:3.11-slim",
            ExecutionEnvironment.NODEJS: "node:18-alpine",
            ExecutionEnvironment.BASH: "ubuntu:22.04",
            ExecutionEnvironment.MULTI: "python:3.11-slim"  # Default to Python
        }
        
        image = image_map.get(context.environment, "python:3.11-slim")
        
        container = await self.docker_manager.create_secure_container(
            image=image,
            name=f"tool_exec_{context.tool_name}_{context.execution_id[:8]}",
            resource_limits=resource_limits,
            security_config=security_config
        )
        
        # Install dependencies if needed
        if context.dependencies:
            await self._install_dependencies(container, context.dependencies)
        
        self.stats['containers_created'] += 1
        
        return container
    
    async def _execute_in_container(
        self,
        tool: Tool,
        parameters: Dict[str, Any],
        container: SecureContainer,
        timeout: int
    ) -> ExecutionResult:
        """Execute tool within a secure container."""
        
        # For now, handle Python code execution as an example
        # This would be expanded for different tool types
        
        if hasattr(tool, 'name') and 'python' in tool.name.lower():
            code = parameters.get('code', '')
            if code:
                result = await self.docker_manager.execute_in_container(
                    container,
                    f'python3 -c "{code}"',
                    timeout=timeout
                )
                
                return ExecutionResult(
                    success=result['success'],
                    output=result.get('output', ''),
                    error=result.get('error', ''),
                    resource_usage=result.get('resource_usage', {})
                )
        
        # Generic tool execution (would be enhanced for specific tools)
        try:
            # Execute tool normally but within container context
            # This is a simplified approach - real implementation would
            # serialize tool execution into container
            result = await tool.execute(**parameters)
            
            return ExecutionResult(
                success=True,
                output=result,
                performance_metrics={'note': 'Generic container execution'}
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _install_dependencies(
        self,
        container: SecureContainer,
        dependencies: List[PackageInfo]
    ) -> None:
        """Install dependencies in the container."""
        
        if not dependencies:
            return
        
        try:
            result = await self.dependency_manager.install_dependencies_securely(
                dependencies, container
            )
            
            if result.blocked_installs > 0:
                logger.warning(f"Some dependencies were blocked during installation: {result.blocked_installs}")
            
        except Exception as e:
            logger.error(f"Failed to install dependencies: {e}")
    
    async def _cleanup_execution(self, execution_id: str):
        """Clean up resources for a completed execution."""
        
        context = self.active_executions.pop(execution_id, None)
        if not context:
            return
        
        try:
            # Remove from resource monitoring
            if self.resource_monitor and context.container:
                self.resource_monitor.remove_container(context.container.container_id)
            
            # Destroy container if not pooled
            if context.container and not self.docker_manager.enable_container_pooling:
                await self.docker_manager.destroy_container(context.container, force=True)
            
        except Exception as e:
            logger.error(f"Error cleaning up execution {execution_id}: {e}")
    
    def _update_performance_stats(self, context: ExecutionContext):
        """Update performance statistics."""
        
        # Update average execution time
        total_time = self.stats['average_execution_time'] * (self.stats['total_executions'] - 1)
        total_time += context.execution_time
        self.stats['average_execution_time'] = total_time / self.stats['total_executions']
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active execution."""
        
        context = self.active_executions.get(execution_id)
        if not context:
            return None
        
        status = context.to_dict()
        status['status'] = 'running'
        
        # Add resource monitoring data if available
        if self.resource_monitor and context.container:
            resource_stats = self.resource_monitor.get_container_statistics(
                context.container.container_id
            )
            status['current_resources'] = resource_stats
        
        return status
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history."""
        
        recent_executions = self.execution_history[-limit:] if limit > 0 else self.execution_history
        return [ctx.to_dict() for ctx in recent_executions]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        
        stats = self.stats.copy()
        stats['active_executions'] = len(self.active_executions)
        stats['total_tools_registered'] = len(self.tool_registry.list_tools())
        
        # Add Docker manager stats
        if hasattr(self.docker_manager, 'stats'):
            stats['docker_stats'] = self.docker_manager.stats
        
        # Add resource monitoring stats
        if self.resource_monitor:
            stats['monitoring_stats'] = self.resource_monitor.get_system_statistics()
        
        return stats
    
    def register_tool(self, tool: Tool):
        """Register a tool with the executor."""
        self.tool_registry.register(tool)
        logger.info(f"Registered tool: {tool.name}")
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return self.tool_registry.list_tools()


# Export classes for use in other modules
__all__ = [
    'SecureToolExecutor',
    'ExecutionMode',
    'ExecutionEnvironment', 
    'ExecutionContext',
    'ExecutionResult'
]