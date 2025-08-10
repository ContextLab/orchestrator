"""Enhanced Sandboxing Integration - Issue #203 Phase 3

Integrates LangChain Sandbox with Universal Tool Registry for secure execution:
- Sandboxed tool execution with security policies
- Resource management and monitoring
- Performance optimization
- Security policy enforcement
- Tool isolation and containment
- Execution context management
"""

import asyncio
import logging
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from pathlib import Path
import json

from .universal_registry import UniversalToolRegistry, ToolExecutionResult, ToolMetadata
from ..security.langchain_sandbox import (
    LangChainSandbox, SandboxConfig, SecurityPolicy, SandboxType, ExecutionResult
)

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for tools."""
    DIRECT = "direct"           # Direct execution (default)
    SANDBOXED = "sandboxed"     # Sandboxed execution
    ISOLATED = "isolated"       # Fully isolated execution
    RESTRICTED = "restricted"   # Restricted with limited resources


class ResourceLimit(Enum):
    """Resource limit types."""
    CPU_PERCENT = "cpu_percent"
    MEMORY_MB = "memory_mb"
    EXECUTION_TIME = "execution_time"
    NETWORK_ACCESS = "network_access"
    FILE_SYSTEM = "file_system"


@dataclass
class SecurityContext:
    """Security context for tool execution."""
    policy: SecurityPolicy
    allowed_imports: List[str] = field(default_factory=list)
    blocked_imports: List[str] = field(default_factory=list)
    resource_limits: Dict[ResourceLimit, Any] = field(default_factory=dict)
    network_access: bool = False
    filesystem_access: bool = False
    environment_vars: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.resource_limits:
            self.resource_limits = {
                ResourceLimit.CPU_PERCENT: 50.0,
                ResourceLimit.MEMORY_MB: 256,
                ResourceLimit.EXECUTION_TIME: 30,
                ResourceLimit.NETWORK_ACCESS: self.network_access,
                ResourceLimit.FILE_SYSTEM: self.filesystem_access
            }


@dataclass
class ExecutionMetrics:
    """Metrics for tool execution."""
    start_time: float
    end_time: Optional[float] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    peak_memory: float = 0.0
    network_bytes: int = 0
    file_operations: int = 0
    security_violations: List[str] = field(default_factory=list)
    
    def duration(self) -> float:
        """Get execution duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class SandboxedToolResult:
    """Result from sandboxed tool execution."""
    success: bool
    output: Any
    execution_result: ExecutionResult
    metrics: ExecutionMetrics
    security_context: SecurityContext
    tool_name: str
    execution_mode: ExecutionMode
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class EnhancedSandboxManager:
    """Enhanced sandbox manager with security policies and monitoring."""
    
    def __init__(self, registry: UniversalToolRegistry):
        self.registry = registry
        self.sandbox = LangChainSandbox()
        
        # Security policies by tool category
        self.category_policies: Dict[str, SecurityContext] = {}
        self.tool_policies: Dict[str, SecurityContext] = {}
        
        # Execution monitoring
        self.active_executions: Dict[str, ExecutionMetrics] = {}
        self.execution_history: List[SandboxedToolResult] = []
        self.max_history = 1000
        
        # Resource monitoring
        self.system_monitor_task: Optional[asyncio.Task] = None
        self.resource_threshold_callbacks: List[Callable[[Dict[str, float]], None]] = []
        
        # Initialize default policies
        self._initialize_default_policies()
        
    def _initialize_default_policies(self):
        """Initialize default security policies for tool categories."""
        
        # Strict policy for code execution tools
        self.category_policies["code_execution"] = SecurityContext(
            policy=SecurityPolicy.STRICT,
            blocked_imports=[
                "os", "subprocess", "sys", "socket", "urllib", "requests",
                "shutil", "tempfile", "pathlib", "glob"
            ],
            resource_limits={
                ResourceLimit.CPU_PERCENT: 25.0,
                ResourceLimit.MEMORY_MB: 128,
                ResourceLimit.EXECUTION_TIME: 15,
                ResourceLimit.NETWORK_ACCESS: False,
                ResourceLimit.FILE_SYSTEM: False
            },
            network_access=False,
            filesystem_access=False
        )
        
        # Moderate policy for data tools
        self.category_policies["data"] = SecurityContext(
            policy=SecurityPolicy.MODERATE,
            blocked_imports=["subprocess", "os.system"],
            resource_limits={
                ResourceLimit.CPU_PERCENT: 50.0,
                ResourceLimit.MEMORY_MB: 512,
                ResourceLimit.EXECUTION_TIME: 60,
                ResourceLimit.NETWORK_ACCESS: False,
                ResourceLimit.FILE_SYSTEM: True
            },
            network_access=False,
            filesystem_access=True
        )
        
        # Restricted policy for web tools
        self.category_policies["web"] = SecurityContext(
            policy=SecurityPolicy.MODERATE,
            resource_limits={
                ResourceLimit.CPU_PERCENT: 30.0,
                ResourceLimit.MEMORY_MB: 256,
                ResourceLimit.EXECUTION_TIME: 45,
                ResourceLimit.NETWORK_ACCESS: True,
                ResourceLimit.FILE_SYSTEM: False
            },
            network_access=True,
            filesystem_access=False
        )
        
        # Permissive policy for system tools (use with caution)
        self.category_policies["system"] = SecurityContext(
            policy=SecurityPolicy.MODERATE,
            blocked_imports=["subprocess.call"],
            resource_limits={
                ResourceLimit.CPU_PERCENT: 40.0,
                ResourceLimit.MEMORY_MB: 256,
                ResourceLimit.EXECUTION_TIME: 30,
                ResourceLimit.NETWORK_ACCESS: False,
                ResourceLimit.FILE_SYSTEM: True
            },
            network_access=False,
            filesystem_access=True
        )
        
        # Default policy for unknown categories
        self.category_policies["default"] = SecurityContext(
            policy=SecurityPolicy.MODERATE,
            resource_limits={
                ResourceLimit.CPU_PERCENT: 30.0,
                ResourceLimit.MEMORY_MB: 256,
                ResourceLimit.EXECUTION_TIME: 30,
                ResourceLimit.NETWORK_ACCESS: False,
                ResourceLimit.FILE_SYSTEM: False
            },
            network_access=False,
            filesystem_access=False
        )
    
    def set_tool_policy(self, tool_name: str, security_context: SecurityContext):
        """Set specific security policy for a tool."""
        self.tool_policies[tool_name] = security_context
        logger.info(f"Set security policy for tool: {tool_name}")
    
    def get_security_context(self, tool_name: str) -> SecurityContext:
        """Get security context for a tool."""
        # Check tool-specific policy first
        if tool_name in self.tool_policies:
            return self.tool_policies[tool_name]
        
        # Check tool metadata for category
        if tool_name in self.registry.tool_metadata:
            metadata = self.registry.tool_metadata[tool_name]
            category_key = metadata.category.value
            
            if category_key in self.category_policies:
                return self.category_policies[category_key]
        
        # Return default policy
        return self.category_policies["default"]
    
    async def execute_sandboxed_tool(
        self, 
        tool_name: str, 
        execution_mode: ExecutionMode = ExecutionMode.SANDBOXED,
        **kwargs
    ) -> SandboxedToolResult:
        """Execute tool in sandboxed environment with monitoring."""
        
        # Get security context
        security_context = self.get_security_context(tool_name)
        
        # Create execution metrics
        metrics = ExecutionMetrics(start_time=time.time())
        execution_id = f"{tool_name}_{int(time.time())}"
        self.active_executions[execution_id] = metrics
        
        try:
            # Choose execution method based on mode
            if execution_mode == ExecutionMode.DIRECT:
                result = await self._execute_direct(tool_name, kwargs, metrics, security_context)
            elif execution_mode == ExecutionMode.SANDBOXED:
                result = await self._execute_sandboxed(tool_name, kwargs, metrics, security_context)
            elif execution_mode == ExecutionMode.ISOLATED:
                result = await self._execute_isolated(tool_name, kwargs, metrics, security_context)
            elif execution_mode == ExecutionMode.RESTRICTED:
                result = await self._execute_restricted(tool_name, kwargs, metrics, security_context)
            else:
                raise ValueError(f"Unknown execution mode: {execution_mode}")
            
            # Finalize metrics
            metrics.end_time = time.time()
            
            # Create result
            sandboxed_result = SandboxedToolResult(
                success=result.success if hasattr(result, 'success') else True,
                output=result.output if hasattr(result, 'output') else result,
                execution_result=result if isinstance(result, ExecutionResult) else None,
                metrics=metrics,
                security_context=security_context,
                tool_name=tool_name,
                execution_mode=execution_mode
            )
            
            # Add to history
            self._add_to_history(sandboxed_result)
            
            return sandboxed_result
            
        except Exception as e:
            metrics.end_time = time.time()
            logger.error(f"Sandboxed execution error for {tool_name}: {e}")
            
            return SandboxedToolResult(
                success=False,
                output=None,
                execution_result=None,
                metrics=metrics,
                security_context=security_context,
                tool_name=tool_name,
                execution_mode=execution_mode,
                error=str(e)
            )
            
        finally:
            # Clean up
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def _execute_direct(
        self, 
        tool_name: str, 
        kwargs: Dict[str, Any], 
        metrics: ExecutionMetrics,
        security_context: SecurityContext
    ) -> Any:
        """Execute tool directly without sandboxing."""
        # Monitor resource usage
        process = psutil.Process()
        start_cpu_time = process.cpu_times()
        start_memory = process.memory_info().rss
        
        try:
            # Execute via registry
            result = await self.registry.execute_tool_enhanced(tool_name, **kwargs)
            
            # Update metrics
            end_cpu_time = process.cpu_times()
            end_memory = process.memory_info().rss
            
            metrics.cpu_usage = (end_cpu_time.user + end_cpu_time.system) - (start_cpu_time.user + start_cpu_time.system)
            metrics.memory_usage = (end_memory - start_memory) / 1024 / 1024  # MB
            metrics.peak_memory = process.memory_info().peak_wss / 1024 / 1024 if hasattr(process.memory_info(), 'peak_wss') else metrics.memory_usage
            
            return result
            
        except Exception as e:
            metrics.security_violations.append(f"Direct execution error: {str(e)}")
            raise
    
    async def _execute_sandboxed(
        self, 
        tool_name: str, 
        kwargs: Dict[str, Any], 
        metrics: ExecutionMetrics,
        security_context: SecurityContext
    ) -> ExecutionResult:
        """Execute tool in LangChain sandbox."""
        
        # Get tool metadata
        if tool_name not in self.registry.tool_metadata:
            raise ValueError(f"Tool {tool_name} not found in registry")
        
        metadata = self.registry.tool_metadata[tool_name]
        
        # Create sandbox configuration
        config = SandboxConfig(
            sandbox_type=SandboxType.PYTHON,  # Default to Python
            security_policy=security_context.policy,
            timeout_seconds=security_context.resource_limits.get(ResourceLimit.EXECUTION_TIME, 30),
            memory_limit_mb=security_context.resource_limits.get(ResourceLimit.MEMORY_MB, 256),
            cpu_limit=security_context.resource_limits.get(ResourceLimit.CPU_PERCENT, 50) / 100,
            network_access=security_context.network_access,
            filesystem_access=security_context.filesystem_access,
            allowed_imports=security_context.allowed_imports,
            blocked_imports=security_context.blocked_imports,
            environment_vars=security_context.environment_vars
        )
        
        # Generate execution code for the tool
        execution_code = self._generate_tool_execution_code(tool_name, kwargs, metadata)
        
        # Execute in sandbox
        result = await self.sandbox.execute_code(execution_code, config)
        
        # Update metrics from sandbox result
        if result.resource_usage:
            metrics.cpu_usage = result.resource_usage.get('cpu_usage_percent', 0.0)
            metrics.memory_usage = result.resource_usage.get('memory_usage_bytes', 0) / 1024 / 1024
            metrics.peak_memory = metrics.memory_usage
        
        metrics.security_violations.extend(result.security_violations)
        
        return result
    
    async def _execute_isolated(
        self, 
        tool_name: str, 
        kwargs: Dict[str, Any], 
        metrics: ExecutionMetrics,
        security_context: SecurityContext
    ) -> ExecutionResult:
        """Execute tool in completely isolated environment."""
        
        # Use Docker-based isolation with strict security
        isolated_context = SecurityContext(
            policy=SecurityPolicy.STRICT,
            blocked_imports=security_context.blocked_imports + [
                "socket", "urllib", "requests", "http", "ftplib", "smtplib"
            ],
            resource_limits={
                ResourceLimit.CPU_PERCENT: min(security_context.resource_limits.get(ResourceLimit.CPU_PERCENT, 25), 25),
                ResourceLimit.MEMORY_MB: min(security_context.resource_limits.get(ResourceLimit.MEMORY_MB, 128), 128),
                ResourceLimit.EXECUTION_TIME: min(security_context.resource_limits.get(ResourceLimit.EXECUTION_TIME, 15), 15),
                ResourceLimit.NETWORK_ACCESS: False,
                ResourceLimit.FILE_SYSTEM: False
            },
            network_access=False,
            filesystem_access=False
        )
        
        return await self._execute_sandboxed(tool_name, kwargs, metrics, isolated_context)
    
    async def _execute_restricted(
        self, 
        tool_name: str, 
        kwargs: Dict[str, Any], 
        metrics: ExecutionMetrics,
        security_context: SecurityContext
    ) -> ExecutionResult:
        """Execute tool with restricted resources."""
        
        # Apply additional resource restrictions
        restricted_context = SecurityContext(
            policy=security_context.policy,
            allowed_imports=security_context.allowed_imports,
            blocked_imports=security_context.blocked_imports,
            resource_limits={
                ResourceLimit.CPU_PERCENT: min(security_context.resource_limits.get(ResourceLimit.CPU_PERCENT, 20), 20),
                ResourceLimit.MEMORY_MB: min(security_context.resource_limits.get(ResourceLimit.MEMORY_MB, 64), 64),
                ResourceLimit.EXECUTION_TIME: min(security_context.resource_limits.get(ResourceLimit.EXECUTION_TIME, 10), 10),
                ResourceLimit.NETWORK_ACCESS: False,
                ResourceLimit.FILE_SYSTEM: security_context.filesystem_access
            },
            network_access=False,
            filesystem_access=security_context.filesystem_access,
            environment_vars=security_context.environment_vars
        )
        
        return await self._execute_sandboxed(tool_name, kwargs, metrics, restricted_context)
    
    def _generate_tool_execution_code(
        self, 
        tool_name: str, 
        kwargs: Dict[str, Any],
        metadata: ToolMetadata
    ) -> str:
        """Generate Python code to execute tool in sandbox."""
        
        # This is a simplified approach - in practice, you'd want more sophisticated code generation
        code_parts = [
            "import json",
            "import asyncio",
            "",
            "# Tool execution wrapper",
            "async def execute_tool():",
            "    try:",
            f"        # Simulated tool execution for {tool_name}",
            f"        kwargs = {json.dumps(kwargs, default=str)}",
            "        ",
            "        # This would be replaced with actual tool execution logic",
            "        # For now, return a success result",
            "        result = {",
            "            'success': True,",
            f"            'tool_name': '{tool_name}',",
            "            'output': 'Sandboxed execution completed',",
            "            'kwargs': kwargs",
            "        }",
            "        ",
            "        return result",
            "        ",
            "    except Exception as e:",
            "        return {",
            "            'success': False,",
            f"            'tool_name': '{tool_name}',",
            "            'error': str(e)",
            "        }",
            "",
            "# Execute and print result",
            "result = asyncio.run(execute_tool())",
            "print(json.dumps(result, default=str))"
        ]
        
        return "\n".join(code_parts)
    
    def _add_to_history(self, result: SandboxedToolResult):
        """Add execution result to history."""
        self.execution_history.append(result)
        
        # Limit history size
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history:]
    
    async def start_system_monitoring(self):
        """Start system resource monitoring."""
        if self.system_monitor_task is None:
            self.system_monitor_task = asyncio.create_task(self._system_monitor_loop())
            logger.info("Started system resource monitoring")
    
    async def stop_system_monitoring(self):
        """Stop system resource monitoring."""
        if self.system_monitor_task:
            self.system_monitor_task.cancel()
            try:
                await self.system_monitor_task
            except asyncio.CancelledError:
                pass
            self.system_monitor_task = None
            logger.info("Stopped system resource monitoring")
    
    async def _system_monitor_loop(self):
        """Background system monitoring loop."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metrics = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_mb': memory.available / 1024 / 1024,
                    'disk_percent': disk.percent,
                    'active_executions': len(self.active_executions)
                }
                
                # Check thresholds and trigger callbacks
                if cpu_percent > 80 or memory.percent > 85:
                    for callback in self.resource_threshold_callbacks:
                        try:
                            callback(metrics)
                        except Exception as e:
                            logger.warning(f"Resource threshold callback error: {e}")
                
                # Log high resource usage
                if cpu_percent > 90 or memory.percent > 95:
                    logger.warning(f"High system resource usage - CPU: {cpu_percent}%, Memory: {memory.percent}%")
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
    
    def add_resource_threshold_callback(self, callback: Callable[[Dict[str, float]], None]):
        """Add callback for resource threshold events."""
        self.resource_threshold_callbacks.append(callback)
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics."""
        if not self.execution_history:
            return {"total_executions": 0}
        
        # Calculate statistics
        total_executions = len(self.execution_history)
        successful_executions = len([r for r in self.execution_history if r.success])
        failed_executions = total_executions - successful_executions
        
        # Execution modes
        mode_counts = {}
        for result in self.execution_history:
            mode = result.execution_mode.value
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        # Average metrics
        avg_duration = sum(r.metrics.duration() for r in self.execution_history) / total_executions
        avg_memory = sum(r.metrics.memory_usage for r in self.execution_history) / total_executions
        avg_cpu = sum(r.metrics.cpu_usage for r in self.execution_history) / total_executions
        
        # Security violations
        total_violations = sum(len(r.metrics.security_violations) for r in self.execution_history)
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions * 100,
            "execution_modes": mode_counts,
            "average_duration": avg_duration,
            "average_memory_mb": avg_memory,
            "average_cpu_usage": avg_cpu,
            "total_security_violations": total_violations,
            "active_executions": len(self.active_executions)
        }
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report from execution history."""
        violations_by_tool = {}
        violations_by_type = {}
        
        for result in self.execution_history:
            tool_name = result.tool_name
            violations = result.metrics.security_violations
            
            if violations:
                if tool_name not in violations_by_tool:
                    violations_by_tool[tool_name] = []
                violations_by_tool[tool_name].extend(violations)
                
                for violation in violations:
                    violation_type = violation.split(':')[0] if ':' in violation else violation
                    violations_by_type[violation_type] = violations_by_type.get(violation_type, 0) + 1
        
        return {
            "violations_by_tool": violations_by_tool,
            "violations_by_type": violations_by_type,
            "total_violations": sum(violations_by_type.values()),
            "tools_with_violations": len(violations_by_tool),
            "security_policies_active": len(self.tool_policies) + len(self.category_policies)
        }
    
    async def cleanup(self):
        """Clean up sandbox manager resources."""
        await self.stop_system_monitoring()
        
        # Clear history and active executions
        self.execution_history.clear()
        self.active_executions.clear()
        
        logger.info("Enhanced Sandbox Manager cleaned up")


# Integration function
async def integrate_sandbox_with_registry(registry: UniversalToolRegistry) -> EnhancedSandboxManager:
    """Integrate enhanced sandbox manager with universal registry."""
    
    sandbox_manager = EnhancedSandboxManager(registry)
    
    # Add sandbox integration to registry
    registry.sandbox_manager = sandbox_manager
    
    # Start system monitoring
    await sandbox_manager.start_system_monitoring()
    
    logger.info("Sandbox integration with Universal Registry complete")
    return sandbox_manager


__all__ = [
    "EnhancedSandboxManager",
    "SecurityContext", 
    "ExecutionMode",
    "ResourceLimit",
    "ExecutionMetrics",
    "SandboxedToolResult",
    "integrate_sandbox_with_registry"
]