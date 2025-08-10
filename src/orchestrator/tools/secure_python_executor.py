"""Secure Python Executor Tool - Issue #206 Task 2.1

Enhanced Python execution tool that uses the secure Docker infrastructure
with advanced threat detection and resource monitoring.
"""

import logging
from typing import Any, Dict, Optional, List
import asyncio

from .base import Tool
from .secure_tool_executor import (
    SecureToolExecutor,
    ExecutionMode,
    ExecutionEnvironment,
    ExecutionResult
)
from ..security.docker_manager import ResourceLimits, SecurityConfig
from ..security.dependency_manager import PackageInfo, PackageEcosystem

logger = logging.getLogger(__name__)


class SecurePythonExecutorTool(Tool):
    """
    Secure Python execution tool that provides sandboxed code execution
    with comprehensive security analysis and resource monitoring.
    """
    
    def __init__(self):
        super().__init__(
            name="secure-python-executor",
            description="Execute Python code in a secure, monitored sandbox environment"
        )
        
        # Required parameters
        self.add_parameter("code", "string", "Python code to execute")
        
        # Optional parameters
        self.add_parameter(
            "timeout", "integer", "Execution timeout in seconds", 
            required=False, default=60
        )
        self.add_parameter(
            "mode", "string", "Execution mode: auto, trusted, sandboxed, isolated", 
            required=False, default="auto"
        )
        self.add_parameter(
            "memory_limit_mb", "integer", "Memory limit in MB", 
            required=False, default=256
        )
        self.add_parameter(
            "cpu_cores", "number", "CPU core limit", 
            required=False, default=0.5
        )
        self.add_parameter(
            "dependencies", "array", "Python packages to install", 
            required=False, default=[]
        )
        self.add_parameter(
            "network_access", "boolean", "Allow network access", 
            required=False, default=False
        )
        self.add_parameter(
            "filesystem_write", "boolean", "Allow filesystem writes", 
            required=False, default=False
        )
        
        self.executor: Optional[SecureToolExecutor] = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure the secure executor is initialized."""
        if not self._initialized:
            self.executor = SecureToolExecutor(
                default_mode=ExecutionMode.AUTO,
                enable_monitoring=True,
                default_timeout=300
            )
            await self.executor.initialize()
            self.executor.register_tool(self)  # Register self for recursive execution if needed
            self._initialized = True
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute Python code with enhanced security and monitoring."""
        
        await self._ensure_initialized()
        
        # Extract parameters
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", 60)
        mode_str = kwargs.get("mode", "auto")
        memory_limit_mb = kwargs.get("memory_limit_mb", 256)
        cpu_cores = kwargs.get("cpu_cores", 0.5)
        dependencies = kwargs.get("dependencies", [])
        network_access = kwargs.get("network_access", False)
        filesystem_write = kwargs.get("filesystem_write", False)
        
        if not code:
            return {
                "success": False,
                "error": "No code provided for execution",
                "output": None
            }
        
        try:
            # Parse execution mode
            mode_map = {
                "auto": ExecutionMode.AUTO,
                "trusted": ExecutionMode.TRUSTED,
                "sandboxed": ExecutionMode.SANDBOXED,
                "isolated": ExecutionMode.ISOLATED
            }
            mode = mode_map.get(mode_str.lower(), ExecutionMode.AUTO)
            
            # Create resource limits
            custom_limits = ResourceLimits(
                memory_mb=memory_limit_mb,
                cpu_cores=cpu_cores,
                execution_timeout=timeout,
                pids_limit=50,
                disk_limit_mb=128
            )
            
            # Create security configuration
            custom_security = SecurityConfig(
                read_only_root=not filesystem_write,
                network_isolation=not network_access,
                no_new_privileges=True,
                drop_all_capabilities=True,
                user_namespace=True
            )
            
            # Parse dependencies
            parsed_deps = []
            if dependencies:
                for dep in dependencies:
                    if isinstance(dep, str):
                        parsed_deps.append(
                            PackageInfo(name=dep, ecosystem=PackageEcosystem.PYPI)
                        )
                    elif isinstance(dep, dict) and 'name' in dep:
                        parsed_deps.append(
                            PackageInfo(
                                name=dep['name'],
                                version=dep.get('version'),
                                ecosystem=PackageEcosystem.PYPI
                            )
                        )
            
            # Execute with secure executor
            result = await self._execute_python_code(
                code=code,
                mode=mode,
                timeout=timeout,
                custom_limits=custom_limits,
                custom_security=custom_security,
                dependencies=parsed_deps
            )
            
            return self._format_result(result)
            
        except Exception as e:
            logger.error(f"Secure Python execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": None
            }
    
    async def _execute_python_code(
        self,
        code: str,
        mode: ExecutionMode,
        timeout: int,
        custom_limits: ResourceLimits,
        custom_security: SecurityConfig,
        dependencies: List[PackageInfo]
    ) -> ExecutionResult:
        """Execute Python code using the secure execution framework."""
        
        # Prepare execution parameters
        parameters = {
            "code": code,
            "timeout": timeout,
            "capture_output": True
        }
        
        # Add dependencies to execution context manually
        # (In a real implementation, this would be passed through the context)
        if dependencies and hasattr(self.executor, '_current_dependencies'):
            self.executor._current_dependencies = dependencies
        
        # Create a temporary Python executor for this execution
        temp_executor = PythonCodeExecutor()
        
        # Execute using the secure executor
        result = await self.executor.execute_tool(
            tool_name="python_code_executor",
            parameters=parameters,
            mode=mode,
            environment=ExecutionEnvironment.PYTHON,
            timeout=timeout,
            custom_limits=custom_limits,
            custom_security=custom_security
        )
        
        return result
    
    def _format_result(self, result: ExecutionResult) -> Dict[str, Any]:
        """Format the execution result for tool response."""
        
        output = {
            "success": result.success,
            "output": result.output,
            "error": result.error
        }
        
        # Add security information
        if result.security_violations:
            output["security_violations"] = result.security_violations
        
        # Add performance metrics
        if result.performance_metrics:
            output["performance"] = result.performance_metrics
        
        # Add resource usage
        if result.resource_usage:
            output["resource_usage"] = result.resource_usage
        
        # Add execution context information
        if result.execution_context:
            context_info = {
                "execution_id": result.execution_context.execution_id,
                "execution_time": result.execution_context.execution_time,
                "mode": result.execution_context.mode.value,
                "environment": result.execution_context.environment.value
            }
            
            if result.execution_context.security_assessment:
                context_info["threat_level"] = result.execution_context.security_assessment.threat_level.value
                context_info["violations_detected"] = len(result.execution_context.security_assessment.violations)
            
            output["execution_context"] = context_info
        
        return output
    
    async def shutdown(self):
        """Shutdown the secure executor."""
        if self.executor:
            await self.executor.shutdown()
            self._initialized = False


class PythonCodeExecutor(Tool):
    """Internal tool for actual Python code execution."""
    
    def __init__(self):
        super().__init__(
            name="python_code_executor",
            description="Internal Python code executor"
        )
        self.add_parameter("code", "string", "Python code to execute")
        self.add_parameter("timeout", "integer", "Execution timeout", required=False, default=60)
        self.add_parameter("capture_output", "boolean", "Capture output", required=False, default=True)
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute Python code directly."""
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", 60)
        capture_output = kwargs.get("capture_output", True)
        
        try:
            # Create a simple Python execution
            # In a real container environment, this would be executed within the container
            import tempfile
            import subprocess
            import sys
            
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                if capture_output:
                    # Execute with output capture
                    process = await asyncio.create_subprocess_exec(
                        sys.executable, temp_file,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            process.communicate(), timeout=timeout
                        )
                        
                        stdout_str = stdout.decode('utf-8') if stdout else ""
                        stderr_str = stderr.decode('utf-8') if stderr else ""
                        
                        return {
                            "success": process.returncode == 0,
                            "output": stdout_str,
                            "error": stderr_str if stderr_str else None,
                            "return_code": process.returncode
                        }
                        
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.communicate()
                        return {
                            "success": False,
                            "output": "",
                            "error": f"Execution timed out after {timeout} seconds",
                            "return_code": -1
                        }
                else:
                    # Execute without output capture
                    process = await asyncio.create_subprocess_exec(sys.executable, temp_file)
                    
                    try:
                        return_code = await asyncio.wait_for(process.wait(), timeout=timeout)
                        return {
                            "success": return_code == 0,
                            "output": "",
                            "error": None,
                            "return_code": return_code
                        }
                        
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
                        return {
                            "success": False,
                            "output": "",
                            "error": f"Execution timed out after {timeout} seconds",
                            "return_code": -1
                        }
            finally:
                # Clean up temp file
                import os
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "return_code": -1
            }


# Factory function to create the secure Python executor
def create_secure_python_executor() -> SecurePythonExecutorTool:
    """Create and return a secure Python executor tool."""
    return SecurePythonExecutorTool()


# Export for use in tool registry
__all__ = [
    'SecurePythonExecutorTool',
    'PythonCodeExecutor',
    'create_secure_python_executor'
]