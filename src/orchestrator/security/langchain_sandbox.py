"""LangChain Sandbox - Phase 3 Security Features

Secure execution environment using LangChain sandbox capabilities with Docker isolation.
Provides safe code execution, dependency management, and resource constraints.
"""

import os
import time
import logging
import tempfile
import subprocess
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import docker
import tarfile
import io

logger = logging.getLogger(__name__)


class SandboxType(Enum):
    """Types of sandbox environments."""
    PYTHON = "python"
    JAVASCRIPT = "javascript" 
    BASH = "bash"
    DOCKER = "docker"


class SecurityPolicy(Enum):
    """Security policy levels."""
    STRICT = "strict"       # Maximum restrictions
    MODERATE = "moderate"   # Balanced restrictions
    PERMISSIVE = "permissive"  # Minimal restrictions


@dataclass
class ExecutionResult:
    """Result of sandbox execution."""
    success: bool
    output: str
    error: str
    execution_time: float
    exit_code: Optional[int] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    security_violations: List[str] = field(default_factory=list)


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    sandbox_type: SandboxType
    security_policy: SecurityPolicy = SecurityPolicy.MODERATE
    timeout_seconds: int = 30
    memory_limit_mb: int = 512
    cpu_limit: float = 1.0  # CPU cores
    network_access: bool = False
    filesystem_access: bool = False
    allowed_imports: Optional[List[str]] = None
    blocked_imports: Optional[List[str]] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.allowed_imports is None:
            self.allowed_imports = []
        if self.blocked_imports is None:
            self.blocked_imports = []


class LangChainSandbox:
    """Secure sandbox for executing code with LangChain integration.
    
    Features:
    - Docker-based isolation
    - Resource limits (CPU, memory, time)
    - Network isolation
    - Filesystem restrictions
    - Import filtering
    - Automatic dependency installation
    - Security policy enforcement
    """
    
    def __init__(self, base_image: str = "python:3.11-slim"):
        """Initialize LangChain sandbox.
        
        Args:
            base_image: Base Docker image for sandbox containers
        """
        self.base_image = base_image
        self.docker_client = None
        self._init_docker()
        
        # Security policies
        self.security_policies = {
            SecurityPolicy.STRICT: {
                "blocked_imports": [
                    "os", "subprocess", "sys", "socket", "urllib", "requests",
                    "http", "ftplib", "telnetlib", "smtplib", "poplib", 
                    "imaplib", "tempfile", "shutil", "glob", "pathlib"
                ],
                "blocked_builtins": ["eval", "exec", "compile", "__import__", "open"],
                "network_access": False,
                "filesystem_access": False
            },
            SecurityPolicy.MODERATE: {
                "blocked_imports": [
                    "subprocess", "os.system", "os.popen", "os.spawn*",
                    "tempfile", "shutil.rmtree"
                ],
                "blocked_builtins": ["eval", "exec"],
                "network_access": False,
                "filesystem_access": True  # Limited
            },
            SecurityPolicy.PERMISSIVE: {
                "blocked_imports": ["os.system", "subprocess.call"],
                "blocked_builtins": [],
                "network_access": True,
                "filesystem_access": True
            }
        }
        
        # Default configurations for different sandbox types
        self.default_configs = {
            SandboxType.PYTHON: SandboxConfig(
                sandbox_type=SandboxType.PYTHON,
                security_policy=SecurityPolicy.MODERATE,
                timeout_seconds=30,
                memory_limit_mb=256,
                cpu_limit=0.5
            ),
            SandboxType.JAVASCRIPT: SandboxConfig(
                sandbox_type=SandboxType.JAVASCRIPT,
                security_policy=SecurityPolicy.MODERATE,
                timeout_seconds=15,
                memory_limit_mb=128,
                cpu_limit=0.5
            ),
            SandboxType.BASH: SandboxConfig(
                sandbox_type=SandboxType.BASH,
                security_policy=SecurityPolicy.STRICT,
                timeout_seconds=10,
                memory_limit_mb=64,
                cpu_limit=0.25
            )
        }
        
    def _init_docker(self) -> None:
        """Initialize Docker client."""
        try:
            self.docker_client = docker.from_env()
            # Test Docker connectivity
            self.docker_client.ping()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None
    
    async def execute_code(
        self,
        code: str,
        config: Optional[SandboxConfig] = None,
        dependencies: Optional[List[str]] = None
    ) -> ExecutionResult:
        """Execute code in secure sandbox.
        
        Args:
            code: Code to execute
            config: Sandbox configuration (uses default if None)
            dependencies: Additional dependencies to install
            
        Returns:
            ExecutionResult with execution details
        """
        if not self.docker_client:
            return ExecutionResult(
                success=False,
                output="",
                error="Docker client not available",
                execution_time=0.0,
                security_violations=["Docker unavailable"]
            )
        
        if config is None:
            config = self.default_configs[SandboxType.PYTHON]
        
        start_time = time.time()
        
        try:
            # Pre-execution security checks
            security_violations = self._check_security_violations(code, config)
            if security_violations:
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Security policy violations detected",
                    execution_time=time.time() - start_time,
                    security_violations=security_violations
                )
            
            # Create execution environment
            container_name = f"langchain-sandbox-{int(time.time())}-{os.getpid()}"
            
            # Execute based on sandbox type
            if config.sandbox_type == SandboxType.PYTHON:
                result = await self._execute_python_code(code, config, dependencies, container_name)
            elif config.sandbox_type == SandboxType.JAVASCRIPT:
                result = await self._execute_javascript_code(code, config, dependencies, container_name)
            elif config.sandbox_type == SandboxType.BASH:
                result = await self._execute_bash_code(code, config, container_name)
            else:
                result = ExecutionResult(
                    success=False,
                    output="",
                    error=f"Unsupported sandbox type: {config.sandbox_type}",
                    execution_time=time.time() - start_time
                )
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
                security_violations=["Execution exception"]
            )
    
    async def execute_python_code(
        self,
        code: str,
        dependencies: Optional[List[str]] = None,
        security_policy: SecurityPolicy = SecurityPolicy.MODERATE
    ) -> ExecutionResult:
        """Convenience method for Python code execution."""
        config = SandboxConfig(
            sandbox_type=SandboxType.PYTHON,
            security_policy=security_policy
        )
        return await self.execute_code(code, config, dependencies)
    
    async def execute_bash_command(
        self,
        command: str,
        security_policy: SecurityPolicy = SecurityPolicy.STRICT
    ) -> ExecutionResult:
        """Convenience method for bash command execution."""
        config = SandboxConfig(
            sandbox_type=SandboxType.BASH,
            security_policy=security_policy,
            timeout_seconds=10
        )
        return await self.execute_code(command, config)
    
    def _check_security_violations(self, code: str, config: SandboxConfig) -> List[str]:
        """Check code for security policy violations."""
        violations = []
        policy = self.security_policies[config.security_policy]
        
        # Check blocked imports
        for blocked_import in policy.get("blocked_imports", []):
            if f"import {blocked_import}" in code or f"from {blocked_import}" in code:
                violations.append(f"Blocked import: {blocked_import}")
        
        # Check blocked builtins
        for blocked_builtin in policy.get("blocked_builtins", []):
            if blocked_builtin in code:
                violations.append(f"Blocked builtin: {blocked_builtin}")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            "__import__",
            "eval(",
            "exec(",
            "compile(",
            "getattr(",
            "setattr(",
            "hasattr(",
            "globals()",
            "locals()",
            "vars()"
        ]
        
        if config.security_policy == SecurityPolicy.STRICT:
            for pattern in dangerous_patterns:
                if pattern in code:
                    violations.append(f"Dangerous pattern: {pattern}")
        
        return violations
    
    async def _execute_python_code(
        self,
        code: str,
        config: SandboxConfig,
        dependencies: Optional[List[str]],
        container_name: str
    ) -> ExecutionResult:
        """Execute Python code in Docker container."""
        try:
            # Prepare execution script
            script_content = self._prepare_python_script(code, config, dependencies)
            
            # Create container
            container = await self._create_container(
                image="python:3.11-slim",
                name=container_name,
                config=config,
                command=["python", "-c", script_content]
            )
            
            # Execute with timeout
            result = await self._run_container_with_timeout(container, config.timeout_seconds)
            
            # Parse resource usage
            resource_usage = await self._get_container_stats(container)
            result.resource_usage = resource_usage
            
            # Cleanup
            await self._cleanup_container(container)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing Python code: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=0.0
            )
    
    async def _execute_javascript_code(
        self,
        code: str,
        config: SandboxConfig,
        dependencies: Optional[List[str]],
        container_name: str
    ) -> ExecutionResult:
        """Execute JavaScript code in Docker container."""
        try:
            # Prepare Node.js script
            script_content = self._prepare_javascript_script(code, config, dependencies)
            
            # Create container with Node.js
            container = await self._create_container(
                image="node:18-slim",
                name=container_name,
                config=config,
                command=["node", "-e", script_content]
            )
            
            # Execute with timeout
            result = await self._run_container_with_timeout(container, config.timeout_seconds)
            
            # Parse resource usage
            resource_usage = await self._get_container_stats(container)
            result.resource_usage = resource_usage
            
            # Cleanup
            await self._cleanup_container(container)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing JavaScript code: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=0.0
            )
    
    async def _execute_bash_code(
        self,
        code: str,
        config: SandboxConfig,
        container_name: str
    ) -> ExecutionResult:
        """Execute bash commands in Docker container."""
        try:
            # Create container with minimal Ubuntu
            container = await self._create_container(
                image="ubuntu:22.04",
                name=container_name,
                config=config,
                command=["bash", "-c", code]
            )
            
            # Execute with timeout
            result = await self._run_container_with_timeout(container, config.timeout_seconds)
            
            # Parse resource usage
            resource_usage = await self._get_container_stats(container)
            result.resource_usage = resource_usage
            
            # Cleanup
            await self._cleanup_container(container)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing bash code: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=0.0
            )
    
    def _prepare_python_script(
        self,
        code: str,
        config: SandboxConfig,
        dependencies: Optional[List[str]]
    ) -> str:
        """Prepare Python script with security and dependency handling."""
        script_parts = []
        
        # Install dependencies if specified
        if dependencies:
            for dep in dependencies:
                script_parts.append(f"import subprocess; subprocess.check_call(['pip', 'install', '{dep}'])")
        
        # Add security wrapper if strict policy
        if config.security_policy == SecurityPolicy.STRICT:
            script_parts.append("""
# Security wrapper - restrict dangerous functions
import builtins
original_import = builtins.__import__

def secure_import(name, *args, **kwargs):
    blocked = ['os', 'subprocess', 'sys', 'socket', 'urllib', 'requests']
    if name in blocked:
        raise ImportError(f'Import of {name} is blocked by security policy')
    return original_import(name, *args, **kwargs)

builtins.__import__ = secure_import

# Block dangerous builtins
for func in ['eval', 'exec', 'compile']:
    if hasattr(builtins, func):
        setattr(builtins, func, lambda *args, **kwargs: None)
""")
        
        # Add user code
        script_parts.append(code)
        
        return "\n".join(script_parts)
    
    def _prepare_javascript_script(
        self,
        code: str,
        config: SandboxConfig,
        dependencies: Optional[List[str]]
    ) -> str:
        """Prepare JavaScript script with security handling."""
        script_parts = []
        
        # Install dependencies if specified
        if dependencies:
            for dep in dependencies:
                script_parts.append(f"require('child_process').execSync('npm install {dep}');")
        
        # Add security restrictions for strict policy
        if config.security_policy == SecurityPolicy.STRICT:
            script_parts.append("""
// Security restrictions
delete require;
delete process;
delete global;
""")
        
        # Add user code
        script_parts.append(code)
        
        return "\n".join(script_parts)
    
    async def _create_container(
        self,
        image: str,
        name: str,
        config: SandboxConfig,
        command: List[str]
    ) -> Any:
        """Create Docker container with specified configuration."""
        # Ensure image is available
        try:
            self.docker_client.images.get(image)
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling image: {image}")
            self.docker_client.images.pull(image)
        
        # Container configuration
        container_config = {
            'image': image,
            'name': name,
            'command': command,
            'detach': True,
            'auto_remove': False,  # Don't auto-remove so we can get logs
            'mem_limit': f"{config.memory_limit_mb}m",
            'cpu_period': 100000,
            'cpu_quota': int(config.cpu_limit * 100000),
            'network_disabled': not config.network_access,
            'read_only': not config.filesystem_access,
        }
        
        # Add environment variables
        if config.environment_vars:
            container_config['environment'] = config.environment_vars
        
        # Security options
        container_config['security_opt'] = ['no-new-privileges:true']
        
        # Create and return container
        container = self.docker_client.containers.create(**container_config)
        return container
    
    async def _run_container_with_timeout(
        self,
        container: Any,
        timeout_seconds: int
    ) -> ExecutionResult:
        """Run container with timeout and capture output."""
        try:
            # Start container
            container.start()
            
            # Wait for completion with timeout
            try:
                exit_code = container.wait(timeout=timeout_seconds)
                if isinstance(exit_code, dict):
                    exit_code = exit_code.get('StatusCode', 0)
            except Exception:
                # Timeout occurred
                container.kill()
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timeout after {timeout_seconds} seconds",
                    execution_time=timeout_seconds,
                    exit_code=-1
                )
            
            # Get output
            try:
                logs = container.logs(stdout=True, stderr=True).decode('utf-8')
                
                # Separate stdout and stderr if possible
                output = logs
                error = ""
                
                # Simple heuristic to separate output and error
                if "Traceback" in logs or "Error:" in logs:
                    error = logs
                    output = ""
                else:
                    output = logs
                    
            except Exception as e:
                output = ""
                error = f"Failed to retrieve container logs: {e}"
            
            return ExecutionResult(
                success=(exit_code == 0),
                output=output.strip(),
                error=error.strip(),
                execution_time=0.0,  # Will be set by caller
                exit_code=exit_code
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=0.0,
                exit_code=-1
            )
    
    async def _get_container_stats(self, container: Any) -> Dict[str, Any]:
        """Get container resource usage statistics."""
        try:
            stats = container.stats(stream=False)
            
            # Extract useful metrics
            resource_usage = {}
            
            if 'memory' in stats:
                memory_stats = stats.get('memory', {})
                resource_usage['memory_usage_bytes'] = memory_stats.get('usage', 0)
                resource_usage['memory_limit_bytes'] = memory_stats.get('limit', 0)
                limit = memory_stats.get('limit', 1)
                resource_usage['memory_usage_percent'] = (
                    memory_stats.get('usage', 0) / limit * 100 if limit > 0 else 0
                )
            
            if 'cpu_stats' in stats and 'precpu_stats' in stats:
                cpu_stats = stats.get('cpu_stats', {})
                precpu_stats = stats.get('precpu_stats', {})
                
                # Calculate CPU usage percentage
                if cpu_stats and precpu_stats:
                    cpu_delta = cpu_stats.get('cpu_usage', {}).get('total_usage', 0) - precpu_stats.get('cpu_usage', {}).get('total_usage', 0)
                    system_cpu_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get('system_cpu_usage', 0)
                    number_cpus = len(cpu_stats.get('cpu_usage', {}).get('percpu_usage', [1]))
                    
                    if system_cpu_delta > 0 and cpu_delta > 0:
                        resource_usage['cpu_usage_percent'] = (cpu_delta / system_cpu_delta) * number_cpus * 100
                    else:
                        resource_usage['cpu_usage_percent'] = 0.0
                else:
                    resource_usage['cpu_usage_percent'] = 0.0
            
            return resource_usage
            
        except Exception as e:
            logger.warning(f"Failed to get container stats: {e}")
            return {}
    
    async def _cleanup_container(self, container: Any) -> None:
        """Clean up container resources."""
        try:
            # Stop container if still running
            container.reload()  # Refresh container status
            if container.status == 'running':
                container.kill()
            
            # Remove container manually since auto_remove is False
            container.remove()
            
        except Exception as e:
            logger.warning(f"Error during container cleanup: {e}")


# Integration with existing tool system
class SecurePythonExecutor:
    """Secure Python executor using LangChain Sandbox."""
    
    def __init__(self, security_policy: SecurityPolicy = SecurityPolicy.MODERATE):
        self.sandbox = LangChainSandbox()
        self.security_policy = security_policy
    
    async def execute(
        self,
        code: str,
        dependencies: Optional[List[str]] = None
    ) -> ExecutionResult:
        """Execute Python code securely."""
        return await self.sandbox.execute_python_code(
            code, dependencies, self.security_policy
        )


# Utility functions
def create_secure_sandbox(
    base_image: str = "python:3.11-slim"
) -> LangChainSandbox:
    """Create a secure sandbox instance."""
    return LangChainSandbox(base_image)


async def execute_code_safely(
    code: str,
    language: str = "python",
    dependencies: Optional[List[str]] = None,
    security_policy: SecurityPolicy = SecurityPolicy.MODERATE
) -> ExecutionResult:
    """Execute code safely in sandbox environment."""
    sandbox = LangChainSandbox()
    
    if language.lower() == "python":
        return await sandbox.execute_python_code(code, dependencies, security_policy)
    elif language.lower() == "javascript":
        config = SandboxConfig(
            sandbox_type=SandboxType.JAVASCRIPT,
            security_policy=security_policy
        )
        return await sandbox.execute_code(code, config, dependencies)
    elif language.lower() == "bash":
        return await sandbox.execute_bash_command(code, security_policy)
    else:
        return ExecutionResult(
            success=False,
            output="",
            error=f"Unsupported language: {language}",
            execution_time=0.0
        )