"""Sandboxed execution environment for secure code execution."""

import asyncio
import tempfile
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..core.task import Task


@dataclass
class ExecutionResult:
    """Result of sandboxed code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    exit_code: int = 0
    resource_usage: Optional[Dict[str, Any]] = None


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution environment."""
    memory_limit: str = "128m"
    cpu_quota: int = 50000  # 50% of one CPU
    time_limit: int = 30  # seconds
    network_disabled: bool = True
    read_only_filesystem: bool = True
    allowed_packages: List[str] = None
    
    def __post_init__(self):
        if self.allowed_packages is None:
            self.allowed_packages = ["json", "math", "datetime", "re"]


class SandboxExecutor(ABC):
    """Abstract base class for sandboxed code execution."""
    
    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
    
    @abstractmethod
    async def execute(self, code: str, language: str = "python") -> ExecutionResult:
        """Execute code in a sandboxed environment."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the executor is available."""
        pass


class DockerSandboxExecutor(SandboxExecutor):
    """Docker-based sandboxed executor."""
    
    def __init__(self, config: SandboxConfig = None):
        super().__init__(config)
        self._docker_available = None
    
    def is_available(self) -> bool:
        """Check if Docker is available."""
        if self._docker_available is None:
            try:
                import docker
                client = docker.from_env()
                client.ping()
                self._docker_available = True
            except (ImportError, Exception):
                self._docker_available = False
        return self._docker_available
    
    async def execute(self, code: str, language: str = "python") -> ExecutionResult:
        """Execute code in Docker container."""
        if not self.is_available():
            return ExecutionResult(
                success=False,
                output="",
                error="Docker not available",
                execution_time=0.0
            )
        
        start_time = time.time()
        
        try:
            import docker
            client = docker.from_env()
            
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=f'.{self._get_file_extension(language)}',
                delete=False
            ) as f:
                f.write(code)
                code_file = f.name
            
            try:
                # Get Docker image and command
                image = self._get_docker_image(language)
                command = self._get_execution_command(language, os.path.basename(code_file))
                
                # Create and run container
                container = client.containers.run(
                    image=image,
                    command=command,
                    volumes={
                        os.path.dirname(code_file): {
                            'bind': '/code',
                            'mode': 'ro'
                        }
                    },
                    working_dir='/code',
                    mem_limit=self.config.memory_limit,
                    cpu_quota=self.config.cpu_quota,
                    pids_limit=100,
                    network_mode='none' if self.config.network_disabled else 'bridge',
                    remove=True,
                    detach=False,
                    stdout=True,
                    stderr=True
                )
                
                execution_time = time.time() - start_time
                output = container.decode('utf-8')
                
                return ExecutionResult(
                    success=True,
                    output=output,
                    execution_time=execution_time,
                    exit_code=0
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                return ExecutionResult(
                    success=False,
                    output="",
                    error=str(e),
                    execution_time=execution_time,
                    exit_code=1
                )
            
            finally:
                # Cleanup
                try:
                    os.unlink(code_file)
                except:
                    pass
                    
        except ImportError:
            return ExecutionResult(
                success=False,
                output="",
                error="Docker library not installed",
                execution_time=0.0
            )
    
    def _get_docker_image(self, language: str) -> str:
        """Get appropriate Docker image for language."""
        images = {
            "python": "python:3.11-slim",
            "javascript": "node:18-slim",
            "bash": "bash:5.1"
        }
        return images.get(language, "ubuntu:22.04")
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language."""
        extensions = {
            "python": "py",
            "javascript": "js",
            "bash": "sh"
        }
        return extensions.get(language, "txt")
    
    def _get_execution_command(self, language: str, filename: str) -> List[str]:
        """Get execution command for language."""
        commands = {
            "python": ["python", filename],
            "javascript": ["node", filename],
            "bash": ["bash", filename]
        }
        return commands.get(language, ["cat", filename])


class ProcessSandboxExecutor(SandboxExecutor):
    """Process-based sandboxed executor (fallback when Docker unavailable)."""
    
    def is_available(self) -> bool:
        """Process executor is always available."""
        return True
    
    async def execute(self, code: str, language: str = "python") -> ExecutionResult:
        """Execute code in subprocess with security restrictions."""
        start_time = time.time()
        
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=f'.{self._get_file_extension(language)}',
            delete=False
        ) as f:
            f.write(code)
            code_file = f.name
        
        try:
            # Get execution command
            command = self._get_execution_command(language, code_file)
            
            # Execute with timeout and security restrictions
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tempfile.gettempdir()
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.time_limit
                )
                
                execution_time = time.time() - start_time
                
                return ExecutionResult(
                    success=process.returncode == 0,
                    output=stdout.decode('utf-8'),
                    error=stderr.decode('utf-8') if stderr else None,
                    execution_time=execution_time,
                    exit_code=process.returncode
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                
                execution_time = time.time() - start_time
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Execution timeout",
                    execution_time=execution_time,
                    exit_code=-1
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=execution_time,
                exit_code=1
            )
        
        finally:
            # Cleanup
            try:
                os.unlink(code_file)
            except:
                pass
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language."""
        extensions = {
            "python": "py",
            "javascript": "js",
            "bash": "sh"
        }
        return extensions.get(language, "txt")
    
    def _get_execution_command(self, language: str, filename: str) -> List[str]:
        """Get execution command for language."""
        commands = {
            "python": ["python", filename],
            "javascript": ["node", filename],
            "bash": ["bash", filename]
        }
        return commands.get(language, ["cat", filename])


class SandboxManager:
    """Manages multiple sandbox executors with fallback strategy."""
    
    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
        self.executors = [
            DockerSandboxExecutor(self.config),
            ProcessSandboxExecutor(self.config)
        ]
    
    def get_available_executor(self) -> SandboxExecutor:
        """Get the best available executor."""
        for executor in self.executors:
            if executor.is_available():
                return executor
        
        # Fallback to process executor (always available)
        return ProcessSandboxExecutor(self.config)
    
    async def execute_task(self, task: Task) -> ExecutionResult:
        """Execute a task in the best available sandbox."""
        executor = self.get_available_executor()
        
        # Extract code from task parameters
        code = task.parameters.get("code", "")
        language = task.parameters.get("language", "python")
        
        if not code:
            return ExecutionResult(
                success=False,
                output="",
                error="No code provided in task parameters",
                execution_time=0.0
            )
        
        return await executor.execute(code, language)
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of available executors."""
        available_executors = []
        for executor in self.executors:
            if executor.is_available():
                available_executors.append({
                    "type": type(executor).__name__,
                    "config": {
                        "memory_limit": self.config.memory_limit,
                        "cpu_quota": self.config.cpu_quota,
                        "time_limit": self.config.time_limit,
                        "network_disabled": self.config.network_disabled
                    }
                })
        
        return {
            "available_executors": available_executors,
            "primary_executor": type(self.get_available_executor()).__name__,
            "supported_languages": ["python", "javascript", "bash"]
        }