"""Multi-Language Sandbox Support - Issue #206 Task 2.2

Enhanced multi-language code execution system providing secure sandboxed environments
for Python, Node.js, Bash, Java, Go, and other programming languages.
"""

import logging
import asyncio
import json
import tempfile
import os
import base64
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .base import Tool
from .secure_tool_executor import (
    SecureToolExecutor,
    ExecutionMode,
    ExecutionEnvironment,
    ExecutionResult
)
from ..security.docker_manager import (
    EnhancedDockerManager,
    SecureContainer,
    ResourceLimits,
    SecurityConfig
)
from ..security.dependency_manager import PackageInfo, PackageEcosystem

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    NODEJS = "nodejs"
    BASH = "bash"
    SHELL = "shell"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    RUBY = "ruby"
    PHP = "php"
    R = "r"


@dataclass
class LanguageConfig:
    """Configuration for a specific programming language."""
    language: Language
    docker_image: str
    file_extension: str
    compile_command: Optional[str] = None
    run_command: str = ""
    package_manager: Optional[str] = None
    dependency_file: Optional[str] = None
    default_memory_mb: int = 256
    default_timeout: int = 60
    supports_repl: bool = False
    ecosystem: Optional[PackageEcosystem] = None
    
    def get_execution_command(self, filename: str, compiled_output: Optional[str] = None) -> str:
        """Generate the execution command for this language."""
        if compiled_output:
            return compiled_output
        return self.run_command.format(filename=filename)


# Language configurations
LANGUAGE_CONFIGS = {
    Language.PYTHON: LanguageConfig(
        language=Language.PYTHON,
        docker_image="python:3.11-slim",
        file_extension=".py",
        run_command="python3 {filename}",
        package_manager="pip",
        dependency_file="requirements.txt",
        default_memory_mb=256,
        default_timeout=60,
        supports_repl=True,
        ecosystem=PackageEcosystem.PYPI
    ),
    
    Language.NODEJS: LanguageConfig(
        language=Language.NODEJS,
        docker_image="node:18-slim",
        file_extension=".js",
        run_command="node {filename}",
        package_manager="npm",
        dependency_file="package.json",
        default_memory_mb=512,
        default_timeout=60,
        supports_repl=True,
        ecosystem=PackageEcosystem.NPM
    ),
    
    Language.JAVASCRIPT: LanguageConfig(
        language=Language.JAVASCRIPT,
        docker_image="node:18-slim",
        file_extension=".js",
        run_command="node {filename}",
        package_manager="npm",
        dependency_file="package.json",
        default_memory_mb=512,
        default_timeout=60,
        supports_repl=True,
        ecosystem=PackageEcosystem.NPM
    ),
    
    Language.BASH: LanguageConfig(
        language=Language.BASH,
        docker_image="ubuntu:22.04",
        file_extension=".sh",
        run_command="bash {filename}",
        default_memory_mb=128,
        default_timeout=30,
        supports_repl=True
    ),
    
    Language.SHELL: LanguageConfig(
        language=Language.SHELL,
        docker_image="ubuntu:22.04",
        file_extension=".sh",
        run_command="bash {filename}",
        default_memory_mb=128,
        default_timeout=30,
        supports_repl=True
    ),
    
    Language.JAVA: LanguageConfig(
        language=Language.JAVA,
        docker_image="openjdk:17-slim",
        file_extension=".java",
        compile_command="javac {filename}",
        run_command="java {classname}",
        package_manager="mvn",
        dependency_file="pom.xml",
        default_memory_mb=512,
        default_timeout=120,
        supports_repl=False,
        ecosystem=PackageEcosystem.UNKNOWN  # Java/Maven not directly supported yet
    ),
    
    Language.GO: LanguageConfig(
        language=Language.GO,
        docker_image="golang:1.21-alpine",
        file_extension=".go",
        run_command="go run {filename}",
        package_manager="go",
        dependency_file="go.mod",
        default_memory_mb=256,
        default_timeout=60,
        supports_repl=False
    ),
    
    Language.RUST: LanguageConfig(
        language=Language.RUST,
        docker_image="rust:1.75-slim",
        file_extension=".rs",
        compile_command="rustc {filename} -o {output}",
        run_command="./{output}",
        package_manager="cargo",
        dependency_file="Cargo.toml",
        default_memory_mb=512,
        default_timeout=180,  # Rust compilation can be slow
        supports_repl=False
    ),
    
    Language.CPP: LanguageConfig(
        language=Language.CPP,
        docker_image="gcc:12-slim",
        file_extension=".cpp",
        compile_command="g++ -o {output} {filename} -std=c++17",
        run_command="./{output}",
        default_memory_mb=256,
        default_timeout=60,
        supports_repl=False
    ),
    
    Language.C: LanguageConfig(
        language=Language.C,
        docker_image="gcc:12-slim",
        file_extension=".c",
        compile_command="gcc -o {output} {filename}",
        run_command="./{output}",
        default_memory_mb=128,
        default_timeout=60,
        supports_repl=False
    ),
    
    Language.RUBY: LanguageConfig(
        language=Language.RUBY,
        docker_image="ruby:3.2-alpine",
        file_extension=".rb",
        run_command="ruby {filename}",
        package_manager="gem",
        dependency_file="Gemfile",
        default_memory_mb=256,
        default_timeout=60,
        supports_repl=True
    ),
    
    Language.PHP: LanguageConfig(
        language=Language.PHP,
        docker_image="php:8.2-cli-alpine",
        file_extension=".php",
        run_command="php {filename}",
        package_manager="composer",
        dependency_file="composer.json",
        default_memory_mb=256,
        default_timeout=60,
        supports_repl=False
    ),
    
    Language.R: LanguageConfig(
        language=Language.R,
        docker_image="r-base:4.3.2",
        file_extension=".R",
        run_command="Rscript {filename}",
        default_memory_mb=512,
        default_timeout=120,
        supports_repl=True
    )
}


class MultiLanguageExecutor:
    """
    Multi-language code executor that provides secure sandboxed environments
    for various programming languages using Docker containers.
    """
    
    def __init__(self, docker_manager: EnhancedDockerManager):
        self.docker_manager = docker_manager
        self.supported_languages = set(LANGUAGE_CONFIGS.keys())
        self.active_containers: Dict[str, SecureContainer] = {}
        
        logger.info(f"MultiLanguageExecutor initialized with {len(self.supported_languages)} languages")
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return [lang.value for lang in self.supported_languages]
    
    def detect_language(self, code: str, filename: Optional[str] = None) -> Optional[Language]:
        """
        Detect programming language from code content or filename.
        
        Args:
            code: Source code content
            filename: Optional filename with extension
            
        Returns:
            Detected Language enum or None if cannot detect
        """
        
        # First try to detect from filename extension
        if filename:
            extension = Path(filename).suffix.lower()
            extension_map = {
                '.py': Language.PYTHON,
                '.js': Language.NODEJS,
                '.sh': Language.BASH,
                '.bash': Language.BASH,
                '.java': Language.JAVA,
                '.go': Language.GO,
                '.rs': Language.RUST,
                '.cpp': Language.CPP,
                '.cxx': Language.CPP,
                '.cc': Language.CPP,
                '.c': Language.C,
                '.rb': Language.RUBY,
                '.php': Language.PHP,
                '.r': Language.R,
                '.R': Language.R
            }
            
            if extension in extension_map:
                return extension_map[extension]
        
        # Try to detect from code content patterns
        code_lower = code.lower().strip()
        
        # Python indicators
        if any(pattern in code for pattern in ['import ', 'def ', 'print(', 'if __name__']):
            return Language.PYTHON
        
        # JavaScript/Node.js indicators
        if any(pattern in code for pattern in ['console.log', 'require(', 'function ', 'const ', 'let ']):
            return Language.NODEJS
        
        # Shell/Bash indicators
        if any(pattern in code_lower for pattern in ['#!/bin/bash', '#!/bin/sh', 'echo ', 'if [', 'for ']):
            return Language.BASH
        
        # Java indicators
        if any(pattern in code for pattern in ['public class', 'public static void main', 'System.out']):
            return Language.JAVA
        
        # Go indicators
        if any(pattern in code for pattern in ['package main', 'func main()', 'fmt.Print']):
            return Language.GO
        
        # Rust indicators
        if any(pattern in code for pattern in ['fn main()', 'println!', 'use std::']):
            return Language.RUST
        
        # C++ indicators
        if any(pattern in code for pattern in ['#include <iostream>', 'std::cout', 'using namespace std']):
            return Language.CPP
        
        # C indicators
        if any(pattern in code for pattern in ['#include <stdio.h>', 'printf(', 'int main(']):
            return Language.C
        
        # Ruby indicators
        if any(pattern in code for pattern in ['puts ', 'def ', 'require ', 'class ']):
            return Language.RUBY
        
        # PHP indicators
        if code.startswith('<?php') or '<?php' in code:
            return Language.PHP
        
        # R indicators
        if any(pattern in code for pattern in ['library(', '<-', 'print(', 'c(']):
            return Language.R
        
        return None
    
    async def execute_code(
        self,
        code: str,
        language: Optional[Union[str, Language]] = None,
        filename: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        resource_limits: Optional[ResourceLimits] = None,
        security_config: Optional[SecurityConfig] = None,
        timeout: Optional[int] = None,
        mode: ExecutionMode = ExecutionMode.SANDBOXED
    ) -> ExecutionResult:
        """
        Execute code in the specified language with full security and monitoring.
        
        Args:
            code: Source code to execute
            language: Programming language (auto-detected if not specified)
            filename: Optional filename for the code
            dependencies: List of dependencies to install
            resource_limits: Custom resource limits
            security_config: Custom security configuration
            timeout: Execution timeout in seconds
            mode: Execution mode (sandboxed, isolated, etc.)
            
        Returns:
            ExecutionResult with comprehensive execution information
        """
        
        try:
            # Determine language
            if isinstance(language, str):
                try:
                    detected_lang = Language(language.lower())
                except ValueError:
                    detected_lang = self.detect_language(code, filename)
            elif isinstance(language, Language):
                detected_lang = language
            else:
                detected_lang = self.detect_language(code, filename)
            
            if not detected_lang:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error="Could not detect programming language"
                )
            
            if detected_lang not in self.supported_languages:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Language {detected_lang.value} is not supported"
                )
            
            config = LANGUAGE_CONFIGS[detected_lang]
            logger.info(f"Executing {detected_lang.value} code")
            
            # Create execution environment
            result = await self._execute_in_language_environment(
                code=code,
                config=config,
                filename=filename,
                dependencies=dependencies or [],
                resource_limits=resource_limits,
                security_config=security_config,
                timeout=timeout,
                mode=mode
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-language execution failed: {e}")
            return ExecutionResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def _execute_in_language_environment(
        self,
        code: str,
        config: LanguageConfig,
        filename: Optional[str],
        dependencies: List[str],
        resource_limits: Optional[ResourceLimits],
        security_config: Optional[SecurityConfig],
        timeout: Optional[int],
        mode: ExecutionMode
    ) -> ExecutionResult:
        """Execute code in a language-specific container environment."""
        
        # Determine resource limits
        if not resource_limits:
            resource_limits = ResourceLimits(
                memory_mb=config.default_memory_mb,
                cpu_cores=1.0,
                execution_timeout=timeout or config.default_timeout,
                pids_limit=50
            )
        
        # Use default security config if not provided
        if not security_config:
            security_config = SecurityConfig()
        
        # Create container for the specific language
        container = await self.docker_manager.create_secure_container(
            image=config.docker_image,
            name=f"multilang_{config.language.value}_{os.getpid()}",
            resource_limits=resource_limits,
            security_config=security_config
        )
        
        try:
            # Install dependencies if needed
            if dependencies:
                await self._install_language_dependencies(
                    container, config, dependencies
                )
            
            # Prepare and execute code
            execution_result = await self._prepare_and_execute_code(
                container, code, config, filename, timeout or config.default_timeout
            )
            
            return execution_result
            
        finally:
            # Return container to pool or destroy
            execution_successful = execution_result.success if 'execution_result' in locals() else False
            await self.docker_manager.return_container_to_pool(
                container=container, 
                execution_time=getattr(execution_result, 'performance_metrics', {}).get('execution_time', 0.0) if 'execution_result' in locals() else 0.0,
                execution_successful=execution_successful
            )
    
    async def _install_language_dependencies(
        self,
        container: SecureContainer,
        config: LanguageConfig,
        dependencies: List[str]
    ) -> None:
        """Install language-specific dependencies in the container."""
        
        if not config.package_manager or not dependencies:
            return
        
        try:
            dependency_commands = {
                "pip": f"pip install {' '.join(dependencies)}",
                "npm": f"npm install {' '.join(dependencies)}",
                "mvn": "# Maven dependencies require pom.xml setup",
                "go": "# Go modules handled automatically",
                "cargo": "# Cargo dependencies require Cargo.toml setup",
                "gem": f"gem install {' '.join(dependencies)}",
                "composer": "# Composer dependencies require composer.json setup"
            }
            
            install_cmd = dependency_commands.get(config.package_manager)
            if install_cmd and not install_cmd.startswith("#"):
                result = await self.docker_manager.execute_in_container(
                    container,
                    install_cmd,
                    timeout=300  # 5 minutes for dependency installation
                )
                
                if not result['success']:
                    logger.warning(f"Failed to install dependencies: {result.get('error')}")
                else:
                    logger.info(f"Successfully installed {len(dependencies)} dependencies")
            
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
    
    async def _prepare_and_execute_code(
        self,
        container: SecureContainer,
        code: str,
        config: LanguageConfig,
        filename: Optional[str],
        timeout: int
    ) -> ExecutionResult:
        """Prepare code file and execute in the container."""
        
        try:
            # Determine filename and ensure it's in /tmp (writable directory)
            if not filename:
                filename = f"/tmp/code{config.file_extension}"
            elif not filename.endswith(config.file_extension):
                filename += config.file_extension
            
            # Ensure filename is in /tmp for writable access
            if not filename.startswith('/tmp/'):
                filename = f"/tmp/{filename}"
            
            # Write code to container using base64 encoding for reliability
            import base64
            
            # Encode the code as base64 to avoid shell escaping issues
            code_b64 = base64.b64encode(code.encode('utf-8')).decode('ascii')
            
            if config.language == Language.PYTHON:
                # Use Python to write the file directly - most reliable approach
                python_write_cmd = f"python3 -c \"import sys, base64; open('{filename}', 'w', encoding='utf-8').write(base64.b64decode('{code_b64}').decode('utf-8'))\""
                write_result = await self.docker_manager.execute_in_container(
                    container,
                    python_write_cmd,
                    timeout=30
                )
            else:
                # For other languages, use base64 decoding to avoid heredoc issues
                write_result = await self.docker_manager.execute_in_container(
                    container,
                    f"echo '{code_b64}' | base64 -d > {filename}",
                    timeout=30
                )
            
            if not write_result['success']:
                return ExecutionResult(
                    success=False,
                    output=None,
                    error=f"Failed to write code file: {write_result.get('error')}"
                )
            
            # Compile if needed
            if config.compile_command:
                compiled_output = filename.rsplit('.', 1)[0]  # Remove extension
                # For Java, classname should be just the filename without path
                classname = filename.split('/')[-1].rsplit('.', 1)[0] if config.language == Language.JAVA else ""
                compile_cmd = config.compile_command.format(
                    filename=filename,
                    output=compiled_output,
                    classname=classname
                )
                
                compile_result = await self.docker_manager.execute_in_container(
                    container,
                    compile_cmd,
                    timeout=min(timeout, 180)  # Max 3 minutes for compilation
                )
                
                if not compile_result['success']:
                    return ExecutionResult(
                        success=False,
                        output=compile_result.get('output', ''),
                        error=f"Compilation failed: {compile_result.get('error')}"
                    )
                
                # Update run command for compiled languages
                run_cmd = config.get_execution_command(filename, compiled_output)
            else:
                run_cmd = config.get_execution_command(filename)
            
            # Execute the code
            execution_result = await self.docker_manager.execute_in_container(
                container,
                run_cmd,
                timeout=timeout
            )
            
            return ExecutionResult(
                success=execution_result['success'],
                output=execution_result.get('output', ''),
                error=execution_result.get('error'),
                resource_usage=execution_result.get('resource_usage', {}),
                performance_metrics={
                    'language': config.language.value,
                    'execution_time': execution_result.get('execution_time', 0),
                    'compiled': config.compile_command is not None,
                    'container_id': container.container_id
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Code execution failed: {e}"
            )


class MultiLanguageExecutorTool(Tool):
    """
    Tool wrapper for multi-language code execution with comprehensive
    language support and security features.
    """
    
    def __init__(self):
        super().__init__(
            name="multi-language-executor",
            description="Execute code in various programming languages with secure sandboxing"
        )
        
        # Required parameters
        self.add_parameter("code", "string", "Source code to execute")
        
        # Optional parameters
        self.add_parameter(
            "language", "string", "Programming language (python, javascript, java, go, etc.)",
            required=False, default="auto"
        )
        self.add_parameter(
            "filename", "string", "Optional filename for the code",
            required=False, default=""
        )
        self.add_parameter(
            "dependencies", "array", "List of dependencies/packages to install",
            required=False, default=[]
        )
        self.add_parameter(
            "timeout", "integer", "Execution timeout in seconds",
            required=False, default=60
        )
        self.add_parameter(
            "memory_limit_mb", "integer", "Memory limit in MB",
            required=False, default=256
        )
        self.add_parameter(
            "mode", "string", "Execution mode: sandboxed, isolated, trusted",
            required=False, default="sandboxed"
        )
        
        self.docker_manager: Optional[EnhancedDockerManager] = None
        self.multi_lang_executor: Optional[MultiLanguageExecutor] = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure the multi-language executor is initialized."""
        if not self._initialized:
            self.docker_manager = EnhancedDockerManager()
            await self.docker_manager.start_background_tasks()
            
            self.multi_lang_executor = MultiLanguageExecutor(self.docker_manager)
            self._initialized = True
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute code in the specified programming language."""
        
        await self._ensure_initialized()
        
        # Extract parameters
        code = kwargs.get("code", "")
        language = kwargs.get("language", "auto")
        filename = kwargs.get("filename", "")
        dependencies = kwargs.get("dependencies", [])
        timeout = kwargs.get("timeout", 60)
        memory_limit_mb = kwargs.get("memory_limit_mb", 256)
        mode_str = kwargs.get("mode", "sandboxed")
        
        if not code:
            return {
                "success": False,
                "error": "No code provided for execution",
                "supported_languages": self.multi_lang_executor.get_supported_languages()
            }
        
        try:
            # Parse execution mode
            mode_map = {
                "sandboxed": ExecutionMode.SANDBOXED,
                "isolated": ExecutionMode.ISOLATED,
                "trusted": ExecutionMode.TRUSTED
            }
            mode = mode_map.get(mode_str.lower(), ExecutionMode.SANDBOXED)
            
            # Create resource limits
            resource_limits = ResourceLimits(
                memory_mb=memory_limit_mb,
                cpu_cores=1.0,
                execution_timeout=timeout,
                pids_limit=50
            )
            
            # Execute code
            result = await self.multi_lang_executor.execute_code(
                code=code,
                language=language if language != "auto" else None,
                filename=filename if filename else None,
                dependencies=dependencies,
                resource_limits=resource_limits,
                timeout=timeout,
                mode=mode
            )
            
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "performance_metrics": result.performance_metrics,
                "resource_usage": result.resource_usage,
                "supported_languages": self.multi_lang_executor.get_supported_languages()
            }
            
        except Exception as e:
            logger.error(f"Multi-language tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "supported_languages": self.multi_lang_executor.get_supported_languages()
            }
    
    async def shutdown(self):
        """Shutdown the multi-language executor."""
        if self.docker_manager:
            await self.docker_manager.shutdown()
            self._initialized = False


# Export classes
__all__ = [
    'MultiLanguageExecutor',
    'MultiLanguageExecutorTool',
    'Language',
    'LanguageConfig',
    'LANGUAGE_CONFIGS'
]