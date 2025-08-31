"""Automatic Setup System - Issue #312 Stream B

Comprehensive automatic setup mechanisms for tool installation and configuration:
- Platform-aware installation strategies (Windows, macOS, Linux)
- Configuration management and validation
- Integration with EnhancedToolRegistry for installation tracking
- Security considerations for safe tool installation
"""

import logging
import platform
import subprocess
import sys
import os
import asyncio
import json
import shutil
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime

from .registry import (
    EnhancedToolRegistry, InstallationRequirement, InstallationStatus,
    SecurityLevel, SecurityPolicy, get_enhanced_registry
)

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Supported platforms for installation."""
    WINDOWS = "windows"
    MACOS = "macos" 
    LINUX = "linux"
    UNKNOWN = "unknown"


class PackageManager(Enum):
    """Supported package managers."""
    PIP = "pip"
    CONDA = "conda"
    NPM = "npm"
    YARN = "yarn"
    APT = "apt"
    YUM = "yum"
    HOMEBREW = "homebrew"
    CHOCOLATEY = "chocolatey"
    WINGET = "winget"


@dataclass
class PlatformInfo:
    """Platform detection information."""
    platform: Platform
    version: str
    architecture: str
    available_managers: List[PackageManager]
    python_version: str
    node_version: Optional[str] = None
    
    def __post_init__(self):
        """Detect available package managers."""
        if not self.available_managers:
            self.available_managers = self._detect_package_managers()
    
    def _detect_package_managers(self) -> List[PackageManager]:
        """Detect which package managers are available on the system."""
        available = []
        
        # Always check for pip since we're in Python
        if shutil.which("pip") or shutil.which("pip3"):
            available.append(PackageManager.PIP)
        
        # Check for conda
        if shutil.which("conda"):
            available.append(PackageManager.CONDA)
        
        # Check Node.js package managers
        if shutil.which("npm"):
            available.append(PackageManager.NPM)
        if shutil.which("yarn"):
            available.append(PackageManager.YARN)
        
        # Platform-specific package managers
        if self.platform == Platform.LINUX:
            if shutil.which("apt-get") or shutil.which("apt"):
                available.append(PackageManager.APT)
            if shutil.which("yum"):
                available.append(PackageManager.YUM)
        elif self.platform == Platform.MACOS:
            if shutil.which("brew"):
                available.append(PackageManager.HOMEBREW)
        elif self.platform == Platform.WINDOWS:
            if shutil.which("choco"):
                available.append(PackageManager.CHOCOLATEY)
            if shutil.which("winget"):
                available.append(PackageManager.WINGET)
        
        return available


@dataclass
class InstallationResult:
    """Result of an installation operation."""
    success: bool
    package_name: str
    package_manager: str
    output: str = ""
    error: str = ""
    duration: float = 0.0
    post_install_success: bool = True
    dependencies_installed: List[str] = field(default_factory=list)


@dataclass
class SetupConfiguration:
    """Configuration for setup operations."""
    allow_system_packages: bool = False
    prefer_user_installs: bool = True
    use_virtual_environments: bool = True
    timeout_seconds: int = 300
    max_retries: int = 2
    verify_installations: bool = True
    create_backups: bool = True
    allowed_package_managers: Optional[Set[PackageManager]] = None
    security_level: SecurityLevel = SecurityLevel.MODERATE
    
    # Environment configuration
    environment_variables: Dict[str, str] = field(default_factory=dict)
    path_additions: List[str] = field(default_factory=list)
    
    # Validation configuration
    validate_checksums: bool = True
    require_signed_packages: bool = False
    allowed_sources: Optional[List[str]] = None


class PlatformDetector:
    """Detects platform information and capabilities."""
    
    @staticmethod
    def detect_platform() -> PlatformInfo:
        """Detect current platform information."""
        system = platform.system().lower()
        
        if system == "windows":
            platform_type = Platform.WINDOWS
        elif system == "darwin":
            platform_type = Platform.MACOS
        elif system == "linux":
            platform_type = Platform.LINUX
        else:
            platform_type = Platform.UNKNOWN
        
        # Get version information
        version = platform.version()
        architecture = platform.machine()
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Detect Node.js version if available
        node_version = None
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                node_version = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return PlatformInfo(
            platform=platform_type,
            version=version,
            architecture=architecture,
            available_managers=[],  # Will be populated by __post_init__
            python_version=python_version,
            node_version=node_version
        )
    
    @staticmethod
    def check_administrator_privileges() -> bool:
        """Check if running with administrator privileges."""
        try:
            if platform.system() == "Windows":
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin()
            else:
                return os.geteuid() == 0
        except:
            return False
    
    @staticmethod
    def check_virtual_environment() -> Tuple[bool, Optional[str]]:
        """Check if running in a virtual environment."""
        # Check for conda environment
        if 'CONDA_DEFAULT_ENV' in os.environ:
            return True, os.environ['CONDA_DEFAULT_ENV']
        
        # Check for venv/virtualenv
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            venv_name = os.path.basename(sys.prefix)
            return True, venv_name
        
        return False, None


class ConfigurationManager:
    """Manages setup configurations and validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".orchestrator" / "setup_config.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = SetupConfiguration()
        
        # Load existing configuration if available
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update configuration with loaded values
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        if key == "security_level":
                            setattr(self.config, key, SecurityLevel(value))
                        elif key == "allowed_package_managers" and value:
                            setattr(self.config, key, {PackageManager(pm) for pm in value})
                        else:
                            setattr(self.config, key, value)
                
                logger.info(f"Configuration loaded from {self.config_path}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
        
        return False
    
    def save_config(self) -> bool:
        """Save current configuration to file."""
        try:
            config_dict = {}
            for key, value in self.config.__dict__.items():
                if isinstance(value, SecurityLevel):
                    config_dict[key] = value.value
                elif isinstance(value, set) and value:
                    config_dict[key] = [item.value for item in value if hasattr(item, 'value')]
                elif not callable(value):
                    config_dict[key] = value
            
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate current configuration."""
        issues = []
        
        # Check timeout values
        if self.config.timeout_seconds <= 0:
            issues.append("Timeout must be positive")
        
        if self.config.max_retries < 0:
            issues.append("Max retries cannot be negative")
        
        # Check security settings
        if self.config.security_level == SecurityLevel.STRICT and self.config.allow_system_packages:
            issues.append("Strict security level should not allow system packages")
        
        # Check virtual environment settings
        in_venv, _ = PlatformDetector.check_virtual_environment()
        if self.config.use_virtual_environments and not in_venv:
            issues.append("Virtual environment usage is enabled but not currently in one")
        
        return len(issues) == 0, issues
    
    def update_config(self, **kwargs) -> bool:
        """Update configuration with new values."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            # Validate and save
            is_valid, issues = self.validate_config()
            if not is_valid:
                logger.error(f"Configuration validation failed: {issues}")
                return False
            
            return self.save_config()
        
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False


class SetupSystem:
    """Main setup system for automatic tool installation and configuration."""
    
    def __init__(self, 
                 registry: Optional[EnhancedToolRegistry] = None,
                 config: Optional[SetupConfiguration] = None):
        self.registry = registry or get_enhanced_registry()
        self.config_manager = ConfigurationManager()
        if config:
            self.config_manager.config = config
        
        # Platform information
        self.platform_info = PlatformDetector.detect_platform()
        
        # Security policy
        self.security_policy = self._create_security_policy()
        
        # Installation tracking
        self.installation_history: List[InstallationResult] = []
        self.pending_installations: Set[str] = set()
        
        logger.info(f"Setup system initialized for {self.platform_info.platform.value}")
        logger.info(f"Available package managers: {[pm.value for pm in self.platform_info.available_managers]}")
    
    def _create_security_policy(self) -> SecurityPolicy:
        """Create security policy based on configuration."""
        config = self.config_manager.config
        
        # Base policy from security level
        if config.security_level == SecurityLevel.STRICT:
            allowed_ops = ["install_user", "verify"]
            blocked_ops = ["install_system", "modify_path", "create_symlinks"]
        elif config.security_level == SecurityLevel.MODERATE:
            allowed_ops = ["install_user", "install_system", "verify", "modify_path"]
            blocked_ops = ["create_symlinks", "modify_registry"]
        else:
            allowed_ops = ["*"]
            blocked_ops = []
        
        return SecurityPolicy(
            level=config.security_level,
            allowed_operations=allowed_ops,
            blocked_operations=blocked_ops,
            sandboxed=config.security_level == SecurityLevel.STRICT,
            network_access=True,
            file_system_access=not config.security_level == SecurityLevel.STRICT,
            max_execution_time=config.timeout_seconds,
            environment_variables=config.environment_variables
        )
    
    async def setup_tool(self, tool_name: str) -> InstallationResult:
        """Setup a tool by installing its requirements."""
        logger.info(f"Setting up tool: {tool_name}")
        
        # Check if tool exists in registry
        if tool_name not in self.registry.enhanced_metadata:
            return InstallationResult(
                success=False,
                package_name=tool_name,
                package_manager="unknown",
                error="Tool not found in registry"
            )
        
        metadata = self.registry.enhanced_metadata[tool_name]
        
        # Check current installation status
        current_status = self.registry.installation_tracker.get(tool_name, InstallationStatus.NEEDS_INSTALL)
        if current_status == InstallationStatus.AVAILABLE:
            logger.info(f"Tool {tool_name} is already available")
            return InstallationResult(
                success=True,
                package_name=tool_name,
                package_manager="none",
                output="Tool already available"
            )
        
        # Mark as installing
        self.registry.installation_tracker[tool_name] = InstallationStatus.INSTALLING
        self.pending_installations.add(tool_name)
        
        try:
            # Install requirements
            for requirement in metadata.installation_requirements:
                result = await self._install_requirement(requirement)
                
                if not result.success:
                    # Installation failed
                    self.registry.installation_tracker[tool_name] = InstallationStatus.FAILED
                    metadata.installation_status = InstallationStatus.FAILED
                    self.pending_installations.discard(tool_name)
                    return result
            
            # All requirements installed successfully
            self.registry.installation_tracker[tool_name] = InstallationStatus.AVAILABLE
            metadata.installation_status = InstallationStatus.AVAILABLE
            self.pending_installations.discard(tool_name)
            
            # Notify callbacks
            if tool_name in self.registry.installation_callbacks:
                for callback in self.registry.installation_callbacks[tool_name]:
                    try:
                        callback(tool_name, InstallationStatus.AVAILABLE)
                    except Exception as e:
                        logger.error(f"Installation callback failed: {e}")
            
            return InstallationResult(
                success=True,
                package_name=tool_name,
                package_manager="multiple",
                output=f"Successfully installed all requirements for {tool_name}"
            )
        
        except Exception as e:
            logger.error(f"Setup failed for {tool_name}: {e}")
            self.registry.installation_tracker[tool_name] = InstallationStatus.FAILED
            metadata.installation_status = InstallationStatus.FAILED
            self.pending_installations.discard(tool_name)
            
            return InstallationResult(
                success=False,
                package_name=tool_name,
                package_manager="unknown",
                error=str(e)
            )
    
    async def _install_requirement(self, requirement: InstallationRequirement) -> InstallationResult:
        """Install a specific requirement."""
        logger.info(f"Installing requirement: {requirement.package_name} via {requirement.package_manager}")
        
        # Import the installers module
        from .installers import PackageInstallerFactory
        
        try:
            # Get appropriate installer
            installer = PackageInstallerFactory.get_installer(
                PackageManager(requirement.package_manager),
                self.platform_info,
                self.config_manager.config
            )
            
            if not installer:
                return InstallationResult(
                    success=False,
                    package_name=requirement.package_name,
                    package_manager=requirement.package_manager,
                    error=f"No installer available for {requirement.package_manager}"
                )
            
            # Perform installation
            result = await installer.install(
                requirement.package_name,
                requirement.version_spec,
                requirement.install_command
            )
            
            # Run post-install command if provided and installation succeeded
            if result.success and requirement.post_install_command:
                post_result = await self._run_post_install_command(
                    requirement.post_install_command,
                    requirement.environment_setup
                )
                result.post_install_success = post_result
            
            # Track installation
            self.installation_history.append(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to install requirement {requirement.package_name}: {e}")
            return InstallationResult(
                success=False,
                package_name=requirement.package_name,
                package_manager=requirement.package_manager,
                error=str(e)
            )
    
    async def _run_post_install_command(self, command: str, env_setup: Dict[str, Any]) -> bool:
        """Run post-installation command."""
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(env_setup)
            env.update(self.config_manager.config.environment_variables)
            
            # Run command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config_manager.config.timeout_seconds
            )
            
            if process.returncode == 0:
                logger.info(f"Post-install command succeeded: {command}")
                return True
            else:
                logger.error(f"Post-install command failed: {command}, stderr: {stderr.decode()}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to run post-install command: {e}")
            return False
    
    async def setup_multiple_tools(self, tool_names: List[str]) -> Dict[str, InstallationResult]:
        """Setup multiple tools concurrently."""
        logger.info(f"Setting up {len(tool_names)} tools")
        
        # Create tasks for concurrent installation
        tasks = [self.setup_tool(tool_name) for tool_name in tool_names]
        
        # Execute all installations
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        setup_results = {}
        for i, result in enumerate(results):
            tool_name = tool_names[i]
            if isinstance(result, Exception):
                setup_results[tool_name] = InstallationResult(
                    success=False,
                    package_name=tool_name,
                    package_manager="unknown",
                    error=str(result)
                )
            else:
                setup_results[tool_name] = result
        
        return setup_results
    
    def check_tool_availability(self, tool_name: str) -> Tuple[bool, Optional[str]]:
        """Check if a tool is available and ready to use."""
        if tool_name not in self.registry.enhanced_metadata:
            return False, "Tool not found in registry"
        
        metadata = self.registry.enhanced_metadata[tool_name]
        status = self.registry.installation_tracker.get(tool_name, InstallationStatus.NEEDS_INSTALL)
        
        if status == InstallationStatus.AVAILABLE:
            return True, None
        elif status == InstallationStatus.INSTALLING:
            return False, "Tool is currently being installed"
        elif status == InstallationStatus.FAILED:
            return False, "Tool installation failed"
        else:
            return False, "Tool needs installation"
    
    def get_installation_status(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed installation status for a tool."""
        if tool_name not in self.registry.enhanced_metadata:
            return {"error": "Tool not found"}
        
        metadata = self.registry.enhanced_metadata[tool_name]
        status = self.registry.installation_tracker.get(tool_name, InstallationStatus.NEEDS_INSTALL)
        
        # Find installation history for this tool
        history = [
            result for result in self.installation_history
            if result.package_name == tool_name
        ]
        
        return {
            "tool_name": tool_name,
            "current_status": status.value,
            "metadata": {
                "version": str(metadata.version_info),
                "requirements_count": len(metadata.installation_requirements),
                "security_level": metadata.security_policy.level.value if metadata.security_policy else "unknown"
            },
            "installation_history": [
                {
                    "success": result.success,
                    "package_manager": result.package_manager,
                    "duration": result.duration,
                    "error": result.error
                } for result in history
            ],
            "is_pending": tool_name in self.pending_installations
        }
    
    def cleanup_failed_installations(self) -> List[str]:
        """Clean up failed installations and reset their status."""
        cleaned = []
        
        for tool_name, status in list(self.registry.installation_tracker.items()):
            if status == InstallationStatus.FAILED:
                # Reset status
                self.registry.installation_tracker[tool_name] = InstallationStatus.NEEDS_INSTALL
                if tool_name in self.registry.enhanced_metadata:
                    self.registry.enhanced_metadata[tool_name].installation_status = InstallationStatus.NEEDS_INSTALL
                
                cleaned.append(tool_name)
                logger.info(f"Reset failed installation status for {tool_name}")
        
        return cleaned
    
    def get_setup_report(self) -> Dict[str, Any]:
        """Generate comprehensive setup system report."""
        total_tools = len(self.registry.enhanced_metadata)
        available_tools = sum(1 for status in self.registry.installation_tracker.values() 
                             if status == InstallationStatus.AVAILABLE)
        failed_tools = sum(1 for status in self.registry.installation_tracker.values() 
                          if status == InstallationStatus.FAILED)
        
        successful_installs = sum(1 for result in self.installation_history if result.success)
        failed_installs = len(self.installation_history) - successful_installs
        
        return {
            "platform_info": {
                "platform": self.platform_info.platform.value,
                "python_version": self.platform_info.python_version,
                "node_version": self.platform_info.node_version,
                "available_managers": [pm.value for pm in self.platform_info.available_managers]
            },
            "tool_statistics": {
                "total_tools": total_tools,
                "available_tools": available_tools,
                "failed_tools": failed_tools,
                "pending_tools": len(self.pending_installations),
                "success_rate": (available_tools / total_tools * 100) if total_tools > 0 else 0
            },
            "installation_statistics": {
                "total_attempts": len(self.installation_history),
                "successful_installs": successful_installs,
                "failed_installs": failed_installs,
                "install_success_rate": (successful_installs / len(self.installation_history) * 100) if self.installation_history else 0
            },
            "configuration": {
                "security_level": self.config_manager.config.security_level.value,
                "use_virtual_environments": self.config_manager.config.use_virtual_environments,
                "timeout_seconds": self.config_manager.config.timeout_seconds
            },
            "timestamp": datetime.now().isoformat()
        }


# Global setup system instance
setup_system = SetupSystem()


def get_setup_system() -> SetupSystem:
    """Get the global setup system instance."""
    return setup_system


# Convenience functions
async def setup_tool(tool_name: str) -> InstallationResult:
    """Convenience function to setup a single tool."""
    return await setup_system.setup_tool(tool_name)


async def setup_tools(tool_names: List[str]) -> Dict[str, InstallationResult]:
    """Convenience function to setup multiple tools."""
    return await setup_system.setup_multiple_tools(tool_names)


def check_tool_availability(tool_name: str) -> Tuple[bool, Optional[str]]:
    """Convenience function to check tool availability."""
    return setup_system.check_tool_availability(tool_name)


__all__ = [
    "SetupSystem",
    "PlatformDetector", 
    "ConfigurationManager",
    "SetupConfiguration",
    "PlatformInfo",
    "InstallationResult",
    "Platform",
    "PackageManager",
    "setup_system",
    "get_setup_system",
    "setup_tool",
    "setup_tools", 
    "check_tool_availability"
]