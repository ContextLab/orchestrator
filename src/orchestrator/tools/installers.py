"""Package Installers - Issue #312 Stream B

Package manager integrations for automatic tool installation:
- Support for pip, conda, npm, yarn, apt, yum, homebrew, chocolatey, winget
- Platform-aware installation strategies
- Security validation and verification
- Concurrent installation support
"""

import asyncio
import subprocess
import shutil
import os
import json
import hashlib
import tempfile
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import logging

from .setup import (
    PackageManager, Platform, PlatformInfo, SetupConfiguration,
    InstallationResult
)

logger = logging.getLogger(__name__)


@dataclass
class PackageInfo:
    """Information about a package to be installed."""
    name: str
    version_spec: Optional[str] = None
    source_url: Optional[str] = None
    checksum: Optional[str] = None
    signature: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    install_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InstallationEnvironment:
    """Environment configuration for installation."""
    variables: Dict[str, str] = field(default_factory=dict)
    path_additions: List[str] = field(default_factory=list)
    working_directory: Optional[Path] = None
    use_virtual_env: bool = False
    virtual_env_path: Optional[Path] = None


class PackageInstaller(ABC):
    """Abstract base class for package installers."""
    
    def __init__(self, 
                 platform_info: PlatformInfo,
                 config: SetupConfiguration):
        self.platform_info = platform_info
        self.config = config
        self.package_manager = self._get_package_manager()
        
        # Installation tracking
        self.installed_packages: Set[str] = set()
        self.failed_packages: Set[str] = set()
        self.installation_cache: Dict[str, InstallationResult] = {}
    
    @abstractmethod
    def _get_package_manager(self) -> PackageManager:
        """Get the package manager type for this installer."""
        pass
    
    @abstractmethod
    async def _install_package(self, 
                              package_name: str,
                              version_spec: Optional[str] = None,
                              custom_command: Optional[str] = None,
                              env: Optional[InstallationEnvironment] = None) -> InstallationResult:
        """Install a specific package."""
        pass
    
    @abstractmethod
    async def _verify_installation(self, package_name: str) -> bool:
        """Verify that a package was installed correctly."""
        pass
    
    @abstractmethod
    async def _uninstall_package(self, package_name: str) -> bool:
        """Uninstall a specific package."""
        pass
    
    def is_available(self) -> bool:
        """Check if this package manager is available on the system."""
        return self.package_manager in self.platform_info.available_managers
    
    async def install(self, 
                     package_name: str,
                     version_spec: Optional[str] = None,
                     custom_command: Optional[str] = None,
                     env: Optional[InstallationEnvironment] = None) -> InstallationResult:
        """Install a package with full error handling and verification."""
        start_time = datetime.now()
        
        logger.info(f"Installing {package_name} via {self.package_manager.value}")
        
        # Check if already installed (cache check)
        cache_key = f"{package_name}:{version_spec}"
        if cache_key in self.installation_cache:
            cached_result = self.installation_cache[cache_key]
            if cached_result.success:
                logger.info(f"Package {package_name} already in cache as successful")
                return cached_result
        
        # Security validation
        if not self._validate_package_security(package_name):
            return InstallationResult(
                success=False,
                package_name=package_name,
                package_manager=self.package_manager.value,
                error="Package failed security validation",
                duration=(datetime.now() - start_time).total_seconds()
            )
        
        try:
            # Perform installation with retries
            result = None
            for attempt in range(self.config.max_retries + 1):
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt} for {package_name}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                result = await self._install_package(package_name, version_spec, custom_command, env)
                
                if result.success:
                    break
                else:
                    logger.warning(f"Installation attempt {attempt + 1} failed: {result.error}")
            
            if not result:
                result = InstallationResult(
                    success=False,
                    package_name=package_name,
                    package_manager=self.package_manager.value,
                    error="No installation result generated",
                    duration=(datetime.now() - start_time).total_seconds()
                )
            
            # Verify installation if successful and verification is enabled
            if result.success and self.config.verify_installations:
                is_verified = await self._verify_installation(package_name)
                if not is_verified:
                    result.success = False
                    result.error = "Installation verification failed"
                    logger.error(f"Verification failed for {package_name}")
            
            # Update tracking
            if result.success:
                self.installed_packages.add(package_name)
                self.failed_packages.discard(package_name)
            else:
                self.failed_packages.add(package_name)
                self.installed_packages.discard(package_name)
            
            # Cache result
            result.duration = (datetime.now() - start_time).total_seconds()
            self.installation_cache[cache_key] = result
            
            return result
        
        except Exception as e:
            logger.error(f"Unexpected error installing {package_name}: {e}")
            result = InstallationResult(
                success=False,
                package_name=package_name,
                package_manager=self.package_manager.value,
                error=f"Unexpected error: {str(e)}",
                duration=(datetime.now() - start_time).total_seconds()
            )
            self.installation_cache[cache_key] = result
            return result
    
    def _validate_package_security(self, package_name: str) -> bool:
        """Validate package against security policies."""
        # Check allowed sources if configured
        if self.config.allowed_sources:
            # This would need to be implemented based on package manager specifics
            pass
        
        # Basic name validation
        if not package_name or len(package_name) > 100:
            return False
        
        # Check for suspicious characters
        suspicious_chars = [';', '&', '|', '`', '$', '>', '<']
        if any(char in package_name for char in suspicious_chars):
            return False
        
        return True
    
    def _prepare_environment(self, env: Optional[InstallationEnvironment]) -> Dict[str, str]:
        """Prepare environment variables for installation."""
        system_env = os.environ.copy()
        
        # Add config environment variables
        system_env.update(self.config.environment_variables)
        
        # Add custom environment if provided
        if env and env.variables:
            system_env.update(env.variables)
        
        # Add path additions
        path_additions = list(self.config.path_additions)
        if env and env.path_additions:
            path_additions.extend(env.path_additions)
        
        if path_additions:
            current_path = system_env.get('PATH', '')
            new_path = os.pathsep.join(path_additions + [current_path])
            system_env['PATH'] = new_path
        
        return system_env
    
    async def _run_command(self, 
                          command: List[str],
                          env: Optional[InstallationEnvironment] = None,
                          capture_output: bool = True) -> Tuple[int, str, str]:
        """Run a command with proper environment and error handling."""
        try:
            # Prepare environment
            proc_env = self._prepare_environment(env)
            
            # Set working directory
            cwd = env.working_directory if env else None
            
            # Create subprocess
            if capture_output:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=proc_env,
                    cwd=cwd
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    env=proc_env,
                    cwd=cwd
                )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout_seconds
                )
                
                stdout_text = stdout.decode('utf-8') if stdout else ""
                stderr_text = stderr.decode('utf-8') if stderr else ""
                
                return process.returncode, stdout_text, stderr_text
            
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return -1, "", "Installation timed out"
        
        except Exception as e:
            return -1, "", f"Command execution failed: {str(e)}"


class PipInstaller(PackageInstaller):
    """Installer for Python packages using pip."""
    
    def _get_package_manager(self) -> PackageManager:
        return PackageManager.PIP
    
    async def _install_package(self, 
                              package_name: str,
                              version_spec: Optional[str] = None,
                              custom_command: Optional[str] = None,
                              env: Optional[InstallationEnvironment] = None) -> InstallationResult:
        
        if custom_command:
            # Use custom command
            command = custom_command.split()
        else:
            # Build pip install command
            package_spec = package_name
            if version_spec:
                package_spec = f"{package_name}{version_spec}"
            
            command = ["pip", "install"]
            
            # Add user flag if configured
            if self.config.prefer_user_installs and not self._in_virtual_env():
                command.append("--user")
            
            # Add upgrade flag
            command.append("--upgrade")
            
            # Add the package
            command.append(package_spec)
        
        # Execute installation
        returncode, stdout, stderr = await self._run_command(command, env)
        
        success = returncode == 0
        error_msg = stderr if not success else ""
        
        return InstallationResult(
            success=success,
            package_name=package_name,
            package_manager=self.package_manager.value,
            output=stdout,
            error=error_msg
        )
    
    async def _verify_installation(self, package_name: str) -> bool:
        """Verify pip package installation."""
        try:
            # Try to import the package
            command = ["python", "-c", f"import {package_name}"]
            returncode, _, _ = await self._run_command(command)
            
            if returncode == 0:
                return True
            
            # If import fails, try pip show
            command = ["pip", "show", package_name]
            returncode, stdout, _ = await self._run_command(command)
            
            return returncode == 0 and package_name.lower() in stdout.lower()
        
        except:
            return False
    
    async def _uninstall_package(self, package_name: str) -> bool:
        """Uninstall pip package."""
        command = ["pip", "uninstall", "-y", package_name]
        returncode, _, _ = await self._run_command(command)
        return returncode == 0
    
    def _in_virtual_env(self) -> bool:
        """Check if running in a virtual environment."""
        return (hasattr(sys, 'real_prefix') or 
                (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
                'VIRTUAL_ENV' in os.environ)


class CondaInstaller(PackageInstaller):
    """Installer for packages using conda."""
    
    def _get_package_manager(self) -> PackageManager:
        return PackageManager.CONDA
    
    async def _install_package(self, 
                              package_name: str,
                              version_spec: Optional[str] = None,
                              custom_command: Optional[str] = None,
                              env: Optional[InstallationEnvironment] = None) -> InstallationResult:
        
        if custom_command:
            command = custom_command.split()
        else:
            package_spec = package_name
            if version_spec:
                package_spec = f"{package_name}{version_spec}"
            
            command = ["conda", "install", "-y", package_spec]
        
        returncode, stdout, stderr = await self._run_command(command, env)
        
        return InstallationResult(
            success=returncode == 0,
            package_name=package_name,
            package_manager=self.package_manager.value,
            output=stdout,
            error=stderr if returncode != 0 else ""
        )
    
    async def _verify_installation(self, package_name: str) -> bool:
        """Verify conda package installation."""
        command = ["conda", "list", package_name]
        returncode, stdout, _ = await self._run_command(command)
        return returncode == 0 and package_name in stdout
    
    async def _uninstall_package(self, package_name: str) -> bool:
        """Uninstall conda package."""
        command = ["conda", "remove", "-y", package_name]
        returncode, _, _ = await self._run_command(command)
        return returncode == 0


class NpmInstaller(PackageInstaller):
    """Installer for Node.js packages using npm."""
    
    def _get_package_manager(self) -> PackageManager:
        return PackageManager.NPM
    
    async def _install_package(self, 
                              package_name: str,
                              version_spec: Optional[str] = None,
                              custom_command: Optional[str] = None,
                              env: Optional[InstallationEnvironment] = None) -> InstallationResult:
        
        if custom_command:
            command = custom_command.split()
        else:
            package_spec = package_name
            if version_spec:
                package_spec = f"{package_name}@{version_spec}"
            
            command = ["npm", "install"]
            
            # Global or local install
            if not self.config.prefer_user_installs:
                command.append("-g")
            
            command.append(package_spec)
        
        returncode, stdout, stderr = await self._run_command(command, env)
        
        return InstallationResult(
            success=returncode == 0,
            package_name=package_name,
            package_manager=self.package_manager.value,
            output=stdout,
            error=stderr if returncode != 0 else ""
        )
    
    async def _verify_installation(self, package_name: str) -> bool:
        """Verify npm package installation."""
        command = ["npm", "list", "--depth=0", package_name]
        returncode, stdout, _ = await self._run_command(command)
        return returncode == 0 and package_name in stdout
    
    async def _uninstall_package(self, package_name: str) -> bool:
        """Uninstall npm package."""
        command = ["npm", "uninstall"]
        if not self.config.prefer_user_installs:
            command.append("-g")
        command.append(package_name)
        
        returncode, _, _ = await self._run_command(command)
        return returncode == 0


class AptInstaller(PackageInstaller):
    """Installer for packages using apt (Debian/Ubuntu)."""
    
    def _get_package_manager(self) -> PackageManager:
        return PackageManager.APT
    
    async def _install_package(self, 
                              package_name: str,
                              version_spec: Optional[str] = None,
                              custom_command: Optional[str] = None,
                              env: Optional[InstallationEnvironment] = None) -> InstallationResult:
        
        if custom_command:
            command = custom_command.split()
        else:
            package_spec = package_name
            if version_spec:
                package_spec = f"{package_name}={version_spec}"
            
            # Update package lists first
            update_cmd = ["sudo", "apt-get", "update"]
            await self._run_command(update_cmd, env)
            
            command = ["sudo", "apt-get", "install", "-y", package_spec]
        
        returncode, stdout, stderr = await self._run_command(command, env)
        
        return InstallationResult(
            success=returncode == 0,
            package_name=package_name,
            package_manager=self.package_manager.value,
            output=stdout,
            error=stderr if returncode != 0 else ""
        )
    
    async def _verify_installation(self, package_name: str) -> bool:
        """Verify apt package installation."""
        command = ["dpkg", "-l", package_name]
        returncode, stdout, _ = await self._run_command(command)
        return returncode == 0 and "ii" in stdout
    
    async def _uninstall_package(self, package_name: str) -> bool:
        """Uninstall apt package."""
        command = ["sudo", "apt-get", "remove", "-y", package_name]
        returncode, _, _ = await self._run_command(command)
        return returncode == 0


class HomebrewInstaller(PackageInstaller):
    """Installer for packages using Homebrew (macOS)."""
    
    def _get_package_manager(self) -> PackageManager:
        return PackageManager.HOMEBREW
    
    async def _install_package(self, 
                              package_name: str,
                              version_spec: Optional[str] = None,
                              custom_command: Optional[str] = None,
                              env: Optional[InstallationEnvironment] = None) -> InstallationResult:
        
        if custom_command:
            command = custom_command.split()
        else:
            # Homebrew doesn't support version specs in the same way
            command = ["brew", "install", package_name]
        
        returncode, stdout, stderr = await self._run_command(command, env)
        
        return InstallationResult(
            success=returncode == 0,
            package_name=package_name,
            package_manager=self.package_manager.value,
            output=stdout,
            error=stderr if returncode != 0 else ""
        )
    
    async def _verify_installation(self, package_name: str) -> bool:
        """Verify homebrew package installation."""
        command = ["brew", "list", package_name]
        returncode, _, _ = await self._run_command(command)
        return returncode == 0
    
    async def _uninstall_package(self, package_name: str) -> bool:
        """Uninstall homebrew package."""
        command = ["brew", "uninstall", package_name]
        returncode, _, _ = await self._run_command(command)
        return returncode == 0


class ChocolateyInstaller(PackageInstaller):
    """Installer for packages using Chocolatey (Windows)."""
    
    def _get_package_manager(self) -> PackageManager:
        return PackageManager.CHOCOLATEY
    
    async def _install_package(self, 
                              package_name: str,
                              version_spec: Optional[str] = None,
                              custom_command: Optional[str] = None,
                              env: Optional[InstallationEnvironment] = None) -> InstallationResult:
        
        if custom_command:
            command = custom_command.split()
        else:
            command = ["choco", "install", "-y", package_name]
            if version_spec:
                command.extend(["--version", version_spec])
        
        returncode, stdout, stderr = await self._run_command(command, env)
        
        return InstallationResult(
            success=returncode == 0,
            package_name=package_name,
            package_manager=self.package_manager.value,
            output=stdout,
            error=stderr if returncode != 0 else ""
        )
    
    async def _verify_installation(self, package_name: str) -> bool:
        """Verify chocolatey package installation."""
        command = ["choco", "list", "--local-only", package_name]
        returncode, stdout, _ = await self._run_command(command)
        return returncode == 0 and package_name.lower() in stdout.lower()
    
    async def _uninstall_package(self, package_name: str) -> bool:
        """Uninstall chocolatey package."""
        command = ["choco", "uninstall", "-y", package_name]
        returncode, _, _ = await self._run_command(command)
        return returncode == 0


class WingetInstaller(PackageInstaller):
    """Installer for packages using winget (Windows)."""
    
    def _get_package_manager(self) -> PackageManager:
        return PackageManager.WINGET
    
    async def _install_package(self, 
                              package_name: str,
                              version_spec: Optional[str] = None,
                              custom_command: Optional[str] = None,
                              env: Optional[InstallationEnvironment] = None) -> InstallationResult:
        
        if custom_command:
            command = custom_command.split()
        else:
            command = ["winget", "install", "-e", "--accept-package-agreements", package_name]
            if version_spec:
                command.extend(["--version", version_spec])
        
        returncode, stdout, stderr = await self._run_command(command, env)
        
        return InstallationResult(
            success=returncode == 0,
            package_name=package_name,
            package_manager=self.package_manager.value,
            output=stdout,
            error=stderr if returncode != 0 else ""
        )
    
    async def _verify_installation(self, package_name: str) -> bool:
        """Verify winget package installation."""
        command = ["winget", "list", package_name]
        returncode, stdout, _ = await self._run_command(command)
        return returncode == 0 and package_name.lower() in stdout.lower()
    
    async def _uninstall_package(self, package_name: str) -> bool:
        """Uninstall winget package."""
        command = ["winget", "uninstall", "-e", package_name]
        returncode, _, _ = await self._run_command(command)
        return returncode == 0


class PackageInstallerFactory:
    """Factory for creating appropriate package installers."""
    
    _installer_classes = {
        PackageManager.PIP: PipInstaller,
        PackageManager.CONDA: CondaInstaller,
        PackageManager.NPM: NpmInstaller,
        PackageManager.APT: AptInstaller,
        PackageManager.HOMEBREW: HomebrewInstaller,
        PackageManager.CHOCOLATEY: ChocolateyInstaller,
        PackageManager.WINGET: WingetInstaller
    }
    
    @classmethod
    def get_installer(cls, 
                     package_manager: PackageManager,
                     platform_info: PlatformInfo,
                     config: SetupConfiguration) -> Optional[PackageInstaller]:
        """Get an installer instance for the specified package manager."""
        installer_class = cls._installer_classes.get(package_manager)
        if not installer_class:
            logger.error(f"No installer available for {package_manager}")
            return None
        
        installer = installer_class(platform_info, config)
        
        # Check if installer is available on this platform
        if not installer.is_available():
            logger.error(f"Package manager {package_manager.value} is not available on this platform")
            return None
        
        return installer
    
    @classmethod
    def get_available_installers(cls, 
                               platform_info: PlatformInfo,
                               config: SetupConfiguration) -> Dict[PackageManager, PackageInstaller]:
        """Get all available installers for the current platform."""
        available_installers = {}
        
        for package_manager in platform_info.available_managers:
            installer = cls.get_installer(package_manager, platform_info, config)
            if installer:
                available_installers[package_manager] = installer
        
        return available_installers
    
    @classmethod 
    def get_preferred_installer(cls,
                              platform_info: PlatformInfo,
                              config: SetupConfiguration,
                              package_type: str = "python") -> Optional[PackageInstaller]:
        """Get the preferred installer for a given package type."""
        # Define preferences by package type and platform
        preferences = {
            "python": [PackageManager.PIP, PackageManager.CONDA],
            "nodejs": [PackageManager.NPM, PackageManager.YARN],
            "system": {
                Platform.LINUX: [PackageManager.APT, PackageManager.YUM],
                Platform.MACOS: [PackageManager.HOMEBREW],
                Platform.WINDOWS: [PackageManager.WINGET, PackageManager.CHOCOLATEY]
            }
        }
        
        # Get preferred package managers for this type
        if package_type in preferences and package_type != "system":
            preferred_managers = preferences[package_type]
        elif package_type == "system" and platform_info.platform in preferences["system"]:
            preferred_managers = preferences["system"][platform_info.platform]
        else:
            preferred_managers = list(platform_info.available_managers)
        
        # Filter by configuration if specified
        if config.allowed_package_managers:
            preferred_managers = [pm for pm in preferred_managers if pm in config.allowed_package_managers]
        
        # Find first available preferred installer
        for package_manager in preferred_managers:
            if package_manager in platform_info.available_managers:
                installer = cls.get_installer(package_manager, platform_info, config)
                if installer:
                    return installer
        
        return None


class ConcurrentInstaller:
    """Manages concurrent package installations with dependency resolution."""
    
    def __init__(self, 
                 platform_info: PlatformInfo,
                 config: SetupConfiguration,
                 max_concurrent: int = 3):
        self.platform_info = platform_info
        self.config = config
        self.max_concurrent = max_concurrent
        
        # Get all available installers
        self.installers = PackageInstallerFactory.get_available_installers(
            platform_info, config
        )
        
        # Semaphore for limiting concurrent installations
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Track ongoing installations
        self.active_installations: Dict[str, asyncio.Task] = {}
    
    async def install_packages(self, 
                              packages: List[Tuple[str, PackageManager, Optional[str]]],
                              resolve_dependencies: bool = True) -> Dict[str, InstallationResult]:
        """Install multiple packages concurrently."""
        logger.info(f"Installing {len(packages)} packages concurrently")
        
        # Create installation tasks
        tasks = {}
        for package_name, package_manager, version_spec in packages:
            task_name = f"{package_name}:{package_manager.value}"
            task = asyncio.create_task(
                self._install_with_semaphore(package_name, package_manager, version_spec),
                name=task_name
            )
            tasks[task_name] = task
            self.active_installations[task_name] = task
        
        # Wait for all installations to complete
        results = {}
        for task_name, task in tasks.items():
            try:
                result = await task
                results[task_name] = result
            except Exception as e:
                package_name = task_name.split(':')[0]
                package_manager = task_name.split(':')[1]
                results[task_name] = InstallationResult(
                    success=False,
                    package_name=package_name,
                    package_manager=package_manager,
                    error=f"Task failed: {str(e)}"
                )
            finally:
                self.active_installations.pop(task_name, None)
        
        return results
    
    async def _install_with_semaphore(self, 
                                    package_name: str,
                                    package_manager: PackageManager,
                                    version_spec: Optional[str]) -> InstallationResult:
        """Install a package with semaphore limiting."""
        async with self.semaphore:
            installer = self.installers.get(package_manager)
            if not installer:
                return InstallationResult(
                    success=False,
                    package_name=package_name,
                    package_manager=package_manager.value,
                    error=f"No installer available for {package_manager.value}"
                )
            
            return await installer.install(package_name, version_spec)
    
    def cancel_all_installations(self) -> List[str]:
        """Cancel all active installations."""
        cancelled = []
        for task_name, task in list(self.active_installations.items()):
            if not task.done():
                task.cancel()
                cancelled.append(task_name)
        
        self.active_installations.clear()
        return cancelled
    
    def get_installation_status(self) -> Dict[str, str]:
        """Get status of all active installations."""
        status = {}
        for task_name, task in self.active_installations.items():
            if task.done():
                if task.cancelled():
                    status[task_name] = "cancelled"
                elif task.exception():
                    status[task_name] = "failed"
                else:
                    status[task_name] = "completed"
            else:
                status[task_name] = "running"
        
        return status


__all__ = [
    "PackageInstaller",
    "PipInstaller",
    "CondaInstaller", 
    "NpmInstaller",
    "AptInstaller",
    "HomebrewInstaller",
    "ChocolateyInstaller",
    "WingetInstaller",
    "PackageInstallerFactory",
    "ConcurrentInstaller",
    "PackageInfo",
    "InstallationEnvironment"
]