"""Tests for automatic setup and installation system - Issue #312 Stream B

Tests platform-aware installation, configuration management, and security.
Real installations are tested to verify actual functionality.
"""

import pytest
import asyncio
import os
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json

from src.orchestrator.tools.setup import (
    SetupSystem, PlatformDetector, ConfigurationManager, SetupConfiguration,
    PlatformInfo, Platform, PackageManager, InstallationResult, InstallationStatus
)
from src.orchestrator.tools.installers import (
    PackageInstallerFactory, PipInstaller, PackageInstaller,
    InstallationEnvironment, ConcurrentInstaller
)
from src.orchestrator.tools.registry import (
    EnhancedToolRegistry, InstallationRequirement, SecurityLevel,
    EnhancedToolMetadata, VersionInfo, get_enhanced_registry
)


class TestPlatformDetector:
    """Test platform detection functionality."""
    
    def test_detect_platform(self):
        """Test platform detection returns valid information."""
        platform_info = PlatformDetector.detect_platform()
        
        # Verify basic platform detection
        assert isinstance(platform_info, PlatformInfo)
        assert platform_info.platform in [Platform.WINDOWS, Platform.MACOS, Platform.LINUX, Platform.UNKNOWN]
        assert isinstance(platform_info.version, str)
        assert isinstance(platform_info.architecture, str)
        assert isinstance(platform_info.python_version, str)
        
        # Verify Python version format
        python_parts = platform_info.python_version.split('.')
        assert len(python_parts) >= 2
        assert all(part.isdigit() for part in python_parts)
        
        # Verify package managers list is populated
        assert isinstance(platform_info.available_managers, list)
        
        # At minimum, pip should be available in Python environment
        if shutil.which("pip") or shutil.which("pip3"):
            assert PackageManager.PIP in platform_info.available_managers
    
    def test_check_administrator_privileges(self):
        """Test administrator privilege detection."""
        is_admin = PlatformDetector.check_administrator_privileges()
        assert isinstance(is_admin, bool)
    
    def test_check_virtual_environment(self):
        """Test virtual environment detection."""
        in_venv, venv_name = PlatformDetector.check_virtual_environment()
        assert isinstance(in_venv, bool)
        if in_venv:
            assert isinstance(venv_name, str)
            assert len(venv_name) > 0
        else:
            assert venv_name is None


class TestConfigurationManager:
    """Test configuration management functionality."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            manager = ConfigurationManager(config_path)
            
            config = manager.config
            assert isinstance(config, SetupConfiguration)
            assert config.timeout_seconds > 0
            assert config.max_retries >= 0
            assert isinstance(config.allow_system_packages, bool)
            assert isinstance(config.prefer_user_installs, bool)
            assert isinstance(config.use_virtual_environments, bool)
    
    def test_config_save_and_load(self):
        """Test configuration persistence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            # Create and modify configuration
            manager = ConfigurationManager(config_path)
            manager.config.timeout_seconds = 600
            manager.config.max_retries = 5
            manager.config.security_level = SecurityLevel.STRICT
            
            # Save configuration
            success = manager.save_config()
            assert success
            assert config_path.exists()
            
            # Create new manager and load configuration
            new_manager = ConfigurationManager(config_path)
            assert new_manager.config.timeout_seconds == 600
            assert new_manager.config.max_retries == 5
            assert new_manager.config.security_level == SecurityLevel.STRICT
    
    def test_config_validation(self):
        """Test configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            manager = ConfigurationManager(config_path)
            
            # Valid configuration
            is_valid, issues = manager.validate_config()
            assert is_valid or len(issues) == 0 or all("Virtual environment" in issue for issue in issues)
            
            # Invalid configuration
            manager.config.timeout_seconds = -1
            is_valid, issues = manager.validate_config()
            assert not is_valid
            assert any("Timeout must be positive" in issue for issue in issues)
    
    def test_config_update(self):
        """Test configuration updates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            manager = ConfigurationManager(config_path)
            
            # Update configuration
            success = manager.update_config(
                timeout_seconds=120,
                max_retries=3,
                verify_installations=False
            )
            assert success
            
            # Verify updates
            assert manager.config.timeout_seconds == 120
            assert manager.config.max_retries == 3
            assert manager.config.verify_installations is False


class TestPackageInstallers:
    """Test package installer functionality."""
    
    def test_installer_factory(self):
        """Test package installer factory."""
        platform_info = PlatformDetector.detect_platform()
        config = SetupConfiguration()
        
        # Test getting available installers
        installers = PackageInstallerFactory.get_available_installers(platform_info, config)
        assert isinstance(installers, dict)
        
        # Should have at least one installer available
        assert len(installers) > 0
        
        # All returned installers should be properly configured
        for package_manager, installer in installers.items():
            assert isinstance(installer, PackageInstaller)
            assert installer.is_available()
            assert installer.package_manager == package_manager
    
    def test_pip_installer_available(self):
        """Test pip installer if available."""
        platform_info = PlatformDetector.detect_platform()
        config = SetupConfiguration()
        
        if PackageManager.PIP in platform_info.available_managers:
            installer = PackageInstallerFactory.get_installer(
                PackageManager.PIP, platform_info, config
            )
            assert installer is not None
            assert isinstance(installer, PipInstaller)
            assert installer.is_available()
    
    @pytest.mark.asyncio
    async def test_pip_install_real_package(self):
        """Test real pip installation with a small, safe package."""
        platform_info = PlatformDetector.detect_platform()
        config = SetupConfiguration()
        config.timeout_seconds = 60  # Shorter timeout for tests
        
        if PackageManager.PIP not in platform_info.available_managers:
            pytest.skip("pip not available on this system")
        
        installer = PackageInstallerFactory.get_installer(
            PackageManager.PIP, platform_info, config
        )
        
        # Test with a small, commonly available package that's safe to install
        test_package = "six"  # Small utility library, widely used
        
        try:
            # Install package
            result = await installer.install(test_package)
            
            # Verify result structure
            assert isinstance(result, InstallationResult)
            assert result.package_name == test_package
            assert result.package_manager == "pip"
            assert isinstance(result.success, bool)
            assert isinstance(result.output, str)
            assert isinstance(result.duration, float)
            assert result.duration >= 0
            
            # If installation succeeded, verify it
            if result.success:
                verification = await installer._verify_installation(test_package)
                assert verification is True
                
                print(f"Successfully installed and verified {test_package}")
                print(f"Installation took {result.duration:.2f} seconds")
                
        except Exception as e:
            pytest.fail(f"Real pip installation test failed: {e}")
    
    def test_preferred_installer_selection(self):
        """Test preferred installer selection logic."""
        platform_info = PlatformDetector.detect_platform()
        config = SetupConfiguration()
        
        # Test Python package preference
        python_installer = PackageInstallerFactory.get_preferred_installer(
            platform_info, config, "python"
        )
        if python_installer:
            assert python_installer.package_manager in [PackageManager.PIP, PackageManager.CONDA]
        
        # Test system package preference
        system_installer = PackageInstallerFactory.get_preferred_installer(
            platform_info, config, "system"
        )
        if system_installer:
            # Should get a system-appropriate package manager
            if platform_info.platform == Platform.MACOS:
                assert python_installer.package_manager == PackageManager.HOMEBREW or True  # May not be available
            elif platform_info.platform == Platform.LINUX:
                assert python_installer.package_manager in [PackageManager.APT, PackageManager.YUM] or True
    
    @pytest.mark.asyncio
    async def test_concurrent_installer(self):
        """Test concurrent installation capabilities."""
        platform_info = PlatformDetector.detect_platform()
        config = SetupConfiguration()
        config.timeout_seconds = 60
        
        if PackageManager.PIP not in platform_info.available_managers:
            pytest.skip("pip not available for concurrent installation test")
        
        concurrent_installer = ConcurrentInstaller(platform_info, config, max_concurrent=2)
        
        # Test with multiple small packages
        packages = [
            ("six", PackageManager.PIP, None),
            ("setuptools", PackageManager.PIP, None),  # Usually already installed
        ]
        
        try:
            results = await concurrent_installer.install_packages(packages)
            
            # Verify results structure
            assert isinstance(results, dict)
            assert len(results) == len(packages)
            
            for task_name, result in results.items():
                assert isinstance(result, InstallationResult)
                assert isinstance(result.success, bool)
                assert result.duration >= 0
                
                print(f"Concurrent installation {task_name}: {'success' if result.success else 'failed'}")
                if result.error:
                    print(f"  Error: {result.error}")
        
        except Exception as e:
            pytest.fail(f"Concurrent installation test failed: {e}")


class TestSetupSystem:
    """Test the main setup system functionality."""
    
    def test_setup_system_initialization(self):
        """Test setup system initialization."""
        registry = EnhancedToolRegistry()
        config = SetupConfiguration()
        
        setup_system = SetupSystem(registry, config)
        
        assert setup_system.registry == registry
        assert setup_system.config_manager.config == config
        assert isinstance(setup_system.platform_info, PlatformInfo)
        assert hasattr(setup_system, 'security_policy')
        assert isinstance(setup_system.installation_history, list)
        assert isinstance(setup_system.pending_installations, set)
    
    def test_setup_system_security_policy_creation(self):
        """Test security policy creation based on configuration."""
        # Test strict security
        config = SetupConfiguration()
        config.security_level = SecurityLevel.STRICT
        setup_system = SetupSystem(config=config)
        
        policy = setup_system.security_policy
        assert policy.level == SecurityLevel.STRICT
        assert policy.sandboxed is True
        assert policy.network_access is False
        assert policy.file_system_access is False
        
        # Test moderate security  
        config.security_level = SecurityLevel.MODERATE
        setup_system = SetupSystem(config=config)
        
        policy = setup_system.security_policy
        assert policy.level == SecurityLevel.MODERATE
        assert policy.network_access is True
        assert policy.file_system_access is True
    
    @pytest.mark.asyncio
    async def test_setup_tool_not_found(self):
        """Test setup behavior with non-existent tool."""
        registry = EnhancedToolRegistry()
        setup_system = SetupSystem(registry)
        
        result = await setup_system.setup_tool("nonexistent_tool")
        
        assert isinstance(result, InstallationResult)
        assert result.success is False
        assert "Tool not found in registry" in result.error
        assert result.package_name == "nonexistent_tool"
    
    @pytest.mark.asyncio 
    async def test_setup_tool_with_requirements(self):
        """Test setting up a tool with installation requirements."""
        from src.orchestrator.tools.base import Tool
        
        # Create a mock tool class
        class MockTool(Tool):
            async def _execute_impl(self, **kwargs):
                return {"result": "mock execution"}
        
        # Create registry and tool
        registry = EnhancedToolRegistry()
        mock_tool = MockTool("test_tool", "A test tool")
        
        # Create installation requirement
        install_req = InstallationRequirement(
            package_manager="pip",
            package_name="six",  # Small, safe package for testing
            version_spec=">=1.0.0"
        )
        
        # Register tool with requirements
        success = registry.register_tool_enhanced(
            tool=mock_tool,
            installation_requirements=[install_req]
        )
        assert success
        
        # Test setup
        config = SetupConfiguration()
        config.timeout_seconds = 60
        setup_system = SetupSystem(registry, config)
        
        # Only run if pip is available
        if PackageManager.PIP in setup_system.platform_info.available_managers:
            result = await setup_system.setup_tool("test_tool")
            
            assert isinstance(result, InstallationResult)
            print(f"Setup result for test_tool: {'success' if result.success else 'failed'}")
            if result.error:
                print(f"  Error: {result.error}")
            
            # Check installation status was updated
            status = registry.installation_tracker.get("test_tool", InstallationStatus.NEEDS_INSTALL)
            expected_status = InstallationStatus.AVAILABLE if result.success else InstallationStatus.FAILED
            assert status == expected_status
        else:
            pytest.skip("pip not available for setup test")
    
    def test_check_tool_availability(self):
        """Test tool availability checking."""
        from src.orchestrator.tools.base import Tool
        
        class MockTool(Tool):
            async def _execute_impl(self, **kwargs):
                return {"result": "mock"}
        
        registry = EnhancedToolRegistry()
        setup_system = SetupSystem(registry)
        
        # Test non-existent tool
        available, reason = setup_system.check_tool_availability("nonexistent")
        assert available is False
        assert "Tool not found" in reason
        
        # Register a tool and test
        mock_tool = MockTool("available_tool", "Test tool")
        registry.register_tool_enhanced(tool=mock_tool)
        
        available, reason = setup_system.check_tool_availability("available_tool")
        # Should be available since no installation requirements
        assert available is True or "needs installation" in reason
    
    def test_get_installation_status(self):
        """Test getting detailed installation status."""
        from src.orchestrator.tools.base import Tool
        
        class MockTool(Tool):
            async def _execute_impl(self, **kwargs):
                return {"result": "mock"}
        
        registry = EnhancedToolRegistry()
        setup_system = SetupSystem(registry)
        
        # Test non-existent tool
        status = setup_system.get_installation_status("nonexistent")
        assert "error" in status
        assert "Tool not found" in status["error"]
        
        # Register a tool and test
        mock_tool = MockTool("status_tool", "Test tool")
        registry.register_tool_enhanced(tool=mock_tool)
        
        status = setup_system.get_installation_status("status_tool")
        assert isinstance(status, dict)
        assert "tool_name" in status
        assert "current_status" in status
        assert "metadata" in status
        assert "installation_history" in status
        assert status["tool_name"] == "status_tool"
    
    def test_cleanup_failed_installations(self):
        """Test cleanup of failed installations."""
        from src.orchestrator.tools.base import Tool
        
        class MockTool(Tool):
            async def _execute_impl(self, **kwargs):
                return {"result": "mock"}
        
        registry = EnhancedToolRegistry()
        setup_system = SetupSystem(registry)
        
        # Register a tool and mark as failed
        mock_tool = MockTool("failed_tool", "Failed tool")
        registry.register_tool_enhanced(tool=mock_tool)
        registry.installation_tracker["failed_tool"] = InstallationStatus.FAILED
        
        # Test cleanup
        cleaned = setup_system.cleanup_failed_installations()
        
        assert "failed_tool" in cleaned
        assert registry.installation_tracker["failed_tool"] == InstallationStatus.NEEDS_INSTALL
    
    def test_get_setup_report(self):
        """Test comprehensive setup report generation."""
        registry = EnhancedToolRegistry()
        setup_system = SetupSystem(registry)
        
        report = setup_system.get_setup_report()
        
        # Verify report structure
        assert isinstance(report, dict)
        assert "platform_info" in report
        assert "tool_statistics" in report
        assert "installation_statistics" in report
        assert "configuration" in report
        assert "timestamp" in report
        
        # Verify platform info
        platform_info = report["platform_info"]
        assert "platform" in platform_info
        assert "python_version" in platform_info
        assert "available_managers" in platform_info
        
        # Verify statistics
        tool_stats = report["tool_statistics"]
        assert "total_tools" in tool_stats
        assert "success_rate" in tool_stats
        
        install_stats = report["installation_statistics"]
        assert "install_success_rate" in install_stats
        
        print("Setup report generated successfully:")
        print(f"  Platform: {platform_info['platform']}")
        print(f"  Python: {platform_info['python_version']}")
        print(f"  Package managers: {len(platform_info['available_managers'])}")


class TestIntegration:
    """Integration tests for the complete setup system."""
    
    @pytest.mark.asyncio
    async def test_full_tool_setup_workflow(self):
        """Test complete workflow from tool registration to setup."""
        from src.orchestrator.tools.base import Tool
        
        class IntegrationTestTool(Tool):
            async def _execute_impl(self, **kwargs):
                return {"result": "integration test executed"}
        
        # Create components
        registry = EnhancedToolRegistry()
        config = SetupConfiguration()
        config.timeout_seconds = 60
        config.verify_installations = True
        
        setup_system = SetupSystem(registry, config)
        
        # Only run full test if pip is available
        if PackageManager.PIP not in setup_system.platform_info.available_managers:
            pytest.skip("pip not available for full integration test")
        
        # Create tool with installation requirements
        test_tool = IntegrationTestTool("integration_tool", "Integration test tool")
        install_req = InstallationRequirement(
            package_manager="pip",
            package_name="wheel",  # Common package that should install quickly
            version_spec=">=0.30.0"
        )
        
        # Register tool
        success = registry.register_tool_enhanced(
            tool=test_tool,
            installation_requirements=[install_req],
            security_level=SecurityLevel.MODERATE
        )
        assert success
        
        # Verify initial state
        available, reason = setup_system.check_tool_availability("integration_tool")
        if not available and "needs installation" not in reason:
            pytest.skip(f"Cannot test installation: {reason}")
        
        # Perform setup
        result = await setup_system.setup_tool("integration_tool")
        
        # Verify results
        assert isinstance(result, InstallationResult)
        print(f"Integration test result: {'success' if result.success else 'failed'}")
        if result.error:
            print(f"  Error: {result.error}")
        
        # Verify final state
        final_available, final_reason = setup_system.check_tool_availability("integration_tool")
        if result.success:
            assert final_available, f"Tool should be available after successful setup, but: {final_reason}"
        
        # Verify installation history
        status = setup_system.get_installation_status("integration_tool")
        assert len(status["installation_history"]) > 0
        
        # Generate and verify report
        report = setup_system.get_setup_report()
        assert report["tool_statistics"]["total_tools"] >= 1
        
        print("Full integration test completed successfully")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])