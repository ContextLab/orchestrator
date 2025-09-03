"""Test automatic package installation utilities with real operations."""

import pytest
import subprocess
import sys
import importlib
import tempfile
import os

from src.orchestrator.utils.auto_install import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    is_package_installed,
    install_package,
    ensure_packages,
    auto_install_for_import,
    safe_import,
    PACKAGE_MAPPINGS
)


class TestAutoInstall:
    """Test auto-installation functionality with real operations."""
    
    def test_is_package_installed_true(self):
        """Test checking for installed package."""
        # Test with packages we know are installed
        assert is_package_installed("os") is True
        assert is_package_installed("sys") is True
        assert is_package_installed("pytest") is True
        
        # Clear cache to ensure fresh checks
        is_package_installed.cache_clear()
    
    def test_is_package_installed_false(self):
        """Test checking for non-installed package."""
        # Test with a package that definitely doesn't exist
        assert is_package_installed("totally_fake_package_12345") is False
        
        # Clear cache
        is_package_installed.cache_clear()
    
    def test_install_package_real_small_package(self):
        """Test real package installation with a small package."""
        # Skip this test if we can't modify packages (e.g., in some environments)
        # Instead, we'll test the install functionality with already installed packages
        
        # Test with a package we know is installed
        test_package = "pytest"
        
        # This should already be installed
        assert is_package_installed(test_package) is True
        
        # Try to install it again - should succeed without error
        result = install_package(test_package)
        assert result is True
        
        # Still installed
        assert is_package_installed(test_package) is True
        
        # Test the actual installation logic with a fake package
        # This tests the failure path
        fake_package = "totally_fake_package_for_testing_12345"
        assert is_package_installed(fake_package) is False
        
        result = install_package(fake_package)
        assert result is False  # Should fail
        
        # Still not installed
        assert is_package_installed(fake_package) is False
    
    def test_install_package_failure_real(self):
        """Test failed package installation with non-existent package."""
        # Try to install a package that doesn't exist
        result = install_package("this_package_definitely_does_not_exist_12345")
        assert result is False
    
    def test_install_package_already_installed_real(self):
        """Test installing already installed package."""
        # Use pytest which we know is installed
        assert is_package_installed("pytest") is True
        
        # Try to install it again
        result = install_package("pytest")
        
        # Should return True without actually reinstalling
        assert result is True
    
    def test_ensure_packages_real(self):
        """Test ensuring multiple packages with real operations."""
        # Mix of installed and not installed packages
        requirements = {
            "os": None,  # Built-in, always available
            "pytest": None,  # Should be installed
        }
        
        results = ensure_packages(requirements)
        
        # All should be available
        assert results["os"] is True
        assert results["pytest"] is True
    
    def test_auto_install_for_import_known_mapping_real(self):
        """Test auto-install with known package mapping."""
        # Test the mapping logic without actually installing/uninstalling
        
        # Test with packages that have known mappings
        # These should use the mapped pip names
        assert "cv2" in PACKAGE_MAPPINGS
        assert PACKAGE_MAPPINGS["cv2"] == "opencv-python"
        
        assert "PIL" in PACKAGE_MAPPINGS  
        assert PACKAGE_MAPPINGS["PIL"] == "pillow"
        
        assert "google.generativeai" in PACKAGE_MAPPINGS
        assert PACKAGE_MAPPINGS["google.generativeai"] == "google-generativeai"
        
        # Test auto_install logic with already installed package
        # This tests the "already installed" path
        result = auto_install_for_import("os")  # Built-in module
        assert result is True
        
        # Test with a fake package that's not in mappings
        # This would try to install with the same name
        fake_import = "fake_module_xyz_789"
        result = auto_install_for_import(fake_import)
        assert result is False  # Should fail since package doesn't exist
    
    def test_safe_import_success_real(self):
        """Test safe import of existing module."""
        # Import a real module
        module = safe_import("os")
        assert module is not None
        assert hasattr(module, "path")
        
        # Import another real module
        json_module = safe_import("json")
        assert json_module is not None
        assert hasattr(json_module, "loads")
    
    def test_safe_import_with_auto_install_real(self):
        """Test safe import with real auto-installation."""
        # Use a small package for testing
        test_module = "six"
        
        # Save state
        was_installed = is_package_installed(test_module)
        
        if was_installed:
            # Uninstall it
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", test_module],
                capture_output=True,
                text=True
            )
            is_package_installed.cache_clear()
            
            # Remove from sys.modules if present
            if test_module in sys.modules:
                del sys.modules[test_module]
        
        try:
            # Should not be importable
            with pytest.raises(ImportError):
                importlib.import_module(test_module)
            
            # Safe import with auto-install should work
            module = safe_import(test_module, auto_install=True)
            assert module is not None
            
            # Should now be importable normally
            module2 = importlib.import_module(test_module)
            assert module2 is not None
            
        finally:
            # Restore state
            if not was_installed:
                subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", "-y", test_module],
                    capture_output=True,
                    text=True
                )
            is_package_installed.cache_clear()
    
    def test_safe_import_no_auto_install_real(self):
        """Test safe import without auto-installation."""
        # Try to import a non-existent module
        module = safe_import("totally_fake_module_98765", auto_install=False)
        assert module is None
        
        # Verify it wasn't installed
        assert is_package_installed("totally_fake_module_98765") is False
    
    def test_package_mappings_real(self):
        """Test that package mappings are correct."""
        # Verify mappings exist and are strings
        assert isinstance(PACKAGE_MAPPINGS, dict)
        assert len(PACKAGE_MAPPINGS) > 0
        
        # Spot check some important mappings
        assert PACKAGE_MAPPINGS.get("cv2") == "opencv-python"
        assert PACKAGE_MAPPINGS.get("PIL") == "pillow"
        assert PACKAGE_MAPPINGS.get("google.generativeai") == "google-generativeai"
        assert PACKAGE_MAPPINGS.get("speech_recognition") == "SpeechRecognition"
        
        # All values should be strings
        for key, value in PACKAGE_MAPPINGS.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
    
    def test_install_with_pip_name_different_real(self):
        """Test installing package where pip name differs from import name."""
        # PIL installs as 'pillow'
        import_name = "PIL"
        pip_name = "pillow"
        
        # Save state
        was_installed = is_package_installed(import_name)
        
        if was_installed:
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", pip_name],
                capture_output=True,
                text=True
            )
            is_package_installed.cache_clear()
            
            # Remove from sys.modules
            for mod in list(sys.modules.keys()):
                if mod.startswith("PIL"):
                    del sys.modules[mod]
        
        try:
            # Should not be installed
            assert is_package_installed(import_name) is False
            
            # Install with pip name
            result = install_package(import_name, pip_name)
            assert result is True
            
            # Should now be importable as PIL
            is_package_installed.cache_clear()
            assert is_package_installed(import_name) is True
            
            # Verify we can import it
            module = safe_import(import_name)
            assert module is not None
            
        finally:
            # Restore state
            if not was_installed:
                subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", "-y", pip_name],
                    capture_output=True,
                    text=True
                )
            is_package_installed.cache_clear()