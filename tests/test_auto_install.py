"""Test automatic package installation utilities."""

import pytest
from unittest.mock import patch, MagicMock, call
import subprocess
import sys

from orchestrator.utils.auto_install import (
    is_package_installed,
    install_package,
    ensure_packages,
    auto_install_for_import,
    safe_import,
    PACKAGE_MAPPINGS
)


class TestAutoInstall:
    """Test auto-installation functionality."""
    
    def test_is_package_installed_true(self):
        """Test checking for installed package."""
        # Test with a package we know is installed
        assert is_package_installed("os") is True
        assert is_package_installed("sys") is True
        assert is_package_installed("pytest") is True
    
    def test_is_package_installed_false(self):
        """Test checking for non-installed package."""
        # Test with a package that definitely doesn't exist
        assert is_package_installed("totally_fake_package_12345") is False
    
    @patch('subprocess.run')
    def test_install_package_success(self, mock_run):
        """Test successful package installation."""
        # Mock successful pip install
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        
        # Clear cache to ensure fresh check
        is_package_installed.cache_clear()
        
        with patch('orchestrator.utils.auto_install.is_package_installed') as mock_installed:
            # First call returns False (not installed), second returns True (after install)
            mock_installed.side_effect = [False, True]
            
            result = install_package("test_package")
            
            assert result is True
            mock_run.assert_called_once_with(
                [sys.executable, "-m", "pip", "install", "test_package"],
                capture_output=True,
                text=True,
                check=False
            )
    
    @patch('subprocess.run')
    def test_install_package_with_pip_name(self, mock_run):
        """Test package installation with different pip name."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        
        with patch('orchestrator.utils.auto_install.is_package_installed') as mock_installed:
            mock_installed.return_value = False
            
            install_package("cv2", "opencv-python")
            
            mock_run.assert_called_once_with(
                [sys.executable, "-m", "pip", "install", "opencv-python"],
                capture_output=True,
                text=True,
                check=False
            )
    
    @patch('subprocess.run')
    def test_install_package_failure(self, mock_run):
        """Test failed package installation."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Error: Package not found")
        
        with patch('orchestrator.utils.auto_install.is_package_installed') as mock_installed:
            mock_installed.return_value = False
            
            result = install_package("fake_package")
            
            assert result is False
    
    def test_install_package_already_installed(self):
        """Test installing already installed package."""
        with patch('orchestrator.utils.auto_install.is_package_installed') as mock_installed:
            mock_installed.return_value = True
            
            with patch('subprocess.run') as mock_run:
                result = install_package("already_installed")
                
                # Should not call pip if already installed
                assert result is True
                mock_run.assert_not_called()
    
    @patch('subprocess.run')
    def test_ensure_packages(self, mock_run):
        """Test ensuring multiple packages."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        
        with patch('orchestrator.utils.auto_install.is_package_installed') as mock_installed:
            # os is installed, fake_package is not
            def is_installed_side_effect(name):
                return name == "os"
            
            mock_installed.side_effect = is_installed_side_effect
            
            requirements = {
                "os": None,
                "fake_package": "fake-package-pip",
            }
            
            results = ensure_packages(requirements)
            
            assert results["os"] is True
            assert results["fake_package"] is True
            
            # Only fake_package should trigger pip install
            mock_run.assert_called_once()
            assert "fake-package-pip" in mock_run.call_args[0][0]
    
    def test_auto_install_for_import_known_package(self):
        """Test auto-install for known package mapping."""
        with patch('orchestrator.utils.auto_install.is_package_installed') as mock_installed:
            mock_installed.return_value = False
            
            with patch('orchestrator.utils.auto_install.install_package') as mock_install:
                mock_install.return_value = True
                
                result = auto_install_for_import("cv2")
                
                # Should use the known mapping
                mock_install.assert_called_once_with("cv2", "opencv-python")
                assert result is True
    
    def test_auto_install_for_import_unknown_package(self):
        """Test auto-install for unknown package."""
        with patch('orchestrator.utils.auto_install.is_package_installed') as mock_installed:
            mock_installed.return_value = False
            
            with patch('orchestrator.utils.auto_install.install_package') as mock_install:
                mock_install.return_value = True
                
                result = auto_install_for_import("unknown_package")
                
                # Should use package name as pip name
                mock_install.assert_called_once_with("unknown_package")
                assert result is True
    
    def test_safe_import_success(self):
        """Test safe import of existing module."""
        module = safe_import("os")
        assert module is not None
        assert hasattr(module, "path")
    
    def test_safe_import_with_auto_install(self):
        """Test safe import with auto-installation."""
        with patch('orchestrator.utils.auto_install.importlib.import_module') as mock_import:
            # First import fails, second succeeds after install
            mock_import.side_effect = [ImportError(), MagicMock()]
            
            with patch('orchestrator.utils.auto_install.auto_install_for_import') as mock_auto:
                mock_auto.return_value = True
                
                module = safe_import("test_module", auto_install=True)
                
                assert module is not None
                mock_auto.assert_called_once_with("test_module")
                assert mock_import.call_count == 2
    
    def test_safe_import_no_auto_install(self):
        """Test safe import without auto-installation."""
        with patch('orchestrator.utils.auto_install.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError()
            
            module = safe_import("test_module", auto_install=False)
            
            assert module is None
            mock_import.assert_called_once()
    
    def test_package_mappings(self):
        """Test that package mappings are correct."""
        # Spot check some important mappings
        assert PACKAGE_MAPPINGS["cv2"] == "opencv-python"
        assert PACKAGE_MAPPINGS["PIL"] == "pillow"
        assert PACKAGE_MAPPINGS["google.generativeai"] == "google-generativeai"
        assert PACKAGE_MAPPINGS["speech_recognition"] == "SpeechRecognition"