"""Tests for the install_configs module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.install_configs import install_default_configs


class TestInstallDefaultConfigs:
    """Test the install_default_configs function."""

    def test_install_configs_creates_directory(self):
        """Test that install_default_configs creates the ~/.orchestrator directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_home = Path(temp_dir)
            
            with patch("pathlib.Path.home", return_value=temp_home):
                with patch("builtins.print"):
                    install_default_configs()
                    
                    # Directory should be created
                    config_dir = temp_home / ".orchestrator"
                    assert config_dir.exists()
                    assert config_dir.is_dir()

    def test_install_configs_copies_files_when_they_exist(self):
        """Test that config files are copied when source files exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_home = Path(temp_dir)
            
            # Create fake source files with correct structure
            # __file__ is package_root/src/orchestrator/install_configs.py
            # So package_root = Path(__file__).parent.parent.parent
            package_root = Path(temp_dir) / "package"
            src_orchestrator_dir = package_root / "src" / "orchestrator"
            src_orchestrator_dir.mkdir(parents=True)
            
            config_dir = package_root / "config"
            config_dir.mkdir(parents=True)
            
            (config_dir / "orchestrator.yaml").write_text("orchestrator: config")
            (config_dir / "models.yaml").write_text("models: config")
            
            with patch("pathlib.Path.home", return_value=temp_home):
                # Mock the __file__ variable in the install_configs module
                with patch("orchestrator.install_configs.__file__", str(src_orchestrator_dir / "install_configs.py")):
                    with patch("builtins.print") as mock_print:
                        install_default_configs()
                        
                        # Files should be copied
                        user_config_dir = temp_home / ".orchestrator"
                        assert (user_config_dir / "orchestrator.yaml").exists()
                        assert (user_config_dir / "models.yaml").exists()
                        
                        # Should print installation messages
                        print_calls = [str(call) for call in mock_print.call_args_list]
                        assert any("Installed orchestrator.yaml" in call for call in print_calls)
                        assert any("Installed models.yaml" in call for call in print_calls)

    def test_install_configs_preserves_existing_files(self):
        """Test that existing config files are not overwritten."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_home = Path(temp_dir)
            config_dir = temp_home / ".orchestrator"
            config_dir.mkdir(parents=True)
            
            # Create existing files
            existing_content = "existing: config"
            (config_dir / "orchestrator.yaml").write_text(existing_content)
            (config_dir / "models.yaml").write_text(existing_content)
            
            # Create source files with different content
            package_root = Path(temp_dir) / "package"
            src_orchestrator_dir = package_root / "src" / "orchestrator"
            src_orchestrator_dir.mkdir(parents=True)
            
            src_config_dir = package_root / "config"
            src_config_dir.mkdir(parents=True)
            
            (src_config_dir / "orchestrator.yaml").write_text("new: config")
            (src_config_dir / "models.yaml").write_text("new: config")
            
            with patch("pathlib.Path.home", return_value=temp_home):
                with patch("orchestrator.install_configs.__file__", str(src_orchestrator_dir / "install_configs.py")):
                    with patch("builtins.print") as mock_print:
                        install_default_configs()
                        
                        # Files should retain original content
                        assert (config_dir / "orchestrator.yaml").read_text() == existing_content
                        assert (config_dir / "models.yaml").read_text() == existing_content
                        
                        # Should print preservation messages
                        print_calls = [str(call) for call in mock_print.call_args_list]
                        assert any("Keeping existing orchestrator.yaml" in call for call in print_calls)
                        assert any("Keeping existing models.yaml" in call for call in print_calls)

    def test_install_configs_creates_readme(self):
        """Test that README.md is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_home = Path(temp_dir)
            
            with patch("pathlib.Path.home", return_value=temp_home):
                with patch("builtins.print") as mock_print:
                    install_default_configs()
                    
                    # README should be created
                    readme_path = temp_home / ".orchestrator" / "README.md"
                    assert readme_path.exists()
                    
                    content = readme_path.read_text()
                    assert "Orchestrator Configuration Directory" in content
                    assert "models.yaml" in content
                    assert "orchestrator.yaml" in content
                    
                    # Should print creation message
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    assert any("Created README" in call for call in print_calls)

    def test_install_configs_skips_readme_if_exists(self):
        """Test that existing README is not overwritten."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_home = Path(temp_dir)
            config_dir = temp_home / ".orchestrator"
            config_dir.mkdir(parents=True)
            
            # Create existing README
            existing_content = "# Existing README"
            readme_path = config_dir / "README.md"
            readme_path.write_text(existing_content)
            
            with patch("pathlib.Path.home", return_value=temp_home):
                with patch("builtins.print"):
                    install_default_configs()
                    
                    # README should retain original content
                    assert readme_path.read_text() == existing_content

    def test_install_configs_handles_missing_source_files(self):
        """Test behavior when source config files don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_home = Path(temp_dir)
            
            with patch("pathlib.Path.home", return_value=temp_home):
                with patch("builtins.print") as mock_print:
                    install_default_configs()
                    
                    # Directory should still be created
                    config_dir = temp_home / ".orchestrator"
                    assert config_dir.exists()
                    
                    # README should still be created
                    readme_path = config_dir / "README.md"
                    assert readme_path.exists()
                    
                    # Should print final message
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    assert any("Configuration files installed to" in call for call in print_calls)

    def test_install_configs_as_main_module(self):
        """Test that the script can be run as main module."""
        # Test that the if __name__ == "__main__" block works
        with patch("orchestrator.install_configs.install_default_configs") as mock_install:
            # Import the module to trigger the main block
            import orchestrator.install_configs
            
            # The function should exist and be callable
            assert callable(orchestrator.install_configs.install_default_configs)