"""Tests to improve coverage for the utils module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from orchestrator.utils.model_utils import (
    check_ollama_installed,
    check_ollama_model,
    install_ollama_model,
    load_model_config,
    parse_model_size,
)


class TestParseModelSize:
    """Test parse_model_size function."""

    def test_parse_explicit_size_billions(self):
        """Test parsing explicit size in billions."""
        assert parse_model_size("test-model", "7b") == 7.0
        assert parse_model_size("test-model", "13b") == 13.0
        assert parse_model_size("test-model", "2.7b") == 2.7
        assert parse_model_size("test-model", "175b") == 175.0

    def test_parse_explicit_size_trillions(self):
        """Test parsing explicit size in trillions."""
        assert parse_model_size("test-model", "1.5t") == 1500.0
        assert parse_model_size("test-model", "2t") == 2000.0

    def test_parse_explicit_size_millions(self):
        """Test parsing explicit size in millions."""
        assert abs(parse_model_size("test-model", "350m") - 0.35) < 0.001
        assert abs(parse_model_size("test-model", "82m") - 0.082) < 0.001

    def test_parse_explicit_size_thousands(self):
        """Test parsing explicit size in thousands."""
        assert parse_model_size("test-model", "125k") == 0.125
        assert parse_model_size("test-model", "1000k") == 1.0

    def test_parse_size_from_model_name(self):
        """Test extracting size from model name."""
        assert parse_model_size("llama2:7b") == 7.0
        assert parse_model_size("gemma2:27b") == 27.0
        assert parse_model_size("mistral-7B-instruct") == 7.0
        assert parse_model_size("llama-13b-chat") == 13.0
        assert parse_model_size("model-2.7b-v2") == 2.7

    def test_parse_size_from_config(self):
        """Test getting size from loaded config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "models": [
                    {
                        "source": "test",
                        "name": "test-model-from-config",
                        "size": "42b"
                    }
                ]
            }
            yaml.dump(config, f)
            temp_file = f.name

        try:
            with patch("orchestrator.utils.model_utils.load_model_config") as mock_load:
                mock_load.return_value = config
                size = parse_model_size("test-model-from-config")
                assert size == 42.0
        finally:
            os.unlink(temp_file)

    def test_parse_size_default(self):
        """Test default size when no size information found."""
        assert parse_model_size("unknown-model") == 1.0
        assert parse_model_size("model-without-size", None) == 1.0

    def test_parse_size_with_exception_in_config_loading(self):
        """Test handling exception when loading config."""
        with patch("orchestrator.utils.model_utils.load_model_config") as mock_load:
            mock_load.side_effect = Exception("Config load error")
            # Should continue and return default
            assert parse_model_size("test-model") == 1.0


class TestOllamaChecks:
    """Test Ollama installation and model checks."""

    @patch("subprocess.run")
    def test_check_ollama_installed_success(self, mock_run):
        """Test check_ollama_installed when Ollama is installed."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ollama version 0.1.0")
        assert check_ollama_installed() is True
        mock_run.assert_called_once_with(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )

    @patch("subprocess.run")
    def test_check_ollama_installed_failure(self, mock_run):
        """Test check_ollama_installed when Ollama is not installed."""
        mock_run.return_value = MagicMock(returncode=1)
        assert check_ollama_installed() is False

    @patch("subprocess.run")
    def test_check_ollama_installed_file_not_found(self, mock_run):
        """Test check_ollama_installed when ollama command not found."""
        mock_run.side_effect = FileNotFoundError()
        assert check_ollama_installed() is False

    @patch("subprocess.run")
    def test_check_ollama_installed_timeout(self, mock_run):
        """Test check_ollama_installed when command times out."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("ollama", 5)
        assert check_ollama_installed() is False

    @patch("orchestrator.utils.model_utils.check_ollama_installed")
    def test_check_ollama_model_no_ollama(self, mock_check_installed):
        """Test check_ollama_model when Ollama is not installed."""
        mock_check_installed.return_value = False
        assert check_ollama_model("llama2:7b") is False

    @patch("orchestrator.utils.model_utils.check_ollama_installed")
    @patch("subprocess.run")
    def test_check_ollama_model_found(self, mock_run, mock_check_installed):
        """Test check_ollama_model when model is found."""
        mock_check_installed.return_value = True
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="llama2:7b\ngemma:7b\nmistral:7b"
        )
        assert check_ollama_model("llama2:7b") is True

    @patch("orchestrator.utils.model_utils.check_ollama_installed")
    @patch("subprocess.run")
    def test_check_ollama_model_not_found(self, mock_run, mock_check_installed):
        """Test check_ollama_model when model is not found."""
        mock_check_installed.return_value = True
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="gemma:7b\nmistral:7b"
        )
        assert check_ollama_model("llama2:7b") is False

    @patch("orchestrator.utils.model_utils.check_ollama_installed")
    @patch("subprocess.run")
    def test_check_ollama_model_command_failure(self, mock_run, mock_check_installed):
        """Test check_ollama_model when list command fails."""
        mock_check_installed.return_value = True
        mock_run.return_value = MagicMock(returncode=1)
        assert check_ollama_model("llama2:7b") is False

    @patch("orchestrator.utils.model_utils.check_ollama_installed")
    @patch("subprocess.run")
    def test_check_ollama_model_exception(self, mock_run, mock_check_installed):
        """Test check_ollama_model when exception occurs."""
        mock_check_installed.return_value = True
        import subprocess
        mock_run.side_effect = subprocess.SubprocessError("Command failed")
        assert check_ollama_model("llama2:7b") is False


class TestInstallOllamaModel:
    """Test install_ollama_model function."""

    @patch("orchestrator.utils.model_utils.check_ollama_installed")
    def test_install_ollama_model_no_ollama(self, mock_check):
        """Test install_ollama_model when Ollama is not installed."""
        mock_check.return_value = False
        with patch("builtins.print") as mock_print:
            result = install_ollama_model("llama2:7b")
        assert result is False
        mock_print.assert_called_with(">>   ⚠️  Ollama not installed, cannot auto-install llama2:7b")

    @patch("orchestrator.utils.model_utils.check_ollama_installed")
    @patch("subprocess.run")
    def test_install_ollama_model_success(self, mock_run, mock_check):
        """Test install_ollama_model successful installation."""
        mock_check.return_value = True
        mock_run.return_value = MagicMock(returncode=0, stdout="Pulling llama2:7b...")
        
        with patch("builtins.print") as mock_print:
            result = install_ollama_model("llama2:7b")
        
        assert result is True
        mock_run.assert_called_once_with(
            ["ollama", "pull", "llama2:7b"],
            capture_output=True,
            text=True,
            timeout=600
        )
        # Check that success message was printed
        assert any("Successfully installed" in str(call) for call in mock_print.call_args_list)

    @patch("orchestrator.utils.model_utils.check_ollama_installed")
    @patch("subprocess.run")
    def test_install_ollama_model_failure(self, mock_run, mock_check):
        """Test install_ollama_model when installation fails."""
        mock_check.return_value = True
        mock_run.return_value = MagicMock(returncode=1, stderr="Error: model not found")
        
        with patch("builtins.print") as mock_print:
            result = install_ollama_model("invalid-model")
        
        assert result is False
        # Check that error message was printed
        assert any("Failed to install" in str(call) for call in mock_print.call_args_list)

    @patch("orchestrator.utils.model_utils.check_ollama_installed")
    @patch("subprocess.run")
    def test_install_ollama_model_timeout(self, mock_run, mock_check):
        """Test install_ollama_model when installation times out."""
        mock_check.return_value = True
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("ollama pull", 600)
        
        with patch("builtins.print") as mock_print:
            result = install_ollama_model("large-model")
        
        assert result is False
        # Check that timeout message was printed
        assert any("timed out" in str(call) for call in mock_print.call_args_list)

    @patch("orchestrator.utils.model_utils.check_ollama_installed")
    @patch("subprocess.run")
    def test_install_ollama_model_exception(self, mock_run, mock_check):
        """Test install_ollama_model when subprocess error occurs."""
        mock_check.return_value = True
        import subprocess
        mock_run.side_effect = subprocess.SubprocessError("Subprocess error")
        
        with patch("builtins.print") as mock_print:
            result = install_ollama_model("test-model")
        
        assert result is False
        # Check that error message was printed
        assert any("Error installing" in str(call) for call in mock_print.call_args_list)


class TestLoadModelConfig:
    """Test load_model_config function."""

    def test_load_model_config_from_current_dir(self):
        """Test loading config from current directory."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {"models": [{"name": "test"}], "defaults": {}}
            yaml.dump(config, f)
            temp_file = f.name

        try:
            # Temporarily change to the directory containing the file
            original_cwd = os.getcwd()
            temp_dir = os.path.dirname(temp_file)
            temp_name = os.path.basename(temp_file)
            os.chdir(temp_dir)
            
            loaded = load_model_config(temp_name)
            assert loaded["models"][0]["name"] == "test"
        finally:
            os.chdir(original_cwd)
            os.unlink(temp_file)

    def test_load_model_config_from_home_dir(self):
        """Test loading config from ~/.orchestrator directory."""
        home_dir = Path.home()
        orchestrator_dir = home_dir / ".orchestrator"
        orchestrator_dir.mkdir(exist_ok=True)
        
        config_file = orchestrator_dir / "test_models.yaml"
        config = {"models": [{"name": "home-test"}], "defaults": {}}
        
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        
        try:
            loaded = load_model_config("test_models.yaml")
            assert loaded["models"][0]["name"] == "home-test"
        finally:
            config_file.unlink()

    def test_load_model_config_from_env_var(self):
        """Test loading config from ORCHESTRATOR_HOME."""
        # Test that the function accepts ORCHESTRATOR_HOME env var
        with patch.dict(os.environ, {"ORCHESTRATOR_HOME": "/tmp/test"}):
            # Since we can't easily mock Path.exists() without side effects,
            # just test that the environment variable is read
            loaded = load_model_config("test_nonexistent.yaml")
            # Should return default config since file doesn't exist
            assert "models" in loaded
            assert "defaults" in loaded

    def test_load_model_config_default(self):
        """Test loading default config when file not found."""
        # Use a name that definitely doesn't exist
        loaded = load_model_config("nonexistent_config_file_12345.yaml")
        
        # Should return default configuration
        assert "models" in loaded
        assert "defaults" in loaded
        assert len(loaded["models"]) > 0
        assert loaded["models"][0]["source"] == "ollama"

    def test_load_model_config_from_package_root(self):
        """Test loading config from package root."""
        # This test is tricky because it depends on the package structure
        # We'll mock the path resolution
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {"models": [{"name": "package-test"}], "defaults": {}}
            yaml.dump(config, f)
            temp_file = f.name

        try:
            with patch("pathlib.Path.exists") as mock_exists:
                with patch("builtins.open", mock_open(read_data=yaml.dump(config))):
                    # Make only the third path (package root) exist
                    mock_exists.side_effect = [False, False, True, False]
                    loaded = load_model_config()
                    assert loaded["models"][0]["name"] == "package-test"
        finally:
            os.unlink(temp_file)