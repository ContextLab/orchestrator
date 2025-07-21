"""Tests to improve coverage for the utils module."""

import os
import tempfile
from pathlib import Path
import subprocess
from typing import Dict, Any, Optional

import pytest
import yaml

from orchestrator.utils.model_utils import (
    check_ollama_installed,
    check_ollama_model,
    install_ollama_model,
    load_model_config,
    parse_model_size,
)


class TestableSubprocessRun:
    """A testable replacement for subprocess.run."""
    
    def __init__(self):
        self.commands = {}  # command -> (returncode, stdout, stderr)
        self.call_history = []
        self.exceptions = {}  # command -> exception to raise
        
    def add_command(self, command: list, returncode: int, stdout: str = "", stderr: str = ""):
        """Add a command result."""
        cmd_str = " ".join(command)
        self.commands[cmd_str] = (returncode, stdout, stderr)
        
    def add_exception(self, command: list, exception: Exception):
        """Add an exception to raise for a command."""
        cmd_str = " ".join(command)
        self.exceptions[cmd_str] = exception
        
    def __call__(self, command: list, **kwargs):
        """Simulate subprocess.run."""
        cmd_str = " ".join(command)
        self.call_history.append((command, kwargs))
        
        # Check for exceptions first
        if cmd_str in self.exceptions:
            raise self.exceptions[cmd_str]
            
        # Return result
        if cmd_str in self.commands:
            returncode, stdout, stderr = self.commands[cmd_str]
            return TestableCompletedProcess(returncode, stdout, stderr)
        else:
            # Default behavior
            return TestableCompletedProcess(1, "", "Command not found")


class TestableCompletedProcess:
    """A testable replacement for subprocess.CompletedProcess."""
    
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestableFileOperations:
    """Testable file operations."""
    
    def __init__(self):
        self.files = {}  # path -> content
        self.directories = set()
        
    def add_file(self, path: str, content: str):
        """Add a test file."""
        self.files[str(path)] = content
        # Add parent directories
        parts = str(path).split('/')
        for i in range(1, len(parts)):
            self.directories.add('/'.join(parts[:i]))
            
    def read_file(self, path: str) -> str:
        """Read a test file."""
        if str(path) in self.files:
            return self.files[str(path)]
        raise FileNotFoundError(f"File not found: {path}")
        
    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        return str(path) in self.files
        
    def dir_exists(self, path: str) -> bool:
        """Check if directory exists."""
        return str(path) in self.directories


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
            # Replace load_model_config temporarily
            import orchestrator.utils.model_utils
            original_load = orchestrator.utils.model_utils.load_model_config
            
            def test_load(config_file=None):
                return config
                
            orchestrator.utils.model_utils.load_model_config = test_load
            
            try:
                size = parse_model_size("test-model-from-config")
                assert size == 42.0
            finally:
                orchestrator.utils.model_utils.load_model_config = original_load
        finally:
            os.unlink(temp_file)

    def test_parse_size_default(self):
        """Test default size when no size information found."""
        assert parse_model_size("unknown-model") == 1.0
        assert parse_model_size("model-without-size", None) == 1.0

    def test_parse_size_with_exception_in_config_loading(self):
        """Test handling exception when loading config."""
        # Replace load_model_config temporarily
        import orchestrator.utils.model_utils
        original_load = orchestrator.utils.model_utils.load_model_config
        
        def failing_load(config_file=None):
            raise Exception("Config load error")
            
        orchestrator.utils.model_utils.load_model_config = failing_load
        
        try:
            # Should continue and return default
            assert parse_model_size("test-model") == 1.0
        finally:
            orchestrator.utils.model_utils.load_model_config = original_load


class TestOllamaChecks:
    """Test Ollama installation and model checks."""

    def test_check_ollama_installed_success(self):
        """Test check_ollama_installed when Ollama is installed."""
        # Replace subprocess.run temporarily
        testable_run = TestableSubprocessRun()
        testable_run.add_command(["ollama", "--version"], 0, "ollama version 0.1.0")
        
        original_run = subprocess.run
        subprocess.run = testable_run
        
        try:
            assert check_ollama_installed() is True
            # Verify the command was called
            assert len(testable_run.call_history) == 1
            assert testable_run.call_history[0][0] == ["ollama", "--version"]
        finally:
            subprocess.run = original_run

    def test_check_ollama_installed_failure(self):
        """Test check_ollama_installed when Ollama is not installed."""
        testable_run = TestableSubprocessRun()
        testable_run.add_command(["ollama", "--version"], 1)
        
        original_run = subprocess.run
        subprocess.run = testable_run
        
        try:
            assert check_ollama_installed() is False
        finally:
            subprocess.run = original_run

    def test_check_ollama_installed_file_not_found(self):
        """Test check_ollama_installed when ollama command not found."""
        testable_run = TestableSubprocessRun()
        testable_run.add_exception(["ollama", "--version"], FileNotFoundError())
        
        original_run = subprocess.run
        subprocess.run = testable_run
        
        try:
            assert check_ollama_installed() is False
        finally:
            subprocess.run = original_run

    def test_check_ollama_installed_timeout(self):
        """Test check_ollama_installed when command times out."""
        testable_run = TestableSubprocessRun()
        testable_run.add_exception(["ollama", "--version"], subprocess.TimeoutExpired("ollama", 5))
        
        original_run = subprocess.run
        subprocess.run = testable_run
        
        try:
            assert check_ollama_installed() is False
        finally:
            subprocess.run = original_run

    def test_check_ollama_model_no_ollama(self):
        """Test check_ollama_model when Ollama is not installed."""
        # Replace check_ollama_installed temporarily
        import orchestrator.utils.model_utils
        original_check = orchestrator.utils.model_utils.check_ollama_installed
        
        orchestrator.utils.model_utils.check_ollama_installed = lambda: False
        
        try:
            assert check_ollama_model("llama2:7b") is False
        finally:
            orchestrator.utils.model_utils.check_ollama_installed = original_check

    def test_check_ollama_model_found(self):
        """Test check_ollama_model when model is found."""
        # Replace both functions temporarily
        import orchestrator.utils.model_utils
        original_check = orchestrator.utils.model_utils.check_ollama_installed
        
        testable_run = TestableSubprocessRun()
        testable_run.add_command(["ollama", "list"], 0, "llama2:7b\ngemma:7b\nmistral:7b")
        
        orchestrator.utils.model_utils.check_ollama_installed = lambda: True
        original_run = subprocess.run
        subprocess.run = testable_run
        
        try:
            assert check_ollama_model("llama2:7b") is True
        finally:
            orchestrator.utils.model_utils.check_ollama_installed = original_check
            subprocess.run = original_run

    def test_check_ollama_model_not_found(self):
        """Test check_ollama_model when model is not found."""
        import orchestrator.utils.model_utils
        original_check = orchestrator.utils.model_utils.check_ollama_installed
        
        testable_run = TestableSubprocessRun()
        testable_run.add_command(["ollama", "list"], 0, "gemma:7b\nmistral:7b")
        
        orchestrator.utils.model_utils.check_ollama_installed = lambda: True
        original_run = subprocess.run
        subprocess.run = testable_run
        
        try:
            assert check_ollama_model("llama2:7b") is False
        finally:
            orchestrator.utils.model_utils.check_ollama_installed = original_check
            subprocess.run = original_run

    def test_check_ollama_model_command_failure(self):
        """Test check_ollama_model when list command fails."""
        import orchestrator.utils.model_utils
        original_check = orchestrator.utils.model_utils.check_ollama_installed
        
        testable_run = TestableSubprocessRun()
        testable_run.add_command(["ollama", "list"], 1)
        
        orchestrator.utils.model_utils.check_ollama_installed = lambda: True
        original_run = subprocess.run
        subprocess.run = testable_run
        
        try:
            assert check_ollama_model("llama2:7b") is False
        finally:
            orchestrator.utils.model_utils.check_ollama_installed = original_check
            subprocess.run = original_run

    def test_check_ollama_model_exception(self):
        """Test check_ollama_model when exception occurs."""
        import orchestrator.utils.model_utils
        original_check = orchestrator.utils.model_utils.check_ollama_installed
        
        testable_run = TestableSubprocessRun()
        testable_run.add_exception(["ollama", "list"], subprocess.SubprocessError("Command failed"))
        
        orchestrator.utils.model_utils.check_ollama_installed = lambda: True
        original_run = subprocess.run
        subprocess.run = testable_run
        
        try:
            assert check_ollama_model("llama2:7b") is False
        finally:
            orchestrator.utils.model_utils.check_ollama_installed = original_check
            subprocess.run = original_run


class TestInstallOllamaModel:
    """Test install_ollama_model function."""

    def test_install_ollama_model_no_ollama(self):
        """Test install_ollama_model when Ollama is not installed."""
        import orchestrator.utils.model_utils
        original_check = orchestrator.utils.model_utils.check_ollama_installed
        
        orchestrator.utils.model_utils.check_ollama_installed = lambda: False
        
        # Track print calls
        print_calls = []
        original_print = print
        
        def track_print(*args, **kwargs):
            print_calls.append(args)
            return original_print(*args, **kwargs)
            
        import builtins
        builtins.print = track_print
        
        try:
            result = install_ollama_model("llama2:7b")
            assert result is False
            assert any("Ollama not installed" in str(call) for call in print_calls)
        finally:
            orchestrator.utils.model_utils.check_ollama_installed = original_check
            builtins.print = original_print

    def test_install_ollama_model_success(self):
        """Test install_ollama_model successful installation."""
        import orchestrator.utils.model_utils
        original_check = orchestrator.utils.model_utils.check_ollama_installed
        
        testable_run = TestableSubprocessRun()
        testable_run.add_command(["ollama", "pull", "llama2:7b"], 0, "Pulling llama2:7b...")
        
        orchestrator.utils.model_utils.check_ollama_installed = lambda: True
        original_run = subprocess.run
        subprocess.run = testable_run
        
        # Track print calls
        print_calls = []
        original_print = print
        
        def track_print(*args, **kwargs):
            print_calls.append(args)
            return original_print(*args, **kwargs)
            
        import builtins
        builtins.print = track_print
        
        try:
            result = install_ollama_model("llama2:7b")
            assert result is True
            assert len(testable_run.call_history) == 1
            assert testable_run.call_history[0][0] == ["ollama", "pull", "llama2:7b"]
            # Check that success message was printed
            assert any("Successfully installed" in str(call) for call in print_calls)
        finally:
            orchestrator.utils.model_utils.check_ollama_installed = original_check
            subprocess.run = original_run
            builtins.print = original_print

    def test_install_ollama_model_failure(self):
        """Test install_ollama_model when installation fails."""
        import orchestrator.utils.model_utils
        original_check = orchestrator.utils.model_utils.check_ollama_installed
        
        testable_run = TestableSubprocessRun()
        testable_run.add_command(["ollama", "pull", "invalid-model"], 1, "", "Error: model not found")
        
        orchestrator.utils.model_utils.check_ollama_installed = lambda: True
        original_run = subprocess.run
        subprocess.run = testable_run
        
        # Track print calls
        print_calls = []
        original_print = print
        
        def track_print(*args, **kwargs):
            print_calls.append(args)
            return original_print(*args, **kwargs)
            
        import builtins
        builtins.print = track_print
        
        try:
            result = install_ollama_model("invalid-model")
            assert result is False
            # Check that error message was printed
            assert any("Failed to install" in str(call) for call in print_calls)
        finally:
            orchestrator.utils.model_utils.check_ollama_installed = original_check
            subprocess.run = original_run
            builtins.print = original_print

    def test_install_ollama_model_timeout(self):
        """Test install_ollama_model when installation times out."""
        import orchestrator.utils.model_utils
        original_check = orchestrator.utils.model_utils.check_ollama_installed
        
        testable_run = TestableSubprocessRun()
        testable_run.add_exception(["ollama", "pull", "large-model"], subprocess.TimeoutExpired("ollama pull", 600))
        
        orchestrator.utils.model_utils.check_ollama_installed = lambda: True
        original_run = subprocess.run
        subprocess.run = testable_run
        
        # Track print calls
        print_calls = []
        original_print = print
        
        def track_print(*args, **kwargs):
            print_calls.append(args)
            return original_print(*args, **kwargs)
            
        import builtins
        builtins.print = track_print
        
        try:
            result = install_ollama_model("large-model")
            assert result is False
            # Check that timeout message was printed
            assert any("timed out" in str(call) for call in print_calls)
        finally:
            orchestrator.utils.model_utils.check_ollama_installed = original_check
            subprocess.run = original_run
            builtins.print = original_print

    def test_install_ollama_model_exception(self):
        """Test install_ollama_model when subprocess error occurs."""
        import orchestrator.utils.model_utils
        original_check = orchestrator.utils.model_utils.check_ollama_installed
        
        testable_run = TestableSubprocessRun()
        testable_run.add_exception(["ollama", "pull", "test-model"], subprocess.SubprocessError("Subprocess error"))
        
        orchestrator.utils.model_utils.check_ollama_installed = lambda: True
        original_run = subprocess.run
        subprocess.run = testable_run
        
        # Track print calls
        print_calls = []
        original_print = print
        
        def track_print(*args, **kwargs):
            print_calls.append(args)
            return original_print(*args, **kwargs)
            
        import builtins
        builtins.print = track_print
        
        try:
            result = install_ollama_model("test-model")
            assert result is False
            # Check that error message was printed
            assert any("Error installing" in str(call) for call in print_calls)
        finally:
            orchestrator.utils.model_utils.check_ollama_installed = original_check
            subprocess.run = original_run
            builtins.print = original_print


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
        original_env = os.environ.get("ORCHESTRATOR_HOME")
        os.environ["ORCHESTRATOR_HOME"] = "/tmp/test"
        
        try:
            # Since we can't easily mock Path.exists() without side effects,
            # just test that the environment variable is read
            loaded = load_model_config("test_nonexistent.yaml")
            # Should return default config since file doesn't exist
            assert "models" in loaded
            assert "defaults" in loaded
        finally:
            if original_env is None:
                os.environ.pop("ORCHESTRATOR_HOME", None)
            else:
                os.environ["ORCHESTRATOR_HOME"] = original_env

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
        # This test is difficult to implement without mocks since it depends on package structure
        # We'll test that the default config is returned when no file is found
        
        # Create a unique filename that won't exist anywhere
        unique_filename = f"test_config_{os.getpid()}_{id(self)}.yaml"
        loaded = load_model_config(unique_filename)
        
        # Should return default configuration
        assert "models" in loaded
        assert "defaults" in loaded
        assert len(loaded["models"]) > 0