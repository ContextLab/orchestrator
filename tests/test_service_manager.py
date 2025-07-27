"""Test service management utilities."""

import pytest
from unittest.mock import patch, MagicMock, call
import subprocess

from orchestrator.utils.service_manager import (
    ServiceManager,
    OllamaServiceManager,
    DockerServiceManager,
    ensure_service_running,
    get_service_status,
    register_service_manager,
)


class MockServiceManager(ServiceManager):
    """Mock service manager for testing."""
    
    def __init__(self, installed=True, running=False):
        self.installed = installed
        self.running = running
        self.start_called = False
        self.stop_called = False
    
    def is_installed(self) -> bool:
        return self.installed
    
    def is_running(self) -> bool:
        return self.running
    
    def start(self) -> bool:
        self.start_called = True
        self.running = True
        return True
    
    def stop(self) -> bool:
        self.stop_called = True
        self.running = False
        return True


class TestServiceManager:
    """Test base ServiceManager functionality."""
    
    def test_ensure_running_already_running(self):
        """Test ensure_running when service is already running."""
        manager = MockServiceManager(installed=True, running=True)
        
        result = manager.ensure_running()
        
        assert result is True
        assert not manager.start_called
    
    def test_ensure_running_not_installed(self):
        """Test ensure_running when service is not installed."""
        manager = MockServiceManager(installed=False, running=False)
        
        result = manager.ensure_running()
        
        assert result is False
        assert not manager.start_called
    
    def test_ensure_running_needs_start(self):
        """Test ensure_running when service needs to be started."""
        manager = MockServiceManager(installed=True, running=False)
        
        result = manager.ensure_running()
        
        assert result is True
        assert manager.start_called
        assert manager.running


class TestOllamaServiceManager:
    """Test Ollama service management."""
    
    @patch('subprocess.run')
    def test_is_installed_true(self, mock_run):
        """Test checking if Ollama is installed."""
        mock_run.return_value = MagicMock(returncode=0)
        
        manager = OllamaServiceManager()
        assert manager.is_installed() is True
        
        mock_run.assert_called_once_with(
            ["which", "ollama"],
            capture_output=True,
            text=True,
            timeout=1
        )
    
    @patch('subprocess.run')
    def test_is_installed_false(self, mock_run):
        """Test checking when Ollama is not installed."""
        mock_run.return_value = MagicMock(returncode=1)
        
        manager = OllamaServiceManager()
        assert manager.is_installed() is False
    
    @patch('requests.get')
    def test_is_running_true(self, mock_get):
        """Test checking if Ollama is running."""
        mock_get.return_value = MagicMock(status_code=200)
        
        manager = OllamaServiceManager()
        assert manager.is_running() is True
        
        mock_get.assert_called_once_with(
            "http://localhost:11434/api/tags",
            timeout=1
        )
    
    @patch('requests.get')
    def test_is_running_false(self, mock_get):
        """Test checking when Ollama is not running."""
        mock_get.side_effect = Exception("Connection refused")
        
        manager = OllamaServiceManager()
        assert manager.is_running() is False
    
    @patch('subprocess.Popen')
    def test_start_success(self, mock_popen):
        """Test starting Ollama successfully."""
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        manager = OllamaServiceManager()
        
        # Mock is_running to return True after "starting"
        with patch.object(manager, 'is_running') as mock_running:
            mock_running.side_effect = [False, False, True]
            
            result = manager.start()
            
            assert result is True
            mock_popen.assert_called_once()
            assert "ollama" in mock_popen.call_args[0][0]
            assert "serve" in mock_popen.call_args[0][0]
    
    @patch('subprocess.run')
    def test_stop(self, mock_run):
        """Test stopping Ollama."""
        manager = OllamaServiceManager()
        
        with patch.object(manager, 'is_running') as mock_running:
            mock_running.return_value = False
            
            result = manager.stop()
            
            assert result is True
            # Should have called pkill
            assert mock_run.call_count >= 1
            assert "pkill" in mock_run.call_args_list[0][0][0]


class TestDockerServiceManager:
    """Test Docker service management."""
    
    @patch('subprocess.run')
    def test_is_installed_true(self, mock_run):
        """Test checking if Docker is installed."""
        mock_run.return_value = MagicMock(returncode=0)
        
        manager = DockerServiceManager()
        assert manager.is_installed() is True
        
        mock_run.assert_called_once_with(
            ["docker", "--version"],
            capture_output=True,
            timeout=1
        )
    
    @patch('subprocess.run')
    def test_is_running_true(self, mock_run):
        """Test checking if Docker is running."""
        mock_run.return_value = MagicMock(returncode=0)
        
        manager = DockerServiceManager()
        assert manager.is_running() is True
        
        mock_run.assert_called_once_with(
            ["docker", "info"],
            capture_output=True,
            timeout=2
        )
    
    @patch('os.path.exists')
    @patch('subprocess.run')
    def test_start_linux(self, mock_run, mock_exists):
        """Test starting Docker on Linux."""
        # Mock systemctl exists
        mock_exists.side_effect = lambda path: path == "/usr/bin/systemctl"
        mock_run.return_value = MagicMock(returncode=0)
        
        manager = DockerServiceManager()
        
        with patch.object(manager, 'is_running') as mock_running:
            mock_running.return_value = True
            
            result = manager.start()
            
            assert result is True
            mock_run.assert_called_with(
                ["sudo", "systemctl", "start", "docker"],
                capture_output=True
            )


class TestServiceRegistry:
    """Test service registry functions."""
    
    def test_ensure_service_running_known_service(self):
        """Test ensuring a known service is running."""
        mock_manager = MockServiceManager(installed=True, running=False)
        
        with patch('orchestrator.utils.service_manager.SERVICE_MANAGERS', {"test": mock_manager}):
            result = ensure_service_running("test")
            
            assert result is True
            assert mock_manager.start_called
    
    def test_ensure_service_running_unknown_service(self):
        """Test ensuring an unknown service returns False."""
        result = ensure_service_running("unknown_service_xyz")
        assert result is False
    
    def test_register_service_manager(self):
        """Test registering a custom service manager."""
        from orchestrator.utils.service_manager import SERVICE_MANAGERS
        
        mock_manager = MockServiceManager()
        register_service_manager("custom", mock_manager)
        
        assert "custom" in SERVICE_MANAGERS
        assert SERVICE_MANAGERS["custom"] is mock_manager
        
        # Clean up
        del SERVICE_MANAGERS["custom"]
    
    def test_get_service_status_known_service(self):
        """Test getting status of a known service."""
        mock_manager = MockServiceManager(installed=True, running=True)
        
        with patch('orchestrator.utils.service_manager.SERVICE_MANAGERS', {"test": mock_manager}):
            status = get_service_status("test")
            
            assert status["installed"] is True
            assert status["running"] is True
    
    def test_get_service_status_unknown_service(self):
        """Test getting status of an unknown service."""
        status = get_service_status("unknown_service_xyz")
        
        assert status["installed"] is False
        assert status["running"] is False