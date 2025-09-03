"""Tests for enhanced service management capabilities - Phase 2."""

import pytest
import asyncio
import subprocess
from unittest.mock import patch, MagicMock

from src.orchestrator.utils.service_manager import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    OllamaServiceManager, 
    DockerServiceManager,
    SERVICE_MANAGERS
)


class TestEnhancedOllamaServiceManager:
    """Test enhanced Ollama service manager with model download capabilities."""
    
    def test_ollama_manager_initialization(self):
        """Test OllamaServiceManager initialization with new parameters."""
        manager = OllamaServiceManager(base_url="http://localhost:11434", timeout=60)
        
        assert manager.base_url == "http://localhost:11434"
        assert manager.timeout == 60
        assert manager._model_cache is None
        assert manager._cache_ttl == 300
        
    @patch('requests.get')
    def test_get_available_models_success(self, mock_get):
        """Test getting available models with successful response."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:1b"},
                {"name": "llama3.2:3b"},
                {"name": "gemma2:9b"}
            ]
        }
        mock_get.return_value = mock_response
        
        manager = OllamaServiceManager()
        models = manager.get_available_models()
        
        assert len(models) == 3
        assert "llama3.2:1b" in models
        assert "llama3.2:3b" in models
        assert "gemma2:9b" in models
        
        # Test caching - second call shouldn't make new request
        models2 = manager.get_available_models()
        assert models == models2
        assert mock_get.call_count == 1
        
    @patch('requests.get')
    def test_get_available_models_cache_refresh(self, mock_get):
        """Test cache refresh functionality."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "test:model"}]}
        mock_get.return_value = mock_response
        
        manager = OllamaServiceManager()
        
        # First call
        models1 = manager.get_available_models()
        
        # Force refresh
        models2 = manager.get_available_models(force_refresh=True)
        
        assert mock_get.call_count == 2
        assert models1 == models2
        
    def test_is_model_available(self):
        """Test checking if specific model is available."""
        manager = OllamaServiceManager()
        manager._model_cache = ["llama3.2:1b", "gemma2:9b"]
        
        assert manager.is_model_available("llama3.2:1b") is True
        assert manager.is_model_available("nonexistent:model") is False
        
    @patch('subprocess.run')
    @patch.object(OllamaServiceManager, 'ensure_running')
    @patch.object(OllamaServiceManager, 'is_model_available')
    def test_ensure_model_available_already_exists(self, mock_is_available, mock_ensure_running, mock_subprocess):
        """Test ensuring model availability when model already exists."""
        mock_ensure_running.return_value = True
        mock_is_available.return_value = True
        
        manager = OllamaServiceManager()
        result = manager.ensure_model_available("llama3.2:1b")
        
        assert result is True
        mock_subprocess.assert_not_called()  # Shouldn't try to pull
        
    @patch('subprocess.run')
    @patch.object(OllamaServiceManager, 'ensure_running')
    @patch.object(OllamaServiceManager, 'is_model_available')
    def test_ensure_model_available_needs_pull(self, mock_is_available, mock_ensure_running, mock_subprocess):
        """Test ensuring model availability when model needs to be pulled."""
        mock_ensure_running.return_value = True
        mock_is_available.return_value = False
        
        # Mock successful pull
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        manager = OllamaServiceManager()
        result = manager.ensure_model_available("llama3.2:1b")
        
        assert result is True
        mock_subprocess.assert_called_once_with(
            ["ollama", "pull", "llama3.2:1b"],
            capture_output=True,
            text=True,
            timeout=600
        )
        
    @patch('subprocess.run')
    def test_pull_model_success(self, mock_subprocess):
        """Test successful model pulling."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        manager = OllamaServiceManager()
        result = manager.pull_model("llama3.2:1b")
        
        assert result is True
        assert manager._model_cache is None  # Cache should be invalidated
        
    @patch('subprocess.run')
    def test_pull_model_failure(self, mock_subprocess):
        """Test model pulling failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Model not found"
        mock_subprocess.return_value = mock_result
        
        manager = OllamaServiceManager()
        result = manager.pull_model("nonexistent:model")
        
        assert result is False
        
    @patch('requests.get')
    def test_get_model_info_success(self, mock_get):
        """Test getting model information."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama3.2:1b",
                    "size": 1234567890,
                    "modified_at": "2024-01-01T00:00:00Z"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        manager = OllamaServiceManager()
        info = manager.get_model_info("llama3.2:1b")
        
        assert info is not None
        assert info["name"] == "llama3.2:1b"
        assert info["size"] == 1234567890
        
    @pytest.mark.asyncio
    @patch('requests.post')
    @patch.object(OllamaServiceManager, 'is_running')
    @patch.object(OllamaServiceManager, 'is_model_available')
    async def test_health_check_model_success(self, mock_is_available, mock_is_running, mock_post):
        """Test successful model health check."""
        mock_is_running.return_value = True
        mock_is_available.return_value = True
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        manager = OllamaServiceManager()
        result = await manager.health_check_model("llama3.2:1b")
        
        assert result is True


class TestEnhancedDockerServiceManager:
    """Test enhanced Docker service manager with container model support."""
    
    def test_docker_manager_initialization(self):
        """Test DockerServiceManager initialization with new parameters."""
        manager = DockerServiceManager(timeout=60)
        
        assert manager.timeout == 60
        assert manager._container_cache == {}
        assert manager._cache_ttl == 300
        
    @patch('subprocess.run')
    def test_get_running_containers_success(self, mock_subprocess):
        """Test getting running containers with successful response."""
        # Mock docker ps output (JSON format)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"Names": "test-container", "Image": "test:image", "Status": "Up 1 hour"}\n'
        mock_subprocess.return_value = mock_result
        
        manager = DockerServiceManager()
        containers = manager.get_running_containers()
        
        assert len(containers) == 1
        assert "test-container" in containers
        assert containers["test-container"]["Image"] == "test:image"
        
    def test_is_container_running(self):
        """Test checking if specific container is running."""
        manager = DockerServiceManager()
        manager._container_cache = {
            "test-container": {"Status": "Up 1 hour"}
        }
        
        # Mock get_running_containers to return cached data
        with patch.object(manager, 'get_running_containers', return_value=manager._container_cache):
            assert manager.is_container_running("test-container") is True
            assert manager.is_container_running("nonexistent-container") is False
        
    @patch('subprocess.run')
    @patch.object(DockerServiceManager, 'ensure_running')
    @patch.object(DockerServiceManager, 'is_container_running')
    def test_ensure_container_running_already_running(self, mock_is_running, mock_ensure_running, mock_subprocess):
        """Test ensuring container is running when already running."""
        mock_ensure_running.return_value = True
        mock_is_running.return_value = True
        
        manager = DockerServiceManager()
        config = {"name": "test-container", "image": "test:image"}
        result = manager.ensure_container_running(config)
        
        assert result is True
        mock_subprocess.assert_not_called()  # Shouldn't try to start or run
        
    @patch('subprocess.run')
    @patch.object(DockerServiceManager, 'ensure_running')
    @patch.object(DockerServiceManager, 'is_container_running')
    @patch.object(DockerServiceManager, 'start_container')
    def test_ensure_container_running_needs_start(self, mock_start, mock_is_running, mock_ensure_running, mock_subprocess):
        """Test ensuring container is running when it needs to be started."""
        mock_ensure_running.return_value = True
        mock_is_running.return_value = False
        mock_start.return_value = True
        
        manager = DockerServiceManager()
        config = {"name": "test-container", "image": "test:image"}
        result = manager.ensure_container_running(config)
        
        assert result is True
        mock_start.assert_called_once_with("test-container")
        
    @patch('subprocess.run')
    def test_run_container_success(self, mock_subprocess):
        """Test successful container creation and running."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        manager = DockerServiceManager()
        config = {
            "name": "test-container",
            "image": "test:image",
            "ports": ["8080:8080"],
            "environment": ["ENV_VAR=value"],
            "volumes": ["/host/path:/container/path"],
            "args": ["--verbose"]
        }
        
        result = manager.run_container(config)
        
        assert result is True
        
        # Verify command construction
        expected_cmd = [
            "docker", "run", "-d", "--name", "test-container",
            "-p", "8080:8080",
            "-e", "ENV_VAR=value", 
            "-v", "/host/path:/container/path",
            "test:image",
            "--verbose"
        ]
        mock_subprocess.assert_called_once_with(
            expected_cmd,
            capture_output=True,
            text=True,
            timeout=90  # 30 * 3
        )
        
    @pytest.mark.asyncio
    @patch('requests.get')
    @patch.object(DockerServiceManager, 'is_container_running')
    async def test_health_check_container_with_endpoint(self, mock_is_running, mock_get):
        """Test container health check with HTTP endpoint."""
        mock_is_running.return_value = True
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        manager = DockerServiceManager()
        result = await manager.health_check_container(
            "test-container", 
            "http://localhost:8080/health"
        )
        
        assert result is True


class TestServiceManagerIntegration:
    """Test integration of enhanced service managers."""
    
    def test_service_managers_registry_updated(self):
        """Test that SERVICE_MANAGERS contains enhanced managers."""
        assert "ollama" in SERVICE_MANAGERS
        assert "docker" in SERVICE_MANAGERS
        
        assert isinstance(SERVICE_MANAGERS["ollama"], OllamaServiceManager)
        assert isinstance(SERVICE_MANAGERS["docker"], DockerServiceManager)
        
    def test_ollama_manager_has_enhanced_methods(self):
        """Test that Ollama manager has all enhanced methods."""
        manager = SERVICE_MANAGERS["ollama"]
        
        # Check enhanced methods exist
        assert hasattr(manager, 'get_available_models')
        assert hasattr(manager, 'is_model_available')
        assert hasattr(manager, 'ensure_model_available')
        assert hasattr(manager, 'pull_model')
        assert hasattr(manager, 'remove_model')
        assert hasattr(manager, 'get_model_info')
        assert hasattr(manager, 'health_check_model')
        
    def test_docker_manager_has_enhanced_methods(self):
        """Test that Docker manager has all enhanced methods."""
        manager = SERVICE_MANAGERS["docker"]
        
        # Check enhanced methods exist
        assert hasattr(manager, 'get_running_containers')
        assert hasattr(manager, 'is_container_running')
        assert hasattr(manager, 'ensure_container_running')
        assert hasattr(manager, 'run_container')
        assert hasattr(manager, 'start_container')
        assert hasattr(manager, 'stop_container')
        assert hasattr(manager, 'remove_container')
        assert hasattr(manager, 'get_container_logs')
        assert hasattr(manager, 'health_check_container')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])