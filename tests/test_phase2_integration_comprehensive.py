"""Comprehensive integration test for Issue #202 Phase 2 implementation."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.utils.service_manager import SERVICE_MANAGERS

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider


class TestPhase2ComprehensiveIntegration:
    """Test complete Phase 2 integration as specified in Issues #202 and #199."""
    
    def test_phase2_service_integration_complete(self):
        """Test that all Phase 2 service integration features are implemented."""
        
        # 2.1 Enhanced OllamaServiceManager with model download capabilities
        ollama_manager = SERVICE_MANAGERS["ollama"]
        
        # Verify enhanced Ollama capabilities
        assert hasattr(ollama_manager, 'get_available_models')
        assert hasattr(ollama_manager, 'is_model_available') 
        assert hasattr(ollama_manager, 'ensure_model_available')
        assert hasattr(ollama_manager, 'pull_model')
        assert hasattr(ollama_manager, 'remove_model')
        assert hasattr(ollama_manager, 'get_model_info')
        assert hasattr(ollama_manager, 'health_check_model')
        
        # 2.2 Extended DockerServiceManager for containerized models
        docker_manager = SERVICE_MANAGERS["docker"]
        
        # Verify enhanced Docker capabilities
        assert hasattr(docker_manager, 'get_running_containers')
        assert hasattr(docker_manager, 'is_container_running')
        assert hasattr(docker_manager, 'ensure_container_running')
        assert hasattr(docker_manager, 'run_container')
        assert hasattr(docker_manager, 'start_container')
        assert hasattr(docker_manager, 'stop_container')
        assert hasattr(docker_manager, 'remove_container')
        assert hasattr(docker_manager, 'get_container_logs')
        assert hasattr(docker_manager, 'health_check_container')
        
        # 2.3 Health monitoring integration
        # Both managers have health check capabilities for models/containers
        assert callable(ollama_manager.health_check_model)
        assert callable(docker_manager.health_check_container)
    
    def test_phase2_registry_integration_complete(self):
        """Test that ModelRegistry LangChain integration preserves all existing functionality."""
        
        registry = ModelRegistry()
        
        # 2.4 Registry Integration - enhance ModelRegistry to support LangChain adapters
        assert hasattr(registry, '_langchain_adapters')
        assert hasattr(registry, '_langchain_enabled')
        assert hasattr(registry, 'register_langchain_model')
        assert hasattr(registry, 'unregister_langchain_model')
        assert hasattr(registry, 'get_langchain_adapters')
        assert hasattr(registry, 'is_langchain_model')
        assert hasattr(registry, 'auto_register_langchain_models')
        
        # 2.5 Preserve UCB selection algorithm and advanced caching logic
        assert hasattr(registry, 'model_selector')
        assert hasattr(registry.model_selector, 'select')
        assert hasattr(registry.model_selector, 'update_reward')
        assert hasattr(registry, 'cache_manager')
        assert hasattr(registry, 'memory_monitor')
        
        # 2.6 Intelligent model selection for LangChain providers
        assert hasattr(registry, '_ensure_service_running')
        assert hasattr(registry, '_auto_install_dependencies')
        
    @patch.object(ModelRegistry, 'register_langchain_model')
    def test_auto_registration_workflow_complete(self, mock_register):
        """Test complete auto-registration workflow as specified in Issue #202."""
        
        registry = ModelRegistry()
        mock_register.side_effect = ["openai:gpt-4-turbo", "anthropic:claude-3-sonnet", "ollama:llama3.2:3b"]
        
        # Configuration matching Issue #202 examples
        config = {
            "models": [
                {
                    "provider": "openai",  # User-facing provider name preserved
                    "model": "gpt-4-turbo",
                    "auto_install": True  # Uses existing auto_install.py
                },
                {
                    "provider": "anthropic",  # User-facing provider name preserved
                    "model": "claude-3-sonnet",
                    "auto_install": True
                },
                {
                    "provider": "ollama",
                    "model": "llama3.2:3b",
                    "ensure_running": True,  # Uses existing service_manager.py
                    "auto_pull": True
                }
            ]
        }
        
        with patch.object(registry, '_ensure_service_running') as mock_service:
            with patch.object(registry, '_auto_install_dependencies') as mock_install:
                registered_keys = registry.auto_register_langchain_models(config)
                
                # Verify all models registered
                assert len(registered_keys) == 3
                
                # Verify auto-installation called for providers with flag
                mock_install.assert_any_call("openai")
                mock_install.assert_any_call("anthropic")
                
                # Verify service startup called for Ollama
                mock_service.assert_called_once_with("ollama")
    
    @patch('subprocess.run')
    @patch('requests.get')
    def test_ollama_integration_workflow(self, mock_requests, mock_subprocess):
        """Test complete Ollama model integration workflow."""
        
        # Mock Ollama API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama3.2:1b"}, {"name": "gemma2:9b"}]
        }
        mock_requests.return_value = mock_response
        
        # Mock successful model pull
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result
        
        ollama_manager = SERVICE_MANAGERS["ollama"]
        
        # Test complete workflow: check availability -> pull if needed -> verify
        available_models = ollama_manager.get_available_models()
        assert len(available_models) == 2
        
        # Test model that needs to be pulled
        if not ollama_manager.is_model_available("llama3.2:3b"):
            success = ollama_manager.pull_model("llama3.2:3b")
            assert success is True
            
        # Verify the CLI was called correctly
        mock_subprocess.assert_called_with(
            ["ollama", "pull", "llama3.2:3b"],
            capture_output=True,
            text=True,
            timeout=600
        )
    
    @patch('subprocess.run')  
    def test_docker_integration_workflow(self, mock_subprocess):
        """Test complete Docker container integration workflow."""
        
        # Mock successful container operations
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"Names": "model-server", "Image": "huggingface/transformers:latest"}\n'
        mock_subprocess.return_value = mock_result
        
        docker_manager = SERVICE_MANAGERS["docker"]
        
        # Test complete workflow: check containers -> run if needed -> verify
        containers = docker_manager.get_running_containers()
        
        container_config = {
            "name": "model-server",
            "image": "huggingface/transformers:latest", 
            "ports": ["8080:8080"],
            "environment": ["MODEL_NAME=gpt2"],
            "volumes": ["/models:/app/models"]
        }
        
        # This should check if running, and create if not
        success = docker_manager.ensure_container_running(container_config)
        
        # Should have called docker ps at minimum
        assert mock_subprocess.called
        
    def test_ucb_algorithm_preservation(self):
        """Test that UCB selection algorithm works identically with LangChain models."""
        
        registry = ModelRegistry()
        
        # Mock LangChain model
        mock_langchain_model = MagicMock()
        mock_langchain_model.metrics = MagicMock()
        mock_langchain_model.metrics.success_rate = 0.85
        
        # Mock regular model  
        mock_regular_model = MagicMock()
        mock_regular_model.metrics = MagicMock()
        mock_regular_model.metrics.success_rate = 0.75
        
        # Initialize both in UCB selector
        registry.model_selector.initialize_model("langchain:model", mock_langchain_model.metrics)
        registry.model_selector.initialize_model("regular:model", mock_regular_model.metrics)
        
        # Test UCB selection works with both types
        available_models = ["langchain:model", "regular:model"]
        selected = registry.model_selector.select(available_models, {})
        
        # Should select one of the available models
        assert selected in available_models
        
        # Test reward update works
        registry.model_selector.update_reward(selected, 0.9)
        
        # Verify stats were updated
        stats = registry.model_selector.model_stats[selected]
        assert stats["total_reward"] == 0.9
        assert stats["successes"] == 1
    
    def test_advanced_caching_preservation(self):
        """Test that advanced caching works with LangChain models."""
        
        registry = ModelRegistry(enable_advanced_caching=True)
        
        # Verify advanced caching infrastructure is preserved
        assert registry._advanced_caching_enabled is True
        assert registry.cache_manager is not None
        
        # Mock cache operations
        with patch.object(registry.cache_manager, 'invalidate_model') as mock_invalidate:
            # LangChain model registration should trigger cache invalidation
            mock_adapter = MagicMock()
            
            with patch.object(registry, '_get_model_key', return_value="openai:gpt-4"):
                with patch('src.orchestrator.models.langchain_adapter.LangChainModelAdapter.__new__', return_value=mock_adapter):
                    registry.register_langchain_model("openai", "gpt-4")
                    
                    # Verify cache invalidation called (preserving existing caching logic)
                    mock_invalidate.assert_called_once_with("openai:gpt-4")
    
    def test_memory_optimization_preservation(self):
        """Test that memory optimization works with LangChain models."""
        
        registry = ModelRegistry(enable_memory_optimization=True)
        
        # Verify memory optimization infrastructure is preserved
        assert registry._memory_optimization_enabled is True  
        assert registry.memory_monitor is not None
        
        # Mock memory operations
        with patch.object(registry.memory_monitor, 'check_memory') as mock_check:
            # LangChain model registration should trigger memory check
            mock_adapter = MagicMock()
            
            with patch.object(registry, '_get_model_key', return_value="anthropic:claude-3"):
                with patch('src.orchestrator.models.langchain_adapter.LangChainModelAdapter.__new__', return_value=mock_adapter):
                    registry.register_langchain_model("anthropic", "claude-3")
                    
                    # Verify memory check called (preserving existing memory logic)
                    mock_check.assert_called()
    
    def test_issue_202_requirements_complete(self):
        """Verify all Issue #202 requirements are met."""
        
        # ✅ Auto-installation system extended with LangChain packages
        from src.orchestrator.utils.auto_install import PACKAGE_MAPPINGS
        assert "langchain_openai" in PACKAGE_MAPPINGS
        assert "langchain_anthropic" in PACKAGE_MAPPINGS
        assert "langchain_google_genai" in PACKAGE_MAPPINGS
        assert "langchain_community" in PACKAGE_MAPPINGS
        assert "langchain_huggingface" in PACKAGE_MAPPINGS
        
        # ✅ Service management enhanced for Ollama and Docker
        assert hasattr(SERVICE_MANAGERS["ollama"], 'ensure_model_available')
        assert hasattr(SERVICE_MANAGERS["docker"], 'ensure_container_running')
        
        # ✅ ModelRegistry enhanced to support LangChain adapters
        registry = ModelRegistry()
        assert hasattr(registry, 'register_langchain_model')
        assert hasattr(registry, 'auto_register_langchain_models')
        
        # ✅ UCB selection algorithm preserved
        assert hasattr(registry, 'model_selector')
        assert hasattr(registry.model_selector, 'select')
        assert registry.model_selector.exploration_factor == 2.0
        
        # ✅ Advanced caching preserved  
        assert hasattr(registry, 'cache_manager')
        assert registry._advanced_caching_enabled is True
        
        # ✅ Memory optimization preserved
        assert hasattr(registry, 'memory_monitor')
        assert registry._memory_optimization_enabled is True
        
        # ✅ Intelligent model selection with service management
        assert hasattr(registry, '_ensure_service_running')
        assert hasattr(registry, '_auto_install_dependencies')
    
    def test_issue_199_broader_scope_addressed(self):
        """Verify broader Issue #199 scope is addressed through Phase 2."""
        
        # Issue #199 focuses on automatic graph generation and enhanced pipelines
        # Phase 2 provides the service integration foundation for this:
        
        # ✅ Service startup automation for models (supporting automatic graph execution)
        ollama_manager = SERVICE_MANAGERS["ollama"]
        docker_manager = SERVICE_MANAGERS["docker"]
        
        assert callable(ollama_manager.ensure_model_available)
        assert callable(docker_manager.ensure_container_running)
        
        # ✅ Health monitoring (supporting automatic graph reliability)
        assert callable(ollama_manager.health_check_model)
        assert callable(docker_manager.health_check_container)
        
        # ✅ Registry integration with auto-configuration (supporting automatic model selection in graphs)
        registry = ModelRegistry()
        assert callable(registry.auto_register_langchain_models)
        assert callable(registry._ensure_service_running)
        assert callable(registry._auto_install_dependencies)
        
        # ✅ Preserved sophisticated model selection (supporting intelligent graph node assignment)
        assert hasattr(registry.model_selector, 'select')
        assert hasattr(registry, 'find_models_by_capability')
        assert hasattr(registry, 'recommend_models_for_task')


class TestPhase2BackwardCompatibility:
    """Test that Phase 2 maintains 100% backward compatibility."""
    
    def test_existing_service_manager_api_unchanged(self):
        """Test that existing service manager APIs are unchanged."""
        
        ollama_manager = SERVICE_MANAGERS["ollama"]
        docker_manager = SERVICE_MANAGERS["docker"]
        
        # Original ServiceManager interface must be preserved
        assert hasattr(ollama_manager, 'is_installed')
        assert hasattr(ollama_manager, 'is_running')
        assert hasattr(ollama_manager, 'start')
        assert hasattr(ollama_manager, 'stop')
        assert hasattr(ollama_manager, 'ensure_running')
        
        assert hasattr(docker_manager, 'is_installed')
        assert hasattr(docker_manager, 'is_running')  
        assert hasattr(docker_manager, 'start')
        assert hasattr(docker_manager, 'stop')
        assert hasattr(docker_manager, 'ensure_running')
    
    def test_existing_model_registry_api_unchanged(self):
        """Test that existing ModelRegistry APIs are unchanged."""
        
        registry = ModelRegistry()
        
        # Original ModelRegistry interface must be preserved
        assert hasattr(registry, 'register_model')
        assert hasattr(registry, 'unregister_model')
        assert hasattr(registry, 'get_model')
        assert hasattr(registry, 'select_model')
        assert hasattr(registry, 'find_models_by_capability')
        assert hasattr(registry, 'recommend_models_for_task')
        assert hasattr(registry, 'get_model_statistics')
        
        # UCB selector interface preserved
        assert hasattr(registry.model_selector, 'select')
        assert hasattr(registry.model_selector, 'update_reward')
        assert hasattr(registry.model_selector, 'initialize_model')
        
    def test_existing_model_registration_unchanged(self):
        """Test that existing model registration still works identically."""
        
        registry = ModelRegistry()
        
        # Mock regular model
        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.provider = "custom"
        mock_model.metrics = MagicMock()
        mock_model.metrics.success_rate = 0.9
        
        # Register model using existing API
        with patch.object(registry, '_get_model_key', return_value="custom:test-model"):
            registry.register_model(mock_model)
            
            # Verify it was registered normally
            assert "custom:test-model" in registry.models
            assert registry.models["custom:test-model"] == mock_model
            
            # Verify UCB initialization still works
            assert "custom:test-model" in registry.model_selector.model_stats
            assert registry.model_selector.model_stats["custom:test-model"]["average_reward"] == 0.9
            
            # Verify it's NOT in LangChain adapters (because it's a regular model)
            assert "custom:test-model" not in registry._langchain_adapters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])