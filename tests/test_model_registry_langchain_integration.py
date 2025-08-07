"""Tests for ModelRegistry LangChain integration - Phase 2."""

import pytest
import asyncio
from unittest.mock import patch, MagicMock

from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.models.langchain_adapter import LangChainModelAdapter
from src.orchestrator.core.exceptions import ModelNotFoundError


class TestModelRegistryLangChainIntegration:
    """Test ModelRegistry LangChain integration while preserving UCB and caching."""
    
    def test_registry_initialization_with_langchain(self):
        """Test registry initialization includes LangChain integration."""
        registry = ModelRegistry()
        
        assert hasattr(registry, '_langchain_adapters')
        assert hasattr(registry, '_langchain_enabled')
        assert registry._langchain_enabled is True
        assert registry._langchain_adapters == {}
        
    @patch.object(LangChainModelAdapter, '__init__', return_value=None)
    def test_register_langchain_model_success(self, mock_adapter_init):
        """Test successful LangChain model registration."""
        registry = ModelRegistry()
        
        # Mock the adapter
        mock_adapter = MagicMock()
        mock_adapter.name = "gpt-3.5-turbo"
        mock_adapter.provider = "openai"
        mock_adapter.capabilities = MagicMock()
        mock_adapter.requirements = MagicMock()
        mock_adapter.metrics = MagicMock()
        
        # Mock the _get_model_key method
        with patch.object(registry, '_get_model_key', return_value="openai:gpt-3.5-turbo"):
            with patch.object(LangChainModelAdapter, '__new__', return_value=mock_adapter):
                model_key = registry.register_langchain_model("openai", "gpt-3.5-turbo")
                
                assert model_key == "openai:gpt-3.5-turbo"
                assert model_key in registry._langchain_adapters
                assert registry._langchain_adapters[model_key] == mock_adapter
                
    def test_register_langchain_model_disabled(self):
        """Test LangChain model registration when integration disabled."""
        registry = ModelRegistry()
        registry.disable_langchain_integration()
        
        with pytest.raises(ValueError, match="LangChain integration is disabled"):
            registry.register_langchain_model("openai", "gpt-3.5-turbo")
            
    def test_unregister_langchain_model(self):
        """Test unregistering LangChain model."""
        registry = ModelRegistry()
        
        # Setup mock adapter
        mock_adapter = MagicMock()
        registry._langchain_adapters["openai:gpt-3.5-turbo"] = mock_adapter
        registry.models["openai:gpt-3.5-turbo"] = mock_adapter
        
        with patch.object(registry, 'unregister_model') as mock_unregister:
            registry.unregister_langchain_model("openai", "gpt-3.5-turbo")
            
            assert "openai:gpt-3.5-turbo" not in registry._langchain_adapters
            mock_unregister.assert_called_once_with("gpt-3.5-turbo", "openai")
            
    def test_get_langchain_adapters(self):
        """Test getting all LangChain adapters."""
        registry = ModelRegistry()
        
        mock_adapter1 = MagicMock()
        mock_adapter2 = MagicMock()
        registry._langchain_adapters["openai:gpt-3.5-turbo"] = mock_adapter1
        registry._langchain_adapters["anthropic:claude-3-haiku"] = mock_adapter2
        
        adapters = registry.get_langchain_adapters()
        
        assert len(adapters) == 2
        assert "openai:gpt-3.5-turbo" in adapters
        assert "anthropic:claude-3-haiku" in adapters
        assert adapters is not registry._langchain_adapters  # Should be a copy
        
    def test_is_langchain_model(self):
        """Test checking if model is a LangChain adapter."""
        registry = ModelRegistry()
        
        registry._langchain_adapters["openai:gpt-3.5-turbo"] = MagicMock()
        
        assert registry.is_langchain_model("openai:gpt-3.5-turbo") is True
        assert registry.is_langchain_model("anthropic:claude-3-haiku") is False
        
    def test_enable_disable_langchain_integration(self):
        """Test enabling/disabling LangChain integration."""
        registry = ModelRegistry()
        
        # Initially enabled
        assert registry._langchain_enabled is True
        
        # Disable
        registry.disable_langchain_integration()
        assert registry._langchain_enabled is False
        
        # Enable
        registry.enable_langchain_integration()
        assert registry._langchain_enabled is True
        
    @patch.object(ModelRegistry, '_ensure_service_running')
    @patch.object(ModelRegistry, '_auto_install_dependencies')
    @patch.object(ModelRegistry, 'register_langchain_model')
    def test_auto_register_langchain_models_success(self, mock_register, mock_install, mock_service):
        """Test auto-registration of LangChain models from configuration."""
        registry = ModelRegistry()
        
        mock_register.side_effect = ["openai:gpt-4-turbo", "anthropic:claude-3-sonnet", "ollama:llama3.2:3b"]
        
        config = {
            "models": [
                {"provider": "openai", "model": "gpt-4-turbo", "auto_install": True},
                {"provider": "anthropic", "model": "claude-3-sonnet", "auto_install": True},
                {"provider": "ollama", "model": "llama3.2:3b", "ensure_running": True, "auto_pull": True}
            ]
        }
        
        registered_keys = registry.auto_register_langchain_models(config)
        
        assert len(registered_keys) == 3
        assert "openai:gpt-4-turbo" in registered_keys
        assert "anthropic:claude-3-sonnet" in registered_keys
        assert "ollama:llama3.2:3b" in registered_keys
        
        # Verify service and dependency management calls
        mock_install.assert_any_call("openai")
        mock_install.assert_any_call("anthropic")
        mock_service.assert_called_once_with("ollama")
        
    def test_auto_register_langchain_models_disabled(self):
        """Test auto-registration fails when LangChain integration disabled."""
        registry = ModelRegistry()
        registry.disable_langchain_integration()
        
        config = {"models": [{"provider": "openai", "model": "gpt-4-turbo"}]}
        
        with pytest.raises(ValueError, match="LangChain integration is disabled"):
            registry.auto_register_langchain_models(config)
            
    @patch('src.orchestrator.utils.service_manager.ensure_service_running')
    def test_ensure_service_running_ollama(self, mock_ensure_service):
        """Test ensuring Ollama service is running."""
        registry = ModelRegistry()
        mock_ensure_service.return_value = True
        
        registry._ensure_service_running("ollama")
        
        mock_ensure_service.assert_called_once_with("ollama")
        
    @patch('src.orchestrator.utils.auto_install.ensure_packages')
    def test_auto_install_dependencies_openai(self, mock_ensure_packages):
        """Test auto-installing OpenAI dependencies."""
        registry = ModelRegistry()
        
        registry._auto_install_dependencies("openai")
        
        mock_ensure_packages.assert_called_once_with(["langchain-openai"])
        
    def test_auto_install_dependencies_mapping(self):
        """Test dependency mapping for all providers."""
        registry = ModelRegistry()
        
        with patch('src.orchestrator.utils.auto_install.ensure_packages') as mock_ensure:
            registry._auto_install_dependencies("openai")
            mock_ensure.assert_called_with(["langchain-openai"])
            
            registry._auto_install_dependencies("anthropic")
            mock_ensure.assert_called_with(["langchain-anthropic"])
            
            registry._auto_install_dependencies("google")
            mock_ensure.assert_called_with(["langchain-google-genai"])
            
            registry._auto_install_dependencies("ollama")
            mock_ensure.assert_called_with(["langchain-community"])
            
            registry._auto_install_dependencies("huggingface")
            mock_ensure.assert_called_with(["langchain-huggingface"])


class TestModelRegistryUCBPreservation:
    """Test that UCB selection algorithm is preserved with LangChain integration."""
    
    def test_langchain_models_use_ucb_selection(self):
        """Test that LangChain models participate in UCB selection."""
        registry = ModelRegistry()
        
        # Mock LangChain adapter
        mock_adapter = MagicMock()
        mock_adapter.name = "gpt-3.5-turbo"
        mock_adapter.provider = "openai"
        mock_adapter.metrics = MagicMock()
        mock_adapter.metrics.success_rate = 0.95
        
        # Mock the registration process
        with patch.object(registry, '_get_model_key', return_value="openai:gpt-3.5-turbo"):
            with patch.object(LangChainModelAdapter, '__new__', return_value=mock_adapter):
                registry.register_langchain_model("openai", "gpt-3.5-turbo")
                
                # Verify UCB selector was initialized with the model
                assert "openai:gpt-3.5-turbo" in registry.model_selector.model_stats
                assert registry.model_selector.model_stats["openai:gpt-3.5-turbo"]["average_reward"] == 0.95
                
    def test_langchain_models_in_selection_process(self):
        """Test that LangChain models are included in selection process."""
        registry = ModelRegistry()
        
        # Add mock models to registry
        mock_langchain_model = MagicMock()
        mock_regular_model = MagicMock()
        
        registry.models["openai:gpt-3.5-turbo"] = mock_langchain_model
        registry.models["custom:local-model"] = mock_regular_model
        registry._langchain_adapters["openai:gpt-3.5-turbo"] = mock_langchain_model
        
        # Initialize UCB stats
        registry.model_selector.model_stats["openai:gpt-3.5-turbo"] = {
            "attempts": 0, "successes": 0, "total_reward": 0.0, "average_reward": 0.5
        }
        registry.model_selector.model_stats["custom:local-model"] = {
            "attempts": 0, "successes": 0, "total_reward": 0.0, "average_reward": 0.5
        }
        
        # Test selection includes both types
        available_models = ["openai:gpt-3.5-turbo", "custom:local-model"]
        selected = registry.model_selector.select(available_models, {})
        
        assert selected in available_models
        
    def test_langchain_model_performance_tracking(self):
        """Test that LangChain model performance is tracked by UCB."""
        registry = ModelRegistry()
        
        # Initialize model in UCB selector
        registry.model_selector.initialize_model("openai:gpt-3.5-turbo", MagicMock(success_rate=0.8))
        
        # Simulate selection and reward
        selected = registry.model_selector.select(["openai:gpt-3.5-turbo"], {})
        assert selected == "openai:gpt-3.5-turbo"
        
        # Update with reward
        registry.model_selector.update_reward("openai:gpt-3.5-turbo", 0.9)
        
        # Verify stats were updated
        stats = registry.model_selector.model_stats["openai:gpt-3.5-turbo"]
        assert stats["successes"] == 1
        assert stats["total_reward"] == 0.9


class TestModelRegistryCachingPreservation:
    """Test that advanced caching is preserved with LangChain integration."""
    
    def test_langchain_models_use_advanced_caching(self):
        """Test that LangChain models use advanced caching when available."""
        registry = ModelRegistry(enable_advanced_caching=True)
        
        # Verify caching is enabled
        assert registry._advanced_caching_enabled is True
        assert registry.cache_manager is not None
        
        # Mock cache invalidation on registration
        with patch.object(registry.cache_manager, 'invalidate_model') as mock_invalidate:
            mock_adapter = MagicMock()
            
            with patch.object(registry, '_get_model_key', return_value="openai:gpt-3.5-turbo"):
                with patch.object(LangChainModelAdapter, '__new__', return_value=mock_adapter):
                    registry.register_langchain_model("openai", "gpt-3.5-turbo")
                    
                    # Verify cache invalidation was called
                    mock_invalidate.assert_called_once_with("openai:gpt-3.5-turbo")
                    
    def test_langchain_models_memory_optimization(self):
        """Test that LangChain models participate in memory optimization."""
        registry = ModelRegistry(enable_memory_optimization=True)
        
        # Verify memory optimization is enabled
        assert registry._memory_optimization_enabled is True
        assert registry.memory_monitor is not None
        
        # Mock memory check on registration
        with patch.object(registry.memory_monitor, 'check_memory') as mock_check:
            mock_adapter = MagicMock()
            
            with patch.object(registry, '_get_model_key', return_value="openai:gpt-3.5-turbo"):
                with patch.object(LangChainModelAdapter, '__new__', return_value=mock_adapter):
                    registry.register_langchain_model("openai", "gpt-3.5-turbo")
                    
                    # Verify memory check was called
                    mock_check.assert_called()


class TestModelRegistryBackwardCompatibility:
    """Test that existing ModelRegistry functionality is preserved."""
    
    def test_regular_model_registration_unchanged(self):
        """Test that regular model registration still works normally."""
        registry = ModelRegistry()
        
        # Create mock regular model
        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.provider = "custom"
        mock_model.metrics = MagicMock()
        
        with patch.object(registry, '_get_model_key', return_value="custom:test-model"):
            registry.register_model(mock_model)
            
            assert "custom:test-model" in registry.models
            assert registry.models["custom:test-model"] == mock_model
            assert "custom:test-model" not in registry._langchain_adapters  # Should not be in LangChain adapters
            
    def test_model_selection_includes_all_types(self):
        """Test that model selection works with mixed regular and LangChain models."""
        registry = ModelRegistry()
        
        # Add mixed models
        regular_model = MagicMock()
        langchain_model = MagicMock()
        
        registry.models["custom:regular"] = regular_model
        registry.models["openai:langchain"] = langchain_model
        registry._langchain_adapters["openai:langchain"] = langchain_model
        
        # Initialize UCB stats
        registry.model_selector.model_stats["custom:regular"] = {
            "attempts": 0, "successes": 0, "total_reward": 0.0, "average_reward": 0.5
        }
        registry.model_selector.model_stats["openai:langchain"] = {
            "attempts": 0, "successes": 0, "total_reward": 0.0, "average_reward": 0.5
        }
        
        # Test selection works with both types
        available_models = ["custom:regular", "openai:langchain"]
        selected = registry.model_selector.select(available_models, {})
        
        assert selected in available_models


if __name__ == "__main__":
    pytest.main([__file__, "-v"])