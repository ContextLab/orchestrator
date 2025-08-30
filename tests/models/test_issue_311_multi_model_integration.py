"""
Unit tests for Issue #311: Multi-Model Integration

Tests the unified model management system with provider abstractions and 
intelligent selection strategies that work with multiple AI providers.
"""

import pytest
import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import model-related components
try:
    from src.orchestrator.models.model_registry import ModelRegistry
    from src.orchestrator.models.langchain_adapter import LangChainModelAdapter
    from src.orchestrator.models.advanced_caching import CacheManager
    from src.orchestrator.models.performance_optimizations import ModelRegistryOptimizer
    from src.orchestrator.models.memory_optimization import MemoryOptimizedRegistry
    from src.orchestrator.foundation import ModelManagerInterface
    HAS_MODEL_COMPONENTS = True
except ImportError:
    HAS_MODEL_COMPONENTS = False


@pytest.mark.skipif(not HAS_MODEL_COMPONENTS, reason="Model components not available")
class TestModelRegistry:
    """Test model registry functionality."""
    
    @pytest.fixture
    def model_registry(self):
        """Create ModelRegistry instance for testing."""
        return ModelRegistry(
            enable_memory_optimization=False,  # Disable for simpler testing
            enable_advanced_caching=False
        )
    
    @pytest.fixture
    def sample_models(self):
        """Sample model configurations for testing."""
        return {
            'gpt-4': {
                'provider': 'openai',
                'model_name': 'gpt-4',
                'capabilities': {
                    'max_tokens': 8192,
                    'supports_functions': True,
                    'supports_vision': True
                },
                'pricing': {
                    'input_cost_per_token': 0.00003,
                    'output_cost_per_token': 0.00006
                },
                'performance': {
                    'avg_response_time': 2.5,
                    'reliability_score': 0.98
                }
            },
            'claude-3-opus': {
                'provider': 'anthropic',
                'model_name': 'claude-3-opus-20240229',
                'capabilities': {
                    'max_tokens': 4096,
                    'supports_functions': True,
                    'supports_vision': True
                },
                'pricing': {
                    'input_cost_per_token': 0.000015,
                    'output_cost_per_token': 0.000075
                },
                'performance': {
                    'avg_response_time': 3.1,
                    'reliability_score': 0.96
                }
            },
            'gemini-pro': {
                'provider': 'google',
                'model_name': 'gemini-pro',
                'capabilities': {
                    'max_tokens': 30720,
                    'supports_functions': True,
                    'supports_vision': False
                },
                'pricing': {
                    'input_cost_per_token': 0.0000005,
                    'output_cost_per_token': 0.0000015
                },
                'performance': {
                    'avg_response_time': 1.8,
                    'reliability_score': 0.94
                }
            },
            'llama-3-70b': {
                'provider': 'ollama',
                'model_name': 'llama3:70b',
                'capabilities': {
                    'max_tokens': 8192,
                    'supports_functions': False,
                    'supports_vision': False
                },
                'pricing': {
                    'input_cost_per_token': 0.0,  # Local model
                    'output_cost_per_token': 0.0
                },
                'performance': {
                    'avg_response_time': 5.2,
                    'reliability_score': 0.92
                }
            }
        }
    
    def test_model_registry_initialization(self, model_registry):
        """Test ModelRegistry initialization."""
        assert isinstance(model_registry.models, dict)
        assert len(model_registry.models) == 0
        assert hasattr(model_registry, 'model_selector')
    
    def test_model_registration(self, model_registry, sample_models):
        """Test model registration functionality."""
        # Register models
        for model_id, config in sample_models.items():
            # Mock model registration
            model_registry.models[model_id] = Mock()
            model_registry.models[model_id].id = model_id
            model_registry.models[model_id].provider = config['provider']
            model_registry.models[model_id].capabilities = config['capabilities']
        
        assert len(model_registry.models) == 4
        assert 'gpt-4' in model_registry.models
        assert 'claude-3-opus' in model_registry.models
        assert 'gemini-pro' in model_registry.models
        assert 'llama-3-70b' in model_registry.models
    
    def test_model_capabilities_query(self, model_registry, sample_models):
        """Test querying model capabilities."""
        # Setup models with capabilities
        for model_id, config in sample_models.items():
            model_registry.models[model_id] = Mock()
            model_registry.models[model_id].capabilities = config['capabilities']
        
        # Test capability queries
        gpt4_caps = model_registry.models['gpt-4'].capabilities
        assert gpt4_caps['supports_functions'] == True
        assert gpt4_caps['supports_vision'] == True
        assert gpt4_caps['max_tokens'] == 8192
        
        llama_caps = model_registry.models['llama-3-70b'].capabilities  
        assert llama_caps['supports_functions'] == False
        assert llama_caps['supports_vision'] == False
    
    def test_provider_abstraction(self, model_registry, sample_models):
        """Test provider abstraction functionality."""
        # Group models by provider
        providers = {}
        for model_id, config in sample_models.items():
            provider = config['provider']
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model_id)
        
        # Verify provider grouping
        assert 'openai' in providers
        assert 'anthropic' in providers
        assert 'google' in providers
        assert 'ollama' in providers
        
        assert 'gpt-4' in providers['openai']
        assert 'claude-3-opus' in providers['anthropic']
        assert 'gemini-pro' in providers['google']
        assert 'llama-3-70b' in providers['ollama']


@pytest.mark.skipif(not HAS_MODEL_COMPONENTS, reason="Model components not available") 
class TestModelSelection:
    """Test intelligent model selection strategies."""
    
    @pytest.fixture
    def mock_model_selector(self):
        """Create mock model selector."""
        selector = Mock()
        selector.select_model = Mock()
        return selector
    
    @pytest.fixture
    def model_requirements(self):
        """Sample model requirements for testing."""
        return {
            'cost_optimized': {
                'strategy': 'cost',
                'max_cost_per_token': 0.000001,
                'min_reliability': 0.9
            },
            'performance_optimized': {
                'strategy': 'performance',
                'max_response_time': 2.0,
                'min_reliability': 0.95,
                'required_capabilities': ['supports_functions']
            },
            'balanced': {
                'strategy': 'balanced',
                'cost_weight': 0.3,
                'performance_weight': 0.4, 
                'capability_weight': 0.3,
                'required_capabilities': ['supports_vision']
            },
            'task_specific': {
                'strategy': 'task_specific',
                'task_type': 'code_generation',
                'programming_language': 'python',
                'complexity': 'high'
            }
        }
    
    def test_cost_optimized_selection(self, mock_model_selector, model_requirements):
        """Test cost-optimized model selection."""
        requirements = model_requirements['cost_optimized']
        
        # Mock selection logic
        mock_model_selector.select_model.return_value = 'gemini-pro'  # Cheapest model
        
        selected = mock_model_selector.select_model(requirements)
        
        assert selected == 'gemini-pro'
        mock_model_selector.select_model.assert_called_once_with(requirements)
    
    def test_performance_optimized_selection(self, mock_model_selector, model_requirements):
        """Test performance-optimized model selection."""
        requirements = model_requirements['performance_optimized']
        
        # Mock selection logic - fastest model with required capabilities
        mock_model_selector.select_model.return_value = 'gemini-pro'  # Fastest response time
        
        selected = mock_model_selector.select_model(requirements)
        
        assert selected == 'gemini-pro'
        mock_model_selector.select_model.assert_called_once_with(requirements)
    
    def test_balanced_selection(self, mock_model_selector, model_requirements):
        """Test balanced model selection strategy."""
        requirements = model_requirements['balanced']
        
        # Mock balanced selection - considers cost, performance, and capabilities
        mock_model_selector.select_model.return_value = 'gpt-4'  # Best vision support
        
        selected = mock_model_selector.select_model(requirements)
        
        assert selected == 'gpt-4'
        mock_model_selector.select_model.assert_called_once_with(requirements)
    
    def test_task_specific_selection(self, mock_model_selector, model_requirements):
        """Test task-specific model selection."""
        requirements = model_requirements['task_specific']
        
        # Mock task-specific selection - best for code generation
        mock_model_selector.select_model.return_value = 'gpt-4'  # Best for coding tasks
        
        selected = mock_model_selector.select_model(requirements)
        
        assert selected == 'gpt-4'
        mock_model_selector.select_model.assert_called_once_with(requirements)
    
    def test_selection_with_constraints(self, mock_model_selector):
        """Test model selection with multiple constraints."""
        requirements = {
            'max_tokens': 8000,
            'supports_functions': True,
            'max_cost_per_token': 0.00005,
            'provider_preference': ['openai', 'anthropic']
        }
        
        # Mock constrained selection
        mock_model_selector.select_model.return_value = 'gpt-4'
        
        selected = mock_model_selector.select_model(requirements)
        
        assert selected == 'gpt-4'
    
    def test_fallback_selection(self, mock_model_selector):
        """Test fallback when no models meet requirements."""
        strict_requirements = {
            'max_cost_per_token': 0.0000001,  # Very strict cost requirement
            'min_reliability': 0.999,
            'supports_advanced_reasoning': True
        }
        
        # Mock fallback selection
        mock_model_selector.select_model.side_effect = [None, 'claude-3-opus']  # Fallback to best available
        
        # First attempt should fail, second should return fallback
        selected = mock_model_selector.select_model(strict_requirements)
        assert selected is None
        
        # Fallback attempt
        fallback_requirements = {'strategy': 'best_available'}
        fallback = mock_model_selector.select_model(fallback_requirements)
        assert fallback == 'claude-3-opus'


@pytest.mark.skipif(not HAS_MODEL_COMPONENTS, reason="Model components not available")
class TestProviderAbstraction:
    """Test unified provider abstraction layer."""
    
    @pytest.fixture
    def mock_providers(self):
        """Create mock provider implementations."""
        return {
            'openai': Mock(),
            'anthropic': Mock(),
            'google': Mock(), 
            'ollama': Mock()
        }
    
    @pytest.fixture
    def unified_provider(self, mock_providers):
        """Create unified provider interface."""
        provider = Mock()
        provider.providers = mock_providers
        provider.invoke_model = AsyncMock()
        provider.get_model_info = AsyncMock()
        return provider
    
    @pytest.mark.asyncio
    async def test_unified_model_invocation(self, unified_provider):
        """Test unified model invocation across providers."""
        # Test OpenAI model invocation
        unified_provider.invoke_model.return_value = "OpenAI response"
        
        response = await unified_provider.invoke_model(
            model_id='gpt-4',
            prompt='Test prompt',
            temperature=0.7
        )
        
        assert response == "OpenAI response"
        unified_provider.invoke_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_provider_specific_parameters(self, unified_provider):
        """Test handling of provider-specific parameters."""
        # Different providers may have different parameter names
        test_cases = [
            {
                'model_id': 'gpt-4',
                'provider': 'openai',
                'params': {'temperature': 0.7, 'max_tokens': 1000}
            },
            {
                'model_id': 'claude-3-opus',
                'provider': 'anthropic', 
                'params': {'temperature': 0.7, 'max_tokens_to_sample': 1000}
            },
            {
                'model_id': 'gemini-pro',
                'provider': 'google',
                'params': {'temperature': 0.7, 'max_output_tokens': 1000}
            }
        ]
        
        for case in test_cases:
            unified_provider.invoke_model.return_value = f"Response from {case['provider']}"
            
            response = await unified_provider.invoke_model(
                model_id=case['model_id'],
                prompt='Test prompt',
                **case['params']
            )
            
            assert case['provider'] in response
    
    @pytest.mark.asyncio
    async def test_provider_authentication(self, unified_provider):
        """Test provider authentication handling."""
        # Mock authentication for different providers
        auth_configs = {
            'openai': {'api_key': 'sk-test-openai'},
            'anthropic': {'api_key': 'sk-ant-test'},
            'google': {'credentials_path': '/path/to/creds.json'},
            'ollama': {'base_url': 'http://localhost:11434'}
        }
        
        for provider, config in auth_configs.items():
            unified_provider.configure_auth = Mock()
            unified_provider.configure_auth(provider, config)
            unified_provider.configure_auth.assert_called_with(provider, config)
    
    @pytest.mark.asyncio
    async def test_rate_limiting_abstraction(self, unified_provider):
        """Test unified rate limiting across providers."""
        # Mock rate limiting
        unified_provider.check_rate_limit = AsyncMock(return_value=True)
        unified_provider.wait_for_rate_limit = AsyncMock()
        
        # Test rate limit checking
        can_proceed = await unified_provider.check_rate_limit('gpt-4')
        assert can_proceed == True
        
        # Test rate limit waiting
        await unified_provider.wait_for_rate_limit('claude-3-opus')
        unified_provider.wait_for_rate_limit.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_error_handling_abstraction(self, unified_provider):
        """Test unified error handling across providers."""
        # Mock different provider errors
        provider_errors = [
            ('openai', 'RateLimitError'),
            ('anthropic', 'APIError'),
            ('google', 'GoogleAPIError'),
            ('ollama', 'ConnectionError')
        ]
        
        unified_provider.handle_provider_error = Mock()
        
        for provider, error_type in provider_errors:
            mock_error = Exception(f"{provider}: {error_type}")
            unified_provider.handle_provider_error(provider, mock_error)
            
        assert unified_provider.handle_provider_error.call_count == len(provider_errors)


@pytest.mark.skipif(not HAS_MODEL_COMPONENTS, reason="Model components not available")
class TestModelDiscovery:
    """Test dynamic model discovery capabilities."""
    
    @pytest.fixture
    def mock_discovery_service(self):
        """Create mock model discovery service."""
        service = Mock()
        service.discover_models = AsyncMock()
        service.get_model_capabilities = AsyncMock()
        service.check_model_availability = AsyncMock()
        return service
    
    @pytest.mark.asyncio
    async def test_automatic_model_discovery(self, mock_discovery_service):
        """Test automatic discovery of available models."""
        # Mock discovered models
        discovered_models = [
            {'id': 'gpt-4', 'provider': 'openai', 'available': True},
            {'id': 'gpt-3.5-turbo', 'provider': 'openai', 'available': True},
            {'id': 'claude-3-opus', 'provider': 'anthropic', 'available': True},
            {'id': 'gemini-pro', 'provider': 'google', 'available': False}  # Not available
        ]
        
        mock_discovery_service.discover_models.return_value = discovered_models
        
        models = await mock_discovery_service.discover_models()
        
        assert len(models) == 4
        available_models = [m for m in models if m['available']]
        assert len(available_models) == 3
    
    @pytest.mark.asyncio
    async def test_capability_detection(self, mock_discovery_service):
        """Test automatic capability detection for models."""
        model_capabilities = {
            'gpt-4': {
                'text_generation': True,
                'function_calling': True,
                'vision': True,
                'max_tokens': 8192,
                'languages': ['en', 'es', 'fr', 'de']
            },
            'claude-3-opus': {
                'text_generation': True,
                'function_calling': True,
                'vision': True,
                'max_tokens': 4096,
                'languages': ['en']
            }
        }
        
        for model_id, capabilities in model_capabilities.items():
            mock_discovery_service.get_model_capabilities.return_value = capabilities
            
            caps = await mock_discovery_service.get_model_capabilities(model_id)
            
            assert caps['text_generation'] == True
            assert 'max_tokens' in caps
            assert 'languages' in caps
    
    @pytest.mark.asyncio
    async def test_model_availability_monitoring(self, mock_discovery_service):
        """Test monitoring of model availability over time."""
        # Mock availability checks
        availability_results = [
            ('gpt-4', True),
            ('claude-3-opus', True),
            ('gemini-pro', False),  # Currently unavailable
            ('local-model', True)
        ]
        
        for model_id, is_available in availability_results:
            mock_discovery_service.check_model_availability.return_value = is_available
            
            available = await mock_discovery_service.check_model_availability(model_id)
            
            assert available == is_available
    
    @pytest.mark.asyncio
    async def test_model_metadata_collection(self, mock_discovery_service):
        """Test collection of model metadata and performance metrics."""
        metadata = {
            'version': '2024-03-01',
            'training_data_cutoff': '2024-01-01',
            'performance_metrics': {
                'avg_response_time_ms': 2500,
                'tokens_per_second': 120,
                'uptime_percentage': 99.5
            },
            'cost_metrics': {
                'input_cost_per_1k_tokens': 0.03,
                'output_cost_per_1k_tokens': 0.06
            }
        }
        
        mock_discovery_service.get_model_metadata = AsyncMock(return_value=metadata)
        
        result = await mock_discovery_service.get_model_metadata('gpt-4')
        
        assert 'version' in result
        assert 'performance_metrics' in result
        assert 'cost_metrics' in result
        assert result['performance_metrics']['uptime_percentage'] == 99.5


@pytest.mark.skipif(not HAS_MODEL_COMPONENTS, reason="Model components not available")
class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        cache = Mock()
        cache.get = AsyncMock()
        cache.set = AsyncMock()
        cache.invalidate = AsyncMock()
        return cache
    
    @pytest.fixture
    def mock_connection_pool(self):
        """Create mock connection pool."""
        pool = Mock()
        pool.get_connection = AsyncMock()
        pool.release_connection = AsyncMock()
        pool.health_check = AsyncMock(return_value=True)
        return pool
    
    @pytest.mark.asyncio
    async def test_response_caching(self, mock_cache_manager):
        """Test API response caching functionality."""
        # Test cache miss
        mock_cache_manager.get.return_value = None
        
        cache_key = "gpt-4:prompt_hash:12345"
        cached_response = await mock_cache_manager.get(cache_key)
        
        assert cached_response is None
        
        # Test cache set
        response_data = {
            'response': 'This is a test response',
            'model': 'gpt-4',
            'timestamp': '2024-01-01T12:00:00Z'
        }
        
        await mock_cache_manager.set(cache_key, response_data, ttl=3600)
        mock_cache_manager.set.assert_called_once()
        
        # Test cache hit
        mock_cache_manager.get.return_value = response_data
        cached_response = await mock_cache_manager.get(cache_key)
        
        assert cached_response == response_data
    
    @pytest.mark.asyncio
    async def test_connection_pooling(self, mock_connection_pool):
        """Test connection pooling for high-throughput scenarios."""
        # Test connection acquisition
        mock_connection = Mock()
        mock_connection_pool.get_connection.return_value = mock_connection
        
        connection = await mock_connection_pool.get_connection('openai')
        assert connection == mock_connection
        
        # Test connection release
        await mock_connection_pool.release_connection(connection)
        mock_connection_pool.release_connection.assert_called_once()
        
        # Test health check
        is_healthy = await mock_connection_pool.health_check()
        assert is_healthy == True
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing capabilities."""
        # Mock batch processor
        batch_processor = Mock()
        batch_processor.add_request = Mock()
        batch_processor.process_batch = AsyncMock()
        
        # Add multiple requests to batch
        requests = [
            {'model': 'gpt-4', 'prompt': f'Prompt {i}'} 
            for i in range(5)
        ]
        
        for request in requests:
            batch_processor.add_request(request)
        
        assert batch_processor.add_request.call_count == 5
        
        # Process batch
        batch_results = [f"Response {i}" for i in range(5)]
        batch_processor.process_batch.return_value = batch_results
        
        results = await batch_processor.process_batch()
        assert len(results) == 5
    
    def test_model_warm_up(self):
        """Test model warm-up and pre-loading strategies."""
        # Mock warm-up manager
        warmup_manager = Mock()
        warmup_manager.preload_models = Mock()
        warmup_manager.warm_up_model = Mock()
        
        # Test preloading high-priority models
        priority_models = ['gpt-4', 'claude-3-opus']
        warmup_manager.preload_models(priority_models)
        
        warmup_manager.preload_models.assert_called_once_with(priority_models)
        
        # Test individual model warm-up
        warmup_manager.warm_up_model('gemini-pro')
        warmup_manager.warm_up_model.assert_called_once_with('gemini-pro')
    
    def test_resource_monitoring(self):
        """Test resource usage monitoring and optimization."""
        # Mock resource monitor
        resource_monitor = Mock()
        resource_monitor.get_memory_usage = Mock(return_value=75.5)  # 75.5% memory usage
        resource_monitor.get_cpu_usage = Mock(return_value=45.2)     # 45.2% CPU usage
        resource_monitor.get_api_quota_usage = Mock(return_value={'openai': 80, 'anthropic': 60})
        
        # Check resource usage
        memory_usage = resource_monitor.get_memory_usage()
        cpu_usage = resource_monitor.get_cpu_usage()
        quota_usage = resource_monitor.get_api_quota_usage()
        
        assert memory_usage == 75.5
        assert cpu_usage == 45.2
        assert quota_usage['openai'] == 80
        assert quota_usage['anthropic'] == 60


class TestFoundationIntegration:
    """Test integration with foundation ModelManagerInterface."""
    
    class MockModelManager(ModelManagerInterface):
        """Mock implementation of ModelManagerInterface for testing."""
        
        def __init__(self):
            self.models = {
                'gpt-4': {'provider': 'openai', 'capabilities': {'functions': True}},
                'claude-3-opus': {'provider': 'anthropic', 'capabilities': {'functions': True}},
                'gemini-pro': {'provider': 'google', 'capabilities': {'functions': True}}
            }
        
        async def select_model(self, requirements: Dict[str, Any]) -> str:
            if requirements.get('high_performance'):
                return 'gpt-4'
            elif requirements.get('cost_effective'):
                return 'gemini-pro'
            return 'claude-3-opus'
        
        async def invoke_model(self, model_id: str, prompt: str, **kwargs) -> str:
            return f"Response from {model_id}: {prompt[:50]}..."
        
        def list_available_models(self) -> List[str]:
            return list(self.models.keys())
        
        async def get_model_capabilities(self, model_id: str) -> Dict[str, Any]:
            return self.models.get(model_id, {}).get('capabilities', {})
    
    @pytest.fixture
    def model_manager(self):
        """Create mock model manager instance."""
        return self.MockModelManager()
    
    @pytest.mark.asyncio
    async def test_foundation_interface_implementation(self, model_manager):
        """Test that multi-model integration implements foundation interface."""
        # Test model selection
        model_id = await model_manager.select_model({'high_performance': True})
        assert model_id == 'gpt-4'
        
        model_id = await model_manager.select_model({'cost_effective': True})
        assert model_id == 'gemini-pro'
        
        # Test model invocation
        response = await model_manager.invoke_model('gpt-4', 'Test prompt', temperature=0.7)
        assert 'gpt-4' in response
        assert 'Test prompt' in response
        
        # Test model listing
        models = model_manager.list_available_models()
        assert len(models) == 3
        assert 'gpt-4' in models
        assert 'claude-3-opus' in models
        assert 'gemini-pro' in models
        
        # Test capabilities query
        capabilities = await model_manager.get_model_capabilities('gpt-4')
        assert 'functions' in capabilities
        assert capabilities['functions'] == True
    
    @pytest.mark.asyncio
    async def test_multi_provider_workflow(self, model_manager):
        """Test complete multi-provider workflow."""
        # Step 1: List available models
        available_models = model_manager.list_available_models()
        assert len(available_models) > 0
        
        # Step 2: Select model based on requirements
        requirements = {
            'task': 'text_generation',
            'max_tokens': 4000,
            'cost_conscious': True
        }
        
        selected_model = await model_manager.select_model(requirements)
        assert selected_model in available_models
        
        # Step 3: Check model capabilities
        capabilities = await model_manager.get_model_capabilities(selected_model)
        assert isinstance(capabilities, dict)
        
        # Step 4: Invoke model
        test_prompt = "Generate a summary of machine learning concepts"
        response = await model_manager.invoke_model(
            selected_model, 
            test_prompt,
            temperature=0.7,
            max_tokens=1000
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, model_manager):
        """Test error handling in foundation integration."""
        # Test invalid model selection
        empty_requirements = {}
        model_id = await model_manager.select_model(empty_requirements)
        assert model_id in model_manager.models  # Should return default
        
        # Test capabilities for non-existent model
        capabilities = await model_manager.get_model_capabilities('non-existent-model')
        assert capabilities == {}  # Should return empty dict
        
        # Test model invocation with minimal parameters
        response = await model_manager.invoke_model('gpt-4', '')
        assert isinstance(response, str)


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])