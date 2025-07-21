Models API Reference
====================

This section documents the model registry, selection algorithms, and model integration components.

.. note::
   For guides on creating custom models, see the :doc:`../advanced/custom_models` documentation.

.. currentmodule:: orchestrator.models

Overview
--------

The Orchestrator model system provides a unified interface for working with different AI models and providers. It includes intelligent model selection, performance tracking, and cost optimization.

**Key Features:**
- **Multi-Provider Support**: OpenAI, Anthropic, Hugging Face, Ollama, and custom models
- **Intelligent Selection**: Multi-armed bandit algorithms for optimal model choice
- **Performance Tracking**: Real-time metrics and cost monitoring
- **Caching**: Response caching to reduce API calls and costs
- **Health Monitoring**: Automatic model health checks and failover

**Usage Pattern:**

.. code-block:: python

    import os
    from orchestrator.models.model_registry import ModelRegistry
    from orchestrator.integrations.openai_model import OpenAIModel
    
    # Create registry
    registry = ModelRegistry()
    
    # Register models
    gpt4 = OpenAIModel(name="gpt-4", api_key=os.environ.get("OPENAI_API_KEY"))
    registry.register_model(gpt4)
    
    # Select best model for task
    best_model = registry.select_model(
        capabilities=["text_generation"],
        max_cost=0.01
    )
    
    # Generate response
    response = await best_model.generate_response("Hello, world!")

Model Registry
--------------

The ModelRegistry manages all available models and provides intelligent selection based on performance, cost, and availability.

**Key Capabilities:**
- **Model Registration**: Register models from different providers
- **Selection Algorithms**: UCB, cost-based, and performance-based selection
- **Health Monitoring**: Continuous health checks and failover
- **Metrics Tracking**: Performance and cost metrics for each model
- **Caching Integration**: Automatic response caching

**Example Usage:**

.. code-block:: python

    import os
    from orchestrator.models.model_registry import ModelRegistry
    from orchestrator.integrations.openai_model import OpenAIModel
    from orchestrator.integrations.anthropic_model import AnthropicModel
    
    # Create registry with custom configuration
    registry = ModelRegistry(
        selection_strategy="ucb",
        health_check_interval=300,
        cache_responses=True
    )
    
    # Register multiple models
    models = [
        OpenAIModel(name="gpt-4", api_key=os.environ.get("OPENAI_API_KEY")),
        OpenAIModel(name="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY")),
        AnthropicModel(name="claude-3-opus", api_key=os.environ.get("ANTHROPIC_API_KEY"))
    ]
    
    for model in models:
        registry.register_model(model)
    
    # Select model based on requirements
    selected_model = registry.select_model(
        capabilities=["text_generation", "code_generation"],
        max_cost=0.05,
        min_speed=100,
        context_length=8000
    )
    
    # Use selected model
    response = await selected_model.generate_response(
        "Write a Python function to calculate fibonacci numbers"
    )

**Classes:**

.. autoclass:: orchestrator.models.model_registry.ModelRegistry
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.models.model_registry.ModelNotFoundError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.models.model_registry.NoEligibleModelsError
   :members:
   :undoc-members:
   :show-inheritance:

Model Selection Algorithms
---------------------------

The Orchestrator includes several model selection algorithms to optimize performance and cost:

Upper Confidence Bound (UCB) Selector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The UCB selector uses multi-armed bandit algorithms to balance exploration and exploitation when selecting models.

**Key Features:**
- **Exploration vs Exploitation**: Balances trying new models with using proven ones
- **Performance Learning**: Learns from historical performance data
- **Confidence Intervals**: Uses statistical confidence to guide selection
- **Adaptive Behavior**: Adjusts selection based on changing conditions

**Example Usage:**

.. code-block:: python

    from orchestrator.models.model_registry import UCBModelSelector
    
    # Create UCB selector with configuration
    selector = UCBModelSelector(
        exploration_factor=2.0,
        min_trials=5,
        confidence_level=0.95
    )
    
    # Use with model registry
    registry = ModelRegistry(model_selector=selector)
    
    # Selection automatically uses UCB algorithm
    model = registry.select_model(capabilities=["text_generation"])

**Classes:**

.. autoclass:: orchestrator.models.model_registry.UCBModelSelector
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.models.model_registry.ModelSelector
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.models.model_registry.CostBasedSelector
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.models.model_registry.PerformanceSelector
   :members:
   :undoc-members:
   :show-inheritance:

Model Integration Classes
-------------------------

The Orchestrator includes built-in integrations for popular model providers:

Base Model Classes
^^^^^^^^^^^^^^^^^^

.. autoclass:: orchestrator.core.model.Model
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.core.model.ModelCapabilities
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.core.model.ModelMetrics
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.core.model.ModelConstraints
   :members:
   :undoc-members:
   :show-inheritance:

Provider Integrations
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: orchestrator.integrations.openai_model.OpenAIModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.integrations.anthropic_model.AnthropicModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.integrations.huggingface_model.HuggingFaceModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.integrations.ollama_model.OllamaModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.integrations.google_model.GoogleModel
   :members:
   :undoc-members:
   :show-inheritance:

Lazy Loading Models
^^^^^^^^^^^^^^^^^^^

For resource-efficient model loading:

.. autoclass:: orchestrator.integrations.lazy_huggingface_model.LazyHuggingFaceModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: orchestrator.integrations.lazy_ollama_model.LazyOllamaModel
   :members:
   :undoc-members:
   :show-inheritance:

Model Health Monitoring
------------------------

The model system includes comprehensive health monitoring capabilities:

**Features:**
- **Automatic Health Checks**: Periodic model availability checks
- **Failover Support**: Automatic switching to backup models
- **Performance Monitoring**: Track response times and error rates
- **Cost Tracking**: Monitor API costs and usage

**Example Usage:**

.. code-block:: python

    from orchestrator.models.health_monitor import ModelHealthMonitor
    
    # Create health monitor
    monitor = ModelHealthMonitor(
        check_interval=300,  # 5 minutes
        failure_threshold=3,
        recovery_threshold=2
    )
    
    # Use with model registry
    registry = ModelRegistry(health_monitor=monitor)
    
    # Get health status
    health_status = await registry.get_model_health("gpt-4")
    
    if not health_status.healthy:
        print(f"Model issues: {health_status.issues}")
        
        # Automatic failover to backup model
        backup_model = registry.select_model(
            capabilities=["text_generation"],
            exclude_models=["gpt-4"]
        )

Model Caching
-------------

The model system includes intelligent caching to improve performance and reduce costs:

**Caching Strategies:**
- **Response Caching**: Cache model responses for identical inputs
- **Embedding Caching**: Cache computed embeddings
- **Metadata Caching**: Cache model metadata and capabilities
- **Multi-Level Caching**: Memory, Redis, and disk-based caching

**Example Usage:**

.. code-block:: python

    from orchestrator.core.cache import ModelCache
    
    # Configure caching
    cache = ModelCache(
        backend="redis",
        ttl=3600,  # 1 hour
        max_size=10000,
        compression=True
    )
    
    # Enable caching for model
    model.enable_caching(cache)
    
    # First call - cache miss
    response1 = await model.generate_response("What is AI?")
    
    # Second call - cache hit
    response2 = await model.generate_response("What is AI?")
    
    # Get cache statistics
    stats = cache.get_statistics()
    print(f"Cache hit rate: {stats.hit_rate:.2%}")

Custom Model Development
------------------------

Create custom models by extending the base Model class:

.. code-block:: python

    from orchestrator.core.model import Model, ModelCapabilities
    from typing import Dict, Any
    
    class CustomModel(Model):
        """Custom model implementation."""
        
        def __init__(self, name: str, config: Dict[str, Any]):
            capabilities = ModelCapabilities(
                text_generation=True,
                max_tokens=4096,
                supports_streaming=True
            )
            
            super().__init__(
                name=name,
                provider="custom",
                capabilities=capabilities
            )
            
            self.config = config
        
        async def generate_response(self, prompt: str, **kwargs) -> str:
            """Generate response from custom model."""
            # Implementation here
            pass
        
        async def health_check(self) -> bool:
            """Check model health."""
            # Implementation here
            pass

For detailed custom model development guides, see the :doc:`../advanced/custom_models` documentation.