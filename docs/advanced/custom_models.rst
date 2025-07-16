Custom Models
==============

The Orchestrator provides a flexible model registry system that allows you to integrate custom models and model providers beyond the built-in integrations. This enables you to use proprietary models, local models, or specialized AI systems within your pipelines.

Model Registry Architecture
----------------------------

The model registry uses a multi-armed bandit approach (Upper Confidence Bound) for intelligent model selection based on performance, cost, and availability.

.. code-block:: python

    from orchestrator.models.model_registry import ModelRegistry
    from orchestrator.core.model import Model, ModelCapabilities
    
    # Initialize model registry
    registry = ModelRegistry()
    
    # Register a custom model
    custom_model = MyCustomModel(
        name="custom-gpt",
        capabilities=ModelCapabilities(
            text_generation=True,
            code_generation=True,
            max_tokens=4096
        )
    )
    registry.register_model(custom_model)
    
    # Model selection based on requirements
    best_model = registry.select_model(
        capabilities=["text_generation"],
        max_cost=0.01,
        min_speed=100
    )

Creating Custom Model Integrations
-----------------------------------

To integrate a custom model, inherit from the base ``Model`` class and implement the required methods:

.. code-block:: python

    from orchestrator.core.model import Model, ModelCapabilities, ModelMetrics
    from typing import Dict, Any, Optional
    
    class CustomModel(Model):
        """Custom model integration example."""
        
        def __init__(self, name: str, api_key: str, endpoint: str):
            capabilities = ModelCapabilities(
                text_generation=True,
                code_generation=True,
                function_calling=True,
                max_tokens=8192,
                supports_streaming=True
            )
            
            super().__init__(
                name=name,
                provider="custom",
                capabilities=capabilities
            )
            
            self.api_key = api_key
            self.endpoint = endpoint
            self.client = None
        
        async def initialize(self) -> None:
            """Initialize the model connection."""
            self.client = CustomAPIClient(
                api_key=self.api_key,
                endpoint=self.endpoint
            )
            await self.client.connect()
        
        async def generate_response(
            self,
            prompt: str,
            parameters: Dict[str, Any] = None
        ) -> str:
            """Generate a response from the model."""
            if not self.client:
                await self.initialize()
            
            response = await self.client.generate(
                prompt=prompt,
                **parameters or {}
            )
            
            # Update metrics
            self.metrics.update_metrics(
                tokens_used=response.token_count,
                cost=response.cost,
                response_time=response.duration
            )
            
            return response.text
        
        async def stream_response(
            self,
            prompt: str,
            parameters: Dict[str, Any] = None
        ):
            """Stream response from the model."""
            if not self.client:
                await self.initialize()
            
            async for chunk in self.client.stream(
                prompt=prompt,
                **parameters or {}
            ):
                yield chunk.text
        
        async def health_check(self) -> bool:
            """Check if the model is healthy and available."""
            try:
                if not self.client:
                    await self.initialize()
                return await self.client.ping()
            except Exception:
                return False
        
        async def cleanup(self) -> None:
            """Clean up resources."""
            if self.client:
                await self.client.disconnect()

Built-in Model Integrations
----------------------------

The Orchestrator includes several built-in model integrations that serve as examples:

OpenAI Integration
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.integrations.openai_model import OpenAIModel
    
    # Initialize OpenAI model
    openai_model = OpenAIModel(
        name="gpt-4",
        api_key="your-api-key",
        organization="your-org-id"
    )
    
    # Register with model registry
    registry.register_model(openai_model)

Anthropic Integration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.integrations.anthropic_model import AnthropicModel
    
    # Initialize Anthropic model
    anthropic_model = AnthropicModel(
        name="claude-3-opus",
        api_key="your-api-key"
    )
    
    registry.register_model(anthropic_model)

Hugging Face Integration
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.integrations.huggingface_model import HuggingFaceModel
    
    # Initialize Hugging Face model
    hf_model = HuggingFaceModel(
        name="microsoft/DialoGPT-medium",
        model_path="microsoft/DialoGPT-medium",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    registry.register_model(hf_model)

Local Model Integration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.integrations.ollama_model import OllamaModel
    
    # Initialize local Ollama model
    ollama_model = OllamaModel(
        name="llama2",
        model_path="llama2:7b",
        host="localhost",
        port=11434
    )
    
    registry.register_model(ollama_model)

Model Capabilities and Constraints
-----------------------------------

Define what your model can do and its limitations:

.. code-block:: python

    from orchestrator.core.model import ModelCapabilities, ModelConstraints
    
    capabilities = ModelCapabilities(
        text_generation=True,
        code_generation=True,
        function_calling=True,
        vision=False,
        audio=False,
        max_tokens=4096,
        supports_streaming=True,
        supports_functions=True,
        input_cost_per_token=0.00001,
        output_cost_per_token=0.00003
    )
    
    constraints = ModelConstraints(
        max_requests_per_minute=100,
        max_concurrent_requests=10,
        required_memory_gb=2.0,
        required_gpu_memory_gb=4.0
    )

Model Selection Strategies
--------------------------

The model registry supports multiple selection strategies:

Upper Confidence Bound (UCB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.models.model_registry import UCBModelSelector
    
    # UCB selector balances exploration vs exploitation
    selector = UCBModelSelector(
        exploration_factor=2.0,
        min_trials=5
    )
    
    registry = ModelRegistry(model_selector=selector)

Cost-Based Selection
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.models.model_registry import CostBasedSelector
    
    # Select models based on cost optimization
    selector = CostBasedSelector(
        max_cost_per_request=0.10,
        prefer_local_models=True
    )

Performance-Based Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from orchestrator.models.model_registry import PerformanceSelector
    
    # Select models based on performance metrics
    selector = PerformanceSelector(
        min_tokens_per_second=50,
        max_response_time=5.0,
        prefer_streaming=True
    )

Model Health Monitoring
-----------------------

The registry continuously monitors model health and availability:

.. code-block:: python

    from orchestrator.models.model_registry import ModelHealthMonitor
    
    # Configure health monitoring
    monitor = ModelHealthMonitor(
        check_interval=300,  # 5 minutes
        failure_threshold=3,
        recovery_threshold=2
    )
    
    registry = ModelRegistry(health_monitor=monitor)
    
    # Check model health
    health_status = await registry.get_model_health("gpt-4")
    if not health_status.is_healthy:
        print(f"Model issues: {health_status.issues}")

Model Caching and Optimization
-------------------------------

Implement caching to improve performance and reduce costs:

.. code-block:: python

    from orchestrator.core.cache import ModelCache
    
    # Configure model response caching
    cache = ModelCache(
        backend="redis",
        ttl=3600,  # 1 hour
        max_size=10000
    )
    
    # Enable caching for specific models
    model.enable_caching(cache)
    
    # Cached responses are automatically used for identical requests
    response1 = await model.generate_response("What is AI?")
    response2 = await model.generate_response("What is AI?")  # From cache

Model Quantization and Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For local models, implement quantization for resource efficiency:

.. code-block:: python

    from orchestrator.integrations.lazy_huggingface_model import LazyHuggingFaceModel
    
    # Initialize model with quantization
    quantized_model = LazyHuggingFaceModel(
        name="llama2-7b-quantized",
        model_path="meta-llama/Llama-2-7b-hf",
        quantization="int8",
        device_map="auto"
    )

Testing Custom Models
----------------------

Create comprehensive tests for your custom models:

.. code-block:: python

    import pytest
    from orchestrator.models.model_registry import ModelRegistry
    
    @pytest.mark.asyncio
    async def test_custom_model_integration():
        """Test custom model integration."""
        registry = ModelRegistry()
        
        # Initialize custom model
        custom_model = CustomModel(
            name="test-model",
            api_key="test-key",
            endpoint="http://localhost:8000"
        )
        
        # Register model
        registry.register_model(custom_model)
        
        # Test model selection
        selected = registry.select_model(
            capabilities=["text_generation"]
        )
        assert selected.name == "test-model"
        
        # Test model generation
        response = await custom_model.generate_response("Hello, world!")
        assert response is not None
        assert len(response) > 0
        
        # Test health check
        is_healthy = await custom_model.health_check()
        assert is_healthy is True
        
        # Cleanup
        await custom_model.cleanup()

Model Configuration Management
------------------------------

Configure models through YAML files:

.. code-block:: yaml

    # config/models.yaml
    models:
      - name: "gpt-4"
        provider: "openai"
        config:
          api_key: "${OPENAI_API_KEY}"
          organization: "${OPENAI_ORG_ID}"
        capabilities:
          text_generation: true
          code_generation: true
          max_tokens: 8192
        constraints:
          max_requests_per_minute: 100
          cost_per_token: 0.00003
      
      - name: "claude-3-opus"
        provider: "anthropic"
        config:
          api_key: "${ANTHROPIC_API_KEY}"
        capabilities:
          text_generation: true
          code_generation: true
          max_tokens: 200000
      
      - name: "local-llama"
        provider: "ollama"
        config:
          host: "localhost"
          port: 11434
          model_path: "llama2:7b"
        capabilities:
          text_generation: true
          max_tokens: 4096

Load models from configuration:

.. code-block:: python

    from orchestrator.models.model_registry import ModelRegistry
    from orchestrator.utils.config import load_model_config
    
    # Load model configuration
    config = load_model_config("config/models.yaml")
    
    # Initialize registry with configured models
    registry = ModelRegistry.from_config(config)

Best Practices
--------------

1. **Error Handling**: Implement robust error handling for network issues and API failures
2. **Resource Management**: Clean up resources properly to avoid memory leaks
3. **Monitoring**: Implement comprehensive health checks and monitoring
4. **Cost Optimization**: Use caching and model selection to minimize costs
5. **Testing**: Write thorough tests for all custom model integrations
6. **Documentation**: Document model capabilities and constraints clearly
7. **Security**: Protect API keys and sensitive configuration data

Example: Complete Custom Model Implementation
---------------------------------------------

Here's a complete example of a custom model integration:

.. code-block:: python

    import asyncio
    import logging
    from typing import Dict, Any, Optional
    
    from orchestrator.core.model import Model, ModelCapabilities
    from orchestrator.models.model_registry import ModelRegistry
    
    class MyCustomModel(Model):
        """Complete custom model implementation."""
        
        def __init__(self, name: str, config: Dict[str, Any]):
            capabilities = ModelCapabilities(
                text_generation=True,
                code_generation=True,
                max_tokens=4096,
                supports_streaming=True
            )
            
            super().__init__(
                name=name,
                provider="custom",
                capabilities=capabilities
            )
            
            self.config = config
            self.client = None
            self.logger = logging.getLogger(__name__)
        
        async def initialize(self) -> None:
            """Initialize the model."""
            try:
                self.client = CustomAPIClient(self.config)
                await self.client.connect()
                self.logger.info(f"Initialized model {self.name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize model {self.name}: {e}")
                raise
        
        async def generate_response(
            self,
            prompt: str,
            parameters: Dict[str, Any] = None
        ) -> str:
            """Generate response from the model."""
            if not self.client:
                await self.initialize()
            
            try:
                response = await self.client.generate(
                    prompt=prompt,
                    **parameters or {}
                )
                
                # Update metrics
                self.metrics.update_metrics(
                    tokens_used=response.token_count,
                    cost=response.cost,
                    response_time=response.duration
                )
                
                return response.text
            except Exception as e:
                self.logger.error(f"Generation failed: {e}")
                raise
        
        async def health_check(self) -> bool:
            """Check model health."""
            try:
                if not self.client:
                    await self.initialize()
                return await self.client.ping()
            except Exception:
                return False
        
        async def cleanup(self) -> None:
            """Clean up resources."""
            if self.client:
                await self.client.disconnect()
    
    # Usage example
    async def main():
        # Initialize model registry
        registry = ModelRegistry()
        
        # Create and register custom model
        custom_model = MyCustomModel(
            name="my-custom-model",
            config={
                "api_key": "your-api-key",
                "endpoint": "https://api.example.com"
            }
        )
        
        registry.register_model(custom_model)
        
        # Use the model
        response = await custom_model.generate_response(
            "Explain quantum computing"
        )
        
        print(f"Model response: {response}")
        
        # Cleanup
        await custom_model.cleanup()
    
    if __name__ == "__main__":
        asyncio.run(main())
