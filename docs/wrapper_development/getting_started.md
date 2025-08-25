# Getting Started with Wrapper Development

This guide will help you create your first wrapper using the unified wrapper framework.

## Prerequisites

- Python 3.8+
- Basic understanding of async/await patterns
- Familiarity with dataclasses and type hints

## Step 1: Define Your Configuration

Every wrapper needs a configuration class that extends `BaseWrapperConfig`:

```python
from dataclasses import dataclass
from src.orchestrator.core.wrapper_config import BaseWrapperConfig, ConfigField

@dataclass
class MyServiceConfig(BaseWrapperConfig):
    """Configuration for MyService wrapper."""
    
    # Service-specific settings
    api_endpoint: str = "https://api.myservice.com"
    api_key: str = ""
    timeout_override: float = 60.0
    max_retries: int = 5
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        """Define configuration fields with validation."""
        return {
            "api_endpoint": ConfigField(
                "api_endpoint", str, self.api_endpoint,
                description="API endpoint URL",
                validator=lambda x: x.startswith('http')
            ),
            "api_key": ConfigField(
                "api_key", str, "",
                description="API authentication key",
                required=True,
                sensitive=True  # Will be masked in logs
            ),
            "timeout_override": ConfigField(
                "timeout_override", float, 60.0,
                description="Request timeout in seconds",
                min_value=1.0, max_value=300.0
            ),
            "max_retries": ConfigField(
                "max_retries", int, 5,
                description="Maximum retry attempts",
                min_value=0, max_value=10
            )
        }
```

## Step 2: Implement Your Wrapper

Create your wrapper class by extending `BaseWrapper`:

```python
from typing import Dict, Any, Optional
from src.orchestrator.core.wrapper_base import (
    BaseWrapper, WrapperContext, WrapperCapability, WrapperException
)

class MyServiceWrapper(BaseWrapper[Dict[str, Any], MyServiceConfig]):
    """Wrapper for MyService external API."""
    
    def __init__(self, name: str, config: MyServiceConfig, **kwargs):
        super().__init__(name, config, **kwargs)
        self._client = None
    
    async def _execute_wrapper_operation(
        self, 
        context: WrapperContext,
        *args, 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the main wrapper operation."""
        
        # Initialize client if needed
        if self._client is None:
            self._client = await self._initialize_client()
        
        # Extract parameters
        query = kwargs.get('query', '')
        if not query:
            raise WrapperException("Query parameter is required", wrapper_name=self.name)
        
        # Call external service
        try:
            response = await self._client.process_query(
                query=query,
                timeout=self.config.timeout_override
            )
            
            return {
                "success": True,
                "result": response.data,
                "service_id": response.id,
                "processing_time": response.processing_time
            }
            
        except Exception as e:
            raise WrapperException(f"Service call failed: {e}", wrapper_name=self.name)
    
    async def _execute_fallback_operation(
        self, 
        context: WrapperContext,
        original_error: Optional[Exception] = None,
        *args, 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute fallback when wrapper fails."""
        
        # Implement your fallback logic here
        # This could be a simpler local implementation or cached result
        
        return {
            "success": True,
            "result": f"Fallback result for: {kwargs.get('query', 'unknown')}",
            "fallback_reason": str(original_error) if original_error else "wrapper_disabled",
            "source": "fallback"
        }
    
    def _validate_config(self) -> bool:
        """Validate wrapper configuration."""
        return (
            bool(self.config.api_key) and
            bool(self.config.api_endpoint) and
            self.config.timeout_override > 0 and
            self.config.max_retries >= 0
        )
    
    def get_capabilities(self) -> List[WrapperCapability]:
        """Return capabilities provided by this wrapper."""
        return [
            WrapperCapability.MONITORING,
            WrapperCapability.VALIDATION
        ]
    
    async def _initialize_client(self):
        """Initialize the external service client."""
        # Your client initialization logic here
        from myservice_sdk import MyServiceClient
        
        client = MyServiceClient(
            api_key=self.config.api_key,
            endpoint=self.config.api_endpoint,
            timeout=self.config.timeout_override
        )
        
        # Test connection
        await client.health_check()
        
        return client
```

## Step 3: Set Up Feature Flags

Register feature flags for your wrapper:

```python
from src.orchestrator.core.feature_flags import FeatureFlagManager, FeatureFlag

# Create feature flag manager
feature_flags = FeatureFlagManager()

# Register wrapper-specific flags (this creates standard flags automatically)
feature_flags.register_wrapper_flags("myservice")

# Add custom flags if needed
custom_flag = FeatureFlag(
    name="myservice_advanced_features",
    enabled=False,
    description="Enable advanced MyService features",
    dependencies=["myservice_enabled"]
)
feature_flags.register_flag(custom_flag)

# Enable the wrapper
feature_flags.enable_flag("myservice_enabled")
```

## Step 4: Configure Monitoring

Set up monitoring for your wrapper:

```python
from src.orchestrator.core.wrapper_monitoring import WrapperMonitoring

# Create monitoring instance
monitoring = WrapperMonitoring(retention_days=30)

# The wrapper will automatically use this for metrics collection
```

## Step 5: Create and Use Your Wrapper

```python
async def main():
    # Create configuration
    config = MyServiceConfig(
        enabled=True,
        api_key="your-api-key-here",
        api_endpoint="https://api.myservice.com",
        timeout_override=30.0
    )
    
    # Create wrapper instance
    wrapper = MyServiceWrapper(
        name="my_service_wrapper",
        config=config,
        feature_flags=feature_flags,
        monitoring=monitoring
    )
    
    # Execute operation
    result = await wrapper.execute(
        operation_type="query_processing",
        query="What is the weather today?"
    )
    
    if result.success:
        print(f"Result: {result.data}")
        if result.fallback_used:
            print(f"Used fallback due to: {result.fallback_reason}")
    else:
        print(f"Operation failed: {result.error}")
    
    # Check wrapper health
    health = wrapper.get_health_info()
    print(f"Wrapper health: {health}")

# Run the example
import asyncio
asyncio.run(main())
```

## Step 6: Add Tests

Create tests using the wrapper testing framework:

```python
import pytest
from src.orchestrator.core.wrapper_testing import (
    WrapperTestHarness, TestScenario, create_basic_scenarios
)

class TestMyServiceWrapper:
    @pytest.fixture
    def test_harness(self):
        return WrapperTestHarness(MyServiceWrapper, MyServiceConfig)
    
    @pytest.mark.asyncio
    async def test_successful_operation(self, test_harness):
        # Add test scenario
        scenario = TestScenario(
            name="successful_query",
            description="Test successful query processing",
            inputs={"query": "test query"},
            expected_outputs={"success": True},
            should_succeed=True
        )
        test_harness.add_test_scenario(scenario)
        
        # Run test
        results = await test_harness.run_all_scenarios(
            config_overrides={"api_key": "test-key"}
        )
        
        assert len(results) == 1
        assert results[0].success
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, test_harness):
        scenario = TestScenario(
            name="performance_test",
            description="Performance benchmark",
            inputs={"query": "benchmark query"},
            expected_outputs={"success": True},
            should_succeed=True
        )
        
        benchmark = await test_harness.run_performance_benchmark(
            scenario, iterations=50
        )
        
        assert benchmark.success_rate >= 0.9
        assert benchmark.average_time_ms < 100  # Should be fast
```

## Step 7: Configuration Management

Set up configuration management:

```python
from src.orchestrator.core.wrapper_config import ConfigurationManager
from pathlib import Path

# Create configuration manager
config_manager = ConfigurationManager(
    config_dir=Path("config"),
    environment="production"
)

# Register your config type
config_manager.register_config_type("myservice", MyServiceConfig)

# Configuration will be automatically loaded from:
# - config/wrappers.json (base config)
# - config/wrappers-production.json (environment specific)
# - Environment variables (WRAPPER_MYSERVICE_*)

# Get configuration
config = config_manager.get_config("myservice")
```

## Common Patterns

### Error Handling
```python
async def _execute_wrapper_operation(self, context, *args, **kwargs):
    try:
        # Your operation logic
        result = await external_service.call()
        return result
    except ExternalServiceTimeout as e:
        # Let this bubble up for automatic fallback
        raise WrapperException(f"Service timeout: {e}", wrapper_name=self.name)
    except ExternalServiceRateLimit as e:
        # Could implement retry logic here
        await asyncio.sleep(1)
        raise WrapperException(f"Rate limited: {e}", wrapper_name=self.name)
```

### Resource Management
```python
def __init__(self, name: str, config: MyConfig, **kwargs):
    super().__init__(name, config, **kwargs)
    self._connection_pool = None

async def _initialize_resources(self):
    if self._connection_pool is None:
        self._connection_pool = await create_connection_pool(
            max_connections=10,
            timeout=self.config.timeout_seconds
        )

async def _cleanup_resources(self):
    if self._connection_pool:
        await self._connection_pool.close()
```

### Custom Metrics
```python
async def _execute_wrapper_operation(self, context, *args, **kwargs):
    start_time = time.time()
    
    result = await self._perform_operation(**kwargs)
    
    # Add custom metrics
    processing_time = time.time() - start_time
    context.set_attribute("processing_time", processing_time)
    context.set_attribute("result_size", len(str(result)))
    
    return result
```

## Next Steps

1. **Read the [Architecture Overview](architecture.md)** - Understand the framework design
2. **Review [Best Practices](best_practices.md)** - Learn recommended patterns  
3. **Study [Examples](examples/)** - See complete wrapper implementations
4. **Set up [Monitoring](monitoring.md)** - Configure monitoring and alerting

## Troubleshooting

### Common Issues

**Wrapper not starting**: Check configuration validation in `_validate_config()`
**Operations timing out**: Adjust `timeout_seconds` in configuration
**Feature flags not working**: Ensure flags are registered and enabled
**Tests failing**: Verify mock configurations and expected outputs

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your wrapper configuration
config = MyServiceConfig(debug_logging=True, ...)
```

### Health Monitoring

Check wrapper health programmatically:

```python
health = wrapper.get_health_info()
print(f"Status: {health['status']}")
print(f"Enabled: {health['enabled']}")
print(f"Last Error: {health.get('initialization_error')}")
```