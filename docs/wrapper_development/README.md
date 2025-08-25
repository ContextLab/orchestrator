# Wrapper Development Framework

## Overview

The orchestrator's unified wrapper framework provides a standardized approach to integrating external tools while maintaining backward compatibility, comprehensive error handling, and robust monitoring capabilities.

## Framework Components

- **[Getting Started Guide](getting_started.md)** - Quick start guide for wrapper development
- **[Architecture Overview](architecture.md)** - Detailed architecture documentation
- **[Best Practices](best_practices.md)** - Development best practices and patterns
- **[Migration Guide](migration_guide.md)** - Migrating existing integrations
- **[API Reference](api_reference.md)** - Complete API documentation
- **[Examples](examples/)** - Working examples and templates

## Key Features

### 🎯 **Unified Architecture**
- Standardized base classes for consistent integration patterns
- Type-safe generic interfaces with runtime validation
- Zero breaking changes to existing functionality

### 🚦 **Feature Flag System**
- Hierarchical flag management with dependencies
- Multiple evaluation strategies (boolean, percentage, whitelist)
- Runtime flag updates with caching optimization

### ⚙️ **Configuration Management**
- Field-level validation with custom rules
- Environment-specific overrides
- Runtime configuration updates with audit trails

### 📊 **Comprehensive Monitoring**
- Built-in metrics collection and health monitoring
- Alert system with configurable rules
- Performance tracking and benchmarking

### 🧪 **Testing Framework**
- Mock implementations for isolated testing
- Performance benchmarking utilities
- Integration testing patterns

## Quick Example

```python
from src.orchestrator.core.wrapper_base import BaseWrapper, WrapperResult
from src.orchestrator.core.wrapper_config import BaseWrapperConfig
from dataclasses import dataclass
from typing import List

@dataclass
class MyWrapperConfig(BaseWrapperConfig):
    api_endpoint: str = "https://api.example.com"
    api_key: str = ""
    
    def get_config_fields(self):
        return {
            "api_endpoint": ConfigField("api_endpoint", str, self.api_endpoint),
            "api_key": ConfigField("api_key", str, "", sensitive=True)
        }

class MyWrapper(BaseWrapper[dict, MyWrapperConfig]):
    async def _execute_wrapper_operation(self, context, *args, **kwargs):
        # Your external tool integration logic here
        result = await self._call_external_api(kwargs.get('query', ''))
        return result
    
    async def _execute_fallback_operation(self, context, original_error=None, *args, **kwargs):
        # Fallback to original implementation
        return {"fallback": True, "error": str(original_error)}
    
    def _validate_config(self):
        return bool(self.config.api_key and self.config.api_endpoint)
    
    def get_capabilities(self):
        return [WrapperCapability.MONITORING]
```

## Architecture Benefits

### For Developers
- **Consistent Patterns**: Standardized integration approach across all wrappers
- **Type Safety**: Full generic type support with IDE assistance
- **Testing Support**: Built-in testing framework and mock implementations
- **Documentation**: Comprehensive examples and API reference

### For Operations
- **Monitoring**: Built-in metrics, health checking, and alerting
- **Configuration**: Centralized configuration with validation and overrides
- **Feature Flags**: Safe rollout and A/B testing capabilities
- **Error Handling**: Comprehensive fallback mechanisms and error tracking

### For System Reliability
- **Zero Downtime**: Graceful fallback to original implementations
- **Performance**: <5ms overhead with optimized async patterns
- **Observability**: Comprehensive logging and metrics collection
- **Scalability**: Efficient resource usage and connection pooling

## Integration Examples

The framework has been successfully used to integrate:

- **RouteLLM**: Intelligent model routing with cost optimization
- **POML**: Structured template markup processing  
- **Future Integrations**: Template for any external tool integration

## Getting Started

1. Read the [Getting Started Guide](getting_started.md)
2. Review the [Architecture Overview](architecture.md)
3. Study the [Examples](examples/)
4. Follow the [Best Practices](best_practices.md)

## Support

- **Issues**: Report issues in the main project repository
- **Examples**: See the `examples/` directory for working implementations
- **Testing**: Use the built-in testing framework for validation
- **Documentation**: Complete API reference available

---

**Framework Version**: 1.0.0  
**Last Updated**: 2025-08-25  
**Compatibility**: Python 3.8+