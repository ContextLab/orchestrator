# Wrapper API Reference

## Overview

This document provides comprehensive API reference for the unified wrapper framework, including all classes, methods, and configuration options.

## Table of Contents

- [Core Classes](#core-classes)
- [Configuration System](#configuration-system) 
- [Feature Flags](#feature-flags)
- [Monitoring](#monitoring)
- [Data Types](#data-types)
- [Examples](#examples)

## Core Classes

### BaseWrapper<T, C>

The foundation class for all wrapper implementations.

```python
from src.orchestrator.core.wrapper_base import BaseWrapper, WrapperResult, WrapperContext

class BaseWrapper(Generic[T, C]):
    """Base class for all wrapper implementations."""
```

#### Constructor

```python
def __init__(self, config: C, flag_manager: Optional[FeatureFlagManager] = None, 
             monitoring: Optional[WrapperMonitoring] = None)
```

**Parameters:**
- `config`: Configuration object extending `BaseWrapperConfig`
- `flag_manager`: Optional feature flag manager instance
- `monitoring`: Optional monitoring system instance

#### Core Methods

##### execute()

Execute a wrapper operation with automatic fallback handling.

```python
async def execute(self, operation_type: str, context_overrides: Optional[Dict[str, Any]] = None, 
                 **kwargs) -> WrapperResult[T]
```

**Parameters:**
- `operation_type`: String identifier for the operation type
- `context_overrides`: Optional context overrides
- `**kwargs`: Operation-specific parameters

**Returns:** `WrapperResult[T]` containing operation results

**Example:**
```python
wrapper = MyWrapper(config)
result = await wrapper.execute("query", query="test query", timeout=10.0)

if result.success:
    print(f"Data: {result.data}")
else:
    print(f"Error: {result.error}")
    if result.fallback_used:
        print(f"Fallback reason: {result.fallback_reason}")
```

##### _execute_wrapper_operation() (Abstract)

Override this method to implement your wrapper logic.

```python
async def _execute_wrapper_operation(self, context: WrapperContext, *args, **kwargs) -> T
```

**Parameters:**
- `context`: Wrapper execution context
- `*args, **kwargs`: Operation parameters

**Returns:** Operation result of type `T`

**Example:**
```python
async def _execute_wrapper_operation(self, context: WrapperContext, query: str, **kwargs) -> Dict[str, Any]:
    # Your external API call logic here
    response = await self.external_api.query(query)
    return {"result": response, "timestamp": time.time()}
```

##### _execute_fallback_operation() (Abstract)

Override this method to implement fallback logic.

```python
async def _execute_fallback_operation(self, context: WrapperContext, 
                                    original_error: Optional[Exception] = None, 
                                    *args, **kwargs) -> T
```

**Parameters:**
- `context`: Wrapper execution context  
- `original_error`: Exception that triggered fallback (if any)
- `*args, **kwargs`: Original operation parameters

**Returns:** Fallback result of type `T`

**Example:**
```python
async def _execute_fallback_operation(self, context: WrapperContext, 
                                    original_error: Optional[Exception] = None,
                                    query: str = "", **kwargs) -> Dict[str, Any]:
    # Fallback to local cache or default response
    return {
        "result": "fallback_response",
        "error": str(original_error) if original_error else None,
        "fallback": True
    }
```

##### _validate_config() (Abstract)

Override this method to validate wrapper configuration.

```python
def _validate_config(self) -> bool
```

**Returns:** `True` if configuration is valid, `False` otherwise

**Example:**
```python
def _validate_config(self) -> bool:
    return bool(self.config.api_key and self.config.endpoint)
```

##### get_capabilities() (Abstract)

Override this method to declare wrapper capabilities.

```python
def get_capabilities(self) -> List[WrapperCapability]
```

**Returns:** List of `WrapperCapability` enum values

**Example:**
```python
def get_capabilities(self) -> List[WrapperCapability]:
    return [
        WrapperCapability.MONITORING,
        WrapperCapability.FALLBACK,
        WrapperCapability.CONFIGURATION_MANAGEMENT
    ]
```

#### Utility Methods

##### get_health()

Get current wrapper health status.

```python
def get_health(self) -> WrapperHealth
```

**Returns:** `WrapperHealth` object with current health metrics

##### cleanup()

Clean up wrapper resources.

```python
async def cleanup(self) -> None
```

**Example:**
```python
await wrapper.cleanup()
```

### WrapperResult<T>

Container for wrapper operation results.

```python
@dataclass
class WrapperResult(Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    execution_time_ms: Optional[float] = None
    operation_id: Optional[str] = None
```

#### Factory Methods

##### success_result()

Create a successful result.

```python
@classmethod
def success_result(cls, data: T, metrics: Optional[Dict[str, Any]] = None, 
                  execution_time_ms: Optional[float] = None) -> WrapperResult[T]
```

**Example:**
```python
result = WrapperResult.success_result(
    data={"response": "success"}, 
    execution_time_ms=150.0
)
```

##### error_result()

Create an error result.

```python
@classmethod  
def error_result(cls, error: str, error_code: Optional[str] = None,
                execution_time_ms: Optional[float] = None) -> WrapperResult[T]
```

**Example:**
```python
result = WrapperResult.error_result(
    error="API timeout",
    error_code="TIMEOUT"
)
```

##### fallback_result()

Create a fallback result.

```python
@classmethod
def fallback_result(cls, data: T, fallback_reason: str,
                   original_error: Optional[str] = None) -> WrapperResult[T]
```

**Example:**
```python
result = WrapperResult.fallback_result(
    data={"fallback": "cached_response"},
    fallback_reason="API unavailable"
)
```

### WrapperContext

Execution context for wrapper operations.

```python
@dataclass
class WrapperContext:
    operation_id: str
    operation_type: str
    start_time: datetime
    timeout_seconds: Optional[float] = None
    feature_flags: Optional[Dict[str, Any]] = None
    config_overrides: Optional[Dict[str, Any]] = None
    custom_attributes: Optional[Dict[str, Any]] = None
```

#### Methods

##### get_feature_flag()

Get feature flag value with fallback.

```python
def get_feature_flag(self, flag_name: str, default: Any = False) -> Any
```

##### get_config_override()

Get configuration override value.

```python
def get_config_override(self, key: str, default: Any = None) -> Any
```

##### add_custom_attribute()

Add custom attribute to context.

```python
def add_custom_attribute(self, key: str, value: Any) -> None
```

## Configuration System

### BaseWrapperConfig

Abstract base class for wrapper configurations.

```python
from src.orchestrator.core.wrapper_config import BaseWrapperConfig, ConfigField

@dataclass
class BaseWrapperConfig:
    """Base configuration class for wrappers."""
    
    # Common fields
    enabled: bool = True
    timeout_seconds: float = 30.0
    max_retry_attempts: int = 3
    enable_monitoring: bool = True
```

#### Abstract Methods

##### get_config_fields()

Define configuration fields with validation.

```python
def get_config_fields(self) -> Dict[str, ConfigField]
```

**Returns:** Dictionary of field names to `ConfigField` objects

**Example:**
```python
def get_config_fields(self) -> Dict[str, ConfigField]:
    return {
        "api_key": ConfigField(
            name="api_key",
            field_type=str,
            default_value="",
            description="API authentication key", 
            required=True,
            sensitive=True,
            environment_var="MY_API_KEY"
        ),
        "timeout_seconds": ConfigField(
            name="timeout_seconds",
            field_type=float,
            default_value=30.0,
            description="Request timeout in seconds",
            min_value=1.0,
            max_value=300.0
        )
    }
```

### ConfigField

Configuration field definition with validation rules.

```python
@dataclass
class ConfigField:
    name: str
    field_type: Type
    default_value: Any
    description: str = ""
    required: bool = False
    sensitive: bool = False
    environment_var: Optional[str] = None
    validator: Optional[Callable] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
```

**Field Parameters:**
- `name`: Field identifier
- `field_type`: Python type for the field
- `default_value`: Default value if not provided
- `description`: Human-readable description
- `required`: Whether field is required
- `sensitive`: Whether field contains sensitive data (masked in logs)
- `environment_var`: Environment variable to check for value
- `validator`: Custom validation function
- `min_value`: Minimum numeric value
- `max_value`: Maximum numeric value  
- `allowed_values`: List of allowed values

### ConfigManager

Configuration management and validation system.

```python
from src.orchestrator.core.wrapper_config import ConfigManager

manager = ConfigManager()
```

#### Methods

##### validate_config()

Validate configuration object.

```python
def validate_config(self, config: BaseWrapperConfig) -> ValidationResult
```

##### load_config_from_env()

Load configuration from environment variables.

```python
def load_config_from_env(self, config_class: Type[BaseWrapperConfig]) -> BaseWrapperConfig
```

##### export_config()

Export configuration (with sensitive data masked).

```python
def export_config(self, config: BaseWrapperConfig, include_sensitive: bool = False) -> Dict[str, Any]
```

## Feature Flags

### FeatureFlagManager

Comprehensive feature flag management system.

```python
from src.orchestrator.core.feature_flags import FeatureFlagManager, FeatureFlag

manager = FeatureFlagManager()
```

#### Methods

##### register_flag()

Register a new feature flag.

```python
def register_flag(self, flag: FeatureFlag) -> None
```

**Example:**
```python
flag = FeatureFlag(
    name="my_wrapper_enabled",
    description="Enable my wrapper integration",
    strategy=FeatureFlagStrategy.PERCENTAGE,
    default_value=0,  # 0% initially
    metadata={"rollout_target": 100}
)

manager.register_flag(flag)
```

##### is_enabled()

Check if a feature flag is enabled.

```python
def is_enabled(self, flag_name: str, context: Optional[Dict[str, Any]] = None) -> bool
```

**Example:**
```python
if manager.is_enabled("my_wrapper_enabled", {"user_id": "12345"}):
    # Use wrapper
    pass
else:
    # Use fallback
    pass
```

##### update_flag()

Update feature flag value.

```python
def update_flag(self, flag_name: str, value: Any) -> None
```

##### create_flag()

Create and register a simple boolean flag.

```python
def create_flag(self, name: str, default_value: bool = False, description: str = "") -> None
```

### FeatureFlag

Feature flag definition.

```python
@dataclass
class FeatureFlag:
    name: str
    description: str
    strategy: FeatureFlagStrategy
    default_value: Any
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### FeatureFlagStrategy

Feature flag evaluation strategies.

```python
class FeatureFlagStrategy(Enum):
    BOOLEAN = "boolean"           # Simple on/off
    PERCENTAGE = "percentage"     # Percentage-based rollout
    WHITELIST = "whitelist"       # Explicit allow list  
    BLACKLIST = "blacklist"       # Explicit deny list
    CUSTOM = "custom"            # Custom evaluation function
```

## Monitoring

### WrapperMonitoring

Comprehensive monitoring and metrics system.

```python
from src.orchestrator.core.wrapper_monitoring import WrapperMonitoring

monitoring = WrapperMonitoring()
```

#### Methods

##### start_operation()

Start tracking an operation.

```python
def start_operation(self, wrapper_name: str, operation_type: str) -> str
```

**Returns:** Operation ID for tracking

##### record_success()

Record successful operation completion.

```python
def record_success(self, operation_id: str, result_data: Any) -> None
```

##### record_error()

Record operation error.

```python
def record_error(self, operation_id: str, error: str, error_code: Optional[str] = None) -> None
```

##### record_fallback()

Record fallback usage.

```python  
def record_fallback(self, operation_id: str, fallback_reason: str) -> None
```

##### end_operation()

End operation tracking.

```python
def end_operation(self, operation_id: str) -> None
```

##### get_wrapper_health()

Get health metrics for a specific wrapper.

```python
def get_wrapper_health(self, wrapper_name: str) -> WrapperHealth
```

##### get_system_health()

Get overall system health.

```python
def get_system_health(self) -> SystemHealth
```

### WrapperHealth

Health metrics for a specific wrapper.

```python
@dataclass
class WrapperHealth:
    wrapper_name: str
    status: HealthStatus
    last_success: Optional[datetime]
    last_error: Optional[datetime]
    success_rate: float
    error_rate: float
    fallback_rate: float
    avg_response_time_ms: float
    operations_per_minute: float
    health_score: float  # 0.0 to 1.0
```

### SystemHealth

Overall system health status.

```python
@dataclass
class SystemHealth:
    overall_status: HealthStatus
    wrapper_healths: Dict[str, WrapperHealth]
    active_alerts: List[Alert]
    system_metrics: Dict[str, Any]
    last_updated: datetime
```

### AlertRule

Alert rule configuration.

```python
@dataclass
class AlertRule:
    name: str
    description: str
    condition: AlertCondition
    threshold: float
    window_minutes: int
    severity: str
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### AlertCondition

Alert condition types.

```python
class AlertCondition(Enum):
    ERROR_RATE_THRESHOLD = "error_rate_threshold"
    RESPONSE_TIME_THRESHOLD = "response_time_threshold"
    FALLBACK_RATE_THRESHOLD = "fallback_rate_threshold"
    SUCCESS_RATE_THRESHOLD = "success_rate_threshold"
    HEALTH_SCORE_THRESHOLD = "health_score_threshold"
```

## Data Types

### WrapperCapability

Wrapper capability declarations.

```python
class WrapperCapability(Enum):
    MONITORING = "monitoring"
    FALLBACK = "fallback"
    CONFIGURATION_MANAGEMENT = "configuration_management"
    FEATURE_FLAGS = "feature_flags"
    CACHING = "caching"
    RATE_LIMITING = "rate_limiting"
    CIRCUIT_BREAKER = "circuit_breaker"
```

### HealthStatus

Health status enumeration.

```python
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
```

### ValidationResult

Configuration validation result.

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    field_errors: Dict[str, List[str]]
```

## Examples

### Basic Wrapper Implementation

```python
from dataclasses import dataclass
from typing import Dict, Any
import aiohttp
from src.orchestrator.core.wrapper_base import BaseWrapper, WrapperResult, WrapperContext
from src.orchestrator.core.wrapper_config import BaseWrapperConfig, ConfigField

@dataclass
class HTTPAPIConfig(BaseWrapperConfig):
    base_url: str = "https://api.example.com"
    api_key: str = ""
    timeout_seconds: float = 30.0
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        return {
            "base_url": ConfigField(
                name="base_url",
                field_type=str,
                default_value=self.base_url,
                description="API base URL",
                required=True
            ),
            "api_key": ConfigField(
                name="api_key", 
                field_type=str,
                default_value="",
                description="API authentication key",
                required=True,
                sensitive=True,
                environment_var="HTTP_API_KEY"
            )
        }

class HTTPAPIWrapper(BaseWrapper[Dict[str, Any], HTTPAPIConfig]):
    async def _execute_wrapper_operation(self, context: WrapperContext, 
                                       endpoint: str, **kwargs) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            url = f"{self.config.base_url}/{endpoint}"
            
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
    
    async def _execute_fallback_operation(self, context: WrapperContext,
                                        original_error: Exception = None,
                                        **kwargs) -> Dict[str, Any]:
        return {
            "error": str(original_error),
            "fallback": True,
            "data": None
        }
    
    def _validate_config(self) -> bool:
        return bool(self.config.base_url and self.config.api_key)
    
    def get_capabilities(self) -> List[WrapperCapability]:
        return [WrapperCapability.MONITORING, WrapperCapability.FALLBACK]

# Usage
async def main():
    config = HTTPAPIConfig(api_key="your-api-key")
    wrapper = HTTPAPIWrapper(config)
    
    result = await wrapper.execute("query", endpoint="users")
    
    if result.success:
        print(f"Data: {result.data}")
    else:
        print(f"Error: {result.error}")

asyncio.run(main())
```

### Advanced Feature Flag Usage

```python
from src.orchestrator.core.feature_flags import FeatureFlagManager, FeatureFlag, FeatureFlagStrategy

# Setup feature flags
manager = FeatureFlagManager()

# Percentage rollout flag
rollout_flag = FeatureFlag(
    name="new_api_rollout",
    description="Gradual rollout of new API",
    strategy=FeatureFlagStrategy.PERCENTAGE,
    default_value=0,  # Start at 0%
    metadata={"target": 100, "increment": 10}
)

# Whitelist flag for beta users
beta_flag = FeatureFlag(
    name="beta_features",
    description="Beta features for selected users",
    strategy=FeatureFlagStrategy.WHITELIST,
    default_value=["user123", "user456"]
)

manager.register_flag(rollout_flag)
manager.register_flag(beta_flag)

# Usage in wrapper
class AdvancedWrapper(BaseWrapper):
    async def _execute_wrapper_operation(self, context: WrapperContext, **kwargs):
        # Check feature flags
        use_new_api = context.get_feature_flag("new_api_rollout", False)
        is_beta_user = context.get_feature_flag("beta_features", False)
        
        if use_new_api:
            return await self._new_api_call(**kwargs)
        else:
            return await self._legacy_api_call(**kwargs)
```

### Monitoring Integration

```python
from src.orchestrator.core.wrapper_monitoring import WrapperMonitoring, AlertRule, AlertCondition

# Setup monitoring
monitoring = WrapperMonitoring()

# Add alert rules
error_alert = AlertRule(
    name="high_error_rate",
    description="Alert when error rate exceeds 5%",
    condition=AlertCondition.ERROR_RATE_THRESHOLD,
    threshold=0.05,
    window_minutes=5,
    severity="high"
)

response_time_alert = AlertRule(
    name="slow_response",
    description="Alert when response time exceeds 2 seconds",
    condition=AlertCondition.RESPONSE_TIME_THRESHOLD,
    threshold=2000,  # milliseconds
    window_minutes=10,
    severity="medium"
)

monitoring.add_alert_rule(error_alert)
monitoring.add_alert_rule(response_time_alert)

# Wrapper with monitoring
class MonitoredWrapper(BaseWrapper):
    def __init__(self, config):
        super().__init__(config, monitoring=monitoring)
    
    async def _execute_wrapper_operation(self, context: WrapperContext, **kwargs):
        # Monitoring is automatic, just implement your logic
        result = await self._external_api_call(**kwargs)
        return result

# Check health
health = monitoring.get_system_health()
print(f"System status: {health.overall_status}")
for name, wrapper_health in health.wrapper_healths.items():
    print(f"{name}: {wrapper_health.health_score:.2f}")
```

## Error Handling

All wrapper methods can raise the following exceptions:

### WrapperException

Base exception for all wrapper-related errors.

```python
class WrapperException(Exception):
    """Base exception for wrapper errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
```

### WrapperConfigurationError

Raised when configuration is invalid.

```python
class WrapperConfigurationError(WrapperException):
    """Raised when wrapper configuration is invalid."""
```

### WrapperOperationError

Raised when wrapper operation fails.

```python
class WrapperOperationError(WrapperException):
    """Raised when wrapper operation fails."""
```

### WrapperTimeoutError

Raised when wrapper operation times out.

```python
class WrapperTimeoutError(WrapperException):
    """Raised when wrapper operation times out."""
```

## Version Information

- **API Version**: 1.0.0
- **Framework Version**: Compatible with orchestrator-framework 2.0+
- **Python Version**: Requires Python 3.8+

## OpenAPI Specification

Complete OpenAPI specification for REST endpoints is available at: `/docs/api/wrappers/openapi.yaml`

This API reference provides comprehensive documentation for building, configuring, and operating wrapper integrations within the orchestrator framework.