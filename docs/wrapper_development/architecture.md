# Wrapper Framework Architecture

## Overview

The unified wrapper framework provides a standardized approach to integrating external tools while maintaining backward compatibility, comprehensive error handling, and robust monitoring capabilities.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Wrapper Framework                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │   BaseWrapper   │    │  FeatureFlags    │    │ WrapperConfig │ │
│  │    (Generic)    │◄───┤   Manager        │    │  (Abstract)  │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
│           │                       │                      │      │
│           ▼                       ▼                      ▼      │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │ WrapperMonitoring│    │ WrapperRegistry  │    │ConfigManager│ │
│  │                 │    │                  │    │             │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Concrete Implementations                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │ RouteLLMWrapper │    │   POMLWrapper    │    │   Custom    │ │
│  │                 │    │                  │    │  Wrappers   │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. BaseWrapper<T, C>

The foundation of all wrapper implementations.

**Key Features:**
- Generic type support for type safety
- Standardized operation lifecycle
- Automatic fallback mechanisms
- Error handling and recovery
- Context management
- Health monitoring integration

**Type Parameters:**
- `T`: Return type of wrapper operations
- `C`: Configuration type (extends BaseWrapperConfig)

**Core Methods:**
```python
async def execute(operation_type: str, **kwargs) -> WrapperResult[T]
async def _execute_wrapper_operation(context: WrapperContext, *args, **kwargs) -> T
async def _execute_fallback_operation(context: WrapperContext, original_error, *args, **kwargs) -> T
def _validate_config() -> bool
def get_capabilities() -> List[WrapperCapability]
```

### 2. WrapperResult<T>

Standardized result container for all wrapper operations.

**Structure:**
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

### 3. WrapperContext

Operation context and metadata container.

**Features:**
- Unique operation tracking
- Configuration overrides
- Feature flag context
- Custom attributes
- Audit trail information

### 4. FeatureFlagManager

Comprehensive feature flag system with hierarchical support.

**Capabilities:**
- Multiple evaluation strategies (boolean, percentage, whitelist, custom)
- Hierarchical dependencies and conflicts
- Runtime flag updates
- Caching for performance
- Wrapper-specific flag patterns

**Flag Evaluation Strategies:**
```python
class FeatureFlagStrategy(Enum):
    BOOLEAN = "boolean"         # Simple on/off
    PERCENTAGE = "percentage"   # Percentage-based rollout  
    WHITELIST = "whitelist"     # Explicit user/context whitelist
    BLACKLIST = "blacklist"     # Explicit user/context blacklist
    CUSTOM = "custom"          # Custom evaluation function
```

### 5. BaseWrapperConfig

Abstract configuration base class with validation framework.

**Features:**
- Field-level validation with custom rules
- Environment variable integration
- Sensitive data masking
- Runtime configuration updates
- Audit trail tracking

**Configuration Fields:**
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
    validator: Optional[callable] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
```

### 6. WrapperMonitoring

Comprehensive monitoring and metrics collection system.

**Capabilities:**
- Operation tracking with unique IDs
- Performance metrics collection
- Health scoring and status reporting
- Alert system with configurable rules
- Export capabilities for external systems

**Monitoring Data Flow:**
```
Operation Start → Metrics Collection → Health Analysis → Alert Evaluation → Reporting
```

### 7. WrapperRegistry

Centralized wrapper management and discovery system.

**Features:**
- Wrapper registration and discovery
- Capability-based querying
- Health status aggregation
- Batch operations
- System-wide health reporting

## Design Patterns

### 1. Generic Type Safety

```python
class MyWrapper(BaseWrapper[MyResultType, MyConfigType]):
    async def _execute_wrapper_operation(self, context, *args, **kwargs) -> MyResultType:
        # Implementation returns MyResultType
        pass
```

**Benefits:**
- Compile-time type checking
- IDE autocomplete and validation
- Runtime type safety
- Clear interface contracts

### 2. Adapter Pattern

Wrappers act as adapters between the orchestrator and external tools:

```python
# External Tool API
external_result = await external_tool.complex_api_call(params)

# Wrapper Adapter
return StandardizedResult(
    success=True,
    data=transform_to_standard_format(external_result),
    metrics=extract_metrics(external_result)
)
```

### 3. Circuit Breaker Pattern

Built-in circuit breaker functionality:

```python
async def execute(self, **kwargs):
    if self._should_use_circuit_breaker():
        return await self._execute_fallback_operation(context, None, **kwargs)
    
    try:
        return await self._execute_wrapper_operation(context, **kwargs)
    except Exception as e:
        self._record_failure()
        return await self._execute_fallback_operation(context, e, **kwargs)
```

### 4. Observer Pattern

Monitoring system observes all wrapper operations:

```python
# Automatic monitoring integration
operation_id = monitoring.start_operation(wrapper_name, operation_type)
try:
    result = await wrapper_operation()
    monitoring.record_success(operation_id, result)
except Exception as e:
    monitoring.record_error(operation_id, str(e))
finally:
    monitoring.end_operation(operation_id)
```

### 5. Strategy Pattern

Feature flags implement strategy pattern for evaluation:

```python
def evaluate(self, context):
    if self.strategy == FeatureFlagStrategy.PERCENTAGE:
        return self._evaluate_percentage(context)
    elif self.strategy == FeatureFlagStrategy.WHITELIST:
        return self._evaluate_whitelist(context)
    # ... other strategies
```

## Data Flow

### 1. Operation Execution Flow

```
1. Client Request
2. Feature Flag Evaluation
3. Configuration Validation
4. Operation Context Creation
5. Monitoring Start
6. Wrapper Operation Execution
   ├─ Success → Result Processing
   └─ Failure → Fallback Execution
7. Monitoring End
8. Result Return
```

### 2. Configuration Flow

```
1. Base Configuration (defaults)
2. File Configuration Loading
3. Environment Variable Overrides
4. Runtime Configuration Updates
5. Validation and Error Handling
6. Configuration Application
```

### 3. Monitoring Flow

```
1. Operation Start Event
2. Metrics Collection During Execution
3. Success/Error/Fallback Recording  
4. Health Status Calculation
5. Alert Rule Evaluation
6. System Health Aggregation
```

## Error Handling Strategy

### 1. Exception Hierarchy

```python
WrapperException (base)
├── WrapperConfigurationError
├── WrapperInitializationError  
├── WrapperOperationError
└── WrapperTimeoutError
```

### 2. Fallback Cascade

```
Primary Operation Failure
↓
Fallback Operation Attempt
↓
Success → Fallback Result
↓
Failure → Error Result with Comprehensive Logging
```

### 3. Error Context Preservation

```python
try:
    result = await wrapper_operation()
except ExternalServiceError as e:
    # Preserve original error context
    fallback_result = await fallback_operation()
    return WrapperResult.fallback_result(
        fallback_result,
        fallback_reason=f"external_service_error: {e}",
        original_error=str(e)
    )
```

## Performance Considerations

### 1. Async/Await Throughout

- All operations are async for non-blocking execution
- Proper resource management with async context managers
- Connection pooling and resource reuse

### 2. Caching Strategies

- Feature flag evaluation caching (TTL-based)
- Configuration caching with invalidation
- Connection pooling for external services

### 3. Monitoring Overhead

- Lightweight metrics collection (<1ms overhead)
- Efficient data structures (deque, defaultdict)
- Background cleanup of old metrics

### 4. Memory Management

- Bounded collections with automatic cleanup
- Weak references where appropriate
- Resource lifecycle management

## Security Considerations

### 1. Sensitive Data Handling

- Configuration fields marked as sensitive
- Automatic masking in logs and exports
- Secure configuration file permissions

### 2. Input Validation

- Configuration validation with custom rules
- Operation parameter validation
- Sanitization of logged data

### 3. Error Information Leakage

- Careful error message construction
- Separation of internal vs. external error details
- Audit logging of security-relevant events

## Scalability Features

### 1. Resource Efficiency

- Minimal memory footprint
- Efficient data structures
- Lazy initialization patterns

### 2. Horizontal Scaling

- Stateless wrapper design
- External configuration management
- Distributed monitoring capabilities

### 3. Performance Monitoring

- Built-in performance tracking
- Bottleneck identification
- Scalability metrics collection

## Testing Architecture

### 1. Mock Framework

- Complete mock implementations
- Configurable behavior simulation
- Isolated testing capabilities

### 2. Test Harness

- Systematic test scenario execution
- Performance benchmarking
- Integration testing patterns

### 3. Quality Assurance

- Comprehensive test coverage
- Integration validation
- Performance regression testing

## Extension Points

### 1. Custom Wrappers

- Clear inheritance patterns
- Well-defined abstract methods
- Comprehensive base functionality

### 2. Monitoring Extensions

- Custom alert rules
- External monitoring integration
- Custom metrics collection

### 3. Configuration Extensions

- Custom validation rules
- External configuration sources
- Dynamic configuration updates

This architecture provides a solid foundation for building reliable, maintainable, and scalable external tool integrations while maintaining consistency and operational excellence across the entire system.