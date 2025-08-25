# Issue #249: Wrapper Architecture - Comprehensive Analysis and Implementation Plan

**Task**: Create unified wrapper architecture that consolidates patterns from completed RouteLLM (#248) and POML (#250) integrations while incorporating lessons from Deep Agents evaluation (#253).

**Created**: 2025-08-25  
**Status**: Implementation Phase  
**Epic**: explore-wrappers  

## Executive Summary

This analysis outlines the creation of a unified wrapper architecture that standardizes external tool integration patterns across the orchestrator codebase. By extracting common patterns from the successfully completed RouteLLM and POML integrations and incorporating lessons learned from the Deep Agents evaluation, we will create a robust framework for future external tool integrations.

## Analysis of Completed Integrations

### RouteLLM Integration (#248) - Patterns Identified

**Key Architectural Components:**
- **Configuration Management**: `RouteLLMConfig` dataclass with comprehensive settings
- **Feature Flag System**: `FeatureFlags` class with domain-specific controls
- **Decision Framework**: `RoutingDecision` for intelligent routing logic  
- **Metrics Tracking**: `RoutingMetrics` and `CostTracker` for performance monitoring
- **Fallback Mechanisms**: Graceful degradation to original implementation
- **Error Handling**: Comprehensive exception handling with logging

**Design Patterns Used:**
- **Wrapper Pattern**: Enhanced existing `DomainRouter` without breaking API
- **Strategy Pattern**: Multiple router types (mf, bert, causal_llm)
- **Observer Pattern**: Cost tracking and performance monitoring
- **Circuit Breaker Pattern**: Fallback on failures
- **Factory Pattern**: Dynamic router configuration

**Key Success Factors:**
- Zero breaking changes (100% backward compatibility)
- Feature flags for safe gradual rollout
- Comprehensive monitoring and cost tracking
- Multiple layers of fallback protection
- Production-ready error handling and logging

### POML Integration (#250) - Patterns Identified  

**Key Architectural Components:**
- **Format Detection**: `TemplateFormatDetector` for automatic format identification
- **Processing Engine**: `POMLTemplateProcessor` with fallback capabilities
- **Migration Tools**: `TemplateMigrationAnalyzer` and `TemplateMigrationEngine`
- **Backward Compatibility**: Enhanced `TemplateResolver` maintains existing API
- **Error Handling**: `POMLIntegrationError` with graceful degradation

**Design Patterns Used:**
- **Adapter Pattern**: Integrate POML while maintaining Jinja2 compatibility
- **Template Method Pattern**: Unified template processing with format-specific implementations
- **Strategy Pattern**: Different processing strategies for different template formats
- **Factory Pattern**: Dynamic template processor selection
- **Null Object Pattern**: Graceful handling of missing POML SDK

**Key Success Factors:**
- 100% backward compatibility with existing templates
- Automatic format detection (100% accuracy)
- Incremental migration path with hybrid support
- Comprehensive validation and error reporting
- Production-ready with optional dependencies

### Deep Agents Evaluation (#253) - Lessons Learned

**Key Insights for Wrapper Architecture:**
- **Experimental Risk Management**: Need robust evaluation criteria for experimental SDKs
- **Integration Complexity Assessment**: Framework for evaluating integration overhead
- **Native vs. Third-party Trade-offs**: When to implement natively vs. integrate external tools
- **Production Readiness Criteria**: Clear criteria for production deployment decisions
- **Monitoring and Evaluation Framework**: Systematic approach to evaluating external integrations

**Architecture Implications:**
- Need standardized evaluation framework for new integrations
- Require production readiness assessment tools
- Must support experimental feature flagging
- Should enable easy rollback and fallback mechanisms

## Common Patterns and Abstractions

### 1. Configuration Management Patterns

**Common Structure:**
```python
@dataclass
class WrapperConfig:
    enabled: bool = False
    fallback_enabled: bool = True
    max_retry_attempts: int = 3
    timeout_seconds: float = 30.0
    monitoring_enabled: bool = True
    # Tool-specific configurations extend this base
```

**Identified Patterns:**
- Enable/disable flags for safe rollout
- Timeout and retry configuration
- Monitoring and metrics collection toggles
- Domain/context-specific overrides
- Fallback behavior configuration

### 2. Feature Flag System Patterns

**Common Structure:**
```python
class FeatureFlags:
    # Core flags
    WRAPPER_ENABLED = "wrapper_enabled"
    WRAPPER_MONITORING = "wrapper_monitoring"
    
    # Domain/context specific flags
    WRAPPER_DOMAIN_MEDICAL = "wrapper_domain_medical"
    # ... other domain flags
    
    # Experimental flags
    WRAPPER_EXPERIMENTAL_FEATURE = "wrapper_experimental_feature"
```

**Identified Patterns:**
- Hierarchical flag structure (core -> domain -> experimental)
- Dynamic flag updates during runtime
- Per-domain/context enable/disable capabilities
- Gradual rollout support
- A/B testing framework integration

### 3. Fallback and Error Handling Patterns

**Common Structure:**
```python
async def execute_with_fallback(self, operation, fallback_operation, context):
    try:
        if not self.is_enabled():
            return await fallback_operation(context)
        return await self._execute_wrapper_operation(operation, context)
    except WrapperException as e:
        logger.warning(f"Wrapper failed: {e}, falling back")
        return await fallback_operation(context)
```

**Identified Patterns:**
- Multi-layer fallback strategies
- Circuit breaker protection
- Graceful degradation to original implementation
- Comprehensive error logging and tracking
- Recovery mechanism for transient failures

### 4. Monitoring and Metrics Patterns

**Common Structure:**
```python
@dataclass 
class WrapperMetrics:
    tracking_id: str
    timestamp: datetime
    operation_type: str
    success: bool
    latency_ms: float
    error_message: Optional[str]
    # Tool-specific metrics extend this base
```

**Identified Patterns:**
- Unique tracking IDs for correlation
- Timestamp-based metrics with retention policies
- Success/failure tracking with detailed error information
- Latency and performance monitoring
- Business metrics (cost savings, quality scores)

## Unified Wrapper Architecture Design

### Base Classes and Interfaces

#### 1. BaseWrapper Abstract Class

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum

T = TypeVar('T')  # Return type for wrapper operations
C = TypeVar('C')  # Configuration type

class WrapperStatus(Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    FALLBACK = "fallback"
    ERROR = "error"

@dataclass
class WrapperResult(Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    fallback_used: bool = False

class BaseWrapper(ABC, Generic[T, C]):
    """Abstract base class for all external tool wrappers."""
    
    def __init__(
        self, 
        name: str,
        config: C,
        feature_flags: FeatureFlagManager,
        monitoring: WrapperMonitoring
    ):
        self.name = name
        self.config = config
        self.feature_flags = feature_flags
        self.monitoring = monitoring
        self._status = WrapperStatus.DISABLED
    
    @abstractmethod
    async def _execute_wrapper_operation(self, *args, **kwargs) -> T:
        """Execute the core wrapper operation."""
        pass
    
    @abstractmethod
    async def _execute_fallback_operation(self, *args, **kwargs) -> T:
        """Execute fallback to original implementation."""
        pass
    
    @abstractmethod
    def _validate_config(self) -> bool:
        """Validate wrapper configuration."""
        pass
    
    async def execute(self, *args, **kwargs) -> WrapperResult[T]:
        """Execute wrapper operation with comprehensive error handling."""
        operation_id = self.monitoring.start_operation(self.name)
        
        try:
            # Check if wrapper is enabled
            if not self._is_enabled():
                result = await self._execute_fallback_operation(*args, **kwargs)
                self.monitoring.record_fallback(operation_id, "wrapper_disabled")
                return WrapperResult(success=True, data=result, fallback_used=True)
            
            # Execute wrapper operation
            result = await self._execute_wrapper_operation(*args, **kwargs)
            self.monitoring.record_success(operation_id, result)
            return WrapperResult(success=True, data=result)
            
        except Exception as e:
            # Execute fallback on error
            self.monitoring.record_error(operation_id, str(e))
            try:
                result = await self._execute_fallback_operation(*args, **kwargs)
                self.monitoring.record_fallback(operation_id, f"error: {e}")
                return WrapperResult(success=True, data=result, fallback_used=True, error=str(e))
            except Exception as fallback_error:
                self.monitoring.record_fatal_error(operation_id, str(fallback_error))
                return WrapperResult(success=False, error=f"Wrapper and fallback failed: {fallback_error}")
        
        finally:
            self.monitoring.end_operation(operation_id)
    
    def _is_enabled(self) -> bool:
        """Check if wrapper is enabled via feature flags."""
        return (
            self.config.enabled and 
            self.feature_flags.is_enabled(f"{self.name}_enabled") and
            self._validate_config()
        )
```

#### 2. BaseWrapperConfig Abstract Class

```python
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class BaseWrapperConfig(ABC):
    """Base configuration for all wrappers."""
    
    # Core wrapper settings
    enabled: bool = False
    fallback_enabled: bool = True
    max_retry_attempts: int = 3
    timeout_seconds: float = 30.0
    
    # Monitoring and metrics
    monitoring_enabled: bool = True
    metrics_retention_days: int = 30
    
    # Feature flag integration
    feature_flag_prefix: str = ""
    
    def get_feature_flag_name(self, flag: str) -> str:
        """Get the full feature flag name for this wrapper."""
        prefix = self.feature_flag_prefix or self.__class__.__name__.lower().replace('config', '')
        return f"{prefix}_{flag}"
    
    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return (
            self.timeout_seconds > 0 and
            self.max_retry_attempts >= 0 and
            self.metrics_retention_days > 0
        )
```

#### 3. FeatureFlagManager - Unified System

```python
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

class FeatureFlagScope(Enum):
    GLOBAL = "global"
    WRAPPER = "wrapper" 
    DOMAIN = "domain"
    USER = "user"
    EXPERIMENTAL = "experimental"

@dataclass
class FeatureFlag:
    name: str
    enabled: bool = False
    scope: FeatureFlagScope = FeatureFlagScope.WRAPPER
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    incompatible_with: List[str] = field(default_factory=list)

class FeatureFlagManager:
    """Unified feature flag management system."""
    
    def __init__(self, config_source: Optional[Dict[str, Any]] = None):
        self._flags: Dict[str, FeatureFlag] = {}
        self._flag_hierarchy: Dict[str, Set[str]] = {}
        
        if config_source:
            self._load_from_config(config_source)
    
    def register_flag(
        self, 
        flag: FeatureFlag,
        parent_flags: Optional[List[str]] = None
    ) -> None:
        """Register a feature flag with optional parent dependencies."""
        self._flags[flag.name] = flag
        
        if parent_flags:
            for parent in parent_flags:
                if parent not in self._flag_hierarchy:
                    self._flag_hierarchy[parent] = set()
                self._flag_hierarchy[parent].add(flag.name)
    
    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled, considering hierarchy."""
        if flag_name not in self._flags:
            return False
        
        flag = self._flags[flag_name]
        
        # Check if flag itself is enabled
        if not flag.enabled:
            return False
        
        # Check dependencies
        for dep in flag.dependencies:
            if not self.is_enabled(dep):
                return False
        
        # Check incompatible flags
        for incompatible in flag.incompatible_with:
            if self.is_enabled(incompatible):
                return False
        
        return True
    
    def enable_flag(self, flag_name: str, enable_dependencies: bool = True) -> bool:
        """Enable a feature flag and optionally its dependencies."""
        if flag_name not in self._flags:
            return False
        
        flag = self._flags[flag_name]
        
        # Enable dependencies first if requested
        if enable_dependencies:
            for dep in flag.dependencies:
                if not self.enable_flag(dep, enable_dependencies=True):
                    return False
        
        # Check for incompatible flags
        for incompatible in flag.incompatible_with:
            if self.is_enabled(incompatible):
                logger.warning(f"Cannot enable {flag_name}: incompatible with {incompatible}")
                return False
        
        flag.enabled = True
        logger.info(f"Feature flag enabled: {flag_name}")
        return True
    
    def disable_flag(self, flag_name: str, disable_dependents: bool = True) -> bool:
        """Disable a feature flag and optionally flags that depend on it."""
        if flag_name not in self._flags:
            return False
        
        # Disable dependent flags first if requested
        if disable_dependents and flag_name in self._flag_hierarchy:
            for dependent in self._flag_hierarchy[flag_name]:
                self.disable_flag(dependent, disable_dependents=True)
        
        self._flags[flag_name].enabled = False
        logger.info(f"Feature flag disabled: {flag_name}")
        return True
    
    def get_wrapper_flags(self, wrapper_name: str) -> Dict[str, bool]:
        """Get all flags relevant to a specific wrapper."""
        prefix = f"{wrapper_name}_"
        return {
            name: flag.enabled 
            for name, flag in self._flags.items() 
            if name.startswith(prefix)
        }
```

#### 4. WrapperMonitoring - Centralized Monitoring

```python
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class OperationMetrics:
    operation_id: str
    wrapper_name: str
    operation_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = True
    error_message: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Calculate operation duration in milliseconds."""
        if self.end_time is None:
            return (datetime.utcnow() - self.start_time).total_seconds() * 1000
        return (self.end_time - self.start_time).total_seconds() * 1000

class WrapperMonitoring:
    """Centralized monitoring system for all wrappers."""
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self._active_operations: Dict[str, OperationMetrics] = {}
        self._completed_operations: List[OperationMetrics] = []
        self._wrapper_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'fallback_operations': 0,
            'average_duration_ms': 0.0,
            'last_error': None,
            'last_success': None
        })
    
    def start_operation(
        self, 
        wrapper_name: str, 
        operation_type: str = "default"
    ) -> str:
        """Start tracking a wrapper operation."""
        operation_id = str(uuid.uuid4())
        metrics = OperationMetrics(
            operation_id=operation_id,
            wrapper_name=wrapper_name,
            operation_type=operation_type,
            start_time=datetime.utcnow()
        )
        
        self._active_operations[operation_id] = metrics
        return operation_id
    
    def record_success(
        self, 
        operation_id: str, 
        result: Any = None,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record successful completion of an operation."""
        if operation_id not in self._active_operations:
            return
        
        metrics = self._active_operations[operation_id]
        metrics.end_time = datetime.utcnow()
        metrics.success = True
        
        if custom_metrics:
            metrics.custom_metrics.update(custom_metrics)
        
        self._finalize_operation(operation_id)
    
    def record_error(
        self, 
        operation_id: str, 
        error_message: str,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record an error in wrapper operation."""
        if operation_id not in self._active_operations:
            return
        
        metrics = self._active_operations[operation_id]
        metrics.success = False
        metrics.error_message = error_message
        
        if custom_metrics:
            metrics.custom_metrics.update(custom_metrics)
        
        # Don't finalize yet - might still use fallback
    
    def record_fallback(
        self, 
        operation_id: str, 
        fallback_reason: str,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record that fallback was used for an operation."""
        if operation_id not in self._active_operations:
            return
        
        metrics = self._active_operations[operation_id]
        metrics.fallback_used = True
        metrics.fallback_reason = fallback_reason
        metrics.success = True  # Fallback succeeded
        
        if custom_metrics:
            metrics.custom_metrics.update(custom_metrics)
        
        # Don't finalize - let the wrapper call end_operation
    
    def record_fatal_error(
        self, 
        operation_id: str, 
        error_message: str,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a fatal error (both wrapper and fallback failed)."""
        if operation_id not in self._active_operations:
            return
        
        metrics = self._active_operations[operation_id]
        metrics.end_time = datetime.utcnow()
        metrics.success = False
        metrics.error_message = error_message
        
        if custom_metrics:
            metrics.custom_metrics.update(custom_metrics)
        
        self._finalize_operation(operation_id)
    
    def end_operation(self, operation_id: str) -> None:
        """End an operation (called from finally block)."""
        if operation_id not in self._active_operations:
            return
        
        metrics = self._active_operations[operation_id]
        if metrics.end_time is None:
            metrics.end_time = datetime.utcnow()
        
        self._finalize_operation(operation_id)
    
    def _finalize_operation(self, operation_id: str) -> None:
        """Move operation from active to completed and update stats."""
        if operation_id not in self._active_operations:
            return
        
        metrics = self._active_operations.pop(operation_id)
        self._completed_operations.append(metrics)
        
        # Update wrapper statistics
        stats = self._wrapper_stats[metrics.wrapper_name]
        stats['total_operations'] += 1
        
        if metrics.success:
            stats['successful_operations'] += 1
            stats['last_success'] = datetime.utcnow()
        else:
            stats['failed_operations'] += 1
            stats['last_error'] = metrics.error_message
        
        if metrics.fallback_used:
            stats['fallback_operations'] += 1
        
        # Update average duration
        total_ops = stats['total_operations']
        current_avg = stats['average_duration_ms']
        new_duration = metrics.duration_ms
        stats['average_duration_ms'] = ((current_avg * (total_ops - 1)) + new_duration) / total_ops
        
        # Cleanup old operations
        self._cleanup_old_operations()
    
    def get_wrapper_stats(self, wrapper_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a wrapper."""
        return self._wrapper_stats.get(wrapper_name, {}).copy()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        total_wrappers = len(self._wrapper_stats)
        active_operations = len(self._active_operations)
        
        if not self._wrapper_stats:
            return {
                'total_wrappers': 0,
                'active_operations': 0,
                'overall_success_rate': 1.0,
                'overall_fallback_rate': 0.0,
                'average_response_time_ms': 0.0
            }
        
        # Calculate aggregate metrics
        total_ops = sum(stats['total_operations'] for stats in self._wrapper_stats.values())
        total_successes = sum(stats['successful_operations'] for stats in self._wrapper_stats.values())
        total_fallbacks = sum(stats['fallback_operations'] for stats in self._wrapper_stats.values())
        avg_duration = sum(stats['average_duration_ms'] for stats in self._wrapper_stats.values()) / total_wrappers
        
        return {
            'total_wrappers': total_wrappers,
            'active_operations': active_operations,
            'overall_success_rate': total_successes / total_ops if total_ops > 0 else 1.0,
            'overall_fallback_rate': total_fallbacks / total_ops if total_ops > 0 else 0.0,
            'average_response_time_ms': avg_duration,
            'total_operations': total_ops
        }
    
    def _cleanup_old_operations(self) -> None:
        """Remove operations older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        original_count = len(self._completed_operations)
        
        self._completed_operations = [
            op for op in self._completed_operations 
            if op.end_time and op.end_time >= cutoff
        ]
        
        cleaned = original_count - len(self._completed_operations)
        if cleaned > 0:
            logger.debug(f"Cleaned up {cleaned} old operation metrics")
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)
1. Create base wrapper classes and interfaces
2. Implement unified feature flag management system
3. Create centralized monitoring infrastructure
4. Implement configuration management framework
5. Add comprehensive error handling patterns

### Phase 2: Integration Refactoring (Week 2)  
1. Refactor RouteLLM integration to use base classes
2. Refactor POML integration to use base classes  
3. Update existing integrations to use unified monitoring
4. Migrate to unified configuration system
5. Standardize feature flag usage across integrations

### Phase 3: Testing and Documentation (Week 3)
1. Create comprehensive test suite for wrapper framework
2. Implement integration tests for refactored wrappers
3. Create wrapper development documentation
4. Write migration guide for existing integrations
5. Performance benchmarking and optimization

### Phase 4: Advanced Features (Week 4)
1. Implement wrapper registry and discovery
2. Add support for wrapper composition and chaining
3. Create wrapper health checking and circuit breaker patterns
4. Implement A/B testing framework for wrappers
5. Add advanced monitoring dashboards and alerts

## File Structure

```
src/orchestrator/core/
├── wrapper_base.py              # BaseWrapper and core interfaces
├── wrapper_config.py            # BaseWrapperConfig and config management  
├── feature_flags.py             # FeatureFlagManager and related classes
├── wrapper_monitoring.py        # WrapperMonitoring and metrics classes
├── wrapper_registry.py          # WrapperRegistry for discovery/management
└── wrapper_testing.py           # Common testing utilities

tests/core/wrapper_framework/
├── test_base_wrapper.py         # BaseWrapper tests
├── test_feature_flags.py        # Feature flag system tests
├── test_monitoring.py           # Monitoring system tests
├── test_wrapper_integration.py  # Integration tests
└── test_wrapper_patterns.py     # Common pattern tests

docs/wrapper_development/
├── getting_started.md           # Quick start guide
├── architecture.md              # Architecture overview
├── migration_guide.md           # Migration from existing patterns
├── best_practices.md            # Development best practices
└── examples/                    # Example implementations
```

## Success Criteria

1. **Unified Architecture**: All wrapper integrations use common base classes and patterns
2. **Zero Breaking Changes**: Existing functionality remains unchanged 
3. **Feature Flag Consistency**: Unified feature flag system across all wrappers
4. **Centralized Monitoring**: All wrappers use common monitoring infrastructure
5. **Comprehensive Testing**: >95% test coverage for wrapper framework
6. **Production Ready**: Full error handling, logging, and documentation
7. **Developer Experience**: Clear patterns and documentation for future integrations

## Risk Mitigation

### Technical Risks
1. **Integration Complexity**: Use incremental migration approach
2. **Performance Impact**: Continuous benchmarking and optimization  
3. **Backward Compatibility**: Comprehensive compatibility testing
4. **Framework Overhead**: Keep abstractions minimal and focused

### Operational Risks  
1. **Migration Issues**: Thorough testing and gradual rollout
2. **Documentation Gap**: Comprehensive documentation with examples
3. **Team Training**: Clear migration guides and best practices
4. **Maintenance Burden**: Focus on simplicity and reusability

## Future Extensions

### Potential Enhancements
1. **Wrapper Marketplace**: Registry of available wrappers with metadata
2. **Dynamic Loading**: Runtime loading and unloading of wrappers
3. **Wrapper Chaining**: Compose multiple wrappers for complex workflows
4. **ML-driven Selection**: Intelligent wrapper selection based on context
5. **Multi-tenant Support**: Per-tenant wrapper configuration and isolation

This unified wrapper architecture provides a solid foundation for standardizing external tool integrations while maintaining the flexibility and reliability demonstrated in the completed RouteLLM and POML integrations.