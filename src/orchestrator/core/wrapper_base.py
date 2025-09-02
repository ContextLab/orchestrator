"""
Base classes and interfaces for the unified wrapper architecture.

This module provides the foundational components for integrating external tools
while maintaining backward compatibility and providing comprehensive error handling,
monitoring, and feature flag support.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, TypeVar, Generic, Union, List
import uuid

logger = logging.getLogger(__name__)

# Type variables for generic wrapper implementations
T = TypeVar('T')  # Return type for wrapper operations
C = TypeVar('C')  # Configuration type for specific wrappers
F = TypeVar('F')  # Fallback operation type


class WrapperStatus(Enum):
    """Status indicators for wrapper operations."""
    
    ENABLED = "enabled"
    DISABLED = "disabled"
    FALLBACK = "fallback"
    ERROR = "error"
    INITIALIZING = "initializing"
    UNAVAILABLE = "unavailable"


class WrapperCapability(Enum):
    """Standard capabilities that wrappers can provide."""
    
    ROUTING = "routing"
    TEMPLATE_PROCESSING = "template_processing"
    MODEL_SELECTION = "model_selection"
    COST_OPTIMIZATION = "cost_optimization"
    PARALLEL_EXECUTION = "parallel_execution"
    STATE_MANAGEMENT = "state_management"
    FORMAT_CONVERSION = "format_conversion"
    VALIDATION = "validation"
    MONITORING = "monitoring"


@dataclass
class WrapperResult(Generic[T]):
    """
    Standardized result container for wrapper operations.
    
    Provides consistent structure for success/failure information,
    data payload, error details, and metrics across all wrappers.
    """
    
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    execution_time_ms: Optional[float] = None
    operation_id: Optional[str] = None
    
    @classmethod
    def success_result(
        cls, 
        data: T, 
        metrics: Optional[Dict[str, Any]] = None,
        operation_id: Optional[str] = None
    ) -> WrapperResult[T]:
        """Create a successful result."""
        return cls(
            success=True, 
            data=data, 
            metrics=metrics,
            operation_id=operation_id
        )
    
    @classmethod
    def error_result(
        cls, 
        error: str, 
        error_code: Optional[str] = None,
        operation_id: Optional[str] = None
    ) -> WrapperResult[T]:
        """Create an error result."""
        return cls(
            success=False, 
            error=error, 
            error_code=error_code,
            operation_id=operation_id
        )
    
    @classmethod
    def fallback_result(
        cls, 
        data: T, 
        fallback_reason: str,
        original_error: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        operation_id: Optional[str] = None
    ) -> WrapperResult[T]:
        """Create a fallback result."""
        return cls(
            success=True,
            data=data,
            fallback_used=True,
            fallback_reason=fallback_reason,
            error=original_error,
            metrics=metrics,
            operation_id=operation_id
        )


@dataclass
class BaseWrapperConfig:
    """Base configuration class for wrapper implementations."""
    
    # Basic configuration
    enabled: bool = True
    timeout_seconds: float = 30.0
    retry_count: int = 3
    retry_delay_seconds: float = 1.0
    
    # Feature flags
    use_fallback: bool = True
    enable_monitoring: bool = True
    enable_caching: bool = False
    
    # Custom configuration
    custom_config: Optional[Dict[str, Any]] = None


@dataclass
class WrapperContext:
    """
    Context information passed to wrapper operations.
    
    Provides standardized context data including operation metadata,
    feature flags, configuration overrides, and custom attributes.
    """
    
    operation_id: str
    wrapper_name: str
    operation_type: str = "default"
    timestamp: datetime = None
    
    # Context data
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Configuration overrides
    config_overrides: Optional[Dict[str, Any]] = None
    feature_flag_overrides: Optional[Dict[str, bool]] = None
    
    # Custom attributes for wrapper-specific context
    custom_attributes: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        
        if self.custom_attributes is None:
            self.custom_attributes = {}
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get a custom attribute value."""
        return self.custom_attributes.get(key, default) if self.custom_attributes else default
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a custom attribute value."""
        if self.custom_attributes is None:
            self.custom_attributes = {}
        self.custom_attributes[key] = value


class WrapperException(Exception):
    """Base exception for all wrapper-related errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        cause: Optional[Exception] = None,
        wrapper_name: Optional[str] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.cause = cause
        self.wrapper_name = wrapper_name


class WrapperConfigurationError(WrapperException):
    """Raised when wrapper configuration is invalid."""
    pass


class WrapperInitializationError(WrapperException):
    """Raised when wrapper fails to initialize properly."""
    pass


class WrapperOperationError(WrapperException):
    """Raised when wrapper operation fails."""
    pass


class WrapperTimeoutError(WrapperException):
    """Raised when wrapper operation times out."""
    pass


class BaseWrapper(ABC, Generic[T, C]):
    """
    Abstract base class for all external tool wrappers.
    
    Provides standardized interface for:
    - Configuration management
    - Feature flag integration
    - Error handling and fallback mechanisms
    - Monitoring and metrics collection
    - Operation lifecycle management
    
    Specific wrappers should inherit from this class and implement the
    abstract methods for their particular external tool integration.
    """
    
    def __init__(
        self, 
        name: str,
        config: C,
        feature_flags: Optional['FeatureFlagManager'] = None,
        monitoring: Optional['WrapperMonitoring'] = None
    ):
        """
        Initialize base wrapper with core dependencies.
        
        Args:
            name: Unique identifier for this wrapper
            config: Wrapper-specific configuration
            feature_flags: Feature flag manager instance
            monitoring: Monitoring system instance
        """
        self.name = name
        self.config = config
        self.feature_flags = feature_flags
        self.monitoring = monitoring
        self._status = WrapperStatus.INITIALIZING
        self._initialization_error: Optional[Exception] = None
        
        try:
            self._initialize_wrapper()
            self._status = WrapperStatus.ENABLED if self._is_enabled() else WrapperStatus.DISABLED
        except Exception as e:
            self._status = WrapperStatus.ERROR
            self._initialization_error = e
            logger.error(f"Failed to initialize wrapper {name}: {e}")
    
    # Abstract methods that must be implemented by concrete wrappers
    
    @abstractmethod
    async def _execute_wrapper_operation(
        self, 
        context: WrapperContext,
        *args, 
        **kwargs
    ) -> T:
        """
        Execute the core wrapper operation.
        
        This method should contain the main logic for interacting with
        the external tool. It should not handle fallbacks or feature flags -
        those are handled by the base class.
        
        Args:
            context: Operation context with metadata and configuration
            *args: Operation-specific positional arguments
            **kwargs: Operation-specific keyword arguments
            
        Returns:
            The result of the wrapper operation
            
        Raises:
            WrapperException: For wrapper-specific errors
        """
        pass
    
    @abstractmethod
    async def _execute_fallback_operation(
        self, 
        context: WrapperContext,
        original_error: Optional[Exception] = None,
        *args, 
        **kwargs
    ) -> T:
        """
        Execute fallback to original implementation.
        
        This method should provide the original functionality that existed
        before the wrapper was added. This ensures that the system continues
        to work even if the external tool integration fails.
        
        Args:
            context: Operation context with metadata and configuration
            original_error: The error that caused fallback (if any)
            *args: Operation-specific positional arguments
            **kwargs: Operation-specific keyword arguments
            
        Returns:
            The result of the fallback operation
            
        Raises:
            Exception: If fallback also fails
        """
        pass
    
    @abstractmethod
    def _validate_config(self) -> bool:
        """
        Validate wrapper configuration.
        
        Check that the wrapper configuration is valid and the wrapper
        can operate with the current settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[WrapperCapability]:
        """
        Get the capabilities provided by this wrapper.
        
        Returns:
            List of capabilities this wrapper provides
        """
        pass
    
    # Concrete methods with default implementations
    
    def _initialize_wrapper(self) -> None:
        """
        Initialize wrapper-specific resources.
        
        Override this method if the wrapper needs to perform initialization
        steps beyond basic configuration validation.
        
        Raises:
            WrapperInitializationError: If initialization fails
        """
        if not self._validate_config():
            raise WrapperInitializationError(
                f"Invalid configuration for wrapper {self.name}",
                wrapper_name=self.name
            )
    
    def _is_enabled(self) -> bool:
        """
        Check if wrapper is enabled via feature flags and configuration.
        
        Returns:
            True if wrapper should be used, False if should fallback
        """
        # Check basic configuration
        if not hasattr(self.config, 'enabled') or not self.config.enabled:
            return False
        
        # Check feature flags if available
        if self.feature_flags:
            primary_flag = f"{self.name}_enabled"
            if not self.feature_flags.is_enabled(primary_flag):
                return False
        
        # Check configuration validity
        return self._validate_config()
    
    def _should_fallback(self, error: Optional[Exception] = None) -> bool:
        """
        Determine if operation should fallback based on error and configuration.
        
        Args:
            error: The error that occurred (if any)
            
        Returns:
            True if should use fallback, False to propagate error
        """
        # Always fallback if wrapper is disabled
        if not self._is_enabled():
            return True
        
        # Check if fallback is enabled in configuration
        if hasattr(self.config, 'fallback_enabled'):
            return self.config.fallback_enabled
        
        # Default to fallback on errors
        return error is not None
    
    def _create_context(
        self, 
        operation_type: str = "default",
        **context_kwargs
    ) -> WrapperContext:
        """
        Create operation context with unique ID and metadata.
        
        Args:
            operation_type: Type of operation being performed
            **context_kwargs: Additional context attributes
            
        Returns:
            WrapperContext instance for this operation
        """
        operation_id = str(uuid.uuid4())
        
        return WrapperContext(
            operation_id=operation_id,
            wrapper_name=self.name,
            operation_type=operation_type,
            **context_kwargs
        )
    
    async def execute(
        self, 
        operation_type: str = "default",
        timeout_seconds: Optional[float] = None,
        *args, 
        **kwargs
    ) -> WrapperResult[T]:
        """
        Execute wrapper operation with comprehensive error handling and monitoring.
        
        This is the main entry point for wrapper operations. It handles:
        - Feature flag checking
        - Error handling and fallback logic
        - Monitoring and metrics collection
        - Operation lifecycle management
        
        Args:
            operation_type: Type of operation for monitoring/logging
            timeout_seconds: Optional timeout override
            *args: Operation-specific positional arguments
            **kwargs: Operation-specific keyword arguments
            
        Returns:
            WrapperResult containing operation outcome and metadata
        """
        # Create operation context
        context = self._create_context(operation_type, **kwargs.pop('context_overrides', {}))
        start_time = datetime.utcnow()
        
        # Start monitoring if available
        if self.monitoring:
            self.monitoring.start_operation(context.operation_id, self.name, operation_type)
        
        try:
            # Check wrapper status
            if self._status == WrapperStatus.ERROR:
                if self._should_fallback(self._initialization_error):
                    logger.info(f"Wrapper {self.name} in error state, using fallback")
                    result = await self._execute_fallback_operation(
                        context, self._initialization_error, *args, **kwargs
                    )
                    return self._create_fallback_result(
                        result, f"wrapper_error: {self._initialization_error}", context
                    )
                else:
                    raise WrapperInitializationError(
                        f"Wrapper {self.name} is in error state",
                        wrapper_name=self.name,
                        cause=self._initialization_error
                    )
            
            # Check if wrapper is enabled
            if not self._is_enabled():
                logger.debug(f"Wrapper {self.name} is disabled, using fallback")
                result = await self._execute_fallback_operation(context, None, *args, **kwargs)
                return self._create_fallback_result(result, "wrapper_disabled", context)
            
            # Execute wrapper operation
            logger.debug(f"Executing wrapper operation: {self.name}.{operation_type}")
            result = await self._execute_wrapper_operation(context, *args, **kwargs)
            
            # Record success
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            if self.monitoring:
                self.monitoring.record_success(context.operation_id, result)
            
            return WrapperResult.success_result(
                result, 
                metrics={"execution_time_ms": execution_time},
                operation_id=context.operation_id
            )
            
        except Exception as e:
            logger.warning(f"Wrapper {self.name} operation failed: {e}")
            
            # Record error
            if self.monitoring:
                self.monitoring.record_error(context.operation_id, str(e))
            
            # Check if should use fallback
            if self._should_fallback(e):
                try:
                    logger.info(f"Using fallback for wrapper {self.name}")
                    result = await self._execute_fallback_operation(context, e, *args, **kwargs)
                    
                    if self.monitoring:
                        self.monitoring.record_fallback(context.operation_id, f"error: {e}")
                    
                    return self._create_fallback_result(result, f"error: {e}", context)
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback failed for wrapper {self.name}: {fallback_error}")
                    
                    if self.monitoring:
                        self.monitoring.record_fatal_error(context.operation_id, str(fallback_error))
                    
                    return WrapperResult.error_result(
                        f"Wrapper and fallback failed: {fallback_error}",
                        error_code="FALLBACK_FAILURE",
                        operation_id=context.operation_id
                    )
            else:
                # Don't use fallback, propagate error
                error_code = getattr(e, 'error_code', 'WRAPPER_ERROR')
                return WrapperResult.error_result(
                    str(e),
                    error_code=error_code,
                    operation_id=context.operation_id
                )
        
        finally:
            # End monitoring
            if self.monitoring:
                self.monitoring.end_operation(context.operation_id)
    
    def _create_fallback_result(
        self, 
        data: T, 
        fallback_reason: str, 
        context: WrapperContext
    ) -> WrapperResult[T]:
        """Create a standardized fallback result."""
        execution_time = (datetime.utcnow() - context.timestamp).total_seconds() * 1000
        
        return WrapperResult.fallback_result(
            data,
            fallback_reason,
            metrics={"execution_time_ms": execution_time},
            operation_id=context.operation_id
        )
    
    def get_status(self) -> WrapperStatus:
        """Get current wrapper status."""
        return self._status
    
    def get_health_info(self) -> Dict[str, Any]:
        """
        Get health information for this wrapper.
        
        Returns:
            Dictionary containing health and status information
        """
        health_info = {
            "name": self.name,
            "status": self._status.value,
            "enabled": self._is_enabled(),
            "capabilities": [cap.value for cap in self.get_capabilities()],
            "config_valid": self._validate_config(),
        }
        
        if self._initialization_error:
            health_info["initialization_error"] = str(self._initialization_error)
        
        if self.monitoring:
            stats = self.monitoring.get_wrapper_stats(self.name)
            health_info["stats"] = stats
        
        return health_info
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, status={self._status.value})"


class WrapperRegistry:
    """
    Registry for managing and discovering wrapper instances.
    
    Provides centralized management of all wrappers including:
    - Registration and discovery
    - Health monitoring  
    - Capability querying
    - Batch operations
    """
    
    def __init__(self):
        self._wrappers: Dict[str, BaseWrapper] = {}
        self._capabilities_index: Dict[WrapperCapability, List[str]] = {}
    
    def register_wrapper(self, wrapper: BaseWrapper) -> None:
        """
        Register a wrapper instance.
        
        Args:
            wrapper: Wrapper instance to register
        """
        self._wrappers[wrapper.name] = wrapper
        
        # Update capabilities index
        for capability in wrapper.get_capabilities():
            if capability not in self._capabilities_index:
                self._capabilities_index[capability] = []
            self._capabilities_index[capability].append(wrapper.name)
        
        logger.info(f"Registered wrapper: {wrapper.name}")
    
    def unregister_wrapper(self, name: str) -> bool:
        """
        Unregister a wrapper.
        
        Args:
            name: Name of wrapper to unregister
            
        Returns:
            True if wrapper was found and removed, False otherwise
        """
        if name not in self._wrappers:
            return False
        
        wrapper = self._wrappers.pop(name)
        
        # Update capabilities index
        for capability in wrapper.get_capabilities():
            if capability in self._capabilities_index:
                try:
                    self._capabilities_index[capability].remove(name)
                except ValueError:
                    pass  # Already removed
                
                # Remove empty capability entries
                if not self._capabilities_index[capability]:
                    del self._capabilities_index[capability]
        
        logger.info(f"Unregistered wrapper: {name}")
        return True
    
    def get_wrapper(self, name: str) -> Optional[BaseWrapper]:
        """Get a wrapper by name."""
        return self._wrappers.get(name)
    
    def get_wrappers_by_capability(
        self, 
        capability: WrapperCapability
    ) -> List[BaseWrapper]:
        """
        Get all wrappers that provide a specific capability.
        
        Args:
            capability: The capability to search for
            
        Returns:
            List of wrappers that provide the capability
        """
        wrapper_names = self._capabilities_index.get(capability, [])
        return [self._wrappers[name] for name in wrapper_names if name in self._wrappers]
    
    def get_all_wrappers(self) -> List[BaseWrapper]:
        """Get all registered wrappers."""
        return list(self._wrappers.values())
    
    def get_healthy_wrappers(self) -> List[BaseWrapper]:
        """Get all wrappers in healthy state."""
        return [
            wrapper for wrapper in self._wrappers.values()
            if wrapper.get_status() not in [WrapperStatus.ERROR, WrapperStatus.UNAVAILABLE]
        ]
    
    def get_registry_health(self) -> Dict[str, Any]:
        """
        Get overall health of all registered wrappers.
        
        Returns:
            Dictionary containing registry health information
        """
        total_wrappers = len(self._wrappers)
        healthy_wrappers = len(self.get_healthy_wrappers())
        
        status_counts = {}
        for wrapper in self._wrappers.values():
            status = wrapper.get_status().value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_wrappers": total_wrappers,
            "healthy_wrappers": healthy_wrappers,
            "health_percentage": (healthy_wrappers / total_wrappers * 100) if total_wrappers > 0 else 100,
            "status_breakdown": status_counts,
            "available_capabilities": list(self._capabilities_index.keys())
        }