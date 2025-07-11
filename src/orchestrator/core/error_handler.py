"""Comprehensive error handling framework for the Orchestrator."""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Type
from collections import defaultdict, deque


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    EXECUTION = "execution"
    RESOURCE = "resource"
    NETWORK = "network"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Detailed error information."""
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True
    retry_count: int = 0
    stack_trace: Optional[str] = None


class RetryStrategy(ABC):
    """Abstract base class for retry strategies."""
    
    @abstractmethod
    def should_retry(self, error_info: ErrorInfo, attempt: int) -> bool:
        """Determine if operation should be retried."""
        pass
    
    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry attempt."""
        pass


class ExponentialBackoffRetry(RetryStrategy):
    """Exponential backoff retry strategy."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def should_retry(self, error_info: ErrorInfo, attempt: int) -> bool:
        """Check if should retry based on attempt count and error type."""
        if attempt >= self.max_retries:
            return False
        
        # Don't retry for certain error types
        non_retryable = {
            ErrorCategory.VALIDATION,
            ErrorCategory.PERMISSION,
            ErrorCategory.CONFIGURATION
        }
        
        if error_info.category in non_retryable:
            return False
        
        # Don't retry critical errors unless explicitly marked as recoverable
        if error_info.severity == ErrorSeverity.CRITICAL and not error_info.recoverable:
            return False
        
        return True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)


class LinearRetry(RetryStrategy):
    """Linear retry strategy with fixed delays."""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
    
    def should_retry(self, error_info: ErrorInfo, attempt: int) -> bool:
        """Check if should retry."""
        return attempt < self.max_retries and error_info.recoverable
    
    def get_delay(self, attempt: int) -> float:
        """Return fixed delay."""
        return self.delay


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2
    timeout: float = 30.0


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time < self.config.recovery_timeout:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
            else:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger.info(f"Circuit breaker {self.name} transitioning to half-open")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise
    
    def _record_success(self):
        """Record successful execution."""
        self.failure_count = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.logger.info(f"Circuit breaker {self.name} closed after recovery")
    
    def _record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            }
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class ErrorClassifier:
    """Classifies errors into categories and severity levels."""
    
    def __init__(self):
        self.classification_rules = {
            # Validation errors
            (ValueError, TypeError, AttributeError): (ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM),
            
            # Execution errors
            (RuntimeError, ImportError, NotImplementedError): (ErrorCategory.EXECUTION, ErrorSeverity.HIGH),
            
            # Resource errors
            (MemoryError, OSError): (ErrorCategory.RESOURCE, ErrorSeverity.HIGH),
            
            # Network errors
            (ConnectionError, TimeoutError): (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM),
            
            # Permission errors
            (PermissionError, FileNotFoundError): (ErrorCategory.PERMISSION, ErrorSeverity.MEDIUM),
        }
    
    def classify(self, exception: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """Classify an exception."""
        for exception_types, (category, severity) in self.classification_rules.items():
            if isinstance(exception, exception_types):
                return category, severity
        
        # Default classification
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    def create_error_info(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Create ErrorInfo from exception."""
        category, severity = self.classify(exception)
        
        return ErrorInfo(
            error_type=type(exception).__name__,
            message=str(exception),
            severity=severity,
            category=category,
            context=context or {},
            recoverable=self._is_recoverable(exception, category),
            stack_trace=self._get_stack_trace(exception)
        )
    
    def _is_recoverable(self, exception: Exception, category: ErrorCategory) -> bool:
        """Determine if error is recoverable."""
        # Non-recoverable categories
        non_recoverable = {
            ErrorCategory.VALIDATION,
            ErrorCategory.PERMISSION,
            ErrorCategory.CONFIGURATION
        }
        
        if category in non_recoverable:
            return False
        
        # Specific non-recoverable exceptions
        non_recoverable_exceptions = (
            ImportError,
            NotImplementedError,
            SystemExit,
            KeyboardInterrupt
        )
        
        return not isinstance(exception, non_recoverable_exceptions)
    
    def _get_stack_trace(self, exception: Exception) -> Optional[str]:
        """Get stack trace from exception."""
        import traceback
        return traceback.format_exc()


class ErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self):
        self.classifier = ErrorClassifier()
        self.retry_strategies: Dict[str, RetryStrategy] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.logger = logging.getLogger("error_handler")
        
        # Default retry strategies
        self.retry_strategies["default"] = ExponentialBackoffRetry()
        self.retry_strategies["network"] = ExponentialBackoffRetry(max_retries=5, base_delay=2.0)
        self.retry_strategies["resource"] = LinearRetry(max_retries=2, delay=5.0)
    
    def register_retry_strategy(self, name: str, strategy: RetryStrategy):
        """Register a custom retry strategy."""
        self.retry_strategies[name] = strategy
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]
    
    async def handle_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Handle an error with classification and logging."""
        error_info = self.classifier.create_error_info(exception, context)
        
        # Log error
        self._log_error(error_info)
        
        # Record in history
        self.error_history.append(error_info)
        self.error_counts[error_info.error_type] += 1
        
        return error_info
    
    async def execute_with_retry(self, 
                               func: Callable, 
                               *args, 
                               strategy_name: str = "default",
                               context: Dict[str, Any] = None,
                               **kwargs) -> Any:
        """Execute function with retry logic."""
        strategy = self.retry_strategies.get(strategy_name, self.retry_strategies["default"])
        attempt = 0
        
        while True:
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                error_info = await self.handle_error(e, context)
                
                if not strategy.should_retry(error_info, attempt):
                    self.logger.error(f"Max retries exceeded for {func.__name__}")
                    raise
                
                delay = strategy.get_delay(attempt)
                self.logger.info(f"Retrying {func.__name__} after {delay}s (attempt {attempt + 1})")
                
                await asyncio.sleep(delay)
                attempt += 1
    
    async def execute_with_circuit_breaker(self,
                                         func: Callable,
                                         breaker_name: str,
                                         *args,
                                         config: CircuitBreakerConfig = None,
                                         **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        breaker = self.get_circuit_breaker(breaker_name, config)
        return await breaker.call(func, *args, **kwargs)
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error information."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_info.severity, logging.ERROR)
        
        self.logger.log(
            log_level,
            f"Error: {error_info.error_type} - {error_info.message} "
            f"(Category: {error_info.category.value}, Severity: {error_info.severity.value})"
        )
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        recent_errors = list(self.error_history)[-100:]  # Last 100 errors
        
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for error in recent_errors:
            category_counts[error.category.value] += 1
            severity_counts[error.severity.value] += 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_types": dict(self.error_counts),
            "category_distribution": dict(category_counts),
            "severity_distribution": dict(severity_counts),
            "circuit_breakers": {
                name: breaker.get_state() 
                for name, breaker in self.circuit_breakers.items()
            }
        }
    
    def reset_statistics(self):
        """Reset error statistics."""
        self.error_history.clear()
        self.error_counts.clear()
        for breaker in self.circuit_breakers.values():
            breaker.failure_count = 0
            breaker.success_count = 0
            breaker.state = CircuitBreakerState.CLOSED