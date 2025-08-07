"""Model Health Monitor - Phase 3 Advanced Features

Comprehensive health monitoring and automatic recovery for models and services.
Builds on Phase 2 service integration to provide intelligent monitoring.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from ..models.model_registry import ModelRegistry
from ..utils.service_manager import SERVICE_MANAGERS

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    RECOVERING = "recovering"


@dataclass
class HealthCheck:
    """Individual health check result."""
    model_key: str
    status: HealthStatus
    timestamp: float
    response_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class HealthMetrics:
    """Aggregated health metrics for a model."""
    model_key: str
    current_status: HealthStatus
    last_check: float
    consecutive_failures: int
    consecutive_successes: int
    total_checks: int
    success_rate: float
    avg_response_time_ms: float
    recovery_attempts: int = 0
    last_recovery_time: Optional[float] = None
    
    # Historical data
    recent_checks: List[HealthCheck] = field(default_factory=list)


class ModelHealthMonitor:
    """Comprehensive health monitoring for models and services.
    
    Features:
    - Real-time health checking with configurable intervals
    - Automatic failure detection and recovery
    - Service restart automation using Phase 2 service managers
    - Health metrics tracking and reporting
    - Intelligent escalation and alerting
    """
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        check_interval: int = 60,  # seconds
        max_history: int = 100,
        recovery_enabled: bool = True
    ):
        """Initialize health monitor.
        
        Args:
            model_registry: Model registry from Phase 2
            check_interval: Health check interval in seconds
            max_history: Maximum number of health checks to retain
            recovery_enabled: Whether to attempt automatic recovery
        """
        self.model_registry = model_registry
        self.check_interval = check_interval
        self.max_history = max_history
        self.recovery_enabled = recovery_enabled
        
        # Health tracking
        self.health_metrics: Dict[str, HealthMetrics] = {}
        self.monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Recovery configuration
        self.recovery_strategies = {
            "ollama": self._recover_ollama_model,
            "docker": self._recover_docker_service,
            "openai": self._recover_api_service,
            "anthropic": self._recover_api_service,
            "google": self._recover_api_service,
        }
        
        # Health check thresholds
        self.failure_threshold = 3  # Consecutive failures before marking unhealthy
        self.recovery_threshold = 2  # Consecutive successes before marking healthy
        self.degraded_response_time = 5000  # ms - response time for degraded status
        self.timeout_threshold = 30000  # ms - timeout threshold
        
        # Callback registry for health change notifications
        self.health_change_callbacks: List[Callable[[str, HealthStatus, HealthStatus], None]] = []
    
    def start_monitoring(self, models: Optional[List[str]] = None) -> None:
        """Start health monitoring for specified models.
        
        Args:
            models: List of model keys to monitor. If None, monitors all registered models.
        """
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
            
        # Determine models to monitor
        if models is None:
            models_to_monitor = list(self.model_registry.models.keys())
        else:
            models_to_monitor = models
        
        # Initialize health metrics for new models
        for model_key in models_to_monitor:
            if model_key not in self.health_metrics:
                self.health_metrics[model_key] = HealthMetrics(
                    model_key=model_key,
                    current_status=HealthStatus.UNKNOWN,
                    last_check=0.0,
                    consecutive_failures=0,
                    consecutive_successes=0,
                    total_checks=0,
                    success_rate=0.0,
                    avg_response_time_ms=0.0
                )
        
        self.monitoring_active = True
        self._shutdown_event.clear()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="ModelHealthMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info(f"Started health monitoring for {len(models_to_monitor)} models")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.monitoring_active:
            return
            
        self.monitoring_active = False
        self._shutdown_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10.0)
            
        logger.info("Stopped health monitoring")
    
    def add_health_change_callback(
        self,
        callback: Callable[[str, HealthStatus, HealthStatus], None]
    ) -> None:
        """Add callback for health status changes.
        
        Args:
            callback: Function called with (model_key, old_status, new_status)
        """
        self.health_change_callbacks.append(callback)
    
    async def check_model_health(self, model_key: str) -> HealthCheck:
        """Perform health check on a specific model.
        
        Args:
            model_key: Model key to check
            
        Returns:
            HealthCheck result
        """
        start_time = time.time()
        provider = model_key.split(':', 1)[0]
        
        try:
            if provider == "ollama":
                return await self._check_ollama_health(model_key, start_time)
            elif provider in ["openai", "anthropic", "google"]:
                return await self._check_api_health(model_key, start_time)
            elif provider == "huggingface":
                return await self._check_huggingface_health(model_key, start_time)
            else:
                return HealthCheck(
                    model_key=model_key,
                    status=HealthStatus.UNKNOWN,
                    timestamp=start_time,
                    response_time_ms=0.0,
                    error_message="Unknown provider type"
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                model_key=model_key,
                status=HealthStatus.UNHEALTHY,
                timestamp=start_time,
                response_time_ms=response_time,
                error_message=str(e)
            )
    
    def get_health_status(self, model_key: str) -> Optional[HealthMetrics]:
        """Get current health metrics for a model.
        
        Args:
            model_key: Model key
            
        Returns:
            HealthMetrics or None if not monitored
        """
        return self.health_metrics.get(model_key)
    
    def get_all_health_status(self) -> Dict[str, HealthMetrics]:
        """Get health status for all monitored models."""
        return self.health_metrics.copy()
    
    def get_unhealthy_models(self) -> List[str]:
        """Get list of currently unhealthy models."""
        return [
            model_key for model_key, metrics in self.health_metrics.items()
            if metrics.current_status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
        ]
    
    async def recover_model(self, model_key: str) -> bool:
        """Attempt to recover an unhealthy model.
        
        Args:
            model_key: Model key to recover
            
        Returns:
            True if recovery was attempted (doesn't guarantee success)
        """
        if not self.recovery_enabled:
            logger.info(f"Recovery disabled, skipping recovery for {model_key}")
            return False
            
        provider = model_key.split(':', 1)[0]
        recovery_func = self.recovery_strategies.get(provider)
        
        if not recovery_func:
            logger.warning(f"No recovery strategy for provider: {provider}")
            return False
        
        try:
            metrics = self.health_metrics.get(model_key)
            if metrics:
                metrics.recovery_attempts += 1
                metrics.last_recovery_time = time.time()
                
            logger.info(f"Attempting recovery for {model_key}")
            success = await recovery_func(model_key)
            
            if success:
                # Mark as recovering and schedule immediate health check
                if metrics:
                    old_status = metrics.current_status
                    metrics.current_status = HealthStatus.RECOVERING
                    self._notify_status_change(model_key, old_status, HealthStatus.RECOVERING)
                    
                logger.info(f"Recovery initiated for {model_key}")
            else:
                logger.error(f"Recovery failed for {model_key}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error during recovery of {model_key}: {e}")
            return False
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Health monitoring loop started")
        
        with ThreadPoolExecutor(max_workers=5, thread_name_prefix="HealthCheck") as executor:
            while self.monitoring_active and not self._shutdown_event.is_set():
                try:
                    # Schedule health checks for all monitored models
                    futures = []
                    for model_key in list(self.health_metrics.keys()):
                        future = executor.submit(self._check_and_update_health, model_key)
                        futures.append(future)
                    
                    # Wait for all checks to complete
                    for future in futures:
                        try:
                            future.result(timeout=30.0)
                        except Exception as e:
                            logger.error(f"Health check failed: {e}")
                    
                    # Wait for next check interval
                    self._shutdown_event.wait(self.check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(5)  # Brief pause before retry
        
        logger.info("Health monitoring loop stopped")
    
    def _check_and_update_health(self, model_key: str) -> None:
        """Check health and update metrics for a single model."""
        try:
            # Perform health check
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                health_check = loop.run_until_complete(self.check_model_health(model_key))
            finally:
                loop.close()
            
            # Update metrics
            self._update_health_metrics(model_key, health_check)
            
            # Check if recovery is needed
            metrics = self.health_metrics[model_key]
            if (metrics.current_status == HealthStatus.UNHEALTHY and 
                metrics.consecutive_failures >= self.failure_threshold):
                
                # Attempt recovery
                if self.recovery_enabled:
                    recovery_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(recovery_loop)
                    try:
                        recovery_loop.run_until_complete(self.recover_model(model_key))
                    finally:
                        recovery_loop.close()
                        
        except Exception as e:
            logger.error(f"Error checking health for {model_key}: {e}")
    
    def _update_health_metrics(self, model_key: str, health_check: HealthCheck) -> None:
        """Update health metrics based on check result."""
        metrics = self.health_metrics[model_key]
        old_status = metrics.current_status
        
        # Add to recent checks
        metrics.recent_checks.append(health_check)
        if len(metrics.recent_checks) > self.max_history:
            metrics.recent_checks.pop(0)
        
        # Update counters
        metrics.total_checks += 1
        metrics.last_check = health_check.timestamp
        
        if health_check.status == HealthStatus.HEALTHY:
            metrics.consecutive_successes += 1
            metrics.consecutive_failures = 0
        else:
            metrics.consecutive_failures += 1
            metrics.consecutive_successes = 0
        
        # Calculate success rate
        recent_successes = sum(1 for check in metrics.recent_checks 
                              if check.status == HealthStatus.HEALTHY)
        metrics.success_rate = recent_successes / len(metrics.recent_checks) if metrics.recent_checks else 0.0
        
        # Calculate average response time
        response_times = [check.response_time_ms for check in metrics.recent_checks]
        metrics.avg_response_time_ms = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Determine new status
        new_status = self._determine_health_status(metrics, health_check)
        
        if new_status != old_status:
            metrics.current_status = new_status
            self._notify_status_change(model_key, old_status, new_status)
    
    def _determine_health_status(self, metrics: HealthMetrics, latest_check: HealthCheck) -> HealthStatus:
        """Determine health status based on metrics and latest check."""
        # If we're recovering, check if we can transition to healthy
        if metrics.current_status == HealthStatus.RECOVERING:
            if metrics.consecutive_successes >= self.recovery_threshold:
                return HealthStatus.HEALTHY
            elif metrics.consecutive_failures >= self.failure_threshold:
                return HealthStatus.UNHEALTHY
            else:
                return HealthStatus.RECOVERING
        
        # Normal status determination
        if latest_check.status == HealthStatus.HEALTHY:
            # Check if response time indicates degradation
            if latest_check.response_time_ms > self.degraded_response_time:
                return HealthStatus.DEGRADED
            elif metrics.consecutive_successes >= self.recovery_threshold:
                return HealthStatus.HEALTHY
        
        # Check for failure patterns
        if metrics.consecutive_failures >= self.failure_threshold:
            return HealthStatus.UNHEALTHY
        elif metrics.success_rate < 0.8 and metrics.total_checks > 10:
            return HealthStatus.DEGRADED
        
        # Default based on latest check
        return latest_check.status
    
    def _notify_status_change(self, model_key: str, old_status: HealthStatus, new_status: HealthStatus) -> None:
        """Notify registered callbacks of status changes."""
        logger.info(f"Health status change for {model_key}: {old_status.value} -> {new_status.value}")
        
        for callback in self.health_change_callbacks:
            try:
                callback(model_key, old_status, new_status)
            except Exception as e:
                logger.error(f"Error in health change callback: {e}")
    
    async def _check_ollama_health(self, model_key: str, start_time: float) -> HealthCheck:
        """Check health of Ollama model using Phase 2 service manager."""
        model_name = model_key.split(':', 1)[1]
        ollama_manager = SERVICE_MANAGERS.get("ollama")
        
        if not ollama_manager:
            return HealthCheck(
                model_key=model_key,
                status=HealthStatus.UNHEALTHY,
                timestamp=start_time,
                response_time_ms=0.0,
                error_message="Ollama service manager not available"
            )
        
        try:
            # Check if service is running
            if not ollama_manager.is_running():
                return HealthCheck(
                    model_key=model_key,
                    status=HealthStatus.UNHEALTHY,
                    timestamp=start_time,
                    response_time_ms=(time.time() - start_time) * 1000,
                    error_message="Ollama service not running"
                )
            
            # Check if model is available
            if not ollama_manager.is_model_available(model_name):
                return HealthCheck(
                    model_key=model_key,
                    status=HealthStatus.UNHEALTHY,
                    timestamp=start_time,
                    response_time_ms=(time.time() - start_time) * 1000,
                    error_message="Model not available in Ollama"
                )
            
            # Perform actual health check
            is_healthy = await ollama_manager.health_check_model(model_name)
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                model_key=model_key,
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                timestamp=start_time,
                response_time_ms=response_time,
                error_message=None if is_healthy else "Health check failed"
            )
            
        except Exception as e:
            return HealthCheck(
                model_key=model_key,
                status=HealthStatus.UNHEALTHY,
                timestamp=start_time,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def _check_api_health(self, model_key: str, start_time: float) -> HealthCheck:
        """Check health of API-based models (OpenAI, Anthropic, etc.)."""
        try:
            provider, model_name = model_key.split(':', 1)
            
            # Get model instance
            model = self.model_registry.get_model(model_name, provider)
            
            # Perform a minimal inference to check health
            test_prompt = "Test"
            
            # Set timeout for health check
            timeout = 30.0
            start_inference = time.time()
            
            try:
                # Simple generation with minimal tokens
                result = await asyncio.wait_for(
                    asyncio.to_thread(model.generate, test_prompt, max_tokens=1),
                    timeout=timeout
                )
                
                response_time = (time.time() - start_time) * 1000
                
                if result and len(result.strip()) > 0:
                    return HealthCheck(
                        model_key=model_key,
                        status=HealthStatus.HEALTHY,
                        timestamp=start_time,
                        response_time_ms=response_time
                    )
                else:
                    return HealthCheck(
                        model_key=model_key,
                        status=HealthStatus.UNHEALTHY,
                        timestamp=start_time,
                        response_time_ms=response_time,
                        error_message="Empty response from model"
                    )
                    
            except asyncio.TimeoutError:
                return HealthCheck(
                    model_key=model_key,
                    status=HealthStatus.UNHEALTHY,
                    timestamp=start_time,
                    response_time_ms=timeout * 1000,
                    error_message="Health check timeout"
                )
                
        except Exception as e:
            return HealthCheck(
                model_key=model_key,
                status=HealthStatus.UNHEALTHY,
                timestamp=start_time,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def _check_huggingface_health(self, model_key: str, start_time: float) -> HealthCheck:
        """Check health of HuggingFace models."""
        try:
            provider, model_name = model_key.split(':', 1)
            model = self.model_registry.get_model(model_name, provider)
            
            # For HuggingFace, check if the model is loaded and accessible
            # This is a simplified check - could be enhanced based on specific model types
            if hasattr(model, 'is_available') and callable(model.is_available):
                is_available = model.is_available()
            else:
                is_available = True  # Assume available if no check method
                
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                model_key=model_key,
                status=HealthStatus.HEALTHY if is_available else HealthStatus.UNHEALTHY,
                timestamp=start_time,
                response_time_ms=response_time,
                error_message=None if is_available else "Model not available"
            )
            
        except Exception as e:
            return HealthCheck(
                model_key=model_key,
                status=HealthStatus.UNHEALTHY,
                timestamp=start_time,
                response_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    async def _recover_ollama_model(self, model_key: str) -> bool:
        """Recover Ollama model using Phase 2 service manager."""
        model_name = model_key.split(':', 1)[1]
        ollama_manager = SERVICE_MANAGERS.get("ollama")
        
        if not ollama_manager:
            return False
        
        try:
            # Try to ensure service is running
            if not ollama_manager.ensure_running():
                return False
            
            # Try to ensure model is available (will pull if needed)
            success = ollama_manager.ensure_model_available(model_name, auto_pull=True)
            return success
            
        except Exception as e:
            logger.error(f"Error recovering Ollama model {model_key}: {e}")
            return False
    
    async def _recover_docker_service(self, model_key: str) -> bool:
        """Recover Docker-based service."""
        docker_manager = SERVICE_MANAGERS.get("docker")
        
        if not docker_manager:
            return False
        
        try:
            # Try to ensure Docker service is running
            return docker_manager.ensure_running()
            
        except Exception as e:
            logger.error(f"Error recovering Docker service for {model_key}: {e}")
            return False
    
    async def _recover_api_service(self, model_key: str) -> bool:
        """Recover API-based service (mostly just wait and retry)."""
        # For API services, recovery is mostly about waiting for service to come back
        # Could implement exponential backoff, circuit breaker patterns, etc.
        
        # Simple recovery: wait a bit and hope the service recovers
        await asyncio.sleep(5.0)
        return True  # Assume recovery attempt is valid


# Utility functions for integration
def create_health_monitor(
    model_registry: ModelRegistry,
    **kwargs
) -> ModelHealthMonitor:
    """Create health monitor from model registry."""
    return ModelHealthMonitor(model_registry, **kwargs)


def setup_basic_health_monitoring(
    model_registry: ModelRegistry,
    models: Optional[List[str]] = None
) -> ModelHealthMonitor:
    """Setup basic health monitoring with reasonable defaults."""
    monitor = ModelHealthMonitor(
        model_registry,
        check_interval=120,  # 2 minutes
        max_history=50,
        recovery_enabled=True
    )
    
    # Add basic logging callback
    def log_status_changes(model_key: str, old_status: HealthStatus, new_status: HealthStatus):
        logger.info(f"Model {model_key} status: {old_status.value} -> {new_status.value}")
    
    monitor.add_health_change_callback(log_status_changes)
    
    # Start monitoring
    monitor.start_monitoring(models)
    
    return monitor