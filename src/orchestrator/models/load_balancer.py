"""Load balancing and failover for model selection."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import random

from ..core.model import Model
from .model_registry import ModelRegistry, NoEligibleModelsError


@dataclass
class ModelPoolConfig:
    """Configuration for a model pool."""
    
    models: List[Dict[str, Any]] = field(default_factory=list)
    # Each model dict contains:
    # - model: str (model identifier)
    # - weight: float (relative weight for selection)
    # - max_concurrent: int (max concurrent requests)
    
    always_available: bool = False  # If true, pool is always considered available
    fallback_pool: Optional[str] = None  # Name of fallback pool if this fails
    
    retry_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "backoff": "exponential",
        "initial_delay": 1.0,
        "max_delay": 60.0,
        "jitter": 0.1
    })


class LoadBalancer:
    """
    Load balancer for distributing requests across models.
    
    Handles:
    - Weighted load distribution
    - Concurrent request limiting
    - Circuit breaking for failed models
    - Automatic failover to backup pools
    - Retry with exponential backoff
    """
    
    def __init__(self, registry: ModelRegistry):
        """
        Initialize load balancer.
        
        Args:
            registry: Model registry to use
        """
        self.registry = registry
        self.pools: Dict[str, ModelPoolConfig] = {}
        self.model_states: Dict[str, ModelState] = defaultdict(ModelState)
        self._lock = asyncio.Lock()
    
    def configure_pool(self, name: str, config: ModelPoolConfig) -> None:
        """
        Configure a model pool.
        
        Args:
            name: Pool name
            config: Pool configuration
        """
        self.pools[name] = config
        
        # Initialize model states
        for model_info in config.models:
            model_id = model_info["model"]
            self.model_states[model_id].max_concurrent = model_info.get("max_concurrent", 10)
    
    async def select_from_pool(self, pool_name: str) -> Model:
        """
        Select a model from a pool with load balancing.
        
        Args:
            pool_name: Name of the pool
            
        Returns:
            Selected model
            
        Raises:
            NoEligibleModelsError: If no models available in pool
        """
        if pool_name not in self.pools:
            raise ValueError(f"Unknown pool: {pool_name}")
        
        pool = self.pools[pool_name]
        
        # Try to select from primary pool
        model = await self._select_from_pool_weighted(pool)
        
        if model is None and pool.fallback_pool:
            # Try fallback pool
            return await self.select_from_pool(pool.fallback_pool)
        
        if model is None:
            raise NoEligibleModelsError(f"No available models in pool {pool_name}")
        
        return model
    
    async def _select_from_pool_weighted(self, pool: ModelPoolConfig) -> Optional[Model]:
        """
        Select model from pool using weighted random selection.
        
        Args:
            pool: Pool configuration
            
        Returns:
            Selected model or None
        """
        available_models = []
        weights = []
        
        async with self._lock:
            for model_info in pool.models:
                model_id = model_info["model"]
                weight = model_info.get("weight", 1.0)
                
                # Check if model is available
                state = self.model_states[model_id]
                
                # Skip if circuit breaker is open
                if state.circuit_open:
                    if time.time() - state.last_failure < state.circuit_timeout:
                        continue
                    else:
                        # Reset circuit breaker
                        state.circuit_open = False
                
                # Skip if at max concurrent requests
                if state.current_requests >= state.max_concurrent:
                    continue
                
                # Try to get model from registry
                try:
                    provider, name = model_id.split(":", 1)
                    model = self.registry.get_model(name, provider)
                    
                    # Check if model is healthy
                    if not pool.always_available:
                        health = await model.health_check()
                        if not health:
                            continue
                    
                    available_models.append((model, model_id))
                    weights.append(weight)
                    
                except Exception:
                    continue
        
        if not available_models:
            return None
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return None
        
        rand = random.uniform(0, total_weight)
        cumulative = 0
        
        for (model, model_id), weight in zip(available_models, weights):
            cumulative += weight
            if rand <= cumulative:
                # Increment concurrent requests
                self.model_states[model_id].current_requests += 1
                return model
        
        # Fallback to last model
        model, model_id = available_models[-1]
        self.model_states[model_id].current_requests += 1
        return model
    
    async def execute_with_retry(
        self,
        model: Model,
        operation: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            model: Model to use
            operation: Operation name (e.g., "generate")
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Operation result
        """
        model_id = f"{model.provider}:{model.name}"
        state = self.model_states[model_id]
        
        # Get retry config from pool or use defaults
        retry_config = self._get_retry_config(model_id)
        max_retries = retry_config.get("max_retries", 3)
        backoff = retry_config.get("backoff", "exponential")
        initial_delay = retry_config.get("initial_delay", 1.0)
        max_delay = retry_config.get("max_delay", 60.0)
        jitter = retry_config.get("jitter", 0.1)
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Get the operation method
                if not hasattr(model, operation):
                    raise AttributeError(f"Model has no operation '{operation}'")
                
                method = getattr(model, operation)
                
                # Execute operation
                start_time = time.time()
                result = await method(*args, **kwargs)
                latency = time.time() - start_time
                
                # Update success metrics
                await self._update_success_metrics(model_id, latency)
                
                return result
                
            except Exception as e:
                last_error = e
                
                # Update failure metrics
                await self._update_failure_metrics(model_id)
                
                if attempt < max_retries - 1:
                    # Calculate delay
                    if backoff == "exponential":
                        delay = initial_delay * (2 ** attempt)
                    elif backoff == "linear":
                        delay = initial_delay * (attempt + 1)
                    else:
                        delay = initial_delay
                    
                    # Apply max delay cap
                    delay = min(delay, max_delay)
                    
                    # Add jitter
                    if jitter > 0:
                        delay += random.uniform(-jitter * delay, jitter * delay)
                    
                    await asyncio.sleep(delay)
                    
            finally:
                # Decrement concurrent requests
                state.current_requests = max(0, state.current_requests - 1)
        
        # All retries failed
        raise last_error
    
    def _get_retry_config(self, model_id: str) -> Dict[str, Any]:
        """Get retry configuration for a model."""
        # Find which pool contains this model
        for pool in self.pools.values():
            for model_info in pool.models:
                if model_info["model"] == model_id:
                    return pool.retry_config
        
        # Default retry config
        return {
            "max_retries": 3,
            "backoff": "exponential",
            "initial_delay": 1.0,
            "max_delay": 60.0,
            "jitter": 0.1
        }
    
    async def _update_success_metrics(self, model_id: str, latency: float) -> None:
        """Update metrics for successful request."""
        state = self.model_states[model_id]
        
        state.total_requests += 1
        state.successful_requests += 1
        state.consecutive_failures = 0
        state.last_success = time.time()
        
        # Update latency tracking
        state.latency_samples.append(latency)
        if len(state.latency_samples) > 100:
            state.latency_samples.pop(0)
        
        # Reset circuit breaker if it was open
        if state.circuit_open:
            state.circuit_open = False
    
    async def _update_failure_metrics(self, model_id: str) -> None:
        """Update metrics for failed request."""
        state = self.model_states[model_id]
        
        state.total_requests += 1
        state.failed_requests += 1
        state.consecutive_failures += 1
        state.last_failure = time.time()
        
        # Open circuit breaker if too many consecutive failures
        if state.consecutive_failures >= 5:
            state.circuit_open = True
            state.circuit_timeout = 60.0  # 1 minute timeout
    
    def get_pool_status(self, pool_name: str) -> Dict[str, Any]:
        """
        Get status of a model pool.
        
        Args:
            pool_name: Pool name
            
        Returns:
            Pool status information
        """
        if pool_name not in self.pools:
            return {"error": "Unknown pool"}
        
        pool = self.pools[pool_name]
        status = {
            "name": pool_name,
            "models": [],
            "fallback_pool": pool.fallback_pool,
            "always_available": pool.always_available
        }
        
        for model_info in pool.models:
            model_id = model_info["model"]
            state = self.model_states[model_id]
            
            model_status = {
                "model": model_id,
                "weight": model_info.get("weight", 1.0),
                "max_concurrent": state.max_concurrent,
                "current_requests": state.current_requests,
                "total_requests": state.total_requests,
                "successful_requests": state.successful_requests,
                "failed_requests": state.failed_requests,
                "success_rate": (
                    state.successful_requests / state.total_requests
                    if state.total_requests > 0 else 0.0
                ),
                "circuit_open": state.circuit_open,
                "consecutive_failures": state.consecutive_failures,
                "avg_latency": (
                    sum(state.latency_samples) / len(state.latency_samples)
                    if state.latency_samples else 0.0
                )
            }
            
            status["models"].append(model_status)
        
        return status


@dataclass
class ModelState:
    """Track state of a model for load balancing."""
    
    max_concurrent: int = 10
    current_requests: int = 0
    
    # Request tracking
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Circuit breaker
    consecutive_failures: int = 0
    circuit_open: bool = False
    circuit_timeout: float = 60.0
    last_failure: float = 0.0
    last_success: float = 0.0
    
    # Performance tracking
    latency_samples: List[float] = field(default_factory=list)