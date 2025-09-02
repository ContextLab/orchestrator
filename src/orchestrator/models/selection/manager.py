"""Intelligent model manager with lifecycle management and performance optimization."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ...core.model import Model, ModelMetrics
from ..registry import ModelRegistry
from ..optimization.caching import ModelResponseCache
from ..optimization.pooling import ConnectionPool
from .strategies import (
    FallbackStrategy,
    SelectionResult,
    SelectionStrategy, 
    TaskRequirements,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelUsageStats:
    """Statistics for model usage and performance."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    total_cost: float = 0.0
    last_used: Optional[float] = None
    error_messages: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property 
    def average_latency(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "average_latency": self.average_latency,
            "total_cost": self.total_cost,
            "last_used": self.last_used,
            "recent_errors": self.error_messages[-5:],  # Last 5 errors
        }


@dataclass
class ModelLoadBalancer:
    """Load balancer for distributing requests across model instances."""
    
    strategy: str = "round_robin"  # "round_robin", "least_loaded", "fastest"
    _current_index: int = field(default=0, init=False)
    _request_counts: Dict[str, int] = field(default_factory=dict, init=False)
    
    def select_instance(self, instances: List[Tuple[Model, str]], stats: Dict[str, ModelUsageStats]) -> Tuple[Model, str]:
        """Select the best instance based on load balancing strategy."""
        if not instances:
            raise ValueError("No model instances available")
        
        if len(instances) == 1:
            return instances[0]
        
        if self.strategy == "round_robin":
            return self._round_robin_select(instances)
        elif self.strategy == "least_loaded":
            return self._least_loaded_select(instances, stats)
        elif self.strategy == "fastest":
            return self._fastest_select(instances, stats)
        else:
            # Default to round robin
            return self._round_robin_select(instances)
    
    def _round_robin_select(self, instances: List[Tuple[Model, str]]) -> Tuple[Model, str]:
        """Round robin selection."""
        instance = instances[self._current_index % len(instances)]
        self._current_index += 1
        return instance
    
    def _least_loaded_select(self, instances: List[Tuple[Model, str]], stats: Dict[str, ModelUsageStats]) -> Tuple[Model, str]:
        """Select instance with least load."""
        min_requests = float('inf')
        selected = instances[0]
        
        for model, provider in instances:
            model_key = f"{provider}:{model.name}"
            request_count = stats.get(model_key, ModelUsageStats()).total_requests
            if request_count < min_requests:
                min_requests = request_count
                selected = (model, provider)
        
        return selected
    
    def _fastest_select(self, instances: List[Tuple[Model, str]], stats: Dict[str, ModelUsageStats]) -> Tuple[Model, str]:
        """Select instance with best latency."""
        best_latency = float('inf')
        selected = instances[0]
        
        for model, provider in instances:
            model_key = f"{provider}:{model.name}"
            avg_latency = stats.get(model_key, ModelUsageStats()).average_latency
            if avg_latency == 0.0:  # New model, give it a chance
                return (model, provider)
            if avg_latency < best_latency:
                best_latency = avg_latency
                selected = (model, provider)
        
        return selected


class ModelManager:
    """
    Intelligent model manager with lifecycle management and performance optimization.
    
    This manager provides:
    - Intelligent model selection based on task requirements
    - Performance monitoring and optimization
    - Caching and connection pooling
    - Load balancing across model instances
    - Health monitoring and failover
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        selection_strategy: Optional[SelectionStrategy] = None,
        enable_caching: bool = True,
        enable_pooling: bool = True,
        max_cache_size: int = 1000,
        pool_size: int = 10,
    ):
        """
        Initialize model manager.
        
        Args:
            registry: Model registry to use
            selection_strategy: Strategy for model selection
            enable_caching: Whether to enable response caching
            enable_pooling: Whether to enable connection pooling
            max_cache_size: Maximum cache size
            pool_size: Connection pool size per provider
        """
        self.registry = registry
        self.selection_strategy = selection_strategy or FallbackStrategy()
        self.enable_caching = enable_caching
        self.enable_pooling = enable_pooling
        
        # Performance tracking
        self._model_stats: Dict[str, ModelUsageStats] = defaultdict(ModelUsageStats)
        self._active_models: Dict[str, List[Tuple[Model, str]]] = defaultdict(list)
        self._load_balancer = ModelLoadBalancer()
        
        # Caching and pooling
        self._cache: Optional[ModelResponseCache] = None
        self._pools: Dict[str, ConnectionPool] = {}
        
        if enable_caching:
            self._cache = ModelResponseCache(max_size=max_cache_size)
        
        if enable_pooling:
            self._pool_size = pool_size
        
        # Health monitoring
        self._last_health_check = 0.0
        self._health_check_interval = 300.0  # 5 minutes
        self._unhealthy_models: Set[str] = set()
        
        logger.info(f"ModelManager initialized with caching={enable_caching}, pooling={enable_pooling}")
    
    async def select_model(
        self,
        requirements: TaskRequirements,
        force_selection: bool = False,
    ) -> SelectionResult:
        """
        Select the best model for given requirements.
        
        Args:
            requirements: Task requirements
            force_selection: Force new selection even if cached
            
        Returns:
            Selection result with chosen model
        """
        # Check if we have a cached selection for this exact requirement
        requirements_key = str(hash(frozenset(requirements.to_dict().items())))
        
        if not force_selection and self._cache:
            cached_selection = await self._cache.get_cached_selection(requirements_key)
            if cached_selection:
                logger.debug(f"Using cached model selection for {requirements.task_type}")
                return cached_selection
        
        # Ensure registry is initialized
        if not self.registry.is_initialized:
            await self.registry.initialize()
        
        # Filter out unhealthy models
        available_models = await self._get_healthy_models()
        
        # Use selection strategy
        selection_result = await self.selection_strategy.select_model(
            self.registry, requirements, available_models
        )
        
        # Cache the selection
        if self._cache:
            await self._cache.cache_selection(requirements_key, selection_result)
        
        # Track model usage
        model_key = f"{selection_result.provider}:{selection_result.model.name}"
        self._active_models[model_key].append((selection_result.model, selection_result.provider))
        
        logger.info(f"Selected model: {selection_result.model.name} from {selection_result.provider}")
        return selection_result
    
    async def generate_with_model(
        self,
        model: Model,
        provider: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text using specified model with performance tracking.
        
        Args:
            model: Model to use
            provider: Provider name
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use cached responses
            **kwargs: Additional model parameters
            
        Returns:
            Tuple of (generated_text, metadata)
        """
        model_key = f"{provider}:{model.name}"
        start_time = time.time()
        
        # Check cache first
        if use_cache and self._cache:
            cache_key = self._cache.generate_cache_key(prompt, temperature, max_tokens, **kwargs)
            cached_response = await self._cache.get(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for model {model.name}")
                return cached_response["response"], {"cached": True, "latency": 0.0}
        
        try:
            # Use connection pool if available
            if self.enable_pooling and provider in self._pools:
                pool = self._pools[provider]
                response = await pool.execute_with_model(
                    model, "generate", prompt=prompt, temperature=temperature, 
                    max_tokens=max_tokens, **kwargs
                )
            else:
                # Direct model call
                response = await model.generate(prompt, temperature, max_tokens, **kwargs)
            
            # Calculate metrics
            latency = time.time() - start_time
            cost = await model.estimate_cost(prompt, max_tokens)
            
            # Update statistics
            stats = self._model_stats[model_key]
            stats.total_requests += 1
            stats.successful_requests += 1
            stats.total_latency += latency
            stats.total_cost += cost
            stats.last_used = time.time()
            
            # Update model metrics
            if hasattr(model, 'metrics') and model.metrics:
                self._update_model_metrics(model, latency, True)
            
            # Cache successful response
            if use_cache and self._cache:
                cache_key = self._cache.generate_cache_key(prompt, temperature, max_tokens, **kwargs)
                await self._cache.put(cache_key, {
                    "response": response,
                    "model": model.name,
                    "provider": provider,
                    "timestamp": time.time(),
                })
            
            metadata = {
                "cached": False,
                "latency": latency,
                "cost": cost,
                "model": model.name,
                "provider": provider,
            }
            
            logger.debug(f"Generated response with {model.name} (latency: {latency:.3f}s, cost: ${cost:.6f})")
            return response, metadata
            
        except Exception as e:
            # Track failures
            error_latency = time.time() - start_time
            stats = self._model_stats[model_key]
            stats.total_requests += 1
            stats.failed_requests += 1
            stats.error_messages.append(str(e))
            stats.last_used = time.time()
            
            # Update model metrics
            if hasattr(model, 'metrics') and model.metrics:
                self._update_model_metrics(model, error_latency, False)
            
            # Mark model as potentially unhealthy if error rate is high
            if stats.success_rate < 0.8 and stats.total_requests > 10:
                self._unhealthy_models.add(model_key)
                logger.warning(f"Marked model {model_key} as unhealthy (success rate: {stats.success_rate:.2f})")
            
            logger.error(f"Generation failed for {model.name}: {e}")
            raise
    
    async def generate_structured_with_model(
        self,
        model: Model,
        provider: str,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate structured output using specified model.
        
        Args:
            model: Model to use
            provider: Provider name
            prompt: Input prompt
            schema: JSON schema for output
            temperature: Sampling temperature
            use_cache: Whether to use cached responses
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (structured_output, metadata)
        """
        model_key = f"{provider}:{model.name}"
        start_time = time.time()
        
        # Check cache first
        if use_cache and self._cache:
            cache_key = self._cache.generate_cache_key(prompt, temperature, schema=schema, **kwargs)
            cached_response = await self._cache.get(cache_key)
            if cached_response:
                return cached_response["response"], {"cached": True, "latency": 0.0}
        
        try:
            # Use connection pool if available
            if self.enable_pooling and provider in self._pools:
                pool = self._pools[provider]
                response = await pool.execute_with_model(
                    model, "generate_structured", prompt=prompt, 
                    schema=schema, temperature=temperature, **kwargs
                )
            else:
                # Direct model call
                response = await model.generate_structured(prompt, schema, temperature, **kwargs)
            
            # Calculate metrics
            latency = time.time() - start_time
            # Estimate cost (approximate for structured generation)
            estimated_tokens = len(prompt.split()) * 1.3  # Rough estimate
            cost = await model.estimate_cost(prompt, int(estimated_tokens))
            
            # Update statistics
            stats = self._model_stats[model_key]
            stats.total_requests += 1
            stats.successful_requests += 1
            stats.total_latency += latency
            stats.total_cost += cost
            stats.last_used = time.time()
            
            # Cache response
            if use_cache and self._cache:
                cache_key = self._cache.generate_cache_key(prompt, temperature, schema=schema, **kwargs)
                await self._cache.put(cache_key, {
                    "response": response,
                    "model": model.name,
                    "provider": provider,
                    "timestamp": time.time(),
                })
            
            metadata = {
                "cached": False,
                "latency": latency,
                "cost": cost,
                "model": model.name,
                "provider": provider,
            }
            
            return response, metadata
            
        except Exception as e:
            # Track failures
            error_latency = time.time() - start_time
            stats = self._model_stats[model_key]
            stats.total_requests += 1
            stats.failed_requests += 1
            stats.error_messages.append(str(e))
            
            logger.error(f"Structured generation failed for {model.name}: {e}")
            raise
    
    async def get_best_model(self, requirements: TaskRequirements) -> Tuple[Model, str]:
        """
        Get the best available model instance for requirements.
        
        Args:
            requirements: Task requirements
            
        Returns:
            Tuple of (model, provider)
        """
        selection_result = await self.select_model(requirements)
        model_key = f"{selection_result.provider}:{selection_result.model.name}"
        
        # Get available instances for this model
        instances = self._active_models.get(model_key, [(selection_result.model, selection_result.provider)])
        
        # Use load balancer to select best instance
        return self._load_balancer.select_instance(instances, self._model_stats)
    
    async def get_model_stats(self, model_name: Optional[str] = None, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model usage statistics.
        
        Args:
            model_name: Specific model name (all models if None)
            provider: Specific provider (all providers if None)
            
        Returns:
            Dictionary with model statistics
        """
        if model_name and provider:
            model_key = f"{provider}:{model_name}"
            if model_key in self._model_stats:
                return {model_key: self._model_stats[model_key].to_dict()}
            return {}
        
        stats = {}
        for model_key, model_stats in self._model_stats.items():
            if model_name and not model_key.endswith(f":{model_name}"):
                continue
            if provider and not model_key.startswith(f"{provider}:"):
                continue
            stats[model_key] = model_stats.to_dict()
        
        return stats
    
    async def health_check(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform health check on all active models.
        
        Args:
            force: Force health check even if recently performed
            
        Returns:
            Health check results
        """
        current_time = time.time()
        if not force and (current_time - self._last_health_check) < self._health_check_interval:
            return {"status": "skipped", "last_check": self._last_health_check}
        
        health_results = {}
        unhealthy_count = 0
        
        # Check registry health first
        registry_health = await self.registry.health_check()
        health_results["registry"] = registry_health
        
        # Check individual model health
        for model_key, instances in self._active_models.items():
            if not instances:
                continue
            
            model, provider = instances[0]  # Use first instance for health check
            
            try:
                is_healthy = await model.health_check()
                stats = self._model_stats[model_key]
                
                health_info = {
                    "healthy": is_healthy and stats.success_rate >= 0.8,
                    "success_rate": stats.success_rate,
                    "average_latency": stats.average_latency,
                    "total_requests": stats.total_requests,
                    "last_used": stats.last_used,
                }
                
                if not health_info["healthy"]:
                    unhealthy_count += 1
                    self._unhealthy_models.add(model_key)
                    health_info["issues"] = []
                    if not is_healthy:
                        health_info["issues"].append("model_health_check_failed")
                    if stats.success_rate < 0.8:
                        health_info["issues"].append(f"low_success_rate_{stats.success_rate:.2f}")
                else:
                    # Remove from unhealthy set if now healthy
                    self._unhealthy_models.discard(model_key)
                
                health_results[model_key] = health_info
                
            except Exception as e:
                unhealthy_count += 1
                self._unhealthy_models.add(model_key)
                health_results[model_key] = {
                    "healthy": False,
                    "error": str(e),
                }
        
        self._last_health_check = current_time
        
        return {
            "status": "completed",
            "timestamp": current_time,
            "total_models": len(self._active_models),
            "healthy_models": len(self._active_models) - unhealthy_count,
            "unhealthy_models": unhealthy_count,
            "results": health_results,
        }
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """
        Optimize performance based on collected statistics.
        
        Returns:
            Optimization results
        """
        optimizations = []
        
        # Analyze model performance
        slow_models = []
        expensive_models = []
        unreliable_models = []
        
        for model_key, stats in self._model_stats.items():
            if stats.average_latency > 5.0:  # > 5 seconds
                slow_models.append(model_key)
            
            if stats.total_cost / max(stats.total_requests, 1) > 0.01:  # > $0.01 per request
                expensive_models.append(model_key)
            
            if stats.success_rate < 0.9 and stats.total_requests > 5:
                unreliable_models.append(model_key)
        
        # Suggest optimizations
        if slow_models:
            optimizations.append(f"Consider faster alternatives for slow models: {', '.join(slow_models)}")
        
        if expensive_models:
            optimizations.append(f"Consider cost-effective alternatives for expensive models: {', '.join(expensive_models)}")
        
        if unreliable_models:
            optimizations.append(f"Investigate reliability issues with: {', '.join(unreliable_models)}")
        
        # Cache optimization
        if self._cache:
            cache_stats = await self._cache.get_stats()
            if cache_stats.get("hit_rate", 0) < 0.2:
                optimizations.append("Low cache hit rate - consider adjusting cache strategy or size")
        
        return {
            "timestamp": time.time(),
            "optimizations": optimizations,
            "slow_models": slow_models,
            "expensive_models": expensive_models,
            "unreliable_models": unreliable_models,
        }
    
    async def cleanup(self) -> None:
        """Clean up manager resources."""
        logger.info("Cleaning up ModelManager resources")
        
        # Clean up cache
        if self._cache:
            await self._cache.cleanup()
        
        # Clean up connection pools
        cleanup_tasks = []
        for pool in self._pools.values():
            cleanup_tasks.append(pool.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear tracking data
        self._model_stats.clear()
        self._active_models.clear()
        self._unhealthy_models.clear()
        self._pools.clear()
        
        logger.info("ModelManager cleanup completed")
    
    def get_manager_info(self) -> Dict[str, Any]:
        """Get manager status and configuration info."""
        return {
            "strategy": self.selection_strategy.name,
            "caching_enabled": self.enable_caching,
            "pooling_enabled": self.enable_pooling,
            "active_models": len(self._active_models),
            "tracked_models": len(self._model_stats),
            "unhealthy_models": len(self._unhealthy_models),
            "connection_pools": len(self._pools),
            "last_health_check": self._last_health_check,
        }
    
    async def _get_healthy_models(self) -> List[Model]:
        """Get list of healthy models."""
        all_models = []
        model_info = self.registry.list_models()
        
        for model_name, info in model_info.items():
            if "error" in info:
                continue
            
            model_key = f"{info['provider']}:{model_name}"
            if model_key in self._unhealthy_models:
                continue
            
            try:
                model = await self.registry.get_model(model_name, info["provider"])
                all_models.append(model)
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
        
        return all_models
    
    def _update_model_metrics(self, model: Model, latency: float, success: bool) -> None:
        """Update model's internal metrics."""
        if not hasattr(model, 'metrics') or not model.metrics:
            model.metrics = ModelMetrics()
        
        # Simple exponential moving average for latency
        if model.metrics.latency_p50 == 0:
            model.metrics.latency_p50 = latency
        else:
            alpha = 0.1  # Smoothing factor
            model.metrics.latency_p50 = alpha * latency + (1 - alpha) * model.metrics.latency_p50
        
        # Update success rate
        current_success_rate = model.metrics.success_rate
        if success:
            model.metrics.success_rate = min(1.0, current_success_rate + 0.01)
        else:
            model.metrics.success_rate = max(0.0, current_success_rate - 0.05)
    
    def __str__(self) -> str:
        """String representation of manager."""
        return f"ModelManager(strategy={self.selection_strategy.name}, models={len(self._active_models)})"


# Alias for backward compatibility
ModelSelectionManager = ModelManager