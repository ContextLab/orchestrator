"""Model registry for managing and selecting models."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, List, Optional

from ..core.model import Model, ModelMetrics


class ModelNotFoundError(Exception):
    """Raised when a requested model is not found."""
    pass


class NoEligibleModelsError(Exception):
    """Raised when no models meet the requirements."""
    pass


class ModelRegistry:
    """
    Central registry for all available models.
    
    Manages model registration, selection, and performance tracking
    using multi-armed bandit algorithms.
    """
    
    def __init__(self) -> None:
        """Initialize model registry."""
        self.models: Dict[str, Model] = {}
        self.model_selector = UCBModelSelector()
        self._model_health_cache: Dict[str, bool] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_health_check = 0  # Will be set to current time on first use
    
    def register_model(self, model: Model) -> None:
        """
        Register a new model.
        
        Args:
            model: Model to register
            
        Raises:
            ValueError: If model with same name already exists
        """
        model_key = self._get_model_key(model)
        
        if model_key in self.models:
            raise ValueError(f"Model '{model_key}' already registered")
        
        self.models[model_key] = model
        
        # Initialize model in selector
        self.model_selector.initialize_model(model_key, model.metrics)
    
    def unregister_model(self, model_name: str, provider: str = "") -> None:
        """
        Unregister a model.
        
        Args:
            model_name: Name of model to unregister
            provider: Provider name (optional)
        """
        if provider:
            model_key = f"{provider}:{model_name}"
        else:
            # Find model by name only
            model_key = None
            for key in self.models:
                if key.split(":")[-1] == model_name:
                    model_key = key
                    break
        
        if model_key and model_key in self.models:
            del self.models[model_key]
            self.model_selector.remove_model(model_key)
        else:
            raise ModelNotFoundError(f"Model '{model_name}' not found")
    
    def get_model(self, model_name: str, provider: str = "") -> Model:
        """
        Get a model by name.
        
        Args:
            model_name: Model name
            provider: Provider name (optional)
            
        Returns:
            Model instance
            
        Raises:
            ModelNotFoundError: If model not found
        """
        if provider:
            model_key = f"{provider}:{model_name}"
        else:
            # Find model by name only
            model_key = None
            for key in self.models:
                if key.split(":")[-1] == model_name:
                    model_key = key
                    break
        
        if model_key and model_key in self.models:
            return self.models[model_key]
        else:
            raise ModelNotFoundError(f"Model '{model_name}' not found")
    
    def list_models(self, provider: Optional[str] = None) -> List[str]:
        """
        List all registered models.
        
        Args:
            provider: Filter by provider (optional)
            
        Returns:
            List of model names
        """
        if provider:
            return [
                key for key in self.models.keys()
                if key.startswith(f"{provider}:")
            ]
        return list(self.models.keys())
    
    def list_providers(self) -> List[str]:
        """
        List all providers.
        
        Returns:
            List of provider names
        """
        providers = set()
        for key in self.models.keys():
            if ":" in key:
                providers.add(key.split(":")[0])
        return sorted(list(providers))
    
    async def get_available_models(self) -> List[str]:
        """
        Get list of available (healthy) models.
        
        Returns:
            List of available model keys
        """
        all_models = list(self.models.values())
        healthy_models = await self._filter_by_health(all_models)
        return [self._get_model_key(model) for model in healthy_models]
    
    async def select_model(self, requirements: Dict[str, Any]) -> Model:
        """
        Select best model for given requirements.
        
        Args:
            requirements: Requirements dictionary
            
        Returns:
            Selected model
            
        Raises:
            NoEligibleModelsError: If no models meet requirements
        """
        # Step 1: Filter by capabilities
        eligible_models = await self._filter_by_capabilities(requirements)
        
        if not eligible_models:
            raise NoEligibleModelsError("No models meet the specified requirements")
        
        # Step 2: Filter by health
        healthy_models = await self._filter_by_health(eligible_models)
        
        if not healthy_models:
            raise NoEligibleModelsError("No healthy models available")
        
        # Step 3: Use bandit algorithm for selection
        selected_key = self.model_selector.select(
            [self._get_model_key(model) for model in healthy_models],
            requirements
        )
        
        return self.models[selected_key]
    
    async def _filter_by_capabilities(self, requirements: Dict[str, Any]) -> List[Model]:
        """Filter models by capabilities."""
        eligible = []
        
        for model in self.models.values():
            if model.meets_requirements(requirements):
                eligible.append(model)
        
        return eligible
    
    async def _filter_by_health(self, models: List[Model]) -> List[Model]:
        """Filter models by health status."""
        # Check if cache is stale or if any models don't have cached health status
        current_time = asyncio.get_event_loop().time()
        
        # Only consider cache stale if we've checked before and enough time has passed
        cache_is_stale = (self._last_health_check > 0 and 
                         current_time - self._last_health_check > self._cache_ttl)
        
        # Check if any models are missing from cache
        missing_models = []
        for model in models:
            model_key = self._get_model_key(model)
            if model_key not in self._model_health_cache:
                missing_models.append(model)
        
        # Only refresh if cache is stale OR if we have missing models
        if cache_is_stale or missing_models:
            # If cache is stale, refresh all models; otherwise just refresh missing ones
            models_to_refresh = models if cache_is_stale else missing_models
            await self._refresh_health_cache(models_to_refresh)
            self._last_health_check = current_time
        
        # Return healthy models
        healthy = []
        for model in models:
            model_key = self._get_model_key(model)
            if self._model_health_cache.get(model_key, False):
                healthy.append(model)
        
        return healthy
    
    async def _refresh_health_cache(self, models: List[Model]) -> None:
        """Refresh health cache for models."""
        tasks = []
        for model in models:
            model_key = self._get_model_key(model)
            tasks.append(self._check_model_health(model_key, model))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_model_health(self, model_key: str, model: Model) -> None:
        """Check health of a single model."""
        try:
            is_healthy = await model.health_check()
            self._model_health_cache[model_key] = is_healthy
        except Exception:
            self._model_health_cache[model_key] = False
    
    def update_model_performance(
        self, 
        model: Model, 
        success: bool, 
        latency: float = 0.0,
        cost: float = 0.0
    ) -> None:
        """
        Update model performance metrics.
        
        Args:
            model: Model that was used
            success: Whether the operation was successful
            latency: Operation latency in seconds
            cost: Operation cost in USD
        """
        model_key = self._get_model_key(model)
        
        # Update bandit algorithm
        reward = self._calculate_reward(success, latency, cost)
        self.model_selector.update_reward(model_key, reward)
        
        # Update model metrics
        self._update_model_metrics(model, success, latency, cost)
    
    def _calculate_reward(self, success: bool, latency: float, cost: float) -> float:
        """Calculate reward for bandit algorithm."""
        if not success:
            return 0.0
        
        # Reward formula: base reward - latency penalty - cost penalty
        base_reward = 1.0
        latency_penalty = min(latency / 10.0, 0.5)  # Cap at 0.5
        cost_penalty = min(cost * 100, 0.3)  # Cap at 0.3
        
        return max(base_reward - latency_penalty - cost_penalty, 0.1)
    
    def _update_model_metrics(
        self, 
        model: Model, 
        success: bool, 
        latency: float, 
        cost: float
    ) -> None:
        """Update model metrics with new data."""
        metrics = model.metrics
        
        # Update success rate (simple exponential moving average)
        alpha = 0.1
        metrics.success_rate = (
            alpha * (1.0 if success else 0.0) + 
            (1 - alpha) * metrics.success_rate
        )
        
        # Update latency (only if successful)
        if success and latency > 0:
            metrics.latency_p50 = (
                alpha * latency + (1 - alpha) * metrics.latency_p50
            )
        
        # Update cost per token (simple average)
        if cost > 0:
            metrics.cost_per_token = (
                alpha * cost + (1 - alpha) * metrics.cost_per_token
            )
    
    def _get_model_key(self, model: Model) -> str:
        """Get unique key for model."""
        return f"{model.provider}:{model.name}"
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_models": len(self.models),
            "providers": len(self.list_providers()),
            "healthy_models": sum(1 for h in self._model_health_cache.values() if h),
            "selection_stats": self.model_selector.get_statistics(),
        }
        
        # Provider breakdown
        provider_counts = {}
        for key in self.models.keys():
            if ":" in key:
                provider = key.split(":")[0]
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        stats["provider_breakdown"] = provider_counts
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset all performance statistics."""
        self.model_selector.reset_statistics()
        self._model_health_cache.clear()
        self._last_health_check = 0


class UCBModelSelector:
    """
    Upper Confidence Bound algorithm for model selection.
    
    Balances exploration and exploitation when selecting models
    based on their performance history.
    """
    
    def __init__(self, exploration_factor: float = 2.0) -> None:
        """
        Initialize UCB selector.
        
        Args:
            exploration_factor: Exploration parameter (higher = more exploration)
        """
        self.exploration_factor = exploration_factor
        self.model_stats: Dict[str, Dict[str, float]] = {}
        self.total_attempts = 0
        self._pending_attempts: set = set()  # Track models with pending attempts
    
    def initialize_model(self, model_key: str, metrics: ModelMetrics) -> None:
        """
        Initialize model in selector.
        
        Args:
            model_key: Unique model key
            metrics: Model performance metrics
        """
        self.model_stats[model_key] = {
            "attempts": 0,
            "successes": 0,
            "total_reward": 0.0,
            "average_reward": metrics.success_rate,
        }
    
    def select(self, model_keys: List[str], context: Dict[str, Any]) -> str:
        """
        Select model using UCB algorithm.
        
        Args:
            model_keys: List of available model keys
            context: Selection context
            
        Returns:
            Selected model key
            
        Raises:
            ValueError: If no models available
        """
        if not model_keys:
            raise ValueError("No models available for selection")
        
        # Initialize any new models
        for key in model_keys:
            if key not in self.model_stats:
                self.model_stats[key] = {
                    "attempts": 0,
                    "successes": 0,
                    "total_reward": 0.0,
                    "average_reward": 0.5,  # Neutral starting point
                }
        
        # Calculate UCB scores
        scores = {}
        
        for key in model_keys:
            stats = self.model_stats[key]
            
            if stats["attempts"] == 0:
                # Explore untried models first
                scores[key] = float('inf')
            else:
                # UCB formula: average_reward + exploration_bonus
                average_reward = stats["average_reward"]
                exploration_bonus = self.exploration_factor * math.sqrt(
                    math.log(self.total_attempts + 1) / stats["attempts"]
                )
                scores[key] = average_reward + exploration_bonus
        
        # Select model with highest score
        selected_key = max(scores, key=scores.get)
        
        # Update attempt count immediately (as expected by tests)
        self.model_stats[selected_key]["attempts"] += 1
        self.total_attempts += 1
        
        # Mark as having a pending attempt (to avoid double counting in update_reward)
        self._pending_attempts.add(selected_key)
        
        return selected_key
    
    def update_reward(self, model_key: str, reward: float) -> None:
        """
        Update model statistics after execution.
        
        Args:
            model_key: Model key
            reward: Reward value (0-1)
        """
        if model_key not in self.model_stats:
            return
        
        stats = self.model_stats[model_key]
        
        # Only increment attempts if this wasn't already counted by select()
        if model_key not in self._pending_attempts:
            stats["attempts"] += 1
            self.total_attempts += 1
        else:
            # Remove from pending attempts (already counted)
            self._pending_attempts.remove(model_key)
        
        # Update statistics
        stats["total_reward"] += reward
        if reward > 0:
            stats["successes"] += 1
        
        # Update average reward
        stats["average_reward"] = stats["total_reward"] / stats["attempts"]
    
    def remove_model(self, model_key: str) -> None:
        """
        Remove model from selector.
        
        Args:
            model_key: Model key to remove
        """
        if model_key in self.model_stats:
            del self.model_stats[model_key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get selection statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_attempts": self.total_attempts,
            "models_tracked": len(self.model_stats),
            "model_performance": {
                key: {
                    "attempts": stats["attempts"],
                    "successes": stats["successes"],
                    "success_rate": stats["successes"] / max(stats["attempts"], 1),
                    "average_reward": stats["average_reward"],
                }
                for key, stats in self.model_stats.items()
            },
        }
    
    def reset_statistics(self) -> None:
        """Reset all selection statistics."""
        for stats in self.model_stats.values():
            stats["attempts"] = 0
            stats["successes"] = 0
            stats["total_reward"] = 0.0
            stats["average_reward"] = 0.5
        
        self.total_attempts = 0
        self._pending_attempts.clear()
    
    def get_model_confidence(self, model_key: str) -> float:
        """
        Get confidence score for a model.
        
        Args:
            model_key: Model key
            
        Returns:
            Confidence score (0-1)
        """
        if model_key not in self.model_stats:
            return 0.0
        
        stats = self.model_stats[model_key]
        
        if stats["attempts"] == 0:
            return 0.0
        
        # Confidence increases with more attempts and higher success rate
        attempt_factor = min(stats["attempts"] / 10.0, 1.0)  # Cap at 10 attempts
        success_factor = stats["average_reward"]
        
        return attempt_factor * success_factor