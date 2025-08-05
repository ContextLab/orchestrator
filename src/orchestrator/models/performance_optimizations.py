"""
Performance optimizations for model selection and registry operations.
Implements caching, indexing, and batch processing for large model registries.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import hashlib

from ..core.model import Model


@dataclass
class ModelIndex:
    """Index structure for fast model lookups."""
    
    # Expertise level indexes
    expertise_low: Set[str] = None
    expertise_medium: Set[str] = None  
    expertise_high: Set[str] = None
    expertise_very_high: Set[str] = None
    
    # Size-based indexes (model_key -> size)
    size_index: Dict[str, float] = None
    size_sorted: List[Tuple[str, float]] = None  # Sorted by size for range queries
    
    # Cost-based indexes
    free_models: Set[str] = None
    paid_models: Dict[str, float] = None  # model_key -> avg_cost_per_1k
    
    # Capability indexes
    vision_capable: Set[str] = None
    code_specialized: Set[str] = None
    function_calling: Set[str] = None
    
    # Provider indexes
    provider_index: Dict[str, Set[str]] = None  # provider -> set of model_keys
    
    def __post_init__(self):
        """Initialize empty collections if not provided."""
        if self.expertise_low is None:
            self.expertise_low = set()
        if self.expertise_medium is None:
            self.expertise_medium = set()
        if self.expertise_high is None:
            self.expertise_high = set()
        if self.expertise_very_high is None:
            self.expertise_very_high = set()
        if self.size_index is None:
            self.size_index = {}
        if self.size_sorted is None:
            self.size_sorted = []
        if self.free_models is None:
            self.free_models = set()
        if self.paid_models is None:
            self.paid_models = {}
        if self.vision_capable is None:
            self.vision_capable = set()
        if self.code_specialized is None:
            self.code_specialized = set()
        if self.function_calling is None:
            self.function_calling = set()
        if self.provider_index is None:
            self.provider_index = defaultdict(set)


class ModelRegistryOptimizer:
    """Performance optimizer for model registry operations."""
    
    def __init__(self):
        self.index = ModelIndex()
        self.selection_cache: Dict[str, Tuple[str, float]] = {}  # criteria_hash -> (model_key, timestamp)
        self.capability_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}  # model_key -> (analysis, timestamp)
        self.cache_ttl = 300  # 5 minutes
        self.index_dirty = True  # Flag to track if index needs rebuilding
        
    def build_index(self, models: Dict[str, Model]) -> None:
        """Build optimized indexes for fast model lookups."""
        start_time = time.time()
        
        # Clear existing indexes
        self.index = ModelIndex()
        
        for model_key, model in models.items():
            self._index_model(model_key, model)
        
        # Sort size index for range queries
        self.index.size_sorted = sorted(self.index.size_index.items(), key=lambda x: x[1])
        
        self.index_dirty = False
        build_time = time.time() - start_time
        
        print(f"Built model index for {len(models)} models in {build_time:.3f}s")
    
    def _index_model(self, model_key: str, model: Model) -> None:
        """Add a single model to the indexes."""
        # Index by expertise level
        model_expertise = getattr(model, "_expertise", ["general"])
        if model_expertise is None:
            model_expertise = ["general"]
        
        # Determine expertise level using same logic as registry
        level = "medium"  # Default
        if "analysis" in model_expertise or "research" in model_expertise:
            level = "very-high"
        elif "code" in model_expertise or "reasoning" in model_expertise:
            level = "high"
        elif "fast" in model_expertise or "compact" in model_expertise:
            level = "low"
        
        # Add to appropriate expertise indexes (hierarchical)
        if level == "low":
            self.index.expertise_low.add(model_key)
        elif level == "medium":
            self.index.expertise_low.add(model_key)
            self.index.expertise_medium.add(model_key)
        elif level == "high":
            self.index.expertise_low.add(model_key)
            self.index.expertise_medium.add(model_key)
            self.index.expertise_high.add(model_key)
        elif level == "very-high":
            self.index.expertise_low.add(model_key)
            self.index.expertise_medium.add(model_key)
            self.index.expertise_high.add(model_key)
            self.index.expertise_very_high.add(model_key)
        
        # Index by size
        size = getattr(model, "_size_billions", 1.0)
        self.index.size_index[model_key] = size
        
        # Index by cost
        if model.cost.is_free:
            self.index.free_models.add(model_key)
        else:
            avg_cost = (model.cost.input_cost_per_1k_tokens + model.cost.output_cost_per_1k_tokens) / 2
            self.index.paid_models[model_key] = avg_cost
        
        # Index by capabilities
        if model.capabilities.vision_capable:
            self.index.vision_capable.add(model_key)
        if model.capabilities.code_specialized:
            self.index.code_specialized.add(model_key)
        if model.capabilities.supports_function_calling:
            self.index.function_calling.add(model_key)
        
        # Index by provider
        self.index.provider_index[model.provider].add(model_key)
    
    def update_model_index(self, model_key: str, model: Model) -> None:
        """Update index for a single model (for incremental updates)."""
        # Remove from existing indexes first
        self._remove_from_index(model_key)
        
        # Add to indexes
        self._index_model(model_key, model)
        
        # Update sorted size index
        self.index.size_sorted = sorted(self.index.size_index.items(), key=lambda x: x[1])
    
    def _remove_from_index(self, model_key: str) -> None:
        """Remove a model from all indexes."""
        # Remove from expertise indexes
        self.index.expertise_low.discard(model_key)
        self.index.expertise_medium.discard(model_key)
        self.index.expertise_high.discard(model_key)
        self.index.expertise_very_high.discard(model_key)
        
        # Remove from size index
        self.index.size_index.pop(model_key, None)
        
        # Remove from cost indexes
        self.index.free_models.discard(model_key)
        self.index.paid_models.pop(model_key, None)
        
        # Remove from capability indexes
        self.index.vision_capable.discard(model_key)
        self.index.code_specialized.discard(model_key)
        self.index.function_calling.discard(model_key)
        
        # Remove from provider indexes
        for provider_models in self.index.provider_index.values():
            provider_models.discard(model_key)
    
    def fast_filter_by_criteria(self, criteria) -> Set[str]:
        """Fast filtering using pre-built indexes."""
        if self.index_dirty:
            raise ValueError("Index is dirty, rebuild required")
        
        candidates = set()
        
        # Start with expertise filtering (most selective)
        if criteria.expertise:
            if criteria.expertise == "low":
                candidates = self.index.expertise_low.copy()
            elif criteria.expertise == "medium":
                candidates = self.index.expertise_medium.copy()
            elif criteria.expertise == "high":
                candidates = self.index.expertise_high.copy()
            elif criteria.expertise == "very-high":
                candidates = self.index.expertise_very_high.copy()
            else:
                # Start with all models if invalid expertise
                candidates = set(self.index.size_index.keys())
        else:
            # Start with all models
            candidates = set(self.index.size_index.keys())
        
        # Filter by size constraints
        if criteria.min_model_size or criteria.max_model_size:
            size_candidates = set()
            min_size = criteria.min_model_size or 0.0
            max_size = criteria.max_model_size or float('inf')
            
            # Use binary search on sorted size index for efficiency
            for model_key in candidates:
                size = self.index.size_index.get(model_key, 1.0)
                if min_size <= size <= max_size:
                    size_candidates.add(model_key)
            
            candidates &= size_candidates
        
        # Filter by cost constraints
        if criteria.cost_limit is not None:
            if criteria.cost_limit == 0.0:
                # Only free models
                candidates &= self.index.free_models
            else:
                # Free models + paid models within budget
                cost_candidates = self.index.free_models.copy()
                
                # Estimate cost for paid models
                for model_key, avg_cost in self.index.paid_models.items():
                    if model_key in candidates:
                        # Simple cost estimation (could be made more sophisticated)
                        estimated_cost = avg_cost * 1.0  # Assume 1k tokens
                        if estimated_cost <= criteria.cost_limit:
                            cost_candidates.add(model_key)
                
                candidates &= cost_candidates
        
        # Filter by capabilities
        if "vision" in criteria.modalities:
            candidates &= self.index.vision_capable
        
        if "code" in criteria.modalities:
            candidates &= self.index.code_specialized
        
        if criteria.required_capabilities:
            if "tools" in criteria.required_capabilities or "function_calling" in criteria.required_capabilities:
                candidates &= self.index.function_calling
        
        # Filter by provider preferences
        if criteria.preferred_providers:
            provider_candidates = set()
            for provider in criteria.preferred_providers:
                provider_candidates.update(self.index.provider_index.get(provider, set()))
            candidates &= provider_candidates
        
        # Exclude providers
        if criteria.excluded_providers:
            for provider in criteria.excluded_providers:
                candidates -= self.index.provider_index.get(provider, set())
        
        return candidates
    
    def get_selection_cache_key(self, criteria) -> str:
        """Generate cache key for selection criteria."""
        # Create hash based on criteria that affect selection
        criteria_str = f"{criteria.expertise}_{criteria.min_model_size}_{criteria.max_model_size}_"
        criteria_str += f"{criteria.cost_limit}_{criteria.budget_period}_"
        criteria_str += f"{'_'.join(sorted(criteria.modalities))}_"
        criteria_str += f"{'_'.join(sorted(criteria.preferred_providers))}_"
        criteria_str += f"{'_'.join(sorted(criteria.excluded_providers))}_"
        criteria_str += f"{criteria.selection_strategy}"
        
        return hashlib.md5(criteria_str.encode()).hexdigest()
    
    def get_cached_selection(self, criteria) -> Optional[str]:
        """Get cached model selection if available and not expired."""
        cache_key = self.get_selection_cache_key(criteria)
        
        if cache_key in self.selection_cache:
            model_key, timestamp = self.selection_cache[cache_key]
            
            # Check if cache is still valid
            if time.time() - timestamp < self.cache_ttl:
                return model_key
            else:
                # Remove expired cache entry
                del self.selection_cache[cache_key]
        
        return None
    
    def cache_selection(self, criteria, model_key: str) -> None:
        """Cache the model selection result."""
        cache_key = self.get_selection_cache_key(criteria)
        self.selection_cache[cache_key] = (model_key, time.time())
        
        # Limit cache size to prevent memory bloat
        if len(self.selection_cache) > 1000:
            self._cleanup_selection_cache()
    
    def _cleanup_selection_cache(self) -> None:
        """Remove expired entries from selection cache."""
        current_time = time.time()
        expired_keys = []
        
        for cache_key, (model_key, timestamp) in self.selection_cache.items():
            if current_time - timestamp >= self.cache_ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.selection_cache[key]
    
    def get_cached_capability_analysis(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get cached capability analysis if available and not expired."""
        if model_key in self.capability_cache:
            analysis, timestamp = self.capability_cache[model_key]
            
            # Check if cache is still valid
            if time.time() - timestamp < self.cache_ttl:
                return analysis
            else:
                # Remove expired cache entry
                del self.capability_cache[model_key]
        
        return None
    
    def cache_capability_analysis(self, model_key: str, analysis: Dict[str, Any]) -> None:
        """Cache capability analysis result."""
        self.capability_cache[model_key] = (analysis, time.time())
        
        # Limit cache size
        if len(self.capability_cache) > 500:
            self._cleanup_capability_cache()
    
    def _cleanup_capability_cache(self) -> None:
        """Remove expired entries from capability cache."""
        current_time = time.time()
        expired_keys = []
        
        for model_key, (analysis, timestamp) in self.capability_cache.items():
            if current_time - timestamp >= self.cache_ttl:
                expired_keys.append(model_key)
        
        for key in expired_keys:
            del self.capability_cache[key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "index_statistics": {
                "total_models": len(self.index.size_index),
                "expertise_distribution": {
                    "low": len(self.index.expertise_low),
                    "medium": len(self.index.expertise_medium),
                    "high": len(self.index.expertise_high),
                    "very_high": len(self.index.expertise_very_high),
                },
                "cost_distribution": {
                    "free_models": len(self.index.free_models),
                    "paid_models": len(self.index.paid_models),
                },
                "capability_distribution": {
                    "vision_capable": len(self.index.vision_capable),
                    "code_specialized": len(self.index.code_specialized),
                    "function_calling": len(self.index.function_calling),
                },
                "provider_distribution": {
                    provider: len(models) for provider, models in self.index.provider_index.items()
                },
            },
            "cache_statistics": {
                "selection_cache_size": len(self.selection_cache),
                "capability_cache_size": len(self.capability_cache),
                "cache_ttl": self.cache_ttl,
            },
            "performance_flags": {
                "index_dirty": self.index_dirty,
            }
        }
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.selection_cache.clear()
        self.capability_cache.clear()
    
    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """Set cache time-to-live in seconds.""" 
        self.cache_ttl = ttl_seconds


class BatchModelProcessor:
    """Batch processor for model operations to improve performance."""
    
    @staticmethod
    async def batch_health_check(models: List[Model], batch_size: int = 10, timeout: float = 5.0) -> Dict[str, bool]:
        """Perform health checks on multiple models in batches."""
        results = {}
        
        # Process models in batches to avoid overwhelming the system
        for i in range(0, len(models), batch_size):
            batch = models[i:i + batch_size]
            
            # Create health check tasks for the batch
            tasks = []
            for model in batch:
                task = asyncio.create_task(
                    asyncio.wait_for(model.health_check(), timeout=timeout)
                )
                tasks.append((model, task))
            
            # Wait for batch to complete
            for model, task in tasks:
                try:
                    is_healthy = await task
                    results[f"{model.provider}:{model.name}"] = is_healthy
                except asyncio.TimeoutError:
                    results[f"{model.provider}:{model.name}"] = False
                except Exception:
                    results[f"{model.provider}:{model.name}"] = False
        
        return results
    
    @staticmethod
    def batch_capability_analysis(models: Dict[str, Model], registry_optimizer) -> Dict[str, Dict[str, Any]]:
        """Perform capability analysis on multiple models with caching."""
        results = {}
        
        # Check cache first
        cache_hits = 0
        for model_key, model in models.items():
            cached_analysis = registry_optimizer.get_cached_capability_analysis(model_key)
            if cached_analysis:
                results[model_key] = cached_analysis
                cache_hits += 1
        
        # Process remaining models
        remaining_models = {k: v for k, v in models.items() if k not in results}
        
        for model_key, model in remaining_models.items():
            # This would normally call the registry's detect_model_capabilities
            # For now, we'll create a placeholder
            analysis = {
                "basic_capabilities": {},
                "advanced_capabilities": {},
                "performance_metrics": {},
                "expertise_analysis": {"level": "medium"},
                "cost_analysis": {"type": "free" if model.cost.is_free else "paid"},
                "suitability_scores": {}
            }
            
            results[model_key] = analysis
            registry_optimizer.cache_capability_analysis(model_key, analysis)
        
        print(f"Batch capability analysis: {cache_hits} cache hits, {len(remaining_models)} computed")
        return results