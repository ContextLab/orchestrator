"""
Memory optimization utilities for model registry operations.
Implements memory-efficient data structures and garbage collection for large model registries.
"""

from __future__ import annotations

import gc
import sys
import weakref
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
import psutil
import time

from ..core.model import Model


@dataclass
class MemoryProfile:
    """Memory usage profile for monitoring."""
    
    # System memory metrics
    total_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    used_memory_mb: float = 0.0
    memory_percent: float = 0.0
    
    # Process-specific metrics
    process_memory_mb: float = 0.0
    process_memory_percent: float = 0.0
    
    # Registry-specific metrics
    models_count: int = 0
    cache_entries_count: int = 0
    index_memory_mb: float = 0.0
    
    # Garbage collection metrics
    gc_collections: Dict[int, int] = field(default_factory=dict)
    gc_collected: Dict[int, int] = field(default_factory=dict)
    
    timestamp: float = field(default_factory=time.time)
    
    @classmethod
    def current(cls) -> "MemoryProfile":
        """Get current memory profile."""
        # System memory
        memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        # GC stats
        gc_stats = gc.get_stats()
        collections = {i: stat['collections'] for i, stat in enumerate(gc_stats)}
        collected = {i: stat['collected'] for i, stat in enumerate(gc_stats)}
        
        return cls(
            total_memory_mb=memory.total / 1024 / 1024,
            available_memory_mb=memory.available / 1024 / 1024,
            used_memory_mb=memory.used / 1024 / 1024,
            memory_percent=memory.percent,
            process_memory_mb=process_memory.rss / 1024 / 1024,
            process_memory_percent=process.memory_percent(),
            gc_collections=collections,
            gc_collected=collected
        )


class MemoryEfficientModelStorage:
    """Memory-efficient storage for model objects with weak references and lazy loading."""
    
    def __init__(self, max_strong_refs: int = 100):
        self.max_strong_refs = max_strong_refs
        
        # Strong references for frequently used models
        self._strong_cache: OrderedDict[str, Model] = OrderedDict()
        
        # Weak references for all models
        self._weak_refs: Dict[str, weakref.ref] = {}
        
        # Model metadata (lightweight)
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Access tracking for LRU eviction
        self._access_times: Dict[str, float] = {}
        
        self._lock = threading.RLock()
    
    def store_model(self, model_key: str, model: Model) -> None:
        """Store a model with memory-efficient caching."""
        with self._lock:
            # Store metadata (lightweight)
            self._metadata[model_key] = {
                'name': model.name,
                'provider': model.provider,
                'size_billions': getattr(model, '_size_billions', 1.0),
                'expertise': getattr(model, '_expertise', ['general']),
                'cost_is_free': model.cost.is_free if model.cost else True,
                'capabilities_hash': self._hash_capabilities(model.capabilities),
                'last_accessed': time.time()
            }
            
            # Create weak reference
            def cleanup_callback(ref):
                with self._lock:
                    self._weak_refs.pop(model_key, None)
                    self._access_times.pop(model_key, None)
            
            self._weak_refs[model_key] = weakref.ref(model, cleanup_callback)
            
            # Add to strong cache if there's room
            if len(self._strong_cache) < self.max_strong_refs:
                self._strong_cache[model_key] = model
                self._access_times[model_key] = time.time()
            else:
                # Evict least recently used from strong cache
                self._evict_lru()
                self._strong_cache[model_key] = model
                self._access_times[model_key] = time.time()
    
    def get_model(self, model_key: str) -> Optional[Model]:
        """Get a model, loading from weak reference if needed."""
        with self._lock:
            # Update access time
            self._access_times[model_key] = time.time()
            
            # Try strong cache first
            if model_key in self._strong_cache:
                # Move to end (most recently used)
                self._strong_cache.move_to_end(model_key)
                return self._strong_cache[model_key]
            
            # Try weak reference
            weak_ref = self._weak_refs.get(model_key)
            if weak_ref:
                model = weak_ref()
                if model is not None:
                    # Promote to strong cache
                    if len(self._strong_cache) >= self.max_strong_refs:
                        self._evict_lru()
                    self._strong_cache[model_key] = model
                    return model
            
            return None
    
    def get_metadata(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get lightweight model metadata without loading full model."""
        return self._metadata.get(model_key)
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all stored models."""
        return self._metadata.copy()
    
    def remove_model(self, model_key: str) -> None:
        """Remove a model from storage."""
        with self._lock:
            self._strong_cache.pop(model_key, None)
            self._weak_refs.pop(model_key, None)
            self._metadata.pop(model_key, None)
            self._access_times.pop(model_key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used model from strong cache."""
        if not self._strong_cache:
            return
        
        # Find least recently used
        lru_key = min(self._access_times.keys(), 
                     key=lambda k: self._access_times.get(k, 0))
        
        # Remove from strong cache (weak ref remains)
        self._strong_cache.pop(lru_key, None)
    
    def _hash_capabilities(self, capabilities) -> str:
        """Create a simple hash of capabilities for metadata."""
        if not capabilities:
            return "none"
        
        # Create a simple hash based on key capabilities
        items = [
            str(capabilities.supports_function_calling),
            str(capabilities.vision_capable),
            str(capabilities.code_specialized),
            str(len(capabilities.supported_tasks)),
            str(capabilities.context_window)
        ]
        return "_".join(items)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self._lock:
            alive_weak_refs = sum(1 for ref in self._weak_refs.values() if ref() is not None)
            
            return {
                'total_models': len(self._metadata),
                'strong_cache_size': len(self._strong_cache),
                'weak_refs_count': len(self._weak_refs),
                'alive_weak_refs': alive_weak_refs,
                'metadata_entries': len(self._metadata),
                'max_strong_refs': self.max_strong_refs,
                'cache_hit_potential': len(self._strong_cache) / max(len(self._metadata), 1)
            }
    
    def cleanup_dead_references(self) -> int:
        """Clean up dead weak references and return count cleaned."""
        with self._lock:
            dead_keys = []
            for key, ref in self._weak_refs.items():
                if ref() is None:
                    dead_keys.append(key)
            
            for key in dead_keys:
                self._weak_refs.pop(key, None)
                self._access_times.pop(key, None)
                # Keep metadata for potential future reloading
            
            return len(dead_keys)
    
    def adjust_cache_size(self, new_max_strong_refs: int) -> None:
        """Adjust the maximum strong reference cache size."""
        with self._lock:
            self.max_strong_refs = new_max_strong_refs
            
            # Evict excess entries if needed
            while len(self._strong_cache) > new_max_strong_refs:
                self._evict_lru()


class MemoryMonitor:
    """Monitor and manage memory usage for model registry operations."""
    
    def __init__(self, warning_threshold: float = 80.0, critical_threshold: float = 90.0):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        self._profiles: List[MemoryProfile] = []
        self._max_profiles = 100  # Keep last 100 profiles
        
        self._callbacks: Dict[str, List[callable]] = {
            'warning': [],
            'critical': [],
            'normal': []
        }
        
        self._last_gc_time = 0
        self._gc_interval = 30  # seconds
    
    def add_callback(self, event_type: str, callback: callable) -> None:
        """Add callback for memory events."""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
    
    def check_memory(self) -> MemoryProfile:
        """Check current memory usage and trigger callbacks if needed."""
        profile = MemoryProfile.current()
        
        # Store profile
        self._profiles.append(profile)
        if len(self._profiles) > self._max_profiles:
            self._profiles.pop(0)
        
        # Check thresholds and trigger callbacks
        if profile.memory_percent >= self.critical_threshold:
            for callback in self._callbacks['critical']:
                try:
                    callback(profile)
                except Exception:
                    pass  # Don't let callback failures break monitoring
        elif profile.memory_percent >= self.warning_threshold:
            for callback in self._callbacks['warning']:
                try:
                    callback(profile)
                except Exception:
                    pass
        else:
            for callback in self._callbacks['normal']:
                try:
                    callback(profile)
                except Exception:
                    pass
        
        # Trigger garbage collection if needed
        current_time = time.time()
        if (current_time - self._last_gc_time > self._gc_interval and 
            profile.memory_percent > self.warning_threshold):
            self.force_garbage_collection()
            self._last_gc_time = current_time
        
        return profile
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return collection stats."""
        collected = {}
        
        # Collect each generation
        for generation in range(3):
            collected[generation] = gc.collect(generation)
        
        return collected
    
    def get_memory_trend(self, minutes: int = 5) -> Dict[str, Any]:
        """Get memory usage trend over specified minutes."""
        if not self._profiles:
            return {'trend': 'unknown', 'profiles': []}
        
        cutoff_time = time.time() - (minutes * 60)
        recent_profiles = [p for p in self._profiles if p.timestamp >= cutoff_time]
        
        if len(recent_profiles) < 2:
            return {'trend': 'insufficient_data', 'profiles': recent_profiles}
        
        # Calculate trend
        start_usage = recent_profiles[0].memory_percent
        end_usage = recent_profiles[-1].memory_percent
        
        if end_usage > start_usage + 5:
            trend = 'increasing'
        elif end_usage < start_usage - 5:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'start_usage': start_usage,
            'end_usage': end_usage,
            'change': end_usage - start_usage,
            'profiles_count': len(recent_profiles),
            'profiles': recent_profiles
        }
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest memory optimizations based on current usage."""
        if not self._profiles:
            return ["No memory data available"]
        
        current = self._profiles[-1]
        suggestions = []
        
        if current.memory_percent > self.critical_threshold:
            suggestions.extend([
                "Critical memory usage - consider reducing model cache size",
                "Force garbage collection immediately",
                "Consider unloading unused models",
                "Reduce batch processing sizes"
            ])
        elif current.memory_percent > self.warning_threshold:
            suggestions.extend([
                "High memory usage - monitor closely",
                "Consider adjusting cache TTL to be more aggressive",
                "Review model storage efficiency",
                "Schedule garbage collection more frequently"
            ])
        
        # Check GC effectiveness
        if len(self._profiles) >= 2:
            prev_gc = self._profiles[-2].gc_collections
            curr_gc = current.gc_collections
            
            total_collections = sum(curr_gc.values()) - sum(prev_gc.values())
            if total_collections > 10:
                suggestions.append("Frequent garbage collection detected - review object lifecycle")
        
        if not suggestions:
            suggestions.append("Memory usage is healthy")
        
        return suggestions


class MemoryOptimizedRegistry:
    """Memory-optimized model registry with efficient storage and monitoring."""
    
    def __init__(self, max_strong_refs: int = 100, memory_warning_threshold: float = 80.0):
        self.storage = MemoryEfficientModelStorage(max_strong_refs)
        self.monitor = MemoryMonitor(memory_warning_threshold)
        
        # Set up memory callbacks
        self.monitor.add_callback('warning', self._on_memory_warning)
        self.monitor.add_callback('critical', self._on_memory_critical)
        
        self._optimization_stats = {
            'cache_adjustments': 0,
            'gc_triggers': 0,
            'reference_cleanups': 0
        }
    
    def register_model(self, model: Model) -> None:
        """Register a model with memory optimization."""
        model_key = f"{model.provider}:{model.name}"
        self.storage.store_model(model_key, model)
        
        # Check memory after registration
        self.monitor.check_memory()
    
    def get_model(self, provider: str, name: str) -> Optional[Model]:
        """Get a model by provider and name."""
        model_key = f"{provider}:{name}"
        return self.storage.get_model(model_key)
    
    def get_model_metadata(self, provider: str, name: str) -> Optional[Dict[str, Any]]:
        """Get lightweight model metadata without loading full model."""
        model_key = f"{provider}:{name}"
        return self.storage.get_metadata(model_key)
    
    def list_models_metadata(self) -> Dict[str, Dict[str, Any]]:
        """List all models using lightweight metadata."""
        return self.storage.get_all_metadata()
    
    def unregister_model(self, provider: str, name: str) -> None:
        """Unregister a model and free memory."""
        model_key = f"{provider}:{name}"
        self.storage.remove_model(model_key)
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization and return stats."""
        # Clean up dead references
        dead_refs = self.storage.cleanup_dead_references()
        self._optimization_stats['reference_cleanups'] += dead_refs
        
        # Force garbage collection
        gc_stats = self.monitor.force_garbage_collection()
        self._optimization_stats['gc_triggers'] += 1
        
        # Check current memory
        profile = self.monitor.check_memory()
        
        return {
            'dead_references_cleaned': dead_refs,
            'gc_collected': gc_stats,
            'current_memory_profile': profile,
            'optimization_stats': self._optimization_stats.copy()
        }
    
    def _on_memory_warning(self, profile: MemoryProfile) -> None:
        """Handle memory warning threshold."""
        # Reduce cache size by 25%
        current_max = self.storage.max_strong_refs
        new_max = max(10, int(current_max * 0.75))
        self.storage.adjust_cache_size(new_max)
        self._optimization_stats['cache_adjustments'] += 1
        
        print(f"Memory warning: {profile.memory_percent:.1f}% - reduced cache to {new_max}")
    
    def _on_memory_critical(self, profile: MemoryProfile) -> None:
        """Handle critical memory threshold."""
        # Aggressive cache reduction
        new_max = max(5, int(self.storage.max_strong_refs * 0.5))
        self.storage.adjust_cache_size(new_max)
        self._optimization_stats['cache_adjustments'] += 1
        
        # Force cleanup
        self.storage.cleanup_dead_references()
        self.monitor.force_garbage_collection()
        self._optimization_stats['gc_triggers'] += 1
        
        print(f"Critical memory: {profile.memory_percent:.1f}% - aggressive cleanup")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        storage_stats = self.storage.get_memory_stats()
        memory_profile = self.monitor.check_memory()
        memory_trend = self.monitor.get_memory_trend()
        suggestions = self.monitor.suggest_optimizations()
        
        return {
            'storage_statistics': storage_stats,
            'current_memory_profile': memory_profile,
            'memory_trend': memory_trend,
            'optimization_suggestions': suggestions,
            'optimization_stats': self._optimization_stats.copy()
        }


def estimate_model_memory_usage(model: Model) -> float:
    """Estimate memory usage of a model object in MB."""
    try:
        # Basic object size
        base_size = sys.getsizeof(model)
        
        # Estimate capabilities size
        if hasattr(model, 'capabilities') and model.capabilities:
            base_size += sys.getsizeof(model.capabilities)
            if hasattr(model.capabilities, 'supported_tasks'):
                base_size += sys.getsizeof(model.capabilities.supported_tasks)
            if hasattr(model.capabilities, 'languages'):
                base_size += sys.getsizeof(model.capabilities.languages)
        
        # Estimate requirements size
        if hasattr(model, 'requirements') and model.requirements:
            base_size += sys.getsizeof(model.requirements)
        
        # Convert to MB
        return base_size / 1024 / 1024
        
    except Exception:
        # Fallback estimate
        return 0.1  # 100KB default estimate


def optimize_model_registry_memory(registry, target_memory_mb: float = 100.0) -> Dict[str, Any]:
    """Optimize memory usage of a model registry to target size."""
    if not hasattr(registry, 'models'):
        return {'error': 'Registry does not have models attribute'}
    
    # Calculate current usage
    current_usage = 0.0
    model_sizes = {}
    
    for model_key, model in registry.models.items():
        size = estimate_model_memory_usage(model)
        model_sizes[model_key] = size
        current_usage += size
    
    # If we're under target, nothing to do
    if current_usage <= target_memory_mb:
        return {
            'current_usage_mb': current_usage,
            'target_mb': target_memory_mb,
            'action': 'no_optimization_needed',
            'models_count': len(registry.models)
        }
    
    # Sort models by size (largest first) for potential removal
    sorted_models = sorted(model_sizes.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate how much we need to free
    excess_mb = current_usage - target_memory_mb
    
    # Identify models to potentially cache out or remove
    candidates_for_removal = []
    freed_mb = 0.0
    
    for model_key, size in sorted_models:
        if freed_mb >= excess_mb:
            break
        candidates_for_removal.append((model_key, size))
        freed_mb += size
    
    return {
        'current_usage_mb': current_usage,
        'target_mb': target_memory_mb,
        'excess_mb': excess_mb,
        'candidates_for_removal': candidates_for_removal,
        'potential_freed_mb': freed_mb,
        'models_count': len(registry.models),
        'action': 'optimization_candidates_identified'
    }