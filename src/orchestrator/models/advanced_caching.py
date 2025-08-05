"""
Advanced caching strategies for model selection and registry operations.
Implements intelligent cache warming, prefetching, and advanced eviction policies.
"""

from __future__ import annotations

import asyncio
import time
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from enum import Enum
import hashlib
import json
import heapq

from ..core.model import Model


class CacheEvictionPolicy(Enum):
    """Cache eviction policy types."""
    LRU = "least_recently_used"
    LFU = "least_frequently_used" 
    FIFO = "first_in_first_out"
    ADAPTIVE = "adaptive_replacement"
    TTL_BASED = "time_to_live_based"


@dataclass
class CacheEntry:
    """Cache entry with metadata for advanced eviction policies."""
    
    key: str
    value: Any
    access_count: int = 0
    creation_time: float = field(default_factory=time.time)
    last_access_time: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    size_bytes: int = 0
    priority: int = 0  # Higher priority = less likely to be evicted
    
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() - self.creation_time > self.ttl
    
    def access(self) -> None:
        """Record an access to this entry."""
        self.access_count += 1
        self.last_access_time = time.time()
    
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.creation_time
    
    def seconds_since_last_access(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_access_time


class AdvancedCache:
    """Advanced cache with intelligent eviction policies and preloading."""
    
    def __init__(
        self, 
        max_size: int = 1000,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.ADAPTIVE,
        default_ttl: Optional[float] = None,
        preload_factor: float = 0.8
    ):
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        self.preload_factor = preload_factor  # Start preloading when cache is 80% full
        
        self._entries: Dict[str, CacheEntry] = {}
        self._access_order: OrderedDict[str, float] = OrderedDict()  # For LRU
        self._frequency_heap: List[Tuple[int, str]] = []  # For LFU
        self._size_bytes: int = 0
        
        # Adaptive Replacement Cache (ARC) specific
        self._t1: OrderedDict[str, None] = OrderedDict()  # Recent cache misses
        self._t2: OrderedDict[str, None] = OrderedDict()  # Frequent items
        self._b1: OrderedDict[str, None] = OrderedDict()  # Ghost entries for T1
        self._b2: OrderedDict[str, None] = OrderedDict()  # Ghost entries for T2
        self._p: int = 0  # Target size for T1
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # Preloading
        self._preload_callbacks: List[Callable] = []
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._entries:
                entry = self._entries[key]
                
                # Check if expired
                if entry.is_expired():
                    self._remove_entry(key)
                    self._misses += 1
                    return None
                
                # Record access
                entry.access()
                self._record_access(key)
                self._hits += 1
                
                return entry.value
            else:
                self._misses += 1
                return None
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        priority: int = 0,
        size_bytes: Optional[int] = None
    ) -> None:
        """Put value in cache."""
        with self._lock:
            # Remove existing entry if present
            if key in self._entries:
                self._remove_entry(key)
            
            # Calculate size if not provided
            if size_bytes is None:
                size_bytes = self._estimate_size(value)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.default_ttl,
                priority=priority,
                size_bytes=size_bytes
            )
            
            # Make room if necessary
            while len(self._entries) >= self.max_size and self._entries:
                evicted_key = self._select_eviction_candidate()
                if evicted_key:
                    self._remove_entry(evicted_key)
                    self._evictions += 1
                else:
                    break
            
            # Add entry
            self._entries[key] = entry
            self._size_bytes += size_bytes
            self._record_access(key)
            
            # Check if we should trigger preloading
            if len(self._entries) >= self.max_size * self.preload_factor:
                self._trigger_preload()
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self._entries:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._entries.clear()
            self._access_order.clear()
            self._frequency_heap.clear()
            self._size_bytes = 0
            
            # Clear ARC structures
            self._t1.clear()
            self._t2.clear()
            self._b1.clear()
            self._b2.clear()
            self._p = 0
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._entries.keys())
    
    def size(self) -> int:
        """Get number of entries in cache."""
        return len(self._entries)
    
    def size_bytes(self) -> int:
        """Get total size of cache in bytes."""
        return self._size_bytes
    
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'entries': len(self._entries),
            'max_size': self.max_size,
            'size_bytes': self._size_bytes,
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'hit_rate': self.hit_rate(),
            'eviction_policy': self.eviction_policy.value,
            'preload_threshold': int(self.max_size * self.preload_factor)
        }
    
    def _record_access(self, key: str) -> None:
        """Record access for eviction policy tracking."""
        current_time = time.time()
        
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            # Move to end for LRU
            self._access_order[key] = current_time
            self._access_order.move_to_end(key)
        
        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            # Update frequency heap for LFU
            entry = self._entries[key]
            heapq.heappush(self._frequency_heap, (entry.access_count, key))
        
        elif self.eviction_policy == CacheEvictionPolicy.ADAPTIVE:
            # ARC algorithm
            if key in self._t1:
                # Move from T1 to T2 (recent to frequent)
                del self._t1[key] 
                self._t2[key] = None
                self._t2.move_to_end(key)
            elif key in self._t2:
                # Move to end of T2
                self._t2.move_to_end(key)
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select entry for eviction based on policy."""
        if not self._entries:
            return None
        
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            return self._select_lru_candidate()
        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            return self._select_lfu_candidate()
        elif self.eviction_policy == CacheEvictionPolicy.FIFO:
            return self._select_fifo_candidate()
        elif self.eviction_policy == CacheEvictionPolicy.ADAPTIVE:
            return self._select_arc_candidate()
        elif self.eviction_policy == CacheEvictionPolicy.TTL_BASED:
            return self._select_ttl_candidate()
        else:
            # Default to LRU
            return self._select_lru_candidate()
    
    def _select_lru_candidate(self) -> Optional[str]:
        """Select least recently used entry."""
        if self._access_order:
            return next(iter(self._access_order))
        elif self._entries:
            # Fallback to oldest entry
            return min(self._entries.keys(), 
                      key=lambda k: self._entries[k].last_access_time)
        return None
    
    def _select_lfu_candidate(self) -> Optional[str]:
        """Select least frequently used entry."""
        # Clean up frequency heap
        while self._frequency_heap:
            count, key = heapq.heappop(self._frequency_heap)
            if key in self._entries and self._entries[key].access_count == count:
                return key
        
        # Fallback to entry with lowest access count
        if self._entries:
            return min(self._entries.keys(),
                      key=lambda k: self._entries[k].access_count)
        return None
    
    def _select_fifo_candidate(self) -> Optional[str]:
        """Select first in, first out entry."""
        if self._entries:
            return min(self._entries.keys(),
                      key=lambda k: self._entries[k].creation_time)
        return None
    
    def _select_arc_candidate(self) -> Optional[str]:
        """Select candidate using Adaptive Replacement Cache algorithm."""
        # Simplified ARC implementation
        if self._t1 and (len(self._t1) > self._p or (not self._t2)):
            # Evict from T1
            key = next(iter(self._t1))
            return key
        elif self._t2:
            # Evict from T2
            key = next(iter(self._t2))
            return key
        else:
            # Fallback to LRU
            return self._select_lru_candidate()
    
    def _select_ttl_candidate(self) -> Optional[str]:
        """Select entry based on TTL (expired first, then oldest)."""
        expired_entries = []
        oldest_entry = None
        oldest_time = float('inf')
        
        for key, entry in self._entries.items():
            if entry.is_expired():
                expired_entries.append((entry.creation_time, key))
            elif entry.creation_time < oldest_time:
                oldest_time = entry.creation_time
                oldest_entry = key
        
        if expired_entries:
            # Return oldest expired entry
            expired_entries.sort()
            return expired_entries[0][1]
        else:
            # Return oldest non-expired entry
            return oldest_entry
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update tracking structures."""
        if key not in self._entries:
            return
        
        entry = self._entries[key]
        self._size_bytes -= entry.size_bytes
        del self._entries[key]
        
        # Update tracking structures
        self._access_order.pop(key, None)
        
        # Clean up frequency heap (lazy removal)
        # Clean up ARC structures
        self._t1.pop(key, None)
        self._t2.pop(key, None)
        self._b1.pop(key, None)
        self._b2.pop(key, None)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(str(value).encode('utf-8'))
        except:
            return 100  # Default estimate
    
    def _trigger_preload(self) -> None:
        """Trigger preload callbacks when cache is getting full."""
        for callback in self._preload_callbacks:
            try:
                callback(self)
            except Exception:
                pass  # Don't let callback failures break cache operation
    
    def add_preload_callback(self, callback: Callable) -> None:
        """Add callback to be triggered when preloading should occur."""
        self._preload_callbacks.append(callback)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        expired_keys = []
        
        with self._lock:
            for key, entry in self._entries.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
        
        return len(expired_keys)


class ModelSelectionCache:
    """Specialized cache for model selection results."""
    
    def __init__(self, max_size: int = 500, default_ttl: float = 300.0):
        self.cache = AdvancedCache(
            max_size=max_size,
            eviction_policy=CacheEvictionPolicy.LRU,
            default_ttl=default_ttl
        )
        self._pattern_cache: Dict[str, List[str]] = {}  # Cache for common patterns
        
    def get_selection(self, criteria_hash: str) -> Optional[str]:
        """Get cached model selection."""
        return self.cache.get(criteria_hash)
    
    def cache_selection(self, criteria_hash: str, model_key: str, ttl: Optional[float] = None) -> None:
        """Cache model selection result."""
        self.cache.put(criteria_hash, model_key, ttl=ttl, priority=1)
    
    def cache_pattern(self, pattern: str, model_keys: List[str]) -> None:
        """Cache common selection patterns."""
        self._pattern_cache[pattern] = model_keys
    
    def get_pattern_suggestions(self, pattern: str) -> Optional[List[str]]:
        """Get suggestions based on cached patterns."""
        return self._pattern_cache.get(pattern)
    
    def invalidate_model(self, model_key: str) -> int:
        """Invalidate all cache entries containing a specific model."""
        invalidated = 0
        
        # Remove from main cache
        keys_to_remove = []
        for key in self.cache.keys():
            cached_model = self.cache.get(key)
            if cached_model == model_key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.cache.remove(key)
            invalidated += 1
        
        # Remove from pattern cache
        patterns_to_remove = []
        for pattern, models in self._pattern_cache.items():
            if model_key in models:
                patterns_to_remove.append(pattern)
        
        for pattern in patterns_to_remove:
            del self._pattern_cache[pattern]
            invalidated += 1
        
        return invalidated
    
    def statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache.statistics()
        stats['pattern_cache_size'] = len(self._pattern_cache)
        return stats


class CapabilityAnalysisCache:
    """Specialized cache for model capability analysis results."""
    
    def __init__(self, max_size: int = 200, default_ttl: float = 3600.0):
        self.cache = AdvancedCache(
            max_size=max_size,
            eviction_policy=CacheEvictionPolicy.TTL_BASED,
            default_ttl=default_ttl
        )
    
    def get_analysis(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get cached capability analysis."""
        return self.cache.get(model_key)
    
    def cache_analysis(self, model_key: str, analysis: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Cache capability analysis result."""
        # Capability analysis is expensive, so give it high priority
        self.cache.put(model_key, analysis, ttl=ttl, priority=2)
    
    def invalidate_model(self, model_key: str) -> bool:
        """Invalidate cached analysis for a model."""
        return self.cache.remove(model_key)
    
    def statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.statistics()


class PredictiveCacheWarmer:
    """Predictive cache warming based on usage patterns."""
    
    def __init__(self, selection_cache: ModelSelectionCache, capability_cache: CapabilityAnalysisCache):
        self.selection_cache = selection_cache
        self.capability_cache = capability_cache
        
        # Usage pattern tracking
        self._usage_patterns: Dict[str, int] = defaultdict(int)
        self._time_patterns: Dict[int, List[str]] = defaultdict(list)  # hour -> criteria hashes
        self._sequence_patterns: List[Tuple[str, str]] = []  # (prev, next) pairs
        
        self._lock = threading.RLock()
    
    def record_selection(self, criteria_hash: str, model_key: str) -> None:
        """Record a model selection for pattern learning."""
        with self._lock:
            # Track overall usage patterns
            self._usage_patterns[criteria_hash] += 1
            
            # Track time-based patterns
            current_hour = int(time.time() // 3600) % 24
            self._time_patterns[current_hour].append(criteria_hash)
            
            # Track sequence patterns (simple bigram)
            if len(self._sequence_patterns) > 0:
                prev_criteria = self._sequence_patterns[-1][1]
                self._sequence_patterns.append((prev_criteria, criteria_hash))
            
            self._sequence_patterns.append((criteria_hash, model_key))
            
            # Limit sequence pattern history
            if len(self._sequence_patterns) > 1000:
                self._sequence_patterns = self._sequence_patterns[-500:]
    
    def warm_cache_predictively(self, registry, selector) -> Dict[str, int]:
        """Warm cache based on predicted usage patterns."""
        warmed = {'selections': 0, 'capabilities': 0}
        
        with self._lock:
            # Get most common patterns
            common_patterns = sorted(self._usage_patterns.items(), 
                                   key=lambda x: x[1], reverse=True)[:20]
            
            # Warm selection cache for common patterns
            for criteria_hash, frequency in common_patterns:
                if not self.selection_cache.get_selection(criteria_hash):
                    # Try to predict selection (simplified)
                    predicted_model = self._predict_model_for_criteria(criteria_hash, registry)
                    if predicted_model:
                        self.selection_cache.cache_selection(criteria_hash, predicted_model, ttl=1800)
                        warmed['selections'] += 1
            
            # Warm capability cache for models likely to be analyzed
            for model_key in self._get_likely_analyzed_models(registry):
                if not self.capability_cache.get_analysis(model_key):
                    # Pre-compute capability analysis
                    try:
                        model = registry.get_model("", model_key.split(":")[-1])
                        analysis = registry.detect_model_capabilities(model)
                        self.capability_cache.cache_analysis(model_key, analysis, ttl=3600)
                        warmed['capabilities'] += 1
                    except Exception:
                        pass
        
        return warmed
    
    def _predict_model_for_criteria(self, criteria_hash: str, registry) -> Optional[str]:
        """Predict which model would be selected for given criteria (simplified)."""
        # This is a simplified prediction - in practice, you'd want more sophisticated ML
        
        # Look for sequence patterns
        for prev, next_item in self._sequence_patterns:
            if prev == criteria_hash and ":" in next_item:  # next_item is a model
                return next_item
        
        # Fallback to most common model
        if registry.models:
            return list(registry.models.keys())[0]
        
        return None
    
    def _get_likely_analyzed_models(self, registry) -> List[str]:
        """Get models likely to need capability analysis."""
        # Return models that have been selected recently
        model_keys = set()
        
        # Extract models from recent sequence patterns
        for prev, next_item in self._sequence_patterns[-50:]:
            if ":" in next_item:  # It's a model key
                model_keys.add(next_item)
        
        return list(model_keys)[:10]  # Limit to 10 models
    
    def get_predictions(self) -> Dict[str, Any]:
        """Get current predictions and patterns."""
        with self._lock:
            current_hour = int(time.time() // 3600) % 24
            
            return {
                'most_common_patterns': dict(sorted(self._usage_patterns.items(), 
                                                  key=lambda x: x[1], reverse=True)[:10]),
                'current_hour_patterns': self._time_patterns.get(current_hour, [])[-10:],
                'recent_sequences': self._sequence_patterns[-10:],
                'total_patterns_tracked': len(self._usage_patterns)
            }


class CacheManager:
    """Central manager for all caching strategies."""
    
    def __init__(self, registry=None):
        self.selection_cache = ModelSelectionCache(max_size=1000, default_ttl=300)
        self.capability_cache = CapabilityAnalysisCache(max_size=500, default_ttl=3600)
        self.cache_warmer = PredictiveCacheWarmer(self.selection_cache, self.capability_cache)
        
        self.registry = registry
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
        self._warming_enabled = True
    
    def get_cached_selection(self, criteria_hash: str) -> Optional[str]:
        """Get cached model selection."""
        return self.selection_cache.get_selection(criteria_hash)
    
    def cache_selection(self, criteria_hash: str, model_key: str, ttl: Optional[float] = None) -> None:
        """Cache model selection and record for pattern learning."""
        self.selection_cache.cache_selection(criteria_hash, model_key, ttl)
        if self._warming_enabled:
            self.cache_warmer.record_selection(criteria_hash, model_key)
    
    def get_cached_capability_analysis(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get cached capability analysis."""
        return self.capability_cache.get_analysis(model_key)
    
    def cache_capability_analysis(self, model_key: str, analysis: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Cache capability analysis."""
        self.capability_cache.cache_analysis(model_key, analysis, ttl)
    
    def invalidate_model(self, model_key: str) -> Dict[str, int]:
        """Invalidate all caches for a specific model."""
        return {
            'selection_entries': self.selection_cache.invalidate_model(model_key),
            'capability_entries': 1 if self.capability_cache.invalidate_model(model_key) else 0
        }
    
    def warm_caches(self) -> Dict[str, int]:
        """Warm caches predictively."""
        if not self._warming_enabled or not self.registry:
            return {'selections': 0, 'capabilities': 0}
        
        return self.cache_warmer.warm_cache_predictively(self.registry, None)
    
    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired cache entries."""
        current_time = time.time()
        
        if current_time - self._last_cleanup < self._cleanup_interval:
            return {'selection_expired': 0, 'capability_expired': 0}
        
        selection_expired = self.selection_cache.cache.cleanup_expired()
        capability_expired = self.capability_cache.cache.cleanup_expired()
        
        self._last_cleanup = current_time
        
        return {
            'selection_expired': selection_expired,
            'capability_expired': capability_expired
        }
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive caching statistics."""
        return {
            'selection_cache': self.selection_cache.statistics(),
            'capability_cache': self.capability_cache.statistics(),
            'predictions': self.cache_warmer.get_predictions(),
            'warming_enabled': self._warming_enabled,
            'last_cleanup': self._last_cleanup
        }
    
    def enable_warming(self) -> None:
        """Enable predictive cache warming."""
        self._warming_enabled = True
    
    def disable_warming(self) -> None:
        """Disable predictive cache warming."""
        self._warming_enabled = False
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.selection_cache.cache.clear()
        self.capability_cache.cache.clear()


async def background_cache_maintenance(cache_manager: CacheManager, interval: int = 300) -> None:
    """Background task for cache maintenance."""
    while True:
        try:
            # Clean up expired entries
            cleanup_stats = cache_manager.cleanup_expired()
            
            # Warm caches if enabled
            if cache_manager._warming_enabled:
                warming_stats = cache_manager.warm_caches()
            
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
        except Exception:
            # Don't let maintenance failures break the system
            await asyncio.sleep(interval)