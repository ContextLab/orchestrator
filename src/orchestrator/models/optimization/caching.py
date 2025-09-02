"""Model response caching system for performance optimization."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..selection.strategies import SelectionResult

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for model response cache."""
    
    max_size: int = 1000
    ttl_seconds: float = 3600.0
    enable_compression: bool = True
    cleanup_interval_seconds: float = 300.0


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hits + self.misses
        if total_requests == 0:
            return 0.0
        return self.hits / total_requests
    
    @property
    def fill_rate(self) -> float:
        """Calculate cache fill rate."""
        if self.max_size == 0:
            return 0.0
        return self.total_size / self.max_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "total_size": self.total_size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate,
            "fill_rate": self.fill_rate,
        }


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl: Optional[float] = None  # Time to live in seconds
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.timestamp + self.ttl)
    
    def access(self) -> None:
        """Record an access to this entry."""
        self.access_count += 1
        self.last_accessed = time.time()


class ModelResponseCache:
    """
    Intelligent caching system for model responses.
    
    Features:
    - LRU eviction policy
    - TTL support
    - Size-based eviction
    - Cache statistics
    - Smart cache key generation
    - Selection result caching
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = 3600.0,  # 1 hour
        max_memory_mb: Optional[int] = 100,  # 100MB max
    ):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (None for no expiration)
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        
        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._selection_cache: OrderedDict[str, SelectionResult] = OrderedDict()
        
        # Statistics
        self._stats = CacheStats(max_size=max_size)
        
        # Memory tracking
        self._current_memory_bytes = 0
        
        logger.info(f"ModelResponseCache initialized (max_size={max_size}, ttl={default_ttl}s)")
    
    def generate_cache_key(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate cache key from request parameters.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        # Create deterministic key from parameters
        key_data = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        
        # Sort keys for consistency
        sorted_data = json.dumps(key_data, sort_keys=True, default=str)
        
        # Generate hash
        cache_key = hashlib.sha256(sorted_data.encode()).hexdigest()[:16]
        return cache_key
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Clean expired entries first
        await self._cleanup_expired()
        
        if key not in self._cache:
            self._stats.misses += 1
            return None
        
        entry = self._cache[key]
        
        # Check expiration
        if entry.is_expired():
            del self._cache[key]
            self._current_memory_bytes -= entry.size_bytes
            self._stats.misses += 1
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.access()
        
        self._stats.hits += 1
        logger.debug(f"Cache hit for key {key[:8]}...")
        return entry.value
    
    async def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl
        
        # Estimate size
        size_bytes = self._estimate_size(value)
        
        # Check memory limits
        if self.max_memory_bytes and (self._current_memory_bytes + size_bytes) > self.max_memory_bytes:
            await self._evict_to_fit(size_bytes)
        
        # Create entry
        entry = CacheEntry(
            value=value,
            timestamp=time.time(),
            size_bytes=size_bytes,
            ttl=ttl,
        )
        
        # If key already exists, update memory tracking
        if key in self._cache:
            old_entry = self._cache[key]
            self._current_memory_bytes -= old_entry.size_bytes
        
        # Store entry
        self._cache[key] = entry
        self._current_memory_bytes += size_bytes
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        
        # Evict if over size limit
        while len(self._cache) > self.max_size:
            await self._evict_lru()
        
        self._stats.total_size = len(self._cache)
        logger.debug(f"Cached value for key {key[:8]}... (size: {size_bytes} bytes)")
    
    async def get_cached_selection(self, requirements_key: str) -> Optional[SelectionResult]:
        """
        Get cached model selection result.
        
        Args:
            requirements_key: Requirements hash key
            
        Returns:
            Cached selection result or None
        """
        if requirements_key in self._selection_cache:
            # Move to end (most recently used)
            self._selection_cache.move_to_end(requirements_key)
            selection_result = self._selection_cache[requirements_key]
            logger.debug(f"Cache hit for selection {requirements_key[:8]}...")
            return selection_result
        return None
    
    async def cache_selection(self, requirements_key: str, selection_result: SelectionResult) -> None:
        """
        Cache model selection result.
        
        Args:
            requirements_key: Requirements hash key
            selection_result: Selection result to cache
        """
        # Store in selection cache (separate from response cache)
        self._selection_cache[requirements_key] = selection_result
        self._selection_cache.move_to_end(requirements_key)
        
        # Keep selection cache smaller
        max_selection_cache = max(100, self.max_size // 10)
        while len(self._selection_cache) > max_selection_cache:
            self._selection_cache.popitem(last=False)
        
        logger.debug(f"Cached selection for key {requirements_key[:8]}...")
    
    async def invalidate(self, pattern: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            pattern: Pattern to match keys (None = invalidate all)
            
        Returns:
            Number of entries invalidated
        """
        if pattern is None:
            # Clear all
            count = len(self._cache)
            self._cache.clear()
            self._selection_cache.clear()
            self._current_memory_bytes = 0
            self._stats.total_size = 0
            return count
        
        # Pattern-based invalidation
        keys_to_remove = []
        for key in self._cache:
            if pattern in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            entry = self._cache[key]
            self._current_memory_bytes -= entry.size_bytes
            del self._cache[key]
        
        self._stats.total_size = len(self._cache)
        return len(keys_to_remove)
    
    async def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        self._stats.total_size = len(self._cache)
        return self._stats
    
    async def cleanup(self) -> None:
        """Clean up cache resources."""
        await self.invalidate()  # Clear all entries
        logger.info("Cache cleanup completed")
    
    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            entry = self._cache[key]
            self._current_memory_bytes -= entry.size_bytes
            del self._cache[key]
        
        if expired_keys:
            self._stats.total_size = len(self._cache)
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        key, entry = self._cache.popitem(last=False)  # Remove oldest
        self._current_memory_bytes -= entry.size_bytes
        self._stats.evictions += 1
        logger.debug(f"Evicted LRU entry {key[:8]}...")
    
    async def _evict_to_fit(self, needed_bytes: int) -> None:
        """Evict entries to fit new data."""
        while (
            self._cache and 
            self.max_memory_bytes and 
            (self._current_memory_bytes + needed_bytes) > self.max_memory_bytes
        ):
            await self._evict_lru()
    
    def _estimate_size(self, value: Any) -> int:
        """
        Estimate size of value in bytes.
        
        Args:
            value: Value to estimate
            
        Returns:
            Estimated size in bytes
        """
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8  # Rough estimate
            elif isinstance(value, dict):
                # Estimate dictionary size
                return len(json.dumps(value, default=str).encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                # Estimate list/tuple size
                return sum(self._estimate_size(item) for item in value)
            else:
                # Fallback: use string representation
                return len(str(value).encode('utf-8'))
        except Exception:
            # If estimation fails, use conservative estimate
            return 1024  # 1KB default
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        return {
            "max_size": self.max_size,
            "current_size": len(self._cache),
            "max_memory_mb": self.max_memory_bytes // (1024 * 1024) if self.max_memory_bytes else None,
            "current_memory_mb": self._current_memory_bytes / (1024 * 1024),
            "default_ttl": self.default_ttl,
            "selection_cache_size": len(self._selection_cache),
            "stats": self._stats.to_dict(),
        }
    
    def __str__(self) -> str:
        """String representation of cache."""
        return f"ModelResponseCache(size={len(self._cache)}/{self.max_size}, hit_rate={self._stats.hit_rate:.2f})"


# Alias for backward compatibility
ModelCache = ModelResponseCache