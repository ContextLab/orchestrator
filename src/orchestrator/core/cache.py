"""Multi-level caching system for the Orchestrator framework."""

import asyncio
import time
import json
import hashlib
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
from collections import OrderedDict
import threading


class CacheLevel(Enum):
    """Cache level priorities."""
    MEMORY = 1
    DISK = 2
    DISTRIBUTED = 3


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    TTL = "ttl"          # Time To Live
    SIZE = "size"        # Size-based


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.size == 0:
            self.size = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate approximate size of cached value."""
        try:
            if isinstance(self.value, (str, bytes)):
                return len(self.value)
            elif isinstance(self.value, (int, float)):
                return 8
            elif isinstance(self.value, (list, tuple, dict)):
                return len(str(self.value))
            else:
                return len(pickle.dumps(self.value))
        except:
            return 1024  # Default size estimate
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access timestamp and count."""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get number of entries in cache."""
        pass
    
    @abstractmethod
    async def keys(self) -> List[str]:
        """Get all cache keys."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend with configurable eviction policies."""
    
    def __init__(self, 
                 max_size: int = 1000, 
                 max_memory: int = 100 * 1024 * 1024,  # 100MB
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                 default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.max_memory = max_memory
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        self._storage = OrderedDict()
        self._lock = threading.RLock()
        self._current_memory = 0
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from memory cache."""
        with self._lock:
            if key not in self._storage:
                return None
            
            entry = self._storage[key]
            
            # Check if expired
            if entry.is_expired():
                del self._storage[key]
                self._current_memory -= entry.size
                return None
            
            # Update access info
            entry.touch()
            
            # Move to end for LRU
            if self.eviction_policy == EvictionPolicy.LRU:
                self._storage.move_to_end(key)
            
            return entry
    
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set value in memory cache."""
        with self._lock:
            # Set default TTL if not specified
            if entry.ttl is None and self.default_ttl:
                entry.ttl = self.default_ttl
            
            # Remove existing entry
            if key in self._storage:
                old_entry = self._storage[key]
                self._current_memory -= old_entry.size
                del self._storage[key]
            
            # Check memory limit
            if self._current_memory + entry.size > self.max_memory:
                await self._evict_by_memory(entry.size)
            
            # Check size limit
            if len(self._storage) >= self.max_size:
                await self._evict_by_count(1)
            
            # Add new entry
            self._storage[key] = entry
            self._current_memory += entry.size
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        with self._lock:
            if key in self._storage:
                entry = self._storage[key]
                self._current_memory -= entry.size
                del self._storage[key]
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            self._storage.clear()
            self._current_memory = 0
            return True
    
    async def size(self) -> int:
        """Get number of entries in cache."""
        return len(self._storage)
    
    async def keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self._storage.keys())
    
    async def _evict_by_memory(self, needed_size: int):
        """Evict entries to free up memory."""
        if self.eviction_policy == EvictionPolicy.LRU:
            while self._current_memory + needed_size > self.max_memory and self._storage:
                key, entry = self._storage.popitem(last=False)
                self._current_memory -= entry.size
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Sort by access count and remove least frequently used
            while self._current_memory + needed_size > self.max_memory and self._storage:
                lfu_key = min(self._storage.keys(), 
                            key=lambda k: self._storage[k].access_count)
                entry = self._storage[lfu_key]
                self._current_memory -= entry.size
                del self._storage[lfu_key]
        
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove expired entries first
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._storage.items()
                if entry.ttl and current_time - entry.created_at > entry.ttl
            ]
            for key in expired_keys:
                entry = self._storage[key]
                self._current_memory -= entry.size
                del self._storage[key]
    
    async def _evict_by_count(self, count: int):
        """Evict specified number of entries."""
        for _ in range(min(count, len(self._storage))):
            if self.eviction_policy == EvictionPolicy.LRU:
                key, entry = self._storage.popitem(last=False)
                self._current_memory -= entry.size
            elif self.eviction_policy == EvictionPolicy.LFU:
                lfu_key = min(self._storage.keys(), 
                            key=lambda k: self._storage[k].access_count)
                entry = self._storage[lfu_key]
                self._current_memory -= entry.size
                del self._storage[lfu_key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._storage),
            "memory_used": self._current_memory,
            "memory_limit": self.max_memory,
            "size_limit": self.max_size,
            "eviction_policy": self.eviction_policy.value,
            "memory_utilization": self._current_memory / self.max_memory if self.max_memory > 0 else 0,
            "size_utilization": len(self._storage) / self.max_size if self.max_size > 0 else 0
        }


class DiskCache(CacheBackend):
    """Disk-based cache backend."""
    
    def __init__(self, cache_dir: str = "/tmp/orchestrator_cache", max_size: int = 10000):
        import os
        self.cache_dir = cache_dir
        self.max_size = max_size
        self._index = {}
        self._lock = threading.RLock()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing index
        self._load_index()
    
    def _get_file_path(self, key: str) -> str:
        """Get file path for cache key."""
        import os
        # Create safe filename from key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_key}.cache")
    
    def _load_index(self):
        """Load cache index from disk."""
        import os
        index_file = os.path.join(self.cache_dir, "index.json")
        try:
            if os.path.exists(index_file):
                with open(index_file, 'r') as f:
                    self._index = json.load(f)
        except:
            self._index = {}
    
    def _save_index(self):
        """Save cache index to disk."""
        import os
        index_file = os.path.join(self.cache_dir, "index.json")
        try:
            with open(index_file, 'w') as f:
                json.dump(self._index, f)
        except:
            pass
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from disk cache."""
        with self._lock:
            if key not in self._index:
                return None
            
            file_path = self._get_file_path(key)
            try:
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f)
                
                # Check if expired
                if entry.is_expired():
                    await self.delete(key)
                    return None
                
                # Update access info
                entry.touch()
                
                # Update index
                self._index[key] = {
                    "accessed_at": entry.accessed_at,
                    "access_count": entry.access_count
                }
                
                return entry
                
            except:
                # File corruption or missing, remove from index
                if key in self._index:
                    del self._index[key]
                    self._save_index()
                return None
    
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set value in disk cache."""
        with self._lock:
            # Check size limit
            if len(self._index) >= self.max_size:
                await self._evict_oldest()
            
            file_path = self._get_file_path(key)
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(entry, f)
                
                # Update index
                self._index[key] = {
                    "created_at": entry.created_at,
                    "accessed_at": entry.accessed_at,
                    "access_count": entry.access_count,
                    "size": entry.size
                }
                
                self._save_index()
                return True
                
            except:
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from disk cache."""
        with self._lock:
            if key not in self._index:
                return False
            
            file_path = self._get_file_path(key)
            try:
                import os
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                del self._index[key]
                self._save_index()
                return True
                
            except:
                return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            import os
            import shutil
            
            try:
                # Remove all cache files
                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                    os.makedirs(self.cache_dir, exist_ok=True)
                
                self._index = {}
                self._save_index()
                return True
                
            except:
                return False
    
    async def size(self) -> int:
        """Get number of entries in cache."""
        return len(self._index)
    
    async def keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self._index.keys())
    
    async def _evict_oldest(self):
        """Evict oldest cache entry."""
        if not self._index:
            return
        
        # Find oldest entry
        oldest_key = min(
            self._index.keys(),
            key=lambda k: self._index[k].get("accessed_at", 0)
        )
        
        await self.delete(oldest_key)


class MultiLevelCache:
    """Multi-level cache with automatic promotion/demotion."""
    
    def __init__(self):
        self.levels = {}
        self.level_order = []
        self.hit_stats = {level: 0 for level in CacheLevel}
        self.miss_stats = 0
        self.promotion_threshold = 3  # Promote after 3 hits
        self._lock = asyncio.Lock()
    
    def add_level(self, level: CacheLevel, backend: CacheBackend):
        """Add a cache level."""
        self.levels[level] = backend
        if level not in self.level_order:
            self.level_order.append(level)
            self.level_order.sort(key=lambda x: x.value)
        self.hit_stats[level] = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache, checking levels in order."""
        async with self._lock:
            for level in self.level_order:
                if level not in self.levels:
                    continue
                
                backend = self.levels[level]
                entry = await backend.get(key)
                
                if entry is not None:
                    self.hit_stats[level] += 1
                    
                    # Promote to higher levels if accessed frequently
                    if entry.access_count >= self.promotion_threshold:
                        await self._promote_entry(key, entry, level)
                    
                    return entry.value
            
            # Cache miss
            self.miss_stats += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, metadata: Dict[str, Any] = None):
        """Set value in cache, starting from highest level."""
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
            metadata=metadata or {}
        )
        
        # Start from highest priority level
        for level in self.level_order:
            if level not in self.levels:
                continue
            
            backend = self.levels[level]
            success = await backend.set(key, entry)
            
            if success:
                return True
        
        return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels."""
        success = False
        for level in self.level_order:
            if level not in self.levels:
                continue
            
            backend = self.levels[level]
            if await backend.delete(key):
                success = True
        
        return success
    
    async def clear(self) -> bool:
        """Clear all cache levels."""
        success = True
        for level in self.level_order:
            if level not in self.levels:
                continue
            
            backend = self.levels[level]
            if not await backend.clear():
                success = False
        
        return success
    
    async def _promote_entry(self, key: str, entry: CacheEntry, current_level: CacheLevel):
        """Promote entry to higher cache levels."""
        current_index = self.level_order.index(current_level)
        
        # Promote to all higher levels
        for i in range(current_index):
            higher_level = self.level_order[i]
            if higher_level in self.levels:
                backend = self.levels[higher_level]
                await backend.set(key, entry)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(self.hit_stats.values())
        total_requests = total_hits + self.miss_stats
        
        stats = {
            "total_requests": total_requests,
            "total_hits": total_hits,
            "total_misses": self.miss_stats,
            "hit_rate": total_hits / total_requests if total_requests > 0 else 0,
            "miss_rate": self.miss_stats / total_requests if total_requests > 0 else 0,
            "level_hits": {level.name: count for level, count in self.hit_stats.items()},
            "level_statistics": {}
        }
        
        # Get individual level statistics
        for level, backend in self.levels.items():
            if hasattr(backend, 'get_statistics'):
                stats["level_statistics"][level.name] = backend.get_statistics()
        
        return stats
    
    def reset_statistics(self):
        """Reset cache statistics."""
        self.hit_stats = {level: 0 for level in CacheLevel}
        self.miss_stats = 0


# Convenience function for creating cache key from task parameters
def create_cache_key(task_id: str, parameters: Dict[str, Any]) -> str:
    """Create cache key from task ID and parameters."""
    # Create deterministic hash from parameters
    param_str = json.dumps(parameters, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    return f"{task_id}:{param_hash}"