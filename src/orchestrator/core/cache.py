"""Multi-level caching system for the Orchestrator framework."""

import asyncio
import hashlib
import json
import pickle
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class CacheLevel(Enum):
    """Cache level priorities."""

    MEMORY = 1
    DISK = 2
    DISTRIBUTED = 3


class EvictionPolicy(Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based


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

    @property
    def last_accessed(self) -> float:
        """Alias for accessed_at for compatibility."""
        return self.accessed_at

    @last_accessed.setter
    def last_accessed(self, value: float):
        """Setter for last_accessed."""
        self.accessed_at = value

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
        except Exception:
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


@dataclass
class CacheStats:
    """Cache statistics and metrics."""

    hits: int = 0
    misses: int = 0
    entries: int = 0
    current_memory: int = 0
    max_memory: Optional[int] = None
    max_entries: Optional[int] = None

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hits + self.misses
        if total_requests == 0:
            return 0.0
        return self.hits / total_requests

    @property
    def memory_utilization(self) -> float:
        """Calculate memory utilization as percentage."""
        if not self.max_memory or self.max_memory == 0:
            return 0.0
        return min(self.current_memory / self.max_memory, 1.0)

    @property
    def entry_utilization(self) -> float:
        """Calculate entry utilization as percentage."""
        if not self.max_entries or self.max_entries == 0:
            return 0.0
        return min(self.entries / self.max_entries, 1.0)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
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

    def __init__(
        self,
        max_size: int = 1000,
        max_memory: int = 100 * 1024 * 1024,  # 100MB
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        default_ttl: Optional[float] = None,
        max_entries: Optional[int] = None,
    ):
        self.max_size = max_size
        self.max_memory = max_memory
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        self.max_entries = max_entries or max_size
        self._storage = OrderedDict()
        self._lock = threading.RLock()
        self._current_memory = 0

    @property
    def maxsize(self) -> int:
        """Alias for max_size for compatibility."""
        return self.max_size

    # Removed sync set method - use async version

    def get_sync(self, key: str) -> Optional[Any]:
        """Synchronous wrapper for get method."""
        import asyncio

        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # We're in an async context, can't use run_until_complete
            # This should not be called from async context
            raise RuntimeError(
                "get_sync() cannot be called from an async context. Use get() instead."
            )
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            entry = asyncio.run(self.get(key))
            return entry.value if entry else None

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

    async def set_entry(self, entry: CacheEntry) -> bool:
        """Set value in memory cache."""
        key = entry.key
        with self._lock:
            # Handle zero limits - reject storage
            if self.max_memory == 0 or self.max_entries == 0:
                return False

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

            # Check size limit (use max_entries if set, otherwise max_size)
            max_entries_limit = self.max_entries
            if len(self._storage) >= max_entries_limit:
                await self._evict_by_count(1)

            # Add new entry
            self._storage[key] = entry
            self._current_memory += entry.size

            return True

    async def async_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Async set method for CacheBackend compatibility."""
        entry = CacheEntry(key=key, value=value, ttl=ttl)
        return await self.set_entry(entry)

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
            while (
                self._current_memory + needed_size > self.max_memory and self._storage
            ):
                key, entry = self._storage.popitem(last=False)
                self._current_memory -= entry.size

        elif self.eviction_policy == EvictionPolicy.LFU:
            # Sort by access count and remove least frequently used
            while (
                self._current_memory + needed_size > self.max_memory and self._storage
            ):
                lfu_key = min(
                    self._storage.keys(), key=lambda k: self._storage[k].access_count
                )
                entry = self._storage[lfu_key]
                self._current_memory -= entry.size
                del self._storage[lfu_key]

        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove expired entries first
            current_time = time.time()
            expired_keys = [
                key
                for key, entry in self._storage.items()
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
                lfu_key = min(
                    self._storage.keys(), key=lambda k: self._storage[k].access_count
                )
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
            "memory_utilization": (
                self._current_memory / self.max_memory if self.max_memory > 0 else 0
            ),
            "size_utilization": (
                len(self._storage) / self.max_size if self.max_size > 0 else 0
            ),
        }

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Async set method - primary interface."""
        return await self.async_set(key, value, ttl)

    def set_sync(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Synchronous set method for cache."""
        import asyncio

        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # We're in an async context, can't use run_until_complete
            # This should not be called from async context
            raise RuntimeError(
                "set_sync() cannot be called from an async context. Use set() or async_set() instead."
            )
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            asyncio.run(self.async_set(key, value, ttl))

    async def get_entry(self, key: str) -> Optional[CacheEntry]:
        """Async get method - returns CacheEntry for MultiLevelCache compatibility."""
        return await self.async_get(key)

    async def get_value(self, key: str) -> Optional[Any]:
        """Async get method that returns value directly."""
        entry = await self.async_get(key)
        return entry.value if entry else None

    def get_value_sync(self, key: str) -> Optional[Any]:
        """Synchronous get method that returns the value directly."""
        import asyncio

        try:
            # Try to get the current event loop
            asyncio.get_running_loop()
            # We're in an async context, can't use run_until_complete
            # This should not be called from async context
            raise RuntimeError(
                "get_value_sync() cannot be called from an async context. Use get_value() instead."
            )
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            entry = asyncio.run(self.async_get(key))
            return entry.value if entry else None

    async def async_get(self, key: str) -> Optional[CacheEntry]:
        """Async get method that returns CacheEntry."""
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

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        import fnmatch

        with self._lock:
            keys_to_delete = []
            for key in self._storage.keys():
                if fnmatch.fnmatch(key, pattern):
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                entry = self._storage[key]
                self._current_memory -= entry.size
                del self._storage[key]

            return len(keys_to_delete)

    def get_stats(self) -> CacheStats:
        """Get cache statistics as CacheStats object."""
        return CacheStats(
            hits=0,  # Would need to track this separately
            misses=0,  # Would need to track this separately
            entries=len(self._storage),
            current_memory=self._current_memory,
            max_memory=self.max_memory,
            max_entries=self.max_entries,
        )


class DiskCache(CacheBackend):
    """Disk-based cache backend."""

    def __init__(
        self, cache_dir: str = "/tmp/orchestrator_cache", max_size: int = 10000
    ):
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
                with open(index_file, "r") as f:
                    self._index = json.load(f)
        except Exception:
            self._index = {}

    def _save_index(self):
        """Save cache index to disk."""
        import os

        index_file = os.path.join(self.cache_dir, "index.json")
        try:
            with open(index_file, "w") as f:
                json.dump(self._index, f)
        except Exception:
            pass

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from disk cache."""
        with self._lock:
            if key not in self._index:
                return None

            file_path = self._get_file_path(key)
            try:
                with open(file_path, "rb") as f:
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
                    "access_count": entry.access_count,
                }

                return entry

            except Exception:
                # File corruption or missing, remove from index
                if key in self._index:
                    del self._index[key]
                    self._save_index()
                return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in disk cache."""
        entry = CacheEntry(key=key, value=value, ttl=ttl)
        return await self.set_entry(entry)

    async def set_entry(self, entry: CacheEntry) -> bool:
        """Set CacheEntry in disk cache."""
        key = entry.key
        with self._lock:
            # Check size limit
            if len(self._index) >= self.max_size:
                await self._evict_oldest()

            file_path = self._get_file_path(key)
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(entry, f)

                # Update index
                self._index[key] = {
                    "created_at": entry.created_at,
                    "accessed_at": entry.accessed_at,
                    "access_count": entry.access_count,
                    "size": entry.size,
                }

                self._save_index()
                return True

            except Exception:
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

            except Exception:
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

            except Exception:
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
            self._index.keys(), key=lambda k: self._index[k].get("accessed_at", 0)
        )

        await self.delete(oldest_key)

    async def cleanup_expired(self):
        """Remove expired entries from cache."""
        with self._lock:
            time.time()
            expired_keys = []

            for key, entry_meta in self._index.items():
                file_path = self._get_file_path(key)
                try:
                    with open(file_path, "rb") as f:
                        entry = pickle.load(f)

                    if entry.is_expired():
                        expired_keys.append(key)
                except Exception:
                    # File corruption or missing, mark for removal
                    expired_keys.append(key)

            # Remove expired entries
            for key in expired_keys:
                await self.delete(key)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_dir": self.cache_dir,
            "total_files": len(self._index),
            "total_size_mb": sum(entry.get("size", 0) for entry in self._index.values())
            / (1024 * 1024),
            "oldest_entry": min(
                self._index.values(), key=lambda x: x.get("created_at", 0), default=None
            ),
            "newest_entry": max(
                self._index.values(), key=lambda x: x.get("created_at", 0), default=None
            ),
        }


class DistributedCache(CacheBackend):
    """Self-contained distributed cache using memory + disk (replacement for Redis)."""

    def __init__(
        self,
        cache_dir: str = None,
        max_memory_entries: int = 1000,
        max_disk_entries: int = 10000,
        memory_cache=None,
        disk_cache=None,
    ):
        self.max_memory_entries = max_memory_entries
        self.max_disk_entries = max_disk_entries

        # Use provided caches or create new ones
        if memory_cache:
            self.memory_cache = memory_cache
        else:
            self.memory_cache = MemoryCache(max_entries=max_memory_entries)

        if disk_cache:
            self.disk_cache = disk_cache
        else:
            if not cache_dir:
                import os
                import tempfile

                cache_dir = os.path.join(tempfile.gettempdir(), "orchestrator_cache")
            self.disk_cache = DiskCache(cache_dir=cache_dir, max_size=max_disk_entries)

        self._available = True

    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache (memory first, then disk)."""
        # Try memory cache first (fastest) - use async method
        try:
            memory_entry = await self.memory_cache.async_get(key)
            if memory_entry is not None:
                return memory_entry.value
        except Exception:
            pass

        # Try disk cache (slower but persistent)
        try:
            disk_entry = await self.disk_cache.get(key)
            if disk_entry is not None:
                # Promote to memory cache for faster future access
                try:
                    self.memory_cache.set(key, disk_entry.value, ttl=disk_entry.ttl)
                except Exception:
                    pass
                return disk_entry.value
        except Exception:
            pass

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in distributed cache (both memory and disk)."""
        try:
            # Set in memory cache (fast access) - use async method
            try:
                memory_success = await self.memory_cache.async_set(key, value, ttl=ttl)
            except Exception:
                memory_success = False

            # Set in disk cache (persistence) - asynchronous
            try:
                disk_success = await self.disk_cache.set(key, value, ttl=ttl)
            except Exception:
                disk_success = False

            # Return True if at least one succeeded
            return memory_success or disk_success
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry from distributed cache (both memory and disk)."""
        try:
            # Delete from memory cache - async
            try:
                memory_success = await self.memory_cache.delete(key)
            except Exception:
                memory_success = False

            # Delete from disk cache - asynchronous
            try:
                disk_success = await self.disk_cache.delete(key)
            except Exception:
                disk_success = False

            # Return True if at least one succeeded
            return memory_success or disk_success
        except Exception:
            return False

    async def clear(self) -> bool:
        """Clear all entries from distributed cache."""
        try:
            # Clear memory cache - async
            try:
                memory_success = await self.memory_cache.clear()
            except Exception:
                memory_success = False

            # Clear disk cache - asynchronous
            try:
                disk_success = await self.disk_cache.clear()
            except Exception:
                disk_success = False

            # Return True if at least one succeeded
            return memory_success or disk_success
        except Exception:
            return False

    async def size(self) -> int:
        """Get total cache size (memory + disk)."""
        try:
            memory_size = len(self.memory_cache._storage)
            disk_size = await self.disk_cache.size()
            return memory_size + disk_size
        except Exception:
            return 0

    async def keys(self, pattern: str = "*") -> List[str]:
        """Get cache keys matching pattern from both memory and disk."""
        try:
            import fnmatch

            # Get keys from memory cache
            memory_keys = list(self.memory_cache._storage.keys())

            # Get keys from disk cache
            disk_keys = await self.disk_cache.keys()

            # Combine and deduplicate
            all_keys = list(set(memory_keys + disk_keys))

            # Filter by pattern
            if pattern == "*":
                return all_keys
            else:
                return [key for key in all_keys if fnmatch.fnmatch(key, pattern)]
        except Exception:
            return []

    async def batch_set(
        self, keys_values: List[tuple], ttl: Optional[int] = None
    ) -> bool:
        """Set multiple key-value pairs in batch."""
        try:
            success_count = 0
            for key, value in keys_values:
                if await self.set(key, value, ttl=ttl):
                    success_count += 1

            # Return True if at least half succeeded
            return success_count >= len(keys_values) // 2
        except Exception:
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        try:
            keys = await self.keys(pattern)
            deleted_count = 0

            for key in keys:
                if await self.delete(key):
                    deleted_count += 1

            return deleted_count
        except Exception:
            return 0


# Backward compatibility wrapper for RedisCache
class RedisCache(DistributedCache):
    """Backward compatibility wrapper for DistributedCache that accepts Redis parameters."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        redis_client=None,
        sync_redis_client=None,
        auto_fallback: bool = True,
        cache_dir: str = None,
        max_memory_entries: int = 1000,
        max_disk_entries: int = 10000,
        memory_cache=None,
        disk_cache=None,
    ):
        """Initialize with Redis-compatible parameters but use DistributedCache internally."""
        # If no Redis clients provided, use standard DistributedCache
        if not redis_client and not sync_redis_client:
            # Try to start Redis server if auto_fallback is False
            if not auto_fallback:
                if not self._check_and_start_redis(redis_url):
                    raise ConnectionError(f"Could not connect to Redis at {redis_url}")

            # Initialize as DistributedCache for self-contained operation
            super().__init__(
                cache_dir=cache_dir,
                max_memory_entries=max_memory_entries,
                max_disk_entries=max_disk_entries,
                memory_cache=memory_cache,
                disk_cache=disk_cache,
            )
            self._redis_available = False
        else:
            # If Redis clients are provided via dependency injection, use them with DistributedCache
            super().__init__(
                cache_dir=cache_dir,
                max_memory_entries=max_memory_entries,
                max_disk_entries=max_disk_entries,
                memory_cache=memory_cache,
                disk_cache=disk_cache,
            )
            self._redis_available = True

        # Store original Redis parameters for compatibility
        self.redis_url = redis_url
        self.auto_fallback = auto_fallback

    def _check_and_start_redis(self, redis_url: str) -> bool:
        """Check if Redis is running and attempt to start it if not."""
        import subprocess
        import time
        from urllib.parse import urlparse

        # Parse Redis URL to get host and port
        parsed = urlparse(redis_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379

        # Check if Redis is already running
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, int(port)))
            sock.close()
            if result == 0:
                return True  # Redis is already running
        except Exception:
            pass

        # Try to start Redis server
        try:
            # Try different Redis server commands
            redis_commands = [
                "redis-server",
                "redis-server.exe",
                "/usr/local/bin/redis-server",
            ]

            for cmd in redis_commands:
                try:
                    # Start Redis in background with custom port if needed
                    if port != 6379:
                        process = subprocess.Popen(
                            [cmd, "--port", str(port)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        process = subprocess.Popen(
                            [cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )

                    # Give Redis time to start
                    time.sleep(0.5)

                    # Check if Redis started successfully
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex((host, int(port)))
                    sock.close()

                    if result == 0:
                        return True  # Redis started successfully
                    else:
                        process.terminate()  # Stop the process if it didn't work

                except (FileNotFoundError, OSError):
                    continue  # Try next command

        except Exception:
            pass

        return False  # Could not start Redis


class MultiLevelCache:
    """Multi-level cache with automatic promotion/demotion."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Initialize cache backends
        self.memory_cache = MemoryCache(
            max_size=config.get("memory_cache_size", 1000),
            eviction_policy=EvictionPolicy.LRU,
        )

        self.disk_cache = DiskCache(
            cache_dir=config.get("disk_cache_dir", "./cache"),
            max_size=config.get("disk_cache_size", 1000),
        )

        # Redis cache (with fallback if Redis not available)
        try:
            self.redis_cache = RedisCache(
                config.get("redis_url", "redis://localhost:6379")
            )
        except Exception:
            # Fallback to memory cache if Redis not available
            self.redis_cache = MemoryCache(
                max_size=1000, eviction_policy=EvictionPolicy.LRU
            )

        # Cache strategy (placeholder)
        self.cache_strategy = "multi_level"

        # Default TTL
        self.default_ttl = config.get("default_ttl", 3600)

        # Internal state
        self.levels = {}
        self.level_order = []
        self.hit_stats = {level: 0 for level in CacheLevel}
        self.miss_stats = 0
        self.promotion_threshold = 3  # Promote after 3 hits
        self._lock = asyncio.Lock()

        # Add default levels
        self.add_level(CacheLevel.MEMORY, self.memory_cache)
        self.add_level(CacheLevel.DISK, self.disk_cache)
        self.add_level(CacheLevel.DISTRIBUTED, self.redis_cache)

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
                if backend is None:
                    continue

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

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Dict[str, Any] = None,
    ):
        """Set value in cache, starting from highest level."""
        CacheEntry(key=key, value=value, ttl=ttl, metadata=metadata or {})

        # Start from highest priority level
        for level in self.level_order:
            if level not in self.levels:
                continue

            backend = self.levels[level]
            ttl_int = int(ttl) if ttl is not None else None
            success = await backend.set(key, value, ttl_int)

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

    async def _promote_entry(
        self, key: str, entry: CacheEntry, current_level: CacheLevel
    ):
        """Promote entry to higher cache levels."""
        current_index = self.level_order.index(current_level)

        # Promote to all higher levels
        for i in range(current_index):
            higher_level = self.level_order[i]
            if higher_level in self.levels:
                backend = self.levels[higher_level]
                ttl_int = int(entry.ttl) if entry.ttl is not None else None
                await backend.set(key, entry.value, ttl_int)

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
            "level_hits": {
                level.name: count for level, count in self.hit_stats.items()
            },
            "level_statistics": {},
        }

        # Get individual level statistics
        for level, backend in self.levels.items():
            if hasattr(backend, "get_statistics"):
                stats["level_statistics"][level.name] = backend.get_statistics()

        return stats

    def reset_statistics(self):
        """Reset cache statistics."""
        self.hit_stats = {level: 0 for level in CacheLevel}
        self.miss_stats = 0

    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        # For memory cache, manually filter keys
        if hasattr(self.memory_cache, "_storage"):
            keys_to_remove = []
            for key in self.memory_cache._storage:
                if pattern.replace("*", "") in key:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                await self.memory_cache.delete(key)

        # For redis and disk cache, use their invalidate methods if available
        if hasattr(self.redis_cache, "invalidate_pattern"):
            await self.redis_cache.invalidate_pattern(pattern)
        if hasattr(self.disk_cache, "invalidate_pattern"):
            await self.disk_cache.invalidate_pattern(pattern)

    async def warmup(self, keys: List[str], data_loader: Callable):
        """Warm up cache with specified keys."""
        for key in keys:
            try:
                value = await data_loader(key)
                await self.set(key, value)
            except Exception:
                # Skip failed loads
                pass

    async def get_or_refresh(self, key: str, refresh_func: Callable) -> Any:
        """Get value from cache or refresh if expired."""
        value = await self.get(key)
        if value is None:
            # Cache miss or expired, refresh
            value = await refresh_func(key)
            await self.set(key, value)
        return value


# Additional cache implementations for testing


class LRUCache:
    """Simple LRU cache implementation."""

    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self.data = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def currsize(self) -> int:
        """Current size of the cache."""
        return len(self.data)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.data:
            # Move to end (most recently used)
            value = self.data[key]
            self.data.move_to_end(key)
            self.hits += 1
            return value
        else:
            self.misses += 1
            return None

    def set(self, key: str, value: Any):
        """Set value in cache."""
        if key in self.data:
            # Update existing
            self.data[key] = value
            self.data.move_to_end(key)
        else:
            # Add new
            if len(self.data) >= self.maxsize:
                # Remove least recently used
                self.data.popitem(last=False)
            self.data[key] = value

    def clear(self):
        """Clear all cache entries."""
        self.data.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        return {
            "size": len(self.data),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total_requests if total_requests > 0 else 0,
        }


class CacheStrategy:
    """Cache strategy management."""

    def __init__(self):
        self.policies = {}
        self.default_ttl = 3600

    def select_policy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Select cache policy based on data."""
        data_type = data.get("type", "default")

        if data_type == "user":
            return {"ttl": 7200, "cache_level": "memory"}
        elif data_type == "session":
            return {"ttl": 1800, "cache_level": "memory"}
        elif data_type == "temp":
            return {"ttl": 300, "cache_level": "memory"}
        else:
            return {"ttl": self.default_ttl, "cache_level": "multi"}

    def generate_key(self, prefix: str, data: Dict[str, Any]) -> str:
        """Generate cache key."""
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()
        return f"{prefix}:{data_hash}"

    def get_invalidation_keys(self, event: Dict[str, Any]) -> List[str]:
        """Get keys to invalidate based on event."""
        event_type = event.get("type", "")

        if event_type == "user_update":
            return [f"user:{event.get('user_id')}"]
        elif event_type == "session_logout":
            return [f"session:{event.get('session_id')}"]
        elif event_type == "config_change" and event.get("scope") == "global":
            return ["*"]  # Invalidate all
        else:
            return []

    def optimize_cache_placement(
        self, access_patterns: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Optimize cache placement based on access patterns."""
        optimizations = {}

        for pattern in access_patterns:
            key = pattern["key"]
            frequency = pattern["frequency"]
            recency = pattern["recency"]

            if frequency >= 50 and recency <= 0.2:
                level = "memory"
            elif frequency >= 10:
                level = "disk"
            else:
                level = "disk"

            optimizations[key] = {"level": level}

        return optimizations

    def select_warmup_keys(
        self, available_keys: List[str], max_keys: int = 10
    ) -> List[str]:
        """Select keys for cache warmup."""
        priority_keys = []

        # High priority patterns
        for key in available_keys:
            if "config:global" in key:
                priority_keys.append(key)
            elif "user:" in key:
                priority_keys.append(key)

        return priority_keys[:max_keys]


# Convenience function for creating cache key from task parameters
def create_cache_key(func_name: str, *args, **kwargs) -> str:
    """Create cache key from function name and arguments."""
    # Combine all arguments into a single data structure
    key_data = {"func": func_name, "args": args, "kwargs": kwargs}

    # Create deterministic hash from the data
    param_str = json.dumps(key_data, sort_keys=True, default=str)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    return f"{func_name}:{param_hash}"


class HybridCache:
    """
    Hybrid cache that combines memory and Redis caching.

    Provides fast memory access with Redis persistence and distributed capability.
    """

    def __init__(self, memory_cache: MemoryCache, redis_cache: RedisCache):
        """Initialize hybrid cache with memory and Redis backends."""
        self.memory_cache = memory_cache
        self.redis_cache = redis_cache

    async def get(self, key: str) -> Any:
        """Get value from cache, checking memory first then Redis."""
        # Try memory cache first
        entry = await self.memory_cache.get(key)
        if entry is not None:
            return entry.value

        # Fallback to Redis cache
        try:
            value = await self.redis_cache.get(key)
            if value is not None:
                # Store in memory for faster future access
                await self.memory_cache.set(key, value)
            return value
        except Exception:
            # If Redis fails, check memory one more time as fallback
            entry = await self.memory_cache.get(key)
            return entry.value if entry else None

    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in both memory and Redis caches."""
        # Always set in memory cache
        await self.memory_cache.set(key, value, ttl=ttl)

        # Try to set in Redis cache
        try:
            await self.redis_cache.set(key, value, ttl=ttl)
        except Exception:
            # Continue if Redis fails - memory cache still works
            pass

    async def delete(self, key: str) -> bool:
        """Delete from both caches."""
        memory_deleted = await self.memory_cache.delete(key)

        try:
            redis_deleted = await self.redis_cache.delete(key)
        except Exception:
            redis_deleted = False

        return memory_deleted or redis_deleted

    async def clear(self) -> None:
        """Clear both caches."""
        await self.memory_cache.clear()
        try:
            await self.redis_cache.clear()
        except Exception:
            pass

    def get_stats(self) -> CacheStats:
        """Get combined cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        # Combine with Redis stats if available
        return memory_stats


def async_cache_wrapper(cache, ttl: Optional[float] = None):
    """
    Async function cache decorator.

    Args:
        cache: Cache backend to use
        ttl: Time to live for cached values

    Returns:
        Decorator function
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = create_cache_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Cache miss - execute function
            result = await func(*args, **kwargs)

            # Store result in cache
            await cache.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator


def sync_cache_wrapper(cache, ttl: Optional[float] = None):
    """
    Synchronous function cache decorator.

    Args:
        cache: Cache backend to use (must support synchronous operations)
        ttl: Time to live for cached values

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = create_cache_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Cache miss - execute function
            result = func(*args, **kwargs)

            # Store result in cache
            cache.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator
