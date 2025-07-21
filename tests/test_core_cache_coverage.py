"""Direct import tests for cache module to achieve coverage measurement."""

import os
import tempfile
import time
from typing import Any, Optional, Dict

import pytest

# Direct import to ensure coverage measurement
from src.orchestrator.core.cache import RedisCache  # Backward compatibility alias
from src.orchestrator.core.cache import (
    CacheEntry,
    CacheLevel,
    CacheStats,
    CacheStrategy,
    DiskCache,
    DistributedCache,
    EvictionPolicy,
    HybridCache,
    LRUCache,
    MemoryCache,
    MultiLevelCache,
    async_cache_wrapper,
    create_cache_key,
    sync_cache_wrapper,
)


class TestCacheLevel:
    """Test CacheLevel enum."""

    def test_cache_level_values(self):
        """Test CacheLevel enum values."""
        assert CacheLevel.MEMORY.value == 1
        assert CacheLevel.DISK.value == 2
        assert CacheLevel.DISTRIBUTED.value == 3


class TestEvictionPolicy:
    """Test EvictionPolicy enum."""

    def test_eviction_policy_values(self):
        """Test EvictionPolicy enum values."""
        assert EvictionPolicy.LRU.value == "lru"
        assert EvictionPolicy.LFU.value == "lfu"
        assert EvictionPolicy.TTL.value == "ttl"
        assert EvictionPolicy.SIZE.value == "size"


class TestCacheEntry:
    """Test CacheEntry class."""

    def test_cache_entry_creation(self):
        """Test CacheEntry creation."""
        entry = CacheEntry(key="test_key", value="test_value")

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert isinstance(entry.created_at, float)
        assert isinstance(entry.accessed_at, float)
        assert entry.access_count == 0
        assert entry.ttl is None
        assert entry.size > 0  # Calculated automatically
        assert entry.metadata == {}

    def test_cache_entry_last_accessed_property(self):
        """Test last_accessed property alias."""
        entry = CacheEntry("key", "value")

        original_time = entry.last_accessed
        assert entry.last_accessed == entry.accessed_at

        # Test setter
        new_time = time.time() + 10
        entry.last_accessed = new_time
        assert entry.accessed_at == new_time

    def test_cache_entry_size_calculation(self):
        """Test size calculation for different value types."""
        # String value
        str_entry = CacheEntry("key", "hello")
        assert str_entry.size == 5

        # Bytes value
        bytes_entry = CacheEntry("key", b"hello")
        assert bytes_entry.size == 5

        # Integer value
        int_entry = CacheEntry("key", 42)
        assert int_entry.size == 8

        # Float value
        float_entry = CacheEntry("key", 3.14)
        assert float_entry.size == 8

        # List value
        list_entry = CacheEntry("key", [1, 2, 3])
        assert list_entry.size > 0

        # Dictionary value
        dict_entry = CacheEntry("key", {"a": 1, "b": 2})
        assert dict_entry.size > 0

    def test_cache_entry_is_expired(self):
        """Test expiration checking."""
        # Entry without TTL never expires
        entry_no_ttl = CacheEntry("key", "value")
        assert entry_no_ttl.is_expired() is False

        # Entry with future TTL not expired
        entry_future = CacheEntry("key", "value", ttl=100)
        assert entry_future.is_expired() is False

        # Entry with past TTL is expired
        entry_past = CacheEntry("key", "value", ttl=0.001)
        time.sleep(0.002)
        assert entry_past.is_expired() is True

    def test_cache_entry_touch(self):
        """Test touch method."""
        entry = CacheEntry("key", "value")

        original_accessed = entry.accessed_at
        original_count = entry.access_count

        time.sleep(0.001)
        entry.touch()

        assert entry.accessed_at > original_accessed
        assert entry.access_count == original_count + 1


class TestCacheStats:
    """Test CacheStats class."""

    def test_cache_stats_creation(self):
        """Test CacheStats creation."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.entries == 0
        assert stats.current_memory == 0
        assert stats.max_memory is None
        assert stats.max_entries is None

    def test_cache_stats_hit_rate(self):
        """Test hit rate calculation."""
        stats = CacheStats()

        # No requests = 0.0 hit rate
        assert stats.hit_rate == 0.0

        # All hits = 1.0 hit rate
        stats.hits = 10
        assert stats.hit_rate == 1.0

        # Mixed hits/misses
        stats.misses = 5
        assert stats.hit_rate == 10 / 15  # 10 hits out of 15 total

    def test_cache_stats_memory_utilization(self):
        """Test memory utilization calculation."""
        # Test with max_memory set
        stats = CacheStats(current_memory=500, max_memory=1000)
        assert stats.memory_utilization == 0.5

        # Test with max_memory = 0
        stats.max_memory = 0
        assert stats.memory_utilization == 0.0

        # Test with max_memory = None
        stats.max_memory = None
        assert stats.memory_utilization == 0.0

    def test_cache_stats_entry_utilization(self):
        """Test entry utilization calculation."""
        # Test with max_entries set
        stats = CacheStats(entries=50, max_entries=100)
        assert stats.entry_utilization == 0.5

        # Test with max_entries = 0
        stats.max_entries = 0
        assert stats.entry_utilization == 0.0

        # Test with max_entries = None
        stats.max_entries = None
        assert stats.entry_utilization == 0.0


class TestCreateCacheKey:
    """Test create_cache_key function."""

    def test_create_cache_key_simple(self):
        """Test simple cache key creation."""
        key = create_cache_key("test_function", "arg1", "arg2")
        assert isinstance(key, str)
        assert len(key) > 0

    def test_create_cache_key_with_kwargs(self):
        """Test cache key creation with kwargs."""
        key1 = create_cache_key("func", "arg1", param1="value1", param2="value2")
        key2 = create_cache_key("func", "arg1", param2="value2", param1="value1")
        # Should be same regardless of kwarg order
        assert key1 == key2

    def test_create_cache_key_different_args(self):
        """Test that different args create different keys."""
        key1 = create_cache_key("func", "arg1")
        key2 = create_cache_key("func", "arg2")
        assert key1 != key2


class TestMemoryCacheBasic:
    """Test basic MemoryCache functionality."""

    @pytest.mark.asyncio
    async def test_memory_cache_creation(self):
        """Test MemoryCache creation."""
        cache = MemoryCache()
        assert cache._storage == {}
        assert cache.max_memory > 0
        assert cache.max_entries > 0

    @pytest.mark.asyncio
    async def test_memory_cache_set_get(self):
        """Test basic set/get operations."""
        cache = MemoryCache()

        # Set value
        result = await cache.set("key1", "value1")
        assert result is True

        # Get value
        entry = await cache.get("key1")
        assert entry is not None
        assert entry.value == "value1"

        # Get non-existent key
        none_entry = await cache.get("nonexistent")
        assert none_entry is None

    @pytest.mark.asyncio
    async def test_memory_cache_delete(self):
        """Test delete operation."""
        cache = MemoryCache()

        await cache.set("key1", "value1")
        assert await cache.get("key1") is not None

        result = await cache.delete("key1")
        assert result is True
        assert await cache.get("key1") is None

        # Delete non-existent key
        result = await cache.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_memory_cache_clear(self):
        """Test clear operation."""
        cache = MemoryCache()

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    def test_memory_cache_get_stats(self):
        """Test statistics."""
        cache = MemoryCache()
        stats = cache.get_stats()

        assert isinstance(stats, CacheStats)
        assert stats.current_memory >= 0
        assert stats.entries >= 0


class TestDistributedCacheBasic:
    """Test basic DistributedCache functionality."""

    @pytest.mark.asyncio
    async def test_distributed_cache_creation(self):
        """Test DistributedCache creation."""
        cache = DistributedCache()
        assert cache._available is True
        assert cache.memory_cache is not None
        assert cache.disk_cache is not None

    @pytest.mark.asyncio
    async def test_distributed_cache_set_get(self):
        """Test basic set/get operations."""
        cache = DistributedCache()

        result = await cache.set("dist_key", "dist_value")
        assert result is True

        value = await cache.get("dist_key")
        assert value == "dist_value"

    @pytest.mark.asyncio
    async def test_distributed_cache_delete(self):
        """Test delete operation."""
        cache = DistributedCache()

        await cache.set("del_key", "del_value")
        result = await cache.delete("del_key")
        assert result is True

        value = await cache.get("del_key")
        assert value is None


class TestMultiLevelCacheBasic:
    """Test basic MultiLevelCache functionality."""

    def test_multi_level_cache_creation(self):
        """Test MultiLevelCache creation."""
        cache = MultiLevelCache()
        assert cache.memory_cache is not None
        assert cache.redis_cache is not None
        assert cache.disk_cache is not None
        assert cache.cache_strategy is not None

    @pytest.mark.asyncio
    async def test_multi_level_cache_set_get(self):
        """Test basic set/get operations."""
        cache = MultiLevelCache()

        result = await cache.set("multi_key", "multi_value")
        assert result is True

        value = await cache.get("multi_key")
        assert value == "multi_value"


class TestRedisCacheBasic:
    """Test basic RedisCache functionality."""

    def test_redis_cache_creation_mock_mode(self):
        """Test RedisCache creation in mock mode."""
        cache = RedisCache(mock_mode=True)
        assert cache._available is True

    @pytest.mark.asyncio
    async def test_redis_cache_mock_operations(self):
        """Test RedisCache operations in mock mode."""
        cache = RedisCache(mock_mode=True)

        result = await cache.set("redis_key", "redis_value")
        assert result is True

        # In mock mode, get returns None to simulate Redis unavailability
        value = await cache.get("redis_key")
        assert value is None


class TestLRUCacheBasic:
    """Test basic LRUCache functionality."""

    def test_lru_cache_creation(self):
        """Test LRUCache creation."""
        cache = LRUCache(maxsize=10)
        assert cache.maxsize == 10
        assert cache.currsize == 0

    def test_lru_cache_operations(self):
        """Test basic LRU operations."""
        cache = LRUCache(maxsize=2)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.set("key2", "value2")
        assert cache.get("key2") == "value2"

        # Adding third item should evict least recently used
        cache.set("key3", "value3")
        assert cache.get("key1") is None  # Should be evicted
        assert cache.get("key3") == "value3"


class TestCacheStrategyBasic:
    """Test basic CacheStrategy functionality."""

    def test_cache_strategy_creation(self):
        """Test CacheStrategy creation."""
        strategy = CacheStrategy()
        assert strategy is not None


class TestableAsyncCache:
    """Testable async cache for wrapper tests."""
    
    def __init__(self):
        self._data = {}
        self.call_history = []
        
    async def get(self, key: str) -> Optional[Any]:
        """Test version of async get."""
        self.call_history.append(('get', key))
        return self._data.get(key)
        
    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Test version of async set."""
        self.call_history.append(('set', key, value, ttl))
        self._data[key] = value


class TestAsyncCacheWrapper:
    """Test async cache wrapper functionality."""

    @pytest.mark.asyncio
    async def test_async_cache_wrapper_basic(self):
        """Test basic async cache wrapper."""
        test_cache = TestableAsyncCache()

        @async_cache_wrapper(cache=test_cache, ttl=300)
        async def test_func(x):
            return f"result_{x}"

        result = await test_func("test")
        assert result == "result_test"
        
        # Verify cache was called
        assert len(test_cache.call_history) == 2  # get and set
        assert test_cache.call_history[0][0] == 'get'
        assert test_cache.call_history[1][0] == 'set'


class TestableSyncCache:
    """Testable sync cache for wrapper tests."""
    
    def __init__(self):
        self._data = {}
        self.call_history = []
        
    def get(self, key: str) -> Optional[Any]:
        """Test version of sync get."""
        self.call_history.append(('get', key))
        return self._data.get(key)
        
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Test version of sync set."""
        self.call_history.append(('set', key, value, ttl))
        self._data[key] = value


def test_sync_cache_wrapper_basic():
    """Test basic sync cache wrapper."""
    test_cache = TestableSyncCache()

    @sync_cache_wrapper(cache=test_cache, ttl=300)
    def test_func(x):
        return f"result_{x}"

    result = test_func("test")
    assert result == "result_test"
    
    # Verify cache was called
    assert len(test_cache.call_history) == 2  # get and set
    assert test_cache.call_history[0][0] == 'get'
    assert test_cache.call_history[1][0] == 'set'


class TestDiskCacheBasic:
    """Test basic DiskCache functionality."""

    @pytest.mark.asyncio
    async def test_disk_cache_creation(self):
        """Test DiskCache creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(cache_dir=temp_dir)
            assert cache.cache_dir == temp_dir
            assert os.path.exists(temp_dir)

    @pytest.mark.asyncio
    async def test_disk_cache_set_get(self):
        """Test basic disk cache operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(cache_dir=temp_dir)

            result = await cache.set("disk_key", "disk_value")
            assert result is True

            entry = await cache.get("disk_key")
            assert entry is not None
            assert entry.value == "disk_value"


class TestHybridCacheBasic:
    """Test basic HybridCache functionality."""

    def test_hybrid_cache_creation(self):
        """Test HybridCache creation."""
        memory_cache = MemoryCache()
        redis_cache = RedisCache(mock_mode=True)

        cache = HybridCache(memory_cache=memory_cache, redis_cache=redis_cache)
        assert cache.memory_cache == memory_cache
        assert cache.redis_cache == redis_cache

    @pytest.mark.asyncio
    async def test_hybrid_cache_operations(self):
        """Test basic hybrid cache operations."""
        memory_cache = MemoryCache()
        redis_cache = RedisCache(mock_mode=True)
        cache = HybridCache(memory_cache=memory_cache, redis_cache=redis_cache)

        await cache.set("hybrid_key", "hybrid_value")
        result = await cache.get("hybrid_key")
        assert result == "hybrid_value"
