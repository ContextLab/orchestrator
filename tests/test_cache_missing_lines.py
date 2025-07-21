"""Tests for missing lines in cache module to achieve 100% coverage."""

import time
import asyncio
from typing import Any, Optional

import pytest

from src.orchestrator.core.cache import (
    CacheStats,
    DistributedCache,
    EvictionPolicy,
    HybridCache,
    MemoryCache,
    async_cache_wrapper,
    sync_cache_wrapper,
    CacheEntry,
)


class TestableMemoryCache(MemoryCache):
    """Testable memory cache with controllable behavior."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._test_get_result = None
        self._test_set_result = True
        self._test_delete_result = True
        self._test_raise_error = None
        self.call_history = []
        
    def set_test_get_result(self, result):
        """Set what get() should return."""
        self._test_get_result = result
        
    def set_test_set_result(self, result: bool):
        """Set what set() should return."""
        self._test_set_result = result
        
    def set_test_delete_result(self, result: bool):
        """Set what delete() should return."""
        self._test_delete_result = result
        
    def set_test_error(self, error: Exception):
        """Set an error to raise."""
        self._test_raise_error = error
        
    async def async_get(self, key: str) -> Optional[CacheEntry]:
        """Test version of async_get."""
        self.call_history.append(('async_get', key))
        if self._test_raise_error:
            raise self._test_raise_error
        if self._test_get_result is not None:
            return CacheEntry(key=key, value=self._test_get_result)
        return None
        
    async def async_set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Test version of async_set."""
        self.call_history.append(('async_set', key, value, ttl))
        if self._test_raise_error:
            raise self._test_raise_error
        return self._test_set_result
        
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Test version of sync set."""
        self.call_history.append(('set', key, value, ttl))
        if self._test_raise_error:
            raise self._test_raise_error
        return self._test_set_result
        
    async def delete(self, key: str) -> bool:
        """Test version of delete."""
        self.call_history.append(('delete', key))
        if self._test_raise_error:
            raise self._test_raise_error
        return self._test_delete_result
        
    async def clear(self):
        """Test version of clear."""
        self.call_history.append(('clear',))
        if self._test_raise_error:
            raise self._test_raise_error
        await super().clear()


class TestableDiskCache:
    """Testable disk cache with controllable behavior."""
    
    def __init__(self):
        self._test_get_result = None
        self._test_set_result = True
        self._test_delete_result = True
        self._test_raise_error = None
        self.call_history = []
        
    def set_test_get_result(self, value: Any, ttl: Optional[float] = None):
        """Set what get() should return."""
        if value is not None:
            self._test_get_result = CacheEntry(key="test", value=value, ttl=ttl)
        else:
            self._test_get_result = None
            
    def set_test_error(self, error: Exception):
        """Set an error to raise."""
        self._test_raise_error = error
        
    def get(self, key: str) -> Optional[CacheEntry]:
        """Test version of get."""
        self.call_history.append(('get', key))
        if self._test_raise_error:
            raise self._test_raise_error
        return self._test_get_result
        
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Test version of set."""
        self.call_history.append(('set', key, value, ttl))
        if self._test_raise_error:
            raise self._test_raise_error
        return self._test_set_result
        
    async def delete(self, key: str) -> bool:
        """Test version of delete."""
        self.call_history.append(('delete', key))
        if self._test_raise_error:
            raise self._test_raise_error
        return self._test_delete_result
        
    async def clear(self):
        """Test version of clear."""
        self.call_history.append(('clear',))
        if self._test_raise_error:
            raise self._test_raise_error


class TestableDistributedCache:
    """Testable distributed cache with controllable behavior."""
    
    def __init__(self):
        self._test_get_result = None
        self._test_set_result = True
        self._test_delete_result = True
        self._test_raise_error = None
        self.call_history = []
        
    def set_test_error(self, error: Exception):
        """Set an error to raise."""
        self._test_raise_error = error
        
    async def get(self, key: str) -> Optional[Any]:
        """Test version of get."""
        self.call_history.append(('get', key))
        if self._test_raise_error:
            raise self._test_raise_error
        return self._test_get_result
        
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Test version of set."""
        self.call_history.append(('set', key, value, ttl))
        if self._test_raise_error:
            raise self._test_raise_error
        return self._test_set_result
        
    async def delete(self, key: str) -> bool:
        """Test version of delete."""
        self.call_history.append(('delete', key))
        if self._test_raise_error:
            raise self._test_raise_error
        return self._test_delete_result
        
    async def clear(self):
        """Test version of clear."""
        self.call_history.append(('clear',))
        if self._test_raise_error:
            raise self._test_raise_error


class TestableCache:
    """Simple testable cache for wrapper tests."""
    
    def __init__(self):
        self._data = {}
        self.call_history = []
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self.call_history.append(('get', key))
        return self._data.get(key)
        
    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        self.call_history.append(('set', key, value, ttl))
        self._data[key] = value


class TestMemoryCacheMissingLines:
    """Tests for missing lines in MemoryCache."""

    @pytest.mark.asyncio
    async def test_memory_cache_ttl_eviction_policy_lines_237_247(self):
        """Test TTL eviction policy (lines 237-247)."""
        cache = MemoryCache(max_memory=1000, eviction_policy=EvictionPolicy.TTL)

        # Add entries with TTL
        await cache.set("key1", "value1", ttl=1)  # Short TTL
        await cache.set("key2", "value2", ttl=100)  # Long TTL

        # Wait for first entry to expire
        time.sleep(1.1)

        # Force eviction by adding a large entry that exceeds memory limit
        large_value = "x" * 900  # Should trigger eviction
        await cache.set("large_key", large_value)

        # TTL eviction should have removed expired key1
        entry1 = await cache.get("key1")  # Expired
        entry2 = await cache.get("key2")  # Still valid
        entry3 = await cache.get("large_key")  # New entry

        assert entry1 is None  # Expired
        assert entry2 is not None  # Still valid
        assert entry3 is not None  # New entry

        # Verify stats
        stats = cache.get_stats()
        assert stats.current_memory < 1000  # Memory was freed by TTL eviction

    def test_memory_cache_sync_get_with_entry_lines_267_269(self):
        """Test sync get that returns entry object (lines 267-269)."""
        cache = MemoryCache()
        
        # Use synchronous method to set a value
        cache.set_sync("sync_key", {"data": "value"})
        
        # Sync get should return the value directly
        result = cache.get_sync("sync_key")
        assert result == {"data": "value"}
        
        # Non-existent key should return None
        result = cache.get_sync("non_existent")
        assert result is None

    def test_cache_stats_hit_rate_zero_requests_line_93(self):
        """Test CacheStats hit_rate with zero requests (line 93)."""
        stats = CacheStats(hits=0, misses=0)
        
        # With zero requests, hit rate should be 0.0
        assert stats.hit_rate == 0.0
        
        # Test with some hits but still zero total
        stats = CacheStats(hits=5, misses=0, entries=0)
        assert stats.hit_rate == 1.0  # All hits

    def test_cache_stats_memory_utilization_zero_max_line_99(self):
        """Test CacheStats memory_utilization with zero max_memory (line 99)."""
        stats = CacheStats(current_memory=1024, max_memory=0)
        
        # With zero max memory, utilization should be 0.0
        assert stats.memory_utilization == 0.0

    def test_cache_stats_entry_utilization_zero_max_line_105(self):
        """Test CacheStats entry_utilization with zero max_entries (line 105)."""
        stats = CacheStats(entries=10, max_entries=0)
        
        # With zero max entries, utilization should be 0.0
        assert stats.entry_utilization == 0.0


class TestDistributedCacheMissingLines:
    """Tests for missing lines in DistributedCache."""

    @pytest.mark.asyncio
    async def test_distributed_cache_memory_hit_lines_325_328(self):
        """Test memory cache hit path (lines 325-328)."""
        # Create testable memory cache with data
        test_memory = TestableMemoryCache()
        test_memory.set_test_get_result({"test": "data"})
        
        # Create testable disk cache that won't be called
        test_disk = TestableDiskCache()
        
        # Use dependency injection to provide test caches
        cache = DistributedCache(memory_cache=test_memory, disk_cache=test_disk)
        
        # Get from cache - should hit memory
        result = await cache.get("test_key")
        assert result == {"test": "data"}
        
        # Verify only memory was called
        assert any(call[0] == 'async_get' for call in test_memory.call_history)
        assert len(test_disk.call_history) == 0

    @pytest.mark.asyncio
    async def test_distributed_cache_disk_hit_promote_lines_331_339(self):
        """Test disk cache hit with promotion to memory (lines 331-339)."""
        # Create testable memory cache that returns None (cache miss)
        test_memory = TestableMemoryCache()
        test_memory.set_test_get_result(None)
        
        # Create testable disk cache with data
        test_disk = TestableDiskCache()
        test_disk.set_test_get_result({"async": "data"}, ttl=300)
        
        # Use dependency injection to provide test caches
        cache = DistributedCache(memory_cache=test_memory, disk_cache=test_disk)
        
        # Get from cache - should miss memory, hit disk, promote to memory
        result = await cache.get("async_key")
        assert result == {"async": "data"}
        
        # Verify both caches were called appropriately
        async_get_calls = [c for c in test_memory.call_history if c[0] == 'async_get']
        assert len(async_get_calls) == 1
        assert len(test_disk.call_history) == 1
        # Verify promotion to memory
        set_calls = [c for c in test_memory.call_history if c[0] == 'set']
        assert len(set_calls) == 1
        assert set_calls[0][2] == {"async": "data"}  # Value
        assert set_calls[0][3] == 300  # TTL

    @pytest.mark.asyncio
    async def test_distributed_cache_miss_lines_342_343(self):
        """Test complete cache miss (lines 342-343)."""
        # Create testable memory and disk caches that return None
        test_memory = TestableMemoryCache()
        test_memory.set_test_get_result(None)
        
        test_disk = TestableDiskCache()
        test_disk.set_test_get_result(None)
        
        # Use dependency injection to provide test caches
        cache = DistributedCache(memory_cache=test_memory, disk_cache=test_disk)
        
        # Get from cache - should miss both
        result = await cache.get("missing_key")
        assert result is None
        
        # Verify both caches were checked
        assert any(call[0] == 'async_get' for call in test_memory.call_history)
        assert any(call[0] == 'get' for call in test_disk.call_history)

    @pytest.mark.asyncio
    async def test_distributed_cache_disk_error_lines_340_341(self):
        """Test disk cache error handling (lines 340-341)."""
        # Create testable memory cache that returns None
        test_memory = TestableMemoryCache()
        test_memory.set_test_get_result(None)
        
        # Create testable disk cache that raises exception
        test_disk = TestableDiskCache()
        test_disk.set_test_error(Exception("Disk error"))
        
        cache = DistributedCache(memory_cache=test_memory, disk_cache=test_disk)
        
        # Get should handle disk error gracefully
        result = await cache.get("error_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_distributed_cache_set_with_ttl_lines_349_354(self):
        """Test set operation with TTL (lines 349-354)."""
        # Test memory and disk caches using async methods
        test_memory = TestableMemoryCache()
        test_disk = TestableDiskCache()
        
        cache = DistributedCache(memory_cache=test_memory, disk_cache=test_disk)
        
        # Set with TTL
        result = await cache.set("ttl_key", {"data": "value"}, ttl=300)
        assert result is True
        
        # Verify both caches were called with correct TTL
        mem_set_calls = [c for c in test_memory.call_history if c[0] == 'async_set']
        assert len(mem_set_calls) == 1
        assert mem_set_calls[0][3] == 300  # TTL
        
        disk_set_calls = [c for c in test_disk.call_history if c[0] == 'set']
        assert len(disk_set_calls) == 1
        assert disk_set_calls[0][2] == 300  # TTL

    @pytest.mark.asyncio
    async def test_distributed_cache_set_without_ttl_line_356(self):
        """Test set operation without TTL (line 356)."""
        test_memory = TestableMemoryCache()
        test_disk = TestableDiskCache()
        
        cache = DistributedCache(memory_cache=test_memory, disk_cache=test_disk)
        
        # Set without TTL
        result = await cache.set("no_ttl_key", {"data": "value"})
        assert result is True
        
        # Verify both set methods called
        assert len(test_memory.call_history) == 1
        assert len(test_disk.call_history) == 1

    @pytest.mark.asyncio
    async def test_distributed_cache_set_disk_failure_line_359(self):
        """Test set with disk failure still succeeds (line 359)."""
        # Test memory cache to succeed
        test_memory = TestableMemoryCache()
        
        # Test disk cache to fail
        test_disk = TestableDiskCache()
        test_disk._test_set_result = False
        
        cache = DistributedCache(memory_cache=test_memory, disk_cache=test_disk)
        
        # Set should still succeed (memory succeeded)
        result = await cache.set("partial_success_key", {"data": "value"}, ttl=60)
        assert result is True
        
        # Verify both caches were attempted
        assert len(test_memory.call_history) == 1
        assert len(test_disk.call_history) == 1

    @pytest.mark.asyncio
    async def test_distributed_cache_delete_lines_365_367(self):
        """Test delete operation (lines 365-367)."""
        # Test memory and disk caches
        test_memory = TestableMemoryCache()
        test_disk = TestableDiskCache()
        
        cache = DistributedCache(memory_cache=test_memory, disk_cache=test_disk)
        
        # Delete from both caches
        result = await cache.delete("delete_key")
        assert result is True
        
        # Verify both caches had delete called
        mem_delete_calls = [c for c in test_memory.call_history if c[0] == 'delete']
        assert len(mem_delete_calls) == 1
        disk_delete_calls = [c for c in test_disk.call_history if c[0] == 'delete']
        assert len(disk_delete_calls) == 1

    @pytest.mark.asyncio
    async def test_distributed_cache_delete_disk_failure_line_370(self):
        """Test delete with disk failure (line 370)."""
        # Test memory cache to succeed (async delete)
        test_memory = TestableMemoryCache()
        
        # Test disk cache to fail
        test_disk = TestableDiskCache()
        test_disk._test_delete_result = False
        
        cache = DistributedCache(memory_cache=test_memory, disk_cache=test_disk)
        
        # Delete should still return True (memory succeeded)
        result = await cache.delete("async_delete_key")
        assert result is True
        
        # Verify both caches were attempted
        mem_calls = [c for c in test_memory.call_history if c[0] == 'delete']
        assert len(mem_calls) == 1
        disk_calls = [c for c in test_disk.call_history if c[0] == 'delete']
        assert len(disk_calls) == 1


class TestHybridCacheMissingLines:
    """Tests for missing lines in HybridCache."""

    @pytest.mark.asyncio
    async def test_hybrid_cache_set_distributed_failure_lines_396_397(self):
        """Test set with distributed cache failure (lines 396-397)."""
        # Test memory cache
        test_memory = TestableMemoryCache()
        
        # Test DistributedCache (formerly Redis) to fail
        test_distributed = TestableDistributedCache()
        test_distributed.set_test_error(Exception("Distributed cache failed"))
        
        cache = HybridCache(memory_cache=test_memory, redis_cache=test_distributed)
        
        # Set should succeed despite distributed failure
        result = await cache.set("hybrid_key", "hybrid_value", ttl=300)
        assert result is True
        
        # Verify memory was set
        mem_calls = [c for c in test_memory.call_history if c[0] == 'set']
        assert len(mem_calls) == 1
        
        # Verify distributed was attempted
        dist_calls = [c for c in test_distributed.call_history if c[0] == 'set']
        assert len(dist_calls) == 1

    @pytest.mark.asyncio
    async def test_hybrid_cache_get_distributed_failure_lines_411_416(self):
        """Test get with distributed cache failure and memory fallback (lines 411-416)."""
        # Test memory cache with data for fallback
        test_memory = TestableMemoryCache()
        # First call returns None, second returns value (simulating fallback)
        test_memory._test_results = [None, {"value": "memory_value"}]
        test_memory._call_count = 0
        
        async def custom_get(key):
            test_memory.call_history.append(('get', key))
            if test_memory._call_count == 0:
                test_memory._call_count += 1
                return None
            else:
                return CacheEntry(key=key, value="memory_value")
                
        test_memory.get = custom_get
        
        # Test DistributedCache to fail
        test_distributed = TestableDistributedCache()
        test_distributed.set_test_error(Exception("Distributed cache failed"))
        
        cache = HybridCache(memory_cache=test_memory, redis_cache=test_distributed)
        
        # Get should fall back to memory after distributed fails
        result = await cache.get("fallback_key")
        assert result == "memory_value"
        
        # Verify distributed was attempted
        assert len(test_distributed.call_history) == 1
        # Verify memory was called twice (once for initial check, once for fallback)
        assert len(test_memory.call_history) == 2

    @pytest.mark.asyncio
    async def test_hybrid_cache_delete_both_success_lines_427_430(self):
        """Test delete from both caches (lines 427-430)."""
        # Test both caches
        test_memory = TestableMemoryCache()
        test_distributed = TestableDistributedCache()
        
        cache = HybridCache(memory_cache=test_memory, redis_cache=test_distributed)
        
        # Delete from both
        result = await cache.delete("delete_both_key")
        assert result is True
        
        # Verify both caches had delete called
        mem_calls = [c for c in test_memory.call_history if c[0] == 'delete']
        assert len(mem_calls) == 1
        dist_calls = [c for c in test_distributed.call_history if c[0] == 'delete']
        assert len(dist_calls) == 1

    @pytest.mark.asyncio
    async def test_hybrid_cache_clear_both_lines_437_439(self):
        """Test clear both caches (lines 437-439)."""
        # Test both caches
        test_memory = TestableMemoryCache()
        test_distributed = TestableDistributedCache()
        
        cache = HybridCache(memory_cache=test_memory, redis_cache=test_distributed)
        
        # Clear both
        await cache.clear()
        
        # Verify both caches had clear called
        mem_calls = [c for c in test_memory.call_history if c[0] == 'clear']
        assert len(mem_calls) == 1
        dist_calls = [c for c in test_distributed.call_history if c[0] == 'clear']
        assert len(dist_calls) == 1


class TestCacheWrappersMissingLines:
    """Tests for missing lines in cache wrappers."""

    @pytest.mark.asyncio
    async def test_async_cache_wrapper_cache_miss_line_512(self):
        """Test async_cache_wrapper with cache miss (line 512)."""
        # Create a simple async function
        call_count = 0
        
        async def test_func(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # Test cache that returns None (cache miss)
        test_cache = TestableCache()
        
        # Wrap the function
        cached_func = async_cache_wrapper(cache=test_cache, ttl=300)(test_func)
        
        # Call the function
        result = await cached_func(5, 10)
        assert result == 15
        assert call_count == 1
        
        # Verify cache operations
        get_calls = [c for c in test_cache.call_history if c[0] == 'get']
        set_calls = [c for c in test_cache.call_history if c[0] == 'set']
        assert len(get_calls) == 1
        assert len(set_calls) == 1

    def test_sync_cache_wrapper_cache_miss_line_532(self):
        """Test sync_cache_wrapper with cache miss (line 532)."""
        # Create a simple sync function
        call_count = 0
        
        def test_func(x, y):
            nonlocal call_count
            call_count += 1
            return x * y
        
        # Test cache that returns None (cache miss)
        test_cache = TestableCache()
        
        # Wrap the function - note sync wrapper expects sync cache methods
        cached_func = sync_cache_wrapper(cache=test_cache, ttl=600)(test_func)
        
        # Call the function
        result = cached_func(3, 4)
        assert result == 12
        assert call_count == 1
        
        # Verify cache operations
        get_calls = [c for c in test_cache.call_history if c[0] == 'get']
        set_calls = [c for c in test_cache.call_history if c[0] == 'set']
        assert len(get_calls) == 1
        assert len(set_calls) == 1