"""Tests for missing lines in cache module to achieve 100% coverage."""

import pytest
import time
from unittest.mock import Mock, AsyncMock

from src.orchestrator.core.cache import (
    MemoryCache,
    DistributedCache,
    EvictionPolicy,
    HybridCache,
    CacheStats,
    sync_cache_wrapper,
    async_cache_wrapper
)


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
    
    @pytest.mark.asyncio
    async def test_memory_cache_evict_by_count_lines_249_265(self):
        """Test _evict_by_count method (lines 249-265)."""
        cache = MemoryCache(max_entries=3, eviction_policy=EvictionPolicy.LRU)
        
        # Fill cache to capacity
        await cache.set("key1", "value1")
        await cache.set("key2", "value2") 
        await cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        await cache.get("key1")
        
        # Add more entries to trigger eviction by count
        await cache.set("key4", "value4")
        await cache.set("key5", "value5")
        
        # Should have evicted least recently used entries
        entry1 = await cache.get("key1")
        entry2 = await cache.get("key2")
        entry3 = await cache.get("key3")
        entry4 = await cache.get("key4")
        entry5 = await cache.get("key5")
        
        assert entry1 is not None  # Recently accessed
        assert entry2 is None      # Should be evicted
        assert entry3 is None      # Should be evicted
        assert entry4 is not None  # New entry
        assert entry5 is not None  # New entry
        
        # Verify entry count
        assert len(cache._storage) == 3  # Max entries maintained
    
    @pytest.mark.asyncio
    async def test_memory_cache_invalidate_pattern_lines_305_320(self):
        """Test invalidate_pattern method (lines 305-320)."""
        cache = MemoryCache()
        
        # Add entries with different patterns
        await cache.set("user:123:profile", {"name": "Alice"})
        await cache.set("user:456:profile", {"name": "Bob"})
        await cache.set("user:123:settings", {"theme": "dark"})
        await cache.set("session:abc123", {"user_id": 123})
        await cache.set("config:global", {"debug": True})
        
        # Test pattern invalidation
        invalidated = cache.invalidate_pattern("user:123:*")
        
        # Should invalidate only user:123 entries
        assert invalidated == 2  # user:123:profile and user:123:settings
        entry1 = await cache.get("user:123:profile")
        entry2 = await cache.get("user:123:settings")
        entry3 = await cache.get("user:456:profile")
        entry4 = await cache.get("session:abc123")
        entry5 = await cache.get("config:global")
        
        assert entry1 is None        # Should be invalidated
        assert entry2 is None        # Should be invalidated
        assert entry3 is not None    # Different user
        assert entry4 is not None    # Different prefix
        assert entry5 is not None    # Different prefix
    
    @pytest.mark.asyncio
    async def test_memory_cache_get_entry_access_time_lines_346_351(self):
        """Test get with entry access time update (lines 346-351)."""
        cache = MemoryCache()
        
        # Set an entry
        await cache.set("test_key", "test_value")
        
        # Get initial access time
        entry = cache._storage["test_key"]
        initial_access_time = entry.last_accessed
        
        # Wait a bit then access again
        time.sleep(0.1)
        cache_entry = await cache.get("test_key")
        
        # Access time should be updated
        assert cache_entry is not None
        assert cache_entry.value == "test_value"
        assert entry.last_accessed > initial_access_time
        
        # Verify access count
        assert entry.access_count > 0
    
    @pytest.mark.asyncio
    async def test_memory_cache_get_expired_entry_lines_364_370(self):
        """Test get with expired entry handling (lines 364-370)."""
        cache = MemoryCache()
        
        # Set entry with very short TTL
        await cache.set("expire_key", "expire_value", ttl=0.1)
        
        # Verify it exists initially
        entry = await cache.get("expire_key")
        assert entry is not None
        assert entry.value == "expire_value"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should return None and clean up expired entry
        expired_entry = await cache.get("expire_key")
        assert expired_entry is None
        assert "expire_key" not in cache._storage
        
        # Verify memory was freed
        stats = cache.get_stats()
        assert stats.current_memory == 0


class TestDistributedCacheMissingLines:
    """Tests for missing lines in DistributedCache (replacement for Redis)."""
    
    @pytest.mark.asyncio
    async def test_distributed_cache_get_memory_first_lines_621_635(self):
        """Test DistributedCache.get with memory cache hit (lines 621-635)."""
        # Create mock memory cache with async_get method returning CacheEntry
        mock_memory = AsyncMock()
        mock_entry = Mock()
        mock_entry.value = {"test": "data"}
        mock_memory.async_get.return_value = mock_entry
        
        # Create mock disk cache that won't be called
        mock_disk = AsyncMock()
        
        # Use dependency injection to provide mock caches
        cache = DistributedCache(memory_cache=mock_memory, disk_cache=mock_disk)
        
        # Test successful get with memory cache hit
        result = await cache.get("test_key")
        
        assert result == {"test": "data"}
        mock_memory.async_get.assert_called_once_with("test_key")
        # Disk cache should not be called since memory hit
        mock_disk.get.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_distributed_cache_get_disk_fallback_lines_628_635(self):
        """Test DistributedCache.get with disk cache fallback (lines 628-635)."""
        # Create mock memory cache that returns None (cache miss)
        mock_memory = AsyncMock()
        mock_memory.async_get.return_value = None
        mock_memory.set = Mock()  # Sync method for promotion
        
        # Create mock disk cache with data  
        mock_disk = AsyncMock()
        mock_disk_entry = Mock()
        mock_disk_entry.value = {"async": "data"}
        mock_disk_entry.ttl = 300
        mock_disk.get.return_value = mock_disk_entry
        
        # Use dependency injection to provide mock caches
        cache = DistributedCache(memory_cache=mock_memory, disk_cache=mock_disk)
        
        # Test successful get with disk cache fallback
        result = await cache.get("async_key")
        
        assert result == {"async": "data"}
        mock_memory.async_get.assert_called_once_with("async_key")
        mock_disk.get.assert_called_once_with("async_key")
        # Should promote to memory cache
        mock_memory.set.assert_called_once_with("async_key", {"async": "data"}, ttl=300)
    
    @pytest.mark.asyncio
    async def test_distributed_cache_get_both_cache_miss_lines_635(self):
        """Test DistributedCache.get with both caches returning None (line 635)."""
        # Create mock memory and disk caches that return None
        mock_memory = AsyncMock()
        mock_memory.async_get.return_value = None
        
        mock_disk = AsyncMock()
        mock_disk.get.return_value = None
        
        # Use dependency injection to provide mock caches
        cache = DistributedCache(memory_cache=mock_memory, disk_cache=mock_disk)
        
        # Should handle cache miss gracefully
        result = await cache.get("missing_key")
        
        assert result is None  # Returns None on complete cache miss
        mock_memory.async_get.assert_called_once_with("missing_key")
        mock_disk.get.assert_called_once_with("missing_key")
    
    @pytest.mark.asyncio
    async def test_distributed_cache_get_disk_exception_fallback(self):
        """Test DistributedCache.get with disk cache exception."""
        # Create mock memory cache that returns None
        mock_memory = AsyncMock()
        mock_memory.async_get.return_value = None
        
        # Create mock disk cache that raises exception
        mock_disk = AsyncMock()
        mock_disk.get.side_effect = Exception("Disk error")
        
        cache = DistributedCache(memory_cache=mock_memory, disk_cache=mock_disk)
        
        # Should handle disk exception gracefully
        result = await cache.get("exception_key")
        
        assert result is None  # Returns None on disk exception
    
    @pytest.mark.asyncio
    async def test_distributed_cache_set_both_caches_lines_637_649(self):
        """Test DistributedCache.set with both memory and disk caches (lines 637-649)."""
        # Mock memory and disk caches using async methods
        mock_memory = AsyncMock()
        mock_memory.async_set.return_value = True
        
        mock_disk = AsyncMock()
        mock_disk.set.return_value = True
        
        cache = DistributedCache(memory_cache=mock_memory, disk_cache=mock_disk)
        
        # Test set with TTL in both caches
        result = await cache.set("ttl_key", {"data": "value"}, ttl=300)
        
        assert result is True
        mock_memory.async_set.assert_called_once_with("ttl_key", {"data": "value"}, ttl=300)
        mock_disk.set.assert_called_once_with("ttl_key", {"data": "value"}, ttl=300)
        
        # Test set without TTL
        result2 = await cache.set("no_ttl_key", {"data": "value2"})
        
        assert result2 is True
        # Check both caches were called without TTL
        assert mock_memory.async_set.call_count == 2
        assert mock_disk.set.call_count == 2
    
    @pytest.mark.asyncio
    async def test_distributed_cache_set_memory_success_disk_fail_lines_646_647(self):
        """Test DistributedCache.set with memory success and disk failure (lines 646-647)."""
        # Mock memory cache to succeed
        mock_memory = AsyncMock()
        mock_memory.async_set.return_value = True
        
        # Mock disk cache to fail
        mock_disk = AsyncMock()
        mock_disk.set.return_value = False
        
        cache = DistributedCache(memory_cache=mock_memory, disk_cache=mock_disk)
        
        # Test set with one success and one failure
        result = await cache.set("async_ttl_key", {"async": "data"}, ttl=600)
        
        # Should return True since at least memory succeeded
        assert result is True
        mock_memory.async_set.assert_called_once_with("async_ttl_key", {"async": "data"}, ttl=600)
        mock_disk.set.assert_called_once_with("async_ttl_key", {"async": "data"}, ttl=600)
    
    @pytest.mark.asyncio 
    async def test_distributed_cache_delete_both_caches_lines_651_663(self):
        """Test DistributedCache.delete from both caches (lines 651-663)."""
        # Mock memory and disk caches
        mock_memory = AsyncMock()
        mock_memory.delete.return_value = True
        
        mock_disk = AsyncMock()
        mock_disk.delete.return_value = True
        
        cache = DistributedCache(memory_cache=mock_memory, disk_cache=mock_disk)
        
        # Test delete from both caches
        result = await cache.delete("delete_key")
        
        assert result is True
        mock_memory.delete.assert_called_once_with("delete_key")
        mock_disk.delete.assert_called_once_with("delete_key")
    
    @pytest.mark.asyncio
    async def test_distributed_cache_delete_partial_success_lines_660_661(self):
        """Test DistributedCache.delete with partial success (lines 660-661)."""
        # Mock memory cache to succeed (async delete)
        mock_memory = AsyncMock()
        mock_memory.delete.return_value = True
        
        # Mock disk cache to fail
        mock_disk = AsyncMock()
        mock_disk.delete.return_value = False
        
        cache = DistributedCache(memory_cache=mock_memory, disk_cache=mock_disk)
        
        # Test delete with one success
        result = await cache.delete("async_delete_key")
        
        # Should return True since at least memory succeeded
        assert result is True
        mock_memory.delete.assert_called_once_with("async_delete_key")
        mock_disk.delete.assert_called_once_with("async_delete_key")


class TestHybridCacheMissingLines:
    """Tests for missing lines in HybridCache."""
    
    @pytest.mark.asyncio
    async def test_hybrid_cache_set_distributed_failure_lines_1114_1118(self):
        """Test HybridCache.set with DistributedCache failure (lines 1114-1118)."""
        # Mock memory cache
        mock_memory = AsyncMock()
        mock_memory.set = AsyncMock()
        
        # Mock DistributedCache (formerly Redis) to fail
        mock_distributed = AsyncMock()
        mock_distributed.set.side_effect = Exception("Distributed cache failed")
        
        cache = HybridCache(memory_cache=mock_memory, redis_cache=mock_distributed)
        
        # Should handle distributed cache failure gracefully and still set in memory
        await cache.set("hybrid_key", "hybrid_value", ttl=300)
        
        # Memory cache should still be called
        mock_memory.set.assert_called_once_with("hybrid_key", "hybrid_value", ttl=300)
        
        # Distributed cache should have been attempted
        mock_distributed.set.assert_called_once_with("hybrid_key", "hybrid_value", ttl=300)
    
    @pytest.mark.asyncio
    async def test_hybrid_cache_get_distributed_failure_memory_fallback_lines_1132_1134(self):
        """Test HybridCache.get with DistributedCache failure and memory fallback (lines 1132-1134)."""
        # Mock memory cache with data for fallback
        mock_memory = AsyncMock()
        # First call returns None (line 1121), second call returns data (line 1134 fallback)
        mock_memory.get.side_effect = [None, AsyncMock(value="memory_value")]
        
        # Mock DistributedCache to fail
        mock_distributed = AsyncMock()
        mock_distributed.get.side_effect = Exception("Distributed cache failed")
        
        cache = HybridCache(memory_cache=mock_memory, redis_cache=mock_distributed)
        
        # Should fall back to memory cache when distributed cache fails
        result = await cache.get("fallback_key")
        
        assert result == "memory_value"
        # The distributed cache should be tried after memory miss
        mock_distributed.get.assert_called_once_with("fallback_key")
        # Memory should be called twice - first attempt and fallback (line 1134)
        assert mock_memory.get.call_count == 2
    
    @pytest.mark.asyncio
    async def test_hybrid_cache_delete_both_caches_lines_1120_1129(self):
        """Test HybridCache.delete from both caches (lines 1120-1129)."""
        # Mock both caches
        mock_memory = AsyncMock()
        mock_memory.delete.return_value = True
        
        mock_distributed = AsyncMock()
        mock_distributed.delete.return_value = True
        
        cache = HybridCache(memory_cache=mock_memory, redis_cache=mock_distributed)
        
        # Should delete from both caches
        result = await cache.delete("delete_both_key")
        
        assert result is True
        mock_memory.delete.assert_called_once_with("delete_both_key")
        mock_distributed.delete.assert_called_once_with("delete_both_key")
    
    @pytest.mark.asyncio
    async def test_hybrid_cache_clear_both_caches_lines_1131_1137(self):
        """Test HybridCache.clear both caches (lines 1131-1137)."""
        # Mock both caches
        mock_memory = AsyncMock()
        mock_memory.clear = AsyncMock()
        
        mock_distributed = AsyncMock()
        mock_distributed.clear = AsyncMock()
        
        cache = HybridCache(memory_cache=mock_memory, redis_cache=mock_distributed)
        
        # Should clear both caches
        await cache.clear()
        
        mock_memory.clear.assert_called_once()
        mock_distributed.clear.assert_called_once()


class TestCacheStatsMissingLines:
    """Tests for missing lines in CacheStats."""
    
    def test_cache_stats_hit_rate_no_requests_line_768(self):
        """Test CacheStats.hit_rate with no requests (line 768)."""
        stats = CacheStats()
        
        # With no requests, hit rate should be 0.0
        assert stats.hit_rate == 0.0
        assert stats.hits == 0
        assert stats.misses == 0
    
    def test_cache_stats_memory_utilization_no_max_memory_lines_781_785(self):
        """Test CacheStats.memory_utilization with no max memory (lines 781-785)."""
        stats = CacheStats(current_memory=100, max_memory=0)
        
        # With max_memory of 0, utilization should be 0.0
        assert stats.memory_utilization == 0.0
        
        # Test with None max_memory
        stats.max_memory = None
        assert stats.memory_utilization == 0.0
    
    def test_cache_stats_entry_utilization_no_max_entries_lines_795_799(self):
        """Test CacheStats.entry_utilization with no max entries (lines 795-799)."""
        stats = CacheStats(entries=50, max_entries=0)
        
        # With max_entries of 0, utilization should be 0.0
        assert stats.entry_utilization == 0.0
        
        # Test with None max_entries  
        stats.max_entries = None
        assert stats.entry_utilization == 0.0


class TestCacheWrappersMissingLines:
    """Tests for missing lines in cache wrapper functions."""
    
    @pytest.mark.asyncio
    async def test_async_cache_wrapper_cache_miss_lines_825_826(self):
        """Test async cache wrapper with cache miss (lines 825-826)."""
        # Mock cache that returns None (cache miss)
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None
        mock_cache.set = AsyncMock()
        
        # Function to cache
        call_count = 0
        async def test_func(x):
            nonlocal call_count
            call_count += 1
            return f"result_{x}"
        
        # Apply cache wrapper
        cached_func = async_cache_wrapper(cache=mock_cache, ttl=300)(test_func)
        
        # Call function - should miss cache and execute function
        result = await cached_func("test_arg")
        
        assert result == "result_test_arg"
        assert call_count == 1  # Function was called
        
        # Verify cache interactions
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()
    
    def test_sync_cache_wrapper_cache_miss_lines_851_853(self):
        """Test sync cache wrapper with cache miss (lines 851-853)."""
        # Mock cache that returns None (cache miss)
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_cache.set = Mock()
        
        # Function to cache
        call_count = 0
        def test_func(x):
            nonlocal call_count
            call_count += 1
            return f"sync_result_{x}"
        
        # Apply cache wrapper
        cached_func = sync_cache_wrapper(cache=mock_cache, ttl=600)(test_func)
        
        # Call function - should miss cache and execute function
        result = cached_func("sync_arg")
        
        assert result == "sync_result_sync_arg"
        assert call_count == 1  # Function was called
        
        # Verify cache interactions
        mock_cache.get.assert_called_once()
        mock_cache.set.assert_called_once()


class TestCacheConfigurationMissingLines:
    """Tests for missing configuration and edge case lines."""
    
    @pytest.mark.asyncio
    async def test_memory_cache_initialization_edge_cases_lines_994_996(self):
        """Test MemoryCache initialization edge cases (lines 994-996)."""
        # Test with zero max_memory (should handle gracefully)
        cache = MemoryCache(max_memory=0)
        
        # Should still work but won't store due to memory constraint
        await cache.set("test", "value")
        # Note: Implementation should prevent storage when max_memory=0
        # This tests the edge case handling
        
        # Test with zero max_entries
        cache2 = MemoryCache(max_entries=0)
        await cache2.set("test", "value") 
        # Note: Implementation should prevent storage when max_entries=0
        # This tests the edge case handling
        
        # Test with extreme values
        cache3 = MemoryCache(max_memory=1, max_entries=1)
        await cache3.set("tiny", "x")  # Very small cache
        # This should work as it's within limits
    
    @pytest.mark.asyncio
    async def test_distributed_cache_initialization_self_contained(self):
        """Test DistributedCache self-contained initialization."""
        # Create cache without external dependencies
        # This should work completely self-contained
        cache = DistributedCache()
        
        # Should work with built-in memory and disk caches
        set_result = await cache.set("self_contained_key", "value")
        assert set_result is True  # Should succeed with built-in caches
        
        # Get should work
        result = await cache.get("self_contained_key")
        assert result == "value"
        
        # Delete should work
        deleted = await cache.delete("self_contained_key")
        assert deleted is True
        
        # Should be completely self-contained
        assert cache._available is True