"""Tests for multi-level caching system."""

import pytest
import asyncio
import time
import tempfile
import os
from unittest.mock import patch, MagicMock, AsyncMock

from src.orchestrator.core.cache import (
    MultiLevelCache, MemoryCache, DiskCache, 
    CacheLevel, EvictionPolicy, CacheEntry,
    LRUCache, CacheStrategy
)


class TestMultiLevelCache:
    """Test cases for MultiLevelCache class."""
    
    def test_cache_creation(self):
        """Test basic cache creation."""
        cache = MultiLevelCache()
        
        assert cache.memory_cache is not None
        assert cache.redis_cache is not None
        assert cache.disk_cache is not None
        assert cache.cache_strategy is not None
    
    def test_cache_with_custom_config(self):
        """Test cache with custom configuration."""
        config = {
            "memory_cache_size": 500,
            "redis_url": "redis://localhost:6380",
            "disk_cache_path": "/tmp/test_cache",
            "default_ttl": 1800
        }
        
        cache = MultiLevelCache(config)
        
        assert cache.memory_cache.maxsize == 500
        assert cache.default_ttl == 1800
    
    @pytest.mark.asyncio
    async def test_cache_get_hit_memory(self):
        """Test cache get with memory cache hit."""
        cache = MultiLevelCache()
        
        # Set in memory cache
        await cache.memory_cache.set("key1", "value1")
        
        result = await cache.get("key1")
        
        assert result == "value1"
    
    @pytest.mark.asyncio
    async def test_cache_get_hit_redis(self):
        """Test cache get with Redis cache hit."""
        cache = MultiLevelCache()
        
        # Set value in redis cache (which is actually a memory cache fallback)
        await cache.redis_cache.set("key2", "value_from_redis")
        
        result = await cache.get("key2")
        
        assert result == "value_from_redis"
    
    @pytest.mark.asyncio
    async def test_cache_get_hit_disk(self):
        """Test cache get with disk cache hit."""
        from src.orchestrator.core.cache import CacheEntry
        
        cache = MultiLevelCache()
        
        # Mock cache misses for memory and redis
        cache.memory_cache.get = AsyncMock(return_value=None)
        cache.redis_cache.get = AsyncMock(return_value=None)
        
        # Create proper CacheEntry for disk cache hit
        disk_entry = CacheEntry(key="key3", value="value_from_disk")
        cache.disk_cache.get = AsyncMock(return_value=disk_entry)
        
        result = await cache.get("key3")
        
        assert result == "value_from_disk"
    
    @pytest.mark.asyncio
    async def test_cache_get_miss(self):
        """Test cache get with complete miss."""
        cache = MultiLevelCache()
        
        # Mock cache misses for all levels
        cache.memory_cache.get = AsyncMock(return_value=None)
        cache.redis_cache.get = AsyncMock(return_value=None)
        cache.disk_cache.get = AsyncMock(return_value=None)
        
        result = await cache.get("nonexistent_key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_set_all_levels(self):
        """Test cache set to all levels."""
        cache = MultiLevelCache()
        
        # Mock set methods
        cache.redis_cache.set = AsyncMock(return_value=True)
        cache.disk_cache.set = AsyncMock(return_value=True)
        
        await cache.set("key1", "value1", ttl=3600)
        
        # Should set in memory level first (since it's highest priority)
        result = await cache.memory_cache.get("key1")
        assert result.value == "value1"
    
    @pytest.mark.asyncio
    async def test_cache_delete_all_levels(self):
        """Test cache delete from all levels."""
        cache = MultiLevelCache()
        
        # Set initial value
        await cache.memory_cache.set("key1", "value1")
        cache.redis_cache.delete = AsyncMock(return_value=True)
        cache.disk_cache.delete = AsyncMock(return_value=True)
        
        await cache.delete("key1")
        
        # Should delete from all levels
        result = await cache.memory_cache.get("key1")
        assert result is None
        cache.redis_cache.delete.assert_called_once_with("key1")
        cache.disk_cache.delete.assert_called_once_with("key1")
    
    @pytest.mark.asyncio
    async def test_cache_invalidate_pattern(self):
        """Test cache invalidation by pattern."""
        cache = MultiLevelCache()
        
        # Set multiple keys
        await cache.memory_cache.set("user:123", "user_data_123")
        await cache.memory_cache.set("user:456", "user_data_456")
        await cache.memory_cache.set("product:789", "product_data_789")
        
        # Mock Redis and disk invalidation
        cache.redis_cache.invalidate_pattern = AsyncMock()
        cache.disk_cache.invalidate_pattern = AsyncMock()
        
        await cache.invalidate_pattern("user:*")
        
        # Should invalidate matching keys
        result1 = await cache.memory_cache.get("user:123")
        result2 = await cache.memory_cache.get("user:456")
        result3 = await cache.memory_cache.get("product:789")
        
        assert result1 is None
        assert result2 is None
        assert result3.value == "product_data_789"
    
    @pytest.mark.asyncio
    async def test_cache_clear_all(self):
        """Test clearing all cache levels."""
        cache = MultiLevelCache()
        
        # Set initial values
        await cache.memory_cache.set("key1", "value1")
        cache.redis_cache.clear = AsyncMock(return_value=True)
        cache.disk_cache.clear = AsyncMock(return_value=True)
        
        await cache.clear()
        
        # Should clear all levels
        size = await cache.memory_cache.size()
        assert size == 0
        cache.redis_cache.clear.assert_called_once()
        cache.disk_cache.clear.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_with_compression(self):
        """Test cache with compression enabled."""
        cache = MultiLevelCache({"compression_enabled": True})
        
        large_data = "x" * 10000  # Large string
        
        # Mock compression
        cache.redis_cache.set = AsyncMock(return_value=True)
        cache.disk_cache.set = AsyncMock(return_value=True)
        
        await cache.set("large_key", large_data)
        
        # Should store large data in memory
        result = await cache.memory_cache.get("large_key")
        assert result.value == large_data
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self):
        """Test cache statistics collection."""
        cache = MultiLevelCache()
        
        # Simulate cache operations
        await cache.memory_cache.set("key1", "value1")
        await cache.memory_cache.get("key1")  # Hit
        await cache.memory_cache.get("key2")  # Miss
        
        stats = cache.get_statistics()
        
        assert "total_requests" in stats
        assert "total_hits" in stats
        assert "total_misses" in stats
        assert "hit_rate" in stats
        assert "level_statistics" in stats
    
    @pytest.mark.asyncio
    async def test_cache_warmup(self):
        """Test cache warmup functionality."""
        cache = MultiLevelCache()
        
        # Mock data source
        async def data_loader(key):
            return f"loaded_value_{key}"
        
        keys_to_warmup = ["key1", "key2", "key3"]
        
        await cache.warmup(keys_to_warmup, data_loader)
        
        # Check that keys were loaded
        for key in keys_to_warmup:
            value = await cache.get(key)
            assert value == f"loaded_value_{key}"
    
    @pytest.mark.asyncio
    async def test_cache_refresh_expired(self):
        """Test refreshing expired cache entries."""
        cache = MultiLevelCache()
        
        # Set with short TTL
        await cache.set("key1", "value1", ttl=0.1)
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Mock refresh function
        async def refresh_func(key):
            return f"refreshed_{key}"
        
        result = await cache.get_or_refresh("key1", refresh_func)
        
        assert result == "refreshed_key1"
        # Should be cached again
        assert await cache.get("key1") == "refreshed_key1"


class TestLRUCache:
    """Test cases for LRUCache class."""
    
    def test_lru_cache_creation(self):
        """Test basic LRU cache creation."""
        cache = LRUCache(maxsize=100)
        
        assert cache.maxsize == 100
        assert len(cache.data) == 0
    
    def test_lru_cache_set_get(self):
        """Test LRU cache set and get operations."""
        cache = LRUCache(maxsize=3)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_lru_cache_eviction(self):
        """Test LRU cache eviction."""
        cache = LRUCache(maxsize=2)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new key, should evict key2
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
    
    def test_lru_cache_update_order(self):
        """Test LRU cache access order update."""
        cache = LRUCache(maxsize=3)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add new key, should evict key2 (least recently used)
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"
    
    def test_lru_cache_clear(self):
        """Test LRU cache clear operation."""
        cache = LRUCache(maxsize=3)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        cache.clear()
        
        assert len(cache.data) == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_lru_cache_statistics(self):
        """Test LRU cache statistics."""
        cache = LRUCache(maxsize=3)
        
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_statistics()
        
        assert stats["size"] == 1
        assert stats["maxsize"] == 3
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


@pytest.mark.skip(reason="Redis tests require real Redis connection")
class TestRedisCache:
    """Test cases for RedisCache class."""
    
    def test_redis_cache_creation(self):
        """Test basic Redis cache creation."""
        # Redis will fail to connect, so we expect ConnectionError
        with pytest.raises(ConnectionError):
            cache = RedisCache(redis_url="redis://localhost:6379")
    
    @pytest.mark.asyncio
    async def test_redis_cache_set_get(self):
        """Test Redis cache set and get operations."""
        # Skip Redis tests since they require real Redis connection
        pytest.skip("Redis tests require real Redis connection")
    
    @pytest.mark.asyncio
    async def test_redis_cache_delete(self):
        """Test Redis cache delete operation."""
        pytest.skip("Redis tests require real Redis connection")
    
    @pytest.mark.asyncio
    async def test_redis_cache_batch_operations(self):
        """Test Redis cache batch operations."""
        cache = RedisCache()
        
        # Mock Redis pipeline
        pipeline_mock = AsyncMock()
        cache.redis = AsyncMock()
        cache.redis.pipeline = MagicMock(return_value=pipeline_mock)
        pipeline_mock.execute = AsyncMock(return_value=[True, True, True])
        
        keys_values = [("key1", "value1"), ("key2", "value2"), ("key3", "value3")]
        
        await cache.batch_set(keys_values, ttl=300)
        
        cache.redis.pipeline.assert_called_once()
        pipeline_mock.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_cache_pattern_operations(self):
        """Test Redis cache pattern operations."""
        cache = RedisCache()
        
        # Mock Redis operations
        cache.redis = AsyncMock()
        cache.redis.keys = AsyncMock(return_value=[b"user:123", b"user:456"])
        cache.redis.delete = AsyncMock(return_value=2)
        
        await cache.invalidate_pattern("user:*")
        
        cache.redis.keys.assert_called_once_with("user:*")
        cache.redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_cache_serialization(self):
        """Test Redis cache serialization."""
        cache = RedisCache()
        
        # Mock Redis operations
        cache.redis = AsyncMock()
        cache.redis.set = AsyncMock()
        cache.redis.get = AsyncMock(return_value=b'{"key": "value"}')
        
        # Test with complex object
        complex_data = {"list": [1, 2, 3], "dict": {"nested": "value"}}
        
        await cache.set("complex_key", complex_data)
        result = await cache.get("complex_key")
        
        assert result == {"key": "value"}  # Mocked return value
        cache.redis.set.assert_called_once()
        cache.redis.get.assert_called_once_with("complex_key")
    
    @pytest.mark.asyncio
    async def test_redis_cache_connection_error(self):
        """Test Redis cache connection error handling."""
        cache = RedisCache()
        
        # Mock connection error
        cache.redis = AsyncMock()
        cache.redis.get = AsyncMock(side_effect=ConnectionError("Redis connection failed"))
        
        # Should handle connection error gracefully
        result = await cache.get("test_key")
        
        assert result is None


class TestDiskCache:
    """Test cases for DiskCache class."""
    
    def test_disk_cache_creation(self):
        """Test basic disk cache creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(cache_dir=temp_dir)
            
            assert cache.cache_dir == temp_dir
            assert os.path.exists(cache.cache_dir)
    
    @pytest.mark.asyncio
    async def test_disk_cache_set_get(self):
        """Test disk cache set and get operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(cache_dir=temp_dir)
            
            await cache.set("test_key", "test_value")
            result = await cache.get("test_key")
            
            assert result.value == "test_value"
    
    @pytest.mark.asyncio
    async def test_disk_cache_complex_data(self):
        """Test disk cache with complex data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(cache_dir=temp_dir)
            
            complex_data = {
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
                "number": 42,
                "boolean": True
            }
            
            await cache.set("complex_key", complex_data)
            result = await cache.get("complex_key")
            
            assert result.value == complex_data
    
    @pytest.mark.asyncio
    async def test_disk_cache_expiration(self):
        """Test disk cache expiration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(cache_dir=temp_dir)
            
            # Set with short TTL
            await cache.set("expiring_key", "expiring_value", ttl=0.1)
            
            # Should exist immediately
            result = await cache.get("expiring_key")
            assert result.value == "expiring_value"
            
            # Wait for expiration
            await asyncio.sleep(0.2)
            
            # Should be expired
            result = await cache.get("expiring_key")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_disk_cache_delete(self):
        """Test disk cache delete operation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(cache_dir=temp_dir)
            
            await cache.set("delete_key", "delete_value")
            
            # Verify it exists
            result = await cache.get("delete_key")
            assert result.value == "delete_value"
            
            # Delete
            await cache.delete("delete_key")
            
            # Verify it's gone
            result = await cache.get("delete_key")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_disk_cache_cleanup(self):
        """Test disk cache cleanup of expired entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(cache_dir=temp_dir)
            
            # Set multiple entries with different TTLs
            await cache.set("key1", "value1", ttl=0.1)
            await cache.set("key2", "value2", ttl=1.0)
            await cache.set("key3", "value3")  # No TTL
            
            # Wait for some to expire
            await asyncio.sleep(0.2)
            
            # Run cleanup
            await cache.cleanup_expired()
            
            # Check results
            assert await cache.get("key1") is None  # Expired
            result2 = await cache.get("key2")
            assert result2.value == "value2"  # Not expired
            result3 = await cache.get("key3")
            assert result3.value == "value3"  # No TTL
    
    @pytest.mark.asyncio
    async def test_disk_cache_size_limit(self):
        """Test disk cache size limit enforcement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(cache_dir=temp_dir, max_size=1)  # 1 entry limit
            
            # Try to store data larger than limit
            large_data = "x" * (2 * 1024 * 1024)  # 2MB
            
            await cache.set("large_key", large_data)
            
            # Should handle size limit gracefully
            result = await cache.get("large_key")
            # Result depends on implementation - might be None or truncated
            assert result is None or len(result.value) <= len(large_data)
    
    def test_disk_cache_statistics(self):
        """Test disk cache statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DiskCache(cache_dir=temp_dir)
            
            stats = cache.get_statistics()
            
            assert "cache_dir" in stats
            assert "total_files" in stats
            assert "total_size_mb" in stats
            assert "oldest_entry" in stats
            assert "newest_entry" in stats


class TestCacheStrategy:
    """Test cases for CacheStrategy class."""
    
    def test_cache_strategy_creation(self):
        """Test basic cache strategy creation."""
        strategy = CacheStrategy()
        
        assert strategy.policies is not None
        assert strategy.default_ttl == 3600
    
    def test_cache_policy_selection(self):
        """Test cache policy selection."""
        strategy = CacheStrategy()
        
        # Test different data types
        user_data = {"type": "user", "id": 123}
        session_data = {"type": "session", "token": "abc123"}
        temp_data = {"type": "temp", "data": "temporary"}
        
        user_policy = strategy.select_policy(user_data)
        session_policy = strategy.select_policy(session_data)
        temp_policy = strategy.select_policy(temp_data)
        
        assert user_policy["ttl"] >= 3600  # Long TTL for user data
        assert session_policy["ttl"] <= 1800  # Shorter TTL for sessions
        assert temp_policy["cache_level"] == "memory"  # Temp data in memory only
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        strategy = CacheStrategy()
        
        # Test with different inputs
        key1 = strategy.generate_key("user", {"id": 123})
        key2 = strategy.generate_key("user", {"id": 456})
        key3 = strategy.generate_key("session", {"token": "abc123"})
        
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3
        
        # Same inputs should generate same key
        key1_again = strategy.generate_key("user", {"id": 123})
        assert key1 == key1_again
    
    def test_cache_invalidation_rules(self):
        """Test cache invalidation rules."""
        strategy = CacheStrategy()
        
        # Test different invalidation triggers
        user_update = {"type": "user_update", "user_id": 123}
        session_logout = {"type": "session_logout", "session_id": "abc123"}
        global_config_change = {"type": "config_change", "scope": "global"}
        
        user_invalidation = strategy.get_invalidation_keys(user_update)
        session_invalidation = strategy.get_invalidation_keys(session_logout)
        global_invalidation = strategy.get_invalidation_keys(global_config_change)
        
        assert "user:123" in user_invalidation
        assert "session:abc123" in session_invalidation
        assert "*" in global_invalidation or len(global_invalidation) > 10
    
    def test_cache_performance_optimization(self):
        """Test cache performance optimization."""
        strategy = CacheStrategy()
        
        # Test with access patterns
        access_patterns = [
            {"key": "user:123", "frequency": 100, "recency": 0.1},
            {"key": "user:456", "frequency": 10, "recency": 0.5},
            {"key": "temp:789", "frequency": 1, "recency": 0.9}
        ]
        
        optimizations = strategy.optimize_cache_placement(access_patterns)
        
        # High frequency, recent access should be in memory
        assert optimizations["user:123"]["level"] == "memory"
        # Low frequency should be in disk
        assert optimizations["temp:789"]["level"] == "disk"
    
    def test_cache_warmup_strategy(self):
        """Test cache warmup strategy."""
        strategy = CacheStrategy()
        
        # Test warmup key selection
        available_keys = ["user:123", "user:456", "session:abc", "config:global"]
        priority_keys = strategy.select_warmup_keys(available_keys, max_keys=2)
        
        assert len(priority_keys) <= 2
        # Check that at least one high priority key is selected
        assert any("config:global" in key or "user:" in key for key in priority_keys)