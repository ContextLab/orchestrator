"""Tests for model response caching system."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock

from src.orchestrator.models.optimization.caching import (
    ModelResponseCache,
    CacheStats,
    CacheEntry,
)
from src.orchestrator.models.selection.strategies import SelectionResult
from src.orchestrator.core.model import Model, ModelCapabilities, ModelCost


class MockModel(Model):
    """Mock model for testing."""
    
    def __init__(self, name: str, provider: str):
        super().__init__(
            name=name,
            provider=provider,
            capabilities=ModelCapabilities(),
            cost=ModelCost(is_free=True),
        )
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = None, **kwargs):
        return f"Response from {self.name}"
    
    async def generate_structured(self, prompt: str, schema: dict, temperature: float = 0.7, **kwargs):
        return {"result": f"Structured from {self.name}"}
    
    async def health_check(self) -> bool:
        return True
    
    async def estimate_cost(self, prompt: str, max_tokens: int = None) -> float:
        return 0.001


class TestCacheEntry:
    """Test CacheEntry functionality."""
    
    def test_initialization(self):
        """Test cache entry initialization."""
        entry = CacheEntry(
            value="test_value",
            timestamp=1234567890.0,
            ttl=3600.0,
        )
        
        assert entry.value == "test_value"
        assert entry.timestamp == 1234567890.0
        assert entry.ttl == 3600.0
        assert entry.access_count == 0
    
    def test_expiration_check(self):
        """Test TTL expiration checking."""
        # Create entry that should be expired
        old_time = time.time() - 7200  # 2 hours ago
        entry = CacheEntry(
            value="test_value",
            timestamp=old_time,
            ttl=3600.0,  # 1 hour TTL
        )
        
        assert entry.is_expired() is True
        
        # Create entry that should not be expired
        recent_time = time.time() - 1800  # 30 minutes ago
        entry_not_expired = CacheEntry(
            value="test_value",
            timestamp=recent_time,
            ttl=3600.0,  # 1 hour TTL
        )
        
        assert entry_not_expired.is_expired() is False
    
    def test_no_ttl_expiration(self):
        """Test entries with no TTL never expire."""
        entry = CacheEntry(
            value="test_value",
            timestamp=time.time() - 86400,  # 1 day ago
            ttl=None,
        )
        
        assert entry.is_expired() is False
    
    def test_access_tracking(self):
        """Test access count and timestamp tracking."""
        entry = CacheEntry(value="test_value", timestamp=time.time())
        
        initial_last_accessed = entry.last_accessed
        initial_count = entry.access_count
        
        # Wait a bit to ensure timestamp difference
        time.sleep(0.01)
        entry.access()
        
        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_last_accessed


class TestCacheStats:
    """Test CacheStats functionality."""
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=8, misses=2)
        assert stats.hit_rate == 0.8
        
        # Test with no requests
        empty_stats = CacheStats()
        assert empty_stats.hit_rate == 0.0
    
    def test_fill_rate_calculation(self):
        """Test cache fill rate calculation."""
        stats = CacheStats(total_size=75, max_size=100)
        assert stats.fill_rate == 0.75
        
        # Test with max_size 0
        stats_no_max = CacheStats(total_size=50, max_size=0)
        assert stats_no_max.fill_rate == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = CacheStats(
            hits=10,
            misses=5,
            evictions=2,
            total_size=50,
            max_size=100,
        )
        
        result = stats.to_dict()
        
        expected_keys = ["hits", "misses", "evictions", "total_size", "max_size", "hit_rate", "fill_rate"]
        for key in expected_keys:
            assert key in result
        
        assert result["hit_rate"] == 10 / 15  # 10 hits out of 15 total
        assert result["fill_rate"] == 0.5  # 50 out of 100


class TestModelResponseCache:
    """Test ModelResponseCache functionality."""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance for testing."""
        return ModelResponseCache(
            max_size=10,
            default_ttl=3600.0,  # 1 hour
            max_memory_mb=1,  # 1MB limit
        )
    
    def test_initialization(self, cache):
        """Test cache initialization."""
        assert cache.max_size == 10
        assert cache.default_ttl == 3600.0
        assert cache.max_memory_bytes == 1024 * 1024
        assert len(cache._cache) == 0
    
    def test_cache_key_generation(self, cache):
        """Test cache key generation."""
        key1 = cache.generate_cache_key(
            prompt="Hello world",
            temperature=0.7,
            max_tokens=100,
        )
        
        key2 = cache.generate_cache_key(
            prompt="Hello world",
            temperature=0.7,
            max_tokens=100,
        )
        
        key3 = cache.generate_cache_key(
            prompt="Different prompt",
            temperature=0.7,
            max_tokens=100,
        )
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different key
        assert key1 != key3
        
        # Keys should be reasonable length (16 chars in current implementation)
        assert len(key1) == 16
    
    @pytest.mark.asyncio
    async def test_basic_caching(self, cache):
        """Test basic cache put/get operations."""
        key = "test_key"
        value = "test_value"
        
        # Should be empty initially
        result = await cache.get(key)
        assert result is None
        
        # Put value in cache
        await cache.put(key, value)
        
        # Should now retrieve value
        result = await cache.get(key)
        assert result == value
        
        # Check stats
        stats = await cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = ModelResponseCache(max_size=10, default_ttl=0.05)  # 50ms TTL
        
        key = "test_key"
        value = "test_value"
        
        await cache.put(key, value)
        
        # Should retrieve immediately
        result = await cache.get(key)
        assert result == value
        
        # Wait for expiration
        await asyncio.sleep(0.1)
        
        # Should be expired now
        result = await cache.get(key)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        for i in range(cache.max_size):
            await cache.put(f"key_{i}", f"value_{i}")
        
        # All keys should be retrievable
        for i in range(cache.max_size):
            result = await cache.get(f"key_{i}")
            assert result == f"value_{i}"
        
        # Add one more item, should evict oldest
        await cache.put("new_key", "new_value")
        
        # First key should be evicted
        result = await cache.get("key_0")
        assert result is None
        
        # New key should be available
        result = await cache.get("new_key")
        assert result == "new_value"
        
        # Check stats
        stats = await cache.get_stats()
        assert stats.evictions >= 1
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache):
        """Test cache invalidation."""
        # Add some items
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("test_key3", "value3")
        
        # Verify items are cached
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"
        assert await cache.get("test_key3") == "value3"
        
        # Pattern-based invalidation
        count = await cache.invalidate("test_")
        assert count == 1
        assert await cache.get("test_key3") is None
        assert await cache.get("key1") == "value1"  # Should still exist
        
        # Full invalidation
        count = await cache.invalidate()
        assert count >= 2  # At least key1 and key2
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
    
    @pytest.mark.asyncio
    async def test_selection_result_caching(self, cache):
        """Test caching of model selection results."""
        model = MockModel("test_model", "test_provider")
        selection_result = SelectionResult(
            model=model,
            provider="test_provider",
            confidence_score=0.95,
            selection_reason="Test selection",
        )
        
        requirements_key = "test_requirements"
        
        # Should be empty initially
        result = await cache.get_cached_selection(requirements_key)
        assert result is None
        
        # Cache selection result
        await cache.cache_selection(requirements_key, selection_result)
        
        # Should retrieve cached result
        result = await cache.get_cached_selection(requirements_key)
        assert result is not None
        assert result.model.name == "test_model"
        assert result.confidence_score == 0.95
    
    @pytest.mark.asyncio
    async def test_memory_management(self):
        """Test memory-based eviction."""
        # Create cache with very small memory limit
        cache = ModelResponseCache(max_size=100, max_memory_mb=0.001)  # ~1KB limit
        
        # Add large values that should trigger memory eviction
        large_value = "x" * 512  # 512 bytes
        
        await cache.put("key1", large_value)
        await cache.put("key2", large_value)
        await cache.put("key3", large_value)  # This should trigger evictions
        
        # First key might be evicted due to memory pressure
        stats = await cache.get_stats()
        assert stats.evictions > 0
    
    @pytest.mark.asyncio
    async def test_size_estimation(self, cache):
        """Test size estimation for different value types."""
        # Test with different types
        test_cases = [
            ("string", "hello world"),
            ("int", 12345),
            ("float", 123.45),
            ("dict", {"key": "value", "number": 42}),
            ("list", [1, 2, 3, "test"]),
        ]
        
        for name, value in test_cases:
            size = cache._estimate_size(value)
            assert size > 0, f"Size estimation failed for {name}"
            assert isinstance(size, int), f"Size should be integer for {name}"
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_entries(self):
        """Test cleanup of expired entries."""
        cache = ModelResponseCache(max_size=10, default_ttl=0.05)  # 50ms TTL
        
        # Add entries
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        
        # Wait for expiration
        await asyncio.sleep(0.1)
        
        # Add new entry, which should trigger cleanup
        await cache.put("key3", "value3")
        
        # Expired entries should be gone
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") == "value3"
    
    @pytest.mark.asyncio
    async def test_cache_info(self, cache):
        """Test cache information retrieval."""
        # Add some items
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        
        info = cache.get_cache_info()
        
        required_keys = [
            "max_size", "current_size", "max_memory_mb", 
            "current_memory_mb", "default_ttl", "stats"
        ]
        
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
        
        assert info["max_size"] == 10
        assert info["current_size"] == 2
        assert isinstance(info["stats"], dict)
    
    @pytest.mark.asyncio
    async def test_cleanup(self, cache):
        """Test cache cleanup."""
        # Add some entries
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        
        # Verify entries exist
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"
        
        # Cleanup
        await cache.cleanup()
        
        # All entries should be cleared
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        
        stats = await cache.get_stats()
        assert stats.total_size == 0
    
    def test_string_representation(self, cache):
        """Test string representation of cache."""
        str_repr = str(cache)
        assert "ModelResponseCache" in str_repr
        assert "0/10" in str_repr  # 0 out of 10 max size
        assert "hit_rate=" in str_repr