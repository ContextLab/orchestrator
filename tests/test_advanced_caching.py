"""
Tests for advanced caching functionality.
Tests intelligent cache warming, prefetching, and advanced eviction policies.
"""

import pytest
import time
import asyncio
from unittest.mock import patch, MagicMock
from orchestrator.models.advanced_caching import (
    CacheEvictionPolicy, CacheEntry, AdvancedCache,
    ModelSelectionCache, CapabilityAnalysisCache, 
    PredictiveCacheWarmer, CacheManager, background_cache_maintenance
)
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.core.model import Model, ModelCapabilities, ModelRequirements, ModelMetrics, ModelCost


class TestModel(Model):
    """Test model implementation for testing."""
    
    async def generate(self, prompt: str, **kwargs):
        return "test response"
    
    async def generate_structured(self, prompt: str, **kwargs):
        return {"response": "test"}
    
    async def health_check(self):
        return True
    
    async def estimate_cost(self, prompt: str, max_tokens=None):
        return 0.001


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return TestModel(
        name="test-model",
        provider="test",
        capabilities=ModelCapabilities(
            supported_tasks=["generate", "chat"],
            context_window=4096,
            supports_function_calling=True,
            supports_structured_output=True,
            languages=["en", "es"],
            max_tokens=2048
        ),
        requirements=ModelRequirements(
            memory_gb=4.0,
            cpu_cores=2,
            min_python_version="3.8"
        ),
        metrics=ModelMetrics(
            latency_p50=1.5,
            throughput=20.0,
            accuracy=0.85,
            cost_per_token=0.001
        ),
        cost=ModelCost(
            input_cost_per_1k_tokens=0.001,
            output_cost_per_1k_tokens=0.002
        )
    )


class TestCacheEntry:
    """Test cache entry functionality."""
    
    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(key="test", value="data", ttl=300.0)
        
        assert entry.key == "test"
        assert entry.value == "data"
        assert entry.ttl == 300.0
        assert entry.access_count == 0
        assert entry.priority == 0
        assert entry.creation_time > 0
        assert entry.last_access_time > 0
    
    def test_cache_entry_access(self):
        """Test cache entry access tracking."""
        entry = CacheEntry(key="test", value="data")
        
        initial_access_count = entry.access_count
        initial_access_time = entry.last_access_time
        
        time.sleep(0.01)  # Small delay
        entry.access()
        
        assert entry.access_count == initial_access_count + 1
        assert entry.last_access_time > initial_access_time
    
    def test_cache_entry_expiration(self):
        """Test cache entry TTL expiration."""
        # Create entry with short TTL
        entry = CacheEntry(key="test", value="data", ttl=0.01)
        
        assert not entry.is_expired()
        
        time.sleep(0.02)  # Wait for expiration
        
        assert entry.is_expired()
    
    def test_cache_entry_age_calculation(self):
        """Test age and last access calculations."""
        entry = CacheEntry(key="test", value="data")
        
        # Age should be small initially
        assert entry.age_seconds() < 1.0
        assert entry.seconds_since_last_access() < 1.0
        
        time.sleep(0.01)
        
        # Age should increase
        assert entry.age_seconds() > 0.01


class TestAdvancedCache:
    """Test advanced cache functionality."""
    
    def test_cache_initialization(self):
        """Test cache initialization with different policies."""
        cache = AdvancedCache(
            max_size=100,
            eviction_policy=CacheEvictionPolicy.LRU,
            default_ttl=300.0
        )
        
        assert cache.max_size == 100
        assert cache.eviction_policy == CacheEvictionPolicy.LRU
        assert cache.default_ttl == 300.0
        assert cache.size() == 0
        assert cache.size_bytes() == 0
    
    def test_cache_put_and_get(self):
        """Test basic cache put and get operations."""
        cache = AdvancedCache(max_size=10)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.size() == 1
        
        # Non-existent key
        assert cache.get("key2") is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = AdvancedCache(max_size=2, eviction_policy=CacheEvictionPolicy.LRU)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it more recently used
        cache.get("key1")
        
        # Add key3, should evict key2 (least recently used)
        cache.put("key3", "value3")
        
        assert cache.size() == 2
        assert cache.get("key1") == "value1"  # Should still exist
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") == "value3"  # Should exist
    
    def test_cache_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = AdvancedCache(max_size=10, default_ttl=0.01)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        time.sleep(0.02)  # Wait for expiration
        
        assert cache.get("key1") is None  # Should be expired
        assert cache.size() == 0  # Should be removed
    
    def test_cache_priority_handling(self):
        """Test cache entry priority handling."""
        cache = AdvancedCache(max_size=2)
        
        # Add low priority entry
        cache.put("low", "value1", priority=0)
        
        # Add high priority entry
        cache.put("high", "value2", priority=10)
        
        # Add another entry to trigger eviction
        cache.put("new", "value3", priority=5)
        
        assert cache.size() == 2
        # High priority should remain
        assert cache.get("high") == "value2"
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = AdvancedCache(max_size=10)
        
        # Test miss
        cache.get("nonexistent")
        
        # Test hit
        cache.put("key1", "value1")
        cache.get("key1")
        
        stats = cache.statistics()
        
        assert stats['entries'] == 1
        assert stats['max_size'] == 10
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = AdvancedCache(max_size=10, default_ttl=0.01)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        time.sleep(0.02)  # Wait for expiration
        
        cleaned = cache.cleanup_expired()
        
        assert cleaned == 2
        assert cache.size() == 0


class TestModelSelectionCache:
    """Test model selection cache."""
    
    def test_selection_cache_basic_operations(self):
        """Test basic cache operations for model selection."""
        cache = ModelSelectionCache(max_size=10, default_ttl=300.0)
        
        # Cache a selection
        cache.cache_selection("criteria1", "model1")
        
        # Retrieve selection
        result = cache.get_selection("criteria1")
        assert result == "model1"
        
        # Non-existent criteria
        assert cache.get_selection("criteria2") is None
    
    def test_selection_cache_pattern_caching(self):
        """Test pattern caching functionality."""
        cache = ModelSelectionCache()
        
        # Cache a pattern
        cache.cache_pattern("code_task", ["model1", "model2"])
        
        # Retrieve pattern suggestions
        suggestions = cache.get_pattern_suggestions("code_task")
        assert suggestions == ["model1", "model2"]
        
        # Non-existent pattern
        assert cache.get_pattern_suggestions("unknown") is None
    
    def test_selection_cache_invalidation(self):
        """Test cache invalidation for specific models."""
        cache = ModelSelectionCache()
        
        # Cache selections
        cache.cache_selection("criteria1", "model1")
        cache.cache_selection("criteria2", "model2")
        cache.cache_selection("criteria3", "model1")  # Same model
        
        # Cache pattern
        cache.cache_pattern("pattern1", ["model1", "model3"])
        
        # Invalidate model1
        invalidated = cache.invalidate_model("model1")
        
        assert invalidated >= 2  # Should invalidate at least the selections
        assert cache.get_selection("criteria1") is None
        assert cache.get_selection("criteria2") == "model2"  # Should remain
        assert cache.get_selection("criteria3") is None


class TestCapabilityAnalysisCache:
    """Test capability analysis cache."""
    
    def test_capability_cache_basic_operations(self):
        """Test basic cache operations for capability analysis."""
        cache = CapabilityAnalysisCache(max_size=10, default_ttl=3600.0)
        
        analysis = {"accuracy": 0.9, "capabilities": ["text", "code"]}
        
        # Cache analysis
        cache.cache_analysis("model1", analysis)
        
        # Retrieve analysis
        result = cache.get_analysis("model1")
        assert result == analysis
        
        # Non-existent model
        assert cache.get_analysis("model2") is None
    
    def test_capability_cache_invalidation(self):
        """Test cache invalidation for capability analysis."""
        cache = CapabilityAnalysisCache()
        
        analysis = {"accuracy": 0.9}
        cache.cache_analysis("model1", analysis)
        
        # Invalidate
        invalidated = cache.invalidate_model("model1")
        
        assert invalidated is True
        assert cache.get_analysis("model1") is None


class TestPredictiveCacheWarmer:
    """Test predictive cache warming."""
    
    def test_cache_warmer_initialization(self):
        """Test cache warmer initialization."""
        selection_cache = ModelSelectionCache()
        capability_cache = CapabilityAnalysisCache()
        
        warmer = PredictiveCacheWarmer(selection_cache, capability_cache)
        
        assert warmer.selection_cache is selection_cache
        assert warmer.capability_cache is capability_cache
        assert len(warmer._usage_patterns) == 0
    
    def test_pattern_recording(self):
        """Test recording of usage patterns."""
        selection_cache = ModelSelectionCache()
        capability_cache = CapabilityAnalysisCache()
        warmer = PredictiveCacheWarmer(selection_cache, capability_cache)
        
        # Record some selections
        warmer.record_selection("criteria1", "model1")
        warmer.record_selection("criteria1", "model1")  # Same pattern
        warmer.record_selection("criteria2", "model2")
        
        predictions = warmer.get_predictions()
        
        assert "most_common_patterns" in predictions
        assert predictions["most_common_patterns"]["criteria1"] == 2
        assert predictions["most_common_patterns"]["criteria2"] == 1
    
    def test_predictive_warming(self):
        """Test predictive cache warming."""
        selection_cache = ModelSelectionCache()
        capability_cache = CapabilityAnalysisCache()
        warmer = PredictiveCacheWarmer(selection_cache, capability_cache)
        
        # Create mock registry
        registry = MagicMock()
        registry.models = {"test:model1": MagicMock(), "test:model2": MagicMock()}
        registry.get_model.return_value = MagicMock()
        registry.detect_model_capabilities.return_value = {"test": "analysis"}
        
        # Record patterns
        warmer.record_selection("criteria1", "test:model1")
        
        # Warm cache
        result = warmer.warm_cache_predictively(registry, None)
        
        assert "selections" in result
        assert "capabilities" in result


class TestCacheManager:
    """Test cache manager functionality."""
    
    def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        manager = CacheManager()
        
        assert manager.selection_cache is not None
        assert manager.capability_cache is not None
        assert manager.cache_warmer is not None
        assert manager._warming_enabled is True
    
    def test_cache_manager_selection_operations(self):
        """Test cache manager selection operations."""
        manager = CacheManager()
        
        # Cache and retrieve selection
        manager.cache_selection("criteria1", "model1")
        result = manager.get_cached_selection("criteria1")
        
        assert result == "model1"
    
    def test_cache_manager_capability_operations(self):
        """Test cache manager capability operations."""
        manager = CacheManager()
        
        analysis = {"accuracy": 0.9}
        
        # Cache and retrieve capability analysis
        manager.cache_capability_analysis("model1", analysis)
        result = manager.get_cached_capability_analysis("model1")
        
        assert result == analysis
    
    def test_cache_manager_invalidation(self):
        """Test cache manager model invalidation."""
        manager = CacheManager()
        
        # Cache data
        manager.cache_selection("criteria1", "model1")
        manager.cache_capability_analysis("model1", {"test": "data"})
        
        # Invalidate
        result = manager.invalidate_model("model1")
        
        assert "selection_entries" in result
        assert "capability_entries" in result
    
    def test_cache_manager_statistics(self):
        """Test cache manager statistics."""
        manager = CacheManager()
        
        # Add some data
        manager.cache_selection("criteria1", "model1")
        manager.cache_capability_analysis("model1", {"test": "data"})
        
        stats = manager.get_comprehensive_statistics()
        
        assert "selection_cache" in stats
        assert "capability_cache" in stats
        assert "predictions" in stats
        assert "warming_enabled" in stats
    
    def test_cache_manager_warming_control(self):
        """Test cache warming enable/disable."""
        manager = CacheManager()
        
        assert manager._warming_enabled is True
        
        manager.disable_warming()
        assert manager._warming_enabled is False
        
        manager.enable_warming()
        assert manager._warming_enabled is True


class TestModelRegistryCachingIntegration:
    """Test caching integration with ModelRegistry."""
    
    def test_registry_with_advanced_caching(self, sample_model):
        """Test ModelRegistry with advanced caching enabled."""
        registry = ModelRegistry(enable_advanced_caching=True)
        
        assert registry.enable_advanced_caching is True
        assert registry._advanced_caching_enabled is True
        assert registry.cache_manager is not None
        
        # Register model
        registry.register_model(sample_model)
        
        # Should be able to get caching statistics
        stats = registry.get_caching_statistics()
        assert "selection_cache" in stats
        assert "capability_cache" in stats
    
    def test_registry_without_advanced_caching(self, sample_model):
        """Test ModelRegistry with advanced caching disabled."""
        registry = ModelRegistry(enable_advanced_caching=False)
        
        assert registry.enable_advanced_caching is False
        assert registry._advanced_caching_enabled is False
        assert registry.cache_manager is None
        
        registry.register_model(sample_model)
        
        # Caching methods should return error
        stats = registry.get_caching_statistics()
        assert "error" in stats
        
        warming = registry.warm_caches()
        assert "error" in warming
    
    def test_cache_warming_control(self):
        """Test cache warming enable/disable controls."""
        registry = ModelRegistry(enable_advanced_caching=True)
        
        # Should be enabled by default
        assert registry.cache_manager._warming_enabled is True
        
        registry.disable_cache_warming()
        assert registry.cache_manager._warming_enabled is False
        
        registry.enable_cache_warming()
        assert registry.cache_manager._warming_enabled is True
    
    def test_capability_analysis_caching(self, sample_model):
        """Test capability analysis with caching."""
        registry = ModelRegistry(enable_advanced_caching=True)
        registry.register_model(sample_model)
        
        # First call should compute and cache
        analysis1 = registry.detect_model_capabilities(sample_model)
        
        # Second call should use cache
        analysis2 = registry.detect_model_capabilities(sample_model)
        
        assert analysis1 == analysis2
        assert "basic_capabilities" in analysis1
        assert "advanced_capabilities" in analysis1
        assert "suitability_scores" in analysis1
    
    @pytest.mark.asyncio
    async def test_model_selection_caching(self, sample_model):
        """Test model selection with caching."""
        registry = ModelRegistry(enable_advanced_caching=True)
        sample_model._expertise = ["general", "chat"]
        sample_model._size_billions = 7.0
        registry.register_model(sample_model)
        
        requirements = {
            "expertise": "medium",
            "min_size": "1B",
            "max_size": "10B"
        }
        
        # First selection should compute and cache
        with patch.object(sample_model, 'health_check', return_value=True):
            model1 = await registry.select_model(requirements)
        
        # Second selection should use cache (verify by checking cache statistics)
        stats_before = registry.get_caching_statistics()
        
        with patch.object(sample_model, 'health_check', return_value=True):
            model2 = await registry.select_model(requirements)
        
        assert model1 == model2
        
        # Cache should have recorded the selection
        stats_after = registry.get_caching_statistics()
        assert stats_after["selection_cache"]["entries"] >= stats_before["selection_cache"]["entries"]
    
    def test_background_maintenance_control(self):
        """Test background maintenance task control."""
        registry = ModelRegistry(enable_advanced_caching=True)
        
        # Start maintenance
        registry.start_background_maintenance()
        assert registry._background_maintenance_task is not None
        
        # Stop maintenance
        registry.stop_background_maintenance()
        assert registry._background_maintenance_task is None
    
    def test_cache_clearing(self, sample_model):
        """Test clearing all caches."""
        registry = ModelRegistry(enable_advanced_caching=True)
        registry.register_model(sample_model)
        
        # Add some cached data
        analysis = registry.detect_model_capabilities(sample_model)
        assert analysis is not None
        
        # Clear caches
        registry.clear_all_caches()
        
        # Verify caches are cleared
        stats = registry.get_caching_statistics()
        assert stats["selection_cache"]["entries"] == 0
        assert stats["capability_cache"]["entries"] == 0


class TestAdvancedCacheEvictionPolicies:
    """Test different cache eviction policies."""
    
    def test_fifo_eviction(self):
        """Test First-In-First-Out eviction."""
        cache = AdvancedCache(max_size=2, eviction_policy=CacheEvictionPolicy.FIFO)
        
        cache.put("first", "value1")
        time.sleep(0.01)
        cache.put("second", "value2")
        time.sleep(0.01)
        cache.put("third", "value3")  # Should evict "first"
        
        assert cache.get("first") is None
        assert cache.get("second") == "value2"
        assert cache.get("third") == "value3"
    
    def test_lfu_eviction(self):
        """Test Least-Frequently-Used eviction."""
        cache = AdvancedCache(max_size=2, eviction_policy=CacheEvictionPolicy.LFU)
        
        cache.put("freq1", "value1")
        cache.put("freq2", "value2")
        
        # Access freq1 multiple times
        cache.get("freq1")
        cache.get("freq1")
        cache.get("freq2")  # Less frequent
        
        cache.put("new", "value3")  # Should evict freq2 (less frequent)
        
        assert cache.get("freq1") == "value1"
        assert cache.get("freq2") is None
        assert cache.get("new") == "value3"
    
    def test_adaptive_eviction(self):
        """Test Adaptive Replacement Cache eviction."""
        cache = AdvancedCache(max_size=4, eviction_policy=CacheEvictionPolicy.ADAPTIVE)
        
        # This is a simplified test as ARC is complex
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access to establish patterns
        cache.get("key1")
        cache.get("key1")  # Make key1 frequent
        
        cache.put("key3", "value3")
        cache.put("key4", "value4")
        
        # Should still have key1 due to frequency
        assert cache.get("key1") == "value1"


@pytest.mark.asyncio
async def test_background_cache_maintenance():
    """Test background cache maintenance task."""
    cache_manager = CacheManager()
    
    # Add some data
    cache_manager.cache_selection("test", "model1")
    
    # Run one cycle of maintenance (with short interval for testing)
    task = asyncio.create_task(background_cache_maintenance(cache_manager, interval=0.01))
    
    # Let it run briefly
    await asyncio.sleep(0.02)
    
    # Cancel the task
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        pass  # Expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])