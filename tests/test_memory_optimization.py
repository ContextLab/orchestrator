"""
Tests for memory optimization functionality.
Tests memory-efficient data structures and memory monitoring capabilities.
"""

import pytest
import gc
import time
import threading
from unittest.mock import patch, MagicMock
from src.orchestrator.models.memory_optimization import (
    MemoryProfile, MemoryEfficientModelStorage, MemoryMonitor, 
    MemoryOptimizedRegistry, estimate_model_memory_usage, 
    optimize_model_registry_memory
)
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.core.model import Model, ModelCapabilities, ModelRequirements, ModelMetrics, ModelCost


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


@pytest.fixture
def sample_models(sample_model):
    """Create multiple sample models for testing."""
    models = []
    for i in range(5):
        model = TestModel(
            name=f"test-model-{i}",
            provider="test",
            capabilities=sample_model.capabilities,
            requirements=sample_model.requirements,
            metrics=sample_model.metrics,
            cost=sample_model.cost
        )
        # Add Issue 194 attributes
        model._size_billions = float(i + 1)
        model._expertise = ["general", "chat"] if i % 2 == 0 else ["code", "reasoning"]
        models.append(model)
    return models


class TestMemoryProfile:
    """Test memory profile functionality."""
    
    def test_memory_profile_creation(self):
        """Test creating a memory profile."""
        profile = MemoryProfile.current()
        
        assert profile.timestamp > 0
        assert profile.total_memory_mb > 0
        assert profile.available_memory_mb >= 0
        assert profile.used_memory_mb >= 0
        assert 0 <= profile.memory_percent <= 100
        assert profile.process_memory_mb > 0
        assert profile.process_memory_percent >= 0
        
        # GC stats should be present
        assert isinstance(profile.gc_collections, dict)
        assert isinstance(profile.gc_collected, dict)
    
    def test_memory_profile_fields(self):
        """Test that all memory profile fields are properly initialized."""
        profile = MemoryProfile()
        
        # Default values
        assert profile.total_memory_mb == 0.0
        assert profile.available_memory_mb == 0.0
        assert profile.used_memory_mb == 0.0
        assert profile.memory_percent == 0.0
        assert profile.process_memory_mb == 0.0
        assert profile.process_memory_percent == 0.0
        assert profile.models_count == 0
        assert profile.cache_entries_count == 0
        assert profile.index_memory_mb == 0.0
        assert isinstance(profile.gc_collections, dict)
        assert isinstance(profile.gc_collected, dict)
        assert profile.timestamp > 0


class TestMemoryEfficientModelStorage:
    """Test memory-efficient model storage."""
    
    def test_storage_initialization(self):
        """Test storage initialization."""
        storage = MemoryEfficientModelStorage(max_strong_refs=50)
        
        assert storage.max_strong_refs == 50
        assert len(storage._strong_cache) == 0
        assert len(storage._weak_refs) == 0
        assert len(storage._metadata) == 0
        assert len(storage._access_times) == 0
    
    def test_store_and_retrieve_model(self, sample_model):
        """Test storing and retrieving models."""
        storage = MemoryEfficientModelStorage(max_strong_refs=2)
        
        # Store model
        storage.store_model("test:model", sample_model)
        
        # Check metadata was created
        metadata = storage.get_metadata("test:model")
        assert metadata is not None
        assert metadata["name"] == "test-model"
        assert metadata["provider"] == "test"
        
        # Retrieve model
        retrieved = storage.get_model("test:model")
        assert retrieved is sample_model
        
        # Check access time was updated
        assert "test:model" in storage._access_times
    
    def test_weak_reference_cleanup(self, sample_model):
        """Test weak reference cleanup when models are deleted."""
        storage = MemoryEfficientModelStorage(max_strong_refs=1)
        
        # Store model
        storage.store_model("test:model", sample_model)
        assert storage.get_model("test:model") is not None
        
        # Delete model reference
        del sample_model
        
        # Force garbage collection
        gc.collect()
        
        # Clean up dead references
        cleaned = storage.cleanup_dead_references()
        
        # Model should still be retrievable from strong cache initially
        # But weak reference might be dead
        stats = storage.get_memory_stats()
        assert stats["total_models"] == 1
    
    def test_lru_eviction(self, sample_models):
        """Test LRU eviction from strong cache."""
        storage = MemoryEfficientModelStorage(max_strong_refs=2)
        
        # Store 3 models (should trigger eviction)
        for i, model in enumerate(sample_models[:3]):
            storage.store_model(f"test:model-{i}", model)
        
        # Should only have 2 models in strong cache
        stats = storage.get_memory_stats()
        assert stats["strong_cache_size"] == 2
        assert stats["total_models"] == 3  # Metadata for all 3
        
        # First model should be evicted from strong cache but metadata remains
        assert storage.get_metadata("test:model-0") is not None
    
    def test_cache_size_adjustment(self, sample_models):
        """Test adjusting cache size dynamically."""
        storage = MemoryEfficientModelStorage(max_strong_refs=5)
        
        # Store 3 models
        for i, model in enumerate(sample_models[:3]):
            storage.store_model(f"test:model-{i}", model)
        
        assert storage.get_memory_stats()["strong_cache_size"] == 3
        
        # Reduce cache size
        storage.adjust_cache_size(2)
        assert storage.max_strong_refs == 2
        assert storage.get_memory_stats()["strong_cache_size"] == 2
    
    def test_model_removal(self, sample_model):
        """Test removing models from storage."""
        storage = MemoryEfficientModelStorage()
        
        # Store model
        storage.store_model("test:model", sample_model)
        assert storage.get_model("test:model") is not None
        
        # Remove model
        storage.remove_model("test:model")
        assert storage.get_model("test:model") is None
        assert storage.get_metadata("test:model") is None


class TestMemoryMonitor:
    """Test memory monitoring functionality."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = MemoryMonitor(warning_threshold=75.0, critical_threshold=85.0)
        
        assert monitor.warning_threshold == 75.0
        assert monitor.critical_threshold == 85.0
        assert len(monitor._profiles) == 0
        assert len(monitor._callbacks["warning"]) == 0
        assert len(monitor._callbacks["critical"]) == 0
        assert len(monitor._callbacks["normal"]) == 0
    
    def test_memory_check(self):
        """Test memory checking."""
        monitor = MemoryMonitor()
        
        profile = monitor.check_memory()
        
        assert isinstance(profile, MemoryProfile)
        assert len(monitor._profiles) == 1
        assert monitor._profiles[0] is profile
    
    def test_callback_registration(self):
        """Test registering callbacks for memory events."""
        monitor = MemoryMonitor()
        
        warning_called = []
        critical_called = []
        
        def warning_callback(profile):
            warning_called.append(profile)
        
        def critical_callback(profile):
            critical_called.append(profile)
        
        monitor.add_callback("warning", warning_callback)
        monitor.add_callback("critical", critical_callback)
        
        assert len(monitor._callbacks["warning"]) == 1
        assert len(monitor._callbacks["critical"]) == 1
    
    @patch('orchestrator.models.memory_optimization.MemoryProfile.current')
    def test_threshold_callbacks(self, mock_current):
        """Test that callbacks are triggered at appropriate thresholds."""
        monitor = MemoryMonitor(warning_threshold=80.0, critical_threshold=90.0)
        
        warning_called = []
        critical_called = []
        
        monitor.add_callback("warning", lambda p: warning_called.append(p))
        monitor.add_callback("critical", lambda p: critical_called.append(p))
        
        # Test warning threshold
        mock_profile = MagicMock()
        mock_profile.memory_percent = 85.0
        mock_current.return_value = mock_profile
        
        monitor.check_memory()
        assert len(warning_called) == 1
        assert len(critical_called) == 0
        
        # Test critical threshold
        mock_profile.memory_percent = 95.0
        monitor.check_memory()
        assert len(warning_called) == 1  # Warning callback not called again
        assert len(critical_called) == 1
    
    def test_garbage_collection(self):
        """Test forcing garbage collection."""
        monitor = MemoryMonitor()
        
        collected = monitor.force_garbage_collection()
        
        assert isinstance(collected, dict)
        # Should have collected counts for each generation
        for generation in range(3):
            assert generation in collected
            assert isinstance(collected[generation], int)
    
    def test_memory_trend_analysis(self):
        """Test memory trend analysis."""
        monitor = MemoryMonitor()
        
        # Add some mock profiles
        base_time = time.time()
        for i in range(5):
            profile = MagicMock()
            profile.timestamp = base_time - (60 * (5 - i))  # 5 minutes of data
            profile.memory_percent = 70.0 + (i * 2)  # Increasing trend
            monitor._profiles.append(profile)
        
        trend = monitor.get_memory_trend(minutes=5)
        
        assert trend["trend"] in ["increasing", "decreasing", "stable"]
        assert "start_usage" in trend
        assert "end_usage" in trend
        assert "change" in trend
        assert len(trend["profiles"]) > 0
    
    def test_optimization_suggestions(self):
        """Test optimization suggestions."""
        monitor = MemoryMonitor()
        
        # Add mock profile
        profile = MagicMock()
        profile.memory_percent = 95.0  # Critical usage
        profile.gc_collections = {0: 5, 1: 3, 2: 1}
        monitor._profiles.append(profile)
        
        suggestions = monitor.suggest_optimizations()
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        # Should suggest critical memory actions
        assert any("critical" in s.lower() for s in suggestions)


class TestMemoryOptimizedRegistry:
    """Test memory-optimized registry."""
    
    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = MemoryOptimizedRegistry(max_strong_refs=50, memory_warning_threshold=75.0)
        
        assert registry.storage.max_strong_refs == 50
        assert registry.monitor.warning_threshold == 75.0
        assert registry._optimization_stats["cache_adjustments"] == 0
        assert registry._optimization_stats["gc_triggers"] == 0
        assert registry._optimization_stats["reference_cleanups"] == 0
    
    def test_model_registration(self, sample_model):
        """Test registering models with memory optimization."""
        registry = MemoryOptimizedRegistry()
        
        registry.register_model(sample_model)
        
        # Should be able to retrieve model
        retrieved = registry.get_model("test", "test-model")
        assert retrieved is sample_model
        
        # Should have metadata
        metadata = registry.get_model_metadata("test", "test-model")
        assert metadata is not None
        assert metadata["name"] == "test-model"
    
    def test_memory_optimization(self, sample_models):
        """Test memory optimization functionality."""
        registry = MemoryOptimizedRegistry(max_strong_refs=2)
        
        # Register multiple models
        for model in sample_models[:3]:
            registry.register_model(model)
        
        # Perform optimization
        result = registry.optimize_memory()
        
        assert "dead_references_cleaned" in result
        assert "gc_collected" in result
        assert "current_memory_profile" in result
        assert "optimization_stats" in result
    
    def test_memory_report(self, sample_model):
        """Test getting memory usage report."""
        registry = MemoryOptimizedRegistry()
        
        registry.register_model(sample_model)
        
        report = registry.get_memory_report()
        
        assert "storage_statistics" in report
        assert "current_memory_profile" in report
        assert "memory_trend" in report
        assert "optimization_suggestions" in report
        assert "optimization_stats" in report
    
    @patch('orchestrator.models.memory_optimization.MemoryProfile.current')
    def test_memory_warning_callback(self, mock_current):
        """Test memory warning callback."""
        registry = MemoryOptimizedRegistry()
        
        # Mock high memory usage
        mock_profile = MagicMock()
        mock_profile.memory_percent = 85.0
        mock_current.return_value = mock_profile
        
        # Trigger callback
        registry._on_memory_warning(mock_profile)
        
        assert registry._optimization_stats["cache_adjustments"] == 1
    
    @patch('orchestrator.models.memory_optimization.MemoryProfile.current')
    def test_memory_critical_callback(self, mock_current):
        """Test memory critical callback."""
        registry = MemoryOptimizedRegistry()
        
        # Mock critical memory usage
        mock_profile = MagicMock()
        mock_profile.memory_percent = 95.0
        mock_current.return_value = mock_profile
        
        # Trigger callback
        registry._on_memory_critical(mock_profile)
        
        assert registry._optimization_stats["cache_adjustments"] == 1
        assert registry._optimization_stats["gc_triggers"] == 1


class TestModelRegistryMemoryIntegration:
    """Test memory optimization integration with ModelRegistry."""
    
    def test_registry_with_memory_optimization(self, sample_models):
        """Test ModelRegistry with memory optimization enabled."""
        registry = ModelRegistry(enable_memory_optimization=True, memory_warning_threshold=75.0)
        
        assert registry.enable_memory_optimization is True
        assert registry._memory_optimization_enabled is True
        assert registry.memory_monitor is not None
        
        # Register models
        for model in sample_models[:3]:
            registry.register_model(model)
        
        # Should be able to get memory report
        report = registry.get_memory_report()
        assert "memory_profile" in report
        assert "registry_metrics" in report
        assert report["registry_metrics"]["total_models"] == 3
    
    def test_registry_without_memory_optimization(self, sample_model):
        """Test ModelRegistry with memory optimization disabled."""
        registry = ModelRegistry(enable_memory_optimization=False)
        
        assert registry.enable_memory_optimization is False
        assert registry._memory_optimization_enabled is False
        assert registry.memory_monitor is None
        
        registry.register_model(sample_model)
        
        # Memory methods should return error
        report = registry.get_memory_report()
        assert "error" in report
        
        optimization = registry.optimize_memory_usage()
        assert "error" in optimization
    
    def test_memory_threshold_adjustment(self):
        """Test adjusting memory thresholds."""
        registry = ModelRegistry(enable_memory_optimization=True)
        
        registry.set_memory_thresholds(warning=70.0, critical=85.0)
        
        assert registry.memory_monitor.warning_threshold == 70.0
        assert registry.memory_monitor.critical_threshold == 85.0
    
    def test_enable_disable_memory_monitoring(self):
        """Test enabling and disabling memory monitoring."""
        # Start with disabled
        registry = ModelRegistry(enable_memory_optimization=False)
        assert not registry._memory_optimization_enabled
        
        # Enable
        registry.enable_memory_monitoring(warning_threshold=80.0)
        assert registry._memory_optimization_enabled
        assert registry.memory_monitor is not None
        
        # Disable
        registry.disable_memory_monitoring()
        assert not registry._memory_optimization_enabled
        assert registry.memory_monitor is None


class TestMemoryUtilityFunctions:
    """Test memory utility functions."""
    
    def test_estimate_model_memory_usage(self, sample_model):
        """Test estimating model memory usage."""
        usage_mb = estimate_model_memory_usage(sample_model)
        
        assert isinstance(usage_mb, float)
        assert usage_mb > 0
        # Should be reasonable estimate (not too large)
        assert usage_mb < 100  # Less than 100MB for a simple model
    
    def test_optimize_model_registry_memory(self, sample_models):
        """Test optimizing model registry memory."""
        registry = ModelRegistry()
        
        # Register models
        for model in sample_models:
            registry.register_model(model)
        
        # Test optimization
        result = optimize_model_registry_memory(registry, target_memory_mb=0.1)
        
        assert "current_usage_mb" in result
        assert "target_mb" in result
        assert "action" in result
        assert "models_count" in result
        
        # With small target, should identify candidates for removal
        if result["current_usage_mb"] > 0.1:
            assert result["action"] == "optimization_candidates_identified"
            assert "candidates_for_removal" in result
        else:
            assert result["action"] == "no_optimization_needed"
    
    def test_optimize_registry_without_models_attr(self):
        """Test optimization with invalid registry."""
        fake_registry = MagicMock()
        del fake_registry.models  # Remove models attribute
        
        result = optimize_model_registry_memory(fake_registry)
        
        assert "error" in result
        assert "does not have models attribute" in result["error"]


class TestMemoryOptimizationConcurrency:
    """Test memory optimization under concurrent access."""
    
    def test_concurrent_model_storage(self, sample_models):
        """Test concurrent access to memory-efficient storage."""
        storage = MemoryEfficientModelStorage(max_strong_refs=10)
        
        def store_models(start_idx, count):
            for i in range(start_idx, start_idx + count):
                if i < len(sample_models):
                    storage.store_model(f"test:model-{i}", sample_models[i])
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=store_models, args=(i * 2, 2))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check that models were stored correctly
        stats = storage.get_memory_stats()
        assert stats["total_models"] > 0
    
    def test_concurrent_memory_monitoring(self):
        """Test concurrent memory monitoring."""
        monitor = MemoryMonitor()
        
        def check_memory():
            for _ in range(5):
                monitor.check_memory()
                time.sleep(0.01)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=check_memory)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have multiple profiles
        assert len(monitor._profiles) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])