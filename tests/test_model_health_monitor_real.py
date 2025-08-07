"""Real integration tests for Model Health Monitor - Phase 3

Tests health monitoring with REAL service calls, recovery mechanisms,
and integration with Phase 2 service managers. NO MOCKS policy enforced.
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import MagicMock

from src.orchestrator.intelligence.model_health_monitor import (
    ModelHealthMonitor,
    HealthStatus,
    HealthCheck,
    HealthMetrics,
    create_health_monitor,
    setup_basic_health_monitoring
)
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.utils.api_keys import load_api_keys_optional


class TestModelHealthMonitorReal:
    """Test health monitor with real integration."""
    
    def setup_method(self):
        """Setup test environment."""
        self.api_keys = load_api_keys_optional()
        self.registry = ModelRegistry()
        self.monitor = ModelHealthMonitor(
            self.registry,
            check_interval=5,  # Short interval for testing
            max_history=10,
            recovery_enabled=True
        )
        
        # Setup test models if API keys available
        self._setup_test_models()
    
    def teardown_method(self):
        """Cleanup after tests."""
        if hasattr(self, 'monitor') and self.monitor:
            self.monitor.stop_monitoring()
    
    def _setup_test_models(self):
        """Setup test models for monitoring."""
        try:
            # Register available models for testing
            if self.api_keys.get("OPENAI_API_KEY"):
                from src.orchestrator.models.openai_model import OpenAIModel
                openai_model = OpenAIModel("gpt-3.5-turbo", api_key=self.api_keys["OPENAI_API_KEY"])
                self.registry.register_model(openai_model)
                
        except ImportError:
            pytest.skip("Model imports not available")
        except Exception as e:
            pytest.skip(f"Error setting up test models: {e}")
    
    def test_health_monitor_initialization(self):
        """Test health monitor initializes correctly."""
        assert self.monitor.model_registry == self.registry
        assert self.monitor.check_interval == 5
        assert self.monitor.max_history == 10
        assert self.monitor.recovery_enabled is True
        assert not self.monitor.monitoring_active
        assert len(self.monitor.health_metrics) == 0
        
        # Test configuration parameters
        assert self.monitor.failure_threshold == 3
        assert self.monitor.recovery_threshold == 2
        assert self.monitor.degraded_response_time == 5000
        assert self.monitor.timeout_threshold == 30000
    
    def test_health_check_data_structures(self):
        """Test health check data structures."""
        # Test HealthCheck
        check = HealthCheck(
            model_key="test:model",
            status=HealthStatus.HEALTHY,
            timestamp=time.time(),
            response_time_ms=100.0,
            error_message=None,
            metadata={"test": "data"}
        )
        
        assert check.model_key == "test:model"
        assert check.status == HealthStatus.HEALTHY
        assert check.response_time_ms == 100.0
        assert check.metadata["test"] == "data"
        
        # Test HealthMetrics
        metrics = HealthMetrics(
            model_key="test:model",
            current_status=HealthStatus.HEALTHY,
            last_check=time.time(),
            consecutive_failures=0,
            consecutive_successes=5,
            total_checks=10,
            success_rate=0.9,
            avg_response_time_ms=150.0
        )
        
        assert metrics.model_key == "test:model"
        assert metrics.success_rate == 0.9
        assert len(metrics.recent_checks) == 0  # Starts empty
    
    @pytest.mark.asyncio
    async def test_check_model_health_basic(self):
        """Test basic model health checking."""
        # Test with a mock model key (will result in unknown status)
        check = await self.monitor.check_model_health("unknown:model")
        
        assert isinstance(check, HealthCheck)
        assert check.model_key == "unknown:model"
        assert check.status in [HealthStatus.UNKNOWN, HealthStatus.UNHEALTHY]
        assert check.timestamp > 0
        assert check.response_time_ms >= 0
    
    @pytest.mark.skipif(
        not load_api_keys_optional().get("OPENAI_API_KEY"),
        reason="OpenAI API key required for real health check test"
    )
    @pytest.mark.asyncio
    async def test_check_openai_health_real(self):
        """Test real OpenAI model health checking."""
        model_key = "openai:gpt-3.5-turbo"
        
        # Ensure model is registered
        if model_key not in [f"{m.provider}:{m.name}" for m in self.registry.models.values()]:
            pytest.skip(f"Model {model_key} not registered")
        
        check = await self.monitor.check_model_health(model_key)
        
        assert isinstance(check, HealthCheck)
        assert check.model_key == model_key
        # Status should be healthy or unhealthy (not unknown for real API)
        assert check.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]
        assert check.response_time_ms > 0
        
        if check.status == HealthStatus.UNHEALTHY:
            assert check.error_message is not None
            print(f"OpenAI health check failed: {check.error_message}")
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        # Add a test model to monitor
        self.monitor.health_metrics["test:model"] = HealthMetrics(
            model_key="test:model",
            current_status=HealthStatus.UNKNOWN,
            last_check=0.0,
            consecutive_failures=0,
            consecutive_successes=0,
            total_checks=0,
            success_rate=0.0,
            avg_response_time_ms=0.0
        )
        
        # Start monitoring
        self.monitor.start_monitoring(["test:model"])
        
        assert self.monitor.monitoring_active is True
        assert self.monitor._monitor_thread is not None
        assert self.monitor._monitor_thread.is_alive()
        
        # Let it run briefly
        time.sleep(0.5)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        assert self.monitor.monitoring_active is False
        # Thread should stop within reasonable time
        time.sleep(1.0)
        assert not self.monitor._monitor_thread.is_alive()
    
    def test_health_status_tracking(self):
        """Test health status tracking and updates."""
        model_key = "test:model"
        
        # Create initial metrics
        self.monitor.health_metrics[model_key] = HealthMetrics(
            model_key=model_key,
            current_status=HealthStatus.UNKNOWN,
            last_check=0.0,
            consecutive_failures=0,
            consecutive_successes=0,
            total_checks=0,
            success_rate=0.0,
            avg_response_time_ms=0.0
        )
        
        # Simulate a healthy check
        healthy_check = HealthCheck(
            model_key=model_key,
            status=HealthStatus.HEALTHY,
            timestamp=time.time(),
            response_time_ms=100.0
        )
        
        self.monitor._update_health_metrics(model_key, healthy_check)
        
        metrics = self.monitor.health_metrics[model_key]
        assert metrics.total_checks == 1
        assert metrics.consecutive_successes == 1
        assert metrics.consecutive_failures == 0
        assert metrics.success_rate == 1.0
        assert metrics.avg_response_time_ms == 100.0
        assert len(metrics.recent_checks) == 1
    
    def test_status_change_callbacks(self):
        """Test health status change callbacks."""
        model_key = "test:model"
        callback_calls = []
        
        def test_callback(model_key: str, old_status: HealthStatus, new_status: HealthStatus):
            callback_calls.append((model_key, old_status, new_status))
        
        self.monitor.add_health_change_callback(test_callback)
        
        # Initialize metrics
        self.monitor.health_metrics[model_key] = HealthMetrics(
            model_key=model_key,
            current_status=HealthStatus.UNKNOWN,
            last_check=0.0,
            consecutive_failures=0,
            consecutive_successes=0,
            total_checks=0,
            success_rate=0.0,
            avg_response_time_ms=0.0
        )
        
        # Simulate status change
        healthy_check = HealthCheck(
            model_key=model_key,
            status=HealthStatus.HEALTHY,
            timestamp=time.time(),
            response_time_ms=100.0
        )
        
        self.monitor._update_health_metrics(model_key, healthy_check)
        
        # Should trigger callback
        assert len(callback_calls) >= 1
        assert callback_calls[-1][0] == model_key
        assert callback_calls[-1][2] == HealthStatus.HEALTHY
    
    def test_failure_detection_and_thresholds(self):
        """Test failure detection and threshold logic."""
        model_key = "test:model"
        
        # Initialize metrics
        self.monitor.health_metrics[model_key] = HealthMetrics(
            model_key=model_key,
            current_status=HealthStatus.HEALTHY,
            last_check=0.0,
            consecutive_failures=0,
            consecutive_successes=0,
            total_checks=0,
            success_rate=0.0,
            avg_response_time_ms=0.0
        )
        
        # Simulate consecutive failures
        for i in range(self.monitor.failure_threshold + 1):
            unhealthy_check = HealthCheck(
                model_key=model_key,
                status=HealthStatus.UNHEALTHY,
                timestamp=time.time(),
                response_time_ms=1000.0,
                error_message=f"Simulated failure {i}"
            )
            
            self.monitor._update_health_metrics(model_key, unhealthy_check)
        
        metrics = self.monitor.health_metrics[model_key]
        
        # Should be marked as unhealthy after threshold failures
        assert metrics.current_status == HealthStatus.UNHEALTHY
        assert metrics.consecutive_failures >= self.monitor.failure_threshold
    
    def test_degraded_status_detection(self):
        """Test detection of degraded status based on response time."""
        model_key = "test:model"
        
        # Initialize metrics
        self.monitor.health_metrics[model_key] = HealthMetrics(
            model_key=model_key,
            current_status=HealthStatus.HEALTHY,
            last_check=0.0,
            consecutive_failures=0,
            consecutive_successes=0,
            total_checks=0,
            success_rate=0.0,
            avg_response_time_ms=0.0
        )
        
        # Simulate slow but successful response
        slow_check = HealthCheck(
            model_key=model_key,
            status=HealthStatus.HEALTHY,
            timestamp=time.time(),
            response_time_ms=self.monitor.degraded_response_time + 1000  # Slower than threshold
        )
        
        self.monitor._update_health_metrics(model_key, slow_check)
        
        metrics = self.monitor.health_metrics[model_key]
        
        # Should be marked as degraded due to slow response
        assert metrics.current_status == HealthStatus.DEGRADED
    
    def test_get_health_status_methods(self):
        """Test health status retrieval methods."""
        # Setup test metrics
        healthy_model = "test:healthy"
        unhealthy_model = "test:unhealthy"
        
        self.monitor.health_metrics[healthy_model] = HealthMetrics(
            model_key=healthy_model,
            current_status=HealthStatus.HEALTHY,
            last_check=time.time(),
            consecutive_failures=0,
            consecutive_successes=3,
            total_checks=5,
            success_rate=1.0,
            avg_response_time_ms=100.0
        )
        
        self.monitor.health_metrics[unhealthy_model] = HealthMetrics(
            model_key=unhealthy_model,
            current_status=HealthStatus.UNHEALTHY,
            last_check=time.time(),
            consecutive_failures=5,
            consecutive_successes=0,
            total_checks=5,
            success_rate=0.0,
            avg_response_time_ms=1000.0
        )
        
        # Test individual status
        healthy_status = self.monitor.get_health_status(healthy_model)
        assert healthy_status is not None
        assert healthy_status.current_status == HealthStatus.HEALTHY
        
        # Test all status
        all_status = self.monitor.get_all_health_status()
        assert len(all_status) == 2
        assert healthy_model in all_status
        assert unhealthy_model in all_status
        
        # Test unhealthy models
        unhealthy_models = self.monitor.get_unhealthy_models()
        assert unhealthy_model in unhealthy_models
        assert healthy_model not in unhealthy_models
    
    @pytest.mark.asyncio
    async def test_recovery_mechanism_basic(self):
        """Test basic recovery mechanism."""
        model_key = "ollama:test"
        
        # Initialize metrics
        self.monitor.health_metrics[model_key] = HealthMetrics(
            model_key=model_key,
            current_status=HealthStatus.UNHEALTHY,
            last_check=time.time(),
            consecutive_failures=5,
            consecutive_successes=0,
            total_checks=5,
            success_rate=0.0,
            avg_response_time_ms=1000.0
        )
        
        # Attempt recovery
        recovery_attempted = await self.monitor.recover_model(model_key)
        
        # Should attempt recovery for Ollama models
        assert isinstance(recovery_attempted, bool)
        
        # Check recovery tracking
        metrics = self.monitor.health_metrics[model_key]
        if recovery_attempted:
            assert metrics.recovery_attempts > 0
            assert metrics.last_recovery_time is not None
    
    def test_recovery_disabled(self):
        """Test behavior when recovery is disabled."""
        # Create monitor with recovery disabled
        monitor_no_recovery = ModelHealthMonitor(
            self.registry,
            recovery_enabled=False
        )
        
        # Setup model
        model_key = "test:model"
        monitor_no_recovery.health_metrics[model_key] = HealthMetrics(
            model_key=model_key,
            current_status=HealthStatus.UNHEALTHY,
            last_check=time.time(),
            consecutive_failures=5,
            consecutive_successes=0,
            total_checks=5,
            success_rate=0.0,
            avg_response_time_ms=1000.0
        )
        
        # Recovery should not be attempted
        async def test_no_recovery():
            result = await monitor_no_recovery.recover_model(model_key)
            assert result is False
            
        asyncio.run(test_no_recovery())
    
    def test_create_health_monitor_helper(self):
        """Test helper function for creating health monitor."""
        monitor = create_health_monitor(
            self.registry,
            check_interval=30,
            max_history=20
        )
        
        assert isinstance(monitor, ModelHealthMonitor)
        assert monitor.model_registry == self.registry
        assert monitor.check_interval == 30
        assert monitor.max_history == 20
    
    def test_setup_basic_health_monitoring_helper(self):
        """Test helper function for basic monitoring setup."""
        # Setup with specific models
        monitor = setup_basic_health_monitoring(
            self.registry,
            models=["test:model1", "test:model2"]
        )
        
        assert isinstance(monitor, ModelHealthMonitor)
        assert monitor.monitoring_active is True
        assert len(monitor.health_change_callbacks) > 0
        
        # Cleanup
        monitor.stop_monitoring()


class TestModelHealthMonitorOllama:
    """Test health monitor with Ollama integration."""
    
    def setup_method(self):
        """Setup Ollama-specific tests."""
        self.registry = ModelRegistry()
        self.monitor = ModelHealthMonitor(self.registry)
    
    def teardown_method(self):
        """Cleanup."""
        if hasattr(self, 'monitor') and self.monitor:
            self.monitor.stop_monitoring()
    
    def test_ollama_health_check_service_not_running(self):
        """Test Ollama health check when service is not running."""
        async def test_health_check():
            # Test with model that likely doesn't exist
            check = await self.monitor._check_ollama_health("ollama:nonexistent", time.time())
            
            assert isinstance(check, HealthCheck)
            assert check.model_key == "ollama:nonexistent"
            # Should be unhealthy if service not running or model not available
            assert check.status in [HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]
            
        asyncio.run(test_health_check())
    
    @pytest.mark.asyncio 
    async def test_ollama_recovery_strategy(self):
        """Test Ollama recovery strategy."""
        model_key = "ollama:test-model"
        
        # This will test the recovery logic even if Ollama isn't running
        # The recovery function should handle the case gracefully
        success = await self.monitor._recover_ollama_model(model_key)
        
        # Should return boolean (may be False if Ollama not available)
        assert isinstance(success, bool)


class TestModelHealthMonitorEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Setup for edge case testing."""
        self.registry = ModelRegistry()
        self.monitor = ModelHealthMonitor(self.registry)
    
    def teardown_method(self):
        """Cleanup."""
        if hasattr(self, 'monitor') and self.monitor:
            self.monitor.stop_monitoring()
    
    def test_malformed_model_key_handling(self):
        """Test handling of malformed model keys."""
        async def test_malformed_keys():
            malformed_keys = ["no-colon", ":missing-provider", "provider:", ""]
            
            for key in malformed_keys:
                if key:  # Skip empty string
                    check = await self.monitor.check_model_health(key)
                    assert isinstance(check, HealthCheck)
                    assert check.status in [HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]
                    
        asyncio.run(test_malformed_keys())
    
    def test_concurrent_monitoring_safety(self):
        """Test thread safety of concurrent monitoring."""
        # Add multiple models
        for i in range(5):
            model_key = f"test:model{i}"
            self.monitor.health_metrics[model_key] = HealthMetrics(
                model_key=model_key,
                current_status=HealthStatus.UNKNOWN,
                last_check=0.0,
                consecutive_failures=0,
                consecutive_successes=0,
                total_checks=0,
                success_rate=0.0,
                avg_response_time_ms=0.0
            )
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Let it run for a short time
        time.sleep(2.0)
        
        # Access metrics from different thread (should be thread-safe)
        def access_metrics():
            for _ in range(10):
                all_metrics = self.monitor.get_all_health_status()
                assert isinstance(all_metrics, dict)
                time.sleep(0.1)
        
        thread = threading.Thread(target=access_metrics)
        thread.start()
        thread.join(timeout=5.0)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        assert not thread.is_alive()
    
    def test_memory_management_with_history_limit(self):
        """Test memory management with history limits."""
        model_key = "test:model"
        
        # Initialize metrics
        self.monitor.health_metrics[model_key] = HealthMetrics(
            model_key=model_key,
            current_status=HealthStatus.HEALTHY,
            last_check=0.0,
            consecutive_failures=0,
            consecutive_successes=0,
            total_checks=0,
            success_rate=0.0,
            avg_response_time_ms=0.0
        )
        
        # Add more checks than max_history
        for i in range(self.monitor.max_history + 10):
            check = HealthCheck(
                model_key=model_key,
                status=HealthStatus.HEALTHY,
                timestamp=time.time(),
                response_time_ms=100.0 + i
            )
            
            self.monitor._update_health_metrics(model_key, check)
        
        metrics = self.monitor.health_metrics[model_key]
        
        # Should not exceed max_history
        assert len(metrics.recent_checks) <= self.monitor.max_history
        assert len(metrics.recent_checks) == self.monitor.max_history  # Should be exactly at limit


@pytest.mark.integration
class TestModelHealthMonitorRealWorld:
    """Real-world integration testing."""
    
    def setup_method(self):
        """Setup for real-world testing."""
        self.api_keys = load_api_keys_optional()
        self.registry = ModelRegistry()
        
        # Setup real models if available
        if self.api_keys.get("OPENAI_API_KEY"):
            try:
                from src.orchestrator.models.openai_model import OpenAIModel
                model = OpenAIModel("gpt-3.5-turbo", api_key=self.api_keys["OPENAI_API_KEY"])
                self.registry.register_model(model)
            except Exception:
                pass
    
    @pytest.mark.skipif(
        not load_api_keys_optional().get("OPENAI_API_KEY"),
        reason="OpenAI API key required for real-world test"
    )
    def test_real_monitoring_scenario(self):
        """Test realistic monitoring scenario with real API."""
        monitor = ModelHealthMonitor(
            self.registry,
            check_interval=10,  # 10 second intervals
            max_history=5,
            recovery_enabled=True
        )
        
        # Track status changes
        status_changes = []
        def track_changes(model_key: str, old_status: HealthStatus, new_status: HealthStatus):
            status_changes.append((model_key, old_status, new_status))
        
        monitor.add_health_change_callback(track_changes)
        
        try:
            # Start monitoring
            monitor.start_monitoring()
            
            # Let it run for a few check cycles
            time.sleep(25)  # Should allow 2-3 health checks
            
            # Check that monitoring is working
            all_status = monitor.get_all_health_status()
            assert len(all_status) > 0
            
            for model_key, metrics in all_status.items():
                # Should have performed at least one check
                assert metrics.total_checks > 0
                assert metrics.last_check > 0
                assert len(metrics.recent_checks) > 0
                
                # Status should not be unknown after real checks
                assert metrics.current_status != HealthStatus.UNKNOWN
                
        finally:
            monitor.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])