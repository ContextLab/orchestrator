"""
Comprehensive test suite for the unified wrapper framework.

Tests all components of the wrapper architecture including base classes,
feature flags, configuration management, monitoring, and testing utilities.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

# Import wrapper framework components
from src.orchestrator.core.wrapper_base import (
    BaseWrapper, WrapperResult, WrapperContext, WrapperException,
    WrapperCapability, WrapperStatus, WrapperRegistry
)
from src.orchestrator.core.feature_flags import (
    FeatureFlagManager, FeatureFlag, FeatureFlagScope, FeatureFlagStrategy
)
from src.orchestrator.core.wrapper_base import BaseWrapperConfig
from src.orchestrator.core.wrapper_config import (
    ConfigurationManager, ConfigField, ConfigSource
)
from src.orchestrator.core.wrapper_monitoring import (
    WrapperMonitoring, OperationMetrics, WrapperHealthStatus, AlertSeverity, Alert
)
from src.orchestrator.core.wrapper_testing import (
    WrapperTestHarness, TestScenario, MockWrapper, MockWrapperConfig,
    IntegrationTestSuite, create_basic_scenarios
)


# Test configuration classes

@dataclass
class TestWrapperConfig(BaseWrapperConfig):
    """Test configuration for wrapper testing."""
    
    test_setting: str = "default"
    numeric_setting: int = 42
    
    def get_config_fields(self) -> Dict[str, ConfigField]:
        return {
            "test_setting": ConfigField(
                "test_setting", str, "default", "Test string setting"
            ),
            "numeric_setting": ConfigField(
                "numeric_setting", int, 42, "Test numeric setting",
                min_value=0, max_value=100
            )
        }


# Test wrapper implementation

class TestWrapper(BaseWrapper[str, TestWrapperConfig]):
    """Test wrapper implementation."""
    
    def __init__(self, name: str, config: TestWrapperConfig, feature_flags=None, monitoring=None):
        super().__init__(name, config, feature_flags, monitoring)
        self.operation_count = 0
    
    async def _execute_wrapper_operation(self, context: WrapperContext, *args, **kwargs) -> str:
        self.operation_count += 1
        
        # Simulate different behaviors based on input
        operation_type = kwargs.get('operation_type', 'default')
        
        if operation_type == 'error':
            raise WrapperException("Test error", wrapper_name=self.name)
        
        if operation_type == 'slow':
            await asyncio.sleep(0.1)  # 100ms delay
        
        return f"test_result_{self.operation_count}_{operation_type}"
    
    async def _execute_fallback_operation(
        self, 
        context: WrapperContext,
        original_error: Optional[Exception] = None,
        *args,
        **kwargs
    ) -> str:
        return f"fallback_result_{context.operation_type}"
    
    def _validate_config(self) -> bool:
        return (
            isinstance(self.config.test_setting, str) and
            0 <= self.config.numeric_setting <= 100
        )
    
    def get_capabilities(self) -> List[WrapperCapability]:
        return [WrapperCapability.MONITORING, WrapperCapability.VALIDATION]


# Test classes for wrapper framework

class TestBaseWrapper:
    """Test the BaseWrapper abstract class."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration."""
        return TestWrapperConfig(enabled=True, test_setting="test", numeric_setting=50)
    
    @pytest.fixture
    def feature_flags(self):
        """Create feature flag manager."""
        return FeatureFlagManager()
    
    @pytest.fixture
    def monitoring(self):
        """Create monitoring instance."""
        return WrapperMonitoring()
    
    @pytest.fixture
    def test_wrapper(self, test_config, feature_flags, monitoring):
        """Create test wrapper instance."""
        return TestWrapper("test_wrapper", test_config, feature_flags, monitoring)
    
    @pytest.mark.asyncio
    async def test_successful_operation(self, test_wrapper):
        """Test successful wrapper operation."""
        result = await test_wrapper.execute(operation_type="success", test_param="value")
        
        assert result.success is True
        assert result.data.startswith("test_result_1_success")
        assert result.fallback_used is False
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_error_with_fallback(self, test_wrapper):
        """Test error handling with fallback."""
        result = await test_wrapper.execute(operation_type="error")
        
        assert result.success is True  # Fallback succeeded
        assert result.fallback_used is True
        assert result.data == "fallback_result_error"
        assert result.error is not None  # Original error preserved
    
    @pytest.mark.asyncio
    async def test_disabled_wrapper_uses_fallback(self, test_config, feature_flags, monitoring):
        """Test that disabled wrapper uses fallback."""
        test_config.enabled = False
        wrapper = TestWrapper("disabled_wrapper", test_config, feature_flags, monitoring)
        
        result = await wrapper.execute(operation_type="test")
        
        assert result.success is True
        assert result.fallback_used is True
        assert result.data == "fallback_result_test"
    
    @pytest.mark.asyncio
    async def test_wrapper_capabilities(self, test_wrapper):
        """Test wrapper capabilities reporting."""
        capabilities = test_wrapper.get_capabilities()
        
        assert WrapperCapability.MONITORING in capabilities
        assert WrapperCapability.VALIDATION in capabilities
    
    @pytest.mark.asyncio
    async def test_wrapper_health_info(self, test_wrapper):
        """Test wrapper health information."""
        health_info = test_wrapper.get_health_info()
        
        assert health_info["name"] == "test_wrapper"
        assert health_info["status"] == WrapperStatus.ENABLED.value
        assert health_info["enabled"] is True
        assert health_info["config_valid"] is True
        assert "capabilities" in health_info
    
    def test_wrapper_status_reporting(self, test_wrapper):
        """Test wrapper status reporting."""
        status = test_wrapper.get_status()
        assert status == WrapperStatus.ENABLED
    
    def test_invalid_config_initialization(self, feature_flags, monitoring):
        """Test wrapper with invalid configuration."""
        invalid_config = TestWrapperConfig(enabled=True, numeric_setting=150)  # Out of range
        
        wrapper = TestWrapper("invalid_wrapper", invalid_config, feature_flags, monitoring)
        assert wrapper.get_status() == WrapperStatus.ERROR


class TestFeatureFlagManager:
    """Test the unified feature flag system."""
    
    @pytest.fixture
    def flag_manager(self):
        """Create feature flag manager."""
        return FeatureFlagManager()
    
    def test_basic_flag_registration(self, flag_manager):
        """Test basic flag registration and evaluation."""
        flag = FeatureFlag(name="test_flag", enabled=True)
        flag_manager.register_flag(flag)
        
        assert flag_manager.is_enabled("test_flag") is True
    
    def test_flag_dependencies(self, flag_manager):
        """Test flag dependency resolution."""
        parent_flag = FeatureFlag(name="parent_flag", enabled=True)
        child_flag = FeatureFlag(
            name="child_flag", 
            enabled=True, 
            dependencies=["parent_flag"]
        )
        
        flag_manager.register_flag(parent_flag)
        flag_manager.register_flag(child_flag)
        
        # Child should be enabled when parent is enabled
        assert flag_manager.is_enabled("child_flag") is True
        
        # Disable parent
        flag_manager.disable_flag("parent_flag")
        assert flag_manager.is_enabled("child_flag") is False
    
    def test_percentage_rollout_strategy(self, flag_manager):
        """Test percentage-based rollout strategy."""
        flag = FeatureFlag(
            name="percentage_flag",
            enabled=True,
            strategy=FeatureFlagStrategy.PERCENTAGE,
            rollout_percentage=50.0
        )
        flag_manager.register_flag(flag)
        
        # Test with consistent user ID
        enabled_for_user1 = flag_manager.is_enabled("percentage_flag", user_id="user1")
        enabled_for_user2 = flag_manager.is_enabled("percentage_flag", user_id="user2")
        
        # Results should be consistent for the same user
        assert flag_manager.is_enabled("percentage_flag", user_id="user1") == enabled_for_user1
        assert flag_manager.is_enabled("percentage_flag", user_id="user2") == enabled_for_user2
    
    def test_whitelist_strategy(self, flag_manager):
        """Test whitelist-based strategy."""
        flag = FeatureFlag(
            name="whitelist_flag",
            enabled=True,
            strategy=FeatureFlagStrategy.WHITELIST,
            whitelist=["user1", "user3"]
        )
        flag_manager.register_flag(flag)
        
        assert flag_manager.is_enabled("whitelist_flag", user_id="user1") is True
        assert flag_manager.is_enabled("whitelist_flag", user_id="user2") is False
        assert flag_manager.is_enabled("whitelist_flag", user_id="user3") is True
    
    def test_wrapper_flag_registration(self, flag_manager):
        """Test wrapper-specific flag registration."""
        flag_manager.register_wrapper_flags("test_wrapper")
        
        # Check that standard wrapper flags were created
        assert flag_manager.is_enabled("test_wrapper_enabled") is False  # Disabled by default
        assert "test_wrapper_monitoring" in flag_manager.get_all_flags()
        assert "test_wrapper_fallback" in flag_manager.get_all_flags()
    
    def test_system_status(self, flag_manager):
        """Test system status reporting."""
        flag_manager.register_wrapper_flags("test_wrapper")
        
        status = flag_manager.get_system_status()
        
        assert "total_flags" in status
        assert "enabled_flags" in status
        assert "scope_breakdown" in status
        assert status["total_flags"] > 0


class TestConfigurationManager:
    """Test the configuration management system."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create configuration manager."""
        return ConfigurationManager(temp_config_dir, environment="test")
    
    def test_config_type_registration(self, config_manager):
        """Test configuration type registration."""
        config_manager.register_config_type("test_wrapper", TestWrapperConfig)
        
        assert "test_wrapper" in config_manager._config_types
        assert config_manager._config_types["test_wrapper"] == TestWrapperConfig
    
    def test_config_creation(self, config_manager):
        """Test configuration creation."""
        config_manager.register_config_type("test_wrapper", TestWrapperConfig)
        
        config_data = {
            "enabled": True,
            "test_setting": "custom_value",
            "numeric_setting": 75
        }
        
        config = config_manager.create_config("test_wrapper", config_data)
        
        assert config is not None
        assert config.enabled is True
        assert config.test_setting == "custom_value"
        assert config.numeric_setting == 75
    
    def test_config_validation(self, config_manager):
        """Test configuration validation."""
        config_manager.register_config_type("test_wrapper", TestWrapperConfig)
        
        # Valid configuration
        valid_config_data = {"enabled": True, "numeric_setting": 50}
        config = config_manager.create_config("test_wrapper", valid_config_data)
        assert config is not None
        
        # Invalid configuration (out of range)
        invalid_config_data = {"enabled": True, "numeric_setting": 150}
        config = config_manager.create_config("test_wrapper", invalid_config_data)
        assert config is None  # Should fail validation
    
    def test_config_updates(self, config_manager):
        """Test configuration updates."""
        config_manager.register_config_type("test_wrapper", TestWrapperConfig)
        config_manager.create_config("test_wrapper", {"enabled": True})
        
        # Update configuration
        success = config_manager.update_config(
            "test_wrapper",
            {"test_setting": "updated_value", "numeric_setting": 80},
            "test_user"
        )
        
        assert success is True
        
        updated_config = config_manager.get_config("test_wrapper")
        assert updated_config.test_setting == "updated_value"
        assert updated_config.numeric_setting == 80
    
    def test_system_summary(self, config_manager):
        """Test system summary reporting."""
        config_manager.register_config_type("test_wrapper", TestWrapperConfig)
        config_manager.create_config("test_wrapper", {"enabled": True})
        
        summary = config_manager.get_system_summary()
        
        assert summary["total_configs"] == 1
        assert summary["enabled_configs"] == 1
        assert summary["valid_configs"] == 1
        assert summary["environment"] == "test"


class TestWrapperMonitoring:
    """Test the monitoring system."""
    
    @pytest.fixture
    def monitoring(self):
        """Create monitoring instance."""
        return WrapperMonitoring()
    
    def test_operation_tracking(self, monitoring):
        """Test basic operation tracking."""
        operation_id = monitoring.start_operation("test_wrapper", "test_operation")
        
        assert operation_id in monitoring._active_operations
        
        # Record success
        monitoring.record_success(operation_id, "test_result")
        monitoring.end_operation(operation_id)
        
        assert operation_id not in monitoring._active_operations
        assert len(monitoring._completed_operations) == 1
    
    def test_error_tracking(self, monitoring):
        """Test error tracking."""
        operation_id = monitoring.start_operation("test_wrapper", "error_operation")
        
        monitoring.record_error(operation_id, "Test error message", "TEST_ERROR")
        monitoring.end_operation(operation_id)
        
        # Get wrapper stats
        stats = monitoring.get_wrapper_stats("test_wrapper")
        assert stats["failed_operations"] == 1
        assert stats["success_rate"] < 1.0
    
    def test_fallback_tracking(self, monitoring):
        """Test fallback tracking."""
        operation_id = monitoring.start_operation("test_wrapper", "fallback_operation")
        
        monitoring.record_fallback(operation_id, "Wrapper failed, using fallback")
        monitoring.end_operation(operation_id)
        
        stats = monitoring.get_wrapper_stats("test_wrapper")
        assert stats["fallback_operations"] == 1
    
    def test_health_monitoring(self, monitoring):
        """Test wrapper health monitoring."""
        # Simulate some operations
        for i in range(10):
            operation_id = monitoring.start_operation("test_wrapper", f"operation_{i}")
            
            if i < 8:  # 80% success rate
                monitoring.record_success(operation_id, f"result_{i}")
            else:
                monitoring.record_error(operation_id, f"error_{i}")
            
            monitoring.end_operation(operation_id)
        
        health = monitoring.get_wrapper_health("test_wrapper")
        
        assert health.wrapper_name == "test_wrapper"
        assert health.total_operations == 10
        assert health.successful_operations == 8
        assert health.failed_operations == 2
        assert 0.7 <= health.success_rate <= 0.9  # Should be around 0.8
    
    def test_system_health(self, monitoring):
        """Test system health reporting."""
        # Add some operations for multiple wrappers
        for wrapper_name in ["wrapper1", "wrapper2"]:
            operation_id = monitoring.start_operation(wrapper_name, "test_operation")
            monitoring.record_success(operation_id, "result")
            monitoring.end_operation(operation_id)
        
        system_health = monitoring.get_system_health()
        
        assert system_health["total_wrappers"] == 2
        assert system_health["completed_operations"] == 2
        assert system_health["overall_success_rate"] == 1.0
    
    def test_alert_system(self, monitoring):
        """Test alert generation."""
        # Generate operations with high error rate to trigger alert
        for i in range(10):
            operation_id = monitoring.start_operation("unhealthy_wrapper", f"operation_{i}")
            monitoring.record_error(operation_id, f"error_{i}")
            monitoring.end_operation(operation_id)
        
        # Check if alerts were generated
        alerts = monitoring.get_alerts(wrapper_name="unhealthy_wrapper")
        
        assert len(alerts) > 0
        assert any(alert.severity == AlertSeverity.ERROR for alert in alerts)


class TestWrapperTestHarness:
    """Test the wrapper testing framework."""
    
    @pytest.fixture
    def test_harness(self):
        """Create test harness."""
        return WrapperTestHarness(MockWrapper, MockWrapperConfig)
    
    def test_scenario_creation(self, test_harness):
        """Test test scenario creation."""
        scenarios = create_basic_scenarios()
        test_harness.add_test_scenarios(scenarios)
        
        assert len(test_harness.test_scenarios) == 3
        assert any(s.name == "basic_success" for s in test_harness.test_scenarios)
    
    @pytest.mark.asyncio
    async def test_mock_wrapper_basic_operation(self):
        """Test mock wrapper basic functionality."""
        config = MockWrapperConfig(enabled=True, mock_delay_ms=10.0)
        wrapper = MockWrapper("test_mock", config)
        
        result = await wrapper.execute(operation_type="test", test_param="value")
        
        assert result.success is True
        assert result.data is not None
        assert wrapper.call_count == 1
    
    @pytest.mark.asyncio
    async def test_mock_wrapper_failure(self):
        """Test mock wrapper failure simulation."""
        config = MockWrapperConfig(enabled=True, mock_should_fail=True)
        wrapper = MockWrapper("test_mock", config)
        
        result = await wrapper.execute(operation_type="test")
        
        assert result.success is True  # Should succeed via fallback
        assert result.fallback_used is True
        assert result.data["fallback"] is True
    
    @pytest.mark.asyncio
    async def test_scenario_execution(self, test_harness):
        """Test running test scenarios."""
        scenario = TestScenario(
            name="simple_test",
            description="Simple test scenario",
            inputs={"test_input": "value"},
            expected_outputs={"success": True},
            should_succeed=True
        )
        
        test_harness.add_test_scenario(scenario)
        results = await test_harness.run_all_scenarios()
        
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].scenario_name == "simple_test"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, test_harness):
        """Test performance benchmarking."""
        scenario = TestScenario(
            name="performance_test",
            description="Performance test",
            inputs={"test_input": "performance"},
            expected_outputs={"success": True},
            should_succeed=True
        )
        
        benchmark_result = await test_harness.run_performance_benchmark(
            scenario, iterations=10
        )
        
        assert benchmark_result.iterations == 10
        assert benchmark_result.success_rate >= 0.0
        assert benchmark_result.average_time_ms >= 0.0
        assert benchmark_result.throughput_ops_per_second >= 0.0
    
    def test_test_report_generation(self, test_harness):
        """Test test report generation."""
        # Add some mock results
        from src.orchestrator.core.wrapper_testing import TestResult
        
        test_harness.test_results = [
            TestResult("test1", True, 100.0),
            TestResult("test2", False, 200.0, error_message="Test error")
        ]
        
        report = test_harness.generate_test_report()
        
        assert report["wrapper_class"] == "MockWrapper"
        assert report["test_summary"]["total_tests"] == 2
        assert report["test_summary"]["successful_tests"] == 1
        assert report["test_summary"]["success_rate"] == 0.5


class TestWrapperRegistry:
    """Test the wrapper registry system."""
    
    @pytest.fixture
    def registry(self):
        """Create wrapper registry."""
        return WrapperRegistry()
    
    @pytest.fixture
    def test_wrapper(self):
        """Create test wrapper for registry."""
        config = TestWrapperConfig(enabled=True)
        return TestWrapper("registry_test_wrapper", config)
    
    def test_wrapper_registration(self, registry, test_wrapper):
        """Test wrapper registration."""
        registry.register_wrapper(test_wrapper)
        
        assert len(registry.get_all_wrappers()) == 1
        assert registry.get_wrapper("registry_test_wrapper") == test_wrapper
    
    def test_capability_querying(self, registry, test_wrapper):
        """Test querying wrappers by capability."""
        registry.register_wrapper(test_wrapper)
        
        monitoring_wrappers = registry.get_wrappers_by_capability(WrapperCapability.MONITORING)
        
        assert len(monitoring_wrappers) == 1
        assert monitoring_wrappers[0] == test_wrapper
    
    def test_wrapper_unregistration(self, registry, test_wrapper):
        """Test wrapper unregistration."""
        registry.register_wrapper(test_wrapper)
        
        success = registry.unregister_wrapper("registry_test_wrapper")
        
        assert success is True
        assert len(registry.get_all_wrappers()) == 0
        assert registry.get_wrapper("registry_test_wrapper") is None
    
    def test_registry_health(self, registry, test_wrapper):
        """Test registry health reporting."""
        registry.register_wrapper(test_wrapper)
        
        health = registry.get_registry_health()
        
        assert health["total_wrappers"] == 1
        assert health["healthy_wrappers"] == 1
        assert health["health_percentage"] == 100.0


class TestIntegrationTestSuite:
    """Test the integration test suite."""
    
    @pytest.mark.asyncio
    async def test_integration_suite_basic(self):
        """Test basic integration test suite functionality."""
        suite = IntegrationTestSuite()
        
        # Mock setup and teardown functions
        setup_called = False
        teardown_called = False
        
        @suite.setup
        async def mock_setup():
            nonlocal setup_called
            setup_called = True
        
        @suite.teardown
        async def mock_teardown():
            nonlocal teardown_called
            teardown_called = True
        
        @suite.test("basic_integration_test")
        async def mock_test():
            assert True  # Simple test
        
        # Run integration tests
        results = await suite.run_integration_tests()
        
        assert setup_called is True
        assert teardown_called is True
        assert results["total_tests"] == 1
        assert results["successful_tests"] == 1
        assert results["success_rate"] == 1.0


# Integration tests that combine multiple components

class TestFullIntegration:
    """Test full integration of all wrapper framework components."""
    
    @pytest.fixture
    def integrated_setup(self):
        """Set up integrated test environment."""
        # Create all components
        feature_flags = FeatureFlagManager()
        monitoring = WrapperMonitoring()
        config = TestWrapperConfig(enabled=True, test_setting="integration")
        
        # Register wrapper flags
        feature_flags.register_wrapper_flags("integration_wrapper")
        feature_flags.enable_flag("integration_wrapper_enabled")
        
        # Create wrapper
        wrapper = TestWrapper("integration_wrapper", config, feature_flags, monitoring)
        
        # Create registry and register wrapper
        registry = WrapperRegistry()
        registry.register_wrapper(wrapper)
        
        return {
            "wrapper": wrapper,
            "feature_flags": feature_flags,
            "monitoring": monitoring,
            "registry": registry
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_operation(self, integrated_setup):
        """Test complete end-to-end operation."""
        wrapper = integrated_setup["wrapper"]
        monitoring = integrated_setup["monitoring"]
        
        # Execute operation
        result = await wrapper.execute(operation_type="integration_test", data="test_data")
        
        # Verify result
        assert result.success is True
        assert "integration_test" in result.data
        
        # Verify monitoring captured the operation
        stats = monitoring.get_wrapper_stats("integration_wrapper")
        assert stats["total_operations"] >= 1
        assert stats["success_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_feature_flag_controlled_behavior(self, integrated_setup):
        """Test that feature flags control wrapper behavior."""
        wrapper = integrated_setup["wrapper"]
        feature_flags = integrated_setup["feature_flags"]
        
        # Disable wrapper via feature flag
        feature_flags.disable_flag("integration_wrapper_enabled")
        
        # Execute operation - should use fallback
        result = await wrapper.execute(operation_type="flag_test")
        
        assert result.success is True
        assert result.fallback_used is True
        assert result.data == "fallback_result_flag_test"
    
    def test_registry_health_with_monitoring(self, integrated_setup):
        """Test registry health reporting with monitoring data."""
        registry = integrated_setup["registry"]
        
        health = registry.get_registry_health()
        
        assert health["total_wrappers"] == 1
        assert health["healthy_wrappers"] == 1
        assert health["health_percentage"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])