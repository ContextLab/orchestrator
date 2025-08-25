"""
Reusable testing framework for wrapper validation.

This module provides comprehensive testing utilities for wrapper development
including test fixtures, mock implementations, integration testing patterns,
and performance benchmarking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Type, TypeVar, Callable
from unittest.mock import Mock, AsyncMock, MagicMock
import uuid

from .wrapper_base import (
    BaseWrapper, BaseWrapperConfig, WrapperResult, WrapperContext,
    WrapperException, WrapperCapability, WrapperStatus
)
from .feature_flags import FeatureFlagManager, FeatureFlag, FeatureFlagScope
from .wrapper_monitoring import WrapperMonitoring, OperationMetrics, WrapperHealthStatus
from .wrapper_config import ConfigurationManager

logger = logging.getLogger(__name__)

T = TypeVar('T')
C = TypeVar('C', bound=BaseWrapperConfig)


@dataclass
class TestScenario:
    """Test scenario definition for wrapper testing."""
    
    name: str
    description: str
    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    should_succeed: bool = True
    should_use_fallback: bool = False
    expected_error: Optional[str] = None
    timeout_seconds: float = 30.0
    tags: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of a wrapper test."""
    
    scenario_name: str
    success: bool
    execution_time_ms: float
    actual_output: Any = None
    expected_output: Any = None
    error_message: Optional[str] = None
    used_fallback: bool = False
    metrics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.success and self.error_message:
            # Inconsistent state - if there's an error, it shouldn't be successful
            self.success = False


@dataclass
class BenchmarkResult:
    """Result of performance benchmarking."""
    
    wrapper_name: str
    scenario_name: str
    iterations: int
    total_time_ms: float
    average_time_ms: float
    min_time_ms: float
    max_time_ms: float
    success_rate: float
    fallback_rate: float
    throughput_ops_per_second: float
    memory_usage_mb: Optional[float] = None


class MockWrapperConfig(BaseWrapperConfig):
    """Mock configuration for testing purposes."""
    
    mock_enabled: bool = True
    mock_should_fail: bool = False
    mock_should_timeout: bool = False
    mock_delay_ms: float = 0.0
    
    def get_config_fields(self) -> Dict[str, Any]:
        """Get mock configuration fields."""
        return {
            "mock_enabled": {"type": bool, "default": True},
            "mock_should_fail": {"type": bool, "default": False},
            "mock_should_timeout": {"type": bool, "default": False},
            "mock_delay_ms": {"type": float, "default": 0.0, "min_value": 0.0}
        }


class MockWrapper(BaseWrapper[Any, MockWrapperConfig]):
    """Mock wrapper implementation for testing."""
    
    def __init__(
        self,
        name: str = "mock_wrapper",
        config: Optional[MockWrapperConfig] = None,
        feature_flags: Optional[FeatureFlagManager] = None,
        monitoring: Optional[WrapperMonitoring] = None
    ):
        config = config or MockWrapperConfig()
        super().__init__(name, config, feature_flags, monitoring)
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []
    
    async def _execute_wrapper_operation(self, context: WrapperContext, *args, **kwargs) -> Any:
        """Execute mock wrapper operation."""
        self.call_count += 1
        self.call_history.append({
            "context": context,
            "args": args,
            "kwargs": kwargs,
            "timestamp": datetime.utcnow()
        })
        
        # Simulate delay
        if self.config.mock_delay_ms > 0:
            await asyncio.sleep(self.config.mock_delay_ms / 1000)
        
        # Simulate failure
        if self.config.mock_should_fail:
            raise WrapperException("Mock wrapper failure", wrapper_name=self.name)
        
        # Simulate timeout
        if self.config.mock_should_timeout:
            await asyncio.sleep(60)  # Will be interrupted by timeout
        
        return {"success": True, "call_count": self.call_count, "args": args, "kwargs": kwargs}
    
    async def _execute_fallback_operation(
        self, 
        context: WrapperContext,
        original_error: Optional[Exception] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute mock fallback operation."""
        return {"fallback": True, "original_error": str(original_error) if original_error else None}
    
    def _validate_config(self) -> bool:
        """Validate mock configuration."""
        return self.config.mock_delay_ms >= 0
    
    def get_capabilities(self) -> List[WrapperCapability]:
        """Get mock wrapper capabilities."""
        return [WrapperCapability.MONITORING]


class WrapperTestHarness:
    """
    Test harness for comprehensive wrapper testing.
    
    Provides utilities for:
    - Setting up test environments
    - Running test scenarios
    - Performance benchmarking
    - Integration testing
    - Mock implementations
    """
    
    def __init__(
        self,
        wrapper_class: Type[BaseWrapper],
        config_class: Type[BaseWrapperConfig]
    ):
        """
        Initialize test harness.
        
        Args:
            wrapper_class: Class of wrapper to test
            config_class: Configuration class for the wrapper
        """
        self.wrapper_class = wrapper_class
        self.config_class = config_class
        self.test_scenarios: List[TestScenario] = []
        self.test_results: List[TestResult] = []
        self.benchmark_results: List[BenchmarkResult] = []
    
    def add_test_scenario(self, scenario: TestScenario) -> None:
        """Add a test scenario."""
        self.test_scenarios.append(scenario)
    
    def add_test_scenarios(self, scenarios: List[TestScenario]) -> None:
        """Add multiple test scenarios."""
        self.test_scenarios.extend(scenarios)
    
    @asynccontextmanager
    async def create_test_wrapper(
        self,
        config_overrides: Optional[Dict[str, Any]] = None,
        feature_flags_config: Optional[Dict[str, bool]] = None
    ) -> AsyncIterator[BaseWrapper]:
        """
        Create a wrapper instance for testing.
        
        Args:
            config_overrides: Configuration overrides
            feature_flags_config: Feature flag overrides
            
        Yields:
            Configured wrapper instance
        """
        # Create test configuration
        config_data = {"enabled": True}
        if config_overrides:
            config_data.update(config_overrides)
        
        config = self.config_class.from_dict(config_data)
        
        # Create feature flag manager
        feature_flags = FeatureFlagManager()
        if feature_flags_config:
            for flag_name, enabled in feature_flags_config.items():
                flag = FeatureFlag(name=flag_name, enabled=enabled)
                feature_flags.register_flag(flag)
        
        # Create monitoring
        monitoring = WrapperMonitoring()
        
        # Create wrapper
        wrapper_name = f"test_{self.wrapper_class.__name__.lower()}_{uuid.uuid4().hex[:8]}"
        wrapper = self.wrapper_class(wrapper_name, config, feature_flags, monitoring)
        
        try:
            yield wrapper
        finally:
            # Cleanup if needed
            pass
    
    async def run_test_scenario(
        self,
        scenario: TestScenario,
        wrapper: BaseWrapper
    ) -> TestResult:
        """
        Run a single test scenario.
        
        Args:
            scenario: Test scenario to run
            wrapper: Wrapper instance to test
            
        Returns:
            Test result
        """
        start_time = time.time()
        
        try:
            # Execute wrapper operation
            result = await asyncio.wait_for(
                wrapper.execute(
                    operation_type=scenario.name,
                    **scenario.inputs
                ),
                timeout=scenario.timeout_seconds
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Check if result matches expectations
            success = self._validate_result(scenario, result)
            
            return TestResult(
                scenario_name=scenario.name,
                success=success,
                execution_time_ms=execution_time,
                actual_output=result.data if result else None,
                expected_output=scenario.expected_outputs,
                error_message=result.error if result else None,
                used_fallback=result.fallback_used if result else False,
                metrics=result.metrics if result else None
            )
            
        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            return TestResult(
                scenario_name=scenario.name,
                success=False,
                execution_time_ms=execution_time,
                error_message="Test timed out"
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Check if this error was expected
            expected_error = scenario.expected_error
            success = (
                not scenario.should_succeed and 
                expected_error and 
                expected_error in str(e)
            )
            
            return TestResult(
                scenario_name=scenario.name,
                success=success,
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    async def run_all_scenarios(
        self,
        config_overrides: Optional[Dict[str, Any]] = None,
        feature_flags_config: Optional[Dict[str, bool]] = None
    ) -> List[TestResult]:
        """
        Run all test scenarios.
        
        Args:
            config_overrides: Configuration overrides for wrapper
            feature_flags_config: Feature flag overrides
            
        Returns:
            List of test results
        """
        results = []
        
        async with self.create_test_wrapper(config_overrides, feature_flags_config) as wrapper:
            for scenario in self.test_scenarios:
                logger.info(f"Running test scenario: {scenario.name}")
                result = await self.run_test_scenario(scenario, wrapper)
                results.append(result)
                self.test_results.append(result)
        
        return results
    
    async def run_performance_benchmark(
        self,
        scenario: TestScenario,
        iterations: int = 100,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Run performance benchmark for a scenario.
        
        Args:
            scenario: Scenario to benchmark
            iterations: Number of iterations to run
            config_overrides: Configuration overrides
            
        Returns:
            Benchmark result
        """
        times: List[float] = []
        successes = 0
        fallbacks = 0
        
        async with self.create_test_wrapper(config_overrides) as wrapper:
            for i in range(iterations):
                start_time = time.time()
                
                try:
                    result = await wrapper.execute(
                        operation_type=f"{scenario.name}_benchmark_{i}",
                        **scenario.inputs
                    )
                    
                    execution_time = (time.time() - start_time) * 1000
                    times.append(execution_time)
                    
                    if result.success:
                        successes += 1
                    
                    if result.fallback_used:
                        fallbacks += 1
                        
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    times.append(execution_time)
                    logger.warning(f"Benchmark iteration {i} failed: {e}")
        
        # Calculate statistics
        total_time = sum(times)
        average_time = total_time / len(times) if times else 0
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0
        success_rate = successes / iterations
        fallback_rate = fallbacks / iterations
        throughput = (iterations / (total_time / 1000)) if total_time > 0 else 0
        
        benchmark_result = BenchmarkResult(
            wrapper_name=wrapper.name,
            scenario_name=scenario.name,
            iterations=iterations,
            total_time_ms=total_time,
            average_time_ms=average_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            success_rate=success_rate,
            fallback_rate=fallback_rate,
            throughput_ops_per_second=throughput
        )
        
        self.benchmark_results.append(benchmark_result)
        return benchmark_result
    
    def _validate_result(self, scenario: TestScenario, result: WrapperResult) -> bool:
        """Validate test result against scenario expectations."""
        if not result:
            return not scenario.should_succeed
        
        # Check success expectation
        if scenario.should_succeed and not result.success:
            return False
        
        if not scenario.should_succeed and result.success:
            return False
        
        # Check fallback expectation
        if scenario.should_use_fallback and not result.fallback_used:
            return False
        
        # Check expected outputs
        if scenario.expected_outputs and result.data:
            for key, expected_value in scenario.expected_outputs.items():
                if isinstance(result.data, dict):
                    if key not in result.data or result.data[key] != expected_value:
                        return False
                else:
                    # For non-dict results, check if expected values are present
                    if not hasattr(result.data, key):
                        return False
                    if getattr(result.data, key) != expected_value:
                        return False
        
        return True
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all test results."""
        if not self.test_results:
            return {"total_tests": 0, "success_rate": 0.0}
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.success])
        failed_tests = total_tests - successful_tests
        
        average_execution_time = sum(r.execution_time_ms for r in self.test_results) / total_tests
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": successful_tests / total_tests,
            "average_execution_time_ms": average_execution_time,
            "test_scenarios": len(self.test_scenarios),
            "benchmark_runs": len(self.benchmark_results)
        }
    
    def generate_test_report(self, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive test report.
        
        Args:
            output_file: Optional file path to save report
            
        Returns:
            Test report as dictionary
        """
        report = {
            "wrapper_class": self.wrapper_class.__name__,
            "config_class": self.config_class.__name__,
            "test_summary": self.get_test_summary(),
            "timestamp": datetime.utcnow().isoformat(),
            "scenarios": [],
            "benchmarks": []
        }
        
        # Add scenario results
        for result in self.test_results:
            scenario_data = {
                "name": result.scenario_name,
                "success": result.success,
                "execution_time_ms": result.execution_time_ms,
                "used_fallback": result.used_fallback,
                "error_message": result.error_message
            }
            report["scenarios"].append(scenario_data)
        
        # Add benchmark results
        for benchmark in self.benchmark_results:
            benchmark_data = {
                "scenario_name": benchmark.scenario_name,
                "iterations": benchmark.iterations,
                "average_time_ms": benchmark.average_time_ms,
                "success_rate": benchmark.success_rate,
                "fallback_rate": benchmark.fallback_rate,
                "throughput_ops_per_second": benchmark.throughput_ops_per_second
            }
            report["benchmarks"].append(benchmark_data)
        
        # Save to file if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Test report saved to {output_file}")
        
        return report


class IntegrationTestSuite:
    """
    Integration test suite for testing wrapper interactions.
    
    Tests how wrappers work together and with the broader system.
    """
    
    def __init__(self):
        self.test_cases: List[Callable] = []
        self.setup_functions: List[Callable] = []
        self.teardown_functions: List[Callable] = []
    
    def setup(self, func: Callable) -> Callable:
        """Decorator to register setup function."""
        self.setup_functions.append(func)
        return func
    
    def teardown(self, func: Callable) -> Callable:
        """Decorator to register teardown function."""
        self.teardown_functions.append(func)
        return func
    
    def test(self, name: str):
        """Decorator to register test case."""
        def decorator(func: Callable) -> Callable:
            func._test_name = name
            self.test_cases.append(func)
            return func
        return decorator
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        results = []
        
        # Run setup functions
        for setup_func in self.setup_functions:
            await self._run_function(setup_func)
        
        try:
            # Run test cases
            for test_func in self.test_cases:
                test_name = getattr(test_func, '_test_name', test_func.__name__)
                
                try:
                    start_time = time.time()
                    await self._run_function(test_func)
                    execution_time = (time.time() - start_time) * 1000
                    
                    results.append({
                        "name": test_name,
                        "success": True,
                        "execution_time_ms": execution_time
                    })
                    
                except Exception as e:
                    execution_time = (time.time() - start_time) * 1000
                    results.append({
                        "name": test_name,
                        "success": False,
                        "execution_time_ms": execution_time,
                        "error": str(e)
                    })
                    logger.error(f"Integration test failed: {test_name} - {e}")
        
        finally:
            # Run teardown functions
            for teardown_func in self.teardown_functions:
                try:
                    await self._run_function(teardown_func)
                except Exception as e:
                    logger.error(f"Teardown function failed: {e}")
        
        # Calculate summary
        total_tests = len(results)
        successful_tests = len([r for r in results if r["success"]])
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 1.0,
            "results": results
        }
    
    async def _run_function(self, func: Callable) -> Any:
        """Run a function (sync or async)."""
        if asyncio.iscoroutinefunction(func):
            return await func()
        else:
            return func()


# Utility functions for creating common test scenarios

def create_basic_scenarios() -> List[TestScenario]:
    """Create basic test scenarios that all wrappers should support."""
    return [
        TestScenario(
            name="basic_success",
            description="Basic successful operation",
            inputs={"test_input": "basic_test"},
            expected_outputs={"success": True},
            should_succeed=True
        ),
        TestScenario(
            name="fallback_test",
            description="Test fallback mechanism",
            inputs={"force_fallback": True},
            expected_outputs={"fallback": True},
            should_succeed=True,
            should_use_fallback=True
        ),
        TestScenario(
            name="error_handling",
            description="Test error handling",
            inputs={"force_error": True},
            expected_outputs={},
            should_succeed=False,
            expected_error="forced error"
        )
    ]


def create_performance_scenarios() -> List[TestScenario]:
    """Create performance-focused test scenarios."""
    return [
        TestScenario(
            name="quick_operation",
            description="Quick operation for latency testing",
            inputs={"size": "small"},
            expected_outputs={"processed": True},
            should_succeed=True,
            tags=["performance", "latency"]
        ),
        TestScenario(
            name="large_operation",
            description="Large operation for throughput testing",
            inputs={"size": "large", "iterations": 1000},
            expected_outputs={"processed": True},
            should_succeed=True,
            tags=["performance", "throughput"]
        )
    ]


def create_stress_scenarios() -> List[TestScenario]:
    """Create stress testing scenarios."""
    return [
        TestScenario(
            name="concurrent_operations",
            description="Multiple concurrent operations",
            inputs={"concurrency": 10},
            expected_outputs={"all_completed": True},
            should_succeed=True,
            tags=["stress", "concurrency"]
        ),
        TestScenario(
            name="resource_exhaustion",
            description="Test resource exhaustion handling",
            inputs={"memory_size": "huge"},
            expected_outputs={},
            should_succeed=False,
            tags=["stress", "resources"]
        )
    ]