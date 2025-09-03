#!/usr/bin/env python3
"""
Performance regression testing for wrapper integrations - Issue #252.

This module provides comprehensive performance testing to ensure wrapper
integrations don't introduce significant performance degradation.
"""

import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import psutil
import sys
import os

import pytest

# Add orchestrator to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orchestrator import Orchestrator, init_models
from src.orchestrator.models import get_model_registry
from src.orchestrator.compiler.yaml_compiler import YAMLCompiler
from src.orchestrator.control_systems.hybrid_control_system import HybridControlSystem

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """Baseline performance metrics."""
    
    pipeline_name: str
    execution_time_ms: float
    memory_usage_mb: float
    api_calls: int
    tokens_used: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass  
class PerformanceMeasurement:
    """Performance measurement result."""
    
    pipeline_name: str
    wrapper_config: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    api_calls: int
    tokens_used: int
    cache_hits: int = 0
    cache_misses: int = 0
    wrapper_overhead_ms: float = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RegressionTestResult:
    """Result of regression testing."""
    
    pipeline_name: str
    wrapper_config: str
    baseline: PerformanceBaseline
    measurement: PerformanceMeasurement
    performance_delta_percent: float
    memory_delta_percent: float
    is_regression: bool
    regression_threshold_percent: float = 10.0  # 10% degradation threshold
    

class PerformanceRegressionTester:
    """
    Performance regression tester for wrapper integrations.
    
    Measures performance impact of wrapper implementations and
    detects regressions beyond acceptable thresholds.
    """
    
    def __init__(self):
        self.examples_dir = Path("examples")
        self.results_dir = Path("tests/performance/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_file = self.results_dir / "performance_baselines.json"
        
        # Performance test pipelines (subset for speed)
        self.test_pipelines = [
            "simple_data_processing.yaml",
            "research_minimal.yaml", 
            "control_flow_conditional.yaml",
            "auto_tags_demo.yaml",
            "validation_pipeline.yaml"
        ]
        
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.measurements: List[PerformanceMeasurement] = []
        self.regression_results: List[RegressionTestResult] = []
        
        self.model_registry = None
        self.control_system = None
        
    async def initialize(self):
        """Initialize testing infrastructure."""
        logger.info("Initializing performance regression tester...")
        
        self.model_registry = init_models()
        if not self.model_registry or not self.model_registry.models:
            raise RuntimeError("No models available for performance testing")
            
        self.control_system = HybridControlSystem(self.model_registry)
        
        # Load existing baselines
        await self.load_baselines()
        
    async def load_baselines(self):
        """Load performance baselines from file."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file) as f:
                    baseline_data = json.load(f)
                
                for name, data in baseline_data.items():
                    self.baselines[name] = PerformanceBaseline(
                        pipeline_name=data["pipeline_name"],
                        execution_time_ms=data["execution_time_ms"],
                        memory_usage_mb=data["memory_usage_mb"],
                        api_calls=data["api_calls"],
                        tokens_used=data["tokens_used"],
                        timestamp=datetime.fromisoformat(data["timestamp"])
                    )
                    
                logger.info(f"Loaded {len(self.baselines)} performance baselines")
            except Exception as e:
                logger.warning(f"Could not load baselines: {e}")
                
    async def save_baselines(self):
        """Save performance baselines to file."""
        baseline_data = {}
        for name, baseline in self.baselines.items():
            baseline_data[name] = {
                "pipeline_name": baseline.pipeline_name,
                "execution_time_ms": baseline.execution_time_ms,
                "memory_usage_mb": baseline.memory_usage_mb,
                "api_calls": baseline.api_calls,
                "tokens_used": baseline.tokens_used,
                "timestamp": baseline.timestamp.isoformat()
            }
            
        with open(self.baseline_file, "w") as f:
            json.dump(baseline_data, f, indent=2)
            
        logger.info(f"Saved {len(self.baselines)} performance baselines")
        
    async def measure_baseline_performance(self, pipeline_name: str) -> PerformanceBaseline:
        """Measure baseline performance for a pipeline."""
        logger.info(f"Measuring baseline performance for {pipeline_name}")
        
        pipeline_path = self.examples_dir / pipeline_name
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_name}")
            
        # Load pipeline
        with open(pipeline_path) as f:
            yaml_content = f.read()
            
        # Get test inputs
        inputs = self._get_test_inputs(pipeline_name)
        
        # Setup clean orchestrator (no wrappers)
        orchestrator = Orchestrator(
            model_registry=self.model_registry,
            control_system=self.control_system
        )
        
        # Measure performance
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        results = await orchestrator.execute_yaml(yaml_content, inputs)
        execution_time = (time.perf_counter() - start_time) * 1000
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        # Collect metrics
        api_calls = getattr(orchestrator, '_api_call_count', 0)
        tokens_used = getattr(orchestrator, '_total_tokens', 0)
        
        baseline = PerformanceBaseline(
            pipeline_name=pipeline_name,
            execution_time_ms=execution_time,
            memory_usage_mb=memory_usage,
            api_calls=api_calls,
            tokens_used=tokens_used
        )
        
        logger.info(f"Baseline for {pipeline_name}: "
                   f"{execution_time:.1f}ms, {memory_usage:.1f}MB, {api_calls} API calls")
        
        return baseline
        
    async def measure_wrapper_performance(
        self, 
        pipeline_name: str, 
        wrapper_config: Dict[str, Any]
    ) -> PerformanceMeasurement:
        """Measure performance with wrapper configuration."""
        logger.info(f"Measuring wrapper performance: {pipeline_name} with {wrapper_config.get('name', 'unknown')}")
        
        pipeline_path = self.examples_dir / pipeline_name
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_name}")
            
        # Load pipeline
        with open(pipeline_path) as f:
            yaml_content = f.read()
            
        # Get test inputs
        inputs = self._get_test_inputs(pipeline_name)
        
        # Setup orchestrator with wrapper configuration
        orchestrator = await self._create_configured_orchestrator(wrapper_config)
        
        # Measure performance
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        cpu_before = process.cpu_percent()
        
        start_time = time.perf_counter()
        results = await orchestrator.execute_yaml(yaml_content, inputs)
        execution_time = (time.perf_counter() - start_time) * 1000
        
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_usage = memory_after - memory_before
        cpu_usage = process.cpu_percent() - cpu_before
        
        # Collect metrics
        api_calls = getattr(orchestrator, '_api_call_count', 0)
        tokens_used = getattr(orchestrator, '_total_tokens', 0)
        cache_hits = getattr(orchestrator, '_cache_hits', 0)
        cache_misses = getattr(orchestrator, '_cache_misses', 0)
        wrapper_overhead = getattr(orchestrator, '_wrapper_overhead_ms', 0)
        
        measurement = PerformanceMeasurement(
            pipeline_name=pipeline_name,
            wrapper_config=wrapper_config.get('name', 'unknown'),
            execution_time_ms=execution_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            api_calls=api_calls,
            tokens_used=tokens_used,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            wrapper_overhead_ms=wrapper_overhead
        )
        
        logger.info(f"Wrapper measurement for {pipeline_name}: "
                   f"{execution_time:.1f}ms, {memory_usage:.1f}MB, overhead: {wrapper_overhead:.1f}ms")
        
        return measurement
        
    async def _create_configured_orchestrator(self, wrapper_config: Dict[str, Any]) -> Orchestrator:
        """Create orchestrator with wrapper configuration."""
        orchestrator = Orchestrator(
            model_registry=self.model_registry,
            control_system=self.control_system
        )
        
        # Apply wrapper configuration
        if hasattr(orchestrator, 'configure_wrappers'):
            await orchestrator.configure_wrappers(wrapper_config)
            
        return orchestrator
        
    def _get_test_inputs(self, pipeline_name: str) -> Dict[str, Any]:
        """Get consistent test inputs for performance testing."""
        return {
            "input_text": "Performance testing with consistent input data for regression analysis.",
            "topic": "performance optimization",
            "query": "system performance metrics",
            "data": {"test": True, "performance": True},
            "output_path": str(self.results_dir / f"perf_test_{int(time.time())}")
        }
        
    def analyze_regression(
        self, 
        baseline: PerformanceBaseline, 
        measurement: PerformanceMeasurement,
        threshold_percent: float = 10.0
    ) -> RegressionTestResult:
        """Analyze performance regression between baseline and measurement."""
        
        # Calculate performance deltas
        performance_delta = ((measurement.execution_time_ms - baseline.execution_time_ms) 
                           / baseline.execution_time_ms * 100)
        memory_delta = ((measurement.memory_usage_mb - baseline.memory_usage_mb) 
                       / max(baseline.memory_usage_mb, 1) * 100)
        
        # Check for regression
        is_regression = (performance_delta > threshold_percent or 
                        memory_delta > threshold_percent * 2)  # More lenient for memory
        
        result = RegressionTestResult(
            pipeline_name=measurement.pipeline_name,
            wrapper_config=measurement.wrapper_config,
            baseline=baseline,
            measurement=measurement,
            performance_delta_percent=performance_delta,
            memory_delta_percent=memory_delta,
            is_regression=is_regression,
            regression_threshold_percent=threshold_percent
        )
        
        return result
        
    async def run_performance_benchmarks(
        self, 
        wrapper_configs: List[Dict[str, Any]],
        iterations: int = 3
    ) -> List[RegressionTestResult]:
        """Run performance benchmarks for all test pipelines."""
        logger.info(f"Running performance benchmarks with {iterations} iterations")
        
        # Ensure we have baselines
        for pipeline_name in self.test_pipelines:
            if pipeline_name not in self.baselines:
                logger.info(f"Creating baseline for {pipeline_name}")
                baseline = await self.measure_baseline_performance(pipeline_name)
                self.baselines[pipeline_name] = baseline
                
        await self.save_baselines()
        
        # Run wrapper performance tests
        results = []
        for wrapper_config in wrapper_configs:
            for pipeline_name in self.test_pipelines:
                # Run multiple iterations and take average
                measurements = []
                for i in range(iterations):
                    logger.info(f"Iteration {i+1}/{iterations} for {pipeline_name} with {wrapper_config.get('name')}")
                    measurement = await self.measure_wrapper_performance(pipeline_name, wrapper_config)
                    measurements.append(measurement)
                    
                # Calculate average measurement
                avg_measurement = self._average_measurements(measurements)
                self.measurements.append(avg_measurement)
                
                # Analyze regression
                baseline = self.baselines[pipeline_name]
                regression_result = self.analyze_regression(baseline, avg_measurement)
                results.append(regression_result)
                self.regression_results.append(regression_result)
                
        return results
        
    def _average_measurements(self, measurements: List[PerformanceMeasurement]) -> PerformanceMeasurement:
        """Calculate average of multiple measurements."""
        if not measurements:
            raise ValueError("No measurements to average")
            
        avg_measurement = PerformanceMeasurement(
            pipeline_name=measurements[0].pipeline_name,
            wrapper_config=measurements[0].wrapper_config,
            execution_time_ms=statistics.mean(m.execution_time_ms for m in measurements),
            memory_usage_mb=statistics.mean(m.memory_usage_mb for m in measurements),
            cpu_usage_percent=statistics.mean(m.cpu_usage_percent for m in measurements),
            api_calls=int(statistics.mean(m.api_calls for m in measurements)),
            tokens_used=int(statistics.mean(m.tokens_used for m in measurements)),
            cache_hits=int(statistics.mean(m.cache_hits for m in measurements)),
            cache_misses=int(statistics.mean(m.cache_misses for m in measurements)),
            wrapper_overhead_ms=statistics.mean(m.wrapper_overhead_ms for m in measurements)
        )
        
        return avg_measurement
        
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance regression report."""
        logger.info("Generating performance regression report")
        
        # Calculate summary statistics
        total_tests = len(self.regression_results)
        regressions = [r for r in self.regression_results if r.is_regression]
        regression_count = len(regressions)
        
        # Performance statistics
        performance_deltas = [r.performance_delta_percent for r in self.regression_results]
        memory_deltas = [r.memory_delta_percent for r in self.regression_results]
        
        report = {
            "summary": {
                "timestamp": datetime.utcnow().isoformat(),
                "total_tests": total_tests,
                "regressions_detected": regression_count,
                "regression_rate": regression_count / total_tests if total_tests > 0 else 0,
                "performance_threshold_percent": 10.0
            },
            "performance_statistics": {
                "average_performance_delta_percent": statistics.mean(performance_deltas) if performance_deltas else 0,
                "max_performance_delta_percent": max(performance_deltas) if performance_deltas else 0,
                "min_performance_delta_percent": min(performance_deltas) if performance_deltas else 0,
                "average_memory_delta_percent": statistics.mean(memory_deltas) if memory_deltas else 0,
                "max_memory_delta_percent": max(memory_deltas) if memory_deltas else 0,
                "min_memory_delta_percent": min(memory_deltas) if memory_deltas else 0
            },
            "regressions": [
                {
                    "pipeline": r.pipeline_name,
                    "wrapper_config": r.wrapper_config,
                    "performance_delta_percent": r.performance_delta_percent,
                    "memory_delta_percent": r.memory_delta_percent,
                    "baseline_time_ms": r.baseline.execution_time_ms,
                    "measured_time_ms": r.measurement.execution_time_ms,
                    "wrapper_overhead_ms": r.measurement.wrapper_overhead_ms
                }
                for r in regressions
            ],
            "detailed_results": [
                {
                    "pipeline": r.pipeline_name,
                    "wrapper_config": r.wrapper_config,
                    "is_regression": r.is_regression,
                    "performance_delta_percent": r.performance_delta_percent,
                    "memory_delta_percent": r.memory_delta_percent,
                    "baseline": {
                        "execution_time_ms": r.baseline.execution_time_ms,
                        "memory_usage_mb": r.baseline.memory_usage_mb,
                        "api_calls": r.baseline.api_calls,
                        "tokens_used": r.baseline.tokens_used
                    },
                    "measurement": {
                        "execution_time_ms": r.measurement.execution_time_ms,
                        "memory_usage_mb": r.measurement.memory_usage_mb,
                        "cpu_usage_percent": r.measurement.cpu_usage_percent,
                        "api_calls": r.measurement.api_calls,
                        "tokens_used": r.measurement.tokens_used,
                        "cache_hits": r.measurement.cache_hits,
                        "cache_misses": r.measurement.cache_misses,
                        "wrapper_overhead_ms": r.measurement.wrapper_overhead_ms
                    }
                }
                for r in self.regression_results
            ]
        }
        
        # Save report
        report_path = self.results_dir / f"performance_regression_report_{int(time.time())}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Performance report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("PERFORMANCE REGRESSION TESTING REPORT")
        print("="*80)
        print(f"\nTotal Tests: {total_tests}")
        print(f"Regressions Detected: {regression_count} ({regression_count/total_tests*100:.1f}%)")
        
        if performance_deltas:
            print(f"\nPerformance Impact:")
            print(f"Average: {statistics.mean(performance_deltas):+.1f}%")
            print(f"Range: {min(performance_deltas):+.1f}% to {max(performance_deltas):+.1f}%")
            
        if memory_deltas:
            print(f"\nMemory Impact:")  
            print(f"Average: {statistics.mean(memory_deltas):+.1f}%")
            print(f"Range: {min(memory_deltas):+.1f}% to {max(memory_deltas):+.1f}%")
        
        if regressions:
            print(f"\n❌ Performance Regressions Detected:")
            for regression in regressions:
                print(f"  {regression['pipeline']} ({regression['wrapper_config']}): "
                      f"{regression['performance_delta_percent']:+.1f}% slower")
        else:
            print("\n✅ No significant performance regressions detected!")
            
        return report


# Test wrapper configurations for performance testing
PERFORMANCE_TEST_CONFIGS = [
    {
        "name": "baseline",
        "wrapper_enabled": False,
        "description": "Baseline without wrappers"
    },
    {
        "name": "routellm_optimized",
        "wrapper_enabled": True,
        "routellm_enabled": True,
        "routing_strategy": "cost",
        "description": "RouteLLM cost optimization"
    },
    {
        "name": "poml_enhanced",
        "wrapper_enabled": True,
        "poml_enabled": True,
        "enhanced_templates": True,
        "description": "POML template enhancements"
    },
    {
        "name": "full_wrappers",
        "wrapper_enabled": True,
        "routellm_enabled": True,
        "poml_enabled": True,
        "monitoring_enabled": True,
        "description": "All wrappers enabled"
    }
]


# pytest fixtures and tests

@pytest.fixture
async def performance_tester():
    """Create and initialize performance tester."""
    tester = PerformanceRegressionTester()
    await tester.initialize()
    return tester


@pytest.mark.asyncio
async def test_wrapper_performance_overhead(performance_tester):
    """Test that wrapper overhead is within acceptable limits."""
    tester = performance_tester
    
    # Test with simple pipeline
    pipeline_name = "simple_data_processing.yaml"
    
    # Measure baseline if not exists
    if pipeline_name not in tester.baselines:
        baseline = await tester.measure_baseline_performance(pipeline_name)
        tester.baselines[pipeline_name] = baseline
        
    # Test wrapper configurations
    for config in PERFORMANCE_TEST_CONFIGS[1:]:  # Skip baseline
        measurement = await tester.measure_wrapper_performance(pipeline_name, config)
        
        # Assert wrapper overhead is minimal
        assert measurement.wrapper_overhead_ms < 5.0, \
            f"Wrapper overhead {measurement.wrapper_overhead_ms}ms exceeds 5ms limit"


@pytest.mark.asyncio
async def test_no_performance_regression(performance_tester):
    """Test that wrapper integration doesn't cause performance regression."""
    tester = performance_tester
    
    results = await tester.run_performance_benchmarks(
        PERFORMANCE_TEST_CONFIGS, iterations=2
    )
    
    # Check for regressions
    regressions = [r for r in results if r.is_regression]
    
    # Allow minor regressions but not major ones
    major_regressions = [r for r in regressions if r.performance_delta_percent > 25]
    
    assert len(major_regressions) == 0, \
        f"Major performance regressions detected: {[r.pipeline_name for r in major_regressions]}"
        
    # Warn about minor regressions
    if regressions:
        logger.warning(f"{len(regressions)} minor performance regressions detected")


@pytest.mark.asyncio
async def test_memory_usage_within_limits(performance_tester):
    """Test that memory usage stays within reasonable limits."""
    tester = performance_tester
    
    # Test each pipeline with full wrapper stack
    full_config = next(c for c in PERFORMANCE_TEST_CONFIGS if c["name"] == "full_wrappers")
    
    for pipeline_name in tester.test_pipelines[:3]:  # Test subset for speed
        measurement = await tester.measure_wrapper_performance(pipeline_name, full_config)
        
        # Assert memory usage is reasonable (< 500MB additional)
        assert measurement.memory_usage_mb < 500, \
            f"Memory usage {measurement.memory_usage_mb}MB exceeds 500MB limit for {pipeline_name}"


if __name__ == "__main__":
    async def main():
        tester = PerformanceRegressionTester()
        await tester.initialize()
        
        results = await tester.run_performance_benchmarks(
            PERFORMANCE_TEST_CONFIGS, iterations=3
        )
        
        report = tester.generate_performance_report()
        
        # Summary for Issue #252
        regressions = len([r for r in results if r.is_regression])
        if regressions == 0:
            print("\n🎉 Issue #252: No performance regressions detected!")
        else:
            print(f"\n⚠️  Issue #252: {regressions} performance regressions need attention")
            
    asyncio.run(main())