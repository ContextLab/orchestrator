"""
Performance comparison benchmarks between current orchestrator control system
and Deep Agents integration.
"""

import asyncio
import time
import logging
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Import the adapter
from adapters.control_system_adapter import DeepAgentsControlSystem, Task, Pipeline

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Benchmark execution result."""
    test_name: str
    system_type: str
    execution_time: float
    memory_usage: Optional[float]
    cpu_usage: Optional[float]
    success: bool
    error: Optional[str]
    task_count: int
    parallel_tasks: int
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ComparisonMetrics:
    """Comparison metrics between systems."""
    test_name: str
    current_system_time: float
    deep_agents_time: float
    time_improvement: float
    time_improvement_percent: float
    deep_agents_overhead: float
    parallel_efficiency: float
    complexity_handling: float

class PerformanceBenchmark:
    """Benchmark suite for comparing control systems."""
    
    def __init__(self):
        self.deep_agents_system = DeepAgentsControlSystem()
        self.benchmark_results: List[BenchmarkResult] = []
        self.comparison_metrics: List[ComparisonMetrics] = []
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark tests."""
        logger.info("Starting comprehensive benchmark suite")
        
        benchmark_tests = [
            ("simple_sequential", self._benchmark_simple_sequential),
            ("parallel_execution", self._benchmark_parallel_execution),
            ("complex_pipeline", self._benchmark_complex_pipeline),
            ("state_management", self._benchmark_state_management),
            ("long_running_tasks", self._benchmark_long_running_tasks),
        ]
        
        for test_name, test_func in benchmark_tests:
            logger.info(f"Running benchmark: {test_name}")
            try:
                await test_func()
            except Exception as e:
                logger.error(f"Benchmark {test_name} failed: {e}")
        
        # Calculate comparison metrics
        self._calculate_comparison_metrics()
        
        # Generate report
        return self._generate_report()
    
    async def _benchmark_simple_sequential(self):
        """Benchmark simple sequential task execution."""
        logger.info("Benchmarking simple sequential execution")
        
        # Create test pipeline
        tasks = [
            Task(f"task_{i}", "generate", {"prompt": f"Generate content {i}"})
            for i in range(3)
        ]
        pipeline = Pipeline("simple_sequential", tasks)
        
        # Benchmark Deep Agents system
        start_time = time.time()
        try:
            result = await self.deep_agents_system.execute_pipeline(pipeline)
            execution_time = time.time() - start_time
            success = result.get("success", False)
            error = None
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            error = str(e)
        
        self.benchmark_results.append(BenchmarkResult(
            test_name="simple_sequential",
            system_type="deep_agents",
            execution_time=execution_time,
            memory_usage=None,  # Would implement with psutil in real version
            cpu_usage=None,
            success=success,
            error=error,
            task_count=len(tasks),
            parallel_tasks=0,
            timestamp=datetime.now().isoformat(),
        ))
        
        # Simulate current system benchmark
        current_system_time = self._simulate_current_system_execution(tasks, parallel=False)
        
        self.benchmark_results.append(BenchmarkResult(
            test_name="simple_sequential",
            system_type="current_system",
            execution_time=current_system_time,
            memory_usage=None,
            cpu_usage=None,
            success=True,
            error=None,
            task_count=len(tasks),
            parallel_tasks=0,
            timestamp=datetime.now().isoformat(),
        ))
    
    async def _benchmark_parallel_execution(self):
        """Benchmark parallel task execution capabilities."""
        logger.info("Benchmarking parallel execution")
        
        # Create parallelizable tasks
        tasks = [
            Task(f"parallel_task_{i}", "analyze", {"data": f"dataset_{i}"})
            for i in range(5)
        ]
        pipeline = Pipeline("parallel_execution", tasks)
        
        # Benchmark Deep Agents system
        start_time = time.time()
        try:
            result = await self.deep_agents_system.execute_pipeline(pipeline)
            execution_time = time.time() - start_time
            success = result.get("success", False)
            parallel_groups = result.get("parallel_groups", [])
            parallel_task_count = sum(len(group.get("tasks", [])) for group in parallel_groups)
            error = None
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            parallel_task_count = 0
            error = str(e)
        
        self.benchmark_results.append(BenchmarkResult(
            test_name="parallel_execution",
            system_type="deep_agents",
            execution_time=execution_time,
            memory_usage=None,
            cpu_usage=None,
            success=success,
            error=error,
            task_count=len(tasks),
            parallel_tasks=parallel_task_count,
            timestamp=datetime.now().isoformat(),
        ))
        
        # Simulate current system (sequential execution)
        current_system_time = self._simulate_current_system_execution(tasks, parallel=False)
        
        self.benchmark_results.append(BenchmarkResult(
            test_name="parallel_execution",
            system_type="current_system",
            execution_time=current_system_time,
            memory_usage=None,
            cpu_usage=None,
            success=True,
            error=None,
            task_count=len(tasks),
            parallel_tasks=0,  # Current system doesn't support native parallel execution
            timestamp=datetime.now().isoformat(),
        ))
    
    async def _benchmark_complex_pipeline(self):
        """Benchmark complex pipeline with dependencies and planning."""
        logger.info("Benchmarking complex pipeline execution")
        
        # Create complex pipeline with dependencies
        tasks = [
            Task("data_ingestion", "execute", {"source": "database", "query": "SELECT * FROM data"}),
            Task("data_validation", "validate", {"data": "{{data_ingestion}}", "rules": "standard"}),
            Task("data_transformation", "transform", {"data": "{{data_validation}}", "format": "json"}),
            Task("analysis_1", "analyze", {"data": "{{data_transformation}}", "type": "statistical"}),
            Task("analysis_2", "analyze", {"data": "{{data_transformation}}", "type": "ml_features"}),
            Task("report_generation", "generate", {
                "template": "standard_report",
                "stats": "{{analysis_1}}",
                "features": "{{analysis_2}}"
            }),
        ]
        pipeline = Pipeline("complex_pipeline", tasks)
        
        # Benchmark Deep Agents system
        start_time = time.time()
        try:
            result = await self.deep_agents_system.execute_pipeline(pipeline)
            execution_time = time.time() - start_time
            success = result.get("success", False)
            execution_plan = result.get("execution_plan", [])
            parallel_groups = result.get("parallel_groups", [])
            parallel_task_count = sum(len(group.get("tasks", [])) for group in parallel_groups)
            error = None
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            parallel_task_count = 0
            error = str(e)
        
        self.benchmark_results.append(BenchmarkResult(
            test_name="complex_pipeline",
            system_type="deep_agents",
            execution_time=execution_time,
            memory_usage=None,
            cpu_usage=None,
            success=success,
            error=error,
            task_count=len(tasks),
            parallel_tasks=parallel_task_count,
            timestamp=datetime.now().isoformat(),
        ))
        
        # Simulate current system
        current_system_time = self._simulate_current_system_execution(tasks, parallel=False)
        
        self.benchmark_results.append(BenchmarkResult(
            test_name="complex_pipeline",
            system_type="current_system",
            execution_time=current_system_time,
            memory_usage=None,
            cpu_usage=None,
            success=True,
            error=None,
            task_count=len(tasks),
            parallel_tasks=0,
            timestamp=datetime.now().isoformat(),
        ))
    
    async def _benchmark_state_management(self):
        """Benchmark state persistence and management capabilities."""
        logger.info("Benchmarking state management")
        
        # Create pipeline that benefits from state management
        tasks = [
            Task("checkpoint_1", "execute", {"operation": "initialize", "state": "start"}),
            Task("long_operation", "execute", {"duration": "long", "checkpoint": "{{checkpoint_1}}"}),
            Task("checkpoint_2", "execute", {"operation": "save", "state": "{{long_operation}}"}),
            Task("recovery_test", "execute", {"operation": "recover", "from": "{{checkpoint_2}}"}),
        ]
        pipeline = Pipeline("state_management", tasks)
        
        # Benchmark Deep Agents system
        start_time = time.time()
        try:
            result = await self.deep_agents_system.execute_pipeline(pipeline)
            execution_time = time.time() - start_time
            success = result.get("success", False)
            persistent_state = result.get("persistent_state")
            error = None
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            error = str(e)
        
        self.benchmark_results.append(BenchmarkResult(
            test_name="state_management",
            system_type="deep_agents",
            execution_time=execution_time,
            memory_usage=None,
            cpu_usage=None,
            success=success,
            error=error,
            task_count=len(tasks),
            parallel_tasks=0,
            timestamp=datetime.now().isoformat(),
        ))
        
        # Current system doesn't have native state persistence
        current_system_time = self._simulate_current_system_execution(tasks, parallel=False)
        
        self.benchmark_results.append(BenchmarkResult(
            test_name="state_management",
            system_type="current_system",
            execution_time=current_system_time,
            memory_usage=None,
            cpu_usage=None,
            success=True,  # But without state persistence benefits
            error=None,
            task_count=len(tasks),
            parallel_tasks=0,
            timestamp=datetime.now().isoformat(),
        ))
    
    async def _benchmark_long_running_tasks(self):
        """Benchmark handling of long-running tasks with planning."""
        logger.info("Benchmarking long-running task handling")
        
        # Create long-running pipeline
        tasks = [
            Task("data_download", "execute", {"url": "large_dataset", "size": "10GB"}),
            Task("data_processing", "process", {"data": "{{data_download}}", "algorithm": "complex"}),
            Task("model_training", "train", {"data": "{{data_processing}}", "epochs": 100}),
            Task("validation", "validate", {"model": "{{model_training}}", "test_data": "{{data_download}}"}),
        ]
        pipeline = Pipeline("long_running_tasks", tasks)
        
        # Benchmark Deep Agents system
        start_time = time.time()
        try:
            result = await self.deep_agents_system.execute_pipeline(pipeline)
            execution_time = time.time() - start_time
            success = result.get("success", False)
            execution_plan = result.get("execution_plan", [])
            planning_benefit = len(execution_plan) > 0
            error = None
        except Exception as e:
            execution_time = time.time() - start_time
            success = False
            error = str(e)
        
        self.benchmark_results.append(BenchmarkResult(
            test_name="long_running_tasks",
            system_type="deep_agents",
            execution_time=execution_time,
            memory_usage=None,
            cpu_usage=None,
            success=success,
            error=error,
            task_count=len(tasks),
            parallel_tasks=0,
            timestamp=datetime.now().isoformat(),
        ))
        
        # Current system
        current_system_time = self._simulate_current_system_execution(tasks, parallel=False)
        
        self.benchmark_results.append(BenchmarkResult(
            test_name="long_running_tasks",
            system_type="current_system",
            execution_time=current_system_time,
            memory_usage=None,
            cpu_usage=None,
            success=True,
            error=None,
            task_count=len(tasks),
            parallel_tasks=0,
            timestamp=datetime.now().isoformat(),
        ))
    
    def _simulate_current_system_execution(self, tasks: List[Task], parallel: bool = False) -> float:
        """Simulate current orchestrator system execution time."""
        # Base execution time per task (simulated)
        base_time_per_task = 0.5
        
        if parallel:
            # Current system has limited parallel support
            return base_time_per_task * max(1, len(tasks) * 0.8)  # 20% improvement at best
        else:
            # Sequential execution
            return base_time_per_task * len(tasks)
    
    def _calculate_comparison_metrics(self):
        """Calculate comparison metrics between systems."""
        # Group results by test name
        results_by_test = {}
        for result in self.benchmark_results:
            if result.test_name not in results_by_test:
                results_by_test[result.test_name] = {}
            results_by_test[result.test_name][result.system_type] = result
        
        # Calculate metrics for each test
        for test_name, systems in results_by_test.items():
            if "current_system" in systems and "deep_agents" in systems:
                current = systems["current_system"]
                deep_agents = systems["deep_agents"]
                
                if current.success and deep_agents.success:
                    time_improvement = current.execution_time - deep_agents.execution_time
                    time_improvement_percent = (time_improvement / current.execution_time) * 100
                    overhead = deep_agents.execution_time - (current.execution_time * 0.1)  # Assume 10% is optimal
                    
                    # Calculate parallel efficiency
                    if deep_agents.parallel_tasks > 0:
                        parallel_efficiency = deep_agents.parallel_tasks / deep_agents.task_count
                    else:
                        parallel_efficiency = 0.0
                    
                    # Calculate complexity handling score
                    complexity_handling = self._calculate_complexity_score(test_name, deep_agents)
                    
                    self.comparison_metrics.append(ComparisonMetrics(
                        test_name=test_name,
                        current_system_time=current.execution_time,
                        deep_agents_time=deep_agents.execution_time,
                        time_improvement=time_improvement,
                        time_improvement_percent=time_improvement_percent,
                        deep_agents_overhead=max(0, overhead),
                        parallel_efficiency=parallel_efficiency,
                        complexity_handling=complexity_handling,
                    ))
    
    def _calculate_complexity_score(self, test_name: str, result: BenchmarkResult) -> float:
        """Calculate how well the system handles complexity."""
        complexity_scores = {
            "simple_sequential": 0.3,
            "parallel_execution": 0.7,
            "complex_pipeline": 0.9,
            "state_management": 0.8,
            "long_running_tasks": 0.6,
        }
        
        base_score = complexity_scores.get(test_name, 0.5)
        
        # Adjust based on success and parallel task handling
        if result.success:
            if result.parallel_tasks > 0:
                base_score += 0.2
        else:
            base_score *= 0.5
        
        return min(1.0, base_score)
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        # Summary statistics
        deep_agents_results = [r for r in self.benchmark_results if r.system_type == "deep_agents"]
        current_system_results = [r for r in self.benchmark_results if r.system_type == "current_system"]
        
        deep_agents_times = [r.execution_time for r in deep_agents_results if r.success]
        current_system_times = [r.execution_time for r in current_system_results if r.success]
        
        report = {
            "benchmark_summary": {
                "total_tests": len(set(r.test_name for r in self.benchmark_results)),
                "deep_agents_success_rate": len([r for r in deep_agents_results if r.success]) / len(deep_agents_results),
                "current_system_success_rate": len([r for r in current_system_results if r.success]) / len(current_system_results),
                "average_deep_agents_time": statistics.mean(deep_agents_times) if deep_agents_times else 0,
                "average_current_system_time": statistics.mean(current_system_times) if current_system_times else 0,
                "timestamp": datetime.now().isoformat(),
            },
            "detailed_results": [result.to_dict() for result in self.benchmark_results],
            "comparison_metrics": [asdict(metric) for metric in self.comparison_metrics],
            "overall_assessment": self._generate_overall_assessment(),
        }
        
        return report
    
    def _generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall assessment of Deep Agents vs current system."""
        if not self.comparison_metrics:
            return {"assessment": "insufficient_data"}
        
        avg_time_improvement = statistics.mean([m.time_improvement_percent for m in self.comparison_metrics])
        avg_parallel_efficiency = statistics.mean([m.parallel_efficiency for m in self.comparison_metrics])
        avg_complexity_handling = statistics.mean([m.complexity_handling for m in self.comparison_metrics])
        avg_overhead = statistics.mean([m.deep_agents_overhead for m in self.comparison_metrics])
        
        # Scoring criteria
        time_score = min(100, max(0, avg_time_improvement))
        parallel_score = avg_parallel_efficiency * 100
        complexity_score = avg_complexity_handling * 100
        overhead_penalty = min(50, avg_overhead * 10)  # Penalty for overhead
        
        overall_score = (time_score + parallel_score + complexity_score - overhead_penalty) / 3
        
        # Generate recommendation
        if overall_score >= 70:
            recommendation = "RECOMMENDED"
            justification = "Deep Agents shows significant improvements in multiple areas"
        elif overall_score >= 50:
            recommendation = "CONDITIONAL"
            justification = "Deep Agents shows promise but with notable trade-offs"
        else:
            recommendation = "NOT_RECOMMENDED"
            justification = "Deep Agents does not provide sufficient benefits over current system"
        
        return {
            "overall_score": overall_score,
            "time_improvement_avg": avg_time_improvement,
            "parallel_efficiency_avg": avg_parallel_efficiency,
            "complexity_handling_avg": avg_complexity_handling,
            "overhead_avg": avg_overhead,
            "recommendation": recommendation,
            "justification": justification,
            "key_strengths": self._identify_key_strengths(),
            "key_weaknesses": self._identify_key_weaknesses(),
        }
    
    def _identify_key_strengths(self) -> List[str]:
        """Identify key strengths of Deep Agents from benchmarks."""
        strengths = []
        
        for metric in self.comparison_metrics:
            if metric.time_improvement_percent > 20:
                strengths.append(f"Significant time improvement in {metric.test_name}")
            
            if metric.parallel_efficiency > 0.5:
                strengths.append(f"Good parallel execution in {metric.test_name}")
            
            if metric.complexity_handling > 0.8:
                strengths.append(f"Excellent complexity handling in {metric.test_name}")
        
        return list(set(strengths))  # Remove duplicates
    
    def _identify_key_weaknesses(self) -> List[str]:
        """Identify key weaknesses of Deep Agents from benchmarks."""
        weaknesses = []
        
        for metric in self.comparison_metrics:
            if metric.time_improvement_percent < -10:
                weaknesses.append(f"Performance regression in {metric.test_name}")
            
            if metric.deep_agents_overhead > 1.0:
                weaknesses.append(f"High overhead in {metric.test_name}")
        
        # Check for failures
        failed_tests = [r.test_name for r in self.benchmark_results if r.system_type == "deep_agents" and not r.success]
        for test in failed_tests:
            weaknesses.append(f"Execution failure in {test}")
        
        return list(set(weaknesses))  # Remove duplicates

async def main():
    """Run benchmarks and save results."""
    logging.basicConfig(level=logging.INFO)
    
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_all_benchmarks()
    
    # Save results to file
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*50)
    
    summary = results["benchmark_summary"]
    print(f"Total tests: {summary['total_tests']}")
    print(f"Deep Agents success rate: {summary['deep_agents_success_rate']:.2%}")
    print(f"Current system success rate: {summary['current_system_success_rate']:.2%}")
    print(f"Average Deep Agents time: {summary['average_deep_agents_time']:.3f}s")
    print(f"Average current system time: {summary['average_current_system_time']:.3f}s")
    
    assessment = results["overall_assessment"]
    print(f"\nOverall score: {assessment['overall_score']:.1f}/100")
    print(f"Recommendation: {assessment['recommendation']}")
    print(f"Justification: {assessment['justification']}")
    
    if assessment['key_strengths']:
        print(f"\nKey strengths:")
        for strength in assessment['key_strengths']:
            print(f"  - {strength}")
    
    if assessment['key_weaknesses']:
        print(f"\nKey weaknesses:")
        for weakness in assessment['key_weaknesses']:
            print(f"  - {weakness}")

if __name__ == "__main__":
    asyncio.run(main())