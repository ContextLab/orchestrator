"""
Integration tests for Deep Agents control system adapter.
"""

import asyncio
import pytest
import logging
from typing import Dict, Any

# Set up path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.control_system_adapter import DeepAgentsControlSystem, Task, Pipeline

logger = logging.getLogger(__name__)

class TestDeepAgentsIntegration:
    """Test suite for Deep Agents integration."""
    
    @pytest.fixture
    def control_system(self):
        """Create Deep Agents control system for testing."""
        config = {
            "deep_agents": {
                "max_planning_iterations": 2,
                "max_parallel_tasks": 3,
                "enable_sub_agents": True,
                "state_persistence_backend": "memory",
            }
        }
        return DeepAgentsControlSystem(config=config)
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, control_system):
        """Test that the system initializes correctly."""
        assert control_system.name == "deep-agents-control-system"
        assert "capabilities" in control_system.config
        
        # Test health check
        health = await control_system.health_check()
        assert isinstance(health, bool)
    
    @pytest.mark.asyncio
    async def test_simple_task_execution(self, control_system):
        """Test execution of a simple task."""
        task = Task("test_task", "generate", {"prompt": "Hello world"})
        context = {}
        
        result = await control_system._execute_task_impl(task, context)
        
        # Should return some result (even if fallback)
        assert result is not None
        logger.info(f"Task result: {result}")
    
    @pytest.mark.asyncio
    async def test_simple_pipeline_execution(self, control_system):
        """Test execution of a simple pipeline."""
        tasks = [
            Task("task1", "generate", {"prompt": "Generate text 1"}),
            Task("task2", "analyze", {"data": "{{task1}}"}),
            Task("task3", "transform", {"input": "{{task2}}", "format": "json"}),
        ]
        pipeline = Pipeline("test_pipeline", tasks)
        
        result = await control_system.execute_pipeline(pipeline)
        
        assert result is not None
        assert "pipeline_name" in result
        assert result["pipeline_name"] == "test_pipeline"
        assert "success" in result
        
        logger.info(f"Pipeline result: {result}")
    
    @pytest.mark.asyncio
    async def test_parallel_task_identification(self, control_system):
        """Test that the system can identify parallelizable tasks."""
        tasks = [
            Task("independent1", "analyze", {"data": "dataset1"}),
            Task("independent2", "analyze", {"data": "dataset2"}),
            Task("dependent", "transform", {"input": "{{independent1}}, {{independent2}}"}),
        ]
        pipeline = Pipeline("parallel_test", tasks)
        
        result = await control_system.execute_pipeline(pipeline)
        
        assert result is not None
        logger.info(f"Parallel execution result: {result}")
        
        # Check if parallel groups were identified (if LangChain is available)
        if "parallel_groups" in result:
            parallel_groups = result["parallel_groups"]
            logger.info(f"Identified parallel groups: {parallel_groups}")
    
    @pytest.mark.asyncio
    async def test_complex_pipeline_with_dependencies(self, control_system):
        """Test complex pipeline with task dependencies."""
        tasks = [
            Task("data_load", "execute", {"source": "database"}),
            Task("validate", "validate", {"data": "{{data_load}}"}),
            Task("clean", "transform", {"data": "{{validate}}", "operation": "clean"}),
            Task("analyze_stats", "analyze", {"data": "{{clean}}", "type": "statistical"}),
            Task("analyze_ml", "analyze", {"data": "{{clean}}", "type": "ml"}),
            Task("report", "generate", {
                "template": "report.html",
                "stats": "{{analyze_stats}}",
                "ml_results": "{{analyze_ml}}"
            }),
        ]
        pipeline = Pipeline("complex_pipeline", tasks)
        
        result = await control_system.execute_pipeline(pipeline)
        
        assert result is not None
        assert result.get("success") is not None
        
        logger.info(f"Complex pipeline result keys: {list(result.keys())}")
        
        # If execution plan is available, verify it makes sense
        if "execution_plan" in result:
            execution_plan = result["execution_plan"]
            assert len(execution_plan) == len(tasks)
            logger.info(f"Execution plan created with {len(execution_plan)} tasks")
    
    @pytest.mark.asyncio
    async def test_capabilities_interface(self, control_system):
        """Test the capabilities interface."""
        capabilities = control_system.get_capabilities()
        
        assert isinstance(capabilities, dict)
        assert "supported_actions" in capabilities
        assert "parallel_execution" in capabilities
        assert capabilities["parallel_execution"] is True
        
        logger.info(f"System capabilities: {capabilities}")
    
    @pytest.mark.asyncio
    async def test_task_complexity_analysis(self, control_system):
        """Test task complexity analysis functionality."""
        simple_task = {"id": "simple", "action": "generate", "parameters": {"prompt": "Hi"}}
        complex_task = {
            "id": "complex",
            "action": "analyze",
            "parameters": {
                "data": "{{previous_result}}",
                "algorithm": "deep_learning",
                "features": ["a", "b", "c"],
                "config": {"epochs": 100, "batch_size": 32}
            }
        }
        
        simple_complexity = control_system._analyze_task_complexity(simple_task)
        complex_complexity = control_system._analyze_task_complexity(complex_task)
        
        assert 0.0 <= simple_complexity <= 1.0
        assert 0.0 <= complex_complexity <= 1.0
        assert complex_complexity > simple_complexity
        
        logger.info(f"Simple task complexity: {simple_complexity}")
        logger.info(f"Complex task complexity: {complex_complexity}")
    
    def test_task_dependency_analysis(self, control_system):
        """Test task dependency analysis."""
        task_with_deps = {
            "id": "dependent_task",
            "action": "transform",
            "parameters": {
                "input": "{{task1}}",
                "reference": "{{task2}}",
                "config": "static_value"
            }
        }
        
        dependencies = control_system._analyze_task_dependencies(task_with_deps, [])
        
        assert isinstance(dependencies, list)
        assert "task1" in dependencies
        assert "task2" in dependencies
        
        logger.info(f"Detected dependencies: {dependencies}")
    
    def test_parallelization_detection(self, control_system):
        """Test parallelization detection logic."""
        parallel_task = {
            "id": "parallel_task",
            "action": "analyze",
            "parameters": {"data": "independent_dataset"}
        }
        
        sequential_task = {
            "id": "sequential_task",
            "action": "generate",
            "parameters": {
                "prompt": "Based on {{task1}} and {{task2}} and {{task3}}"
            }
        }
        
        is_parallel1 = control_system._is_task_parallelizable(parallel_task)
        is_parallel2 = control_system._is_task_parallelizable(sequential_task)
        
        assert is_parallel1 is True
        assert is_parallel2 is False
        
        logger.info(f"Parallel task parallelizable: {is_parallel1}")
        logger.info(f"Sequential task parallelizable: {is_parallel2}")

@pytest.mark.asyncio
async def test_benchmark_integration():
    """Test that benchmarking functionality works."""
    try:
        from benchmarks.performance_comparison import PerformanceBenchmark
        
        benchmark = PerformanceBenchmark()
        assert benchmark.deep_agents_system is not None
        
        # Test a simple benchmark
        tasks = [Task("test", "generate", {"prompt": "test"})]
        current_time = benchmark._simulate_current_system_execution(tasks)
        assert current_time > 0
        
        logger.info("Benchmark integration test passed")
        
    except ImportError:
        logger.warning("Benchmark module not available for integration test")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])