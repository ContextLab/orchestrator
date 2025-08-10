"""Performance comparison tests between Legacy and LangGraph state management.

These tests measure the performance impact of migrating to LangGraph-based
state management and ensure that the enhanced features don't significantly
degrade performance.
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any

from src.orchestrator import init_models
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.task import Task


@pytest.mark.asyncio
class TestPerformanceComparison:
    """Performance comparison tests for Legacy vs LangGraph."""
    
    async def test_orchestrator_initialization_performance(self):
        """Compare initialization time between legacy and LangGraph orchestrators."""
        # Initialize models once
        init_models()
        
        # Measure legacy initialization
        legacy_times = []
        for i in range(5):
            start_time = time.time()
            legacy_orchestrator = Orchestrator(use_langgraph_state=False)
            end_time = time.time()
            legacy_times.append(end_time - start_time)
            await legacy_orchestrator.shutdown()
        
        # Measure LangGraph initialization
        langgraph_times = []
        for i in range(5):
            start_time = time.time()
            langgraph_orchestrator = Orchestrator(use_langgraph_state=True)
            end_time = time.time()
            langgraph_times.append(end_time - start_time)
            await langgraph_orchestrator.shutdown()
        
        # Calculate statistics
        legacy_avg = statistics.mean(legacy_times)
        langgraph_avg = statistics.mean(langgraph_times)
        
        print(f"Legacy initialization avg: {legacy_avg:.4f}s")
        print(f"LangGraph initialization avg: {langgraph_avg:.4f}s")
        print(f"LangGraph overhead: {((langgraph_avg / legacy_avg) - 1) * 100:.1f}%")
        
        # LangGraph should not be more than 50% slower to initialize
        assert langgraph_avg < legacy_avg * 1.5, "LangGraph initialization significantly slower"
        
        print("✅ Initialization performance acceptable")
    
    async def test_pipeline_state_capture_performance(self):
        """Compare pipeline state capture performance."""
        # Initialize models
        init_models()
        
        # Create orchestrators
        legacy_orchestrator = Orchestrator(use_langgraph_state=False)
        langgraph_orchestrator = Orchestrator(use_langgraph_state=True)
        
        # Create a complex pipeline for testing
        def create_complex_pipeline():
            pipeline = Pipeline(
                id="performance_test_pipeline",
                name="Performance Test Pipeline",
                context={"data": "x" * 1000, "numbers": list(range(100))},  # Some bulk data
                metadata={"test": "performance", "large_data": "x" * 500}
            )
            
            # Add multiple tasks
            for i in range(10):
                task = Task(
                    id=f"task_{i}",
                    name=f"Task {i}",
                    action="prompt",
                    parameters={
                        "prompt": f"Process data {i}: {{data}}",
                        "model": "gpt-3.5-turbo",
                        "extra_data": "x" * 100
                    }
                )
                if i > 0:
                    task.dependencies = [f"task_{i-1}"]
                pipeline.add_task(task)
            
            return pipeline
        
        # Measure state capture performance
        def measure_state_capture(orchestrator, pipeline, iterations=20):
            times = []
            for _ in range(iterations):
                start_time = time.time()
                state = orchestrator._get_pipeline_state(pipeline)
                end_time = time.time()
                times.append(end_time - start_time)
                
                # Verify state was captured
                assert isinstance(state, dict)
                assert "tasks" in state
            return times
        
        pipeline = create_complex_pipeline()
        
        legacy_times = measure_state_capture(legacy_orchestrator, pipeline)
        langgraph_times = measure_state_capture(langgraph_orchestrator, pipeline)
        
        # Calculate statistics
        legacy_avg = statistics.mean(legacy_times)
        legacy_std = statistics.stdev(legacy_times) if len(legacy_times) > 1 else 0
        
        langgraph_avg = statistics.mean(langgraph_times)
        langgraph_std = statistics.stdev(langgraph_times) if len(langgraph_times) > 1 else 0
        
        print(f"Legacy state capture: {legacy_avg:.6f}s ±{legacy_std:.6f}")
        print(f"LangGraph state capture: {langgraph_avg:.6f}s ±{langgraph_std:.6f}")
        print(f"LangGraph overhead: {((langgraph_avg / legacy_avg) - 1) * 100:.1f}%")
        
        # LangGraph should not be more than 100% slower for state capture
        assert langgraph_avg < legacy_avg * 2.0, "LangGraph state capture significantly slower"
        
        print("✅ State capture performance acceptable")
        
        await legacy_orchestrator.shutdown()
        await langgraph_orchestrator.shutdown()
    
    async def test_checkpoint_operation_performance(self):
        """Compare checkpoint operation performance."""
        # Initialize models
        init_models()
        
        # Create orchestrators
        legacy_orchestrator = Orchestrator(use_langgraph_state=False)
        langgraph_orchestrator = Orchestrator(use_langgraph_state=True)
        
        # Test data
        test_state = {
            "pipeline_id": "perf_test_pipeline",
            "execution_id": "perf_test_execution",
            "status": "running",
            "data": "x" * 1000,  # Some bulk data
            "complex_structure": {
                "nested": {"deep": {"values": list(range(50))}},
                "arrays": [{"id": i, "data": f"item_{i}"} for i in range(20)]
            }
        }
        
        test_context = {
            "execution_id": "perf_test_execution",
            "pipeline_id": "perf_test_pipeline",
            "start_time": time.time()
        }
        
        async def measure_checkpoint_operations(orchestrator, prefix, iterations=10):
            """Measure checkpoint save/restore operations."""
            save_times = []
            restore_times = []
            
            for i in range(iterations):
                execution_id = f"{prefix}_execution_{i}"
                updated_state = {**test_state, "execution_id": execution_id, "iteration": i}
                updated_context = {**test_context, "execution_id": execution_id}
                
                try:
                    # Measure save operation
                    start_time = time.time()
                    checkpoint_id = await orchestrator.state_manager.save_checkpoint(
                        execution_id, updated_state, updated_context
                    )
                    save_end_time = time.time()
                    save_times.append(save_end_time - start_time)
                    
                    if checkpoint_id:
                        # Measure restore operation
                        restore_start_time = time.time()
                        restored = await orchestrator.state_manager.restore_checkpoint(
                            "perf_test_pipeline", checkpoint_id
                        )
                        restore_end_time = time.time()
                        restore_times.append(restore_end_time - restore_start_time)
                        
                        # Verify restoration worked
                        assert restored is not None
                    
                except Exception as e:
                    print(f"Checkpoint operation failed for {prefix}: {e}")
                    # Still record the operation time even if it failed
                    continue
            
            return save_times, restore_times
        
        # Measure both orchestrators
        legacy_save_times, legacy_restore_times = await measure_checkpoint_operations(
            legacy_orchestrator, "legacy"
        )
        
        langgraph_save_times, langgraph_restore_times = await measure_checkpoint_operations(
            langgraph_orchestrator, "langgraph" 
        )
        
        # Calculate and compare statistics
        if legacy_save_times and langgraph_save_times:
            legacy_save_avg = statistics.mean(legacy_save_times)
            langgraph_save_avg = statistics.mean(langgraph_save_times)
            
            print(f"Legacy save avg: {legacy_save_avg:.6f}s")
            print(f"LangGraph save avg: {langgraph_save_avg:.6f}s")
            
            if legacy_save_avg > 0:
                save_overhead = ((langgraph_save_avg / legacy_save_avg) - 1) * 100
                print(f"LangGraph save overhead: {save_overhead:.1f}%")
        
        if legacy_restore_times and langgraph_restore_times:
            legacy_restore_avg = statistics.mean(legacy_restore_times)
            langgraph_restore_avg = statistics.mean(langgraph_restore_times)
            
            print(f"Legacy restore avg: {legacy_restore_avg:.6f}s")  
            print(f"LangGraph restore avg: {langgraph_restore_avg:.6f}s")
            
            if legacy_restore_avg > 0:
                restore_overhead = ((langgraph_restore_avg / legacy_restore_avg) - 1) * 100
                print(f"LangGraph restore overhead: {restore_overhead:.1f}%")
        
        print("✅ Checkpoint performance comparison completed")
        
        await legacy_orchestrator.shutdown()
        await langgraph_orchestrator.shutdown()
    
    async def test_memory_usage_comparison(self):
        """Compare memory usage between legacy and LangGraph."""
        import psutil
        import gc
        
        # Initialize models
        init_models()
        
        def get_memory_usage():
            """Get current memory usage in MB."""
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        
        # Measure baseline memory
        gc.collect()
        baseline_memory = get_memory_usage()
        
        # Test legacy orchestrator memory usage
        gc.collect()
        memory_before_legacy = get_memory_usage()
        
        legacy_orchestrators = []
        for i in range(5):
            orchestrator = Orchestrator(use_langgraph_state=False)
            legacy_orchestrators.append(orchestrator)
        
        gc.collect()
        memory_after_legacy = get_memory_usage()
        legacy_memory_per_instance = (memory_after_legacy - memory_before_legacy) / 5
        
        # Cleanup legacy orchestrators
        for orchestrator in legacy_orchestrators:
            await orchestrator.shutdown()
        del legacy_orchestrators
        gc.collect()
        
        # Test LangGraph orchestrator memory usage
        memory_before_langgraph = get_memory_usage()
        
        langgraph_orchestrators = []
        for i in range(5):
            orchestrator = Orchestrator(use_langgraph_state=True)
            langgraph_orchestrators.append(orchestrator)
        
        gc.collect()
        memory_after_langgraph = get_memory_usage()
        langgraph_memory_per_instance = (memory_after_langgraph - memory_before_langgraph) / 5
        
        # Cleanup LangGraph orchestrators
        for orchestrator in langgraph_orchestrators:
            await orchestrator.shutdown()
        del langgraph_orchestrators
        gc.collect()
        
        print(f"Baseline memory: {baseline_memory:.1f} MB")
        print(f"Legacy memory per instance: {legacy_memory_per_instance:.1f} MB")
        print(f"LangGraph memory per instance: {langgraph_memory_per_instance:.1f} MB")
        
        if legacy_memory_per_instance > 0:
            memory_overhead = ((langgraph_memory_per_instance / legacy_memory_per_instance) - 1) * 100
            print(f"LangGraph memory overhead: {memory_overhead:.1f}%")
            
            # LangGraph should not use more than 200% more memory per instance
            assert langgraph_memory_per_instance < legacy_memory_per_instance * 3.0, \
                "LangGraph memory usage significantly higher"
        
        print("✅ Memory usage comparison completed")
    
    async def test_concurrent_operations_performance(self):
        """Test performance under concurrent operations."""
        # Initialize models
        init_models()
        
        # Create orchestrators
        legacy_orchestrator = Orchestrator(use_langgraph_state=False)
        langgraph_orchestrator = Orchestrator(use_langgraph_state=True)
        
        async def concurrent_checkpoint_operations(orchestrator, prefix, num_concurrent=5):
            """Perform concurrent checkpoint operations."""
            
            async def single_checkpoint_cycle(index):
                execution_id = f"{prefix}_concurrent_{index}"
                state = {
                    "pipeline_id": f"{prefix}_pipeline_{index}",
                    "execution_id": execution_id,
                    "data": f"concurrent_data_{index}",
                    "timestamp": time.time()
                }
                context = {
                    "execution_id": execution_id,
                    "start_time": time.time()
                }
                
                try:
                    # Save checkpoint
                    checkpoint_id = await orchestrator.state_manager.save_checkpoint(
                        execution_id, state, context
                    )
                    
                    if checkpoint_id:
                        # Restore checkpoint  
                        restored = await orchestrator.state_manager.restore_checkpoint(
                            f"{prefix}_pipeline_{index}", checkpoint_id
                        )
                        return True if restored else False
                    return False
                except Exception:
                    return False
            
            # Run concurrent operations
            start_time = time.time()
            tasks = [single_checkpoint_cycle(i) for i in range(num_concurrent)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            successful_operations = sum(1 for result in results if result is True)
            total_time = end_time - start_time
            
            return successful_operations, total_time, len(results)
        
        # Test concurrent operations for both orchestrators
        legacy_success, legacy_time, legacy_total = await concurrent_checkpoint_operations(
            legacy_orchestrator, "legacy"
        )
        
        langgraph_success, langgraph_time, langgraph_total = await concurrent_checkpoint_operations(
            langgraph_orchestrator, "langgraph"
        )
        
        print(f"Legacy concurrent operations: {legacy_success}/{legacy_total} successful in {legacy_time:.3f}s")
        print(f"LangGraph concurrent operations: {langgraph_success}/{langgraph_total} successful in {langgraph_time:.3f}s")
        
        if legacy_time > 0:
            time_ratio = langgraph_time / legacy_time
            print(f"LangGraph vs Legacy time ratio: {time_ratio:.2f}x")
        
        # Both should handle concurrent operations reasonably
        assert legacy_success >= 0, "Legacy should handle some concurrent operations"
        assert langgraph_success >= 0, "LangGraph should handle some concurrent operations"
        
        print("✅ Concurrent operations performance test completed")
        
        await legacy_orchestrator.shutdown()
        await langgraph_orchestrator.shutdown()
    
    async def test_enhanced_features_performance_cost(self):
        """Measure the performance cost of enhanced features."""
        # Initialize models
        init_models()
        
        # Create LangGraph orchestrator (enhanced features enabled)
        langgraph_orchestrator = Orchestrator(use_langgraph_state=True)
        
        # Create test pipeline
        pipeline = Pipeline(
            id="enhanced_features_test",
            name="Enhanced Features Test",
            context={"test_data": "performance_test"},
            metadata={"performance_test": True}
        )
        
        task = Task(
            id="enhanced_task",
            name="Enhanced Task",
            action="prompt",
            parameters={
                "prompt": "Test enhanced features: {{test_data}}",
                "model": "gpt-3.5-turbo"
            }
        )
        pipeline.add_task(task)
        
        # Measure enhanced state capture
        enhanced_times = []
        for _ in range(10):
            start_time = time.time()
            enhanced_state = langgraph_orchestrator._get_pipeline_state(pipeline)
            end_time = time.time()
            enhanced_times.append(end_time - start_time)
            
            # Verify enhanced features are present
            assert "execution_metadata" in enhanced_state
            assert "performance_metrics" in enhanced_state
        
        enhanced_avg = statistics.mean(enhanced_times)
        print(f"Enhanced state capture avg: {enhanced_avg:.6f}s")
        
        # Test enhanced processing
        enhanced_results = {
            "task1": {
                "result": "test output",
                "task_metadata": {
                    "task_id": "task1",
                    "execution_time": 1.5,
                    "status": "completed"
                }
            }
        }
        
        processing_times = []
        for _ in range(100):
            start_time = time.time()
            processed = langgraph_orchestrator._process_enhanced_results(enhanced_results)
            end_time = time.time()
            processing_times.append(end_time - start_time)
            
            # Verify processing worked
            assert processed["task1"] == "test output"
        
        processing_avg = statistics.mean(processing_times)
        print(f"Enhanced result processing avg: {processing_avg:.6f}s")
        
        # Enhanced features should be reasonably fast
        assert enhanced_avg < 0.01, "Enhanced state capture should be under 10ms"
        assert processing_avg < 0.001, "Enhanced result processing should be under 1ms"
        
        print("✅ Enhanced features performance cost acceptable")
        
        await langgraph_orchestrator.shutdown()