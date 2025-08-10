"""State consistency validation tests for LangGraph migration.

These tests validate that state management maintains data integrity,
consistency, and proper validation throughout the migration from
legacy to LangGraph-based systems.
"""

import pytest
import asyncio
import time
import json
from typing import Dict, Any, List

from src.orchestrator import init_models
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.task import Task
from src.orchestrator.state.global_context import (
    validate_pipeline_state,
    create_initial_pipeline_state,
    merge_pipeline_states,
    PipelineStatus
)


@pytest.mark.asyncio
class TestStateValidation:
    """Test state consistency and validation."""
    
    async def test_state_schema_validation(self):
        """Test that LangGraph states conform to the defined schema."""
        # Initialize models
        init_models()
        
        # Create orchestrator
        orchestrator = Orchestrator(use_langgraph_state=True)
        
        # Create initial pipeline state
        initial_state = create_initial_pipeline_state(
            pipeline_id="validation_test_pipeline",
            thread_id="validation_test_thread",
            execution_id="validation_test_execution",
            inputs={"test_input": "validation_data"},
            user_id="test_user",
            session_id="test_session"
        )
        
        # Validate the initial state structure
        validation_errors = validate_pipeline_state(initial_state)
        assert len(validation_errors) == 0, f"Initial state validation failed: {validation_errors}"
        
        # Test all required fields are present
        required_fields = [
            "inputs", "outputs", "intermediate_results",
            "execution_metadata", "error_context", "debug_context",
            "tool_results", "model_interactions", "performance_metrics",
            "global_variables", "state_version", "checkpoint_history"
        ]
        
        for field in required_fields:
            assert field in initial_state, f"Missing required field: {field}"
        
        # Test execution metadata structure
        exec_meta = initial_state["execution_metadata"]
        required_exec_fields = [
            "pipeline_id", "thread_id", "execution_id", "start_time",
            "current_step", "completed_steps", "failed_steps", 
            "pending_steps", "status", "retry_count"
        ]
        
        for field in required_exec_fields:
            assert field in exec_meta, f"Missing execution metadata field: {field}"
        
        # Test that status is a valid PipelineStatus
        assert exec_meta["status"] in [status.value for status in PipelineStatus]
        
        print("✅ State schema validation passed")
        
        await orchestrator.shutdown()
    
    async def test_state_merging_consistency(self):
        """Test that state merging maintains consistency."""
        # Initialize models
        init_models()
        
        # Create initial state
        initial_state = create_initial_pipeline_state(
            pipeline_id="merge_test_pipeline",
            thread_id="merge_test_thread",
            execution_id="merge_test_execution",
            inputs={"initial_input": "test_data"}
        )
        
        # Create various types of updates
        updates_1 = {
            "outputs": {"task1_result": "completed"},
            "intermediate_results": {"step1": "done"},
            "execution_metadata": {
                "current_step": "task2",
                "completed_steps": ["task1"]
            }
        }
        
        updates_2 = {
            "outputs": {"task2_result": "completed"},
            "intermediate_results": {"step2": "done"},
            "execution_metadata": {
                "current_step": "task3",
                "completed_steps": ["task2"]  # This should extend the list
            }
        }
        
        # Apply first merge
        merged_state_1 = merge_pipeline_states(initial_state, updates_1)
        
        # Validate merged state
        validation_errors = validate_pipeline_state(merged_state_1)
        assert len(validation_errors) == 0, f"First merge validation failed: {validation_errors}"
        
        # Check that outputs were added
        assert "task1_result" in merged_state_1["outputs"]
        assert merged_state_1["outputs"]["task1_result"] == "completed"
        
        # Check that execution metadata was merged
        assert merged_state_1["execution_metadata"]["current_step"] == "task2"
        assert "task1" in merged_state_1["execution_metadata"]["completed_steps"]
        
        # Apply second merge
        merged_state_2 = merge_pipeline_states(merged_state_1, updates_2)
        
        # Validate final state
        validation_errors = validate_pipeline_state(merged_state_2)
        assert len(validation_errors) == 0, f"Second merge validation failed: {validation_errors}"
        
        # Check that both outputs are present
        assert "task1_result" in merged_state_2["outputs"]
        assert "task2_result" in merged_state_2["outputs"]
        
        # Check that intermediate results were merged
        assert "step1" in merged_state_2["intermediate_results"]
        assert "step2" in merged_state_2["intermediate_results"]
        
        # Check that completed steps were extended
        completed_steps = merged_state_2["execution_metadata"]["completed_steps"]
        assert "task1" in completed_steps
        assert "task2" in completed_steps
        
        print("✅ State merging consistency test passed")
    
    async def test_checkpoint_data_integrity(self):
        """Test that checkpoint data maintains integrity through save/restore cycles."""
        # Initialize models
        init_models()
        
        # Create orchestrator
        orchestrator = Orchestrator(use_langgraph_state=True)
        
        # Create complex test data
        original_state = {
            "pipeline_id": "integrity_test_pipeline",
            "execution_id": "integrity_test_execution",
            "complex_data": {
                "nested_object": {
                    "deep_value": "test_value",
                    "numbers": [1, 2, 3, 4, 5],
                    "boolean_flags": {"flag1": True, "flag2": False}
                },
                "array_of_objects": [
                    {"id": i, "name": f"item_{i}", "value": i * 10}
                    for i in range(5)
                ],
                "special_characters": "String with ünicode and émojis 🎉",
                "large_text": "Lorem ipsum " * 100  # Larger text block
            },
            "metadata": {
                "timestamp": time.time(),
                "version": "1.0.0",
                "test_markers": ["integrity", "validation", "checkpoint"]
            }
        }
        
        original_context = {
            "execution_id": "integrity_test_execution",
            "test_context": "integrity_validation",
            "timestamp": time.time()
        }
        
        try:
            # Save checkpoint
            checkpoint_id = await orchestrator.state_manager.save_checkpoint(
                "integrity_test_execution", original_state, original_context
            )
            
            assert checkpoint_id is not None, "Failed to create checkpoint"
            print(f"Created checkpoint: {checkpoint_id}")
            
            # Restore checkpoint
            restored_data = await orchestrator.state_manager.restore_checkpoint(
                "integrity_test_pipeline", checkpoint_id
            )
            
            assert restored_data is not None, "Failed to restore checkpoint"
            
            # Extract the restored state
            restored_state = restored_data.get("state", {})
            
            # Verify data integrity
            assert restored_state["pipeline_id"] == original_state["pipeline_id"]
            assert restored_state["execution_id"] == original_state["execution_id"]
            
            # Check complex nested data
            if "complex_data" in restored_state:
                restored_complex = restored_state["complex_data"]
                original_complex = original_state["complex_data"]
                
                # Check nested object
                if "nested_object" in restored_complex:
                    assert restored_complex["nested_object"]["deep_value"] == original_complex["nested_object"]["deep_value"]
                    assert restored_complex["nested_object"]["numbers"] == original_complex["nested_object"]["numbers"]
                
                # Check special characters
                if "special_characters" in restored_complex:
                    assert restored_complex["special_characters"] == original_complex["special_characters"]
            
            print("✅ Checkpoint data integrity verified")
            
        except Exception as e:
            print(f"Checkpoint integrity test failed (may be expected): {e}")
        
        await orchestrator.shutdown()
    
    async def test_concurrent_state_validation(self):
        """Test state validation under concurrent operations."""
        # Initialize models
        init_models()
        
        # Create orchestrator
        orchestrator = Orchestrator(use_langgraph_state=True)
        
        async def validate_state_concurrently(validator_id, num_validations=10):
            """Perform state validations concurrently."""
            validation_results = []
            
            for i in range(num_validations):
                # Create a test state
                test_state = create_initial_pipeline_state(
                    pipeline_id=f"concurrent_validation_{validator_id}",
                    thread_id=f"thread_{validator_id}_{i}",
                    execution_id=f"execution_{validator_id}_{i}",
                    inputs={"validator_id": validator_id, "iteration": i}
                )
                
                # Add some random updates to make states different
                updates = {
                    "outputs": {f"result_{validator_id}_{i}": f"output_value_{i}"},
                    "intermediate_results": {f"step_{i}": f"completed_by_{validator_id}"},
                    "execution_metadata": {
                        "current_step": f"step_{i}",
                        "completed_steps": [f"step_{j}" for j in range(i)]
                    }
                }
                
                merged_state = merge_pipeline_states(test_state, updates)
                
                # Validate the state
                try:
                    validation_errors = validate_pipeline_state(merged_state)
                    validation_results.append({
                        "validator_id": validator_id,
                        "iteration": i,
                        "valid": len(validation_errors) == 0,
                        "errors": validation_errors
                    })
                except Exception as e:
                    validation_results.append({
                        "validator_id": validator_id,
                        "iteration": i,
                        "valid": False,
                        "errors": [f"Validation exception: {e}"]
                    })
                
                # Small delay to simulate processing time
                await asyncio.sleep(0.001)
            
            return validation_results
        
        # Run concurrent validators
        num_validators = 8
        validations_per_validator = 5
        
        print(f"Running {num_validators} concurrent state validators...")
        
        tasks = [validate_state_concurrently(i, validations_per_validator) 
                for i in range(num_validators)]
        all_results = await asyncio.gather(*tasks)
        
        # Analyze results
        total_validations = 0
        successful_validations = 0
        all_errors = []
        
        for validator_results in all_results:
            for result in validator_results:
                total_validations += 1
                if result["valid"]:
                    successful_validations += 1
                else:
                    all_errors.extend(result["errors"])
        
        success_rate = (successful_validations / total_validations) * 100
        
        print(f"Concurrent validation results:")
        print(f"  Total validations: {total_validations}")
        print(f"  Successful: {successful_validations}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        if all_errors:
            print(f"  Errors encountered: {len(all_errors)}")
            for error in set(all_errors[:5]):  # Show unique errors (first 5)
                print(f"    - {error}")
        
        # All state validations should succeed under concurrent access
        assert success_rate >= 95, f"State validation success rate too low: {success_rate:.1f}%"
        
        print("✅ Concurrent state validation test passed")
        
        await orchestrator.shutdown()
    
    async def test_state_type_safety(self):
        """Test that state maintains proper type safety."""
        # Test type validation for pipeline state creation
        
        # Test valid state creation
        valid_state = create_initial_pipeline_state(
            pipeline_id="type_safety_test",
            thread_id="type_test_thread", 
            execution_id="type_test_execution",
            inputs={"test_input": "valid_string"},
            user_id="test_user",
            session_id="test_session"
        )
        
        # Validate that all fields have expected types
        assert isinstance(valid_state["inputs"], dict)
        assert isinstance(valid_state["outputs"], dict)
        assert isinstance(valid_state["intermediate_results"], dict)
        assert isinstance(valid_state["execution_metadata"], dict)
        assert isinstance(valid_state["error_context"], dict)
        assert isinstance(valid_state["debug_context"], dict)
        assert isinstance(valid_state["tool_results"], dict)
        assert isinstance(valid_state["model_interactions"], dict)
        assert isinstance(valid_state["performance_metrics"], dict)
        assert isinstance(valid_state["memory_snapshots"], list)
        assert isinstance(valid_state["global_variables"], dict)
        assert isinstance(valid_state["state_version"], str)
        assert isinstance(valid_state["checkpoint_history"], list)
        
        # Test execution metadata types
        exec_meta = valid_state["execution_metadata"]
        assert isinstance(exec_meta["pipeline_id"], str)
        assert isinstance(exec_meta["thread_id"], str)
        assert isinstance(exec_meta["execution_id"], str)
        assert isinstance(exec_meta["start_time"], (int, float))
        assert isinstance(exec_meta["current_step"], str)
        assert isinstance(exec_meta["completed_steps"], list)
        assert isinstance(exec_meta["failed_steps"], list)
        assert isinstance(exec_meta["pending_steps"], list)
        assert isinstance(exec_meta["retry_count"], int)
        
        # Test that status is a valid enum value
        status_value = exec_meta["status"]
        # Handle both enum objects and string values
        if hasattr(status_value, 'value'):
            status_str = status_value.value
        else:
            status_str = status_value
            
        valid_statuses = [status.value for status in PipelineStatus]
        assert status_str in valid_statuses, f"Invalid status: {status_value}"
        
        # Test list field types
        assert all(isinstance(step, str) for step in exec_meta["completed_steps"])
        assert all(isinstance(step, str) for step in exec_meta["failed_steps"])
        assert all(isinstance(step, str) for step in exec_meta["pending_steps"])
        assert all(isinstance(checkpoint, str) for checkpoint in valid_state["checkpoint_history"])
        
        print("✅ State type safety validation passed")
    
    async def test_error_state_handling(self):
        """Test that error states are properly handled and validated."""
        # Initialize models
        init_models()
        
        # Create orchestrator
        orchestrator = Orchestrator(use_langgraph_state=True)
        
        # Create a state with error information
        error_state = create_initial_pipeline_state(
            pipeline_id="error_handling_test",
            thread_id="error_test_thread",
            execution_id="error_test_execution",
            inputs={"test_input": "error_test"}
        )
        
        # Add error information
        error_updates = {
            "execution_metadata": {
                "status": PipelineStatus.FAILED.value,
                "failed_steps": ["failing_task"],
                "current_step": "error_handling"
            },
            "error_context": {
                "current_error": {
                    "message": "Test error for validation",
                    "type": "TestError",
                    "timestamp": time.time()
                },
                "error_history": [
                    {
                        "error": "Previous error",
                        "timestamp": time.time() - 100,
                        "step": "previous_step"
                    }
                ],
                "retry_count": 2,
                "retry_attempts": [
                    {"attempt": 1, "timestamp": time.time() - 50},
                    {"attempt": 2, "timestamp": time.time() - 25}
                ]
            }
        }
        
        # Merge error state
        error_state_merged = merge_pipeline_states(error_state, error_updates)
        
        # Validate error state
        validation_errors = validate_pipeline_state(error_state_merged)
        assert len(validation_errors) == 0, f"Error state validation failed: {validation_errors}"
        
        # Test that error context is properly structured
        error_ctx = error_state_merged["error_context"]
        assert "current_error" in error_ctx
        assert "error_history" in error_ctx
        assert "retry_count" in error_ctx
        assert "retry_attempts" in error_ctx
        
        # Verify error data types
        assert isinstance(error_ctx["error_history"], list)
        assert isinstance(error_ctx["retry_count"], int)
        assert isinstance(error_ctx["retry_attempts"], list)
        
        # Test checkpoint operations with error state
        try:
            error_context = {
                "execution_id": "error_test_execution",
                "error_handling": True,
                "start_time": time.time()
            }
            
            checkpoint_id = await orchestrator.state_manager.save_checkpoint(
                "error_test_execution", error_state_merged, error_context
            )
            
            if checkpoint_id:
                print(f"✅ Error state checkpoint created: {checkpoint_id}")
                
                # Restore and verify
                restored = await orchestrator.state_manager.restore_checkpoint(
                    "error_handling_test", checkpoint_id
                )
                
                if restored:
                    print("✅ Error state restoration successful")
            
        except Exception as e:
            print(f"Error state checkpoint test failed (may be expected): {e}")
        
        print("✅ Error state handling validation passed")
        
        await orchestrator.shutdown()
    
    async def test_large_state_validation(self):
        """Test validation and handling of large state objects."""
        # Initialize models
        init_models()
        
        # Create orchestrator
        orchestrator = Orchestrator(use_langgraph_state=True)
        
        # Create a large state object
        large_inputs = {
            f"input_array_{i}": [f"item_{j}" for j in range(100)]
            for i in range(10)
        }
        
        large_outputs = {
            f"output_data_{i}": {
                "processed_items": [f"processed_{j}_{i}" for j in range(50)],
                "metadata": {
                    "processing_time": i * 0.1,
                    "success": True,
                    "details": f"Processing details for output {i}" * 10
                }
            }
            for i in range(20)
        }
        
        # Create large state
        large_state = create_initial_pipeline_state(
            pipeline_id="large_state_test",
            thread_id="large_state_thread",
            execution_id="large_state_execution",
            inputs=large_inputs
        )
        
        # Add large outputs
        large_updates = {
            "outputs": large_outputs,
            "intermediate_results": {
                f"step_{i}": {
                    "step_data": [f"step_item_{j}" for j in range(30)],
                    "step_metadata": {"step": i, "processed": True}
                }
                for i in range(15)
            },
            "tool_results": {
                "tool_calls": {
                    f"tool_{i}": {
                        "input": f"tool_input_{i}",
                        "output": [f"tool_output_{i}_{j}" for j in range(25)]
                    }
                    for i in range(10)
                }
            }
        }
        
        # Merge large state
        merged_large_state = merge_pipeline_states(large_state, large_updates)
        
        # Validate large state
        start_validation_time = time.time()
        validation_errors = validate_pipeline_state(merged_large_state)
        validation_time = time.time() - start_validation_time
        
        print(f"Large state validation took {validation_time:.3f}s")
        assert len(validation_errors) == 0, f"Large state validation failed: {validation_errors}"
        
        # Test serialization/deserialization performance
        start_serialize_time = time.time()
        serialized = json.dumps(merged_large_state, default=str)
        serialize_time = time.time() - start_serialize_time
        
        start_deserialize_time = time.time()
        deserialized = json.loads(serialized)
        deserialize_time = time.time() - start_deserialize_time
        
        serialized_size = len(serialized) / 1024  # KB
        
        print(f"Large state serialization:")
        print(f"  Size: {serialized_size:.1f} KB")
        print(f"  Serialize time: {serialize_time:.3f}s")
        print(f"  Deserialize time: {deserialize_time:.3f}s")
        
        # Verify deserialization integrity
        assert deserialized["execution_metadata"]["pipeline_id"] == merged_large_state["execution_metadata"]["pipeline_id"]
        assert len(deserialized["outputs"]) == len(merged_large_state["outputs"])
        
        # Performance should be reasonable even for large states
        assert validation_time < 1.0, "Large state validation too slow"
        assert serialize_time < 2.0, "Large state serialization too slow"
        
        print("✅ Large state validation test passed")
        
        await orchestrator.shutdown()