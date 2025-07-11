"""Tests for state manager functionality."""

import pytest
import asyncio
import tempfile
import os
import json
import pickle
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock

from orchestrator.state.state_manager import StateManager, StateManagerError
from orchestrator.state.checkpoint_strategy import AdaptiveCheckpointStrategy, CheckpointStrategy
from orchestrator.state.persistence_backends import (
    PostgresBackend, 
    FileBackend, 
    RedisBackend,
    InMemoryBackend
)


class TestStateManager:
    """Test cases for StateManager class."""
    
    def test_state_manager_creation(self):
        """Test basic state manager creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(backend_type="file", backend_config={"path": temp_dir})
            
            assert manager.backend is not None
            assert manager.checkpoint_strategy is not None
            assert manager.compression_enabled is False
            assert manager.retention_days == 7
    
    def test_state_manager_with_postgres(self):
        """Test state manager with PostgreSQL backend."""
        config = {
            "connection_string": "postgresql://localhost/test",
            "pool_size": 5
        }
        
        manager = StateManager(backend_type="postgres", backend_config=config)
        
        assert isinstance(manager.backend, PostgresBackend)
        assert manager.backend.connection_string == "postgresql://localhost/test"
        assert manager.backend.pool_size == 5
    
    def test_state_manager_with_redis(self):
        """Test state manager with Redis backend."""
        config = {
            "url": "redis://localhost:6379",
            "db": 1
        }
        
        manager = StateManager(backend_type="redis", backend_config=config)
        
        assert isinstance(manager.backend, RedisBackend)
        assert manager.backend.url == "redis://localhost:6379"
        assert manager.backend.db == 1
    
    def test_state_manager_with_custom_strategy(self):
        """Test state manager with custom checkpoint strategy."""
        strategy = AdaptiveCheckpointStrategy(checkpoint_interval=10)
        
        manager = StateManager(
            backend_type="memory",
            checkpoint_strategy=strategy
        )
        
        assert manager.checkpoint_strategy is strategy
        assert manager.checkpoint_strategy.checkpoint_interval == 10
    
    @pytest.mark.asyncio
    async def test_save_checkpoint_basic(self):
        """Test basic checkpoint saving."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        state = {
            "completed_tasks": ["task1", "task2"],
            "current_task": "task3",
            "pipeline_context": {"user": "test_user"}
        }
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0"
        }
        
        checkpoint_id = await manager.save_checkpoint(pipeline_id, state, metadata)
        
        assert checkpoint_id is not None
        assert isinstance(checkpoint_id, str)
        assert len(checkpoint_id) > 0
    
    @pytest.mark.asyncio
    async def test_save_checkpoint_with_compression(self):
        """Test checkpoint saving with compression."""
        manager = StateManager(
            backend_type="memory",
            compression_enabled=True
        )
        
        pipeline_id = "test_pipeline_123"
        # Large state that should trigger compression
        large_state = {
            "data": "x" * 10000,
            "completed_tasks": ["task1", "task2"],
            "large_results": {f"result_{i}": f"data_{i}" * 100 for i in range(100)}
        }
        
        checkpoint_id = await manager.save_checkpoint(pipeline_id, large_state)
        
        assert checkpoint_id is not None
        
        # Verify compression was applied
        restored_state = await manager.restore_checkpoint(pipeline_id, checkpoint_id)
        assert restored_state["state"]["data"] == "x" * 10000
        assert len(restored_state["state"]["large_results"]) == 100
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint_success(self):
        """Test successful checkpoint restoration."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        original_state = {
            "completed_tasks": ["task1", "task2"],
            "current_task": "task3",
            "context": {"user": "test_user"}
        }
        original_metadata = {"version": "1.0.0"}
        
        # Save checkpoint
        checkpoint_id = await manager.save_checkpoint(pipeline_id, original_state, original_metadata)
        
        # Restore checkpoint
        restored = await manager.restore_checkpoint(pipeline_id, checkpoint_id)
        
        assert restored is not None
        assert restored["pipeline_id"] == pipeline_id
        assert restored["state"]["completed_tasks"] == ["task1", "task2"]
        assert restored["state"]["current_task"] == "task3"
        assert restored["state"]["context"]["user"] == "test_user"
        assert restored["metadata"]["version"] == "1.0.0"
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint_not_found(self):
        """Test restoring non-existent checkpoint."""
        manager = StateManager(backend_type="memory")
        
        result = await manager.restore_checkpoint("nonexistent_pipeline", "nonexistent_checkpoint")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint_by_timestamp(self):
        """Test restoring checkpoint by timestamp."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        
        # Save first checkpoint
        state1 = {"step": 1, "data": "first"}
        checkpoint1_id = await manager.save_checkpoint(pipeline_id, state1)
        
        # Small delay to ensure different timestamps
        await asyncio.sleep(0.01)
        
        # Save second checkpoint
        state2 = {"step": 2, "data": "second"}
        checkpoint2_id = await manager.save_checkpoint(pipeline_id, state2)
        
        # Restore latest checkpoint
        latest = await manager.restore_checkpoint(pipeline_id)
        assert latest["state"]["step"] == 2
        assert latest["state"]["data"] == "second"
        
        # Restore specific checkpoint
        specific = await manager.restore_checkpoint(pipeline_id, checkpoint1_id)
        assert specific["state"]["step"] == 1
        assert specific["state"]["data"] == "first"
    
    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        """Test listing checkpoints for a pipeline."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        
        # Save multiple checkpoints
        for i in range(5):
            state = {"step": i, "data": f"checkpoint_{i}"}
            await manager.save_checkpoint(pipeline_id, state)
            await asyncio.sleep(0.01)  # Ensure different timestamps
        
        # List all checkpoints
        checkpoints = await manager.list_checkpoints(pipeline_id)
        
        assert len(checkpoints) == 5
        assert all("checkpoint_id" in cp for cp in checkpoints)
        assert all("timestamp" in cp for cp in checkpoints)
        assert all("metadata" in cp for cp in checkpoints)
        
        # Verify order (should be newest first)
        timestamps = [cp["timestamp"] for cp in checkpoints]
        assert timestamps == sorted(timestamps, reverse=True)
    
    @pytest.mark.asyncio
    async def test_list_checkpoints_with_limit(self):
        """Test listing checkpoints with limit."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        
        # Save multiple checkpoints
        for i in range(10):
            state = {"step": i}
            await manager.save_checkpoint(pipeline_id, state)
            await asyncio.sleep(0.01)
        
        # List with limit
        checkpoints = await manager.list_checkpoints(pipeline_id, limit=3)
        
        assert len(checkpoints) == 3
    
    @pytest.mark.asyncio
    async def test_delete_checkpoint(self):
        """Test deleting a checkpoint."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        state = {"data": "test"}
        
        # Save checkpoint
        checkpoint_id = await manager.save_checkpoint(pipeline_id, state)
        
        # Verify it exists
        restored = await manager.restore_checkpoint(pipeline_id, checkpoint_id)
        assert restored is not None
        
        # Delete checkpoint
        deleted = await manager.delete_checkpoint(pipeline_id, checkpoint_id)
        assert deleted is True
        
        # Verify it's gone
        restored_after_delete = await manager.restore_checkpoint(pipeline_id, checkpoint_id)
        assert restored_after_delete is None
    
    @pytest.mark.asyncio
    async def test_delete_all_checkpoints(self):
        """Test deleting all checkpoints for a pipeline."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        
        # Save multiple checkpoints
        for i in range(3):
            state = {"step": i}
            await manager.save_checkpoint(pipeline_id, state)
        
        # Verify they exist
        checkpoints = await manager.list_checkpoints(pipeline_id)
        assert len(checkpoints) == 3
        
        # Delete all
        deleted_count = await manager.delete_all_checkpoints(pipeline_id)
        assert deleted_count == 3
        
        # Verify they're gone
        checkpoints_after_delete = await manager.list_checkpoints(pipeline_id)
        assert len(checkpoints_after_delete) == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_checkpoints(self):
        """Test cleanup of expired checkpoints."""
        manager = StateManager(backend_type="memory", retention_days=0)  # Immediate expiration
        
        pipeline_id = "test_pipeline_123"
        
        # Save checkpoint
        state = {"data": "test"}
        await manager.save_checkpoint(pipeline_id, state)
        
        # Verify it exists
        checkpoints = await manager.list_checkpoints(pipeline_id)
        assert len(checkpoints) == 1
        
        # Run cleanup
        cleaned_count = await manager.cleanup_expired_checkpoints()
        assert cleaned_count == 1
        
        # Verify it's gone
        checkpoints_after_cleanup = await manager.list_checkpoints(pipeline_id)
        assert len(checkpoints_after_cleanup) == 0
    
    @pytest.mark.asyncio
    async def test_checkpoint_with_context_manager(self):
        """Test checkpoint context manager."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        task_id = "test_task"
        
        # Use context manager
        async with manager.checkpoint_context(pipeline_id, task_id) as checkpoint_id:
            # Do some work
            await asyncio.sleep(0.01)
            
            # Checkpoint should be created
            assert checkpoint_id is not None
        
        # Verify checkpoint was saved
        checkpoints = await manager.list_checkpoints(pipeline_id)
        assert len(checkpoints) == 1
        assert checkpoints[0]["checkpoint_id"] == checkpoint_id
    
    @pytest.mark.asyncio
    async def test_checkpoint_context_manager_with_error(self):
        """Test checkpoint context manager with error."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        task_id = "test_task"
        
        # Use context manager with error
        with pytest.raises(ValueError):
            async with manager.checkpoint_context(pipeline_id, task_id):
                # Simulate error
                raise ValueError("Test error")
        
        # Verify error checkpoint was saved
        checkpoints = await manager.list_checkpoints(pipeline_id)
        assert len(checkpoints) == 1
        
        # Check error information
        restored = await manager.restore_checkpoint(pipeline_id, checkpoints[0]["checkpoint_id"])
        assert "error" in restored["state"]
        assert restored["state"]["error"] == "Test error"
    
    @pytest.mark.asyncio
    async def test_get_pipeline_state(self):
        """Test getting current pipeline state."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        
        # Save some checkpoints
        for i in range(3):
            state = {"step": i, "progress": i / 3}
            await manager.save_checkpoint(pipeline_id, state)
            await asyncio.sleep(0.01)
        
        # Get current state
        current_state = await manager.get_pipeline_state(pipeline_id)
        
        assert current_state is not None
        assert current_state["state"]["step"] == 2  # Latest checkpoint
        assert current_state["state"]["progress"] == 2/3
    
    @pytest.mark.asyncio
    async def test_get_pipeline_history(self):
        """Test getting pipeline execution history."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        
        # Save checkpoints with different states
        states = [
            {"status": "started", "task": "task1"},
            {"status": "running", "task": "task2"},
            {"status": "completed", "task": "task3"}
        ]
        
        for state in states:
            await manager.save_checkpoint(pipeline_id, state)
            await asyncio.sleep(0.01)
        
        # Get history
        history = await manager.get_pipeline_history(pipeline_id)
        
        assert len(history) == 3
        assert history[0]["state"]["status"] == "completed"  # Most recent first
        assert history[1]["state"]["status"] == "running"
        assert history[2]["state"]["status"] == "started"
    
    @pytest.mark.asyncio
    async def test_state_diff(self):
        """Test generating state diff between checkpoints."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        
        # Save first state
        state1 = {
            "completed_tasks": ["task1"],
            "current_task": "task2",
            "data": {"count": 1}
        }
        checkpoint1_id = await manager.save_checkpoint(pipeline_id, state1)
        
        # Save second state
        state2 = {
            "completed_tasks": ["task1", "task2"],
            "current_task": "task3",
            "data": {"count": 2}
        }
        checkpoint2_id = await manager.save_checkpoint(pipeline_id, state2)
        
        # Generate diff
        diff = await manager.get_state_diff(pipeline_id, checkpoint1_id, checkpoint2_id)
        
        assert diff is not None
        assert "added" in diff
        assert "removed" in diff
        assert "modified" in diff
        
        # Check specific changes
        assert "task2" in str(diff["added"])
        assert diff["modified"]["current_task"]["from"] == "task2"
        assert diff["modified"]["current_task"]["to"] == "task3"
    
    @pytest.mark.asyncio
    async def test_rollback_to_checkpoint(self):
        """Test rolling back to a specific checkpoint."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        
        # Save checkpoints
        state1 = {"step": 1, "data": "first"}
        checkpoint1_id = await manager.save_checkpoint(pipeline_id, state1)
        
        state2 = {"step": 2, "data": "second"}
        checkpoint2_id = await manager.save_checkpoint(pipeline_id, state2)
        
        state3 = {"step": 3, "data": "third"}
        checkpoint3_id = await manager.save_checkpoint(pipeline_id, state3)
        
        # Rollback to checkpoint 1
        rollback_result = await manager.rollback_to_checkpoint(pipeline_id, checkpoint1_id)
        
        assert rollback_result is True
        
        # Verify current state is back to checkpoint 1
        current_state = await manager.get_pipeline_state(pipeline_id)
        assert current_state["state"]["step"] == 1
        assert current_state["state"]["data"] == "first"
    
    def test_compression_detection(self):
        """Test compression detection logic."""
        manager = StateManager(backend_type="memory", compression_enabled=True)
        
        # Small state - should not compress
        small_state = {"data": "small"}
        assert manager._should_compress(small_state) is False
        
        # Large state - should compress
        large_state = {"data": "x" * 10000}
        assert manager._should_compress(large_state) is True
        
        # Complex state - should compress if large
        complex_state = {
            "results": {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        }
        assert manager._should_compress(complex_state) is True
    
    def test_state_serialization(self):
        """Test state serialization and deserialization."""
        manager = StateManager(backend_type="memory")
        
        # Test various data types
        complex_state = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "datetime": datetime.now(timezone.utc).isoformat()
        }
        
        # Serialize
        serialized = manager._serialize_state(complex_state)
        assert isinstance(serialized, (str, bytes))
        
        # Deserialize
        deserialized = manager._deserialize_state(serialized)
        assert deserialized == complex_state
    
    @pytest.mark.asyncio
    async def test_concurrent_checkpoint_operations(self):
        """Test concurrent checkpoint operations."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        
        # Concurrent save operations
        async def save_checkpoint_task(task_id):
            state = {"task_id": task_id, "data": f"data_{task_id}"}
            return await manager.save_checkpoint(pipeline_id, state)
        
        # Run multiple save operations concurrently
        tasks = [save_checkpoint_task(i) for i in range(10)]
        checkpoint_ids = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(checkpoint_ids) == 10
        assert all(isinstance(cid, str) for cid in checkpoint_ids)
        
        # Verify all checkpoints exist
        checkpoints = await manager.list_checkpoints(pipeline_id)
        assert len(checkpoints) == 10
    
    @pytest.mark.asyncio
    async def test_checkpoint_metadata(self):
        """Test checkpoint metadata handling."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        state = {"data": "test"}
        
        # Custom metadata
        custom_metadata = {
            "user": "test_user",
            "pipeline_version": "1.2.3",
            "execution_environment": "production",
            "custom_field": "custom_value"
        }
        
        checkpoint_id = await manager.save_checkpoint(pipeline_id, state, custom_metadata)
        
        # Restore and verify metadata
        restored = await manager.restore_checkpoint(pipeline_id, checkpoint_id)
        
        assert restored["metadata"]["user"] == "test_user"
        assert restored["metadata"]["pipeline_version"] == "1.2.3"
        assert restored["metadata"]["execution_environment"] == "production"
        assert restored["metadata"]["custom_field"] == "custom_value"
        
        # Should also have system metadata
        assert "timestamp" in restored["metadata"]
        assert "checkpoint_id" in restored["metadata"]
    
    @pytest.mark.asyncio
    async def test_state_validation(self):
        """Test state validation before saving."""
        manager = StateManager(backend_type="memory")
        
        pipeline_id = "test_pipeline_123"
        
        # Valid state
        valid_state = {
            "completed_tasks": ["task1", "task2"],
            "current_task": "task3",
            "context": {"user": "test"}
        }
        
        checkpoint_id = await manager.save_checkpoint(pipeline_id, valid_state)
        assert checkpoint_id is not None
        
        # Invalid state (non-serializable)
        invalid_state = {
            "data": lambda x: x  # Function is not serializable
        }
        
        with pytest.raises(StateManagerError):
            await manager.save_checkpoint(pipeline_id, invalid_state)
    
    def test_get_statistics(self):
        """Test getting state manager statistics."""
        manager = StateManager(backend_type="memory")
        
        stats = manager.get_statistics()
        
        assert "total_checkpoints" in stats
        assert "total_pipelines" in stats
        assert "storage_backend" in stats
        assert "compression_enabled" in stats
        assert "retention_days" in stats
        assert "checkpoint_strategy" in stats
        
        # Initial values
        assert stats["total_checkpoints"] == 0
        assert stats["total_pipelines"] == 0
        assert stats["storage_backend"] == "memory"


class TestAdaptiveCheckpointStrategy:
    """Test cases for AdaptiveCheckpointStrategy class."""
    
    def test_strategy_creation(self):
        """Test basic strategy creation."""
        strategy = AdaptiveCheckpointStrategy()
        
        assert strategy.task_history == {}
        assert strategy.checkpoint_interval == 5
        assert strategy.critical_task_patterns == []
    
    def test_strategy_with_custom_config(self):
        """Test strategy with custom configuration."""
        strategy = AdaptiveCheckpointStrategy(
            checkpoint_interval=10,
            critical_task_patterns=[".*_critical", "important_.*"]
        )
        
        assert strategy.checkpoint_interval == 10
        assert len(strategy.critical_task_patterns) == 2
        assert ".*_critical" in strategy.critical_task_patterns
    
    def test_should_checkpoint_critical_task(self):
        """Test checkpointing for critical tasks."""
        strategy = AdaptiveCheckpointStrategy(
            critical_task_patterns=[".*_critical", "important_.*"]
        )
        
        # Critical tasks should always trigger checkpoint
        assert strategy.should_checkpoint("pipeline1", "task_critical") is True
        assert strategy.should_checkpoint("pipeline1", "important_task") is True
        
        # Non-critical tasks should not trigger immediately
        assert strategy.should_checkpoint("pipeline1", "normal_task") is False
    
    def test_should_checkpoint_interval_based(self):
        """Test interval-based checkpointing."""
        strategy = AdaptiveCheckpointStrategy(checkpoint_interval=3)
        
        pipeline_id = "test_pipeline"
        
        # First few tasks should not trigger
        assert strategy.should_checkpoint(pipeline_id, "task1") is False
        assert strategy.should_checkpoint(pipeline_id, "task2") is False
        
        # At interval, should trigger
        assert strategy.should_checkpoint(pipeline_id, "task3") is True
        
        # Reset and continue
        assert strategy.should_checkpoint(pipeline_id, "task4") is False
        assert strategy.should_checkpoint(pipeline_id, "task5") is False
        assert strategy.should_checkpoint(pipeline_id, "task6") is True
    
    def test_should_checkpoint_execution_time_based(self):
        """Test execution time-based checkpointing."""
        strategy = AdaptiveCheckpointStrategy(
            time_threshold_minutes=5.0
        )
        
        pipeline_id = "test_pipeline"
        
        # Mock execution time tracking
        strategy.last_checkpoint_time[pipeline_id] = datetime.now(timezone.utc).timestamp() - 360  # 6 minutes ago
        
        # Should trigger due to time threshold
        assert strategy.should_checkpoint(pipeline_id, "any_task") is True
        
        # Reset time
        strategy.last_checkpoint_time[pipeline_id] = datetime.now(timezone.utc).timestamp()
        
        # Should not trigger immediately
        assert strategy.should_checkpoint(pipeline_id, "any_task") is False
    
    def test_should_checkpoint_failure_rate_based(self):
        """Test failure rate-based checkpointing."""
        strategy = AdaptiveCheckpointStrategy(
            failure_rate_threshold=0.3
        )
        
        pipeline_id = "test_pipeline"
        
        # Simulate task history with failures
        strategy.task_history[pipeline_id] = [
            {"task_id": "task1", "success": True},
            {"task_id": "task2", "success": False},
            {"task_id": "task3", "success": False},
            {"task_id": "task4", "success": True},
            {"task_id": "task5", "success": False},
        ]
        
        # High failure rate should trigger checkpoint
        assert strategy.should_checkpoint(pipeline_id, "task6") is True
    
    def test_record_task_execution(self):
        """Test recording task execution."""
        strategy = AdaptiveCheckpointStrategy()
        
        pipeline_id = "test_pipeline"
        
        # Record successful execution
        strategy.record_task_execution(pipeline_id, "task1", success=True, duration=1.5)
        
        # Record failed execution
        strategy.record_task_execution(pipeline_id, "task2", success=False, duration=3.0)
        
        # Verify history
        assert len(strategy.task_history[pipeline_id]) == 2
        assert strategy.task_history[pipeline_id][0]["task_id"] == "task1"
        assert strategy.task_history[pipeline_id][0]["success"] is True
        assert strategy.task_history[pipeline_id][0]["duration"] == 1.5
        assert strategy.task_history[pipeline_id][1]["task_id"] == "task2"
        assert strategy.task_history[pipeline_id][1]["success"] is False
        assert strategy.task_history[pipeline_id][1]["duration"] == 3.0
    
    def test_get_checkpoint_priority(self):
        """Test getting checkpoint priority."""
        strategy = AdaptiveCheckpointStrategy()
        
        # Critical task should have high priority
        priority = strategy.get_checkpoint_priority("pipeline1", "critical_task")
        assert priority >= 0.8
        
        # Normal task should have lower priority
        priority = strategy.get_checkpoint_priority("pipeline1", "normal_task")
        assert priority < 0.8
    
    def test_optimize_checkpoint_frequency(self):
        """Test optimizing checkpoint frequency."""
        strategy = AdaptiveCheckpointStrategy(checkpoint_interval=5)
        
        pipeline_id = "test_pipeline"
        
        # Simulate stable execution
        for i in range(20):
            strategy.record_task_execution(pipeline_id, f"task{i}", success=True, duration=1.0)
        
        # Should increase interval for stable execution
        strategy.optimize_checkpoint_frequency(pipeline_id)
        assert strategy.checkpoint_interval > 5
        
        # Simulate unstable execution
        for i in range(10):
            strategy.record_task_execution(pipeline_id, f"fail_task{i}", success=False, duration=2.0)
        
        # Should decrease interval for unstable execution
        strategy.optimize_checkpoint_frequency(pipeline_id)
        assert strategy.checkpoint_interval < 5
    
    def test_get_strategy_statistics(self):
        """Test getting strategy statistics."""
        strategy = AdaptiveCheckpointStrategy()
        
        pipeline_id = "test_pipeline"
        
        # Record some executions
        strategy.record_task_execution(pipeline_id, "task1", success=True, duration=1.0)
        strategy.record_task_execution(pipeline_id, "task2", success=False, duration=2.0)
        strategy.record_task_execution(pipeline_id, "task3", success=True, duration=1.5)
        
        stats = strategy.get_statistics()
        
        assert "total_pipelines" in stats
        assert "total_executions" in stats
        assert "success_rate" in stats
        assert "average_duration" in stats
        assert "checkpoint_frequency" in stats
        
        assert stats["total_pipelines"] == 1
        assert stats["total_executions"] == 3
        assert stats["success_rate"] == 2/3
        assert stats["average_duration"] == (1.0 + 2.0 + 1.5) / 3


class TestPersistenceBackends:
    """Test cases for different persistence backends."""
    
    def test_memory_backend(self):
        """Test in-memory backend."""
        backend = InMemoryBackend()
        
        assert backend.data == {}
        assert backend.name == "memory"
        assert backend.persistent is False
    
    @pytest.mark.asyncio
    async def test_memory_backend_operations(self):
        """Test memory backend operations."""
        backend = InMemoryBackend()
        
        # Save data
        key = "test_key"
        data = {"test": "data"}
        await backend.save(key, data)
        
        # Load data
        loaded = await backend.load(key)
        assert loaded == data
        
        # Delete data
        deleted = await backend.delete(key)
        assert deleted is True
        
        # Load deleted data
        loaded_after_delete = await backend.load(key)
        assert loaded_after_delete is None
    
    def test_file_backend(self):
        """Test file backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileBackend(path=temp_dir)
            
            assert backend.path == temp_dir
            assert backend.name == "file"
            assert backend.persistent is True
            assert os.path.exists(backend.path)
    
    @pytest.mark.asyncio
    async def test_file_backend_operations(self):
        """Test file backend operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileBackend(path=temp_dir)
            
            # Save data
            key = "test_key"
            data = {"test": "data", "number": 42}
            await backend.save(key, data)
            
            # Verify file exists
            file_path = os.path.join(temp_dir, f"{key}.json")
            assert os.path.exists(file_path)
            
            # Load data
            loaded = await backend.load(key)
            assert loaded == data
            
            # Delete data
            deleted = await backend.delete(key)
            assert deleted is True
            assert not os.path.exists(file_path)
    
    def test_postgres_backend_config(self):
        """Test PostgreSQL backend configuration."""
        config = {
            "connection_string": "postgresql://user:pass@localhost/db",
            "pool_size": 10,
            "table_name": "checkpoints"
        }
        
        backend = PostgresBackend(**config)
        
        assert backend.connection_string == "postgresql://user:pass@localhost/db"
        assert backend.pool_size == 10
        assert backend.table_name == "checkpoints"
        assert backend.name == "postgres"
        assert backend.persistent is True
    
    def test_redis_backend_config(self):
        """Test Redis backend configuration."""
        config = {
            "url": "redis://localhost:6379",
            "db": 2,
            "key_prefix": "checkpoints:"
        }
        
        backend = RedisBackend(**config)
        
        assert backend.url == "redis://localhost:6379"
        assert backend.db == 2
        assert backend.key_prefix == "checkpoints:"
        assert backend.name == "redis"
        assert backend.persistent is True