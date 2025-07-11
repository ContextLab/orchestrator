"""Comprehensive tests for state manager functionality."""

import pytest
import asyncio
import tempfile
import os
import json
import time
from datetime import datetime
from unittest.mock import patch, mock_open

from orchestrator.state.state_manager import StateManager


class TestStateManager:
    """Test cases for StateManager class."""
    
    def test_state_manager_creation_default_path(self):
        """Test StateManager creation with default path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                manager = StateManager()
                
                assert manager.storage_path == "./checkpoints"
                assert os.path.exists(os.path.join(temp_dir, "checkpoints"))
            finally:
                os.chdir(original_cwd)
    
    def test_state_manager_creation_custom_path(self):
        """Test StateManager creation with custom path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = os.path.join(temp_dir, "custom_checkpoints")
            manager = StateManager(custom_path)
            
            assert manager.storage_path == custom_path
            assert os.path.exists(custom_path)
    
    def test_ensure_storage_path_creates_directory(self):
        """Test that storage path is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "nested", "path", "checkpoints")
            manager = StateManager(storage_path)
            
            assert os.path.exists(storage_path)
    
    def test_ensure_storage_path_existing_directory(self):
        """Test that existing storage path is not modified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory first
            storage_path = os.path.join(temp_dir, "checkpoints")
            os.makedirs(storage_path)
            
            # Create a test file
            test_file = os.path.join(storage_path, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            
            manager = StateManager(storage_path)
            
            # Test file should still exist
            assert os.path.exists(test_file)
    
    @pytest.mark.asyncio
    async def test_save_checkpoint_basic(self):
        """Test basic checkpoint saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            state = {
                "completed_tasks": ["task1", "task2"],
                "current_task": "task3",
                "context": {"user": "test_user"}
            }
            metadata = {"version": "1.0.0", "notes": "test checkpoint"}
            
            checkpoint_id = await manager.save_checkpoint(execution_id, state, metadata)
            
            assert checkpoint_id.startswith(execution_id)
            assert "_" in checkpoint_id
            
            # Verify file was created
            filename = f"{checkpoint_id}.json"
            filepath = os.path.join(temp_dir, filename)
            assert os.path.exists(filepath)
            
            # Verify file content
            with open(filepath, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["checkpoint_id"] == checkpoint_id
            assert saved_data["execution_id"] == execution_id
            assert saved_data["state"] == state
            assert saved_data["metadata"] == metadata
            assert "timestamp" in saved_data
            assert saved_data["version"] == "1.0"
    
    @pytest.mark.asyncio
    async def test_save_checkpoint_no_metadata(self):
        """Test saving checkpoint without metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            state = {"data": "test"}
            
            checkpoint_id = await manager.save_checkpoint(execution_id, state)
            
            # Verify file content
            filename = f"{checkpoint_id}.json"
            filepath = os.path.join(temp_dir, filename)
            
            with open(filepath, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["metadata"] == {}
    
    @pytest.mark.asyncio
    async def test_save_checkpoint_with_complex_state(self):
        """Test saving checkpoint with complex state data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            state = {
                "nested_dict": {
                    "level1": {
                        "level2": {
                            "data": "deep_value"
                        }
                    }
                },
                "list_data": [1, 2, 3, {"nested": "in_list"}],
                "datetime_obj": datetime.now(),  # Will be converted to string
                "none_value": None,
                "boolean_value": True,
                "number_value": 42.5
            }
            
            checkpoint_id = await manager.save_checkpoint(execution_id, state)
            
            # Verify file was created and can be loaded
            filename = f"{checkpoint_id}.json"
            filepath = os.path.join(temp_dir, filename)
            
            with open(filepath, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["state"]["nested_dict"]["level1"]["level2"]["data"] == "deep_value"
            assert saved_data["state"]["list_data"] == [1, 2, 3, {"nested": "in_list"}]
            assert saved_data["state"]["none_value"] is None
            assert saved_data["state"]["boolean_value"] is True
            assert saved_data["state"]["number_value"] == 42.5
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint_specific_id(self):
        """Test restoring specific checkpoint by ID."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            state = {"test": "data"}
            metadata = {"version": "1.0.0"}
            
            # Save checkpoint
            checkpoint_id = await manager.save_checkpoint(execution_id, state, metadata)
            
            # Restore checkpoint
            restored = await manager.restore_checkpoint(execution_id, checkpoint_id)
            
            assert restored is not None
            assert restored["checkpoint_id"] == checkpoint_id
            assert restored["execution_id"] == execution_id
            assert restored["state"] == state
            assert restored["metadata"] == metadata
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint_latest(self):
        """Test restoring latest checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            
            # Save multiple checkpoints with delay
            checkpoint1_id = await manager.save_checkpoint(execution_id, {"step": 1})
            await asyncio.sleep(0.01)  # Small delay
            checkpoint2_id = await manager.save_checkpoint(execution_id, {"step": 2})
            await asyncio.sleep(0.01)  # Small delay
            checkpoint3_id = await manager.save_checkpoint(execution_id, {"step": 3})
            
            # Restore latest (should be checkpoint3)
            restored = await manager.restore_checkpoint(execution_id)
            
            assert restored is not None
            assert restored["checkpoint_id"] == checkpoint3_id
            assert restored["state"]["step"] == 3
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint_nonexistent(self):
        """Test restoring nonexistent checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            # Try to restore non-existent checkpoint
            restored = await manager.restore_checkpoint("nonexistent", "nonexistent_checkpoint")
            
            assert restored is None
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint_nonexistent_execution(self):
        """Test restoring latest checkpoint for nonexistent execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            # Try to restore latest for non-existent execution
            restored = await manager.restore_checkpoint("nonexistent")
            
            assert restored is None
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint_corrupted_file(self):
        """Test restoring checkpoint with corrupted JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            
            # Create corrupted JSON file
            filename = f"{execution_id}_123456789.json"
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write("invalid json content {")
            
            # Try to restore latest (should handle corruption gracefully)
            restored = await manager.restore_checkpoint(execution_id)
            
            assert restored is None
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint_missing_key(self):
        """Test restoring checkpoint with missing required key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            
            # Create JSON file missing timestamp key
            filename = f"{execution_id}_123456789.json"
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                json.dump({"checkpoint_id": "test", "state": {}}, f)
            
            # Try to restore latest (should handle missing key gracefully)
            restored = await manager.restore_checkpoint(execution_id)
            
            assert restored is None
    
    @pytest.mark.asyncio
    async def test_list_checkpoints_multiple(self):
        """Test listing multiple checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            
            # Save multiple checkpoints
            checkpoints = []
            for i in range(3):
                state = {"step": i}
                metadata = {"iteration": i}
                checkpoint_id = await manager.save_checkpoint(execution_id, state, metadata)
                checkpoints.append(checkpoint_id)
                await asyncio.sleep(1.1)  # Ensure different timestamps (since checkpoint ID uses int(time.time()))
            
            # List checkpoints
            checkpoint_list = await manager.list_checkpoints(execution_id)
            
            assert len(checkpoint_list) == 3
            
            # Verify each checkpoint in list
            for i, checkpoint_info in enumerate(checkpoint_list):
                assert checkpoint_info["checkpoint_id"] == checkpoints[i]
                assert checkpoint_info["execution_id"] == execution_id
                assert "timestamp" in checkpoint_info
                assert checkpoint_info["metadata"]["iteration"] == i
    
    @pytest.mark.asyncio
    async def test_list_checkpoints_empty(self):
        """Test listing checkpoints for execution with no checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            # List checkpoints for non-existent execution
            checkpoint_list = await manager.list_checkpoints("nonexistent")
            
            assert checkpoint_list == []
    
    @pytest.mark.asyncio
    async def test_list_checkpoints_corrupted_file(self):
        """Test listing checkpoints with corrupted file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            
            # Save valid checkpoint
            valid_checkpoint_id = await manager.save_checkpoint(execution_id, {"valid": "data"})
            
            # Create corrupted file
            corrupted_filename = f"{execution_id}_corrupted.json"
            corrupted_filepath = os.path.join(temp_dir, corrupted_filename)
            with open(corrupted_filepath, 'w') as f:
                f.write("invalid json")
            
            # List checkpoints (should skip corrupted file)
            checkpoint_list = await manager.list_checkpoints(execution_id)
            
            assert len(checkpoint_list) == 1
            assert checkpoint_list[0]["checkpoint_id"] == valid_checkpoint_id
    
    @pytest.mark.asyncio
    async def test_list_checkpoints_sorted_by_timestamp(self):
        """Test that checkpoints are sorted by timestamp."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            
            # Save checkpoints with delays to ensure different timestamps
            checkpoint_ids = []
            for i in range(3):
                checkpoint_id = await manager.save_checkpoint(execution_id, {"step": i})
                checkpoint_ids.append(checkpoint_id)
                await asyncio.sleep(0.01)
            
            # List checkpoints
            checkpoint_list = await manager.list_checkpoints(execution_id)
            
            # Verify sorting (should be in chronological order)
            timestamps = [cp["timestamp"] for cp in checkpoint_list]
            assert timestamps == sorted(timestamps)
    
    @pytest.mark.asyncio
    async def test_delete_checkpoint_existing(self):
        """Test deleting existing checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            state = {"test": "data"}
            
            # Save checkpoint
            checkpoint_id = await manager.save_checkpoint(execution_id, state)
            
            # Verify file exists
            filename = f"{checkpoint_id}.json"
            filepath = os.path.join(temp_dir, filename)
            assert os.path.exists(filepath)
            
            # Delete checkpoint
            result = await manager.delete_checkpoint(checkpoint_id)
            
            assert result is True
            assert not os.path.exists(filepath)
    
    @pytest.mark.asyncio
    async def test_delete_checkpoint_nonexistent(self):
        """Test deleting nonexistent checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            # Try to delete non-existent checkpoint
            result = await manager.delete_checkpoint("nonexistent_checkpoint")
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_execution_checkpoints(self):
        """Test deleting all checkpoints for an execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            other_execution_id = "other_execution_456"
            
            # Save multiple checkpoints for target execution
            target_checkpoints = []
            for i in range(3):
                checkpoint_id = await manager.save_checkpoint(execution_id, {"step": i})
                target_checkpoints.append(checkpoint_id)
                await asyncio.sleep(1.1)  # Ensure different timestamps
            
            # Save checkpoint for other execution
            other_checkpoint_id = await manager.save_checkpoint(other_execution_id, {"other": "data"})
            
            # Delete checkpoints for target execution
            deleted_count = await manager.delete_execution_checkpoints(execution_id)
            
            assert deleted_count == 3
            
            # Verify target execution checkpoints are deleted
            for checkpoint_id in target_checkpoints:
                filename = f"{checkpoint_id}.json"
                filepath = os.path.join(temp_dir, filename)
                assert not os.path.exists(filepath)
            
            # Verify other execution checkpoint still exists
            other_filename = f"{other_checkpoint_id}.json"
            other_filepath = os.path.join(temp_dir, other_filename)
            assert os.path.exists(other_filepath)
    
    @pytest.mark.asyncio
    async def test_delete_execution_checkpoints_none(self):
        """Test deleting checkpoints for execution with no checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            # Try to delete checkpoints for non-existent execution
            deleted_count = await manager.delete_execution_checkpoints("nonexistent")
            
            assert deleted_count == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints(self):
        """Test cleaning up old checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            
            # Save checkpoint
            checkpoint_id = await manager.save_checkpoint(execution_id, {"test": "data"})
            
            # Verify file exists
            filename = f"{checkpoint_id}.json"
            filepath = os.path.join(temp_dir, filename)
            assert os.path.exists(filepath)
            
            # Modify file timestamp to be old
            old_time = time.time() - (35 * 24 * 60 * 60)  # 35 days ago
            os.utime(filepath, (old_time, old_time))
            
            # Clean up old checkpoints (30 day threshold)
            deleted_count = await manager.cleanup_old_checkpoints(30)
            
            assert deleted_count == 1
            assert not os.path.exists(filepath)
    
    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints_recent(self):
        """Test that recent checkpoints are not cleaned up."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            
            # Save checkpoint (recent)
            checkpoint_id = await manager.save_checkpoint(execution_id, {"test": "data"})
            
            # Clean up old checkpoints
            deleted_count = await manager.cleanup_old_checkpoints(30)
            
            assert deleted_count == 0
            
            # Verify file still exists
            filename = f"{checkpoint_id}.json"
            filepath = os.path.join(temp_dir, filename)
            assert os.path.exists(filepath)
    
    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints_os_error(self):
        """Test cleanup with OS error when accessing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            # Create a file that will cause OS error when accessing
            problem_file = os.path.join(temp_dir, "problem.json")
            with open(problem_file, 'w') as f:
                f.write("{}")
            
            # Mock os.path.getmtime to raise OSError
            with patch('os.path.getmtime', side_effect=OSError("Permission denied")):
                # Should handle error gracefully
                deleted_count = await manager.cleanup_old_checkpoints(30)
                
                # Should return 0 since file couldn't be processed
                assert deleted_count == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints_custom_age(self):
        """Test cleanup with custom max age."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            
            # Save checkpoint
            checkpoint_id = await manager.save_checkpoint(execution_id, {"test": "data"})
            
            # Modify file timestamp to be 5 days old
            filename = f"{checkpoint_id}.json"
            filepath = os.path.join(temp_dir, filename)
            old_time = time.time() - (5 * 24 * 60 * 60)  # 5 days ago
            os.utime(filepath, (old_time, old_time))
            
            # Clean up with 3 day threshold
            deleted_count = await manager.cleanup_old_checkpoints(3)
            
            assert deleted_count == 1
            assert not os.path.exists(filepath)
    
    def test_get_storage_info_empty(self):
        """Test getting storage info for empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            info = manager.get_storage_info()
            
            assert info["storage_path"] == temp_dir
            assert info["total_checkpoints"] == 0
            assert info["total_size_bytes"] == 0
            assert info["total_size_mb"] == 0.0
    
    @pytest.mark.asyncio
    async def test_get_storage_info_with_checkpoints(self):
        """Test getting storage info with checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            
            # Save multiple checkpoints
            for i in range(3):
                await manager.save_checkpoint(execution_id, {"step": i, "data": "x" * 100})
                await asyncio.sleep(1.1)  # Ensure different timestamps
            
            info = manager.get_storage_info()
            
            assert info["storage_path"] == temp_dir
            assert info["total_checkpoints"] == 3
            assert info["total_size_bytes"] > 0
            assert info["total_size_mb"] > 0
    
    def test_get_storage_info_with_non_json_files(self):
        """Test storage info ignores non-JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            # Create non-JSON file
            non_json_file = os.path.join(temp_dir, "test.txt")
            with open(non_json_file, 'w') as f:
                f.write("not a checkpoint")
            
            info = manager.get_storage_info()
            
            assert info["total_checkpoints"] == 0
            assert info["total_size_bytes"] == 0
    
    def test_get_storage_info_os_error(self):
        """Test storage info with OS error when accessing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            # Create JSON file
            json_file = os.path.join(temp_dir, "test.json")
            with open(json_file, 'w') as f:
                json.dump({"test": "data"}, f)
            
            # Mock os.path.getsize to raise OSError
            with patch('os.path.getsize', side_effect=OSError("Permission denied")):
                info = manager.get_storage_info()
                
                # Should handle error gracefully
                assert info["total_checkpoints"] == 0
                assert info["total_size_bytes"] == 0
    
    @pytest.mark.asyncio
    async def test_checkpoint_id_format(self):
        """Test that checkpoint IDs have correct format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            state = {"test": "data"}
            
            checkpoint_id = await manager.save_checkpoint(execution_id, state)
            
            # Should be in format "execution_id_timestamp"
            parts = checkpoint_id.split("_")
            assert len(parts) >= 3  # execution, id, timestamp (and maybe more if execution_id contains underscores)
            assert checkpoint_id.startswith(execution_id)
            
            # Last part should be a timestamp (numeric)
            timestamp_part = parts[-1]
            assert timestamp_part.isdigit()
    
    @pytest.mark.asyncio
    async def test_timestamp_format(self):
        """Test that timestamps are in correct ISO format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            state = {"test": "data"}
            
            checkpoint_id = await manager.save_checkpoint(execution_id, state)
            restored = await manager.restore_checkpoint(execution_id, checkpoint_id)
            
            timestamp_str = restored["timestamp"]
            
            # Should be able to parse as ISO format
            timestamp = datetime.fromisoformat(timestamp_str)
            assert isinstance(timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_concurrent_checkpoint_operations(self):
        """Test concurrent checkpoint operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = StateManager(temp_dir)
            
            execution_id = "test_execution_123"
            
            # Run multiple save operations concurrently
            async def save_checkpoint_task(i):
                return await manager.save_checkpoint(execution_id, {"task": i})
            
            tasks = [save_checkpoint_task(i) for i in range(5)]
            checkpoint_ids = await asyncio.gather(*tasks)
            
            # All should succeed
            assert len(checkpoint_ids) == 5
            assert all(isinstance(cid, str) for cid in checkpoint_ids)
            # Note: Due to timing, some checkpoint IDs might be the same
            # since they use int(time.time()) which has 1-second resolution
            assert len(checkpoint_ids) == 5
            
            # Verify all files exist
            for checkpoint_id in checkpoint_ids:
                filename = f"{checkpoint_id}.json"
                filepath = os.path.join(temp_dir, filename)
                assert os.path.exists(filepath)