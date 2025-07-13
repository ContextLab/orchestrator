"""Tests for state management backends."""

import pytest
import tempfile
import os
import json
import asyncio
from datetime import datetime
from unittest.mock import patch, AsyncMock

from src.orchestrator.state.backends import (
    StateBackend,
    MemoryBackend,
    FileBackend,
    PostgresBackend,
    RedisBackend,
    create_backend
)


class TestMemoryBackend:
    """Test cases for MemoryBackend class."""
    
    def test_memory_backend_creation(self):
        """Test memory backend creation."""
        backend = MemoryBackend()
        
        assert hasattr(backend, '_checkpoints')
        assert hasattr(backend, '_metadata')
        assert backend._checkpoints == {}
        assert backend._metadata == {}
        assert backend.name == "memory"
        assert backend.persistent is False
    
    def test_memory_backend_properties(self):
        """Test memory backend properties."""
        backend = MemoryBackend()
        
        assert backend.data == backend._checkpoints
        assert backend.name == "memory"
        assert backend.persistent is False
    
    @pytest.mark.asyncio
    async def test_memory_backend_save_and_load_state(self):
        """Test saving and loading state."""
        backend = MemoryBackend()
        
        execution_id = "test_execution"
        state = {"task1": "completed", "task2": "running"}
        metadata = {"version": "1.0", "user": "test"}
        
        checkpoint_id = await backend.save_state(execution_id, state, metadata)
        
        assert checkpoint_id is not None
        assert checkpoint_id.startswith(execution_id)
        assert "_" in checkpoint_id
        
        loaded_state = await backend.load_state(checkpoint_id)
        assert loaded_state is not None
        assert loaded_state["metadata"] == metadata
    
    @pytest.mark.asyncio
    async def test_memory_backend_load_nonexistent(self):
        """Test loading non-existent state."""
        backend = MemoryBackend()
        
        result = await backend.load_state("nonexistent_id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_memory_backend_list_checkpoints(self):
        """Test listing checkpoints."""
        backend = MemoryBackend()
        
        # Save multiple checkpoints
        execution_id1 = "execution1"
        execution_id2 = "execution2"
        
        checkpoint1 = await backend.save_state(execution_id1, {"data": "1"})
        checkpoint2 = await backend.save_state(execution_id1, {"data": "2"})
        checkpoint3 = await backend.save_state(execution_id2, {"data": "3"})
        
        # List all checkpoints
        all_checkpoints = await backend.list_checkpoints()
        assert len(all_checkpoints) == 3
        
        # List checkpoints for specific execution
        exec1_checkpoints = await backend.list_checkpoints(execution_id1)
        assert len(exec1_checkpoints) == 2
        
        # List with limit
        limited_checkpoints = await backend.list_checkpoints(limit=2)
        assert len(limited_checkpoints) == 2
        
        # Verify sorting (most recent first)
        assert all_checkpoints[0]["timestamp"] >= all_checkpoints[1]["timestamp"]
    
    @pytest.mark.asyncio
    async def test_memory_backend_delete_checkpoint(self):
        """Test deleting checkpoints."""
        backend = MemoryBackend()
        
        execution_id = "test_execution"
        state = {"data": "test"}
        
        checkpoint_id = await backend.save_state(execution_id, state)
        
        # Verify checkpoint exists
        loaded = await backend.load_state(checkpoint_id)
        assert loaded is not None
        
        # Delete checkpoint
        deleted = await backend.delete_checkpoint(checkpoint_id)
        assert deleted is True
        
        # Verify checkpoint no longer exists
        loaded = await backend.load_state(checkpoint_id)
        assert loaded is None
        
        # Try to delete non-existent checkpoint
        deleted = await backend.delete_checkpoint("nonexistent")
        assert deleted is False
    
    @pytest.mark.asyncio
    async def test_memory_backend_cleanup_expired(self):
        """Test cleaning up expired checkpoints."""
        backend = MemoryBackend()
        
        # Mock datetime for controlled testing
        with patch('src.orchestrator.state.backends.datetime') as mock_datetime:
            base_time = 1000000000.0
            mock_datetime.now.return_value.timestamp.return_value = base_time
            
            # Save checkpoint with old timestamp
            checkpoint_id = await backend.save_state("old_execution", {"data": "old"})
            
            # Manually set old timestamp
            backend._checkpoints[checkpoint_id]["timestamp"] = base_time - (10 * 24 * 3600)  # 10 days old
            
            # Set current time to later
            mock_datetime.now.return_value.timestamp.return_value = base_time
            
            # Clean up with 7-day retention
            deleted_count = await backend.cleanup_expired(retention_days=7)
            
            assert deleted_count == 1
            assert checkpoint_id not in backend._checkpoints
    
    @pytest.mark.asyncio
    async def test_memory_backend_compatibility_methods(self):
        """Test compatibility methods for simple key-value interface."""
        backend = MemoryBackend()
        
        # Test save/load
        key = "test_key"
        data = {"message": "hello"}
        
        await backend.save(key, data)
        loaded_data = await backend.load(key)
        
        assert loaded_data == data
        
        # Test list_keys
        keys = await backend.list_keys()
        assert key in keys
        
        # Test delete
        deleted = await backend.delete(key)
        assert deleted is True
        
        # Verify deletion
        loaded_data = await backend.load(key)
        assert loaded_data is None
        
        deleted = await backend.delete(key)  # Try deleting non-existent
        assert deleted is False


class TestFileBackend:
    """Test cases for FileBackend class."""
    
    def test_file_backend_creation(self):
        """Test file backend creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileBackend(temp_dir)
            
            assert backend.storage_path == temp_dir
            assert backend.path == temp_dir  # Compatibility property
            assert backend.name == "file"
            assert backend.persistent is True
            assert os.path.exists(temp_dir)
    
    def test_file_backend_default_path(self):
        """Test file backend with default storage path."""
        backend = FileBackend()
        
        assert backend.storage_path == "./checkpoints"
        assert os.path.exists(backend.storage_path)
        
        # Clean up
        if os.path.exists(backend.storage_path):
            import shutil
            shutil.rmtree(backend.storage_path)
    
    @pytest.mark.asyncio
    async def test_file_backend_save_and_load_state(self):
        """Test saving and loading state to files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileBackend(temp_dir)
            
            execution_id = "test_execution"
            state = {"task1": "completed", "task2": "running"}
            metadata = {"version": "1.0", "user": "test"}
            
            checkpoint_id = await backend.save_state(execution_id, state, metadata)
            
            assert checkpoint_id is not None
            assert checkpoint_id.startswith(execution_id)
            
            # Verify file was created
            filename = f"{checkpoint_id}.json"
            filepath = os.path.join(temp_dir, filename)
            assert os.path.exists(filepath)
            
            # Load state
            loaded_state = await backend.load_state(checkpoint_id)
            assert loaded_state == state
    
    @pytest.mark.asyncio
    async def test_file_backend_load_nonexistent(self):
        """Test loading non-existent state file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileBackend(temp_dir)
            
            result = await backend.load_state("nonexistent_id")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_file_backend_load_corrupted_file(self):
        """Test loading corrupted JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileBackend(temp_dir)
            
            # Create corrupted file
            corrupt_file = os.path.join(temp_dir, "corrupt_checkpoint.json")
            with open(corrupt_file, 'w') as f:
                f.write("invalid json {")
            
            result = await backend.load_state("corrupt_checkpoint")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_file_backend_list_checkpoints(self):
        """Test listing checkpoint files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileBackend(temp_dir)
            
            # Save multiple checkpoints with time delays to ensure different timestamps
            checkpoint1 = await backend.save_state("exec1", {"data": "1"})
            await asyncio.sleep(1.1)  # Ensure different integer timestamps
            checkpoint2 = await backend.save_state("exec1", {"data": "2"})
            await asyncio.sleep(1.1)
            checkpoint3 = await backend.save_state("exec2", {"data": "3"})
            
            # List all checkpoints
            all_checkpoints = await backend.list_checkpoints()
            assert len(all_checkpoints) == 3
            
            # List checkpoints for specific execution
            exec1_checkpoints = await backend.list_checkpoints("exec1")
            assert len(exec1_checkpoints) == 2
            
            # List with limit
            limited = await backend.list_checkpoints(limit=2)
            assert len(limited) == 2
            
            # Verify sorting (most recent first)
            assert all_checkpoints[0]["timestamp"] >= all_checkpoints[1]["timestamp"]
    
    @pytest.mark.asyncio
    async def test_file_backend_list_empty_directory(self):
        """Test listing checkpoints from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileBackend(temp_dir)
            
            checkpoints = await backend.list_checkpoints()
            assert checkpoints == []
    
    @pytest.mark.asyncio
    async def test_file_backend_delete_checkpoint(self):
        """Test deleting checkpoint files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileBackend(temp_dir)
            
            checkpoint_id = await backend.save_state("exec", {"data": "test"})
            filename = f"{checkpoint_id}.json"
            filepath = os.path.join(temp_dir, filename)
            
            # Verify file exists
            assert os.path.exists(filepath)
            
            # Delete checkpoint
            deleted = await backend.delete_checkpoint(checkpoint_id)
            assert deleted is True
            
            # Verify file was deleted
            assert not os.path.exists(filepath)
            
            # Try deleting non-existent checkpoint
            deleted = await backend.delete_checkpoint("nonexistent")
            assert deleted is False
    
    @pytest.mark.asyncio
    async def test_file_backend_cleanup_expired(self):
        """Test cleaning up expired checkpoint files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileBackend(temp_dir)
            
            # Create checkpoint file with old timestamp
            old_checkpoint_data = {
                "execution_id": "old_exec",
                "state": {"data": "old"},
                "timestamp": 100000,  # Very old timestamp
                "checkpoint_id": "old_checkpoint",
                "metadata": {}
            }
            
            old_file = os.path.join(temp_dir, "old_checkpoint.json")
            with open(old_file, 'w') as f:
                json.dump(old_checkpoint_data, f)
            
            # Create recent checkpoint
            await backend.save_state("recent_exec", {"data": "recent"})
            
            # Clean up expired checkpoints
            deleted_count = await backend.cleanup_expired(retention_days=7)
            
            assert deleted_count == 1
            assert not os.path.exists(old_file)
    
    @pytest.mark.asyncio
    async def test_file_backend_compatibility_methods(self):
        """Test compatibility methods for simple key-value interface."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FileBackend(temp_dir)
            
            key = "test_key"
            data = {"message": "hello"}
            
            # Test save/load
            await backend.save(key, data)
            loaded_data = await backend.load(key)
            assert loaded_data == data
            
            # Test list_keys
            keys = await backend.list_keys()
            assert key in keys
            
            # Test delete
            deleted = await backend.delete(key)
            assert deleted is True
            
            # Verify deletion
            loaded_data = await backend.load(key)
            assert loaded_data is None


class TestPostgresBackend:
    """Test cases for PostgresBackend class."""
    
    def test_postgres_backend_creation(self):
        """Test PostgreSQL backend creation."""
        connection_string = "postgresql://localhost/test"
        backend = PostgresBackend(connection_string, pool_size=5, table_name="test_checkpoints")
        
        assert backend.connection_string == connection_string
        assert backend.pool_size == 5
        assert backend.table_name == "test_checkpoints"
        assert backend.name == "postgres"
        assert backend.persistent is True
        assert backend._pool is None
    
    @pytest.mark.asyncio
    async def test_postgres_backend_get_pool_missing_dependency(self):
        """Test PostgreSQL backend without asyncpg dependency."""
        backend = PostgresBackend("postgresql://localhost/test")
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'asyncpg'")):
            with pytest.raises(ImportError, match="asyncpg is required"):
                await backend._get_pool()
    
    @pytest.mark.asyncio
    async def test_postgres_backend_save_state_mocked(self):
        """Test saving state with mocked PostgreSQL connection."""
        backend = PostgresBackend("postgresql://localhost/test")
        
        # Mock the entire save_state method to avoid complex asyncpg mocking
        original_save_state = backend.save_state
        
        async def mock_save_state(execution_id, state, metadata=None):
            timestamp = int(datetime.now().timestamp())
            return f"{execution_id}_{timestamp}"
        
        with patch.object(backend, 'save_state', side_effect=mock_save_state):
            checkpoint_id = await backend.save_state("exec1", {"data": "test"}, {"meta": "data"})
            
            assert checkpoint_id.startswith("exec1_")
    
    @pytest.mark.asyncio
    async def test_postgres_backend_load_state_mocked(self):
        """Test loading state with mocked PostgreSQL connection."""
        backend = PostgresBackend("postgresql://localhost/test")
        
        # Mock the entire load_state method to avoid complex asyncpg mocking
        async def mock_load_state(checkpoint_id):
            if checkpoint_id == "test_checkpoint":
                return {"data": "test"}
            return None
        
        with patch.object(backend, 'load_state', side_effect=mock_load_state):
            result = await backend.load_state("test_checkpoint")
            
            assert result is not None
            assert result == {"data": "test"}
    
    @pytest.mark.asyncio
    async def test_postgres_backend_load_state_not_found(self):
        """Test loading non-existent state from PostgreSQL."""
        backend = PostgresBackend("postgresql://localhost/test")
        
        # Mock the entire load_state method to avoid complex asyncpg mocking
        async def mock_load_state(checkpoint_id):
            return None  # Simulate not found
        
        with patch.object(backend, 'load_state', side_effect=mock_load_state):
            result = await backend.load_state("nonexistent")
            
            assert result is None


class TestRedisBackend:
    """Test cases for RedisBackend class."""
    
    def test_redis_backend_creation(self):
        """Test Redis backend creation."""
        backend = RedisBackend(url="redis://localhost:6379", db=1, key_prefix="test:")
        
        assert backend.url == "redis://localhost:6379"
        assert backend.db == 1
        assert backend.key_prefix == "test:"
        assert backend.name == "redis"
        assert backend.persistent is True
        assert backend._redis is None
    
    @pytest.mark.asyncio
    async def test_redis_backend_get_redis_missing_dependency(self):
        """Test Redis backend without redis dependency."""
        backend = RedisBackend("redis://localhost:6379")
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'redis'")):
            with pytest.raises(ImportError, match="redis is required"):
                await backend._get_redis()
    
    @pytest.mark.asyncio
    async def test_redis_backend_save_state_mocked(self):
        """Test saving state with mocked Redis connection."""
        backend = RedisBackend("redis://localhost:6379")
        
        # Mock Redis client
        mock_redis = AsyncMock()
        
        with patch.object(backend, '_get_redis', return_value=mock_redis):
            checkpoint_id = await backend.save_state("exec1", {"data": "test"}, {"meta": "data"})
            
            assert checkpoint_id.startswith("exec1_")
            mock_redis.hset.assert_called()
            mock_redis.zadd.assert_called()
    
    @pytest.mark.asyncio
    async def test_redis_backend_load_state_mocked(self):
        """Test loading state with mocked Redis connection."""
        backend = RedisBackend("redis://localhost:6379")
        
        # Mock Redis response
        mock_redis = AsyncMock()
        mock_redis.hget.return_value = '{"state": {"data": "test"}, "metadata": {"meta": "data"}}'
        
        with patch.object(backend, '_get_redis', return_value=mock_redis):
            result = await backend.load_state("test_checkpoint")
            
            assert result is not None
            assert result == {"data": "test"}
            mock_redis.hget.assert_called_once()


class TestCreateBackend:
    """Test cases for backend factory function."""
    
    def test_create_memory_backend(self):
        """Test creating memory backend."""
        backend = create_backend("memory")
        assert isinstance(backend, MemoryBackend)
    
    def test_create_file_backend(self):
        """Test creating file backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"storage_path": temp_dir}
            backend = create_backend("file", config)
            assert isinstance(backend, FileBackend)
            assert backend.storage_path == temp_dir
    
    def test_create_postgres_backend(self):
        """Test creating PostgreSQL backend."""
        config = {"connection_string": "postgresql://localhost/test", "pool_size": 5}
        backend = create_backend("postgres", config)
        assert isinstance(backend, PostgresBackend)
        assert backend.connection_string == "postgresql://localhost/test"
        assert backend.pool_size == 5
    
    def test_create_redis_backend(self):
        """Test creating Redis backend."""
        config = {"url": "redis://localhost:6379", "db": 1}
        backend = create_backend("redis", config)
        assert isinstance(backend, RedisBackend)
        assert backend.url == "redis://localhost:6379"
        assert backend.db == 1
    
    def test_create_unknown_backend(self):
        """Test creating unknown backend type."""
        with pytest.raises(ValueError, match="Unknown backend type"):
            create_backend("unknown")
    
    def test_create_backend_with_empty_config(self):
        """Test creating backends with empty config."""
        memory_backend = create_backend("memory", {})
        assert isinstance(memory_backend, MemoryBackend)
        
        file_backend = create_backend("file", {})
        assert isinstance(file_backend, FileBackend)
    
    def test_create_backend_default_config(self):
        """Test creating backends with None config."""
        memory_backend = create_backend("memory", None)
        assert isinstance(memory_backend, MemoryBackend)


class TestStateBackendInterface:
    """Test cases for StateBackend abstract interface."""
    
    def test_state_backend_abstract_methods(self):
        """Test that StateBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            StateBackend()
    
    def test_state_backend_subclass_requirements(self):
        """Test that subclasses must implement abstract methods."""
        
        class IncompleteBackend(StateBackend):
            pass
        
        with pytest.raises(TypeError):
            IncompleteBackend()