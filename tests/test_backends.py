"""Tests for state management backends."""

import asyncio
import json
import os
import tempfile
from datetime import datetime

import pytest

from src.orchestrator.state.backends import (
    FileBackend,
    MemoryBackend,
    PostgresBackend,
    RedisBackend,
    StateBackend,
    create_backend,
)
from tests.test_helpers.backend_helpers import (
    TestableDateTime,
    TestableRedisClient,
    TestableAsyncpgPool,
    TestableAsyncpgConnection,
)


class TestMemoryBackend:
    """Test cases for MemoryBackend class."""

    def test_memory_backend_creation(self):
        """Test memory backend creation."""
        backend = MemoryBackend()

        assert hasattr(backend, "_checkpoints")
        assert hasattr(backend, "_metadata")
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

        # Use testable datetime for controlled testing
        import src.orchestrator.state.backends
        original_datetime = src.orchestrator.state.backends.datetime
        test_datetime = TestableDateTime(base_time=1000000000.0)
        
        try:
            # Replace datetime temporarily
            src.orchestrator.state.backends.datetime = test_datetime

            # Save checkpoint with old timestamp
            checkpoint_id = await backend.save_state("old_execution", {"data": "old"})

            # Manually set old timestamp
            backend._checkpoints[checkpoint_id]["timestamp"] = test_datetime._base_time - (
                10 * 24 * 3600
            )  # 10 days old

            # Clean up with 7-day retention
            deleted_count = await backend.cleanup_expired(retention_days=7)

            assert deleted_count == 1
            assert checkpoint_id not in backend._checkpoints
        finally:
            # Restore original datetime
            src.orchestrator.state.backends.datetime = original_datetime

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
            with open(corrupt_file, "w") as f:
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
                "metadata": {},
            }

            old_file = os.path.join(temp_dir, "old_checkpoint.json")
            with open(old_file, "w") as f:
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
        backend = PostgresBackend(
            connection_string, pool_size=5, table_name="test_checkpoints"
        )

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

        # Store original __import__
        original_import = __builtins__['__import__']
        
        def mock_import(name, *args, **kwargs):
            if name == 'asyncpg':
                raise ImportError("No module named 'asyncpg'")
            return original_import(name, *args, **kwargs)
        
        try:
            __builtins__['__import__'] = mock_import
            with pytest.raises(ImportError, match="asyncpg is required"):
                await backend._get_pool()
        finally:
            __builtins__['__import__'] = original_import

    @pytest.mark.asyncio
    async def test_postgres_backend_save_state_mocked(self):
        """Test saving state with testable PostgreSQL connection."""
        backend = PostgresBackend("postgresql://localhost/test")

        # Use testable connection
        test_connection = TestableAsyncpgConnection()
        
        # Replace _get_pool to return our test pool
        async def get_test_pool():
            return test_connection.pool
            
        backend._get_pool = get_test_pool

        checkpoint_id = await backend.save_state(
            "exec1", {"data": "test"}, {"meta": "data"}
        )

        assert checkpoint_id.startswith("exec1_")
        
        # Verify the data was stored
        assert len(test_connection.pool.call_history) > 0
        execute_calls = [c for c in test_connection.pool.call_history if c[0] == 'execute']
        assert len(execute_calls) >= 1  # At least table creation and insert

    @pytest.mark.asyncio
    async def test_postgres_backend_load_state_mocked(self):
        """Test loading state with testable PostgreSQL connection."""
        backend = PostgresBackend("postgresql://localhost/test")

        # Use testable connection with pre-populated data
        test_connection = TestableAsyncpgConnection()
        test_connection.pool._data["checkpoints"] = {
            "test_checkpoint": {
                "checkpoint_id": "test_checkpoint",
                "execution_id": "test",
                "state": json.dumps({"data": "test"}),
                "metadata": json.dumps({}),
                "timestamp": 1234567890
            }
        }
        
        # Replace _get_pool to return our test pool
        async def get_test_pool():
            return test_connection.pool
            
        backend._get_pool = get_test_pool

        result = await backend.load_state("test_checkpoint")

        assert result is not None
        assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_postgres_backend_load_state_not_found(self):
        """Test loading non-existent state from PostgreSQL."""
        backend = PostgresBackend("postgresql://localhost/test")

        # Use testable connection with no data
        test_connection = TestableAsyncpgConnection()
        
        # Replace _get_pool to return our test pool
        async def get_test_pool():
            return test_connection.pool
            
        backend._get_pool = get_test_pool

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

        # Store original __import__
        original_import = __builtins__['__import__']
        
        def mock_import(name, *args, **kwargs):
            if name == 'redis':
                raise ImportError("No module named 'redis'")
            return original_import(name, *args, **kwargs)
        
        try:
            __builtins__['__import__'] = mock_import
            with pytest.raises(ImportError, match="redis is required"):
                await backend._get_redis()
        finally:
            __builtins__['__import__'] = original_import

    @pytest.mark.asyncio
    async def test_redis_backend_save_state_mocked(self):
        """Test saving state with testable Redis connection."""
        backend = RedisBackend("redis://localhost:6379")

        # Use testable Redis client
        test_redis = TestableRedisClient()
        
        # Replace _get_redis to return our test client
        async def get_test_redis():
            return test_redis
            
        backend._get_redis = get_test_redis

        checkpoint_id = await backend.save_state(
            "exec1", {"data": "test"}, {"meta": "data"}
        )

        assert checkpoint_id.startswith("exec1_")
        
        # Verify calls were made
        hset_calls = [c for c in test_redis.call_history if c[0] == 'hset']
        zadd_calls = [c for c in test_redis.call_history if c[0] == 'zadd']
        assert len(hset_calls) > 0
        assert len(zadd_calls) > 0

    @pytest.mark.asyncio
    async def test_redis_backend_load_state_mocked(self):
        """Test loading state with testable Redis connection."""
        backend = RedisBackend("redis://localhost:6379")

        # Use testable Redis client with pre-populated data
        test_redis = TestableRedisClient()
        test_redis._data["checkpoints"] = {
            "test_checkpoint": '{"state": {"data": "test"}, "metadata": {"meta": "data"}}'
        }
        
        # Replace _get_redis to return our test client
        async def get_test_redis():
            return test_redis
            
        backend._get_redis = get_test_redis

        result = await backend.load_state("test_checkpoint")

        assert result is not None
        assert result == {"data": "test"}
        
        # Verify the call was made
        hget_calls = [c for c in test_redis.call_history if c[0] == 'hget']
        assert len(hget_calls) == 1


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
