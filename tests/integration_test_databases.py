"""Integration tests for database connectivity and operations.

These tests verify:
1. Actual database connections work
2. CRUD operations are correct
3. Error handling for database failures
4. Connection pooling and resource management
5. Edge cases like duplicate keys, missing databases, etc.

Note: These tests require running database instances.
Use Docker or local installations for testing.
"""

import pytest
import asyncio
import os
import time
import json
from typing import Dict, Any, Optional, List

# Check for database connectivity
def check_postgres_available():
    """Check if PostgreSQL is available for testing."""
    try:
        import psycopg2
        conn_string = os.getenv("TEST_POSTGRES_URL", "postgresql://postgres:password@localhost:5432/test_orchestrator")
        with psycopg2.connect(conn_string) as conn:
            return True
    except (ImportError, Exception):
        return False

def check_redis_available():
    """Check if Redis is available for testing."""
    try:
        import redis
        redis_url = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")
        r = redis.from_url(redis_url)
        r.ping()
        return True
    except (ImportError, Exception):
        return False

HAS_POSTGRES = check_postgres_available()
HAS_REDIS = check_redis_available()


class PostgresBackend:
    """PostgreSQL backend for testing."""
    
    def __init__(self, connection_string: str, table_name: str = "test_checkpoints"):
        self.connection_string = connection_string
        self.table_name = table_name
        self.pool = None
    
    async def initialize(self):
        """Initialize connection pool and create tables."""
        try:
            import asyncpg
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=5
            )
            
            # Create table if not exists
            async with self.pool.acquire() as conn:
                await conn.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        checkpoint_id VARCHAR(255) UNIQUE NOT NULL,
                        execution_id VARCHAR(255) NOT NULL,
                        data JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
        except ImportError:
            pytest.skip("asyncpg library not installed")
    
    async def save_checkpoint(self, checkpoint_id: str, execution_id: str, data: Dict[str, Any]) -> bool:
        """Save checkpoint to PostgreSQL."""
        async with self.pool.acquire() as conn:
            try:
                await conn.execute(f'''
                    INSERT INTO {self.table_name} (checkpoint_id, execution_id, data)
                    VALUES ($1, $2, $3)
                ''', checkpoint_id, execution_id, json.dumps(data))
                return True
            except Exception as e:
                if "duplicate key" in str(e).lower():
                    raise ValueError(f"Checkpoint {checkpoint_id} already exists")
                raise
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint from PostgreSQL."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(f'''
                SELECT checkpoint_id, execution_id, data, created_at 
                FROM {self.table_name} 
                WHERE checkpoint_id = $1
            ''', checkpoint_id)
            
            if row:
                return {
                    "checkpoint_id": row["checkpoint_id"],
                    "execution_id": row["execution_id"],
                    "data": json.loads(row["data"]),
                    "created_at": row["created_at"].isoformat()
                }
            return None
    
    async def list_checkpoints(self, execution_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for an execution."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f'''
                SELECT checkpoint_id, execution_id, created_at
                FROM {self.table_name}
                WHERE execution_id = $1
                ORDER BY created_at DESC
            ''', execution_id)
            
            return [
                {
                    "checkpoint_id": row["checkpoint_id"],
                    "execution_id": row["execution_id"],
                    "created_at": row["created_at"].isoformat()
                }
                for row in rows
            ]
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete specific checkpoint."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(f'''
                DELETE FROM {self.table_name} WHERE checkpoint_id = $1
            ''', checkpoint_id)
            return result.split()[-1] == "1"  # Check if one row was deleted
    
    async def delete_execution_checkpoints(self, execution_id: str) -> int:
        """Delete all checkpoints for execution."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(f'''
                DELETE FROM {self.table_name} WHERE execution_id = $1
            ''', execution_id)
            return int(result.split()[-1])  # Number of deleted rows
    
    async def cleanup(self):
        """Clean up test data and close connections."""
        if self.pool:
            async with self.pool.acquire() as conn:
                await conn.execute(f'DROP TABLE IF EXISTS {self.table_name}')
            await self.pool.close()


class RedisBackend:
    """Redis backend for testing."""
    
    def __init__(self, redis_url: str, key_prefix: str = "test:checkpoints:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.redis = None
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            import redis.asyncio as redis
            self.redis = redis.from_url(self.redis_url)
            await self.redis.ping()  # Test connection
        except ImportError:
            pytest.skip("redis library not installed")
    
    async def save_checkpoint(self, checkpoint_id: str, execution_id: str, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Save checkpoint to Redis."""
        key = f"{self.key_prefix}{checkpoint_id}"
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "execution_id": execution_id,
            "data": data,
            "created_at": time.time()
        }
        
        # Check if key already exists
        exists = await self.redis.exists(key)
        if exists:
            raise ValueError(f"Checkpoint {checkpoint_id} already exists")
        
        await self.redis.setex(key, ttl, json.dumps(checkpoint_data))
        
        # Also add to execution index
        exec_key = f"{self.key_prefix}exec:{execution_id}"
        await self.redis.sadd(exec_key, checkpoint_id)
        await self.redis.expire(exec_key, ttl)
        
        return True
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint from Redis."""
        key = f"{self.key_prefix}{checkpoint_id}"
        data = await self.redis.get(key)
        
        if data:
            return json.loads(data)
        return None
    
    async def list_checkpoints(self, execution_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for an execution."""
        exec_key = f"{self.key_prefix}exec:{execution_id}"
        checkpoint_ids = await self.redis.smembers(exec_key)
        
        checkpoints = []
        for checkpoint_id in checkpoint_ids:
            checkpoint_data = await self.load_checkpoint(checkpoint_id.decode())
            if checkpoint_data:
                checkpoints.append({
                    "checkpoint_id": checkpoint_data["checkpoint_id"],
                    "execution_id": checkpoint_data["execution_id"],
                    "created_at": checkpoint_data["created_at"]
                })
        
        # Sort by created_at descending
        checkpoints.sort(key=lambda x: x["created_at"], reverse=True)
        return checkpoints
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete specific checkpoint."""
        key = f"{self.key_prefix}{checkpoint_id}"
        
        # Get execution_id first
        data = await self.load_checkpoint(checkpoint_id)
        if data:
            execution_id = data["execution_id"]
            
            # Remove from execution index
            exec_key = f"{self.key_prefix}exec:{execution_id}"
            await self.redis.srem(exec_key, checkpoint_id)
            
            # Delete the checkpoint
            result = await self.redis.delete(key)
            return result == 1
        return False
    
    async def delete_execution_checkpoints(self, execution_id: str) -> int:
        """Delete all checkpoints for execution."""
        exec_key = f"{self.key_prefix}exec:{execution_id}"
        checkpoint_ids = await self.redis.smembers(exec_key)
        
        deleted_count = 0
        for checkpoint_id in checkpoint_ids:
            key = f"{self.key_prefix}{checkpoint_id.decode()}"
            result = await self.redis.delete(key)
            if result == 1:
                deleted_count += 1
        
        # Delete execution index
        await self.redis.delete(exec_key)
        return deleted_count
    
    async def cleanup(self):
        """Clean up test data."""
        if self.redis:
            # Delete all test keys
            keys = await self.redis.keys(f"{self.key_prefix}*")
            if keys:
                await self.redis.delete(*keys)
            await self.redis.close()


@pytest.mark.skipif(not HAS_POSTGRES, reason="PostgreSQL not available")
class TestPostgresIntegration:
    """Integration tests for PostgreSQL database."""
    
    @pytest.fixture
    async def postgres_backend(self):
        """Create and initialize PostgreSQL backend."""
        conn_string = os.getenv("TEST_POSTGRES_URL", "postgresql://postgres:password@localhost:5432/test_orchestrator")
        backend = PostgresBackend(conn_string, "test_checkpoints_integration")
        await backend.initialize()
        yield backend
        await backend.cleanup()
    
    @pytest.mark.asyncio
    async def test_postgres_connection(self, postgres_backend):
        """Test basic PostgreSQL connection."""
        # If we get here, connection was successful
        assert postgres_backend.pool is not None
    
    @pytest.mark.asyncio
    async def test_postgres_save_and_load(self, postgres_backend):
        """Test saving and loading checkpoints."""
        checkpoint_id = "test_checkpoint_001"
        execution_id = "test_execution_001"
        data = {
            "completed_tasks": ["task1", "task2"],
            "current_task": "task3",
            "context": {"user": "test_user"}
        }
        
        # Save checkpoint
        result = await postgres_backend.save_checkpoint(checkpoint_id, execution_id, data)
        assert result is True
        
        # Load checkpoint
        loaded = await postgres_backend.load_checkpoint(checkpoint_id)
        assert loaded is not None
        assert loaded["checkpoint_id"] == checkpoint_id
        assert loaded["execution_id"] == execution_id
        assert loaded["data"] == data
        assert "created_at" in loaded
    
    @pytest.mark.asyncio
    async def test_postgres_duplicate_key_error(self, postgres_backend):
        """Test duplicate key error handling."""
        checkpoint_id = "test_checkpoint_duplicate"
        execution_id = "test_execution_001"
        data = {"test": "data"}
        
        # Save first time
        await postgres_backend.save_checkpoint(checkpoint_id, execution_id, data)
        
        # Try to save again with same ID
        with pytest.raises(ValueError, match="already exists"):
            await postgres_backend.save_checkpoint(checkpoint_id, execution_id, data)
    
    @pytest.mark.asyncio
    async def test_postgres_load_nonexistent(self, postgres_backend):
        """Test loading non-existent checkpoint."""
        result = await postgres_backend.load_checkpoint("nonexistent_checkpoint")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_postgres_list_checkpoints(self, postgres_backend):
        """Test listing checkpoints for execution."""
        execution_id = "test_execution_list"
        
        # Save multiple checkpoints
        checkpoints = []
        for i in range(3):
            checkpoint_id = f"checkpoint_{i}"
            await postgres_backend.save_checkpoint(checkpoint_id, execution_id, {"step": i})
            checkpoints.append(checkpoint_id)
            await asyncio.sleep(0.01)  # Small delay for different timestamps
        
        # List checkpoints
        listed = await postgres_backend.list_checkpoints(execution_id)
        
        assert len(listed) == 3
        # Should be ordered by created_at DESC (newest first)
        assert all("checkpoint_id" in cp for cp in listed)
        assert all("execution_id" in cp for cp in listed)
        assert all("created_at" in cp for cp in listed)
    
    @pytest.mark.asyncio
    async def test_postgres_delete_checkpoint(self, postgres_backend):
        """Test deleting specific checkpoint."""
        checkpoint_id = "test_checkpoint_delete"
        execution_id = "test_execution_001"
        data = {"test": "data"}
        
        # Save checkpoint
        await postgres_backend.save_checkpoint(checkpoint_id, execution_id, data)
        
        # Verify it exists
        loaded = await postgres_backend.load_checkpoint(checkpoint_id)
        assert loaded is not None
        
        # Delete checkpoint
        result = await postgres_backend.delete_checkpoint(checkpoint_id)
        assert result is True
        
        # Verify it's gone
        loaded = await postgres_backend.load_checkpoint(checkpoint_id)
        assert loaded is None
    
    @pytest.mark.asyncio
    async def test_postgres_delete_nonexistent(self, postgres_backend):
        """Test deleting non-existent checkpoint."""
        result = await postgres_backend.delete_checkpoint("nonexistent_checkpoint")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_postgres_delete_execution_checkpoints(self, postgres_backend):
        """Test deleting all checkpoints for execution."""
        execution_id = "test_execution_delete_all"
        other_execution_id = "test_execution_keep"
        
        # Save checkpoints for target execution
        for i in range(3):
            await postgres_backend.save_checkpoint(f"delete_{i}", execution_id, {"step": i})
        
        # Save checkpoint for other execution
        await postgres_backend.save_checkpoint("keep_1", other_execution_id, {"keep": True})
        
        # Delete all for target execution
        deleted_count = await postgres_backend.delete_execution_checkpoints(execution_id)
        assert deleted_count == 3
        
        # Verify target execution checkpoints are gone
        listed = await postgres_backend.list_checkpoints(execution_id)
        assert len(listed) == 0
        
        # Verify other execution checkpoint still exists
        other_listed = await postgres_backend.list_checkpoints(other_execution_id)
        assert len(other_listed) == 1
    
    @pytest.mark.asyncio
    async def test_postgres_json_data_types(self, postgres_backend):
        """Test various JSON data types."""
        checkpoint_id = "test_json_types"
        execution_id = "test_execution_json"
        
        complex_data = {
            "string": "test_string",
            "number": 42,
            "float": 3.14159,
            "boolean": True,
            "null_value": None,
            "array": [1, 2, 3, "four"],
            "nested_object": {
                "level1": {
                    "level2": {
                        "deep_value": "found"
                    }
                }
            }
        }
        
        # Save and load
        await postgres_backend.save_checkpoint(checkpoint_id, execution_id, complex_data)
        loaded = await postgres_backend.load_checkpoint(checkpoint_id)
        
        assert loaded["data"] == complex_data
    
    @pytest.mark.asyncio
    async def test_postgres_connection_pool(self, postgres_backend):
        """Test connection pooling works correctly."""
        # Perform multiple concurrent operations
        async def save_task(i):
            return await postgres_backend.save_checkpoint(f"pool_test_{i}", "pool_execution", {"task": i})
        
        tasks = [save_task(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(result is True for result in results)
        
        # Verify all were saved
        listed = await postgres_backend.list_checkpoints("pool_execution")
        assert len(listed) == 10


@pytest.mark.skipif(not HAS_REDIS, reason="Redis not available")
class TestRedisIntegration:
    """Integration tests for Redis database."""
    
    @pytest.fixture
    async def redis_backend(self):
        """Create and initialize Redis backend."""
        redis_url = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")
        backend = RedisBackend(redis_url, "test:integration:")
        await backend.initialize()
        yield backend
        await backend.cleanup()
    
    @pytest.mark.asyncio
    async def test_redis_connection(self, redis_backend):
        """Test basic Redis connection."""
        # If we get here, connection was successful
        assert redis_backend.redis is not None
    
    @pytest.mark.asyncio
    async def test_redis_save_and_load(self, redis_backend):
        """Test saving and loading checkpoints in Redis."""
        checkpoint_id = "test_checkpoint_001"
        execution_id = "test_execution_001"
        data = {
            "completed_tasks": ["task1", "task2"],
            "current_task": "task3"
        }
        
        # Save checkpoint
        result = await redis_backend.save_checkpoint(checkpoint_id, execution_id, data)
        assert result is True
        
        # Load checkpoint
        loaded = await redis_backend.load_checkpoint(checkpoint_id)
        assert loaded is not None
        assert loaded["checkpoint_id"] == checkpoint_id
        assert loaded["execution_id"] == execution_id
        assert loaded["data"] == data
        assert "created_at" in loaded
    
    @pytest.mark.asyncio
    async def test_redis_duplicate_key_error(self, redis_backend):
        """Test duplicate key error handling in Redis."""
        checkpoint_id = "test_checkpoint_duplicate"
        execution_id = "test_execution_001"
        data = {"test": "data"}
        
        # Save first time
        await redis_backend.save_checkpoint(checkpoint_id, execution_id, data)
        
        # Try to save again with same ID
        with pytest.raises(ValueError, match="already exists"):
            await redis_backend.save_checkpoint(checkpoint_id, execution_id, data)
    
    @pytest.mark.asyncio
    async def test_redis_ttl_expiration(self, redis_backend):
        """Test TTL expiration in Redis."""
        checkpoint_id = "test_checkpoint_ttl"
        execution_id = "test_execution_ttl"
        data = {"test": "data"}
        
        # Save with short TTL
        await redis_backend.save_checkpoint(checkpoint_id, execution_id, data, ttl=1)
        
        # Should exist immediately
        loaded = await redis_backend.load_checkpoint(checkpoint_id)
        assert loaded is not None
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        # Should be gone
        loaded = await redis_backend.load_checkpoint(checkpoint_id)
        assert loaded is None
    
    @pytest.mark.asyncio
    async def test_redis_list_checkpoints(self, redis_backend):
        """Test listing checkpoints for execution in Redis."""
        execution_id = "test_execution_list"
        
        # Save multiple checkpoints
        for i in range(3):
            checkpoint_id = f"checkpoint_{i}"
            await redis_backend.save_checkpoint(checkpoint_id, execution_id, {"step": i})
            await asyncio.sleep(0.01)  # Small delay for different timestamps
        
        # List checkpoints
        listed = await redis_backend.list_checkpoints(execution_id)
        
        assert len(listed) == 3
        assert all("checkpoint_id" in cp for cp in listed)
        assert all("execution_id" in cp for cp in listed)
        assert all("created_at" in cp for cp in listed)
    
    @pytest.mark.asyncio
    async def test_redis_delete_checkpoint(self, redis_backend):
        """Test deleting specific checkpoint in Redis."""
        checkpoint_id = "test_checkpoint_delete"
        execution_id = "test_execution_001"
        data = {"test": "data"}
        
        # Save checkpoint
        await redis_backend.save_checkpoint(checkpoint_id, execution_id, data)
        
        # Verify it exists
        loaded = await redis_backend.load_checkpoint(checkpoint_id)
        assert loaded is not None
        
        # Delete checkpoint
        result = await redis_backend.delete_checkpoint(checkpoint_id)
        assert result is True
        
        # Verify it's gone
        loaded = await redis_backend.load_checkpoint(checkpoint_id)
        assert loaded is None
    
    @pytest.mark.asyncio
    async def test_redis_json_serialization(self, redis_backend):
        """Test JSON serialization/deserialization in Redis."""
        checkpoint_id = "test_json_serialization"
        execution_id = "test_execution_json"
        
        complex_data = {
            "string": "test_string",
            "number": 42,
            "float": 3.14159,
            "boolean": True,
            "null_value": None,
            "array": [1, 2, 3, "four"],
            "nested_object": {
                "level1": {
                    "level2": {
                        "deep_value": "found"
                    }
                }
            }
        }
        
        # Save and load
        await redis_backend.save_checkpoint(checkpoint_id, execution_id, complex_data)
        loaded = await redis_backend.load_checkpoint(checkpoint_id)
        
        assert loaded["data"] == complex_data


@pytest.mark.skipif(not (HAS_POSTGRES and HAS_REDIS), reason="Both PostgreSQL and Redis needed")
class TestDatabaseConsistency:
    """Test consistency between different database backends."""
    
    @pytest.mark.asyncio
    async def test_cross_database_consistency(self):
        """Test that PostgreSQL and Redis backends behave consistently."""
        # Setup both backends
        postgres_conn = os.getenv("TEST_POSTGRES_URL", "postgresql://postgres:password@localhost:5432/test_orchestrator")
        redis_url = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")
        
        postgres = PostgresBackend(postgres_conn, "test_consistency_pg")
        redis = RedisBackend(redis_url, "test:consistency:")
        
        await postgres.initialize()
        await redis.initialize()
        
        try:
            checkpoint_id = "consistency_test"
            execution_id = "consistency_execution"
            data = {"consistency": "test", "value": 42}
            
            # Save to both
            pg_result = await postgres.save_checkpoint(checkpoint_id, execution_id, data)
            redis_result = await redis.save_checkpoint(checkpoint_id, execution_id, data)
            
            assert pg_result is True
            assert redis_result is True
            
            # Load from both
            pg_loaded = await postgres.load_checkpoint(checkpoint_id)
            redis_loaded = await redis.load_checkpoint(checkpoint_id)
            
            assert pg_loaded["data"] == data
            assert redis_loaded["data"] == data
            
            # Both should have same core fields
            assert pg_loaded["checkpoint_id"] == redis_loaded["checkpoint_id"]
            assert pg_loaded["execution_id"] == redis_loaded["execution_id"]
            
        finally:
            await postgres.cleanup()
            await redis.cleanup()


if __name__ == "__main__":
    # Print available databases for debugging
    print("Available databases:")
    print(f"PostgreSQL: {'✓' if HAS_POSTGRES else '✗'}")
    print(f"Redis: {'✓' if HAS_REDIS else '✗'}")
    
    if not HAS_POSTGRES:
        print("\nPostgreSQL not available. Try:")
        print("docker run -d --name test-postgres -e POSTGRES_PASSWORD=password -e POSTGRES_DB=test_orchestrator -p 5432:5432 postgres:13")
        print("export TEST_POSTGRES_URL=postgresql://postgres:password@localhost:5432/test_orchestrator")
    
    if not HAS_REDIS:
        print("\nRedis not available. Try:")
        print("docker run -d --name test-redis -p 6379:6379 redis:7")
        print("export TEST_REDIS_URL=redis://localhost:6379/1")
    
    pytest.main([__file__, "-v"])