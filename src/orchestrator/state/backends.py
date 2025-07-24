"""Backend implementations for state management."""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional


class StateBackend(ABC):
    """Abstract base class for state persistence backends."""

    @abstractmethod
    async def save_state(
        self, execution_id: str, state: Dict[str, Any], metadata: Dict[str, Any] = None
    ) -> str:
        """Save state to backend."""
        pass

    @abstractmethod
    async def load_state(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load state from backend."""
        pass

    @abstractmethod
    async def list_checkpoints(
        self, execution_id: str = None, limit: int = None
    ) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        pass

    @abstractmethod
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        pass

    @abstractmethod
    async def cleanup_expired(self, retention_days: int = 7) -> int:
        """Clean up expired checkpoints."""
        pass


class MemoryBackend(StateBackend):
    """In-memory state backend for testing."""

    def __init__(self):
        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    @property
    def data(self) -> Dict[str, Any]:
        """Access to internal data for testing."""
        return self._checkpoints

    @property
    def name(self) -> str:
        """Backend name."""
        return "memory"

    @property
    def persistent(self) -> bool:
        """Whether backend persists data."""
        return False

    # Compatibility methods for test interface
    async def save(self, key: str, data: Dict[str, Any]) -> None:
        """Save data with simple key-value interface."""
        self._checkpoints[key] = {
            "state": data,
            "execution_id": key,
            "timestamp": datetime.now().timestamp(),
        }

    async def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load data with simple key-value interface."""
        if key in self._checkpoints:
            return self._checkpoints[key]["state"]
        return None

    async def delete(self, key: str) -> bool:
        """Delete data with simple key-value interface."""
        if key in self._checkpoints:
            del self._checkpoints[key]
            return True
        return False

    async def list_keys(self) -> List[str]:
        """List all keys."""
        return list(self._checkpoints.keys())

    async def save_state(
        self, execution_id: str, state: Dict[str, Any], metadata: Dict[str, Any] = None
    ) -> str:
        """Save state to memory."""
        timestamp = datetime.now().timestamp()  # Keep full precision
        checkpoint_id = f"{execution_id}_{timestamp:.6f}"  # Include microseconds

        self._checkpoints[checkpoint_id] = {
            "execution_id": execution_id,
            "state": state,
            "timestamp": timestamp,
            "checkpoint_id": checkpoint_id,
        }

        self._metadata[checkpoint_id] = metadata or {}

        return checkpoint_id

    async def load_state(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load state from memory."""
        if checkpoint_id in self._checkpoints:
            checkpoint_data = self._checkpoints[checkpoint_id]["state"].copy()
            # Add metadata to the returned state
            checkpoint_data["metadata"] = self._metadata.get(checkpoint_id, {})
            return checkpoint_data
        return None

    async def list_checkpoints(
        self, execution_id: str = None, limit: int = None
    ) -> List[Dict[str, Any]]:
        """List checkpoints in memory."""
        checkpoints = []

        for checkpoint_id, checkpoint in self._checkpoints.items():
            if execution_id and checkpoint["execution_id"] != execution_id:
                continue

            metadata = self._metadata.get(checkpoint_id, {})

            checkpoints.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "execution_id": checkpoint["execution_id"],
                    "timestamp": checkpoint["timestamp"],
                    "created_at": datetime.fromtimestamp(
                        checkpoint["timestamp"]
                    ).isoformat(),
                    "metadata": metadata,
                }
            )

        # Sort by timestamp descending
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)

        if limit:
            checkpoints = checkpoints[:limit]

        return checkpoints

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from memory."""
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]
            if checkpoint_id in self._metadata:
                del self._metadata[checkpoint_id]
            return True
        return False

    async def cleanup_expired(self, retention_days: int = 7) -> int:
        """Clean up expired checkpoints."""
        current_time = datetime.now().timestamp()
        cutoff_time = current_time - (retention_days * 24 * 3600)

        expired_checkpoints = []
        for checkpoint_id, checkpoint in self._checkpoints.items():
            if checkpoint["timestamp"] < cutoff_time:
                expired_checkpoints.append(checkpoint_id)

        for checkpoint_id in expired_checkpoints:
            await self.delete_checkpoint(checkpoint_id)

        return len(expired_checkpoints)


class FileBackend(StateBackend):
    """File-based state backend."""

    def __init__(self, storage_path: str = "./checkpoints"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    @property
    def path(self) -> str:
        """Storage path for compatibility."""
        return self.storage_path

    @property
    def name(self) -> str:
        """Backend name."""
        return "file"

    @property
    def persistent(self) -> bool:
        """Whether backend persists data."""
        return True

    # Compatibility methods for test interface
    async def save(self, key: str, data: Dict[str, Any]) -> None:
        """Save data with simple key-value interface."""
        filepath = os.path.join(self.storage_path, f"{key}.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

    async def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load data with simple key-value interface."""
        filepath = os.path.join(self.storage_path, f"{key}.json")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        return None

    async def delete(self, key: str) -> bool:
        """Delete data with simple key-value interface."""
        filepath = os.path.join(self.storage_path, f"{key}.json")
        if os.path.exists(filepath):
            os.unlink(filepath)
            return True
        return False

    async def list_keys(self) -> List[str]:
        """List all keys."""
        keys = []
        if os.path.exists(self.storage_path):
            for filename in os.listdir(self.storage_path):
                if filename.endswith(".json"):
                    keys.append(filename[:-5])  # Remove .json extension
        return keys

    async def save_state(
        self, execution_id: str, state: Dict[str, Any], metadata: Dict[str, Any] = None
    ) -> str:
        """Save state to file."""
        timestamp = int(datetime.now().timestamp())
        checkpoint_id = f"{execution_id}_{timestamp}"

        checkpoint_data = {
            "execution_id": execution_id,
            "state": state,
            "timestamp": timestamp,
            "checkpoint_id": checkpoint_id,
            "metadata": metadata or {},
        }

        filename = f"{checkpoint_id}.json"
        filepath = os.path.join(self.storage_path, filename)

        with open(filepath, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        return checkpoint_id

    async def load_state(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load state from file."""
        filename = f"{checkpoint_id}.json"
        filepath = os.path.join(self.storage_path, filename)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "r") as f:
                checkpoint_data = json.load(f)
            return checkpoint_data["state"]
        except (json.JSONDecodeError, KeyError):
            return None

    async def list_checkpoints(
        self, execution_id: str = None, limit: int = None
    ) -> List[Dict[str, Any]]:
        """List checkpoints from files."""
        checkpoints = []

        if not os.path.exists(self.storage_path):
            return checkpoints

        for filename in os.listdir(self.storage_path):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.storage_path, filename)

            try:
                with open(filepath, "r") as f:
                    checkpoint_data = json.load(f)

                if execution_id and checkpoint_data["execution_id"] != execution_id:
                    continue

                checkpoints.append(
                    {
                        "checkpoint_id": checkpoint_data["checkpoint_id"],
                        "execution_id": checkpoint_data["execution_id"],
                        "timestamp": checkpoint_data["timestamp"],
                        "created_at": datetime.fromtimestamp(
                            checkpoint_data["timestamp"]
                        ).isoformat(),
                        "metadata": checkpoint_data.get("metadata", {}),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by timestamp descending
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)

        if limit:
            checkpoints = checkpoints[:limit]

        return checkpoints

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint file."""
        filename = f"{checkpoint_id}.json"
        filepath = os.path.join(self.storage_path, filename)

        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                return True
            except OSError:
                return False
        return False

    async def cleanup_expired(self, retention_days: int = 7) -> int:
        """Clean up expired checkpoint files."""
        current_time = datetime.now().timestamp()
        cutoff_time = current_time - (retention_days * 24 * 3600)

        expired_count = 0
        if not os.path.exists(self.storage_path):
            return expired_count

        for filename in os.listdir(self.storage_path):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.storage_path, filename)

            try:
                with open(filepath, "r") as f:
                    checkpoint_data = json.load(f)

                if checkpoint_data["timestamp"] < cutoff_time:
                    os.remove(filepath)
                    expired_count += 1
            except (json.JSONDecodeError, KeyError, OSError):
                continue

        return expired_count


class PostgresBackend(StateBackend):
    """PostgreSQL state backend."""

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        table_name: str = "checkpoints",
    ):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.table_name = table_name
        self._pool = None

    @property
    def name(self) -> str:
        """Backend name."""
        return "postgres"

    @property
    def persistent(self) -> bool:
        """Whether backend persists data."""
        return True

    async def _get_pool(self):
        """Get or create database connection pool."""
        if self._pool is None:
            try:
                import asyncpg

                self._pool = await asyncpg.create_pool(
                    self.connection_string, max_size=self.pool_size
                )
                await self._create_tables()
            except ImportError:
                raise ImportError("asyncpg is required for PostgreSQL backend")
        return self._pool

    async def _create_tables(self):
        """Create necessary tables."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id VARCHAR(255) PRIMARY KEY,
                    execution_id VARCHAR(255) NOT NULL,
                    state JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    timestamp INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_checkpoints_execution_id
                ON checkpoints(execution_id);

                CREATE INDEX IF NOT EXISTS idx_checkpoints_timestamp
                ON checkpoints(timestamp);
            """
            )

    async def save_state(
        self, execution_id: str, state: Dict[str, Any], metadata: Dict[str, Any] = None
    ) -> str:
        """Save state to PostgreSQL."""
        pool = await self._get_pool()
        timestamp = int(datetime.now().timestamp())
        checkpoint_id = f"{execution_id}_{timestamp}"

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO checkpoints (checkpoint_id, execution_id, state, metadata, timestamp)
                VALUES ($1, $2, $3, $4, $5)
            """,
                checkpoint_id,
                execution_id,
                json.dumps(state),
                json.dumps(metadata or {}),
                timestamp,
            )

        return checkpoint_id

    async def load_state(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load state from PostgreSQL."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT state FROM checkpoints WHERE checkpoint_id = $1
            """,
                checkpoint_id,
            )

            if row:
                return json.loads(row["state"])

        return None

    async def list_checkpoints(
        self, execution_id: str = None, limit: int = None
    ) -> List[Dict[str, Any]]:
        """List checkpoints from PostgreSQL."""
        pool = await self._get_pool()

        query = """
            SELECT checkpoint_id, execution_id, timestamp, created_at, metadata
            FROM checkpoints
        """
        params = []

        if execution_id:
            query += " WHERE execution_id = $1"
            params.append(execution_id)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT ${len(params) + 1}"
            params.append(limit)

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            checkpoints = []
            for row in rows:
                checkpoints.append(
                    {
                        "checkpoint_id": row["checkpoint_id"],
                        "execution_id": row["execution_id"],
                        "timestamp": row["timestamp"],
                        "created_at": row["created_at"].isoformat(),
                        "metadata": (
                            json.loads(row["metadata"]) if row["metadata"] else {}
                        ),
                    }
                )

        return checkpoints

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from PostgreSQL."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM checkpoints WHERE checkpoint_id = $1
            """,
                checkpoint_id,
            )

            return result.split()[1] == "1"  # DELETE 1 means one row was deleted

    async def cleanup_expired(self, retention_days: int = 7) -> int:
        """Clean up expired checkpoints from PostgreSQL."""
        pool = await self._get_pool()
        cutoff_time = int(datetime.now().timestamp()) - (retention_days * 24 * 3600)

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM checkpoints WHERE timestamp < $1
            """,
                cutoff_time,
            )

            return int(result.split()[1])  # Number of deleted rows


class RedisBackend(StateBackend):
    """Redis state backend."""

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        db: int = 0,
        key_prefix: str = "orchestrator:",
    ):
        self.url = url
        self.db = db
        self.key_prefix = key_prefix
        self._redis = None

    @property
    def name(self) -> str:
        """Backend name."""
        return "redis"

    @property
    def persistent(self) -> bool:
        """Whether backend persists data."""
        return True

    async def _get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis

                self._redis = redis.from_url(self.url, db=self.db)
            except ImportError:
                raise ImportError("redis is required for Redis backend")
        return self._redis

    async def save_state(
        self, execution_id: str, state: Dict[str, Any], metadata: Dict[str, Any] = None
    ) -> str:
        """Save state to Redis."""
        redis_client = await self._get_redis()
        timestamp = int(datetime.now().timestamp())
        checkpoint_id = f"{execution_id}_{timestamp}"

        checkpoint_data = {
            "execution_id": execution_id,
            "state": state,
            "timestamp": timestamp,
            "checkpoint_id": checkpoint_id,
            "metadata": metadata or {},
        }

        # Save checkpoint data
        await redis_client.hset(
            f"checkpoint:{checkpoint_id}",
            mapping={
                "data": json.dumps(checkpoint_data),
                "execution_id": execution_id,
                "timestamp": timestamp,
            },
        )

        # Add to execution index
        await redis_client.zadd(
            f"execution:{execution_id}:checkpoints", {checkpoint_id: timestamp}
        )

        return checkpoint_id

    async def load_state(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load state from Redis."""
        redis_client = await self._get_redis()

        data = await redis_client.hget(f"checkpoint:{checkpoint_id}", "data")
        if data:
            checkpoint_data = json.loads(data)
            return checkpoint_data["state"]

        return None

    async def list_checkpoints(
        self, execution_id: str = None, limit: int = None
    ) -> List[Dict[str, Any]]:
        """List checkpoints from Redis."""
        redis_client = await self._get_redis()
        checkpoints = []

        if execution_id:
            # Get checkpoints for specific execution
            checkpoint_ids = await redis_client.zrevrange(
                f"execution:{execution_id}:checkpoints", 0, limit - 1 if limit else -1
            )
        else:
            # Get all checkpoint keys
            keys = await redis_client.keys("checkpoint:*")
            checkpoint_ids = [key.decode("utf-8").split(":", 1)[1] for key in keys]

            # Sort by timestamp (extract from checkpoint_id)
            checkpoint_ids.sort(key=lambda x: int(x.split("_")[-1]), reverse=True)

            if limit:
                checkpoint_ids = checkpoint_ids[:limit]

        for checkpoint_id in checkpoint_ids:
            data = await redis_client.hget(f"checkpoint:{checkpoint_id}", "data")
            if data:
                checkpoint_data = json.loads(data)
                checkpoints.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "execution_id": checkpoint_data["execution_id"],
                        "timestamp": checkpoint_data["timestamp"],
                        "created_at": datetime.fromtimestamp(
                            checkpoint_data["timestamp"]
                        ).isoformat(),
                        "metadata": checkpoint_data.get("metadata", {}),
                    }
                )

        return checkpoints

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from Redis."""
        redis_client = await self._get_redis()

        # Get execution_id first
        execution_id = await redis_client.hget(
            f"checkpoint:{checkpoint_id}", "execution_id"
        )

        # Delete checkpoint data
        result = await redis_client.delete(f"checkpoint:{checkpoint_id}")

        # Remove from execution index
        if execution_id:
            await redis_client.zrem(
                f"execution:{execution_id}:checkpoints", checkpoint_id
            )

        return result > 0

    async def cleanup_expired(self, retention_days: int = 7) -> int:
        """Clean up expired checkpoints from Redis."""
        redis_client = await self._get_redis()
        cutoff_time = int(datetime.now().timestamp()) - (retention_days * 24 * 3600)

        # Get all checkpoint keys
        keys = await redis_client.keys("checkpoint:*")
        expired_count = 0

        for key in keys:
            timestamp = await redis_client.hget(key, "timestamp")
            if timestamp and int(timestamp) < cutoff_time:
                checkpoint_id = key.decode("utf-8").split(":", 1)[1]
                if await self.delete_checkpoint(checkpoint_id):
                    expired_count += 1

        return expired_count


def create_backend(
    backend_type: str, backend_config: Dict[str, Any] = None
) -> StateBackend:
    """Factory function to create state backends."""
    backend_config = backend_config or {}

    if backend_type == "memory":
        return MemoryBackend()
    elif backend_type == "file":
        return FileBackend(backend_config.get("storage_path", "./checkpoints"))
    elif backend_type == "postgres":
        return PostgresBackend(
            connection_string=backend_config.get(
                "connection_string", "postgresql://localhost/test"
            ),
            pool_size=backend_config.get("pool_size", 10),
        )
    elif backend_type == "redis":
        return RedisBackend(
            url=backend_config.get("url", "redis://localhost:6379"),
            db=backend_config.get("db", 0),
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
