"""State management and checkpointing system.

NOTE: This legacy state manager is maintained for backward compatibility.
For new projects, consider using the LangGraph-based state management system:

- orchestrator.Orchestrator(use_langgraph_state=True) for enhanced features
- orchestrator.state.langgraph_state_manager.LangGraphGlobalContextManager for direct access
- orchestrator.state.legacy_compatibility.LegacyStateManagerAdapter for migration support

The legacy system will continue to be supported but new features will focus on LangGraph integration.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from .adaptive_checkpoint import AdaptiveStrategy
from .backends import create_backend
from ..core.exceptions import StateManagerError


class StateManager:
    """
    Manages pipeline state and checkpointing.

    Provides functionality to save and restore pipeline states
    for recovery and resumption of failed executions.
    """

    def __init__(
        self,
        storage_path: str = "./checkpoints",
        backend_type: str = "file",
        backend_config: Dict[str, Any] = None,
        checkpoint_strategy: Any = None,
        compression_enabled: bool = True,
    ) -> None:
        """
        Initialize state manager.

        Args:
            storage_path: Path to store checkpoint files (for file backend)
            backend_type: Type of backend ('file', 'memory', 'postgres', 'redis')
            backend_config: Backend-specific configuration
            checkpoint_strategy: Checkpoint strategy instance
            compression_enabled: Whether to compress checkpoint data
        """
        self.storage_path = storage_path
        self.compression_enabled = compression_enabled

        # Create backend
        backend_config = backend_config or {}
        if backend_type == "file" and "storage_path" not in backend_config:
            backend_config["storage_path"] = storage_path

        self.backend = create_backend(backend_type, backend_config)

        # Set up checkpoint strategy
        self.checkpoint_strategy = checkpoint_strategy or AdaptiveStrategy()

        # Statistics tracking
        self.stats = {
            "checkpoints_created": 0,
            "checkpoints_restored": 0,
            "checkpoints_deleted": 0,
            "bytes_saved": 0,
            "bytes_compressed": 0,
        }

        # Ensure storage path exists for file backend
        if backend_type == "file":
            self._ensure_storage_path()

    def _ensure_storage_path(self) -> None:
        """Ensure storage path exists."""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        if self.compression_enabled:
            return gzip.compress(data)
        return data

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data using gzip."""
        if self.compression_enabled:
            try:
                return gzip.decompress(data)
            except gzip.BadGzipFile:
                # Data might not be compressed
                return data
        return data

    def _detect_compression(self, data: bytes) -> bool:
        """Detect if data is compressed."""
        return data.startswith(b"\x1f\x8b")

    def _should_compress(self, state: Dict[str, Any]) -> bool:
        """
        Determine if state should be compressed.

        Args:
            state: State to check

        Returns:
            True if state should be compressed
        """
        if not self.compression_enabled:
            return False

        # Compress if state is large (serialize to check size)
        state_bytes = self._serialize_state(state)
        return len(state_bytes) > 1000  # Compress if larger than 1KB

    def _serialize_state(self, state: Dict[str, Any]) -> bytes:
        """Serialize state to bytes."""
        # Use lenient serialization with default=str to handle non-serializable objects
        try:
            json_str = json.dumps(state, default=str)
            return json_str.encode("utf-8")
        except (TypeError, ValueError) as e:
            raise StateManagerError(f"Failed to serialize state: {e}")

    def _deserialize_state(self, data: bytes) -> Dict[str, Any]:
        """Deserialize state from bytes."""
        json_str = data.decode("utf-8")
        return json.loads(json_str)

    async def save_checkpoint(
        self,
        execution_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save pipeline state checkpoint.

        Args:
            execution_id: Execution ID
            state: Pipeline state to save
            metadata: Optional metadata

        Returns:
            Checkpoint ID
        """
        try:
            # Serialize and optionally compress state
            state_bytes = self._serialize_state(state)
            original_size = len(state_bytes)
        except Exception as e:
            raise StateManagerError(f"Failed to serialize state: {e}")

        if self.compression_enabled:
            state_bytes = self._compress_data(state_bytes)

        # Create checkpoint with compressed state
        checkpoint_state = {
            "data": state_bytes.hex(),  # Store as hex string for JSON compatibility
            "compressed": self.compression_enabled,
            "original_size": original_size,
            "compressed_size": len(state_bytes),
        }

        checkpoint_id = await self.backend.save_state(
            execution_id, checkpoint_state, metadata
        )

        # Update statistics
        self.stats["checkpoints_created"] += 1
        self.stats["bytes_saved"] += original_size
        if self.compression_enabled:
            self.stats["bytes_compressed"] += len(state_bytes)

        return checkpoint_id

    async def restore_checkpoint(
        self,
        pipeline_id: str,
        checkpoint_id: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Restore pipeline state from checkpoint.

        Args:
            pipeline_id: Pipeline ID (for compatibility)
            checkpoint_id: Checkpoint ID to restore (if None, use pipeline_id as checkpoint_id)

        Returns:
            Checkpoint state or None if not found
        """
        # Handle both interfaces: restore_checkpoint(checkpoint_id) and restore_checkpoint(pipeline_id, checkpoint_id)
        if checkpoint_id is None:
            # Called as restore_checkpoint(pipeline_id) - find latest checkpoint for this pipeline
            checkpoints = await self.backend.list_checkpoints(pipeline_id)
            if not checkpoints:
                return None
            # Get the most recent checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: x.get("timestamp", 0))
            checkpoint_id = latest_checkpoint.get("checkpoint_id")
            if not checkpoint_id:
                return None

        checkpoint_state = await self.backend.load_state(checkpoint_id)

        if checkpoint_state is None:
            return None

        # Extract state data
        state_hex = checkpoint_state["data"]
        is_compressed = checkpoint_state.get("compressed", False)

        # Convert from hex and decompress if needed
        state_bytes = bytes.fromhex(state_hex)

        if is_compressed:
            state_bytes = self._decompress_data(state_bytes)

        # Deserialize state
        state = self._deserialize_state(state_bytes)

        # Update statistics
        self.stats["checkpoints_restored"] += 1

        # Merge user metadata with system metadata
        user_metadata = checkpoint_state.get("metadata", {})
        system_metadata = {
            "timestamp": checkpoint_state.get("timestamp", time.time()),
            "checkpoint_id": checkpoint_id,
            "compressed": checkpoint_state.get("compressed", False),
        }

        # Return in expected format
        return {
            "pipeline_id": pipeline_id,
            "checkpoint_id": checkpoint_id,
            "state": state,
            "metadata": {**user_metadata, **system_metadata},
            "timestamp": checkpoint_state.get("timestamp", time.time()),
            "compressed": checkpoint_state.get("compressed", False),
        }

    async def restore_checkpoint_by_timestamp(
        self,
        execution_id: str,
        timestamp: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Restore checkpoint closest to given timestamp.

        Args:
            execution_id: Execution ID
            timestamp: Target timestamp

        Returns:
            Checkpoint state or None if not found
        """
        checkpoints = await self.list_checkpoints(execution_id)

        if not checkpoints:
            return None

        # Find checkpoint closest to timestamp
        closest_checkpoint = None
        closest_diff = float("inf")

        for checkpoint in checkpoints:
            diff = abs(checkpoint["timestamp"] - timestamp)
            if diff < closest_diff:
                closest_diff = diff
                closest_checkpoint = checkpoint

        if closest_checkpoint:
            return await self.restore_checkpoint(closest_checkpoint["checkpoint_id"])

        return None

    async def list_checkpoints(
        self,
        execution_id: str = None,
        limit: int = None,
    ) -> List[Dict[str, Any]]:
        """
        List available checkpoints.

        Args:
            execution_id: Filter by execution ID
            limit: Maximum number of checkpoints to return

        Returns:
            List of checkpoint metadata
        """
        return await self.backend.list_checkpoints(execution_id, limit)

    async def delete_checkpoint(
        self, pipeline_id: str, checkpoint_id: str = None
    ) -> bool:
        """
        Delete a checkpoint.

        Args:
            pipeline_id: Pipeline ID (for compatibility)
            checkpoint_id: Checkpoint ID to delete (if None, use pipeline_id as checkpoint_id)

        Returns:
            True if deleted, False otherwise
        """
        # Handle both interfaces: delete_checkpoint(checkpoint_id) and delete_checkpoint(pipeline_id, checkpoint_id)
        if checkpoint_id is None:
            checkpoint_id = pipeline_id
        result = await self.backend.delete_checkpoint(checkpoint_id)

        if result:
            self.stats["checkpoints_deleted"] += 1

        return result

    async def delete_all_checkpoints(self, execution_id: str) -> int:
        """
        Delete all checkpoints for an execution.

        Args:
            execution_id: Execution ID

        Returns:
            Number of deleted checkpoints
        """
        checkpoints = await self.list_checkpoints(execution_id)
        deleted_count = 0

        for checkpoint in checkpoints:
            if await self.delete_checkpoint(checkpoint["checkpoint_id"]):
                deleted_count += 1

        return deleted_count

    async def cleanup_expired_checkpoints(
        self,
        retention_days: int = 7,
    ) -> int:
        """
        Clean up expired checkpoints.

        Args:
            retention_days: Number of days to retain checkpoints

        Returns:
            Number of deleted checkpoints
        """
        deleted_count = await self.backend.cleanup_expired(retention_days)

        # Update statistics
        if deleted_count > 0:
            self.stats["checkpoints_deleted"] += deleted_count

        return deleted_count

    @asynccontextmanager
    async def checkpoint_context(
        self,
        execution_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for checkpoint operations.

        Creates a checkpoint on entry and optionally on exit.
        """
        checkpoint_id = await self.save_checkpoint(execution_id, state, metadata)

        try:
            yield checkpoint_id
        except Exception as e:
            # Save checkpoint on error with updated state that includes error info
            error_state = {**state, "error": str(e), "error_type": type(e).__name__}
            error_metadata = {
                "error": str(e),
                "error_type": type(e).__name__,
                **(metadata or {}),
            }
            await self.save_checkpoint(execution_id, error_state, error_metadata)
            raise

    async def get_pipeline_state(
        self,
        execution_id: str,
        checkpoint_id: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get pipeline state from checkpoint.

        Args:
            execution_id: Execution ID
            checkpoint_id: Specific checkpoint ID (latest if None)

        Returns:
            Pipeline state or None if not found
        """
        if checkpoint_id:
            return await self.restore_checkpoint(checkpoint_id)

        # Get latest checkpoint for execution
        checkpoints = await self.list_checkpoints(execution_id, limit=1)
        if checkpoints:
            return await self.restore_checkpoint(checkpoints[0]["checkpoint_id"])

        return None

    async def get_pipeline_history(
        self,
        execution_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get pipeline execution history.

        Args:
            execution_id: Execution ID
            limit: Maximum number of history entries

        Returns:
            List of checkpoint metadata
        """
        return await self.list_checkpoints(execution_id, limit)

    def calculate_state_diff(
        self,
        state1: Dict[str, Any],
        state2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate difference between two states.

        Args:
            state1: First state
            state2: Second state

        Returns:
            Dictionary containing differences
        """
        diff = {"added": {}, "removed": {}, "modified": {}}

        # Find added and modified keys
        for key, value2 in state2.items():
            if key not in state1:
                diff["added"][key] = value2
            elif state1[key] != value2:
                # For arrays, check if items were added
                if isinstance(state1[key], list) and isinstance(value2, list):
                    old_set = (
                        set(state1[key])
                        if all(isinstance(x, (str, int, float)) for x in state1[key])
                        else state1[key]
                    )
                    new_set = (
                        set(value2)
                        if all(isinstance(x, (str, int, float)) for x in value2)
                        else value2
                    )
                    if isinstance(old_set, set) and isinstance(new_set, set):
                        added_items = new_set - old_set
                        if added_items:
                            diff["added"].update(
                                {f"{key}_item_{item}": item for item in added_items}
                            )

                diff["modified"][key] = {"from": state1[key], "to": value2}

        # Find removed keys
        for key, value1 in state1.items():
            if key not in state2:
                diff["removed"][key] = value1

        return diff

    async def get_state_diff(
        self,
        pipeline_id: str,
        checkpoint_id1: str,
        checkpoint_id2: str,
    ) -> Dict[str, Any]:
        """
        Get difference between two checkpoints.

        Args:
            pipeline_id: Pipeline ID
            checkpoint_id1: First checkpoint ID
            checkpoint_id2: Second checkpoint ID

        Returns:
            Dictionary containing differences
        """
        # Restore both checkpoints
        state1_result = await self.restore_checkpoint(pipeline_id, checkpoint_id1)
        state2_result = await self.restore_checkpoint(pipeline_id, checkpoint_id2)

        if state1_result is None or state2_result is None:
            raise StateManagerError("One or both checkpoints not found")

        state1 = state1_result["state"]
        state2 = state2_result["state"]

        return self.calculate_state_diff(state1, state2)

    async def rollback_to_checkpoint(
        self,
        pipeline_id: str,
        checkpoint_id: str,
    ) -> bool:
        """
        Rollback to a specific checkpoint.

        Args:
            pipeline_id: Pipeline ID
            checkpoint_id: Checkpoint ID to rollback to

        Returns:
            True if rollback successful, False otherwise
        """
        restored = await self.restore_checkpoint(pipeline_id, checkpoint_id)

        if restored:
            # Save a new checkpoint for the rollback
            metadata = {
                "rollback_from": checkpoint_id,
                "rollback_timestamp": datetime.now().isoformat(),
            }
            state = restored["state"]
            await self.save_checkpoint(pipeline_id, state, metadata)
            return True

        return False

    def _validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate state structure.

        Args:
            state: State to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(state, dict):
            return False

        # Check for required fields
        required_fields = ["pipeline_id", "execution_id"]
        for field in required_fields:
            if field not in state:
                return False

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get state manager statistics.

        Returns:
            Statistics dictionary
        """
        storage_info = {}

        # Add storage-specific information
        if hasattr(self.backend, "storage_path"):
            storage_info["storage_path"] = self.backend.storage_path

        # Calculate compression ratio
        compression_ratio = 0.0
        if self.stats["bytes_saved"] > 0:
            compression_ratio = (
                1.0 - (self.stats["bytes_compressed"] / self.stats["bytes_saved"])
            ) * 100

        # Map backend type to expected name
        backend_name_map = {
            "MemoryBackend": "memory",
            "FileBackend": "file",
            "PostgresBackend": "postgres",
            "RedisBackend": "redis",
        }
        backend_type_name = type(self.backend).__name__
        storage_backend = backend_name_map.get(
            backend_type_name, backend_type_name.lower()
        )

        return {
            **self.stats,
            "total_checkpoints": self.stats.get("checkpoints_created", 0),
            "total_pipelines": len(set()),  # Will need to track this properly
            "storage_backend": storage_backend,
            "compression_enabled": self.compression_enabled,
            "retention_days": getattr(self, "retention_days", 7),
            "checkpoint_strategy": (
                getattr(self.checkpoint_strategy, "__class__.__name__", "default")
                if hasattr(self, "checkpoint_strategy")
                else "default"
            ),
            "compression_ratio": compression_ratio,
            "backend_type": backend_type_name,
            **storage_info,
        }

    async def is_healthy(self) -> bool:
        """Check if the state manager is healthy and operational."""
        try:
            # Check if backend is available
            if not hasattr(self, "backend") or self.backend is None:
                return False

            # Try to perform a simple operation to verify backend is working
            # List checkpoints with a small limit to test connectivity
            await self.backend.list_checkpoints(limit=1)

            # Check if the storage path exists for file backend
            if (
                hasattr(self.backend, "__class__")
                and self.backend.__class__.__name__ == "FileBackend"
            ):
                if hasattr(self, "storage_path") and not os.path.exists(
                    self.storage_path
                ):
                    return False

            return True

        except Exception:
            # Any exception means the state manager is not healthy
            return False

    async def shutdown(self) -> None:
        """Shutdown state manager and clean up resources."""
        # Clean up backend connections and resources
        if hasattr(self.backend, "shutdown"):
            await self.backend.shutdown()
        elif hasattr(self.backend, "cleanup"):
            await self.backend.cleanup()
        elif hasattr(self.backend, "close"):
            if asyncio.iscoroutinefunction(self.backend.close):
                await self.backend.close()
            else:
                self.backend.close()


# For backward compatibility
async def create_state_manager(
    backend_type: str = "file", backend_config: Dict[str, Any] = None, **kwargs
) -> StateManager:
    """
    Create a state manager instance.

    Args:
        backend_type: Type of backend
        backend_config: Backend configuration
        **kwargs: Additional arguments

    Returns:
        StateManager instance
    """
    return StateManager(
        backend_type=backend_type, backend_config=backend_config, **kwargs
    )
