"""Simple file-based state manager for basic use cases."""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class StateManager:
    """
    Simple file-based state manager for pipeline checkpointing.

    This is a simplified version focused on direct file storage
    without the complexity of the backend architecture.
    """

    def __init__(self, storage_path: str = "./checkpoints") -> None:
        """
        Initialize state manager with storage path.

        Args:
            storage_path: Directory to store checkpoint files
        """
        self.storage_path = storage_path
        self._ensure_storage_path()

    def _ensure_storage_path(self) -> None:
        """Ensure storage directory exists."""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)

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
        timestamp_int = int(time.time())
        timestamp_iso = datetime.fromtimestamp(timestamp_int).isoformat()
        checkpoint_id = f"{execution_id}_{timestamp_int}"

        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "execution_id": execution_id,
            "state": state,
            "metadata": metadata or {},
            "timestamp": timestamp_iso,
            "version": "1.0",
        }

        filename = f"{checkpoint_id}.json"
        filepath = os.path.join(self.storage_path, filename)

        with open(filepath, "w") as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        return checkpoint_id

    async def restore_checkpoint(
        self, execution_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Restore pipeline state from checkpoint.

        Args:
            execution_id: Execution ID
            checkpoint_id: Specific checkpoint ID (if None, get latest)

        Returns:
            Checkpoint data or None if not found
        """
        if checkpoint_id is None:
            # Find latest checkpoint for execution
            checkpoints = await self.list_checkpoints(execution_id)
            if not checkpoints:
                return None
            # Get most recent
            latest = max(checkpoints, key=lambda x: x.get("timestamp", 0))
            checkpoint_id = latest["checkpoint_id"]

        filename = f"{checkpoint_id}.json"
        filepath = os.path.join(self.storage_path, filename)

        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, KeyError):
            return None

    async def list_checkpoints(self, execution_id: str) -> List[Dict[str, Any]]:
        """
        List checkpoints for an execution.

        Args:
            execution_id: Execution ID to filter by

        Returns:
            List of checkpoint metadata
        """
        if not os.path.exists(self.storage_path):
            return []

        checkpoints = []

        for filename in os.listdir(self.storage_path):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.storage_path, filename)

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                if data.get("execution_id") == execution_id:
                    checkpoints.append(data)
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by timestamp
        checkpoints.sort(key=lambda x: x.get("timestamp", 0))

        return checkpoints

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a specific checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to delete

        Returns:
            True if deleted, False if not found
        """
        filename = f"{checkpoint_id}.json"
        filepath = os.path.join(self.storage_path, filename)

        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                return True
            except OSError:
                return False

        return False

    async def delete_execution_checkpoints(self, execution_id: str) -> int:
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

    async def cleanup_old_checkpoints(self, max_age_days: int) -> int:
        """
        Clean up checkpoints older than specified age.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of deleted checkpoints
        """
        if not os.path.exists(self.storage_path):
            return 0

        current_time = time.time()
        cutoff_time = current_time - (max_age_days * 24 * 60 * 60)
        deleted_count = 0

        for filename in os.listdir(self.storage_path):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.storage_path, filename)

            try:
                # Check file modification time
                file_mtime = os.path.getmtime(filepath)

                if file_mtime < cutoff_time:
                    os.remove(filepath)
                    deleted_count += 1
            except OSError:
                # Skip files we can't access
                continue

        return deleted_count

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about storage usage.

        Returns:
            Storage information dictionary
        """
        info = {
            "storage_path": self.storage_path,
            "total_checkpoints": 0,
            "total_size_bytes": 0,
            "total_size_mb": 0.0,
        }

        if not os.path.exists(self.storage_path):
            return info

        total_size = 0
        checkpoint_count = 0

        for filename in os.listdir(self.storage_path):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.storage_path, filename)

            try:
                file_size = os.path.getsize(filepath)
                total_size += file_size
                checkpoint_count += 1
            except OSError:
                # Skip files we can't access
                continue

        info["total_checkpoints"] = checkpoint_count
        info["total_size_bytes"] = total_size
        info["total_size_mb"] = total_size / (1024 * 1024)

        return info


# For backward compatibility, also create InMemoryStateManager
class InMemoryStateManager:
    """
    In-memory state manager for testing and temporary use.
    """

    def __init__(self) -> None:
        """Initialize in-memory state manager."""
        self._checkpoints = {}
        self._execution_checkpoints = {}

    async def save_checkpoint(
        self,
        execution_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save checkpoint in memory."""
        timestamp = int(time.time())
        checkpoint_id = f"{execution_id}_{timestamp}"

        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "execution_id": execution_id,
            "state": state,
            "metadata": metadata or {},
            "timestamp": timestamp,
            "version": "1.0",
        }

        self._checkpoints[checkpoint_id] = checkpoint_data

        if execution_id not in self._execution_checkpoints:
            self._execution_checkpoints[execution_id] = []
        self._execution_checkpoints[execution_id].append(checkpoint_id)

        return checkpoint_id

    async def restore_checkpoint(
        self, execution_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Restore checkpoint from memory."""
        if checkpoint_id is None:
            # Find latest checkpoint for execution
            if execution_id not in self._execution_checkpoints:
                return None

            checkpoint_ids = self._execution_checkpoints[execution_id]
            if not checkpoint_ids:
                return None

            # Get most recent (last in list)
            checkpoint_id = checkpoint_ids[-1]

        return self._checkpoints.get(checkpoint_id)

    async def list_checkpoints(self, execution_id: str) -> List[Dict[str, Any]]:
        """List checkpoints for execution."""
        if execution_id not in self._execution_checkpoints:
            return []

        checkpoint_ids = self._execution_checkpoints[execution_id]
        checkpoints = []

        for checkpoint_id in checkpoint_ids:
            if checkpoint_id in self._checkpoints:
                checkpoints.append(self._checkpoints[checkpoint_id])

        # Sort by timestamp
        checkpoints.sort(key=lambda x: x.get("timestamp", 0))

        return checkpoints

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from memory."""
        if checkpoint_id in self._checkpoints:
            checkpoint_data = self._checkpoints.pop(checkpoint_id)
            execution_id = checkpoint_data["execution_id"]

            if execution_id in self._execution_checkpoints:
                try:
                    self._execution_checkpoints[execution_id].remove(checkpoint_id)
                except ValueError:
                    pass

            return True

        return False

    async def delete_execution_checkpoints(self, execution_id: str) -> int:
        """Delete all checkpoints for execution."""
        if execution_id not in self._execution_checkpoints:
            return 0

        checkpoint_ids = self._execution_checkpoints[execution_id][:]
        deleted_count = 0

        for checkpoint_id in checkpoint_ids:
            if await self.delete_checkpoint(checkpoint_id):
                deleted_count += 1

        return deleted_count

    def get_storage_info(self) -> Dict[str, Any]:
        """Get memory storage info."""
        return {
            "storage_path": "memory",
            "total_checkpoints": len(self._checkpoints),
            "total_size_bytes": 0,  # Not applicable for memory
            "total_size_mb": 0.0,
        }
