"""State management and checkpointing system."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


class StateManager:
    """
    Manages pipeline state and checkpointing.
    
    Provides functionality to save and restore pipeline states
    for recovery and resumption of failed executions.
    """
    
    def __init__(self, storage_path: str = "./checkpoints") -> None:
        """
        Initialize state manager.
        
        Args:
            storage_path: Path to store checkpoint files
        """
        self.storage_path = storage_path
        self._ensure_storage_path()
    
    def _ensure_storage_path(self) -> None:
        """Ensure storage path exists."""
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
        checkpoint_id = f"{execution_id}_{int(time.time())}"
        
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "execution_id": execution_id,
            "state": state,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0",
        }
        
        # Save to file
        filename = f"{checkpoint_id}.json"
        filepath = os.path.join(self.storage_path, filename)
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        return checkpoint_id
    
    async def restore_checkpoint(
        self,
        execution_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Restore pipeline state from checkpoint.
        
        Args:
            execution_id: Execution ID
            checkpoint_id: Specific checkpoint ID (latest if None)
            
        Returns:
            Checkpoint data or None if not found
        """
        if checkpoint_id:
            # Load specific checkpoint
            filename = f"{checkpoint_id}.json"
            filepath = os.path.join(self.storage_path, filename)
            
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        else:
            # Find latest checkpoint for execution
            latest_checkpoint = None
            latest_time = 0
            
            for filename in os.listdir(self.storage_path):
                if filename.startswith(execution_id) and filename.endswith('.json'):
                    filepath = os.path.join(self.storage_path, filename)
                    try:
                        with open(filepath, 'r') as f:
                            checkpoint = json.load(f)
                            
                        # Parse timestamp
                        timestamp = datetime.fromisoformat(checkpoint["timestamp"])
                        if timestamp.timestamp() > latest_time:
                            latest_time = timestamp.timestamp()
                            latest_checkpoint = checkpoint
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            return latest_checkpoint
        
        return None
    
    async def list_checkpoints(self, execution_id: str) -> List[Dict[str, Any]]:
        """
        List all checkpoints for an execution.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            List of checkpoint metadata
        """
        checkpoints = []
        
        for filename in os.listdir(self.storage_path):
            if filename.startswith(execution_id) and filename.endswith('.json'):
                filepath = os.path.join(self.storage_path, filename)
                try:
                    with open(filepath, 'r') as f:
                        checkpoint = json.load(f)
                    
                    # Return metadata only
                    checkpoints.append({
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "execution_id": checkpoint["execution_id"],
                        "timestamp": checkpoint["timestamp"],
                        "metadata": checkpoint.get("metadata", {}),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x["timestamp"])
        
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
            os.remove(filepath)
            return True
        
        return False
    
    async def delete_execution_checkpoints(self, execution_id: str) -> int:
        """
        Delete all checkpoints for an execution.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Number of checkpoints deleted
        """
        deleted_count = 0
        
        for filename in os.listdir(self.storage_path):
            if filename.startswith(execution_id) and filename.endswith('.json'):
                filepath = os.path.join(self.storage_path, filename)
                os.remove(filepath)
                deleted_count += 1
        
        return deleted_count
    
    async def cleanup_old_checkpoints(self, max_age_days: int = 30) -> int:
        """
        Clean up old checkpoints.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of checkpoints cleaned up
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        deleted_count = 0
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_path, filename)
                
                try:
                    # Check file modification time
                    file_time = os.path.getmtime(filepath)
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        deleted_count += 1
                except OSError:
                    continue
        
        return deleted_count
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage information.
        
        Returns:
            Storage statistics
        """
        total_files = 0
        total_size = 0
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_path, filename)
                try:
                    file_size = os.path.getsize(filepath)
                    total_files += 1
                    total_size += file_size
                except OSError:
                    continue
        
        return {
            "storage_path": self.storage_path,
            "total_checkpoints": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }