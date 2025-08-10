"""Legacy Compatibility Adapter - Issue #204

Provides backward compatibility for existing StateManager interface
while using the new LangGraph-based state management system underneath.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from contextlib import asynccontextmanager

from .langgraph_state_manager import LangGraphGlobalContextManager
from .global_context import PipelineGlobalState, PipelineStatus

logger = logging.getLogger(__name__)


class LegacyStateManagerAdapter:
    """
    Adapter that provides the legacy StateManager interface
    while using LangGraph state management underneath.
    
    This allows existing code to work without modification during
    the migration to LangGraph-based state management.
    """
    
    def __init__(self, langgraph_manager: LangGraphGlobalContextManager):
        """
        Initialize legacy adapter.
        
        Args:
            langgraph_manager: LangGraph state manager instance
        """
        self.langgraph_manager = langgraph_manager
        self._execution_to_thread_mapping: Dict[str, str] = {}
        self._checkpoint_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Statistics for compatibility
        self.stats = {
            "checkpoints_created": 0,
            "checkpoints_restored": 0, 
            "checkpoints_deleted": 0,
            "bytes_saved": 0,
            "bytes_compressed": 0
        }
        
        logger.info("Legacy StateManager compatibility adapter initialized")
        
    async def save_checkpoint(self,
                            execution_id: str,
                            state: Dict[str, Any],
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save checkpoint using legacy interface.
        
        Args:
            execution_id: Legacy execution ID
            state: Pipeline state to save
            metadata: Optional metadata
            
        Returns:
            Checkpoint ID
        """
        try:
            # Map execution_id to thread_id
            if execution_id not in self._execution_to_thread_mapping:
                # Create new pipeline state
                thread_id = await self.langgraph_manager.initialize_pipeline_state(
                    pipeline_id=state.get("pipeline_id", execution_id),
                    inputs=state.get("inputs", {}),
                    user_id=state.get("user_id"),
                    session_id=state.get("session_id")
                )
                self._execution_to_thread_mapping[execution_id] = thread_id
            else:
                thread_id = self._execution_to_thread_mapping[execution_id]
                
            # Convert legacy state to LangGraph format
            langgraph_updates = self._convert_legacy_state_to_langgraph(state)
            
            # Update state
            await self.langgraph_manager.update_global_state(thread_id, langgraph_updates)
            
            # Create checkpoint
            description = metadata.get("description", f"Legacy checkpoint for {execution_id}") if metadata else ""
            checkpoint_id = await self.langgraph_manager.create_checkpoint(
                thread_id,
                description=description,
                metadata=metadata
            )
            
            # Store metadata for legacy compatibility
            self._checkpoint_metadata[checkpoint_id] = {
                "execution_id": execution_id,
                "thread_id": thread_id,
                "timestamp": time.time(),
                "metadata": metadata or {}
            }
            
            # Update stats
            self.stats["checkpoints_created"] += 1
            # Estimate bytes (rough approximation)
            import json
            state_size = len(json.dumps(state, default=str).encode('utf-8'))
            self.stats["bytes_saved"] += state_size
            
            logger.debug(f"Legacy checkpoint saved: {checkpoint_id} for execution: {execution_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save legacy checkpoint for {execution_id}: {e}")
            raise
            
    async def restore_checkpoint(self,
                               pipeline_id: str,
                               checkpoint_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Restore checkpoint using legacy interface.
        
        Args:
            pipeline_id: Pipeline ID (used as execution_id in legacy interface)
            checkpoint_id: Specific checkpoint ID (optional)
            
        Returns:
            Legacy format checkpoint data or None
        """
        try:
            execution_id = pipeline_id  # In legacy interface, pipeline_id is execution_id
            
            if checkpoint_id is None:
                # Get latest checkpoint for execution_id
                if execution_id not in self._execution_to_thread_mapping:
                    logger.warning(f"No thread mapping found for execution_id: {execution_id}")
                    return None
                    
                thread_id = self._execution_to_thread_mapping[execution_id]
                state = await self.langgraph_manager.get_global_state(thread_id)
                
                if state is None:
                    return None
                    
                # Convert to legacy format
                legacy_state = self._convert_langgraph_state_to_legacy(state)
                
                return {
                    "pipeline_id": pipeline_id,
                    "checkpoint_id": "latest",
                    "state": legacy_state,
                    "metadata": {
                        "timestamp": time.time(),
                        "thread_id": thread_id
                    },
                    "timestamp": time.time(),
                    "compressed": False
                }
            else:
                # Restore specific checkpoint
                if checkpoint_id not in self._checkpoint_metadata:
                    logger.warning(f"Checkpoint metadata not found: {checkpoint_id}")
                    return None
                    
                checkpoint_meta = self._checkpoint_metadata[checkpoint_id]
                thread_id = checkpoint_meta["thread_id"]
                
                state = await self.langgraph_manager.restore_from_checkpoint(thread_id, checkpoint_id)
                
                if state is None:
                    return None
                    
                # Convert to legacy format
                legacy_state = self._convert_langgraph_state_to_legacy(state)
                
                # Update stats
                self.stats["checkpoints_restored"] += 1
                
                return {
                    "pipeline_id": pipeline_id,
                    "checkpoint_id": checkpoint_id,
                    "state": legacy_state,
                    "metadata": checkpoint_meta["metadata"],
                    "timestamp": checkpoint_meta["timestamp"],
                    "compressed": False
                }
                
        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            return None
            
    async def restore_checkpoint_by_timestamp(self,
                                            execution_id: str,
                                            timestamp: float) -> Optional[Dict[str, Any]]:
        """
        Restore checkpoint closest to given timestamp.
        
        Args:
            execution_id: Execution ID
            timestamp: Target timestamp
            
        Returns:
            Closest checkpoint or None
        """
        try:
            if execution_id not in self._execution_to_thread_mapping:
                return None
                
            thread_id = self._execution_to_thread_mapping[execution_id]
            checkpoints = await self.langgraph_manager.list_checkpoints(thread_id)
            
            if not checkpoints:
                return None
                
            # Find closest checkpoint by timestamp
            closest_checkpoint = min(
                checkpoints,
                key=lambda x: abs(x["timestamp"] - timestamp)
            )
            
            return await self.restore_checkpoint(execution_id, closest_checkpoint["checkpoint_id"])
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint by timestamp: {e}")
            return None
            
    async def list_checkpoints(self,
                             execution_id: str = None,
                             limit: int = None) -> List[Dict[str, Any]]:
        """
        List available checkpoints in legacy format.
        
        Args:
            execution_id: Optional execution ID filter
            limit: Maximum number of results
            
        Returns:
            List of checkpoint metadata in legacy format
        """
        try:
            if execution_id:
                if execution_id not in self._execution_to_thread_mapping:
                    return []
                    
                thread_id = self._execution_to_thread_mapping[execution_id]
                checkpoints = await self.langgraph_manager.list_checkpoints(thread_id)
            else:
                # List all checkpoints across all threads
                checkpoints = []
                for exec_id, thread_id in self._execution_to_thread_mapping.items():
                    thread_checkpoints = await self.langgraph_manager.list_checkpoints(thread_id)
                    for cp in thread_checkpoints:
                        cp["execution_id"] = exec_id
                    checkpoints.extend(thread_checkpoints)
                    
            # Convert to legacy format
            legacy_checkpoints = []
            for checkpoint in checkpoints:
                legacy_checkpoint = {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "timestamp": checkpoint["timestamp"],
                    "description": checkpoint.get("description", ""),
                    "execution_id": checkpoint.get("execution_id", execution_id),
                    "metadata": {
                        "step": checkpoint.get("step", 0),
                        "thread_id": checkpoint["thread_id"]
                    }
                }
                legacy_checkpoints.append(legacy_checkpoint)
                
            # Apply limit if specified
            if limit:
                legacy_checkpoints = legacy_checkpoints[:limit]
                
            return legacy_checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
            
    async def delete_checkpoint(self,
                              pipeline_id: str,
                              checkpoint_id: str = None) -> bool:
        """
        Delete checkpoint using legacy interface.
        
        Args:
            pipeline_id: Pipeline ID (execution_id in legacy interface)
            checkpoint_id: Checkpoint ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            if checkpoint_id is None:
                checkpoint_id = pipeline_id  # Legacy interface variation
                
            if checkpoint_id in self._checkpoint_metadata:
                checkpoint_meta = self._checkpoint_metadata[checkpoint_id]
                thread_id = checkpoint_meta["thread_id"]
                
                success = await self.langgraph_manager.delete_checkpoint(thread_id, checkpoint_id)
                
                if success:
                    del self._checkpoint_metadata[checkpoint_id]
                    self.stats["checkpoints_deleted"] += 1
                    
                return success
            else:
                logger.warning(f"Checkpoint not found for deletion: {checkpoint_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
            
    async def delete_all_checkpoints(self, execution_id: str) -> int:
        """
        Delete all checkpoints for an execution.
        
        Args:
            execution_id: Execution ID
            
        Returns:
            Number of deleted checkpoints
        """
        try:
            checkpoints = await self.list_checkpoints(execution_id)
            deleted_count = 0
            
            for checkpoint in checkpoints:
                if await self.delete_checkpoint(execution_id, checkpoint["checkpoint_id"]):
                    deleted_count += 1
                    
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete all checkpoints for {execution_id}: {e}")
            return 0
            
    async def cleanup_expired_checkpoints(self, retention_days: int = 7) -> int:
        """
        Clean up expired checkpoints.
        
        Args:
            retention_days: Number of days to retain
            
        Returns:
            Number of cleaned up checkpoints
        """
        return await self.langgraph_manager.cleanup_expired_checkpoints(retention_days)
        
    @asynccontextmanager
    async def checkpoint_context(self,
                               execution_id: str,
                               state: Dict[str, Any],
                               metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for checkpoint operations (legacy compatibility).
        
        Args:
            execution_id: Execution ID
            state: Pipeline state
            metadata: Optional metadata
        """
        checkpoint_id = await self.save_checkpoint(execution_id, state, metadata)
        
        try:
            yield checkpoint_id
        except Exception as e:
            # Save error checkpoint
            error_state = {**state, "error": str(e), "error_type": type(e).__name__}
            error_metadata = {
                "error": str(e),
                "error_type": type(e).__name__,
                **(metadata or {})
            }
            await self.save_checkpoint(execution_id, error_state, error_metadata)
            raise
            
    async def get_pipeline_state(self,
                               execution_id: str,
                               checkpoint_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Get pipeline state from checkpoint (legacy compatibility).
        
        Args:
            execution_id: Execution ID
            checkpoint_id: Optional specific checkpoint
            
        Returns:
            Pipeline state or None
        """
        checkpoint_data = await self.restore_checkpoint(execution_id, checkpoint_id)
        return checkpoint_data["state"] if checkpoint_data else None
        
    async def get_pipeline_history(self,
                                 execution_id: str,
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get pipeline execution history (legacy compatibility).
        
        Args:
            execution_id: Execution ID
            limit: Maximum number of history entries
            
        Returns:
            List of checkpoint metadata
        """
        return await self.list_checkpoints(execution_id, limit)
        
    def calculate_state_diff(self,
                           state1: Dict[str, Any],
                           state2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate difference between two states.
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            State differences
        """
        diff = {"added": {}, "removed": {}, "modified": {}}
        
        # Find added and modified keys
        for key, value2 in state2.items():
            if key not in state1:
                diff["added"][key] = value2
            elif state1[key] != value2:
                diff["modified"][key] = {"from": state1[key], "to": value2}
                
        # Find removed keys
        for key, value1 in state1.items():
            if key not in state2:
                diff["removed"][key] = value1
                
        return diff
        
    async def get_state_diff(self,
                           pipeline_id: str,
                           checkpoint_id1: str,
                           checkpoint_id2: str) -> Dict[str, Any]:
        """
        Get difference between two checkpoints.
        
        Args:
            pipeline_id: Pipeline ID
            checkpoint_id1: First checkpoint
            checkpoint_id2: Second checkpoint
            
        Returns:
            State differences
        """
        state1_result = await self.restore_checkpoint(pipeline_id, checkpoint_id1)
        state2_result = await self.restore_checkpoint(pipeline_id, checkpoint_id2)
        
        if state1_result is None or state2_result is None:
            raise ValueError("One or both checkpoints not found")
            
        return self.calculate_state_diff(state1_result["state"], state2_result["state"])
        
    async def rollback_to_checkpoint(self,
                                   pipeline_id: str,
                                   checkpoint_id: str) -> bool:
        """
        Rollback to specific checkpoint.
        
        Args:
            pipeline_id: Pipeline ID
            checkpoint_id: Checkpoint to rollback to
            
        Returns:
            True if successful
        """
        try:
            restored = await self.restore_checkpoint(pipeline_id, checkpoint_id)
            
            if restored:
                # Save new checkpoint for rollback
                metadata = {
                    "rollback_from": checkpoint_id,
                    "rollback_timestamp": datetime.now().isoformat()
                }
                state = restored["state"]
                await self.save_checkpoint(pipeline_id, state, metadata)
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to rollback to checkpoint {checkpoint_id}: {e}")
            return False
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get legacy-compatible statistics."""
        langgraph_metrics = self.langgraph_manager.get_metrics()
        
        return {
            **self.stats,
            "total_checkpoints": self.stats["checkpoints_created"],
            "total_pipelines": len(self._execution_to_thread_mapping),
            "storage_backend": langgraph_metrics["storage_backend"],
            "compression_enabled": langgraph_metrics["memory_optimization_enabled"],
            "retention_days": 7,  # Default
            "checkpoint_strategy": "langgraph_adaptive",
            "compression_ratio": 0.0,  # Would need to calculate
            "backend_type": "LangGraphAdapter",
            "active_sessions": langgraph_metrics["active_sessions"]
        }
        
    async def is_healthy(self) -> bool:
        """Check if state manager is healthy."""
        try:
            health_status = await self.langgraph_manager.health_check()
            return health_status["status"] == "healthy"
        except Exception:
            return False
            
    async def shutdown(self) -> None:
        """Shutdown and clean up resources."""
        await self.langgraph_manager.shutdown()
        self._execution_to_thread_mapping.clear()
        self._checkpoint_metadata.clear()
        
    def _convert_legacy_state_to_langgraph(self, legacy_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert legacy state format to LangGraph format.
        
        Args:
            legacy_state: Legacy state dictionary
            
        Returns:
            LangGraph-compatible state updates
        """
        updates = {}
        
        # Map common legacy fields to LangGraph structure
        if "inputs" in legacy_state:
            updates["inputs"] = legacy_state["inputs"]
            
        if "outputs" in legacy_state:
            updates["outputs"] = legacy_state["outputs"]
            
        if "intermediate_results" in legacy_state:
            updates["intermediate_results"] = legacy_state["intermediate_results"]
            
        # Map execution metadata
        execution_metadata_updates = {}
        if "pipeline_id" in legacy_state:
            execution_metadata_updates["pipeline_id"] = legacy_state["pipeline_id"]
        if "execution_id" in legacy_state:
            execution_metadata_updates["execution_id"] = legacy_state["execution_id"]
        if "current_step" in legacy_state:
            execution_metadata_updates["current_step"] = legacy_state["current_step"]
        if "completed_tasks" in legacy_state:
            execution_metadata_updates["completed_steps"] = legacy_state["completed_tasks"]
        if "status" in legacy_state:
            execution_metadata_updates["status"] = legacy_state["status"]
            
        if execution_metadata_updates:
            updates["execution_metadata"] = execution_metadata_updates
            
        # Map context to global variables
        if "context" in legacy_state:
            updates["global_variables"] = legacy_state["context"]
            
        # Map error information
        if "error" in legacy_state:
            updates["error_context"] = {
                "current_error": {
                    "message": legacy_state["error"],
                    "type": legacy_state.get("error_type", "Unknown"),
                    "timestamp": time.time()
                }
            }
            
        return updates
        
    def _convert_langgraph_state_to_legacy(self, langgraph_state: PipelineGlobalState) -> Dict[str, Any]:
        """
        Convert LangGraph state to legacy format.
        
        Args:
            langgraph_state: LangGraph state
            
        Returns:
            Legacy-compatible state dictionary
        """
        legacy_state = {}
        
        # Core data
        legacy_state["inputs"] = langgraph_state.get("inputs", {})
        legacy_state["outputs"] = langgraph_state.get("outputs", {})
        legacy_state["intermediate_results"] = langgraph_state.get("intermediate_results", {})
        
        # Execution metadata
        exec_meta = langgraph_state.get("execution_metadata", {})
        legacy_state["pipeline_id"] = exec_meta.get("pipeline_id", "unknown")
        legacy_state["execution_id"] = exec_meta.get("execution_id", "unknown")
        legacy_state["current_step"] = exec_meta.get("current_step", "unknown")
        legacy_state["completed_tasks"] = exec_meta.get("completed_steps", [])
        legacy_state["status"] = exec_meta.get("status", "unknown")
        
        # Context from global variables
        legacy_state["context"] = langgraph_state.get("global_variables", {})
        
        # Error information
        error_ctx = langgraph_state.get("error_context", {})
        if "current_error" in error_ctx:
            current_error = error_ctx["current_error"]
            legacy_state["error"] = current_error.get("message", "Unknown error")
            legacy_state["error_type"] = current_error.get("type", "Unknown")
            
        return legacy_state


__all__ = [
    "LegacyStateManagerAdapter"
]