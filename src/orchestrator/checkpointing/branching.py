"""Checkpoint Branching and Rollback - Issue #205 Phase 2

Implements checkpoint branching from any execution point, rollback capabilities
with state restoration, and merge operations between branches.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass
from enum import Enum

# Internal imports
from ..state.global_context import (
    PipelineGlobalState,
    validate_pipeline_state,
    merge_pipeline_states,
    PipelineStatus
)
from ..state.langgraph_state_manager import LangGraphGlobalContextManager
from ..core.exceptions import PipelineExecutionError

logger = logging.getLogger(__name__)


class BranchStatus(Enum):
    """Status of execution branches."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    MERGED = "merged"
    ABANDONED = "abandoned"


class MergeStrategy(Enum):
    """Strategies for merging branches."""
    OVERWRITE = "overwrite"  # Target branch overwrites source
    MERGE_RECURSIVE = "merge_recursive"  # Recursive merge with conflict resolution
    CHERRY_PICK = "cherry_pick"  # Select specific changes to merge
    MANUAL = "manual"  # Manual merge with human intervention


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    FAVOR_SOURCE = "favor_source"
    FAVOR_TARGET = "favor_target"
    MERGE_BOTH = "merge_both"
    REQUIRE_MANUAL = "require_manual"


@dataclass
class BranchInfo:
    """Information about an execution branch."""
    branch_id: str
    thread_id: str
    parent_thread_id: str
    parent_checkpoint_id: str
    branch_name: str
    description: str
    status: BranchStatus
    created_at: float
    created_by: Optional[str] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MergeConflict:
    """Represents a merge conflict between branches."""
    field_path: str
    source_value: Any
    target_value: Any
    conflict_type: str  # "value_mismatch", "type_mismatch", "missing_field"
    resolution: Optional[ConflictResolution] = None
    resolved_value: Optional[Any] = None


@dataclass
class MergeResult:
    """Result of a branch merge operation."""
    success: bool
    merged_thread_id: str
    source_branch_id: str
    target_thread_id: str
    conflicts: List[MergeConflict] = None
    merge_checkpoint_id: Optional[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.conflicts is None:
            self.conflicts = []


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    success: bool
    thread_id: str
    rollback_checkpoint_id: str
    target_checkpoint_id: str
    steps_rolled_back: int
    restored_state: Optional[PipelineGlobalState] = None
    error_message: Optional[str] = None


class CheckpointBranchingManager:
    """
    Manages checkpoint branching, rollback, and merge operations.
    
    Provides capabilities for:
    - Creating execution branches from any checkpoint
    - Rolling back to previous checkpoints with state restoration
    - Merging branch results with conflict resolution
    - Managing branch lifecycle and cleanup
    """
    
    def __init__(
        self,
        langgraph_manager: LangGraphGlobalContextManager,
        max_branch_depth: int = 10,
        auto_cleanup_abandoned_branches: bool = True,
        default_merge_strategy: MergeStrategy = MergeStrategy.MERGE_RECURSIVE,
        branch_retention_hours: int = 168,  # 7 days
    ):
        """
        Initialize checkpoint branching manager.
        
        Args:
            langgraph_manager: LangGraph state manager for persistence
            max_branch_depth: Maximum allowed branch depth
            auto_cleanup_abandoned_branches: Automatically clean up abandoned branches
            default_merge_strategy: Default strategy for merge operations
            branch_retention_hours: Hours to retain completed/abandoned branches
        """
        self.langgraph_manager = langgraph_manager
        self.max_branch_depth = max_branch_depth
        self.auto_cleanup_abandoned_branches = auto_cleanup_abandoned_branches
        self.default_merge_strategy = default_merge_strategy
        self.branch_retention_hours = branch_retention_hours
        
        # Branch tracking
        self.active_branches: Dict[str, BranchInfo] = {}
        self.branch_hierarchy: Dict[str, List[str]] = {}  # parent -> children
        
        # Merge and rollback operations
        self._operation_locks: Dict[str, asyncio.Lock] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = {
            "branches_created": 0,
            "branches_merged": 0,
            "rollbacks_performed": 0,
            "conflicts_resolved": 0,
            "abandoned_branches": 0,
            "active_branches": 0
        }
        
        # Start cleanup task
        self._start_cleanup_task()
        
        logger.info("CheckpointBranchingManager initialized")
    
    async def create_branch(
        self,
        source_thread_id: str,
        source_checkpoint_id: str,
        branch_name: str,
        description: str = "",
        created_by: Optional[str] = None
    ) -> BranchInfo:
        """
        Create a new execution branch from a checkpoint.
        
        Args:
            source_thread_id: Source thread to branch from
            source_checkpoint_id: Checkpoint to branch from
            branch_name: Human-readable name for the branch
            description: Optional description of the branch purpose
            created_by: Optional user who created the branch
            
        Returns:
            Branch information object
            
        Raises:
            RuntimeError: If branch depth limit exceeded
        """
        # Check branch depth limit
        depth = await self._calculate_branch_depth(source_thread_id)
        if depth >= self.max_branch_depth:
            raise RuntimeError(f"Maximum branch depth ({self.max_branch_depth}) exceeded")
        
        # Generate branch identifiers
        branch_id = f"branch_{uuid.uuid4().hex[:8]}"
        new_thread_id = f"{source_thread_id}_branch_{branch_id}"
        
        # Get source state from checkpoint
        source_state = await self.langgraph_manager.get_global_state(source_thread_id)
        if not source_state:
            raise ValueError(f"Could not get state for source thread: {source_thread_id}")
        
        # Create branch state
        branch_state = source_state.copy()
        branch_state["thread_id"] = new_thread_id
        branch_state["execution_metadata"]["branch_info"] = {
            "branch_id": branch_id,
            "parent_thread_id": source_thread_id,
            "parent_checkpoint_id": source_checkpoint_id,
            "branch_name": branch_name,
            "created_at": time.time(),
            "created_by": created_by
        }
        
        # Initialize branch in LangGraph
        await self.langgraph_manager.initialize_pipeline_state(
            pipeline_id=branch_state["execution_metadata"]["pipeline_id"],
            inputs=branch_state.get("global_variables", {}).get("inputs", {}),
            user_id=created_by,
            session_id=f"branch_{branch_id}"
        )
        
        # Create initial branch checkpoint
        branch_checkpoint_id = await self.langgraph_manager.create_checkpoint(
            thread_id=new_thread_id,
            description=f"Branch created: {branch_name}",
            metadata={
                "branch_operation": "create",
                "source_thread_id": source_thread_id,
                "source_checkpoint_id": source_checkpoint_id,
                "branch_id": branch_id,
                "created_by": created_by
            }
        )
        
        # Create branch info
        branch_info = BranchInfo(
            branch_id=branch_id,
            thread_id=new_thread_id,
            parent_thread_id=source_thread_id,
            parent_checkpoint_id=source_checkpoint_id,
            branch_name=branch_name,
            description=description,
            status=BranchStatus.ACTIVE,
            created_at=time.time(),
            created_by=created_by,
            metadata={
                "initial_checkpoint_id": branch_checkpoint_id,
                "depth": depth + 1
            }
        )
        
        # Track branch
        self.active_branches[branch_id] = branch_info
        
        if source_thread_id not in self.branch_hierarchy:
            self.branch_hierarchy[source_thread_id] = []
        self.branch_hierarchy[source_thread_id].append(branch_id)
        
        # Update metrics
        self.metrics["branches_created"] += 1
        self.metrics["active_branches"] = len([b for b in self.active_branches.values() if b.status == BranchStatus.ACTIVE])
        
        logger.info(f"Created branch {branch_id} ({branch_name}) from {source_thread_id}")
        
        return branch_info
    
    async def rollback_to_checkpoint(
        self,
        thread_id: str,
        target_checkpoint_id: str,
        create_rollback_branch: bool = True,
        rollback_reason: Optional[str] = None
    ) -> RollbackResult:
        """
        Rollback thread execution to a specific checkpoint.
        
        Args:
            thread_id: Thread to rollback
            target_checkpoint_id: Checkpoint to rollback to
            create_rollback_branch: Whether to create a branch before rollback
            rollback_reason: Optional reason for the rollback
            
        Returns:
            Rollback result object
        """
        # Acquire operation lock
        async with await self._get_operation_lock(thread_id):
            
            try:
                # Get current state
                current_state = await self.langgraph_manager.get_global_state(thread_id)
                if not current_state:
                    return RollbackResult(
                        success=False,
                        thread_id=thread_id,
                        rollback_checkpoint_id="",
                        target_checkpoint_id=target_checkpoint_id,
                        steps_rolled_back=0,
                        error_message="Could not get current state"
                    )
                
                # Create rollback branch if requested
                rollback_branch_id = None
                if create_rollback_branch:
                    try:
                        branch = await self.create_branch(
                            source_thread_id=thread_id,
                            source_checkpoint_id="current",
                            branch_name=f"pre_rollback_{int(time.time())}",
                            description=f"State before rollback: {rollback_reason or 'No reason provided'}"
                        )
                        rollback_branch_id = branch.branch_id
                    except Exception as e:
                        logger.warning(f"Could not create rollback branch: {e}")
                
                # Find target checkpoint and restore state
                checkpoints = await self.langgraph_manager.list_checkpoints(thread_id)
                target_checkpoint = None
                
                for checkpoint in checkpoints:
                    if checkpoint.get("checkpoint_id") == target_checkpoint_id:
                        target_checkpoint = checkpoint
                        break
                
                if not target_checkpoint:
                    return RollbackResult(
                        success=False,
                        thread_id=thread_id,
                        rollback_checkpoint_id="",
                        target_checkpoint_id=target_checkpoint_id,
                        steps_rolled_back=0,
                        error_message=f"Target checkpoint {target_checkpoint_id} not found"
                    )
                
                # Restore state from checkpoint
                restored_state = await self.langgraph_manager.restore_from_checkpoint(
                    thread_id=thread_id,
                    checkpoint_id=target_checkpoint_id
                )
                
                if not restored_state:
                    return RollbackResult(
                        success=False,
                        thread_id=thread_id,
                        rollback_checkpoint_id="",
                        target_checkpoint_id=target_checkpoint_id,
                        steps_rolled_back=0,
                        error_message="Failed to restore state from checkpoint"
                    )
                
                # Calculate steps rolled back
                current_steps = len(current_state.get("execution_metadata", {}).get("completed_steps", []))
                restored_steps = len(restored_state.get("execution_metadata", {}).get("completed_steps", []))
                steps_rolled_back = max(0, current_steps - restored_steps)
                
                # Create rollback checkpoint
                rollback_checkpoint_id = await self.langgraph_manager.create_checkpoint(
                    thread_id=thread_id,
                    description=f"Rollback to {target_checkpoint_id}",
                    metadata={
                        "rollback_operation": True,
                        "target_checkpoint_id": target_checkpoint_id,
                        "steps_rolled_back": steps_rolled_back,
                        "rollback_reason": rollback_reason,
                        "rollback_branch_id": rollback_branch_id,
                        "rollback_timestamp": time.time()
                    }
                )
                
                # Update metrics
                self.metrics["rollbacks_performed"] += 1
                
                logger.info(f"Successfully rolled back thread {thread_id} to checkpoint {target_checkpoint_id}")
                
                return RollbackResult(
                    success=True,
                    thread_id=thread_id,
                    rollback_checkpoint_id=rollback_checkpoint_id,
                    target_checkpoint_id=target_checkpoint_id,
                    steps_rolled_back=steps_rolled_back,
                    restored_state=restored_state
                )
                
            except Exception as e:
                logger.error(f"Rollback operation failed for thread {thread_id}: {e}")
                return RollbackResult(
                    success=False,
                    thread_id=thread_id,
                    rollback_checkpoint_id="",
                    target_checkpoint_id=target_checkpoint_id,
                    steps_rolled_back=0,
                    error_message=str(e)
                )
    
    async def merge_branch(
        self,
        source_branch_id: str,
        target_thread_id: str,
        merge_strategy: Optional[MergeStrategy] = None,
        conflict_resolution: ConflictResolution = ConflictResolution.REQUIRE_MANUAL,
        merge_description: Optional[str] = None
    ) -> MergeResult:
        """
        Merge a branch back into target thread.
        
        Args:
            source_branch_id: Branch to merge from
            target_thread_id: Target thread to merge into
            merge_strategy: Strategy for merging (uses default if None)
            conflict_resolution: How to handle conflicts
            merge_description: Optional description of the merge
            
        Returns:
            Merge result object
        """
        strategy = merge_strategy or self.default_merge_strategy
        
        # Validate branch exists
        if source_branch_id not in self.active_branches:
            return MergeResult(
                success=False,
                merged_thread_id="",
                source_branch_id=source_branch_id,
                target_thread_id=target_thread_id,
                error_message=f"Branch {source_branch_id} not found"
            )
        
        source_branch = self.active_branches[source_branch_id]
        
        # Acquire operation locks
        async with await self._get_operation_lock(source_branch.thread_id):
            async with await self._get_operation_lock(target_thread_id):
                
                try:
                    # Get states
                    source_state = await self.langgraph_manager.get_global_state(source_branch.thread_id)
                    target_state = await self.langgraph_manager.get_global_state(target_thread_id)
                    
                    if not source_state or not target_state:
                        return MergeResult(
                            success=False,
                            merged_thread_id="",
                            source_branch_id=source_branch_id,
                            target_thread_id=target_thread_id,
                            error_message="Could not get source or target state"
                        )
                    
                    # Perform merge based on strategy
                    if strategy == MergeStrategy.OVERWRITE:
                        merged_state, conflicts = await self._merge_overwrite(source_state, target_state)
                    elif strategy == MergeStrategy.MERGE_RECURSIVE:
                        merged_state, conflicts = await self._merge_recursive(source_state, target_state, conflict_resolution)
                    elif strategy == MergeStrategy.CHERRY_PICK:
                        merged_state, conflicts = await self._merge_cherry_pick(source_state, target_state)
                    else:  # MANUAL
                        return MergeResult(
                            success=False,
                            merged_thread_id="",
                            source_branch_id=source_branch_id,
                            target_thread_id=target_thread_id,
                            error_message="Manual merge strategy requires human intervention"
                        )
                    
                    # Check for unresolved conflicts
                    unresolved_conflicts = [c for c in conflicts if c.resolution is None]
                    if unresolved_conflicts and conflict_resolution == ConflictResolution.REQUIRE_MANUAL:
                        return MergeResult(
                            success=False,
                            merged_thread_id="",
                            source_branch_id=source_branch_id,
                            target_thread_id=target_thread_id,
                            conflicts=conflicts,
                            error_message=f"Manual resolution required for {len(unresolved_conflicts)} conflicts"
                        )
                    
                    # Apply merged state
                    await self.langgraph_manager.update_global_state(target_thread_id, merged_state)
                    
                    # Create merge checkpoint
                    merge_checkpoint_id = await self.langgraph_manager.create_checkpoint(
                        thread_id=target_thread_id,
                        description=merge_description or f"Merged branch {source_branch.branch_name}",
                        metadata={
                            "merge_operation": True,
                            "source_branch_id": source_branch_id,
                            "merge_strategy": strategy.value,
                            "conflicts_count": len(conflicts),
                            "resolved_conflicts": len([c for c in conflicts if c.resolution is not None]),
                            "merge_timestamp": time.time()
                        }
                    )
                    
                    # Update branch status
                    source_branch.status = BranchStatus.MERGED
                    source_branch.completed_at = time.time()
                    
                    # Update metrics
                    self.metrics["branches_merged"] += 1
                    self.metrics["conflicts_resolved"] += len([c for c in conflicts if c.resolution is not None])
                    self.metrics["active_branches"] = len([b for b in self.active_branches.values() if b.status == BranchStatus.ACTIVE])
                    
                    logger.info(f"Successfully merged branch {source_branch_id} into {target_thread_id}")
                    
                    return MergeResult(
                        success=True,
                        merged_thread_id=target_thread_id,
                        source_branch_id=source_branch_id,
                        target_thread_id=target_thread_id,
                        conflicts=conflicts,
                        merge_checkpoint_id=merge_checkpoint_id
                    )
                    
                except Exception as e:
                    logger.error(f"Merge operation failed: {e}")
                    return MergeResult(
                        success=False,
                        merged_thread_id="",
                        source_branch_id=source_branch_id,
                        target_thread_id=target_thread_id,
                        error_message=str(e)
                    )
    
    async def _merge_overwrite(
        self,
        source_state: PipelineGlobalState,
        target_state: PipelineGlobalState
    ) -> Tuple[PipelineGlobalState, List[MergeConflict]]:
        """Merge using overwrite strategy (source overwrites target)."""
        merged_state = source_state.copy()
        
        # Preserve critical target state fields
        merged_state["thread_id"] = target_state["thread_id"]
        merged_state["execution_id"] = target_state["execution_id"]
        
        # No conflicts in overwrite strategy
        return merged_state, []
    
    async def _merge_recursive(
        self,
        source_state: PipelineGlobalState,
        target_state: PipelineGlobalState,
        conflict_resolution: ConflictResolution
    ) -> Tuple[PipelineGlobalState, List[MergeConflict]]:
        """Merge using recursive strategy with conflict detection."""
        
        # Start with target state as base
        merged_state = target_state.copy()
        conflicts = []
        
        # Recursively merge fields
        await self._recursive_merge_dict(
            source_state,
            merged_state,
            conflicts,
            conflict_resolution,
            ""
        )
        
        return merged_state, conflicts
    
    async def _recursive_merge_dict(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
        conflicts: List[MergeConflict],
        conflict_resolution: ConflictResolution,
        field_path: str
    ):
        """Recursively merge dictionaries and track conflicts."""
        
        for key, source_value in source.items():
            current_path = f"{field_path}.{key}" if field_path else key
            
            # Skip critical system fields
            if current_path in ["thread_id", "execution_id"]:
                continue
            
            if key not in target:
                # New field from source
                target[key] = source_value
            elif target[key] == source_value:
                # No conflict - values are the same
                continue
            elif isinstance(source_value, dict) and isinstance(target[key], dict):
                # Recursively merge nested dicts
                await self._recursive_merge_dict(
                    source_value,
                    target[key],
                    conflicts,
                    conflict_resolution,
                    current_path
                )
            else:
                # Conflict detected
                conflict = MergeConflict(
                    field_path=current_path,
                    source_value=source_value,
                    target_value=target[key],
                    conflict_type="value_mismatch"
                )
                
                # Apply conflict resolution
                if conflict_resolution == ConflictResolution.FAVOR_SOURCE:
                    target[key] = source_value
                    conflict.resolution = conflict_resolution
                    conflict.resolved_value = source_value
                elif conflict_resolution == ConflictResolution.FAVOR_TARGET:
                    # Keep target value
                    conflict.resolution = conflict_resolution
                    conflict.resolved_value = target[key]
                elif conflict_resolution == ConflictResolution.MERGE_BOTH:
                    # Attempt to merge both values
                    if isinstance(source_value, list) and isinstance(target[key], list):
                        merged_list = target[key] + [item for item in source_value if item not in target[key]]
                        target[key] = merged_list
                        conflict.resolution = conflict_resolution
                        conflict.resolved_value = merged_list
                    else:
                        # Can't merge, require manual resolution
                        pass
                # REQUIRE_MANUAL - leave unresolved
                
                conflicts.append(conflict)
    
    async def _merge_cherry_pick(
        self,
        source_state: PipelineGlobalState,
        target_state: PipelineGlobalState
    ) -> Tuple[PipelineGlobalState, List[MergeConflict]]:
        """Merge using cherry-pick strategy (selective merge)."""
        # For now, implement as recursive merge
        # In a full implementation, this would allow selecting specific changes
        return await self._merge_recursive(source_state, target_state, ConflictResolution.REQUIRE_MANUAL)
    
    async def _calculate_branch_depth(self, thread_id: str) -> int:
        """Calculate the depth of a branch in the hierarchy."""
        depth = 0
        current_thread = thread_id
        
        # Look for branch info in active branches
        for branch in self.active_branches.values():
            if branch.thread_id == current_thread:
                depth = branch.metadata.get("depth", 0)
                break
        
        return depth
    
    async def _get_operation_lock(self, thread_id: str) -> asyncio.Lock:
        """Get or create operation lock for thread-safe operations."""
        if thread_id not in self._operation_locks:
            self._operation_locks[thread_id] = asyncio.Lock()
        return self._operation_locks[thread_id]
    
    def abandon_branch(
        self,
        branch_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """Mark a branch as abandoned."""
        if branch_id not in self.active_branches:
            return False
        
        branch = self.active_branches[branch_id]
        branch.status = BranchStatus.ABANDONED
        branch.completed_at = time.time()
        branch.metadata["abandonment_reason"] = reason
        
        # Update metrics
        self.metrics["abandoned_branches"] += 1
        self.metrics["active_branches"] = len([b for b in self.active_branches.values() if b.status == BranchStatus.ACTIVE])
        
        logger.info(f"Abandoned branch {branch_id}: {reason or 'No reason provided'}")
        return True
    
    def get_branch_hierarchy(self, root_thread_id: str) -> Dict[str, Any]:
        """Get the complete branch hierarchy for a thread."""
        def build_hierarchy(thread_id: str) -> Dict[str, Any]:
            node = {"thread_id": thread_id, "branches": []}
            
            if thread_id in self.branch_hierarchy:
                for branch_id in self.branch_hierarchy[thread_id]:
                    if branch_id in self.active_branches:
                        branch = self.active_branches[branch_id]
                        branch_node = {
                            "branch_id": branch_id,
                            "branch_name": branch.branch_name,
                            "status": branch.status.value,
                            "created_at": branch.created_at,
                            "thread_id": branch.thread_id,
                            "branches": build_hierarchy(branch.thread_id).get("branches", [])
                        }
                        node["branches"].append(branch_node)
            
            return node
        
        return build_hierarchy(root_thread_id)
    
    def get_active_branches(self) -> List[Dict[str, Any]]:
        """Get information about all active branches."""
        branches = []
        for branch in self.active_branches.values():
            if branch.status == BranchStatus.ACTIVE:
                branches.append({
                    "branch_id": branch.branch_id,
                    "thread_id": branch.thread_id,
                    "parent_thread_id": branch.parent_thread_id,
                    "branch_name": branch.branch_name,
                    "description": branch.description,
                    "created_at": branch.created_at,
                    "created_by": branch.created_by,
                    "depth": branch.metadata.get("depth", 0)
                })
        return branches
    
    def _start_cleanup_task(self):
        """Start background cleanup task for old branches."""
        async def cleanup_old_branches():
            while True:
                try:
                    await asyncio.sleep(3600)  # Check every hour
                    
                    if not self.auto_cleanup_abandoned_branches:
                        continue
                    
                    current_time = time.time()
                    retention_seconds = self.branch_retention_hours * 3600
                    
                    # Find old branches to clean up
                    branches_to_remove = []
                    for branch_id, branch in self.active_branches.items():
                        if branch.status in [BranchStatus.ABANDONED, BranchStatus.MERGED, BranchStatus.FAILED]:
                            if branch.completed_at and (current_time - branch.completed_at) > retention_seconds:
                                branches_to_remove.append(branch_id)
                    
                    # Remove old branches
                    for branch_id in branches_to_remove:
                        branch = self.active_branches[branch_id]
                        del self.active_branches[branch_id]
                        
                        # Remove from hierarchy
                        parent_id = branch.parent_thread_id
                        if parent_id in self.branch_hierarchy:
                            if branch_id in self.branch_hierarchy[parent_id]:
                                self.branch_hierarchy[parent_id].remove(branch_id)
                        
                        logger.info(f"Cleaned up old branch: {branch_id}")
                
                except Exception as e:
                    logger.error(f"Error in branch cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_old_branches())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get branching and merge metrics."""
        return {
            **self.metrics,
            "current_active_branches": len([b for b in self.active_branches.values() if b.status == BranchStatus.ACTIVE])
        }
    
    async def shutdown(self):
        """Shutdown branching manager and cleanup resources."""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clear tracking data
        self.active_branches.clear()
        self.branch_hierarchy.clear()
        self._operation_locks.clear()
        
        logger.info("CheckpointBranchingManager shutdown complete")