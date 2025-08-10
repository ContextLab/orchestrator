"""Human-in-the-Loop System - Issue #205 Phase 2

Manages human intervention during pipeline execution with state inspection,
modification, and interactive debugging capabilities.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum

# Internal imports
from ..state.global_context import (
    PipelineGlobalState,
    validate_pipeline_state,
    PipelineStatus
)
from ..state.langgraph_state_manager import LangGraphGlobalContextManager
from ..core.exceptions import PipelineExecutionError

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of human interactions."""
    INSPECTION = "inspection"
    MODIFICATION = "modification"
    APPROVAL = "approval"
    DEBUG = "debug"
    DECISION = "decision"


class SessionStatus(Enum):
    """Status of human interaction sessions."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class StateModification:
    """Represents a state modification made by human."""
    field_path: str  # e.g., "intermediate_results.step_1.output"
    old_value: Any
    new_value: Any
    modification_type: str  # "update", "delete", "insert"
    timestamp: float
    user_id: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class InspectionSession:
    """Represents a human inspection session."""
    session_id: str
    thread_id: str
    checkpoint_id: str
    current_step: str
    interaction_type: InteractionType
    status: SessionStatus
    created_at: float
    user_id: Optional[str] = None
    modifications: List[StateModification] = None
    approval_required: bool = False
    timeout_seconds: Optional[float] = None
    
    def __post_init__(self):
        if self.modifications is None:
            self.modifications = []


@dataclass
class ApprovalRequest:
    """Represents an approval workflow request."""
    request_id: str
    session_id: str
    thread_id: str
    step_id: str
    approval_type: str  # "execute", "modify", "skip", "abort"
    description: str
    required_approvers: List[str]
    current_approvals: List[str] = None
    created_at: float = None
    expires_at: Optional[float] = None
    
    def __post_init__(self):
        if self.current_approvals is None:
            self.current_approvals = []
        if self.created_at is None:
            self.created_at = time.time()


class HumanInteractionManager:
    """
    Manages human intervention during pipeline execution.
    
    Provides capabilities for:
    - Runtime state inspection and modification
    - Interactive debugging sessions
    - Approval workflows for sensitive operations
    - Manual checkpoint creation at decision points
    """
    
    def __init__(
        self,
        langgraph_manager: LangGraphGlobalContextManager,
        default_session_timeout: float = 3600.0,  # 1 hour
        max_concurrent_sessions: int = 10,
        enable_approval_workflows: bool = True,
        auto_checkpoint_on_modification: bool = True,
    ):
        """
        Initialize human interaction manager.
        
        Args:
            langgraph_manager: LangGraph state manager for persistence
            default_session_timeout: Default timeout for interaction sessions
            max_concurrent_sessions: Maximum concurrent interaction sessions
            enable_approval_workflows: Enable approval workflow features
            auto_checkpoint_on_modification: Automatically create checkpoints on modifications
        """
        self.langgraph_manager = langgraph_manager
        self.default_session_timeout = default_session_timeout
        self.max_concurrent_sessions = max_concurrent_sessions
        self.enable_approval_workflows = enable_approval_workflows
        self.auto_checkpoint_on_modification = auto_checkpoint_on_modification
        
        # Active sessions tracking
        self.active_sessions: Dict[str, InspectionSession] = {}
        self.pending_approvals: Dict[str, ApprovalRequest] = {}
        
        # Session management
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = {
            "total_sessions": 0,
            "active_sessions": 0,
            "completed_sessions": 0,
            "expired_sessions": 0,
            "total_modifications": 0,
            "approval_requests": 0,
            "approved_requests": 0,
            "rejected_requests": 0
        }
        
        # Start cleanup task
        self._start_cleanup_task()
        
        logger.info("HumanInteractionManager initialized")
    
    async def pause_for_inspection(
        self,
        thread_id: str,
        step_id: str,
        interaction_type: InteractionType = InteractionType.INSPECTION,
        user_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        approval_required: bool = False,
        description: Optional[str] = None
    ) -> InspectionSession:
        """
        Pause pipeline execution for human inspection.
        
        Args:
            thread_id: Thread identifier for the execution
            step_id: Current step being inspected
            interaction_type: Type of interaction required
            user_id: Optional user identifier
            timeout_seconds: Optional timeout override
            approval_required: Whether approval is required to continue
            description: Optional description of the inspection
            
        Returns:
            Inspection session object
            
        Raises:
            RuntimeError: If max concurrent sessions exceeded
        """
        # Check concurrent session limit
        if len(self.active_sessions) >= self.max_concurrent_sessions:
            raise RuntimeError(f"Maximum concurrent sessions ({self.max_concurrent_sessions}) exceeded")
        
        # Create session
        session_id = f"inspection_{uuid.uuid4().hex[:8]}"
        timeout = timeout_seconds or self.default_session_timeout
        
        # Create inspection checkpoint
        checkpoint_description = f"Human inspection: {interaction_type.value}"
        if description:
            checkpoint_description += f" - {description}"
            
        checkpoint_id = await self.langgraph_manager.create_checkpoint(
            thread_id=thread_id,
            description=checkpoint_description,
            metadata={
                "interaction_type": interaction_type.value,
                "session_id": session_id,
                "user_id": user_id,
                "step_id": step_id,
                "approval_required": approval_required
            }
        )
        
        # Create session
        session = InspectionSession(
            session_id=session_id,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            current_step=step_id,
            interaction_type=interaction_type,
            status=SessionStatus.ACTIVE,
            created_at=time.time(),
            user_id=user_id,
            approval_required=approval_required,
            timeout_seconds=timeout
        )
        
        # Track session
        self.active_sessions[session_id] = session
        self._session_locks[session_id] = asyncio.Lock()
        
        # Update metrics
        self.metrics["total_sessions"] += 1
        self.metrics["active_sessions"] += 1
        
        logger.info(f"Created inspection session {session_id} for thread {thread_id} at step {step_id}")
        
        return session
    
    async def get_current_state(self, session_id: str) -> Optional[PipelineGlobalState]:
        """Get current pipeline state for an inspection session."""
        if session_id not in self.active_sessions:
            return None
            
        session = self.active_sessions[session_id]
        return await self.langgraph_manager.get_global_state(session.thread_id)
    
    async def inspect_state_field(
        self,
        session_id: str,
        field_path: str
    ) -> Optional[Any]:
        """
        Inspect a specific field in the pipeline state.
        
        Args:
            session_id: Session identifier
            field_path: Dot-separated path to field (e.g., "intermediate_results.step_1.output")
            
        Returns:
            Field value or None if not found
        """
        state = await self.get_current_state(session_id)
        if not state:
            return None
        
        # Navigate to field using path
        try:
            current = state
            for part in field_path.split('.'):
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
            return current
        except Exception as e:
            logger.error(f"Error inspecting field {field_path}: {e}")
            return None
    
    async def modify_state(
        self,
        session_id: str,
        field_path: str,
        new_value: Any,
        modification_type: str = "update",
        user_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> bool:
        """
        Modify pipeline state during inspection session.
        
        Args:
            session_id: Session identifier
            field_path: Dot-separated path to field
            new_value: New value to set
            modification_type: Type of modification ("update", "delete", "insert")
            user_id: User making the modification
            reason: Optional reason for the modification
            
        Returns:
            True if modification was successful
        """
        if session_id not in self.active_sessions:
            logger.error(f"Session {session_id} not found")
            return False
        
        session = self.active_sessions[session_id]
        
        async with self._session_locks[session_id]:
            # Get current state
            current_state = await self.get_current_state(session_id)
            if not current_state:
                logger.error(f"Could not get state for session {session_id}")
                return False
            
            # Get current value for change tracking
            old_value = await self.inspect_state_field(session_id, field_path)
            
            # Validate modification
            if not await self._validate_state_modification(field_path, new_value, current_state):
                logger.error(f"Invalid state modification: {field_path} = {new_value}")
                return False
            
            try:
                # Apply modification
                modified_state = current_state.copy()
                if await self._apply_field_modification(modified_state, field_path, new_value, modification_type):
                    
                    # Validate modified state
                    validation_errors = validate_pipeline_state(modified_state)
                    if validation_errors:
                        logger.error(f"State validation failed after modification: {validation_errors}")
                        return False
                    
                    # Update state in LangGraph
                    await self.langgraph_manager.update_global_state(
                        session.thread_id,
                        modified_state
                    )
                    
                    # Track modification
                    modification = StateModification(
                        field_path=field_path,
                        old_value=old_value,
                        new_value=new_value,
                        modification_type=modification_type,
                        timestamp=time.time(),
                        user_id=user_id,
                        reason=reason
                    )
                    session.modifications.append(modification)
                    
                    # Create checkpoint if enabled
                    if self.auto_checkpoint_on_modification:
                        await self.langgraph_manager.create_checkpoint(
                            thread_id=session.thread_id,
                            description=f"Human modification: {field_path}",
                            metadata={
                                "modification": {
                                    "field_path": field_path,
                                    "modification_type": modification_type,
                                    "user_id": user_id,
                                    "reason": reason
                                },
                                "session_id": session_id
                            }
                        )
                    
                    # Update metrics
                    self.metrics["total_modifications"] += 1
                    
                    logger.info(f"Successfully modified {field_path} in session {session_id}")
                    return True
                
            except Exception as e:
                logger.error(f"Error applying modification {field_path}: {e}")
                return False
        
        return False
    
    async def _apply_field_modification(
        self,
        state: PipelineGlobalState,
        field_path: str,
        new_value: Any,
        modification_type: str
    ) -> bool:
        """Apply field modification to state."""
        try:
            parts = field_path.split('.')
            current = state
            
            # Navigate to parent of target field
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            target_field = parts[-1]
            
            if modification_type == "update":
                current[target_field] = new_value
            elif modification_type == "delete":
                if target_field in current:
                    del current[target_field]
            elif modification_type == "insert":
                if isinstance(current.get(target_field), list):
                    current[target_field].append(new_value)
                else:
                    current[target_field] = new_value
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying field modification: {e}")
            return False
    
    async def _validate_state_modification(
        self,
        field_path: str,
        new_value: Any,
        current_state: PipelineGlobalState
    ) -> bool:
        """Validate that a state modification is safe and valid."""
        
        # Prevent modification of critical system fields
        protected_fields = [
            "thread_id",
            "execution_id", 
            "pipeline_id",
            "execution_metadata.start_time",
            "execution_metadata.pipeline_id"
        ]
        
        for protected in protected_fields:
            if field_path.startswith(protected):
                logger.warning(f"Attempted modification of protected field: {field_path}")
                return False
        
        # Validate value types for known fields
        if field_path.startswith("execution_metadata.status"):
            if not isinstance(new_value, (str, PipelineStatus)):
                return False
        
        # Additional validation can be added here
        return True
    
    async def request_approval(
        self,
        session_id: str,
        approval_type: str,
        description: str,
        required_approvers: List[str],
        expires_in_seconds: Optional[float] = None
    ) -> ApprovalRequest:
        """
        Create an approval request for a human interaction session.
        
        Args:
            session_id: Session requiring approval
            approval_type: Type of approval ("execute", "modify", "skip", "abort")
            description: Description of what needs approval
            required_approvers: List of user IDs who can approve
            expires_in_seconds: Optional expiration time
            
        Returns:
            Approval request object
        """
        if not self.enable_approval_workflows:
            raise RuntimeError("Approval workflows are disabled")
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        request_id = f"approval_{uuid.uuid4().hex[:8]}"
        
        expires_at = None
        if expires_in_seconds:
            expires_at = time.time() + expires_in_seconds
        
        approval_request = ApprovalRequest(
            request_id=request_id,
            session_id=session_id,
            thread_id=session.thread_id,
            step_id=session.current_step,
            approval_type=approval_type,
            description=description,
            required_approvers=required_approvers,
            expires_at=expires_at
        )
        
        self.pending_approvals[request_id] = approval_request
        
        # Update metrics
        self.metrics["approval_requests"] += 1
        
        logger.info(f"Created approval request {request_id} for session {session_id}")
        
        return approval_request
    
    async def provide_approval(
        self,
        request_id: str,
        approver_id: str,
        approved: bool,
        comments: Optional[str] = None
    ) -> bool:
        """
        Provide approval for a pending request.
        
        Args:
            request_id: Approval request ID
            approver_id: ID of the approver
            approved: Whether the request is approved
            comments: Optional approval comments
            
        Returns:
            True if approval was processed successfully
        """
        if request_id not in self.pending_approvals:
            logger.error(f"Approval request {request_id} not found")
            return False
        
        request = self.pending_approvals[request_id]
        
        # Check if approver is authorized
        if approver_id not in request.required_approvers:
            logger.error(f"User {approver_id} not authorized to approve request {request_id}")
            return False
        
        # Check if request has expired
        if request.expires_at and time.time() > request.expires_at:
            logger.error(f"Approval request {request_id} has expired")
            return False
        
        # Record approval
        if approved:
            if approver_id not in request.current_approvals:
                request.current_approvals.append(approver_id)
            self.metrics["approved_requests"] += 1
        else:
            self.metrics["rejected_requests"] += 1
            # Remove request on rejection
            del self.pending_approvals[request_id]
            logger.info(f"Approval request {request_id} rejected by {approver_id}")
            return True
        
        # Check if all approvals received
        if set(request.current_approvals) == set(request.required_approvers):
            # All approvals received, remove from pending
            del self.pending_approvals[request_id]
            logger.info(f"Approval request {request_id} fully approved")
            return True
        
        logger.info(f"Partial approval received for {request_id}: {len(request.current_approvals)}/{len(request.required_approvers)}")
        return True
    
    async def resume_execution(
        self,
        session_id: str,
        modifications_applied: bool = False
    ) -> str:
        """
        Resume pipeline execution after human interaction.
        
        Args:
            session_id: Session to resume from
            modifications_applied: Whether state modifications were applied
            
        Returns:
            Thread ID for continued execution
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Mark session as completed
        session.status = SessionStatus.COMPLETED
        
        # Create resume checkpoint
        resume_description = f"Resumed from human interaction: {session.interaction_type.value}"
        if modifications_applied:
            resume_description += f" (with {len(session.modifications)} modifications)"
        
        resume_checkpoint_id = await self.langgraph_manager.create_checkpoint(
            thread_id=session.thread_id,
            description=resume_description,
            metadata={
                "session_id": session_id,
                "modifications_count": len(session.modifications),
                "resume_timestamp": time.time()
            }
        )
        
        # Clean up session
        await self._cleanup_session(session_id)
        
        logger.info(f"Resumed execution for thread {session.thread_id} from session {session_id}")
        
        return session.thread_id
    
    async def cancel_session(self, session_id: str, reason: Optional[str] = None) -> bool:
        """Cancel an active inspection session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.status = SessionStatus.CANCELLED
        
        # Clean up session
        await self._cleanup_session(session_id)
        
        logger.info(f"Cancelled session {session_id}: {reason or 'No reason provided'}")
        return True
    
    async def _cleanup_session(self, session_id: str):
        """Clean up a completed or cancelled session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        if session_id in self._session_locks:
            del self._session_locks[session_id]
        
        # Update metrics
        self.metrics["active_sessions"] = len(self.active_sessions)
        self.metrics["completed_sessions"] += 1
    
    def _start_cleanup_task(self):
        """Start background cleanup task for expired sessions."""
        async def cleanup_expired_sessions():
            while True:
                try:
                    await asyncio.sleep(60)  # Check every minute
                    
                    current_time = time.time()
                    expired_sessions = []
                    
                    # Find expired sessions
                    for session_id, session in self.active_sessions.items():
                        if session.timeout_seconds:
                            expires_at = session.created_at + session.timeout_seconds
                            if current_time > expires_at:
                                expired_sessions.append(session_id)
                    
                    # Clean up expired sessions
                    for session_id in expired_sessions:
                        self.active_sessions[session_id].status = SessionStatus.EXPIRED
                        await self._cleanup_session(session_id)
                        self.metrics["expired_sessions"] += 1
                        logger.info(f"Cleaned up expired session: {session_id}")
                    
                    # Clean up expired approval requests
                    expired_approvals = []
                    for request_id, request in self.pending_approvals.items():
                        if request.expires_at and current_time > request.expires_at:
                            expired_approvals.append(request_id)
                    
                    for request_id in expired_approvals:
                        del self.pending_approvals[request_id]
                        logger.info(f"Cleaned up expired approval request: {request_id}")
                
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_expired_sessions())
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get information about all active sessions."""
        sessions = []
        for session in self.active_sessions.values():
            sessions.append({
                "session_id": session.session_id,
                "thread_id": session.thread_id,
                "current_step": session.current_step,
                "interaction_type": session.interaction_type.value,
                "status": session.status.value,
                "created_at": session.created_at,
                "user_id": session.user_id,
                "modifications_count": len(session.modifications),
                "approval_required": session.approval_required
            })
        return sessions
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get information about all pending approval requests."""
        approvals = []
        for request in self.pending_approvals.values():
            approvals.append({
                "request_id": request.request_id,
                "session_id": request.session_id,
                "approval_type": request.approval_type,
                "description": request.description,
                "required_approvers": request.required_approvers,
                "current_approvals": request.current_approvals,
                "created_at": request.created_at,
                "expires_at": request.expires_at
            })
        return approvals
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get human interaction metrics."""
        return {
            **self.metrics,
            "pending_approvals": len(self.pending_approvals),
            "current_active_sessions": len(self.active_sessions)
        }
    
    async def shutdown(self):
        """Shutdown human interaction manager and cleanup resources."""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.cancel_session(session_id, "System shutdown")
        
        # Clear pending approvals
        self.pending_approvals.clear()
        
        logger.info("HumanInteractionManager shutdown complete")