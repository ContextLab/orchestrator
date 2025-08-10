"""Global Pipeline State Schema - Issue #204

Comprehensive TypedDict-based state schema for LangGraph memory management.
Provides type safety, validation, and structured access to global pipeline context.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict, NotRequired
from datetime import datetime
from enum import Enum


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ExecutionMetadata(TypedDict):
    """Execution tracking metadata."""
    pipeline_id: str
    thread_id: str
    execution_id: str
    start_time: float
    current_step: str
    completed_steps: List[str]
    failed_steps: List[str]
    pending_steps: List[str]
    status: PipelineStatus
    total_execution_time: NotRequired[float]
    resource_usage: NotRequired[Dict[str, float]]
    user_id: NotRequired[str]
    session_id: NotRequired[str]
    parent_execution_id: NotRequired[str]
    retry_count: int
    last_checkpoint_id: NotRequired[str]
    last_checkpoint_time: NotRequired[float]
    last_updated: NotRequired[float]
    end_time: NotRequired[float]


class ErrorContext(TypedDict):
    """Error handling and debugging context."""
    current_error: NotRequired[Dict[str, Any]]
    error_history: List[Dict[str, Any]]
    retry_count: int
    retry_attempts: List[Dict[str, Any]]
    last_successful_step: NotRequired[str]
    debugging_session_id: NotRequired[str]
    recovery_attempts: NotRequired[List[Dict[str, Any]]]
    error_patterns: NotRequired[List[str]]
    failure_analysis: NotRequired[Dict[str, Any]]


class ToolExecutionResults(TypedDict):
    """Tool execution results and metadata."""
    tool_calls: Dict[str, Dict[str, Any]]
    tool_outputs: Dict[str, Any]
    tool_errors: Dict[str, str]
    execution_times: Dict[str, float]
    tool_metadata: Dict[str, Dict[str, Any]]
    security_violations: NotRequired[List[str]]
    resource_usage: NotRequired[Dict[str, Dict[str, float]]]


class ModelInteractions(TypedDict):
    """Model call tracking and analytics."""
    model_calls: List[Dict[str, Any]]
    token_usage: Dict[str, int]
    model_responses: Dict[str, Any] 
    auto_resolutions: Dict[str, Any]
    model_performance: Dict[str, Dict[str, float]]
    cost_tracking: NotRequired[Dict[str, float]]
    rate_limiting: NotRequired[Dict[str, Any]]


class DebugContext(TypedDict):
    """Debugging and development context."""
    debug_enabled: bool
    debug_level: str
    debug_session_id: NotRequired[str]
    debug_snapshots: List[Dict[str, Any]]
    debug_logs: List[str]
    breakpoints: NotRequired[List[str]]
    debug_metadata: NotRequired[Dict[str, Any]]
    trace_data: NotRequired[List[Dict[str, Any]]]


class PerformanceMetrics(TypedDict):
    """Performance monitoring and analytics."""
    cpu_usage: Dict[str, float]
    memory_usage: Dict[str, float]
    disk_usage: Dict[str, float]
    network_usage: Dict[str, float]
    step_timings: Dict[str, float]
    bottlenecks: List[str]
    optimization_suggestions: NotRequired[List[str]]
    resource_alerts: NotRequired[List[str]]


class SecurityContext(TypedDict):
    """Security and access control context."""
    user_permissions: Dict[str, List[str]]
    access_tokens: NotRequired[Dict[str, str]]
    security_policies: List[str]
    audit_trail: List[Dict[str, Any]]
    sensitive_data_locations: NotRequired[List[str]]
    encryption_status: NotRequired[Dict[str, bool]]


class PipelineGlobalState(TypedDict):
    """
    Global state schema for LangGraph-based pipeline execution.
    
    This schema provides comprehensive state management for:
    - Pipeline execution tracking
    - Tool and model interactions  
    - Error handling and debugging
    - Performance monitoring
    - Security and audit
    - Cross-session persistence
    """
    
    # Core execution data
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    intermediate_results: Dict[str, Any]
    
    # Execution tracking
    execution_metadata: ExecutionMetadata
    
    # Error handling and debugging
    error_context: ErrorContext
    debug_context: DebugContext
    
    # Tool and model interactions
    tool_results: ToolExecutionResults
    model_interactions: ModelInteractions
    
    # Performance and monitoring
    performance_metrics: PerformanceMetrics
    memory_snapshots: List[Dict[str, Any]]
    
    # Security and access control
    security_context: NotRequired[SecurityContext]
    
    # User and session context
    user_context: NotRequired[Dict[str, Any]]
    session_context: NotRequired[Dict[str, Any]]
    global_variables: Dict[str, Any]
    
    # Cross-pipeline data sharing
    shared_state: NotRequired[Dict[str, Any]]
    pipeline_dependencies: NotRequired[List[str]]
    
    # Versioning and history
    state_version: str
    checkpoint_history: List[str]
    state_diffs: NotRequired[List[Dict[str, Any]]]


def create_initial_pipeline_state(
    pipeline_id: str,
    thread_id: str, 
    execution_id: str,
    inputs: Dict[str, Any],
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> PipelineGlobalState:
    """
    Create initial pipeline state with required fields.
    
    Args:
        pipeline_id: Unique pipeline identifier
        thread_id: LangGraph thread identifier
        execution_id: Unique execution identifier
        inputs: Pipeline input data
        user_id: Optional user identifier
        session_id: Optional session identifier
        
    Returns:
        Initialized PipelineGlobalState
    """
    current_time = time.time()
    
    return PipelineGlobalState(
        # Core execution data
        inputs=inputs,
        outputs={},
        intermediate_results={},
        
        # Execution tracking
        execution_metadata=ExecutionMetadata(
            pipeline_id=pipeline_id,
            thread_id=thread_id,
            execution_id=execution_id,
            start_time=current_time,
            current_step="initialization",
            completed_steps=[],
            failed_steps=[],
            pending_steps=[],
            status=PipelineStatus.PENDING.value,
            user_id=user_id,
            session_id=session_id,
            retry_count=0
        ),
        
        # Error handling and debugging
        error_context=ErrorContext(
            error_history=[],
            retry_count=0,
            retry_attempts=[]
        ),
        debug_context=DebugContext(
            debug_enabled=False,
            debug_level="INFO",
            debug_snapshots=[],
            debug_logs=[]
        ),
        
        # Tool and model interactions
        tool_results=ToolExecutionResults(
            tool_calls={},
            tool_outputs={},
            tool_errors={},
            execution_times={},
            tool_metadata={}
        ),
        model_interactions=ModelInteractions(
            model_calls=[],
            token_usage={},
            model_responses={},
            auto_resolutions={},
            model_performance={}
        ),
        
        # Performance and monitoring
        performance_metrics=PerformanceMetrics(
            cpu_usage={},
            memory_usage={},
            disk_usage={},
            network_usage={},
            step_timings={},
            bottlenecks=[]
        ),
        memory_snapshots=[],
        
        # User and session context
        global_variables={},
        
        # Versioning and history
        state_version="1.0.0",
        checkpoint_history=[]
    )


def validate_pipeline_state(state: Dict[str, Any]) -> List[str]:
    """
    Validate pipeline state structure and required fields.
    
    Args:
        state: State dictionary to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check if it's a dictionary
    if not isinstance(state, dict):
        errors.append("State must be a dictionary")
        return errors
    
    # Check required top-level keys
    required_keys = [
        "inputs", "outputs", "intermediate_results",
        "execution_metadata", "error_context", "debug_context",
        "tool_results", "model_interactions", "performance_metrics",
        "global_variables", "state_version", "checkpoint_history"
    ]
    
    for key in required_keys:
        if key not in state:
            errors.append(f"Missing required key: {key}")
    
    # Validate execution_metadata structure
    if "execution_metadata" in state:
        metadata = state["execution_metadata"]
        required_metadata_keys = [
            "pipeline_id", "thread_id", "execution_id", "start_time",
            "current_step", "completed_steps", "failed_steps", 
            "pending_steps", "status", "retry_count"
        ]
        
        for key in required_metadata_keys:
            if key not in metadata:
                errors.append(f"Missing required execution_metadata key: {key}")
    
    # Validate error_context structure
    if "error_context" in state:
        error_ctx = state["error_context"]
        required_error_keys = ["error_history", "retry_count", "retry_attempts"]
        
        for key in required_error_keys:
            if key not in error_ctx:
                errors.append(f"Missing required error_context key: {key}")
    
    # Validate lists are actually lists
    list_fields = {
        "execution_metadata.completed_steps": ["execution_metadata", "completed_steps"],
        "execution_metadata.failed_steps": ["execution_metadata", "failed_steps"],
        "execution_metadata.pending_steps": ["execution_metadata", "pending_steps"],
        "error_context.error_history": ["error_context", "error_history"],
        "error_context.retry_attempts": ["error_context", "retry_attempts"],
        "debug_context.debug_snapshots": ["debug_context", "debug_snapshots"],
        "debug_context.debug_logs": ["debug_context", "debug_logs"],
        "checkpoint_history": ["checkpoint_history"]
    }
    
    for field_path, keys in list_fields.items():
        current = state
        for key in keys[:-1]:
            if key in current:
                current = current[key]
            else:
                current = None
                break
        
        if current is not None:
            final_key = keys[-1]
            if final_key in current and not isinstance(current[final_key], list):
                errors.append(f"{field_path} must be a list")
    
    return errors


def merge_pipeline_states(
    current_state: PipelineGlobalState,
    updates: Dict[str, Any]
) -> PipelineGlobalState:
    """
    Merge state updates with current state, handling nested structures.
    
    Args:
        current_state: Current pipeline state
        updates: Updates to merge
        
    Returns:
        Merged state
    """
    import copy
    
    # Deep copy current state to avoid mutation
    merged_state = copy.deepcopy(current_state)
    
    def merge_recursive(target, updates):
        """Recursively merge nested dictionaries and lists."""
        for key, value in updates.items():
            if key in target:
                if isinstance(target[key], dict) and isinstance(value, dict):
                    # Recursively merge nested dictionaries
                    merge_recursive(target[key], value)
                elif isinstance(target[key], list) and isinstance(value, list):
                    # Extend lists (avoid duplicates for specific fields)
                    if key == "completed_steps":
                        # For completed_steps, extend and deduplicate
                        target[key].extend([item for item in value if item not in target[key]])
                    else:
                        target[key].extend(value)
                else:
                    # Direct replacement
                    target[key] = value
            else:
                # New key
                target[key] = value
    
    merge_recursive(merged_state, updates)
    
    return merged_state


__all__ = [
    "PipelineGlobalState",
    "ExecutionMetadata", 
    "ErrorContext",
    "ToolExecutionResults",
    "ModelInteractions",
    "DebugContext",
    "PerformanceMetrics",
    "SecurityContext",
    "PipelineStatus",
    "create_initial_pipeline_state",
    "validate_pipeline_state", 
    "merge_pipeline_states"
]