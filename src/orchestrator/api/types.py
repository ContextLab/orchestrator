"""
Comprehensive Type Definitions and API Documentation for the Orchestrator Framework.

This module provides complete type definitions for all API components, including:
- Request and response types for all API operations
- Status and progress tracking types
- Configuration and validation types
- Integration types for external systems
- Complete API documentation with examples

The type system ensures type safety, provides clear API contracts, and enables
comprehensive validation and error handling throughout the framework.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import (
    Any, Dict, List, Optional, Union, Protocol, TypeVar, Generic,
    Callable, Awaitable, AsyncIterator, Literal, TypedDict, get_type_hints
)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod

# Import foundation types
from ..execution import (
    ExecutionStatus,
    ExecutionMetrics,
    ProgressEvent,
    ProgressEventType,
    StepStatus,
    VariableScope,
    VariableType,
    RecoveryStrategy,
    ErrorSeverity,
)
from ..core.pipeline import Pipeline
from ..core.task import Task

# Generic type variables
T = TypeVar('T')
P = TypeVar('P', bound=Pipeline)
R = TypeVar('R')


# API Operation Types

class APIOperation(Enum):
    """Supported API operations."""
    COMPILE_PIPELINE = "compile_pipeline"
    EXECUTE_PIPELINE = "execute_pipeline"
    VALIDATE_YAML = "validate_yaml"
    GET_STATUS = "get_execution_status"
    STOP_EXECUTION = "stop_execution"
    LIST_EXECUTIONS = "list_active_executions"
    CLEANUP_EXECUTION = "cleanup_execution"
    GET_COMPILATION_REPORT = "get_compilation_report"
    EXTRACT_VARIABLES = "get_template_variables"
    MONITOR_EXECUTION = "monitor_execution"
    CONTROL_EXECUTION = "control_execution"


class ValidationLevel(Enum):
    """Pipeline validation levels."""
    STRICT = "strict"           # Full validation with strict rules
    PERMISSIVE = "permissive"   # Relaxed validation allowing some issues
    DEVELOPMENT = "development" # Development mode with warnings only
    DISABLED = "disabled"       # Disable validation (not recommended)


class CompilationMode(Enum):
    """Pipeline compilation modes."""
    STANDARD = "standard"       # Standard compilation
    FAST = "fast"              # Skip non-essential validation
    SAFE = "safe"              # Extra safety checks
    DEBUG = "debug"            # Include debug information


class ExecutionMode(Enum):
    """Pipeline execution modes."""
    NORMAL = "normal"          # Standard execution
    DRY_RUN = "dry_run"       # Validate without executing
    STEP_BY_STEP = "step_by_step"  # Interactive step execution
    PARALLEL = "parallel"      # Parallel step execution where possible
    RECOVERY = "recovery"      # Recovery mode execution


# Request Types

@dataclass
class CompilationRequest:
    """Request for pipeline compilation."""
    # Required fields
    yaml_content: Union[str, Path]
    
    # Optional compilation parameters
    context: Optional[Dict[str, Any]] = None
    resolve_ambiguities: bool = True
    validate: bool = True
    validation_level: ValidationLevel = ValidationLevel.STRICT
    compilation_mode: CompilationMode = CompilationMode.STANDARD
    
    # Template processing options
    enable_preprocessing: bool = True
    template_strict_mode: bool = True
    
    # Advanced options
    cache_result: bool = True
    include_metadata: bool = False
    debug_mode: bool = False
    
    # Request metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "yaml_content": str(self.yaml_content) if isinstance(self.yaml_content, Path) else self.yaml_content,
            "context": self.context,
            "resolve_ambiguities": self.resolve_ambiguities,
            "validate": self.validate,
            "validation_level": self.validation_level.value,
            "compilation_mode": self.compilation_mode.value,
            "enable_preprocessing": self.enable_preprocessing,
            "template_strict_mode": self.template_strict_mode,
            "cache_result": self.cache_result,
            "include_metadata": self.include_metadata,
            "debug_mode": self.debug_mode,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id
        }


@dataclass
class ExecutionRequest:
    """Request for pipeline execution."""
    # Required fields
    pipeline: Union[Pipeline, str, Path]
    
    # Optional execution parameters
    context: Optional[Dict[str, Any]] = None
    execution_id: Optional[str] = None
    execution_mode: ExecutionMode = ExecutionMode.NORMAL
    
    # Execution control options
    timeout: Optional[int] = None  # seconds
    max_retries: int = 3
    enable_recovery: bool = True
    enable_checkpointing: bool = True
    
    # Monitoring options
    enable_monitoring: bool = True
    progress_callback: Optional[Callable] = None
    status_callback: Optional[Callable] = None
    
    # Advanced options
    resource_limits: Optional[Dict[str, Any]] = None
    environment_vars: Optional[Dict[str, str]] = None
    debug_mode: bool = False
    
    # Request metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "pipeline": self.pipeline.id if isinstance(self.pipeline, Pipeline) else str(self.pipeline),
            "context": self.context,
            "execution_id": self.execution_id,
            "execution_mode": self.execution_mode.value,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "enable_recovery": self.enable_recovery,
            "enable_checkpointing": self.enable_checkpointing,
            "enable_monitoring": self.enable_monitoring,
            "resource_limits": self.resource_limits,
            "environment_vars": self.environment_vars,
            "debug_mode": self.debug_mode,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id
        }


# Response Types

@dataclass
class APIResponse(Generic[T]):
    """Base response type for API operations."""
    # Response data
    success: bool
    data: Optional[T] = None
    
    # Response metadata
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[float] = None
    
    # Error information (if success = False)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Additional context
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        result = {
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
        }
        
        if self.data is not None:
            if hasattr(self.data, 'to_dict'):
                result["data"] = self.data.to_dict()
            elif hasattr(self.data, '__dict__'):
                result["data"] = self.data.__dict__
            else:
                result["data"] = self.data
        
        # Add optional fields
        for key, value in [
            ("request_id", self.request_id),
            ("duration_ms", self.duration_ms),
            ("error_code", self.error_code),
            ("error_message", self.error_message),
            ("error_details", self.error_details),
        ]:
            if value is not None:
                result[key] = value
        
        if self.warnings:
            result["warnings"] = self.warnings
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result


@dataclass
class CompilationResult:
    """Result of pipeline compilation."""
    pipeline: Pipeline
    compilation_time: timedelta
    validation_passed: bool = True
    
    # Compilation details
    template_variables: List[str] = field(default_factory=list)
    resolved_ambiguities: List[str] = field(default_factory=list)
    compilation_warnings: List[str] = field(default_factory=list)
    
    # Validation details
    validation_report: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    
    # Metadata
    compilation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    compiled_at: datetime = field(default_factory=datetime.now)
    compiler_version: str = "2.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "pipeline_id": self.pipeline.id,
            "pipeline_name": getattr(self.pipeline, 'name', None),
            "compilation_time_ms": self.compilation_time.total_seconds() * 1000,
            "validation_passed": self.validation_passed,
            "template_variables": self.template_variables,
            "resolved_ambiguities": self.resolved_ambiguities,
            "compilation_warnings": self.compilation_warnings,
            "validation_report": self.validation_report,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "compilation_id": self.compilation_id,
            "compiled_at": self.compiled_at.isoformat(),
            "compiler_version": self.compiler_version
        }


@dataclass
class ExecutionResult:
    """Result of pipeline execution initialization."""
    execution_id: str
    pipeline_id: str
    status: ExecutionStatus
    
    # Execution details
    started_at: datetime
    estimated_duration: Optional[timedelta] = None
    total_steps: int = 0
    
    # Monitoring details
    monitoring_enabled: bool = True
    progress_url: Optional[str] = None
    status_url: Optional[str] = None
    
    # Control details
    stop_url: Optional[str] = None
    control_enabled: bool = True
    
    # Metadata
    execution_mode: ExecutionMode = ExecutionMode.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "execution_id": self.execution_id,
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "estimated_duration_seconds": self.estimated_duration.total_seconds() if self.estimated_duration else None,
            "total_steps": self.total_steps,
            "monitoring_enabled": self.monitoring_enabled,
            "progress_url": self.progress_url,
            "status_url": self.status_url,
            "stop_url": self.stop_url,
            "control_enabled": self.control_enabled,
            "execution_mode": self.execution_mode.value,
            "metadata": self.metadata
        }


# Status and Progress Types

@dataclass
class ExecutionStatusInfo:
    """Comprehensive execution status information."""
    # Basic status
    execution_id: str
    pipeline_id: str
    status: ExecutionStatus
    
    # Timing information
    started_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[timedelta] = None
    
    # Progress information
    current_step: Optional[str] = None
    current_step_name: Optional[str] = None
    steps_completed: int = 0
    steps_total: int = 0
    progress_percentage: float = 0.0
    
    # Detailed metrics
    metrics: Optional[ExecutionMetrics] = None
    
    # Step details
    step_statuses: Dict[str, StepStatus] = field(default_factory=dict)
    step_progress: Dict[str, float] = field(default_factory=dict)
    step_messages: Dict[str, List[str]] = field(default_factory=dict)
    
    # Error information
    error_count: int = 0
    warning_count: int = 0
    last_error: Optional[str] = None
    error_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recovery information
    recovery_attempts: int = 0
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_in_progress: bool = False
    
    # Context information
    variables: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary."""
        return {
            "execution_id": self.execution_id,
            "pipeline_id": self.pipeline_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
            "current_step": self.current_step,
            "current_step_name": self.current_step_name,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "progress_percentage": self.progress_percentage,
            "metrics": self.metrics.__dict__ if self.metrics else None,
            "step_statuses": {k: v.value for k, v in self.step_statuses.items()},
            "step_progress": self.step_progress,
            "step_messages": self.step_messages,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "last_error": self.last_error,
            "error_details": self.error_details,
            "recovery_attempts": self.recovery_attempts,
            "recovery_strategy": self.recovery_strategy.value if self.recovery_strategy else None,
            "recovery_in_progress": self.recovery_in_progress,
            "variables": self.variables,
            "environment": self.environment,
            "resource_usage": self.resource_usage
        }


@dataclass
class ProgressUpdate:
    """Progress update information."""
    execution_id: str
    timestamp: datetime
    
    # Progress details
    event_type: ProgressEventType
    step_id: Optional[str] = None
    step_name: Optional[str] = None
    message: Optional[str] = None
    
    # Progress metrics
    step_progress: float = 0.0
    overall_progress: float = 0.0
    steps_completed: int = 0
    steps_total: int = 0
    
    # Additional data
    data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert progress update to dictionary."""
        return {
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "step_id": self.step_id,
            "step_name": self.step_name,
            "message": self.message,
            "step_progress": self.step_progress,
            "overall_progress": self.overall_progress,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "data": self.data,
            "metadata": self.metadata
        }


# Configuration Types

@dataclass
class APIConfiguration:
    """Configuration for the API interface."""
    # Model registry configuration
    model_registry_config: Optional[Dict[str, Any]] = None
    auto_model_selection: bool = True
    
    # Validation configuration  
    default_validation_level: ValidationLevel = ValidationLevel.STRICT
    enable_validation_caching: bool = True
    validation_timeout: int = 30  # seconds
    
    # Execution configuration
    default_execution_timeout: int = 3600  # seconds
    max_concurrent_executions: int = 10
    enable_execution_recovery: bool = True
    enable_execution_checkpointing: bool = True
    
    # Performance configuration
    enable_compilation_caching: bool = True
    cache_size_limit: int = 1000  # number of entries
    memory_limit_mb: Optional[int] = None
    
    # Monitoring configuration
    enable_detailed_monitoring: bool = True
    progress_update_interval: int = 5  # seconds
    status_cleanup_interval: int = 3600  # seconds
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "structured"
    enable_audit_logging: bool = True
    
    # Security configuration
    enable_authentication: bool = False
    api_key_required: bool = False
    rate_limiting: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_registry_config": self.model_registry_config,
            "auto_model_selection": self.auto_model_selection,
            "default_validation_level": self.default_validation_level.value,
            "enable_validation_caching": self.enable_validation_caching,
            "validation_timeout": self.validation_timeout,
            "default_execution_timeout": self.default_execution_timeout,
            "max_concurrent_executions": self.max_concurrent_executions,
            "enable_execution_recovery": self.enable_execution_recovery,
            "enable_execution_checkpointing": self.enable_execution_checkpointing,
            "enable_compilation_caching": self.enable_compilation_caching,
            "cache_size_limit": self.cache_size_limit,
            "memory_limit_mb": self.memory_limit_mb,
            "enable_detailed_monitoring": self.enable_detailed_monitoring,
            "progress_update_interval": self.progress_update_interval,
            "status_cleanup_interval": self.status_cleanup_interval,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "enable_audit_logging": self.enable_audit_logging,
            "enable_authentication": self.enable_authentication,
            "api_key_required": self.api_key_required,
            "rate_limiting": self.rate_limiting
        }


# Protocol Types for Integration

class PipelineCompilerProtocol(Protocol):
    """Protocol for pipeline compiler implementations."""
    
    async def compile(
        self,
        yaml_content: str,
        context: Optional[Dict[str, Any]] = None,
        resolve_ambiguities: bool = True
    ) -> Pipeline:
        """Compile YAML content to pipeline."""
        ...
    
    def validate_yaml(self, yaml_content: str) -> bool:
        """Validate YAML content."""
        ...
    
    def get_template_variables(self, yaml_content: str) -> List[str]:
        """Extract template variables from YAML."""
        ...


class ExecutionManagerProtocol(Protocol):
    """Protocol for execution manager implementations."""
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get execution status information."""
        ...
    
    def start_execution(self, total_steps: int) -> None:
        """Start execution tracking."""
        ...
    
    def complete_execution(self, success: bool) -> None:
        """Complete execution tracking."""
        ...
    
    def cleanup(self) -> None:
        """Cleanup execution resources."""
        ...


class ProgressMonitorProtocol(Protocol):
    """Protocol for progress monitoring implementations."""
    
    def start_monitoring(self, execution_id: str) -> None:
        """Start monitoring execution progress."""
        ...
    
    async def get_progress_updates(self, execution_id: str) -> AsyncIterator[ProgressUpdate]:
        """Get progress update stream."""
        ...
    
    def stop_monitoring(self, execution_id: str) -> None:
        """Stop monitoring execution progress."""
        ...


# Callback Types

ProgressCallback = Callable[[ProgressUpdate], None]
StatusCallback = Callable[[ExecutionStatusInfo], None]
ErrorCallback = Callable[[Exception], None]
CompletionCallback = Callable[[ExecutionResult], None]

# Async variants
AsyncProgressCallback = Callable[[ProgressUpdate], Awaitable[None]]
AsyncStatusCallback = Callable[[ExecutionStatusInfo], Awaitable[None]]
AsyncErrorCallback = Callable[[Exception], Awaitable[None]]
AsyncCompletionCallback = Callable[[ExecutionResult], Awaitable[None]]


# Typed Dictionary Types for JSON API

class PipelineCompilationDict(TypedDict, total=False):
    """Type definition for pipeline compilation JSON requests."""
    yaml_content: Union[str, Path]
    context: Optional[Dict[str, Any]]
    resolve_ambiguities: bool
    validate: bool
    validation_level: str
    compilation_mode: str
    enable_preprocessing: bool
    template_strict_mode: bool
    cache_result: bool
    include_metadata: bool
    debug_mode: bool
    request_id: str
    user_id: Optional[str]


class PipelineExecutionDict(TypedDict, total=False):
    """Type definition for pipeline execution JSON requests."""
    pipeline: Union[str, Path]  # Pipeline ID or YAML content/path
    context: Optional[Dict[str, Any]]
    execution_id: Optional[str]
    execution_mode: str
    timeout: Optional[int]
    max_retries: int
    enable_recovery: bool
    enable_checkpointing: bool
    enable_monitoring: bool
    resource_limits: Optional[Dict[str, Any]]
    environment_vars: Optional[Dict[str, str]]
    debug_mode: bool
    request_id: str
    user_id: Optional[str]


class ExecutionStatusDict(TypedDict, total=False):
    """Type definition for execution status JSON responses."""
    execution_id: str
    pipeline_id: str
    status: str
    started_at: str
    updated_at: str
    completed_at: Optional[str]
    duration_seconds: Optional[float]
    current_step: Optional[str]
    current_step_name: Optional[str]
    steps_completed: int
    steps_total: int
    progress_percentage: float
    metrics: Optional[Dict[str, Any]]
    step_statuses: Dict[str, str]
    step_progress: Dict[str, float]
    step_messages: Dict[str, List[str]]
    error_count: int
    warning_count: int
    last_error: Optional[str]
    error_details: List[Dict[str, Any]]
    recovery_attempts: int
    recovery_strategy: Optional[str]
    recovery_in_progress: bool
    variables: Dict[str, Any]
    environment: Dict[str, str]
    resource_usage: Dict[str, Any]


# Validation and Utility Types

class ValidationResult(TypedDict):
    """Result of validation operations."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]


class ResourceUsage(TypedDict):
    """Resource usage information."""
    memory_mb: float
    cpu_percent: float
    disk_mb: float
    network_kb: float
    execution_time_seconds: float


class StepSummary(TypedDict):
    """Summary information for a pipeline step."""
    step_id: str
    step_name: str
    step_type: str
    status: str
    progress: float
    duration_seconds: Optional[float]
    error_message: Optional[str]
    resource_usage: ResourceUsage


# API Documentation Types

@dataclass
class APIEndpoint:
    """Documentation for an API endpoint."""
    name: str
    method: str
    path: str
    description: str
    
    # Request information
    request_type: Optional[type] = None
    request_schema: Optional[Dict[str, Any]] = None
    request_example: Optional[Dict[str, Any]] = None
    
    # Response information
    response_type: Optional[type] = None
    response_schema: Optional[Dict[str, Any]] = None
    response_example: Optional[Dict[str, Any]] = None
    
    # Additional documentation
    parameters: List[str] = field(default_factory=list)
    error_codes: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert endpoint documentation to dictionary."""
        return {
            "name": self.name,
            "method": self.method,
            "path": self.path,
            "description": self.description,
            "request_type": str(self.request_type) if self.request_type else None,
            "request_schema": self.request_schema,
            "request_example": self.request_example,
            "response_type": str(self.response_type) if self.response_type else None,
            "response_schema": self.response_schema,
            "response_example": self.response_example,
            "parameters": self.parameters,
            "error_codes": self.error_codes,
            "examples": self.examples
        }


# API Documentation

API_DOCUMENTATION = {
    "title": "Orchestrator API Framework",
    "version": "2.0.0",
    "description": """
    Comprehensive API for pipeline compilation, execution, and management.
    
    The Orchestrator API provides clean, intuitive interfaces for:
    - Compiling YAML pipeline specifications into executable pipelines
    - Executing pipelines with real-time status tracking and monitoring
    - Managing pipeline state and controlling execution flow
    - Comprehensive error handling and automatic recovery mechanisms
    """,
    
    "endpoints": [
        APIEndpoint(
            name="compile_pipeline",
            method="POST",
            path="/api/v2/pipelines/compile",
            description="Compile a YAML pipeline specification into an executable pipeline",
            request_type=CompilationRequest,
            response_type=CompilationResult,
            request_example={
                "yaml_content": "steps:\n  - name: example\n    type: text_generation\n    model: AUTO",
                "context": {"variable": "value"},
                "validation_level": "strict"
            },
            response_example={
                "success": True,
                "data": {
                    "pipeline_id": "pipeline_12345",
                    "compilation_time_ms": 150,
                    "validation_passed": True,
                    "validation_warnings": []
                }
            }
        ),
        
        APIEndpoint(
            name="execute_pipeline",
            method="POST", 
            path="/api/v2/pipelines/execute",
            description="Execute a compiled pipeline with monitoring and control",
            request_type=ExecutionRequest,
            response_type=ExecutionResult,
            request_example={
                "pipeline": "pipeline_12345",
                "context": {"input": "test data"},
                "enable_monitoring": True
            },
            response_example={
                "success": True,
                "data": {
                    "execution_id": "exec_67890",
                    "status": "running",
                    "total_steps": 5,
                    "progress_url": "/api/v2/executions/exec_67890/progress"
                }
            }
        ),
        
        APIEndpoint(
            name="get_execution_status",
            method="GET",
            path="/api/v2/executions/{execution_id}/status",
            description="Get comprehensive status information for a pipeline execution",
            response_type=ExecutionStatusInfo,
            response_example={
                "success": True,
                "data": {
                    "execution_id": "exec_67890",
                    "status": "running",
                    "progress_percentage": 60.0,
                    "steps_completed": 3,
                    "steps_total": 5
                }
            }
        ),
        
        APIEndpoint(
            name="monitor_execution",
            method="GET",
            path="/api/v2/executions/{execution_id}/progress",
            description="Real-time progress monitoring for pipeline execution",
            response_type=ProgressUpdate,
            response_example={
                "success": True,
                "data": {
                    "execution_id": "exec_67890",
                    "event_type": "step_completed",
                    "overall_progress": 80.0,
                    "message": "Step 4 completed successfully"
                }
            }
        ),
        
        APIEndpoint(
            name="stop_execution",
            method="POST",
            path="/api/v2/executions/{execution_id}/stop",
            description="Stop a running pipeline execution",
            parameters=["graceful: bool = true"],
            response_example={
                "success": True,
                "data": {"stopped": True, "graceful": True}
            }
        )
    ]
}

# Type aliases for convenience
PipelineCompilationResponse = APIResponse[CompilationResult]
PipelineExecutionResponse = APIResponse[ExecutionResult]
ExecutionStatusResponse = APIResponse[ExecutionStatusInfo]
ProgressUpdateResponse = APIResponse[ProgressUpdate]

# Export all types for easy importing
__all__ = [
    # Enums
    "APIOperation", "ValidationLevel", "CompilationMode", "ExecutionMode",
    
    # Request types
    "CompilationRequest", "ExecutionRequest",
    
    # Response types
    "APIResponse", "CompilationResult", "ExecutionResult",
    
    # Status types
    "ExecutionStatusInfo", "ProgressUpdate",
    
    # Configuration types
    "APIConfiguration",
    
    # Protocol types
    "PipelineCompilerProtocol", "ExecutionManagerProtocol", "ProgressMonitorProtocol",
    
    # Callback types
    "ProgressCallback", "StatusCallback", "ErrorCallback", "CompletionCallback",
    "AsyncProgressCallback", "AsyncStatusCallback", "AsyncErrorCallback", "AsyncCompletionCallback",
    
    # Typed dictionary types
    "PipelineCompilationDict", "PipelineExecutionDict", "ExecutionStatusDict",
    "ValidationResult", "ResourceUsage", "StepSummary",
    
    # Documentation types
    "APIEndpoint", "API_DOCUMENTATION",
    
    # Response aliases
    "PipelineCompilationResponse", "PipelineExecutionResponse", 
    "ExecutionStatusResponse", "ProgressUpdateResponse"
]