"""
Compatibility layer for migrating from foundation module to new architecture.

This module provides aliases and compatibility classes to maintain backward compatibility
while transitioning to the new execution and API modules.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


# Compatibility aliases for foundation types (using late imports to avoid circular deps)
def _get_pipeline_class():
    from ..core.pipeline import Pipeline
    return Pipeline

def _get_task_class():
    from ..core.task import Task
    return Task

# Set up compatibility aliases lazily
class _LazyAlias:
    def __init__(self, factory):
        self._factory = factory
        self._value = None
    
    def __call__(self, *args, **kwargs):
        if self._value is None:
            self._value = self._factory()
        return self._value(*args, **kwargs)
    
    def __getattr__(self, name):
        if self._value is None:
            self._value = self._factory()
        return getattr(self._value, name)

PipelineSpecification = _LazyAlias(_get_pipeline_class)
PipelineStep = _LazyAlias(_get_task_class)


@dataclass
class FoundationConfig:
    """Compatibility class for FoundationConfig - maps to APIConfiguration."""
    
    # Model Management
    default_model: Optional[str] = None
    model_selection_strategy: str = "balanced"
    
    # Execution
    max_concurrent_steps: int = 5
    execution_timeout: int = 3600
    
    # Tool Registry
    auto_install_tools: bool = True
    tool_timeout: int = 300
    
    # Quality Control
    enable_quality_checks: bool = True
    quality_threshold: float = 0.7
    
    # LangGraph Integration
    enable_persistence: bool = False
    storage_backend: str = "memory"
    database_url: Optional[str] = None
    
    # Progress Monitoring
    show_progress_bars: bool = True
    log_level: str = "INFO"
    
    def to_api_config(self):
        """Convert to new APIConfiguration."""
        from ..api.types import APIConfiguration
        return APIConfiguration(
            default_execution_timeout=self.execution_timeout,
            max_concurrent_executions=self.max_concurrent_steps,
            enable_execution_recovery=True,
            enable_execution_checkpointing=self.enable_persistence,
            log_level=self.log_level,
        )


@dataclass
class PipelineHeader:
    """Compatibility class for pipeline headers."""
    name: str
    version: str
    description: Optional[str] = None
    author: Optional[str] = None
    created: Optional[datetime] = None
    tags: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None


@dataclass
class StepResult:
    """Compatibility class for step execution results."""
    
    step_id: str
    status: str
    output: Dict[str, Any]
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PipelineResult:
    """Compatibility class for pipeline execution results."""
    
    pipeline_name: str
    status: str
    step_results: List[StepResult]
    total_steps: int
    executed_steps: Optional[int] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set executed_steps to length of step_results if not provided."""
        if self.executed_steps is None:
            self.executed_steps = len(self.step_results)


class ExecutionEngineInterface:
    """Compatibility interface for execution engines."""
    
    async def execute(self, spec: PipelineSpecification, inputs: Dict[str, Any]) -> PipelineResult:
        """Execute a pipeline specification."""
        raise NotImplementedError
    
    async def execute_step(self, step_id: str, context: Dict[str, Any]) -> StepResult:
        """Execute a single pipeline step."""
        raise NotImplementedError
    
    def get_execution_progress(self) -> Dict[str, Any]:
        """Get current execution progress."""
        raise NotImplementedError