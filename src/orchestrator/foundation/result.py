"""
Result structures for pipeline execution.

This module defines the data structures for capturing pipeline and step execution results.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class StepResult:
    """
    Result of executing a single pipeline step.
    
    Attributes:
        step_id: Unique identifier for the step
        status: Execution status (success, failure, pending, etc.)
        output: The output data produced by the step
        error: Error information if the step failed
        execution_time: Time taken to execute the step
        metadata: Additional metadata about step execution
    """
    step_id: str
    status: str
    output: Dict[str, Any]
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class PipelineResult:
    """
    Result of executing a complete pipeline.
    
    Attributes:
        pipeline_name: Name of the pipeline that was executed
        status: Overall pipeline execution status
        step_results: List of individual step results
        total_steps: Total number of steps in the pipeline
        executed_steps: Number of steps that were executed
        execution_time: Total pipeline execution time
        metadata: Additional metadata about pipeline execution
    """
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