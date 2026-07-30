"""Base class for pipeline testing with comprehensive validation and utilities."""

import asyncio
import json
import tempfile
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
import yaml

from orchestrator import Orchestrator
from orchestrator.core.exceptions import OrchestratorError
from orchestrator.models.model_registry import ModelRegistry


@dataclass
class PipelineExecutionResult:
    """Result of a pipeline execution with metadata and analysis."""
    
    success: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error: Optional[Exception] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Performance metrics
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    api_calls_count: int = 0
    tokens_used: int = 0
    estimated_cost: float = 0.0
    
    # Quality metrics
    validation_results: Dict[str, bool] = field(default_factory=dict)
    template_validation: bool = True
    dependency_validation: bool = True
    output_validation: bool = True
    
    # Execution details
    task_results: Dict[str, Any] = field(default_factory=dict)
    intermediate_outputs: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.end_time and self.start_time:
            self.execution_time = self.end_time - self.start_time
    
    @property
    def is_valid(self) -> bool:
        """Check if execution result is valid based on all validations."""
        return (
            self.success and
            self.template_validation and
            self.dependency_validation and
            self.output_validation and
            not self.error
        )
    
    @property
    def performance_score(self) -> float:
        """Calculate performance score (0-1) based on execution metrics."""
        if not self.success:
            return 0.0
        
        # Base score for successful execution
        score = 0.5
        
        # Time performance (faster is better, penalize over 60s)
        if self.execution_time < 30:
            score += 0.2
        elif self.execution_time < 60:
            score += 0.1
        
        # Cost performance (cheaper is better, penalize over $0.10)
        if self.estimated_cost < 0.05:
            score += 0.2
        elif self.estimated_cost < 0.10:
            score += 0.1
        
        # Quality performance (all validations pass)
        if all(self.validation_results.values()):
            score += 0.1
        
        return min(1.0, score)


@dataclass
class PipelineTestConfiguration:
    """Configuration for pipeline test execution."""
    
    timeout_seconds: int = 180
    max_cost_dollars: float = 1.0
    enable_performance_tracking: bool = True
    enable_validation: bool = True
    save_intermediate_outputs: bool = True
    parallel_execution: bool = False
    retry_on_failure: bool = True
    max_retries: int = 2
    
    # Validation settings
    validate_templates: bool = True
    validate_dependencies: bool = True
    validate_outputs: bool = True
    check_output_types: bool = True
    
    # Performance thresholds
    max_execution_time: float = 300
    max_memory_mb: float = 1000
    min_success_rate: float = 0.90


class BasePipelineTest(ABC):
    """
    Base class for comprehensive pipeline testing.
    
    Provides utilities for:
    - Pipeline execution with async support
    - Output validation and quality scoring
    - Performance tracking and cost analysis
    - Error reporting and debugging
    - Template validation and dependency checking
    """
    
    def __init__(self, 
                 orchestrator: Orchestrator,
                 model_registry: ModelRegistry,
                 config: Optional[PipelineTestConfiguration] = None):
        """
        Initialize pipeline test instance.
        
        Args:
            orchestrator: Orchestrator instance for pipeline execution
            model_registry: Model registry for cost estimation
            config: Test configuration (uses defaults if None)
        """
        self.orchestrator = orchestrator
        self.model_registry = model_registry
        self.config = config or PipelineTestConfiguration()
        
        # Track execution history
        self.execution_history: List[PipelineExecutionResult] = []
        self.total_cost = 0.0
        self.total_executions = 0
        
    async def execute_pipeline_async(self,
                                   yaml_content: str,
                                   inputs: Optional[Dict[str, Any]] = None,
                                   output_dir: Optional[Path] = None) -> PipelineExecutionResult:
        """
        Execute a pipeline asynchronously with comprehensive tracking.
        
        Args:
            yaml_content: YAML pipeline definition
            inputs: Pipeline input parameters
            output_dir: Directory for saving outputs
            
        Returns:
            PipelineExecutionResult: Comprehensive execution result
        """
        result = PipelineExecutionResult(success=False, start_time=time.time())
        inputs = inputs or {}
        
        try:
            # Pre-execution validation
            if self.config.validate_templates:
                result.template_validation = self._validate_templates(yaml_content)
            
            if self.config.validate_dependencies:
                result.dependency_validation = self._validate_dependencies(yaml_content)
            
            # Create temporary file for pipeline
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_content)
                temp_pipeline_path = Path(f.name)
            
            try:
                # Execute with timeout
                execution_task = asyncio.create_task(
                    self._execute_with_orchestrator(temp_pipeline_path, inputs, output_dir)
                )
                
                outputs = await asyncio.wait_for(
                    execution_task, 
                    timeout=self.config.timeout_seconds
                )
                
                result.outputs = outputs
                result.success = True
                
                # Post-execution validation
                if self.config.validate_outputs:
                    result.output_validation = self._validate_outputs(outputs)
                
                # Extract performance metrics
                if self.config.enable_performance_tracking:
                    self._extract_performance_metrics(result, outputs)
                
            finally:
                # Clean up temp file
                if temp_pipeline_path.exists():
                    temp_pipeline_path.unlink()
                    
        except asyncio.TimeoutError:
            result.error = TimeoutError(f"Pipeline execution timed out after {self.config.timeout_seconds}s")
            result.error_message = str(result.error)
        except Exception as e:
            result.error = e
            result.error_message = str(e)
            result.error_traceback = traceback.format_exc()
        
        finally:
            result.end_time = time.time()
            
            # Track total cost and executions
            self.total_cost += result.estimated_cost
            self.total_executions += 1
            self.execution_history.append(result)
        
        return result
    
    def execute_pipeline_sync(self,
                            yaml_content: str,
                            inputs: Optional[Dict[str, Any]] = None,
                            output_dir: Optional[Path] = None) -> PipelineExecutionResult:
        """
        Execute a pipeline synchronously (wrapper for async method).
        
        Args:
            yaml_content: YAML pipeline definition
            inputs: Pipeline input parameters
            output_dir: Directory for saving outputs
            
        Returns:
            PipelineExecutionResult: Comprehensive execution result
        """
        return asyncio.run(self.execute_pipeline_async(yaml_content, inputs, output_dir))
    
    async def _execute_with_orchestrator(self,
                                       pipeline_path: Path,
                                       inputs: Dict[str, Any],
                                       output_dir: Optional[Path]) -> Dict[str, Any]:
        """Execute pipeline using orchestrator."""
        # Prepare execution context
        context = inputs.copy()
        if output_dir:
            context["output_path"] = str(output_dir)
        
        # Execute pipeline
        results = await self.orchestrator.execute_yaml_file(
            str(pipeline_path), 
            context=context
        )
        
        return results or {}
    
    def _validate_templates(self, yaml_content: str) -> bool:
        """
        Validate template syntax and structure.
        
        Args:
            yaml_content: YAML content to validate
            
        Returns:
            bool: True if templates are valid
        """
        try:
            # Parse YAML
            pipeline_data = yaml.safe_load(yaml_content)
            
            if not isinstance(pipeline_data, dict):
                return False
            
            # Check required top-level fields
            if 'tasks' not in pipeline_data:
                return False
            
            # Validate each task
            for task in pipeline_data.get('tasks', []):
                if not isinstance(task, dict):
                    return False
                
                # Check required task fields
                if 'name' not in task or 'type' not in task:
                    return False
                
                # Validate template field if present
                if 'template' in task:
                    template = task['template']
                    if not isinstance(template, str):
                        return False
                    
                    # Basic template syntax validation
                    if '{{' in template:
                        # Check for balanced braces
                        open_count = template.count('{{')
                        close_count = template.count('}}')
                        if open_count != close_count:
                            return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_dependencies(self, yaml_content: str) -> bool:
        """
        Validate task dependencies are properly defined.
        
        Args:
            yaml_content: YAML content to validate
            
        Returns:
            bool: True if dependencies are valid
        """
        try:
            pipeline_data = yaml.safe_load(yaml_content)
            
            if not isinstance(pipeline_data, dict):
                return False
            
            tasks = pipeline_data.get('tasks', [])
            task_names = {task.get('name') for task in tasks if isinstance(task, dict)}
            
            # Check all dependency references are valid
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                
                dependencies = task.get('dependencies', [])
                if dependencies:
                    for dep in dependencies:
                        if dep not in task_names:
                            return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_outputs(self, outputs: Dict[str, Any]) -> bool:
        """
        Validate pipeline outputs meet quality standards.
        
        Args:
            outputs: Pipeline execution outputs
            
        Returns:
            bool: True if outputs are valid
        """
        if not outputs:
            return False
        
        # Check for common error indicators
        for key, value in outputs.items():
            if isinstance(value, str):
                # Check for common error patterns
                error_indicators = [
                    'error:', 'failed:', 'exception:', 'traceback:',
                    'null', 'undefined', 'none'
                ]
                value_lower = value.lower()
                if any(indicator in value_lower for indicator in error_indicators):
                    return False
                
                # Check for empty or placeholder content
                if not value.strip() or value.strip() in ['', '{}', '[]', 'null']:
                    return False
        
        return True
    
    def _extract_performance_metrics(self, 
                                   result: PipelineExecutionResult,
                                   outputs: Dict[str, Any]):
        """
        Extract performance metrics from execution.
        
        Args:
            result: Result object to populate
            outputs: Pipeline outputs containing metadata
        """
        # Extract execution metadata if available
        metadata = outputs.get('execution_metadata', {})
        
        if isinstance(metadata, dict):
            result.api_calls_count = metadata.get('api_calls', 0)
            result.tokens_used = metadata.get('tokens_used', 0)
            result.estimated_cost = metadata.get('total_cost', 0.0)
            result.memory_usage_mb = metadata.get('memory_peak_mb', 0.0)
        
        # Estimate cost if not provided
        if result.estimated_cost == 0.0:
            result.estimated_cost = self._estimate_execution_cost(result)
    
    def _estimate_execution_cost(self, result: PipelineExecutionResult) -> float:
        """
        Estimate execution cost based on tokens and model usage.
        
        Args:
            result: Execution result
            
        Returns:
            float: Estimated cost in dollars
        """
        # Simple cost estimation - can be enhanced
        if result.tokens_used > 0:
            # Rough estimate: $0.02 per 1000 tokens for GPT-4o-mini
            return (result.tokens_used / 1000) * 0.02
        elif result.api_calls_count > 0:
            # Fallback: estimate based on API calls
            return result.api_calls_count * 0.01
        else:
            # Default minimal cost
            return 0.001
    
    def assert_pipeline_success(self, 
                              result: PipelineExecutionResult,
                              message: Optional[str] = None):
        """
        Assert that pipeline execution was successful.
        
        Args:
            result: Pipeline execution result
            message: Optional custom error message
        """
        error_msg = message or self._format_error_message(result)
        assert result.success, error_msg
        assert result.is_valid, f"Pipeline validation failed: {error_msg}"
    
    def assert_output_contains(self,
                             result: PipelineExecutionResult,
                             key: str,
                             expected_content: Union[str, List[str]],
                             case_sensitive: bool = False):
        """
        Assert that output contains expected content.
        
        Args:
            result: Pipeline execution result
            key: Output key to check
            expected_content: Content that should be present
            case_sensitive: Whether to perform case-sensitive matching
        """
        assert key in result.outputs, f"Output key '{key}' not found in results"
        
        output_value = result.outputs[key]
        output_str = str(output_value)
        
        if not case_sensitive:
            output_str = output_str.lower()
        
        if isinstance(expected_content, str):
            expected_content = [expected_content]
        
        for content in expected_content:
            search_content = content if case_sensitive else content.lower()
            assert search_content in output_str, (
                f"Expected content '{content}' not found in output '{key}': {output_str[:200]}..."
            )
    
    def assert_performance_within_limits(self,
                                       result: PipelineExecutionResult,
                                       max_time: Optional[float] = None,
                                       max_cost: Optional[float] = None):
        """
        Assert that performance metrics are within acceptable limits.
        
        Args:
            result: Pipeline execution result
            max_time: Maximum execution time in seconds
            max_cost: Maximum cost in dollars
        """
        max_time = max_time or self.config.max_execution_time
        max_cost = max_cost or self.config.max_cost_dollars
        
        assert result.execution_time <= max_time, (
            f"Execution time {result.execution_time:.2f}s exceeds limit {max_time}s"
        )
        
        assert result.estimated_cost <= max_cost, (
            f"Execution cost ${result.estimated_cost:.4f} exceeds limit ${max_cost}"
        )
    
    def _format_error_message(self, result: PipelineExecutionResult) -> str:
        """
        Format comprehensive error message for failed pipeline.
        
        Args:
            result: Failed pipeline execution result
            
        Returns:
            str: Formatted error message
        """
        msg_parts = ["Pipeline execution failed:"]
        
        if result.error_message:
            msg_parts.append(f"Error: {result.error_message}")
        
        if result.execution_time > 0:
            msg_parts.append(f"Execution time: {result.execution_time:.2f}s")
        
        if result.estimated_cost > 0:
            msg_parts.append(f"Estimated cost: ${result.estimated_cost:.4f}")
        
        validation_issues = []
        if not result.template_validation:
            validation_issues.append("template")
        if not result.dependency_validation:
            validation_issues.append("dependency")
        if not result.output_validation:
            validation_issues.append("output")
        
        if validation_issues:
            msg_parts.append(f"Validation failures: {', '.join(validation_issues)}")
        
        if result.warnings:
            msg_parts.append(f"Warnings: {'; '.join(result.warnings)}")
        
        return "\n".join(msg_parts)
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of all pipeline executions.
        
        Returns:
            Dict[str, Any]: Execution summary statistics
        """
        if not self.execution_history:
            return {"total_executions": 0}
        
        successful = [r for r in self.execution_history if r.success]
        
        return {
            "total_executions": self.total_executions,
            "successful_executions": len(successful),
            "success_rate": len(successful) / self.total_executions,
            "total_cost": self.total_cost,
            "average_cost": self.total_cost / self.total_executions,
            "average_execution_time": sum(r.execution_time for r in self.execution_history) / len(self.execution_history),
            "total_api_calls": sum(r.api_calls_count for r in self.execution_history),
            "total_tokens": sum(r.tokens_used for r in self.execution_history),
            "average_performance_score": sum(r.performance_score for r in self.execution_history) / len(self.execution_history)
        }
    
    @abstractmethod
    def test_basic_execution(self):
        """Test basic pipeline execution - must be implemented by subclasses."""
        pass
    
    @abstractmethod  
    def test_error_handling(self):
        """Test error handling scenarios - must be implemented by subclasses."""
        pass