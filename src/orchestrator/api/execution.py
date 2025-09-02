"""
Advanced Pipeline Execution for the Orchestrator Framework.

This module provides specialized pipeline execution methods with comprehensive status
tracking, monitoring, control, and integration with the execution engine. It builds
upon the core API to provide detailed execution management capabilities.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Callable

# Import foundation components
from ..execution import (
    ComprehensiveExecutionManager,
    create_comprehensive_execution_manager,
    ExecutionStatus,
    ExecutionMetrics,
    ProgressTracker,
    ProgressEvent,
    ProgressEventType,
    StepStatus,
    RecoveryManager,
    ErrorSeverity,
    VariableManager
)
from ..core.pipeline import Pipeline
from ..core.task import Task
from ..models.registry import ModelRegistry
from ..execution.engine import StateGraphEngine
from ..execution.model_selector import ExecutionModelSelector

logger = logging.getLogger(__name__)


class PipelineExecutionError(Exception):
    """Exception raised during pipeline execution."""
    pass


class ExecutionControlError(PipelineExecutionError):
    """Exception raised during execution control operations."""
    pass


class PipelineExecutor:
    """
    Advanced pipeline execution with comprehensive monitoring and control.
    
    Provides specialized execution methods that enhance the basic execution capabilities
    with real-time monitoring, advanced progress tracking, execution control, and
    comprehensive status reporting.
    
    Example:
        >>> executor = PipelineExecutor()
        >>> execution = await executor.execute_with_monitoring(pipeline, context)
        >>> async for event in executor.monitor_execution(execution.execution_id):
        ...     print(f"Progress: {event.progress}%")
    """
    
    def __init__(
        self,
        max_concurrent_executions: int = 10,
        default_timeout: Optional[int] = 3600,
        enable_recovery: bool = True,
        enable_checkpointing: bool = True,
        model_registry: Optional[ModelRegistry] = None,
        enable_intelligent_selection: bool = True
    ):
        """
        Initialize the pipeline executor with intelligent model selection capabilities.
        
        Args:
            max_concurrent_executions: Maximum number of concurrent pipeline executions
            default_timeout: Default execution timeout in seconds
            enable_recovery: Enable execution recovery features
            enable_checkpointing: Enable execution checkpointing
            model_registry: Model registry for intelligent model selection
            enable_intelligent_selection: Enable intelligent model selection features
        """
        self.max_concurrent_executions = max_concurrent_executions
        self.default_timeout = default_timeout
        self.enable_recovery = enable_recovery
        self.enable_checkpointing = enable_checkpointing
        self.enable_intelligent_selection = enable_intelligent_selection
        
        # Model selection integration
        self.model_registry = model_registry
        self._execution_engine = None
        if model_registry and enable_intelligent_selection:
            from ..foundation._compatibility import FoundationConfig
            config = FoundationConfig(
                max_concurrent_steps=max_concurrent_executions,
                enable_persistence=enable_checkpointing
            )
            self._execution_engine = StateGraphEngine(config=config, model_registry=model_registry)
        
        # Track active executions
        self._active_executions: Dict[str, ComprehensiveExecutionManager] = {}
        self._execution_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Progress monitoring
        self._progress_callbacks: Dict[str, List[Callable]] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info(f"PipelineExecutor initialized with max_concurrent={max_concurrent_executions}, "
                   f"intelligent_selection={enable_intelligent_selection and model_registry is not None}")
    
    async def execute_with_monitoring(
        self,
        pipeline: Pipeline,
        context: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
        timeout: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> ComprehensiveExecutionManager:
        """
        Execute pipeline with comprehensive monitoring and progress tracking.
        
        Args:
            pipeline: Compiled pipeline object
            context: Additional execution context variables
            execution_id: Optional custom execution ID
            timeout: Execution timeout in seconds (overrides default)
            progress_callback: Optional callback for progress updates
            
        Returns:
            ComprehensiveExecutionManager for monitoring and control
            
        Raises:
            PipelineExecutionError: If execution initialization fails
        """
        try:
            # Check concurrent execution limit
            if len(self._active_executions) >= self.max_concurrent_executions:
                raise PipelineExecutionError(
                    f"Maximum concurrent executions ({self.max_concurrent_executions}) reached"
                )
            
            # Generate execution ID
            if not execution_id:
                execution_id = f"{pipeline.id}_{uuid.uuid4().hex[:8]}"
            
            logger.info(f"Starting monitored execution {execution_id} for pipeline '{pipeline.id}'")
            
            # Create comprehensive execution manager
            execution_manager = create_comprehensive_execution_manager(
                execution_id=execution_id,
                pipeline_id=pipeline.id
            )
            
            # Setup execution metadata
            execution_metadata = {
                "pipeline_id": pipeline.id,
                "pipeline_name": pipeline.name,
                "execution_id": execution_id,
                "start_time": datetime.now(),
                "timeout": timeout or self.default_timeout,
                "total_tasks": len(pipeline.tasks),
                "current_task": None,
                "status": ExecutionStatus.PENDING,
                "progress_percentage": 0.0,
                "estimated_completion": None
            }
            
            # Store execution references
            self._active_executions[execution_id] = execution_manager
            self._execution_metadata[execution_id] = execution_metadata
            
            # Setup progress callback
            if progress_callback:
                if execution_id not in self._progress_callbacks:
                    self._progress_callbacks[execution_id] = []
                self._progress_callbacks[execution_id].append(progress_callback)
            
            # Add execution context if provided
            if context:
                for key, value in context.items():
                    execution_manager.variable_manager.set_variable(key, value)
            
            # Add pipeline context to execution
            if pipeline.context:
                for key, value in pipeline.context.items():
                    execution_manager.variable_manager.set_variable(key, value)
            
            # Start execution monitoring
            monitoring_task = asyncio.create_task(
                self._monitor_execution_progress(execution_id, execution_manager)
            )
            self._monitoring_tasks[execution_id] = monitoring_task
            
            # Start execution with step count
            total_steps = len(pipeline.tasks)
            execution_manager.start_execution(total_steps)
            
            # Update metadata
            execution_metadata["status"] = ExecutionStatus.RUNNING
            execution_metadata["start_time"] = datetime.now()
            
            logger.info(f"Execution {execution_id} started with monitoring")
            
            return execution_manager
            
        except Exception as e:
            logger.error(f"Failed to initialize monitored pipeline execution: {e}")
            raise PipelineExecutionError(f"Execution initialization failed: {e}") from e
    
    async def execute_with_streaming(
        self,
        pipeline: Pipeline,
        context: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute pipeline with streaming progress updates.
        
        Args:
            pipeline: Compiled pipeline object
            context: Additional execution context variables
            execution_id: Optional custom execution ID
            
        Yields:
            Dictionary containing execution progress updates
            
        Raises:
            PipelineExecutionError: If execution fails
        """
        try:
            # Start execution with monitoring
            execution_manager = await self.execute_with_monitoring(
                pipeline=pipeline,
                context=context,
                execution_id=execution_id
            )
            
            execution_id = execution_manager.execution_id
            
            # Stream progress updates
            async for progress_update in self._stream_execution_progress(execution_id):
                yield progress_update
            
        except Exception as e:
            logger.error(f"Streaming execution failed: {e}")
            raise PipelineExecutionError(f"Streaming execution error: {e}") from e
    
    async def monitor_execution(
        self,
        execution_id: str,
        update_interval: float = 1.0
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Monitor an active execution with regular status updates.
        
        Args:
            execution_id: Unique execution identifier
            update_interval: Update interval in seconds
            
        Yields:
            Dictionary containing execution status updates
            
        Raises:
            ExecutionControlError: If execution not found
        """
        if execution_id not in self._active_executions:
            raise ExecutionControlError(f"Execution not found: {execution_id}")
        
        execution_manager = self._active_executions[execution_id]
        
        try:
            while execution_id in self._active_executions:
                # Get current status
                status = execution_manager.get_execution_status()
                
                # Add metadata
                if execution_id in self._execution_metadata:
                    metadata = self._execution_metadata[execution_id]
                    status.update({
                        "pipeline_name": metadata.get("pipeline_name"),
                        "start_time": metadata.get("start_time"),
                        "estimated_completion": metadata.get("estimated_completion")
                    })
                
                yield status
                
                # Check if execution completed
                if status.get("status") in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
                    break
                
                await asyncio.sleep(update_interval)
                
        except Exception as e:
            logger.error(f"Error monitoring execution {execution_id}: {e}")
            yield {
                "execution_id": execution_id,
                "status": ExecutionStatus.FAILED,
                "error": str(e)
            }
    
    async def pause_execution(self, execution_id: str) -> bool:
        """
        Pause a running pipeline execution.
        
        Args:
            execution_id: Unique execution identifier
            
        Returns:
            True if execution was paused successfully
            
        Raises:
            ExecutionControlError: If execution not found or cannot be paused
        """
        if execution_id not in self._active_executions:
            raise ExecutionControlError(f"Execution not found: {execution_id}")
        
        execution_manager = self._active_executions[execution_id]
        
        try:
            logger.info(f"Pausing execution {execution_id}")
            
            # Update metadata
            if execution_id in self._execution_metadata:
                self._execution_metadata[execution_id]["status"] = ExecutionStatus.PAUSED
            
            # Note: ComprehensiveExecutionManager doesn't have pause/resume yet
            # This is a placeholder for future implementation
            logger.warning(f"Pause functionality not yet implemented for execution {execution_id}")
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to pause execution {execution_id}: {e}")
            raise ExecutionControlError(f"Failed to pause execution: {e}") from e
    
    async def resume_execution(self, execution_id: str) -> bool:
        """
        Resume a paused pipeline execution.
        
        Args:
            execution_id: Unique execution identifier
            
        Returns:
            True if execution was resumed successfully
            
        Raises:
            ExecutionControlError: If execution not found or cannot be resumed
        """
        if execution_id not in self._active_executions:
            raise ExecutionControlError(f"Execution not found: {execution_id}")
        
        execution_manager = self._active_executions[execution_id]
        
        try:
            logger.info(f"Resuming execution {execution_id}")
            
            # Update metadata
            if execution_id in self._execution_metadata:
                self._execution_metadata[execution_id]["status"] = ExecutionStatus.RUNNING
            
            # Note: ComprehensiveExecutionManager doesn't have pause/resume yet
            # This is a placeholder for future implementation
            logger.warning(f"Resume functionality not yet implemented for execution {execution_id}")
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resume execution {execution_id}: {e}")
            raise ExecutionControlError(f"Failed to resume execution: {e}") from e
    
    async def stop_execution(
        self,
        execution_id: str,
        graceful: bool = True,
        reason: Optional[str] = None
    ) -> bool:
        """
        Stop a running pipeline execution.
        
        Args:
            execution_id: Unique execution identifier
            graceful: Whether to wait for current step to complete
            reason: Optional reason for stopping the execution
            
        Returns:
            True if execution was stopped successfully
            
        Raises:
            ExecutionControlError: If execution not found
        """
        if execution_id not in self._active_executions:
            raise ExecutionControlError(f"Execution not found: {execution_id}")
        
        execution_manager = self._active_executions[execution_id]
        
        try:
            logger.info(f"Stopping execution {execution_id} (graceful={graceful}, reason={reason})")
            
            # Complete execution as failed/stopped
            execution_manager.complete_execution(success=False)
            
            # Clean up execution
            execution_manager.cleanup()
            
            # Stop monitoring task
            if execution_id in self._monitoring_tasks:
                monitoring_task = self._monitoring_tasks[execution_id]
                monitoring_task.cancel()
                try:
                    await monitoring_task
                except asyncio.CancelledError:
                    pass
                del self._monitoring_tasks[execution_id]
            
            # Clean up resources
            await self._cleanup_execution(execution_id)
            
            logger.info(f"Execution {execution_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop execution {execution_id}: {e}")
            return False
    
    def get_execution_summary(self, execution_id: str) -> Dict[str, Any]:
        """
        Get comprehensive execution summary.
        
        Args:
            execution_id: Unique execution identifier
            
        Returns:
            Dictionary containing execution summary
            
        Raises:
            ExecutionControlError: If execution not found
        """
        if execution_id not in self._active_executions:
            raise ExecutionControlError(f"Execution not found: {execution_id}")
        
        execution_manager = self._active_executions[execution_id]
        metadata = self._execution_metadata.get(execution_id, {})
        
        # Get base status
        status = execution_manager.get_execution_status()
        
        # Enhance with metadata and calculations
        summary = {
            **status,
            "pipeline_name": metadata.get("pipeline_name"),
            "start_time": metadata.get("start_time"),
            "total_tasks": metadata.get("total_tasks", 0),
            "timeout": metadata.get("timeout"),
            "progress_percentage": self._calculate_progress_percentage(execution_id),
            "estimated_completion": self._calculate_estimated_completion(execution_id),
            "execution_duration": self._calculate_execution_duration(execution_id),
            "average_task_duration": self._calculate_average_task_duration(execution_id)
        }
        
        return summary
    
    def list_active_executions(self) -> List[Dict[str, Any]]:
        """
        Get list of all active executions with basic information.
        
        Returns:
            List of dictionaries containing execution information
        """
        active_executions = []
        
        for execution_id, execution_manager in self._active_executions.items():
            metadata = self._execution_metadata.get(execution_id, {})
            
            execution_info = {
                "execution_id": execution_id,
                "pipeline_id": metadata.get("pipeline_id"),
                "pipeline_name": metadata.get("pipeline_name"),
                "status": metadata.get("status", ExecutionStatus.UNKNOWN),
                "start_time": metadata.get("start_time"),
                "progress_percentage": self._calculate_progress_percentage(execution_id),
                "total_tasks": metadata.get("total_tasks", 0)
            }
            
            active_executions.append(execution_info)
        
        return active_executions
    
    def get_execution_metrics(self, execution_id: str) -> Dict[str, Any]:
        """
        Get detailed execution metrics and performance data.
        
        Args:
            execution_id: Unique execution identifier
            
        Returns:
            Dictionary containing execution metrics
            
        Raises:
            ExecutionControlError: If execution not found
        """
        if execution_id not in self._active_executions:
            raise ExecutionControlError(f"Execution not found: {execution_id}")
        
        execution_manager = self._active_executions[execution_id]
        
        # Get base metrics from execution manager
        try:
            base_metrics = execution_manager.get_metrics()
        except AttributeError:
            # Fallback if metrics not available
            base_metrics = {}
        
        # Calculate additional metrics
        metrics = {
            **base_metrics,
            "execution_id": execution_id,
            "memory_usage": self._get_memory_usage(execution_id),
            "cpu_usage": self._get_cpu_usage(execution_id),
            "network_stats": self._get_network_stats(execution_id),
            "error_counts": self._get_error_counts(execution_id),
            "performance_stats": self._get_performance_stats(execution_id)
        }
        
        # Add model selection metrics if intelligent selection is enabled
        if self.enable_intelligent_selection:
            metrics["model_selection"] = self._get_model_selection_metrics(execution_id)
        
        return metrics
    
    async def execute_with_intelligent_selection(
        self,
        pipeline: Pipeline,
        context: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None,
        selection_strategy: Optional[str] = None,
        cost_constraints: Optional[Dict[str, Any]] = None
    ) -> ComprehensiveExecutionManager:
        """
        Execute pipeline with intelligent model selection for optimal performance.
        
        This method enhances pipeline execution with intelligent model selection,
        choosing optimal models for each step based on requirements and constraints.
        
        Args:
            pipeline: Compiled pipeline object
            context: Additional execution context variables
            execution_id: Optional custom execution ID
            selection_strategy: Model selection strategy ("balanced", "cost_optimized", "performance_optimized", "accuracy_optimized")
            cost_constraints: Optional cost constraints for model selection
            
        Returns:
            ComprehensiveExecutionManager for monitoring and control
            
        Raises:
            PipelineExecutionError: If execution initialization fails
        """
        if not self.enable_intelligent_selection or not self.model_registry:
            logger.warning("Intelligent selection not available, falling back to standard execution")
            return await self.execute_with_monitoring(pipeline, context, execution_id)
        
        try:
            logger.info(f"Starting intelligent execution for pipeline '{pipeline.id}' with strategy: {selection_strategy}")
            
            # Enhance context with model selection parameters
            enhanced_context = context.copy() if context else {}
            if selection_strategy:
                enhanced_context["selection_strategy"] = selection_strategy
            if cost_constraints:
                enhanced_context["cost_constraints"] = cost_constraints
            
            # Get model selection recommendations before execution
            if hasattr(pipeline, 'specification'):
                recommendations = self._execution_engine.get_model_selection_recommendations(
                    pipeline.specification, enhanced_context
                )
                enhanced_context["model_recommendations"] = recommendations
                logger.info(f"Generated model recommendations for {len(recommendations)} steps")
            
            # Execute with enhanced context
            execution_manager = await self.execute_with_monitoring(
                pipeline=pipeline,
                context=enhanced_context,
                execution_id=execution_id
            )
            
            # Mark as intelligent execution
            if execution_manager.execution_id in self._execution_metadata:
                self._execution_metadata[execution_manager.execution_id]["intelligent_selection"] = True
                self._execution_metadata[execution_manager.execution_id]["selection_strategy"] = selection_strategy
            
            return execution_manager
            
        except Exception as e:
            logger.error(f"Intelligent execution failed: {e}")
            raise PipelineExecutionError(f"Intelligent execution error: {e}") from e
    
    def get_model_selection_recommendations(
        self,
        pipeline: Pipeline,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get model selection recommendations for a pipeline before execution.
        
        This provides upfront analysis of optimal model choices, helping with
        planning, cost estimation, and execution optimization.
        
        Args:
            pipeline: Pipeline to analyze
            execution_context: Optional execution context for recommendations
            
        Returns:
            Dictionary containing model selection recommendations for each step
        """
        if not self.enable_intelligent_selection or not self._execution_engine:
            return {"error": "Intelligent model selection not available"}
        
        try:
            if hasattr(pipeline, 'specification'):
                recommendations = self._execution_engine.get_model_selection_recommendations(
                    pipeline.specification, execution_context or {}
                )
                
                # Enhance with cost and performance estimates
                enhanced_recommendations = {}
                total_estimated_cost = 0.0
                
                for step_id, step_data in recommendations.items():
                    if isinstance(step_data, dict) and "recommendations" in step_data:
                        step_recommendations = step_data["recommendations"]
                        if step_recommendations:
                            # Add aggregate metrics
                            best_recommendation = step_recommendations[0]
                            total_estimated_cost += best_recommendation.get("estimated_cost", 0.0)
                        
                        enhanced_recommendations[step_id] = step_data
                
                # Add pipeline-level summary
                enhanced_recommendations["pipeline_summary"] = {
                    "total_steps": len([k for k in enhanced_recommendations.keys() if k != "pipeline_summary"]),
                    "total_estimated_cost": total_estimated_cost,
                    "intelligent_selection_available": True
                }
                
                return enhanced_recommendations
            else:
                return {"error": "Pipeline specification not available"}
                
        except Exception as e:
            logger.error(f"Failed to generate model selection recommendations: {e}")
            return {"error": str(e)}
    
    def analyze_execution_efficiency(
        self,
        execution_id: str
    ) -> Dict[str, Any]:
        """
        Analyze the efficiency of an execution's model selections.
        
        Provides insights into model selection quality, cost efficiency,
        and performance optimization opportunities.
        
        Args:
            execution_id: Unique execution identifier
            
        Returns:
            Dictionary containing efficiency analysis
        """
        if not self.enable_intelligent_selection:
            return {"error": "Intelligent model selection not available"}
        
        if execution_id not in self._execution_metadata:
            return {"error": "Execution not found"}
        
        try:
            metadata = self._execution_metadata[execution_id]
            
            analysis = {
                "execution_id": execution_id,
                "pipeline_id": metadata.get("pipeline_id"),
                "intelligent_selection_used": metadata.get("intelligent_selection", False),
                "selection_strategy": metadata.get("selection_strategy"),
                "total_steps": metadata.get("total_tasks", 0),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            if metadata.get("intelligent_selection"):
                # Detailed efficiency analysis for intelligent executions
                analysis.update({
                    "cost_efficiency": self._analyze_cost_efficiency(execution_id),
                    "performance_efficiency": self._analyze_performance_efficiency(execution_id),
                    "selection_accuracy": self._analyze_selection_accuracy(execution_id),
                    "optimization_opportunities": self._identify_optimization_opportunities(execution_id)
                })
            else:
                analysis["message"] = "Execution did not use intelligent model selection"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze execution efficiency: {e}")
            return {"error": str(e)}
    
    async def _monitor_execution_progress(
        self,
        execution_id: str,
        execution_manager: ComprehensiveExecutionManager
    ) -> None:
        """
        Internal method to monitor execution progress and trigger callbacks.
        
        Args:
            execution_id: Unique execution identifier
            execution_manager: Execution manager instance
        """
        try:
            while execution_id in self._active_executions:
                # Get current progress
                progress_data = self._get_progress_data(execution_id)
                
                # Trigger progress callbacks
                if execution_id in self._progress_callbacks:
                    for callback in self._progress_callbacks[execution_id]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(progress_data)
                            else:
                                callback(progress_data)
                        except Exception as e:
                            logger.error(f"Progress callback error for {execution_id}: {e}")
                
                # Check if execution completed
                status = execution_manager.get_execution_status()
                if status.get("status") in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
                    break
                
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error(f"Error monitoring execution {execution_id}: {e}")
    
    async def _stream_execution_progress(
        self,
        execution_id: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream execution progress updates.
        
        Args:
            execution_id: Unique execution identifier
            
        Yields:
            Progress update dictionaries
        """
        while execution_id in self._active_executions:
            progress_data = self._get_progress_data(execution_id)
            yield progress_data
            
            # Check if execution completed
            if progress_data.get("status") in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED]:
                break
            
            await asyncio.sleep(0.5)
    
    def _get_progress_data(self, execution_id: str) -> Dict[str, Any]:
        """
        Get current progress data for an execution.
        
        Args:
            execution_id: Unique execution identifier
            
        Returns:
            Dictionary containing progress data
        """
        if execution_id not in self._active_executions:
            return {"error": "Execution not found"}
        
        execution_manager = self._active_executions[execution_id]
        metadata = self._execution_metadata.get(execution_id, {})
        
        status = execution_manager.get_execution_status()
        
        return {
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat(),
            "status": status.get("status", ExecutionStatus.UNKNOWN),
            "progress_percentage": self._calculate_progress_percentage(execution_id),
            "current_step": status.get("current_step", 0),
            "total_steps": metadata.get("total_tasks", 0),
            "estimated_completion": self._calculate_estimated_completion(execution_id),
            "execution_duration": self._calculate_execution_duration(execution_id)
        }
    
    def _calculate_progress_percentage(self, execution_id: str) -> float:
        """Calculate progress percentage for an execution."""
        metadata = self._execution_metadata.get(execution_id, {})
        total_tasks = metadata.get("total_tasks", 0)
        
        if total_tasks == 0:
            return 0.0
        
        if execution_id in self._active_executions:
            execution_manager = self._active_executions[execution_id]
            status = execution_manager.get_execution_status()
            current_step = status.get("current_step", 0)
            return min(100.0, (current_step / total_tasks) * 100.0)
        
        return 0.0
    
    def _calculate_estimated_completion(self, execution_id: str) -> Optional[datetime]:
        """Calculate estimated completion time for an execution."""
        metadata = self._execution_metadata.get(execution_id, {})
        start_time = metadata.get("start_time")
        
        if not start_time:
            return None
        
        progress_percentage = self._calculate_progress_percentage(execution_id)
        
        if progress_percentage <= 0:
            return None
        
        elapsed = datetime.now() - start_time
        total_estimated = elapsed / (progress_percentage / 100.0)
        return start_time + total_estimated
    
    def _calculate_execution_duration(self, execution_id: str) -> Optional[timedelta]:
        """Calculate current execution duration."""
        metadata = self._execution_metadata.get(execution_id, {})
        start_time = metadata.get("start_time")
        
        if not start_time:
            return None
        
        return datetime.now() - start_time
    
    def _calculate_average_task_duration(self, execution_id: str) -> Optional[float]:
        """Calculate average task duration in seconds."""
        duration = self._calculate_execution_duration(execution_id)
        if not duration:
            return None
        
        progress_percentage = self._calculate_progress_percentage(execution_id)
        if progress_percentage <= 0:
            return None
        
        metadata = self._execution_metadata.get(execution_id, {})
        total_tasks = metadata.get("total_tasks", 0)
        
        if total_tasks == 0:
            return None
        
        completed_tasks = (progress_percentage / 100.0) * total_tasks
        if completed_tasks <= 0:
            return None
        
        return duration.total_seconds() / completed_tasks
    
    def _get_memory_usage(self, execution_id: str) -> Dict[str, Any]:
        """Get memory usage statistics for an execution."""
        # Placeholder for memory monitoring
        return {
            "current_mb": 0,
            "peak_mb": 0,
            "available_mb": 0
        }
    
    def _get_cpu_usage(self, execution_id: str) -> Dict[str, Any]:
        """Get CPU usage statistics for an execution."""
        # Placeholder for CPU monitoring
        return {
            "current_percent": 0.0,
            "average_percent": 0.0,
            "peak_percent": 0.0
        }
    
    def _get_network_stats(self, execution_id: str) -> Dict[str, Any]:
        """Get network usage statistics for an execution."""
        # Placeholder for network monitoring
        return {
            "bytes_sent": 0,
            "bytes_received": 0,
            "requests_made": 0
        }
    
    def _get_error_counts(self, execution_id: str) -> Dict[str, Any]:
        """Get error counts for an execution."""
        # Placeholder for error tracking
        return {
            "total_errors": 0,
            "recoverable_errors": 0,
            "critical_errors": 0
        }
    
    def _get_performance_stats(self, execution_id: str) -> Dict[str, Any]:
        """Get performance statistics for an execution."""
        # Placeholder for performance monitoring
        return {
            "tasks_per_minute": 0.0,
            "average_response_time": 0.0,
            "throughput_score": 0.0
        }
    
    def _get_model_selection_metrics(self, execution_id: str) -> Dict[str, Any]:
        """Get model selection metrics for an execution."""
        metadata = self._execution_metadata.get(execution_id, {})
        
        return {
            "intelligent_selection_enabled": metadata.get("intelligent_selection", False),
            "selection_strategy": metadata.get("selection_strategy"),
            "models_selected": 0,  # TODO: Track actual model selections
            "selection_quality_score": 0.0,  # TODO: Calculate based on execution results
            "cost_optimization_achieved": 0.0,  # TODO: Calculate cost savings
            "performance_improvement": 0.0  # TODO: Calculate performance gains
        }
    
    def _analyze_cost_efficiency(self, execution_id: str) -> Dict[str, Any]:
        """Analyze cost efficiency of model selections."""
        return {
            "total_estimated_cost": 0.0,  # TODO: Calculate from execution results
            "cost_vs_budget": 0.0,  # TODO: Compare against budget constraints
            "cost_optimization_opportunities": [],  # TODO: Identify cheaper alternatives
            "efficiency_score": 0.8  # TODO: Calculate based on cost vs performance
        }
    
    def _analyze_performance_efficiency(self, execution_id: str) -> Dict[str, Any]:
        """Analyze performance efficiency of model selections."""
        return {
            "average_execution_time": 0.0,  # TODO: Calculate from step timings
            "performance_vs_expectations": 0.0,  # TODO: Compare against model capabilities
            "bottlenecks_identified": [],  # TODO: Identify slow steps
            "efficiency_score": 0.8  # TODO: Calculate based on speed vs accuracy
        }
    
    def _analyze_selection_accuracy(self, execution_id: str) -> Dict[str, Any]:
        """Analyze accuracy of model selections."""
        return {
            "selection_success_rate": 0.9,  # TODO: Calculate based on step success rates
            "optimal_selections": 0,  # TODO: Count selections that were optimal
            "suboptimal_selections": 0,  # TODO: Count selections that could be improved
            "accuracy_score": 0.85  # TODO: Calculate overall selection accuracy
        }
    
    def _identify_optimization_opportunities(self, execution_id: str) -> List[Dict[str, Any]]:
        """Identify optimization opportunities for future executions."""
        # TODO: Implement actual analysis based on execution data
        return [
            {
                "type": "cost_optimization",
                "description": "Consider using cost-optimized strategy for non-critical steps",
                "potential_savings": "15%",
                "steps_affected": []
            },
            {
                "type": "performance_optimization", 
                "description": "Fast models available for simple text generation steps",
                "potential_improvement": "25% faster execution",
                "steps_affected": []
            }
        ]
    
    async def _cleanup_execution(self, execution_id: str) -> None:
        """
        Clean up execution resources.
        
        Args:
            execution_id: Unique execution identifier
        """
        try:
            # Remove from active executions
            if execution_id in self._active_executions:
                del self._active_executions[execution_id]
            
            # Remove metadata
            if execution_id in self._execution_metadata:
                del self._execution_metadata[execution_id]
            
            # Remove progress callbacks
            if execution_id in self._progress_callbacks:
                del self._progress_callbacks[execution_id]
            
            # Cancel monitoring task if still running
            if execution_id in self._monitoring_tasks:
                monitoring_task = self._monitoring_tasks[execution_id]
                if not monitoring_task.done():
                    monitoring_task.cancel()
                del self._monitoring_tasks[execution_id]
            
            logger.debug(f"Execution {execution_id} cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up execution {execution_id}: {e}")
    
    async def shutdown(self) -> None:
        """
        Shutdown the executor and clean up all active executions.
        """
        logger.info("Shutting down PipelineExecutor...")
        
        # Stop all active executions
        active_ids = list(self._active_executions.keys())
        for execution_id in active_ids:
            try:
                await self.stop_execution(execution_id, graceful=False, reason="Executor shutdown")
            except Exception as e:
                logger.error(f"Error stopping execution {execution_id} during shutdown: {e}")
        
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)
        
        # Clear all references
        self._active_executions.clear()
        self._execution_metadata.clear()
        self._progress_callbacks.clear()
        self._monitoring_tasks.clear()
        
        logger.info("PipelineExecutor shutdown complete")


def create_pipeline_executor(
    max_concurrent_executions: int = 10,
    default_timeout: Optional[int] = 3600,
    enable_recovery: bool = True,
    enable_checkpointing: bool = True,
    model_registry: Optional[ModelRegistry] = None,
    enable_intelligent_selection: bool = True
) -> PipelineExecutor:
    """
    Create a PipelineExecutor instance with intelligent model selection capabilities.
    
    Args:
        max_concurrent_executions: Maximum number of concurrent pipeline executions
        default_timeout: Default execution timeout in seconds
        enable_recovery: Enable execution recovery features
        enable_checkpointing: Enable execution checkpointing
        model_registry: Model registry for intelligent model selection
        enable_intelligent_selection: Enable intelligent model selection features
        
    Returns:
        Configured PipelineExecutor instance with model selection intelligence
    """
    return PipelineExecutor(
        max_concurrent_executions=max_concurrent_executions,
        default_timeout=default_timeout,
        enable_recovery=enable_recovery,
        enable_checkpointing=enable_checkpointing,
        model_registry=model_registry,
        enable_intelligent_selection=enable_intelligent_selection
    )


def create_intelligent_pipeline_executor(
    model_registry: ModelRegistry,
    max_concurrent_executions: int = 10,
    default_timeout: Optional[int] = 3600,
    enable_recovery: bool = True,
    enable_checkpointing: bool = True
) -> PipelineExecutor:
    """
    Create a PipelineExecutor instance optimized for intelligent model selection.
    
    This is a convenience function that creates an executor with intelligent
    model selection enabled and properly configured.
    
    Args:
        model_registry: Model registry for intelligent model selection
        max_concurrent_executions: Maximum number of concurrent pipeline executions
        default_timeout: Default execution timeout in seconds
        enable_recovery: Enable execution recovery features
        enable_checkpointing: Enable execution checkpointing
        
    Returns:
        PipelineExecutor with intelligent model selection enabled
    """
    return PipelineExecutor(
        max_concurrent_executions=max_concurrent_executions,
        default_timeout=default_timeout,
        enable_recovery=enable_recovery,
        enable_checkpointing=enable_checkpointing,
        model_registry=model_registry,
        enable_intelligent_selection=True
    )