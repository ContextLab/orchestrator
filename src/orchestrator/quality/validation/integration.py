"""
Real-time integration with pipeline execution for automated quality control.

This module provides seamless integration between the validation system and
the orchestrator's execution engine, enabling automatic quality checks
during pipeline execution.
"""

from __future__ import annotations

import logging
import asyncio
import threading
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass
from pathlib import Path

from .validator import OutputQualityValidator, ValidationResult, ValidationSeverity
from .rules import RuleCategory
from ...execution.state import ExecutionContext, ExecutionStatus
from ...execution.progress import ProgressTracker, ProgressEvent, ProgressEventType

logger = logging.getLogger(__name__)


@dataclass
class ValidationTrigger:
    """Configuration for when validation should be triggered."""
    on_step_completion: bool = True
    on_execution_completion: bool = True
    on_output_creation: bool = True
    on_checkpoint: bool = False
    step_filter: Optional[Set[str]] = None  # Only validate specific steps
    output_pattern_filter: Optional[str] = None  # Only validate matching paths


class ExecutionQualityMonitor:
    """
    Real-time quality monitor that integrates with pipeline execution.
    
    This monitor automatically validates outputs as they are created during
    pipeline execution, providing continuous quality assurance.
    """
    
    def __init__(
        self,
        validator: OutputQualityValidator,
        trigger_config: Optional[ValidationTrigger] = None
    ):
        """
        Initialize execution quality monitor.
        
        Args:
            validator: Quality validator instance
            trigger_config: Configuration for validation triggers
        """
        self.validator = validator
        self.trigger_config = trigger_config or ValidationTrigger()
        
        # Monitoring state
        self.monitored_executions: Dict[str, ExecutionContext] = {}
        self.validation_results: Dict[str, List[ValidationResult]] = {}
        self.quality_alerts: List[ValidationResult] = []
        
        # Event handlers
        self.quality_handlers: List[Callable[[str, ValidationResult], None]] = []
        self.alert_handlers: List[Callable[[ValidationResult], None]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Initialized ExecutionQualityMonitor")
    
    def start_monitoring(self, execution_context: ExecutionContext) -> None:
        """
        Start monitoring an execution context for quality control.
        
        Args:
            execution_context: Execution context to monitor
        """
        with self._lock:
            execution_id = execution_context.execution_id
            self.monitored_executions[execution_id] = execution_context
            self.validation_results[execution_id] = []
            
            # Register with validator
            self.validator.register_execution_context(execution_context)
            
            # Add our handlers to execution context
            execution_context.add_step_handler(self._on_step_completed)
            execution_context.add_status_handler(self._on_status_changed)
            
            logger.info(f"Started monitoring execution: {execution_id}")
    
    def stop_monitoring(self, execution_id: str) -> None:
        """
        Stop monitoring an execution context.
        
        Args:
            execution_id: Execution ID to stop monitoring
        """
        with self._lock:
            if execution_id in self.monitored_executions:
                del self.monitored_executions[execution_id]
                
                # Keep validation results for reporting
                logger.info(f"Stopped monitoring execution: {execution_id}")
    
    def add_quality_handler(
        self, 
        handler: Callable[[str, ValidationResult], None]
    ) -> None:
        """Add handler for quality validation results."""
        self.quality_handlers.append(handler)
    
    def add_alert_handler(self, handler: Callable[[ValidationResult], None]) -> None:
        """Add handler for quality alerts."""
        self.alert_handlers.append(handler)
    
    def get_execution_quality_summary(self, execution_id: str) -> Dict[str, Any]:
        """
        Get quality summary for a specific execution.
        
        Args:
            execution_id: Execution ID to get summary for
            
        Returns:
            Quality summary dictionary
        """
        results = self.validation_results.get(execution_id, [])
        
        if not results:
            return {
                "execution_id": execution_id,
                "total_validations": 0,
                "average_quality_score": 0.0,
                "quality_trend": "no_data"
            }
        
        # Calculate metrics
        total_validations = len(results)
        quality_scores = [r.quality_score for r in results]
        average_score = sum(quality_scores) / len(quality_scores)
        
        # Count by severity
        severity_counts = {
            "pass": sum(1 for r in results if r.severity == ValidationSeverity.PASS),
            "warning": sum(1 for r in results if r.severity == ValidationSeverity.WARNING),
            "fail": sum(1 for r in results if r.severity == ValidationSeverity.FAIL),
            "critical": sum(1 for r in results if r.severity == ValidationSeverity.CRITICAL)
        }
        
        # Determine trend (simplified)
        trend = "stable"
        if len(quality_scores) >= 2:
            recent_avg = sum(quality_scores[-3:]) / len(quality_scores[-3:])
            early_avg = sum(quality_scores[:3]) / min(3, len(quality_scores))
            if recent_avg > early_avg + 10:
                trend = "improving"
            elif recent_avg < early_avg - 10:
                trend = "declining"
        
        return {
            "execution_id": execution_id,
            "total_validations": total_validations,
            "average_quality_score": average_score,
            "severity_distribution": severity_counts,
            "quality_trend": trend,
            "alerts_count": sum(1 for r in results if r.has_critical_issues()),
            "latest_validation": results[-1].get_summary_for_stream_c() if results else None
        }
    
    def _on_step_completed(
        self, 
        execution_context: ExecutionContext, 
        step_id: str, 
        **kwargs
    ) -> None:
        """Handle step completion event."""
        if not self.trigger_config.on_step_completion:
            return
            
        # Check step filter
        if (self.trigger_config.step_filter and 
            step_id not in self.trigger_config.step_filter):
            return
        
        logger.debug(f"Step completed: {step_id}, checking for outputs to validate")
        
        # Get step outputs (this would need integration with step output tracking)
        step_outputs = self._get_step_outputs(execution_context, step_id)
        
        # Validate each output
        for output_path in step_outputs:
            self._validate_output(execution_context, output_path, step_id)
    
    def _on_status_changed(self, execution_context: ExecutionContext, **kwargs) -> None:
        """Handle execution status change."""
        if (execution_context.status == ExecutionStatus.COMPLETED and 
            self.trigger_config.on_execution_completion):
            
            logger.info(f"Execution completed: {execution_context.execution_id}, running final validation")
            self._run_final_validation(execution_context)
    
    def _get_step_outputs(
        self, 
        execution_context: ExecutionContext, 
        step_id: str
    ) -> List[str]:
        """
        Get output paths for a completed step.
        
        This is a placeholder - in a real implementation, this would integrate
        with the orchestrator's output tracking system.
        """
        # This would integrate with the actual output tracking system
        # For now, return empty list as placeholder
        return []
    
    def _validate_output(
        self, 
        execution_context: ExecutionContext, 
        output_path: str,
        step_id: Optional[str] = None
    ) -> None:
        """Validate a single output and handle the result."""
        try:
            # Apply output pattern filter if configured
            if (self.trigger_config.output_pattern_filter and 
                not self._matches_pattern(output_path, self.trigger_config.output_pattern_filter)):
                return
            
            # Run validation
            result = self.validator.validate_output(
                execution_context=execution_context,
                output_path=output_path,
                real_time=True
            )
            
            # Store result
            execution_id = execution_context.execution_id
            with self._lock:
                if execution_id not in self.validation_results:
                    self.validation_results[execution_id] = []
                self.validation_results[execution_id].append(result)
            
            # Execute handlers
            self._execute_quality_handlers(execution_id, result)
            
            # Check for alerts
            if result.has_critical_issues() or result.quality_score < self.validator.quality_threshold:
                self.quality_alerts.append(result)
                self._execute_alert_handlers(result)
                
                logger.warning(
                    f"Quality alert for {output_path}: "
                    f"score={result.quality_score:.1f}, severity={result.severity.value}"
                )
            
        except Exception as e:
            logger.error(f"Failed to validate output {output_path}: {e}")
    
    def _run_final_validation(self, execution_context: ExecutionContext) -> None:
        """Run final comprehensive validation for completed execution."""
        try:
            # Get all execution outputs (placeholder)
            all_outputs = self._get_execution_outputs(execution_context)
            
            if all_outputs:
                # Run batch validation
                results = self.validator.validate_pipeline_outputs(
                    execution_context=execution_context,
                    output_paths=all_outputs
                )
                
                # Store results
                execution_id = execution_context.execution_id
                with self._lock:
                    if execution_id not in self.validation_results:
                        self.validation_results[execution_id] = []
                    self.validation_results[execution_id].extend(results)
                
                # Process results
                for result in results:
                    self._execute_quality_handlers(execution_id, result)
                    
                    if result.has_critical_issues():
                        self.quality_alerts.append(result)
                        self._execute_alert_handlers(result)
                
                logger.info(f"Final validation completed for {execution_id}: {len(results)} outputs validated")
            
        except Exception as e:
            logger.error(f"Final validation failed for {execution_context.execution_id}: {e}")
    
    def _get_execution_outputs(self, execution_context: ExecutionContext) -> List[str]:
        """Get all output paths for an execution (placeholder)."""
        # This would integrate with the actual output tracking system
        return []
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches pattern."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)
    
    def _execute_quality_handlers(self, execution_id: str, result: ValidationResult) -> None:
        """Execute quality result handlers."""
        for handler in self.quality_handlers:
            try:
                handler(execution_id, result)
            except Exception as e:
                logger.error(f"Quality handler failed: {e}")
    
    def _execute_alert_handlers(self, result: ValidationResult) -> None:
        """Execute alert handlers."""
        for handler in self.alert_handlers:
            try:
                handler(result)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get overall monitoring statistics."""
        with self._lock:
            total_executions = len(self.validation_results)
            total_validations = sum(len(results) for results in self.validation_results.values())
            total_alerts = len(self.quality_alerts)
            
            # Calculate average quality score across all validations
            all_results = [
                result for results in self.validation_results.values() 
                for result in results
            ]
            
            if all_results:
                avg_quality_score = sum(r.quality_score for r in all_results) / len(all_results)
                severity_distribution = {
                    "pass": sum(1 for r in all_results if r.severity == ValidationSeverity.PASS),
                    "warning": sum(1 for r in all_results if r.severity == ValidationSeverity.WARNING),
                    "fail": sum(1 for r in all_results if r.severity == ValidationSeverity.FAIL),
                    "critical": sum(1 for r in all_results if r.severity == ValidationSeverity.CRITICAL)
                }
            else:
                avg_quality_score = 0.0
                severity_distribution = {"pass": 0, "warning": 0, "fail": 0, "critical": 0}
            
            return {
                "monitored_executions": len(self.monitored_executions),
                "total_executions_processed": total_executions,
                "total_validations": total_validations,
                "total_quality_alerts": total_alerts,
                "average_quality_score": avg_quality_score,
                "severity_distribution": severity_distribution,
                "active_handlers": {
                    "quality_handlers": len(self.quality_handlers),
                    "alert_handlers": len(self.alert_handlers)
                }
            }


class QualityControlManager:
    """
    High-level manager for orchestrator quality control integration.
    
    This manager coordinates the validation system with the execution engine
    and provides a unified interface for quality control operations.
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize quality control manager.
        
        Args:
            config_path: Path to validation configuration file
            enable_monitoring: Whether to enable real-time monitoring
        """
        # Initialize core components
        self.validator = OutputQualityValidator(config_path=config_path)
        self.monitor = ExecutionQualityMonitor(
            validator=self.validator
        ) if enable_monitoring else None
        
        # Integration state
        self.registered_executions: Set[str] = set()
        
        logger.info("Initialized QualityControlManager")
    
    def register_execution_context(
        self, 
        execution_context: ExecutionContext,
        enable_monitoring: bool = True
    ) -> None:
        """
        Register an execution context for quality control.
        
        Args:
            execution_context: Execution context to register
            enable_monitoring: Whether to enable real-time monitoring
        """
        execution_id = execution_context.execution_id
        
        if execution_id in self.registered_executions:
            logger.warning(f"Execution {execution_id} already registered")
            return
        
        # Register with validator
        self.validator.register_execution_context(execution_context)
        
        # Start monitoring if enabled
        if enable_monitoring and self.monitor:
            self.monitor.start_monitoring(execution_context)
        
        self.registered_executions.add(execution_id)
        logger.info(f"Registered execution context: {execution_id}")
    
    def validate_output(
        self,
        execution_context: ExecutionContext,
        output_path: Path,
        **kwargs
    ) -> ValidationResult:
        """Validate a specific output."""
        return self.validator.validate_output(
            execution_context=execution_context,
            output_path=output_path,
            **kwargs
        )
    
    def get_quality_metrics(self, execution_id: str) -> Dict[str, Any]:
        """Get comprehensive quality metrics for an execution."""
        # Get validator metrics
        validator_metrics = self.validator.get_quality_metrics(execution_id)
        
        # Get monitor metrics if available
        monitor_metrics = {}
        if self.monitor:
            monitor_metrics = self.monitor.get_execution_quality_summary(execution_id)
        
        return {
            "execution_id": execution_id,
            "validator_metrics": validator_metrics,
            "monitor_metrics": monitor_metrics,
            "overall_quality_score": monitor_metrics.get("average_quality_score", 0.0),
            "quality_status": self._determine_quality_status(monitor_metrics),
            "timestamp": validator_metrics.get("timestamp", 0)
        }
    
    def _determine_quality_status(self, monitor_metrics: Dict[str, Any]) -> str:
        """Determine overall quality status."""
        if not monitor_metrics:
            return "unknown"
        
        severity_dist = monitor_metrics.get("severity_distribution", {})
        
        if severity_dist.get("critical", 0) > 0:
            return "critical"
        elif severity_dist.get("fail", 0) > 0:
            return "failed"
        elif severity_dist.get("warning", 0) > 0:
            return "warning"
        else:
            return "passed"
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        validator_stats = self.validator.get_validation_statistics()
        
        monitor_stats = {}
        if self.monitor:
            monitor_stats = self.monitor.get_monitoring_statistics()
        
        return {
            "registered_executions": len(self.registered_executions),
            "validator_statistics": validator_stats,
            "monitor_statistics": monitor_stats
        }
    
    def cleanup_execution(self, execution_id: str) -> None:
        """Clean up resources for a completed execution."""
        if execution_id in self.registered_executions:
            self.registered_executions.remove(execution_id)
        
        if self.monitor:
            self.monitor.stop_monitoring(execution_id)
        
        logger.info(f"Cleaned up execution: {execution_id}")


# Factory function for easy integration
def create_quality_control_manager(
    config_path: Optional[Path] = None,
    enable_monitoring: bool = True
) -> QualityControlManager:
    """
    Create and configure a quality control manager.
    
    Args:
        config_path: Optional path to validation configuration
        enable_monitoring: Whether to enable real-time monitoring
        
    Returns:
        Configured QualityControlManager instance
    """
    return QualityControlManager(
        config_path=config_path,
        enable_monitoring=enable_monitoring
    )