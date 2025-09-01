"""
Core validation engine with execution context integration.

This module provides the primary OutputQualityValidator that integrates with the
orchestrator's execution engine to provide real-time quality control validation
during pipeline execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

from .engine import ValidationEngine, ValidationSession, RuleExecutionContext
from .rules import (
    RuleRegistry, ValidationContext, RuleViolation, 
    RuleSeverity, RuleCategory
)
from ...execution.state import ExecutionContext, ExecutionStatus
from ...execution.progress import ProgressTracker, ProgressEvent, ProgressEventType

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Overall validation severity levels."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """
    Comprehensive validation result for pipeline output quality control.
    
    This result integrates with the execution context to provide actionable
    quality insights and supports Stream C reporting requirements.
    """
    session_id: str
    pipeline_id: str
    execution_id: str
    step_id: Optional[str]
    output_path: str
    
    # Overall results
    severity: ValidationSeverity
    passed: bool
    quality_score: float  # 0-100 score for Stream C analytics
    
    # Detailed metrics
    total_violations: int = 0
    violations_by_severity: Dict[str, int] = None
    violations_by_category: Dict[str, int] = None
    
    # Execution metrics
    validation_duration_ms: float = 0.0
    rules_executed: int = 0
    rules_failed: int = 0
    
    # Detailed data for Stream C
    violation_details: List[Dict[str, Any]] = None
    rule_results: List[Dict[str, Any]] = None
    recommendations: List[str] = None
    
    # Integration metadata
    execution_metrics: Dict[str, Any] = None
    pipeline_context: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default collections."""
        if self.violations_by_severity is None:
            self.violations_by_severity = {}
        if self.violations_by_category is None:
            self.violations_by_category = {}
        if self.violation_details is None:
            self.violation_details = []
        if self.rule_results is None:
            self.rule_results = []
        if self.recommendations is None:
            self.recommendations = []
        if self.execution_metrics is None:
            self.execution_metrics = {}
        if self.pipeline_context is None:
            self.pipeline_context = {}
    
    def has_critical_issues(self) -> bool:
        """Check if there are critical validation issues."""
        return self.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.FAIL]
    
    def has_warnings(self) -> bool:
        """Check if there are warning-level issues."""
        return self.severity == ValidationSeverity.WARNING or self.total_violations > 0
    
    def get_summary_for_stream_c(self) -> Dict[str, Any]:
        """Get validation summary optimized for Stream C reporting and analytics."""
        return {
            "validation_id": self.session_id,
            "pipeline_id": self.pipeline_id,
            "execution_id": self.execution_id,
            "step_id": self.step_id,
            "quality_metrics": {
                "quality_score": self.quality_score,
                "severity": self.severity.value,
                "passed": self.passed,
                "total_violations": self.total_violations,
                "violation_breakdown": self.violations_by_severity,
                "category_breakdown": self.violations_by_category
            },
            "performance_metrics": {
                "validation_duration_ms": self.validation_duration_ms,
                "rules_executed": self.rules_executed,
                "rules_failed": self.rules_failed,
                "success_rate": ((self.rules_executed - self.rules_failed) / max(1, self.rules_executed)) * 100
            },
            "execution_context": self.execution_metrics,
            "recommendations": self.recommendations,
            "timestamp": time.time()
        }


class OutputQualityValidator:
    """
    Primary output quality validator with execution engine integration.
    
    This validator provides automated quality control for pipeline outputs with:
    - Real-time integration with execution context
    - Configurable validation rules and thresholds
    - Quality metrics for Stream C reporting
    - Performance-aware validation execution
    - Comprehensive quality insights and recommendations
    """
    
    def __init__(
        self,
        rule_registry: Optional[RuleRegistry] = None,
        config_path: Optional[Union[str, Path]] = None,
        enable_real_time: bool = True,
        quality_threshold: float = 70.0
    ):
        """
        Initialize output quality validator.
        
        Args:
            rule_registry: Custom rule registry (creates default if None)
            config_path: Optional path to validation configuration file
            enable_real_time: Whether to enable real-time validation hooks
            quality_threshold: Minimum quality score threshold (0-100)
        """
        # Initialize core components
        self.rule_registry = rule_registry or RuleRegistry()
        self.validation_engine = ValidationEngine(rule_registry=self.rule_registry)
        self.quality_threshold = quality_threshold
        self.enable_real_time = enable_real_time
        
        # Load configuration if provided
        if config_path:
            self.load_configuration(config_path)
        
        # Execution context integration
        self.execution_contexts: Dict[str, ExecutionContext] = {}
        self.validation_sessions: Dict[str, ValidationSession] = {}
        
        # Real-time validation state
        self.real_time_handlers: List[Callable[[ValidationResult], None]] = []
        self.quality_alerts: List[Callable[[ValidationResult], None]] = []
        
        # Setup execution hooks for real-time validation
        if self.enable_real_time:
            self._setup_execution_hooks()
        
        logger.info(f"Initialized OutputQualityValidator with {len(self.rule_registry.rules)} rules")
    
    def load_configuration(self, config_path: Union[str, Path]) -> None:
        """Load validator configuration from file."""
        try:
            self.rule_registry.load_rules_from_config(config_path)
            logger.info(f"Loaded validation configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def register_execution_context(self, execution_context: ExecutionContext) -> None:
        """
        Register an execution context for real-time validation.
        
        Args:
            execution_context: Execution context to monitor
        """
        self.execution_contexts[execution_context.execution_id] = execution_context
        
        if self.enable_real_time:
            # Add validation hooks to the execution context
            execution_context.add_step_handler(self._on_step_completed)
            execution_context.add_status_handler(self._on_execution_status_changed)
        
        logger.debug(f"Registered execution context: {execution_context.execution_id}")
    
    def validate_output(
        self,
        execution_context: ExecutionContext,
        output_path: Union[str, Path],
        output_content: Optional[Any] = None,
        rule_filters: Optional[List[str]] = None,
        category_filters: Optional[List[RuleCategory]] = None,
        real_time: bool = True
    ) -> ValidationResult:
        """
        Validate pipeline output with comprehensive quality control.
        
        Args:
            execution_context: Current execution context
            output_path: Path to output file/directory
            output_content: Optional pre-loaded content
            rule_filters: Optional list of specific rules to execute
            category_filters: Optional list of rule categories to execute
            real_time: Whether this is a real-time validation (affects performance)
            
        Returns:
            ValidationResult with comprehensive quality assessment
        """
        start_time = time.time()
        output_path = str(output_path)
        
        logger.info(f"Starting output validation for {output_path}")
        
        try:
            # Execute validation session
            session = self.validation_engine.validate_output(
                execution_context=execution_context,
                output_path=output_path,
                output_content=output_content,
                rule_filters=rule_filters,
                category_filters=category_filters,
                parallel=not real_time  # Use sequential for real-time to reduce overhead
            )
            
            # Store session for tracking
            self.validation_sessions[session.session_id] = session
            
            # Convert session to ValidationResult
            result = self._create_validation_result(session, execution_context, output_path)
            
            # Execute real-time handlers
            if real_time and self.enable_real_time:
                self._execute_real_time_handlers(result)
            
            logger.info(
                f"Completed validation for {output_path}: "
                f"score={result.quality_score:.1f}, violations={result.total_violations}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed for {output_path}: {e}")
            # Create error result
            return ValidationResult(
                session_id=f"error_{int(time.time())}",
                pipeline_id=execution_context.pipeline_id,
                execution_id=execution_context.execution_id,
                step_id=execution_context.current_step_id,
                output_path=output_path,
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                quality_score=0.0,
                validation_duration_ms=(time.time() - start_time) * 1000,
                recommendations=[f"Validation system error: {str(e)}"]
            )
    
    def validate_pipeline_outputs(
        self,
        execution_context: ExecutionContext,
        output_paths: List[Union[str, Path]],
        batch_size: int = 5
    ) -> List[ValidationResult]:
        """
        Validate multiple pipeline outputs in batches.
        
        Args:
            execution_context: Current execution context
            output_paths: List of output paths to validate
            batch_size: Number of outputs to validate concurrently
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        logger.info(f"Starting batch validation of {len(output_paths)} outputs")
        
        # Process in batches to manage resource usage
        for i in range(0, len(output_paths), batch_size):
            batch = output_paths[i:i + batch_size]
            batch_results = []
            
            for output_path in batch:
                result = self.validate_output(
                    execution_context=execution_context,
                    output_path=output_path,
                    real_time=False  # Batch mode - use parallel validation
                )
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Log batch progress
            logger.info(f"Completed batch {i//batch_size + 1}/{(len(output_paths) + batch_size - 1)//batch_size}")
        
        return results
    
    def get_quality_metrics(self, execution_id: str) -> Dict[str, Any]:
        """
        Get aggregated quality metrics for an execution.
        
        Args:
            execution_id: Execution ID to get metrics for
            
        Returns:
            Aggregated quality metrics for Stream C analytics
        """
        # Find all validation sessions for this execution
        execution_sessions = [
            session for session in self.validation_sessions.values()
            if session.session_id.startswith(f"validation_{execution_id}")
        ]
        
        if not execution_sessions:
            return {
                "execution_id": execution_id,
                "total_validations": 0,
                "average_quality_score": 0.0,
                "total_violations": 0,
                "validation_coverage": 0.0
            }
        
        # Aggregate metrics
        total_validations = len(execution_sessions)
        total_violations = sum(session.total_violations for session in execution_sessions)
        total_rules_executed = sum(session.rules_executed for session in execution_sessions)
        total_duration_ms = sum(session.duration_ms for session in execution_sessions)
        
        # Calculate average quality score
        quality_scores = [
            self.validation_engine._calculate_quality_score(session)
            for session in execution_sessions
        ]
        average_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return {
            "execution_id": execution_id,
            "total_validations": total_validations,
            "average_quality_score": average_quality_score,
            "total_violations": total_violations,
            "violations_per_validation": total_violations / max(1, total_validations),
            "average_validation_time_ms": total_duration_ms / max(1, total_validations),
            "rules_executed": total_rules_executed,
            "validation_coverage": 100.0,  # Assume full coverage for now
            "quality_trend": "stable",  # Could implement trend analysis
            "timestamp": time.time()
        }
    
    def add_real_time_handler(self, handler: Callable[[ValidationResult], None]) -> None:
        """Add handler for real-time validation results."""
        self.real_time_handlers.append(handler)
    
    def add_quality_alert_handler(self, handler: Callable[[ValidationResult], None]) -> None:
        """Add handler for quality threshold alerts."""
        self.quality_alerts.append(handler)
    
    def set_quality_threshold(self, threshold: float) -> None:
        """Set minimum quality score threshold."""
        self.quality_threshold = max(0.0, min(100.0, threshold))
        logger.info(f"Set quality threshold to {self.quality_threshold}")
    
    def _create_validation_result(
        self,
        session: ValidationSession,
        execution_context: ExecutionContext,
        output_path: str
    ) -> ValidationResult:
        """Convert validation session to ValidationResult."""
        # Get validation summary
        summary = self.validation_engine.get_validation_summary(session)
        
        # Determine overall severity
        severity = self._determine_severity(session)
        
        # Extract detailed violation information for Stream C
        violation_details = []
        rule_results = []
        
        for result in session.rule_results:
            rule_data = {
                "rule_id": result.rule_id,
                "rule_name": result.rule_name,
                "success": result.success,
                "execution_time_ms": result.execution_time_ms,
                "violation_count": len(result.violations)
            }
            rule_results.append(rule_data)
            
            for violation in result.violations:
                violation_data = {
                    "rule_id": violation.rule_id,
                    "message": violation.message,
                    "severity": violation.severity.value,
                    "category": violation.category.value,
                    "file_path": violation.file_path,
                    "line_number": violation.line_number,
                    "metadata": violation.metadata
                }
                violation_details.append(violation_data)
        
        # Extract execution context metrics
        execution_metrics = {}
        if execution_context.metrics:
            execution_metrics = {
                "execution_duration_s": execution_context.metrics.duration.total_seconds() if execution_context.metrics.duration else 0,
                "steps_completed": execution_context.metrics.steps_completed,
                "steps_failed": execution_context.metrics.steps_failed,
                "memory_peak_mb": execution_context.metrics.memory_peak_mb,
                "cpu_time_seconds": execution_context.metrics.cpu_time_seconds
            }
        
        return ValidationResult(
            session_id=session.session_id,
            pipeline_id=execution_context.pipeline_id,
            execution_id=execution_context.execution_id,
            step_id=execution_context.current_step_id,
            output_path=output_path,
            severity=severity,
            passed=severity not in [ValidationSeverity.FAIL, ValidationSeverity.CRITICAL],
            quality_score=summary["quality_score"],
            total_violations=session.total_violations,
            violations_by_severity={k: v for k, v in summary["violations_by_severity"].items()},
            violations_by_category=summary["violations_by_category"],
            validation_duration_ms=session.duration_ms,
            rules_executed=session.rules_executed,
            rules_failed=session.rules_failed,
            violation_details=violation_details,
            rule_results=rule_results,
            recommendations=summary["recommendations"],
            execution_metrics=execution_metrics,
            pipeline_context={
                "pipeline_id": execution_context.pipeline_id,
                "execution_status": execution_context.status.value if execution_context.status else "unknown"
            }
        )
    
    def _determine_severity(self, session: ValidationSession) -> ValidationSeverity:
        """Determine overall validation severity."""
        # Check for critical violations
        if session.violations_by_severity.get(RuleSeverity.CRITICAL, 0) > 0:
            return ValidationSeverity.CRITICAL
        
        # Check for errors
        if session.violations_by_severity.get(RuleSeverity.ERROR, 0) > 0:
            return ValidationSeverity.FAIL
        
        # Check for warnings
        if session.violations_by_severity.get(RuleSeverity.WARNING, 0) > 0:
            return ValidationSeverity.WARNING
        
        return ValidationSeverity.PASS
    
    def _execute_real_time_handlers(self, result: ValidationResult) -> None:
        """Execute real-time validation handlers."""
        try:
            # Execute general handlers
            for handler in self.real_time_handlers:
                try:
                    handler(result)
                except Exception as e:
                    logger.error(f"Real-time handler failed: {e}")
            
            # Execute quality alert handlers if threshold breached
            if result.quality_score < self.quality_threshold:
                for handler in self.quality_alerts:
                    try:
                        handler(result)
                    except Exception as e:
                        logger.error(f"Quality alert handler failed: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to execute real-time handlers: {e}")
    
    def _setup_execution_hooks(self) -> None:
        """Setup hooks for real-time execution integration."""
        # Add validation engine hooks
        self.validation_engine.add_hook('after_validation', self._on_validation_completed)
        self.validation_engine.add_hook('on_violation', self._on_violation_detected)
    
    def _on_step_completed(self, execution_context: ExecutionContext, step_id: str, **kwargs) -> None:
        """Handle step completion for real-time validation."""
        # This could trigger automatic output validation
        # Implementation depends on step output tracking
        logger.debug(f"Step completed: {step_id} in execution {execution_context.execution_id}")
    
    def _on_execution_status_changed(self, execution_context: ExecutionContext, **kwargs) -> None:
        """Handle execution status changes."""
        logger.debug(f"Execution status changed: {execution_context.status}")
    
    def _on_validation_completed(self, context: RuleExecutionContext, **kwargs) -> None:
        """Handle validation completion."""
        logger.debug(f"Validation completed for session {context.session.session_id}")
    
    def _on_violation_detected(self, context: RuleExecutionContext, violation: RuleViolation, **kwargs) -> None:
        """Handle violation detection."""
        # Could implement real-time violation alerts here
        if violation.severity in [RuleSeverity.CRITICAL, RuleSeverity.ERROR]:
            logger.warning(f"Quality violation detected: {violation.message}")
    
    def cleanup_session(self, session_id: str) -> bool:
        """Clean up validation session data."""
        if session_id in self.validation_sessions:
            del self.validation_sessions[session_id]
            return True
        return False
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get overall validation statistics."""
        total_sessions = len(self.validation_sessions)
        total_violations = sum(session.total_violations for session in self.validation_sessions.values())
        
        if total_sessions == 0:
            return {
                "total_sessions": 0,
                "average_violations": 0.0,
                "average_quality_score": 0.0,
                "rule_usage": {}
            }
        
        # Calculate averages
        quality_scores = [
            self.validation_engine._calculate_quality_score(session)
            for session in self.validation_sessions.values()
        ]
        average_quality_score = sum(quality_scores) / len(quality_scores)
        
        # Rule usage statistics
        rule_usage = {}
        for session in self.validation_sessions.values():
            for result in session.rule_results:
                rule_id = result.rule_id
                if rule_id not in rule_usage:
                    rule_usage[rule_id] = {"executions": 0, "violations": 0}
                rule_usage[rule_id]["executions"] += 1
                rule_usage[rule_id]["violations"] += len(result.violations)
        
        return {
            "total_sessions": total_sessions,
            "total_violations": total_violations,
            "average_violations": total_violations / total_sessions,
            "average_quality_score": average_quality_score,
            "rule_usage": rule_usage,
            "active_execution_contexts": len(self.execution_contexts)
        }