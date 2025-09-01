"""
Integration module for quality logging with existing validation system.

This module provides seamless integration between the comprehensive logging
framework and the existing validation system components, ensuring that all
quality events, validation results, and performance metrics are properly
logged and monitored.

Key Integration Points:
- ValidationResult logging with structured data
- Quality event generation from validation events
- Performance metric collection during validation
- Alert generation for quality threshold breaches
- Context propagation from execution engine
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import asdict
from pathlib import Path

from .logger import StructuredLogger, LogLevel, LogCategory, QualityEvent, get_logger
from .monitoring import QualityMonitor, MonitoringAlert, create_monitoring_setup
from ..validation.validator import ValidationResult, ValidationSeverity, OutputQualityValidator
from ..validation.rules import RuleViolation, RuleSeverity, ValidationContext
from ..validation.engine import ValidationEngine, ValidationSession
from ..validation.integration import ExecutionQualityMonitor, QualityControlManager
from ...execution.state import ExecutionContext, ExecutionStatus
from ...execution.progress import ProgressTracker, ProgressEvent, ProgressEventType


class QualityLoggingIntegrator:
    """
    Integrates quality logging with validation system components.
    
    Provides comprehensive logging for validation events, quality metrics,
    and performance data while maintaining compatibility with existing
    validation system interfaces.
    """
    
    def __init__(
        self,
        logger: Optional[StructuredLogger] = None,
        monitor: Optional[QualityMonitor] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.logger = logger or get_logger("quality_integration")
        self.monitor = monitor
        self.config = config or {}
        
        # Track validation sessions and metrics
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._quality_metrics: Dict[str, Any] = {}
        
        # Integration state
        self._execution_contexts: Dict[str, ExecutionContext] = {}
        self._progress_trackers: Dict[str, ProgressTracker] = {}

    async def start(self):
        """Start quality logging integration."""
        self.logger.info("Starting quality logging integration", category=LogCategory.SYSTEM)
        
        if self.monitor:
            await self.monitor.start()
            self.logger.info("Quality monitoring system started", category=LogCategory.MONITORING)

    async def stop(self):
        """Stop quality logging integration gracefully."""
        self.logger.info("Stopping quality logging integration", category=LogCategory.SYSTEM)
        
        # Complete any active validation sessions
        for session_id in list(self._active_sessions.keys()):
            await self.complete_validation_session(session_id, forced=True)
        
        if self.monitor:
            await self.monitor.stop()
            self.logger.info("Quality monitoring system stopped", category=LogCategory.MONITORING)

    # Validation System Integration
    async def start_validation_session(
        self,
        session_id: str,
        pipeline_id: str,
        execution_id: str,
        context: ValidationContext
    ) -> None:
        """Start tracking a validation session."""
        session_data = {
            'session_id': session_id,
            'pipeline_id': pipeline_id,
            'execution_id': execution_id,
            'start_time': time.time(),
            'context': context,
            'rules_executed': 0,
            'violations_found': 0,
            'quality_score': 0.0
        }
        
        self._active_sessions[session_id] = session_data
        
        # Log session start with context
        with self.logger.context(
            execution_id=execution_id,
            pipeline_id=pipeline_id,
            session_id=session_id
        ):
            self.logger.info(
                f"Validation session started: {session_id}",
                category=LogCategory.VALIDATION,
                metadata={
                    'validation_context': asdict(context) if hasattr(context, '__dict__') else str(context),
                    'session_type': 'validation'
                }
            )
            
            # Generate quality event
            if self.monitor:
                quality_event = QualityEvent(
                    event_type="validation_session_started",
                    severity="INFO",
                    source_component="validation_integrator",
                    validation_result={
                        'session_id': session_id,
                        'pipeline_id': pipeline_id,
                        'execution_id': execution_id
                    }
                )
                await self.monitor.record_quality_event(quality_event)

    async def log_validation_result(
        self,
        session_id: str,
        result: ValidationResult
    ) -> None:
        """Log a validation result with comprehensive quality metrics."""
        if session_id not in self._active_sessions:
            self.logger.warning(f"Validation result received for unknown session: {session_id}")
            return
        
        session_data = self._active_sessions[session_id]
        
        # Update session metrics
        session_data['rules_executed'] += 1
        if result.violations:
            session_data['violations_found'] += len(result.violations)
        session_data['quality_score'] = result.quality_score
        
        # Log with full context
        with self.logger.context(
            execution_id=session_data['execution_id'],
            pipeline_id=session_data['pipeline_id'],
            session_id=session_id,
            step_id=result.step_id
        ):
            # Determine log level based on severity
            log_level = self._map_validation_severity_to_log_level(result.severity)
            
            self.logger.log_validation_result(
                {
                    'session_id': session_id,
                    'severity': result.severity.value,
                    'quality_score': result.quality_score,
                    'violations': [asdict(v) for v in result.violations],
                    'performance_metrics': result.performance_metrics,
                    'recommendations': result.recommendations,
                    'output_path': result.output_path
                },
                level=log_level,
                validation_score=result.quality_score,
                rule_violations=len(result.violations) if result.violations else 0
            )
            
            # Generate quality event for monitoring
            if self.monitor:
                quality_event = QualityEvent(
                    event_type="validation_result_generated",
                    severity=result.severity.value,
                    source_component="validation_integrator",
                    validation_result=asdict(result),
                    quality_score=result.quality_score,
                    rule_violations=[asdict(v) for v in result.violations] if result.violations else None,
                    recommendations=result.recommendations
                )
                await self.monitor.record_quality_event(quality_event)
                
                # Record metrics
                await self.monitor.record_metric(
                    "orchestrator_validation_quality_score",
                    result.quality_score,
                    {
                        'pipeline_id': session_data['pipeline_id'],
                        'execution_id': session_data['execution_id'],
                        'severity': result.severity.value
                    }
                )
                
                if result.violations:
                    await self.monitor.record_metric(
                        "orchestrator_validation_violations",
                        len(result.violations),
                        {
                            'pipeline_id': session_data['pipeline_id'],
                            'execution_id': session_data['execution_id'],
                            'severity': result.severity.value
                        }
                    )

    async def log_rule_violation(
        self,
        session_id: str,
        violation: RuleViolation,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a specific rule violation with detailed context."""
        if session_id not in self._active_sessions:
            self.logger.warning(f"Rule violation received for unknown session: {session_id}")
            return
        
        session_data = self._active_sessions[session_id]
        
        with self.logger.context(
            execution_id=session_data['execution_id'],
            pipeline_id=session_data['pipeline_id'],
            session_id=session_id
        ):
            log_level = self._map_rule_severity_to_log_level(violation.severity)
            
            self.logger.log(
                log_level,
                f"Rule violation detected: {violation.rule_name}",
                category=LogCategory.VALIDATION,
                metadata={
                    'rule_violation': asdict(violation),
                    'violation_context': context,
                    'remediation_suggestions': getattr(violation, 'remediation_suggestions', [])
                }
            )
            
            # Generate quality event
            if self.monitor:
                quality_event = QualityEvent(
                    event_type="rule_violation_detected",
                    severity=violation.severity.value,
                    source_component="validation_integrator",
                    rule_violations=[asdict(violation)],
                    recommendations=getattr(violation, 'remediation_suggestions', [])
                )
                await self.monitor.record_quality_event(quality_event)

    async def complete_validation_session(
        self,
        session_id: str,
        forced: bool = False
    ) -> None:
        """Complete a validation session and generate summary metrics."""
        if session_id not in self._active_sessions:
            if not forced:
                self.logger.warning(f"Attempted to complete unknown session: {session_id}")
            return
        
        session_data = self._active_sessions.pop(session_id)
        session_duration = time.time() - session_data['start_time']
        
        with self.logger.context(
            execution_id=session_data['execution_id'],
            pipeline_id=session_data['pipeline_id'],
            session_id=session_id
        ):
            self.logger.info(
                f"Validation session completed: {session_id}",
                category=LogCategory.VALIDATION,
                duration_ms=session_duration * 1000,
                metadata={
                    'session_summary': {
                        'duration_seconds': session_duration,
                        'rules_executed': session_data['rules_executed'],
                        'violations_found': session_data['violations_found'],
                        'final_quality_score': session_data['quality_score']
                    }
                }
            )
            
            # Record performance metrics
            if self.monitor:
                await self.monitor.record_metric(
                    "orchestrator_validation_session_duration",
                    session_duration,
                    {
                        'pipeline_id': session_data['pipeline_id'],
                        'execution_id': session_data['execution_id']
                    }
                )
                
                await self.monitor.record_metric(
                    "orchestrator_validation_rules_executed",
                    session_data['rules_executed'],
                    {
                        'pipeline_id': session_data['pipeline_id'],
                        'execution_id': session_data['execution_id']
                    }
                )

    # Execution Engine Integration
    def register_execution_context(
        self,
        execution_id: str,
        context: ExecutionContext
    ) -> None:
        """Register execution context for logging correlation."""
        self._execution_contexts[execution_id] = context
        
        self.logger.info(
            f"Execution context registered: {execution_id}",
            category=LogCategory.EXECUTION,
            metadata={
                'status': context.status.value if hasattr(context.status, 'value') else str(context.status),
                'pipeline_id': getattr(context, 'pipeline_id', None)
            }
        )

    def register_progress_tracker(
        self,
        execution_id: str,
        tracker: ProgressTracker
    ) -> None:
        """Register progress tracker for performance monitoring."""
        self._progress_trackers[execution_id] = tracker
        
        self.logger.debug(
            f"Progress tracker registered: {execution_id}",
            category=LogCategory.PERFORMANCE
        )

    async def log_execution_event(
        self,
        execution_id: str,
        event: ProgressEvent
    ) -> None:
        """Log execution progress events with quality context."""
        context = self._execution_contexts.get(execution_id)
        
        with self.logger.context(
            execution_id=execution_id,
            pipeline_id=getattr(context, 'pipeline_id', None) if context else None
        ):
            # Map event type to log level
            if event.event_type == ProgressEventType.ERROR:
                log_level = LogLevel.ERROR
            elif event.event_type == ProgressEventType.WARNING:
                log_level = LogLevel.WARNING
            else:
                log_level = LogLevel.INFO
            
            self.logger.log(
                log_level,
                f"Execution event: {event.message}",
                category=LogCategory.EXECUTION,
                metadata={
                    'event_type': event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
                    'event_data': event.data,
                    'timestamp': event.timestamp
                }
            )

    # Quality Control Manager Integration
    def create_integrated_quality_manager(
        self,
        validator: OutputQualityValidator,
        execution_monitor: ExecutionQualityMonitor
    ) -> 'IntegratedQualityControlManager':
        """Create quality control manager with integrated logging."""
        return IntegratedQualityControlManager(
            validator=validator,
            execution_monitor=execution_monitor,
            logging_integrator=self
        )

    # Utility methods
    def _map_validation_severity_to_log_level(self, severity: ValidationSeverity) -> LogLevel:
        """Map validation severity to logging level."""
        mapping = {
            ValidationSeverity.PASS: LogLevel.INFO,
            ValidationSeverity.WARNING: LogLevel.WARNING,
            ValidationSeverity.FAIL: LogLevel.ERROR,
            ValidationSeverity.CRITICAL: LogLevel.CRITICAL
        }
        return mapping.get(severity, LogLevel.INFO)

    def _map_rule_severity_to_log_level(self, severity: RuleSeverity) -> LogLevel:
        """Map rule severity to logging level."""
        mapping = {
            RuleSeverity.INFO: LogLevel.INFO,
            RuleSeverity.WARNING: LogLevel.WARNING,
            RuleSeverity.ERROR: LogLevel.ERROR,
            RuleSeverity.CRITICAL: LogLevel.CRITICAL
        }
        return mapping.get(severity, LogLevel.INFO)


class IntegratedQualityControlManager(QualityControlManager):
    """
    Quality Control Manager with integrated comprehensive logging.
    
    Extends the existing QualityControlManager to provide seamless
    integration with the quality logging framework.
    """
    
    def __init__(
        self,
        validator: OutputQualityValidator,
        execution_monitor: ExecutionQualityMonitor,
        logging_integrator: QualityLoggingIntegrator
    ):
        super().__init__(validator, execution_monitor)
        self.logging_integrator = logging_integrator

    async def validate_output(
        self,
        output_path: str,
        context: ExecutionContext
    ) -> ValidationResult:
        """Validate output with comprehensive logging integration."""
        # Generate session ID for tracking
        session_id = f"session_{context.execution_id}_{int(time.time())}"
        
        # Start validation session tracking
        validation_context = ValidationContext(
            execution_id=context.execution_id,
            pipeline_id=getattr(context, 'pipeline_id', None),
            output_path=output_path,
            metadata={}
        )
        
        await self.logging_integrator.start_validation_session(
            session_id=session_id,
            pipeline_id=getattr(context, 'pipeline_id', 'unknown'),
            execution_id=context.execution_id,
            context=validation_context
        )
        
        try:
            # Perform validation with timing
            with self.logging_integrator.logger.operation_timer(
                "output_validation", 
                LogCategory.VALIDATION
            ):
                result = await super().validate_output(output_path, context)
            
            # Log validation result
            await self.logging_integrator.log_validation_result(session_id, result)
            
            return result
            
        except Exception as e:
            self.logging_integrator.logger.error(
                f"Validation failed for {output_path}",
                category=LogCategory.VALIDATION,
                exception=e
            )
            raise
        finally:
            # Complete validation session
            await self.logging_integrator.complete_validation_session(session_id)


def create_integrated_quality_logging(
    config_path: Optional[str] = None,
    log_level: LogLevel = LogLevel.INFO,
    enable_monitoring: bool = True
) -> QualityLoggingIntegrator:
    """
    Create integrated quality logging setup with validation system.
    
    Args:
        config_path: Path to logging configuration file
        log_level: Default logging level
        enable_monitoring: Enable external monitoring integration
    
    Returns:
        Configured QualityLoggingIntegrator instance
    """
    # Load configuration
    config = {}
    if config_path:
        import yaml
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
    
    # Create logger
    logger = get_logger("quality_integration", level=log_level)
    
    # Create monitoring if enabled
    monitor = None
    if enable_monitoring and config.get('monitoring', {}).get('enabled', True):
        try:
            monitor = create_monitoring_setup(config.get('monitoring', {}), logger)
        except Exception as e:
            logger.warning(f"Failed to create monitoring setup: {e}")
    
    return QualityLoggingIntegrator(logger=logger, monitor=monitor, config=config)