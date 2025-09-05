"""
Comprehensive tests for quality logging integration with validation system.

This test suite validates the seamless integration between the logging framework
and the existing validation system, ensuring proper event correlation, context
propagation, and quality metrics collection.
"""

import asyncio
import time
from dataclasses import asdict
from unittest.mock import Mock, patch, AsyncMock
import pytest

from src.orchestrator.quality.logging.integration import (
    QualityLoggingIntegrator,
    IntegratedQualityControlManager,
    create_integrated_quality_logging
)
from src.orchestrator.quality.logging.logger import (
    StructuredLogger,
    LogLevel,
    LogCategory,
    QualityEvent
)
from src.orchestrator.quality.logging.monitoring import QualityMonitor
from src.orchestrator.quality.validation.validator import (
    ValidationResult,
    ValidationSeverity,
    OutputQualityValidator
)
from src.orchestrator.quality.validation.rules import (
    RuleViolation,
    RuleSeverity,
    ValidationContext
)
from src.orchestrator.quality.validation.integration import (
    ExecutionQualityMonitor,
    QualityControlManager
)
from src.orchestrator.execution.state import ExecutionContext, ExecutionStatus
from src.orchestrator.execution.progress import ProgressTracker, ProgressEvent, ProgressEventType


class TestQualityLoggingIntegrator:
    """Test QualityLoggingIntegrator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_logger = Mock(spec=StructuredLogger)
        self.mock_monitor = AsyncMock(spec=QualityMonitor)
        self.config = {"test": "config"}
        
        self.integrator = QualityLoggingIntegrator(
            logger=self.mock_logger,
            monitor=self.mock_monitor,
            config=self.config
        )

    @pytest.mark.asyncio
    async def test_integrator_creation(self):
        """Test creating quality logging integrator."""
        assert self.integrator.logger is self.mock_logger
        assert self.integrator.monitor is self.mock_monitor
        assert self.integrator.config == self.config
        assert isinstance(self.integrator._active_sessions, dict)
        assert isinstance(self.integrator._quality_metrics, dict)

    @pytest.mark.asyncio
    async def test_start_integration(self):
        """Test starting the integration."""
        await self.integrator.start()
        
        self.mock_logger.info.assert_called_with(
            "Starting quality logging integration", 
            category=LogCategory.SYSTEM
        )
        self.mock_monitor.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_integration(self):
        """Test stopping the integration."""
        # Add an active session
        self.integrator._active_sessions["test_session"] = {
            "session_id": "test_session",
            "pipeline_id": "pipe-123",
            "execution_id": "exec-456",
            "start_time": time.time()
        }
        
        with patch.object(self.integrator, 'complete_validation_session') as mock_complete:
            await self.integrator.stop()
            
            mock_complete.assert_called_once_with("test_session", forced=True)
        
        self.mock_monitor.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_validation_session(self):
        """Test starting a validation session."""
        session_id = "test_session"
        pipeline_id = "pipe-123"
        execution_id = "exec-456"
        context = ValidationContext(
            execution_id=execution_id,
            pipeline_id=pipeline_id,
            output_path="/test/path",
            metadata={}
        )
        
        await self.integrator.start_validation_session(
            session_id=session_id,
            pipeline_id=pipeline_id,
            execution_id=execution_id,
            context=context
        )
        
        # Check session was registered
        assert session_id in self.integrator._active_sessions
        session_data = self.integrator._active_sessions[session_id]
        assert session_data["pipeline_id"] == pipeline_id
        assert session_data["execution_id"] == execution_id
        assert session_data["rules_executed"] == 0
        
        # Check logging was called
        self.mock_logger.context.assert_called()
        self.mock_logger.info.assert_called()
        
        # Check quality event was recorded
        self.mock_monitor.record_quality_event.assert_called()

    @pytest.mark.asyncio
    async def test_log_validation_result(self):
        """Test logging validation results."""
        session_id = "test_session"
        
        # Set up session first
        self.integrator._active_sessions[session_id] = {
            "session_id": session_id,
            "pipeline_id": "pipe-123",
            "execution_id": "exec-456",
            "start_time": time.time(),
            "rules_executed": 0,
            "violations_found": 0,
            "quality_score": 0.0
        }
        
        # Create validation result
        validation_result = ValidationResult(
            session_id=session_id,
            pipeline_id="pipe-123",
            execution_id="exec-456",
            step_id="step-789",
            output_path="/test/output",
            severity=ValidationSeverity.WARNING,
            quality_score=0.75,
            violations=[
                RuleViolation(
                    rule_name="test_rule",
                    severity=RuleSeverity.WARNING,
                    message="Test violation",
                    context={}
                )
            ],
            performance_metrics={"duration_ms": 150.0},
            recommendations=["Fix the issue"],
            execution_time=0.15
        )
        
        await self.integrator.log_validation_result(session_id, validation_result)
        
        # Check session metrics were updated
        session_data = self.integrator._active_sessions[session_id]
        assert session_data["rules_executed"] == 1
        assert session_data["violations_found"] == 1
        assert session_data["quality_score"] == 0.75
        
        # Check logging was called with proper context
        self.mock_logger.context.assert_called()
        self.mock_logger.log_validation_result.assert_called()
        
        # Check monitoring was called
        self.mock_monitor.record_quality_event.assert_called()
        self.mock_monitor.record_metric.assert_called()

    @pytest.mark.asyncio
    async def test_log_validation_result_unknown_session(self):
        """Test logging validation result for unknown session."""
        validation_result = ValidationResult(
            session_id="unknown_session",
            pipeline_id="pipe-123",
            execution_id="exec-456",
            step_id=None,
            output_path="/test/output",
            severity=ValidationSeverity.PASS,
            quality_score=0.95,
            violations=[],
            performance_metrics={},
            recommendations=[],
            execution_time=0.1
        )
        
        await self.integrator.log_validation_result("unknown_session", validation_result)
        
        # Should log warning about unknown session
        self.mock_logger.warning.assert_called_with(
            "Validation result received for unknown session: unknown_session"
        )

    @pytest.mark.asyncio
    async def test_log_rule_violation(self):
        """Test logging individual rule violations."""
        session_id = "test_session"
        
        # Set up session
        self.integrator._active_sessions[session_id] = {
            "session_id": session_id,
            "pipeline_id": "pipe-123",
            "execution_id": "exec-456",
            "start_time": time.time()
        }
        
        violation = RuleViolation(
            rule_name="test_rule",
            severity=RuleSeverity.ERROR,
            message="Critical validation error",
            context={"file": "test.txt", "line": 42}
        )
        
        context = {"additional": "context"}
        
        await self.integrator.log_rule_violation(session_id, violation, context)
        
        # Check logging was called
        self.mock_logger.context.assert_called()
        self.mock_logger.log.assert_called()
        
        # Check monitoring was called
        self.mock_monitor.record_quality_event.assert_called()

    @pytest.mark.asyncio
    async def test_complete_validation_session(self):
        """Test completing a validation session."""
        session_id = "test_session"
        start_time = time.time() - 1.0  # 1 second ago
        
        # Set up session
        self.integrator._active_sessions[session_id] = {
            "session_id": session_id,
            "pipeline_id": "pipe-123",
            "execution_id": "exec-456",
            "start_time": start_time,
            "rules_executed": 5,
            "violations_found": 2,
            "quality_score": 0.80
        }
        
        await self.integrator.complete_validation_session(session_id)
        
        # Session should be removed
        assert session_id not in self.integrator._active_sessions
        
        # Check logging was called
        self.mock_logger.context.assert_called()
        self.mock_logger.info.assert_called()
        
        # Check metrics were recorded
        self.mock_monitor.record_metric.assert_called()

    @pytest.mark.asyncio
    async def test_complete_unknown_session(self):
        """Test completing unknown validation session."""
        await self.integrator.complete_validation_session("unknown_session")
        
        # Should log warning
        self.mock_logger.warning.assert_called_with(
            "Attempted to complete unknown session: unknown_session"
        )

    def test_register_execution_context(self):
        """Test registering execution context."""
        execution_id = "exec-123"
        context = Mock(spec=ExecutionContext)
        context.status = ExecutionStatus.RUNNING
        context.pipeline_id = "pipe-456"
        
        self.integrator.register_execution_context(execution_id, context)
        
        assert execution_id in self.integrator._execution_contexts
        assert self.integrator._execution_contexts[execution_id] is context
        
        # Check logging was called
        self.mock_logger.info.assert_called()

    def test_register_progress_tracker(self):
        """Test registering progress tracker."""
        execution_id = "exec-123"
        tracker = Mock(spec=ProgressTracker)
        
        self.integrator.register_progress_tracker(execution_id, tracker)
        
        assert execution_id in self.integrator._progress_trackers
        assert self.integrator._progress_trackers[execution_id] is tracker
        
        # Check logging was called
        self.mock_logger.debug.assert_called()

    @pytest.mark.asyncio
    async def test_log_execution_event(self):
        """Test logging execution events."""
        execution_id = "exec-123"
        
        # Register context
        context = Mock(spec=ExecutionContext)
        context.pipeline_id = "pipe-456"
        self.integrator._execution_contexts[execution_id] = context
        
        event = ProgressEvent(
            event_type=ProgressEventType.ERROR,
            message="Test error occurred",
            timestamp=time.time(),
            data={"error_code": "E001"}
        )
        
        await self.integrator.log_execution_event(execution_id, event)
        
        # Check logging was called with proper context
        self.mock_logger.context.assert_called()
        self.mock_logger.log.assert_called()

    def test_severity_mapping(self):
        """Test validation severity to log level mapping."""
        # Test all validation severity mappings
        assert self.integrator._map_validation_severity_to_log_level(
            ValidationSeverity.PASS
        ) == LogLevel.INFO
        
        assert self.integrator._map_validation_severity_to_log_level(
            ValidationSeverity.WARNING
        ) == LogLevel.WARNING
        
        assert self.integrator._map_validation_severity_to_log_level(
            ValidationSeverity.FAIL
        ) == LogLevel.ERROR
        
        assert self.integrator._map_validation_severity_to_log_level(
            ValidationSeverity.CRITICAL
        ) == LogLevel.CRITICAL

    def test_rule_severity_mapping(self):
        """Test rule severity to log level mapping."""
        # Test all rule severity mappings
        assert self.integrator._map_rule_severity_to_log_level(
            RuleSeverity.INFO
        ) == LogLevel.INFO
        
        assert self.integrator._map_rule_severity_to_log_level(
            RuleSeverity.WARNING
        ) == LogLevel.WARNING
        
        assert self.integrator._map_rule_severity_to_log_level(
            RuleSeverity.ERROR
        ) == LogLevel.ERROR
        
        assert self.integrator._map_rule_severity_to_log_level(
            RuleSeverity.CRITICAL
        ) == LogLevel.CRITICAL

    def test_create_integrated_quality_manager(self):
        """Test creating integrated quality control manager."""
        validator = Mock(spec=OutputQualityValidator)
        execution_monitor = Mock(spec=ExecutionQualityMonitor)
        
        manager = self.integrator.create_integrated_quality_manager(
            validator, execution_monitor
        )
        
        assert isinstance(manager, IntegratedQualityControlManager)
        assert manager.logging_integrator is self.integrator


class TestIntegratedQualityControlManager:
    """Test IntegratedQualityControlManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = Mock(spec=OutputQualityValidator)
        self.execution_monitor = Mock(spec=ExecutionQualityMonitor)
        self.logging_integrator = Mock(spec=QualityLoggingIntegrator)
        
        self.manager = IntegratedQualityControlManager(
            validator=self.validator,
            execution_monitor=self.execution_monitor,
            logging_integrator=self.logging_integrator
        )

    @pytest.mark.asyncio
    async def test_validate_output_success(self):
        """Test successful output validation with logging integration."""
        output_path = "/test/output.txt"
        context = Mock(spec=ExecutionContext)
        context.execution_id = "exec-123"
        context.pipeline_id = "pipe-456"
        
        # Mock validation result
        validation_result = ValidationResult(
            session_id="session-789",
            pipeline_id="pipe-456",
            execution_id="exec-123",
            step_id=None,
            output_path=output_path,
            severity=ValidationSeverity.PASS,
            quality_score=0.95,
            violations=[],
            performance_metrics={"duration_ms": 100.0},
            recommendations=[],
            execution_time=0.1
        )
        
        # Mock parent class method
        with patch.object(QualityControlManager, 'validate_output', 
                         new_callable=AsyncMock, return_value=validation_result):
            
            result = await self.manager.validate_output(output_path, context)
            
            # Check that validation session was started and completed
            self.logging_integrator.start_validation_session.assert_called_once()
            self.logging_integrator.log_validation_result.assert_called_once()
            self.logging_integrator.complete_validation_session.assert_called_once()
            
            # Check operation timer was used
            self.logging_integrator.logger.operation_timer.assert_called_once()
            
            assert result is validation_result

    @pytest.mark.asyncio
    async def test_validate_output_exception(self):
        """Test validation with exception handling."""
        output_path = "/test/output.txt"
        context = Mock(spec=ExecutionContext)
        context.execution_id = "exec-123"
        context.pipeline_id = "pipe-456"
        
        test_exception = Exception("Validation failed")
        
        # Mock parent class to raise exception
        with patch.object(QualityControlManager, 'validate_output',
                         new_callable=AsyncMock, side_effect=test_exception):
            
            with pytest.raises(Exception, match="Validation failed"):
                await self.manager.validate_output(output_path, context)
            
            # Check that validation session was started
            self.logging_integrator.start_validation_session.assert_called_once()
            
            # Check that error was logged
            self.logging_integrator.logger.error.assert_called_once()
            
            # Check that session was completed even on exception
            self.logging_integrator.complete_validation_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_propagation(self):
        """Test that execution context is properly propagated."""
        output_path = "/test/output.txt"
        context = Mock(spec=ExecutionContext)
        context.execution_id = "exec-123"
        context.pipeline_id = "pipe-456"
        
        validation_result = ValidationResult(
            session_id="session-789",
            pipeline_id="pipe-456",
            execution_id="exec-123",
            step_id=None,
            output_path=output_path,
            severity=ValidationSeverity.PASS,
            quality_score=0.95,
            violations=[],
            performance_metrics={},
            recommendations=[],
            execution_time=0.1
        )
        
        with patch.object(QualityControlManager, 'validate_output',
                         new_callable=AsyncMock, return_value=validation_result):
            
            await self.manager.validate_output(output_path, context)
            
            # Check that session was started with proper context
            start_call = self.logging_integrator.start_validation_session.call_args
            assert start_call[1]["execution_id"] == "exec-123"
            assert start_call[1]["pipeline_id"] == "pipe-456"


class TestCreateIntegratedQualityLogging:
    """Test create_integrated_quality_logging function."""
    
    def test_basic_creation(self):
        """Test basic integrated logging creation."""
        integrator = create_integrated_quality_logging(
            log_level=LogLevel.INFO,
            enable_monitoring=False
        )
        
        assert isinstance(integrator, QualityLoggingIntegrator)
        assert integrator.logger is not None
        assert integrator.monitor is None

    def test_with_monitoring_enabled(self):
        """Test creation with monitoring enabled."""
        with patch('src.orchestrator.quality.logging.integration.create_monitoring_setup') as mock_create_monitoring:
            mock_monitor = Mock(spec=QualityMonitor)
            mock_create_monitoring.return_value = mock_monitor
            
            integrator = create_integrated_quality_logging(
                enable_monitoring=True
            )
            
            assert integrator.monitor is mock_monitor
            mock_create_monitoring.assert_called_once()

    def test_with_config_file(self):
        """Test creation with configuration file."""
        config_data = {
            "monitoring": {
                "enabled": True,
                "backends": []
            }
        }
        
        with patch('builtins.open', mock_open(read_data="monitoring:\n  enabled: true\n  backends: []")), \
             patch('yaml.safe_load', return_value=config_data), \
             patch('src.orchestrator.quality.logging.integration.create_monitoring_setup') as mock_create_monitoring:
            
            mock_monitor = Mock(spec=QualityMonitor)
            mock_create_monitoring.return_value = mock_monitor
            
            integrator = create_integrated_quality_logging(
                config_path="/test/config.yaml"
            )
            
            assert integrator.config == config_data
            mock_create_monitoring.assert_called_once_with(config_data.get('monitoring', {}), integrator.logger)

    def test_config_loading_failure(self):
        """Test handling of configuration loading failure."""
        with patch('builtins.open', side_effect=FileNotFoundError), \
             patch('logging.warning') as mock_warning:
            
            integrator = create_integrated_quality_logging(
                config_path="/nonexistent/config.yaml"
            )
            
            # Should still create integrator despite config failure
            assert isinstance(integrator, QualityLoggingIntegrator)
            mock_warning.assert_called()

    def test_monitoring_setup_failure(self):
        """Test handling of monitoring setup failure."""
        config_data = {
            "monitoring": {
                "enabled": True,
                "backends": [{"invalid": "config"}]
            }
        }
        
        with patch('yaml.safe_load', return_value=config_data), \
             patch('src.orchestrator.quality.logging.integration.create_monitoring_setup',
                   side_effect=Exception("Setup failed")):
            
            # Should create mock logger with warning method
            mock_logger = Mock()
            mock_logger.warning = Mock()
            
            with patch('src.orchestrator.quality.logging.integration.get_logger',
                      return_value=mock_logger):
                
                integrator = create_integrated_quality_logging(
                    enable_monitoring=True
                )
                
                # Should still create integrator without monitoring
                assert isinstance(integrator, QualityLoggingIntegrator)
                assert integrator.monitor is None
                mock_logger.warning.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])