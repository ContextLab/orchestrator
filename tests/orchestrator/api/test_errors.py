"""
Comprehensive tests for API error handling system.

Tests all error classes, recovery mechanisms, error handling integration,
and error context management for the orchestrator API framework.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

from orchestrator.api.errors import (
    OrchestratorAPIError,
    PipelineCompilationError,
    YAMLValidationError,
    TemplateProcessingError,
    PipelineExecutionError,
    ExecutionTimeoutError,
    StepExecutionError,
    APIConfigurationError,
    ModelRegistryError,
    ResourceError,
    NetworkError,
    UserInputError,
    APIErrorHandler,
    APIErrorCategory,
    APIErrorContext,
    RecoveryGuidance,
    create_api_error_handler,
    handle_api_exception,
)
from orchestrator.execution import (
    ErrorSeverity,
    RecoveryStrategy,
    ErrorCategory,
    RecoveryManager,
)


class TestAPIErrorContext:
    """Test API error context functionality."""
    
    def test_default_context_creation(self):
        """Test creating default error context."""
        context = APIErrorContext()
        
        assert context.error_id is not None
        assert len(context.error_id) == 8  # UUID short form
        assert isinstance(context.timestamp, datetime)
        assert context.operation is None
        assert context.pipeline_id is None
        assert context.metadata == {}
        assert context.related_errors == []
    
    def test_context_with_values(self):
        """Test creating context with specific values."""
        context = APIErrorContext(
            operation="test_operation",
            pipeline_id="test_pipeline",
            execution_id="test_execution",
            step_name="test_step",
            metadata={"key": "value"}
        )
        
        assert context.operation == "test_operation"
        assert context.pipeline_id == "test_pipeline"
        assert context.execution_id == "test_execution"
        assert context.step_name == "test_step"
        assert context.metadata == {"key": "value"}
    
    def test_context_to_dict(self):
        """Test converting context to dictionary."""
        context = APIErrorContext(
            operation="test_op",
            pipeline_id="test_pipeline",
            metadata={"test": "data"}
        )
        
        result = context.to_dict()
        
        assert result["error_id"] == context.error_id
        assert result["operation"] == "test_op"
        assert result["pipeline_id"] == "test_pipeline"
        assert result["metadata"] == {"test": "data"}
        assert "timestamp" in result
        
        # Check None values are excluded
        assert "execution_id" not in result


class TestRecoveryGuidance:
    """Test recovery guidance functionality."""
    
    def test_recovery_guidance_creation(self):
        """Test creating recovery guidance."""
        guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.RETRY,
            automatic_recovery=True,
            user_actions=["Check input", "Retry operation"],
            recovery_steps=["1. Validate", "2. Retry"],
            confidence_level=0.8
        )
        
        assert guidance.strategy == RecoveryStrategy.RETRY
        assert guidance.automatic_recovery is True
        assert guidance.user_actions == ["Check input", "Retry operation"]
        assert guidance.recovery_steps == ["1. Validate", "2. Retry"]
        assert guidance.confidence_level == 0.8
    
    def test_recovery_guidance_to_dict(self):
        """Test converting recovery guidance to dictionary."""
        guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
            user_actions=["Action 1"],
            system_actions=["System action"],
            estimated_recovery_time=30
        )
        
        result = guidance.to_dict()
        
        assert result["strategy"] == "retry_with_backoff"
        assert result["user_actions"] == ["Action 1"]
        assert result["system_actions"] == ["System action"]
        assert result["estimated_recovery_time"] == 30


class TestOrchestratorAPIError:
    """Test base OrchestratorAPIError functionality."""
    
    def test_basic_error_creation(self):
        """Test creating basic API error."""
        error = OrchestratorAPIError(
            message="Test error",
            category=APIErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM
        )
        
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.category == APIErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context is not None
        assert isinstance(error.context, APIErrorContext)
    
    def test_error_with_context_and_recovery(self):
        """Test error with custom context and recovery guidance."""
        context = APIErrorContext(operation="test_op")
        guidance = RecoveryGuidance(
            strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            user_actions=["Fix the issue"]
        )
        
        error = OrchestratorAPIError(
            message="Test error with context",
            category=APIErrorCategory.EXECUTION,
            context=context,
            recovery_guidance=guidance
        )
        
        assert error.context.operation == "test_op"
        assert error.recovery_guidance.strategy == RecoveryStrategy.MANUAL_INTERVENTION
    
    def test_error_with_original_exception(self):
        """Test error wrapping original exception."""
        original = ValueError("Original error")
        
        error = OrchestratorAPIError(
            message="Wrapped error",
            category=APIErrorCategory.USER_CONFIGURATION,
            original_exception=original
        )
        
        assert error.original_exception == original
        assert error.traceback_info is not None
    
    def test_error_to_error_info(self):
        """Test converting API error to foundation ErrorInfo."""
        error = OrchestratorAPIError(
            message="Test conversion",
            category=APIErrorCategory.COMPILATION,
            severity=ErrorSeverity.HIGH
        )
        
        error_info = error.to_error_info()
        
        assert error_info.message == "Test conversion"
        assert error_info.category == ErrorCategory.VALIDATION  # Mapped from COMPILATION
        assert error_info.severity == ErrorSeverity.HIGH
        assert error_info.error_id == error.context.error_id
    
    def test_error_to_dict(self):
        """Test converting error to dictionary."""
        error = OrchestratorAPIError(
            message="Dict test",
            category=APIErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM
        )
        
        result = error.to_dict()
        
        assert result["error_type"] == "OrchestratorAPIError"
        assert result["message"] == "Dict test"
        assert result["category"] == "network"
        assert result["severity"] == "medium"
        assert "context" in result


class TestSpecificErrorTypes:
    """Test specific error type implementations."""
    
    def test_pipeline_compilation_error(self):
        """Test pipeline compilation error."""
        error = PipelineCompilationError(
            message="Compilation failed",
            yaml_content="steps:\n- invalid",
            context_variables={"var1": "value1"},
            validation_errors=["Invalid step format"]
        )
        
        assert error.category == APIErrorCategory.COMPILATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.context.operation == "pipeline_compilation"
        assert "yaml_length" in error.context.metadata
        assert "validation_errors" in error.context.metadata
        assert error.recovery_guidance.strategy == RecoveryStrategy.MANUAL_INTERVENTION
    
    def test_yaml_validation_error(self):
        """Test YAML validation error."""
        error = YAMLValidationError(
            message="YAML syntax error",
            yaml_line=5,
            yaml_column=10
        )
        
        assert error.context.operation == "yaml_validation"
        assert error.context.metadata["yaml_line"] == 5
        assert error.context.metadata["yaml_column"] == 10
        assert "Fix YAML syntax errors" in error.recovery_guidance.user_actions
    
    def test_template_processing_error(self):
        """Test template processing error."""
        error = TemplateProcessingError(
            message="Missing variables",
            template_variables=["var1", "var2"],
            missing_variables=["var2"]
        )
        
        assert error.context.operation == "template_processing"
        assert error.context.metadata["missing_variables"] == ["var2"]
        assert any("missing variables" in action.lower() 
                  for action in error.recovery_guidance.user_actions)
    
    def test_pipeline_execution_error(self):
        """Test pipeline execution error."""
        error = PipelineExecutionError(
            message="Execution failed",
            pipeline_id="test_pipeline",
            execution_id="test_execution",
            failed_step="step1"
        )
        
        assert error.category == APIErrorCategory.EXECUTION
        assert error.context.pipeline_id == "test_pipeline"
        assert error.context.execution_id == "test_execution"
        assert error.context.step_name == "step1"
        assert error.recovery_guidance.automatic_recovery is True
    
    def test_execution_timeout_error(self):
        """Test execution timeout error."""
        error = ExecutionTimeoutError(
            message="Execution timed out",
            timeout_seconds=300,
            elapsed_seconds=350
        )
        
        assert error.context.operation == "execution_timeout"
        assert error.context.metadata["timeout_seconds"] == 300
        assert error.context.metadata["elapsed_seconds"] == 350
        assert error.recovery_guidance.strategy == RecoveryStrategy.RETRY_WITH_BACKOFF
    
    def test_step_execution_error(self):
        """Test step execution error."""
        error = StepExecutionError(
            message="Step failed",
            step_id="step_123",
            step_type="text_generation",
            step_config={"model": "test"}
        )
        
        assert error.context.step_id == "step_123"
        assert error.context.metadata["step_type"] == "text_generation"
        assert error.context.metadata["step_config"] == {"model": "test"}
    
    def test_api_configuration_error(self):
        """Test API configuration error."""
        error = APIConfigurationError(
            message="Invalid config",
            config_key="model_registry",
            config_value="invalid_value"
        )
        
        assert error.category == APIErrorCategory.CONFIGURATION
        assert error.context.metadata["config_key"] == "model_registry"
        assert error.context.metadata["config_value"] == "invalid_value"
    
    def test_model_registry_error(self):
        """Test model registry error."""
        error = ModelRegistryError(
            message="Model not found",
            model_name="test_model",
            registry_type="local"
        )
        
        assert error.context.metadata["model_name"] == "test_model"
        assert error.context.metadata["registry_type"] == "local"
    
    def test_resource_error(self):
        """Test resource error."""
        error = ResourceError(
            message="Resource unavailable",
            resource_type="memory",
            resource_id="mem_pool_1"
        )
        
        assert error.category == APIErrorCategory.RESOURCE_MANAGEMENT
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.recovery_guidance.automatic_recovery is True
    
    def test_network_error(self):
        """Test network error."""
        error = NetworkError(
            message="Connection failed",
            endpoint="https://api.example.com",
            status_code=503
        )
        
        assert error.category == APIErrorCategory.NETWORK
        assert error.context.metadata["endpoint"] == "https://api.example.com"
        assert error.context.metadata["status_code"] == 503
        assert error.recovery_guidance.estimated_recovery_time == 30
    
    def test_user_input_error(self):
        """Test user input error."""
        error = UserInputError(
            message="Invalid input",
            input_field="yaml_content",
            expected_type="string",
            provided_value=None
        )
        
        assert error.category == APIErrorCategory.INPUT_VALIDATION
        assert error.severity == ErrorSeverity.LOW
        assert error.context.metadata["input_field"] == "yaml_content"
        assert error.context.metadata["expected_type"] == "string"


class TestAPIErrorHandler:
    """Test API error handler functionality."""
    
    def test_error_handler_creation(self):
        """Test creating error handler."""
        handler = APIErrorHandler()
        
        assert handler.recovery_manager is None
        assert len(handler._error_handlers) > 0  # Default handlers registered
        assert handler._error_history == []
    
    def test_error_handler_with_recovery_manager(self):
        """Test creating error handler with recovery manager."""
        recovery_manager = Mock(spec=RecoveryManager)
        handler = APIErrorHandler(recovery_manager=recovery_manager)
        
        assert handler.recovery_manager == recovery_manager
    
    def test_handle_value_error(self):
        """Test handling ValueError."""
        handler = APIErrorHandler()
        original_error = ValueError("Invalid value provided")
        
        api_error = handler.handle_error(original_error, operation="test_op")
        
        assert isinstance(api_error, UserInputError)
        assert api_error.original_exception == original_error
        assert api_error.context.operation == "test_op"
        assert "Invalid input value" in api_error.message
    
    def test_handle_file_not_found_error(self):
        """Test handling FileNotFoundError."""
        handler = APIErrorHandler()
        original_error = FileNotFoundError("File not found: test.yaml")
        
        api_error = handler.handle_error(original_error)
        
        assert isinstance(api_error, APIConfigurationError)
        assert "Required file not found" in api_error.message
    
    def test_handle_timeout_error(self):
        """Test handling TimeoutError."""
        handler = APIErrorHandler()
        original_error = TimeoutError("Operation timed out")
        
        api_error = handler.handle_error(original_error)
        
        assert isinstance(api_error, ExecutionTimeoutError)
        assert "Operation timed out" in api_error.message
    
    def test_handle_connection_error(self):
        """Test handling ConnectionError."""
        handler = APIErrorHandler()
        original_error = ConnectionError("Connection refused")
        
        api_error = handler.handle_error(original_error)
        
        assert isinstance(api_error, NetworkError)
        assert "Network error" in api_error.message
    
    def test_handle_unknown_error(self):
        """Test handling unknown error type."""
        handler = APIErrorHandler()
        original_error = RuntimeError("Unknown error")
        
        api_error = handler.handle_error(original_error)
        
        assert isinstance(api_error, OrchestratorAPIError)
        assert "Unexpected error" in api_error.message
        assert api_error.category == APIErrorCategory.VALIDATION
    
    def test_error_history_tracking(self):
        """Test error history tracking."""
        handler = APIErrorHandler()
        
        error1 = ValueError("Error 1")
        error2 = TypeError("Error 2")
        
        api_error1 = handler.handle_error(error1)
        api_error2 = handler.handle_error(error2)
        
        history = handler.get_error_history()
        assert len(history) == 2
        assert history[0] == api_error1
        assert history[1] == api_error2
        
        handler.clear_error_history()
        assert handler.get_error_history() == []
    
    @patch('orchestrator.api.errors.logger')
    def test_error_logging(self, mock_logger):
        """Test error logging functionality."""
        handler = APIErrorHandler()
        original_error = ValueError("Test error for logging")
        
        handler.handle_error(original_error, operation="test_logging")
        
        # Should log the error
        mock_logger.error.assert_called()
        
        # Check log call
        log_call = mock_logger.error.call_args
        assert "UserInputError" in log_call[0][0]
        assert "Invalid input value" in log_call[0][0]


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_api_error_handler(self):
        """Test creating error handler via convenience function."""
        handler = create_api_error_handler()
        
        assert isinstance(handler, APIErrorHandler)
        assert handler.recovery_manager is None
    
    def test_create_api_error_handler_with_recovery(self):
        """Test creating error handler with recovery manager."""
        recovery_manager = Mock(spec=RecoveryManager)
        handler = create_api_error_handler(recovery_manager=recovery_manager)
        
        assert handler.recovery_manager == recovery_manager
    
    def test_handle_api_exception(self):
        """Test handle_api_exception convenience function."""
        original_error = ValueError("Test exception")
        
        api_error = handle_api_exception(
            original_error,
            operation="test_operation",
            context={"test": "data"}
        )
        
        assert isinstance(api_error, UserInputError)
        assert api_error.context.operation == "test_operation"
        assert api_error.context.metadata["test"] == "data"


class TestErrorIntegration:
    """Test error handling integration with foundation components."""
    
    def test_error_info_conversion(self):
        """Test conversion to foundation ErrorInfo."""
        error = PipelineExecutionError(
            message="Integration test",
            pipeline_id="test_pipeline"
        )
        
        error_info = error.to_error_info()
        
        assert error_info.message == "Integration test"
        assert error_info.category == ErrorCategory.EXECUTION
        assert error_info.severity == ErrorSeverity.HIGH
        assert error_info.error_id == error.context.error_id
    
    def test_category_mapping(self):
        """Test API category to foundation category mapping."""
        test_cases = [
            (APIErrorCategory.COMPILATION, ErrorCategory.VALIDATION),
            (APIErrorCategory.EXECUTION, ErrorCategory.EXECUTION),
            (APIErrorCategory.NETWORK, ErrorCategory.NETWORK),
            (APIErrorCategory.AUTHENTICATION, ErrorCategory.AUTHENTICATION),
            (APIErrorCategory.RESOURCE_MANAGEMENT, ErrorCategory.RESOURCE),
            (APIErrorCategory.DEPENDENCY_RESOLUTION, ErrorCategory.DEPENDENCY),
        ]
        
        for api_category, expected_foundation_category in test_cases:
            error = OrchestratorAPIError(
                message="Test mapping",
                category=api_category
            )
            
            error_info = error.to_error_info()
            assert error_info.category == expected_foundation_category


if __name__ == "__main__":
    pytest.main([__file__])