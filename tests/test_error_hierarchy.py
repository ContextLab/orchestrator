"""Comprehensive tests for the Orchestrator error hierarchy."""

import pytest
from src.orchestrator.core.exceptions import (
    # Base
    OrchestratorError,
    # Pipeline errors
    PipelineError,
    PipelineCompilationError,
    PipelineExecutionError,
    CircularDependencyError,
    InvalidDependencyError,
    # Task errors
    TaskError,
    TaskExecutionError,
    TaskValidationError,
    TaskTimeoutError,
    # Model errors
    ModelError,
    ModelNotFoundError,
    NoEligibleModelsError,
    ModelExecutionError,
    ModelConfigurationError,
    # Validation errors
    ValidationError,
    SchemaValidationError,
    YAMLValidationError,
    ParameterValidationError,
    # Resource errors
    ResourceError,
    ResourceAllocationError,
    ResourceLimitError,
    # State errors
    StateError,
    StateManagerError,
    StateCorruptionError,
    # Tool errors
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    # Control system errors
    ControlSystemError,
    CircuitBreakerOpenError,
    SystemUnavailableError,
    # Compilation errors
    CompilationError,
    YAMLCompilerError,
    AmbiguityResolutionError,
    # Adapter errors
    AdapterError,
    AdapterConfigurationError,
    AdapterConnectionError,
    # Configuration errors
    ConfigurationError,
    MissingConfigurationError,
    InvalidConfigurationError,
    # Network errors
    NetworkError,
    APIError,
    RateLimitError,
    AuthenticationError,
    # Timeout errors
    TimeoutError,
    # Helper function
    get_error_hierarchy
)


class TestErrorHierarchyStructure:
    """Test the error hierarchy structure and inheritance."""
    
    def test_base_error_is_exception(self):
        """Test that OrchestratorError inherits from Exception."""
        assert issubclass(OrchestratorError, Exception)
        
    def test_all_errors_inherit_from_base(self):
        """Test that all custom errors inherit from OrchestratorError."""
        error_classes = [
            PipelineError, TaskError, ModelError, ValidationError,
            ResourceError, StateError, ToolError, ControlSystemError,
            CompilationError, AdapterError, ConfigurationError, NetworkError,
            TimeoutError
        ]
        
        for error_class in error_classes:
            assert issubclass(error_class, OrchestratorError)
            
    def test_specific_error_inheritance(self):
        """Test specific error inheritance chains."""
        # Pipeline errors
        assert issubclass(PipelineCompilationError, PipelineError)
        assert issubclass(PipelineExecutionError, PipelineError)
        assert issubclass(CircularDependencyError, PipelineError)
        assert issubclass(InvalidDependencyError, PipelineError)
        
        # Task errors
        assert issubclass(TaskExecutionError, TaskError)
        assert issubclass(TaskValidationError, TaskError)
        assert issubclass(TaskTimeoutError, TaskError)
        
        # Model errors
        assert issubclass(ModelNotFoundError, ModelError)
        assert issubclass(NoEligibleModelsError, ModelError)
        assert issubclass(ModelExecutionError, ModelError)
        assert issubclass(ModelConfigurationError, ModelError)
        
        # Network errors
        assert issubclass(APIError, NetworkError)
        assert issubclass(RateLimitError, APIError)
        assert issubclass(AuthenticationError, APIError)
        
    def test_get_error_hierarchy(self):
        """Test the get_error_hierarchy helper function."""
        hierarchy = get_error_hierarchy()
        
        # Check that the hierarchy includes major categories
        assert "OrchestratorError" in hierarchy
        assert "PipelineError" in hierarchy["OrchestratorError"]
        assert "TaskError" in hierarchy["OrchestratorError"]
        assert "ModelError" in hierarchy["OrchestratorError"]
        
        # Check some specific subclasses
        assert "PipelineCompilationError" in hierarchy.get("PipelineError", [])
        assert "ModelNotFoundError" in hierarchy.get("ModelError", [])


class TestBaseErrorFunctionality:
    """Test base OrchestratorError functionality."""
    
    def test_error_creation_with_message_only(self):
        """Test creating error with just a message."""
        error = OrchestratorError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}
        assert error.error_code is None
        
    def test_error_creation_with_details(self):
        """Test creating error with details."""
        details = {"key": "value", "count": 42}
        error = OrchestratorError("Test error", details=details)
        assert error.details == details
        
    def test_error_creation_with_error_code(self):
        """Test creating error with error code."""
        error = OrchestratorError("Test error", error_code="E001")
        assert error.error_code == "E001"
        assert str(error) == "[E001] Test error"
        
    def test_error_to_dict(self):
        """Test converting error to dictionary."""
        error = OrchestratorError(
            "Test error",
            details={"key": "value"},
            error_code="E001"
        )
        
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "OrchestratorError"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"] == {"key": "value"}
        assert error_dict["error_code"] == "E001"


class TestSpecificErrorTypes:
    """Test specific error types with their custom behavior."""
    
    def test_circular_dependency_error(self):
        """Test CircularDependencyError with cycle information."""
        cycle = ["task1", "task2", "task3", "task1"]
        error = CircularDependencyError(cycle)
        
        assert "Circular dependency detected" in str(error)
        assert "task1 -> task2 -> task3 -> task1" in str(error)
        assert error.details["cycle"] == cycle
        
    def test_invalid_dependency_error(self):
        """Test InvalidDependencyError with task information."""
        error = InvalidDependencyError("task1", "missing_task")
        
        assert "task1" in str(error)
        assert "missing_task" in str(error)
        assert error.details["task_id"] == "task1"
        assert error.details["missing_dependency"] == "missing_task"
        
    def test_task_execution_error(self):
        """Test TaskExecutionError with task details."""
        error = TaskExecutionError("task1", "Division by zero")
        
        assert "task1" in str(error)
        assert "Division by zero" in str(error)
        assert error.details["task_id"] == "task1"
        assert error.details["reason"] == "Division by zero"
        
    def test_task_timeout_error(self):
        """Test TaskTimeoutError with timeout information."""
        error = TaskTimeoutError("long_task", 30.0)
        
        assert "long_task" in str(error)
        assert "30" in str(error)
        assert error.details["task_id"] == "long_task"
        assert error.details["timeout"] == 30.0
        
    def test_model_not_found_error(self):
        """Test ModelNotFoundError with model ID."""
        error = ModelNotFoundError("gpt-4")
        
        assert "gpt-4" in str(error)
        assert "not found" in str(error)
        assert error.details["model_id"] == "gpt-4"
        
    def test_no_eligible_models_error(self):
        """Test NoEligibleModelsError with requirements."""
        requirements = {"tasks": ["generate"], "min_context": 8192}
        error = NoEligibleModelsError(requirements)
        
        assert "No models meet" in str(error)
        assert error.details["requirements"] == requirements
        
    def test_schema_validation_error(self):
        """Test SchemaValidationError with validation errors."""
        validation_errors = [
            "Missing required field: name",
            "Invalid type for field: age"
        ]
        error = SchemaValidationError(validation_errors)
        
        assert "2 errors" in str(error)
        assert error.details["validation_errors"] == validation_errors
        
    def test_parameter_validation_error(self):
        """Test ParameterValidationError with parameter details."""
        error = ParameterValidationError("max_tokens", "must be positive integer")
        
        assert "max_tokens" in str(error)
        assert "must be positive integer" in str(error)
        assert error.details["parameter"] == "max_tokens"
        assert error.details["reason"] == "must be positive integer"
        
    def test_resource_allocation_error(self):
        """Test ResourceAllocationError with resource details."""
        error = ResourceAllocationError("memory", "8GB", "4GB")
        
        assert "8GB" in str(error)
        assert "4GB" in str(error)
        assert "memory" in str(error)
        assert error.details["resource_type"] == "memory"
        assert error.details["requested"] == "8GB"
        assert error.details["available"] == "4GB"
        
    def test_state_corruption_error(self):
        """Test StateCorruptionError with corruption details."""
        error = StateCorruptionError("Checksum mismatch")
        
        assert "State corruption" in str(error)
        assert "Checksum mismatch" in str(error)
        assert error.details["reason"] == "Checksum mismatch"
        
    def test_tool_not_found_error(self):
        """Test ToolNotFoundError with tool name."""
        error = ToolNotFoundError("web_scraper")
        
        assert "web_scraper" in str(error)
        assert "not found" in str(error)
        assert error.details["tool_name"] == "web_scraper"
        
    def test_tool_execution_error(self):
        """Test ToolExecutionError with execution details."""
        error = ToolExecutionError("file_reader", "Permission denied")
        
        assert "file_reader" in str(error)
        assert "Permission denied" in str(error)
        assert error.details["tool_name"] == "file_reader"
        assert error.details["reason"] == "Permission denied"
        
    def test_circuit_breaker_open_error(self):
        """Test CircuitBreakerOpenError with system name."""
        error = CircuitBreakerOpenError("external_api")
        
        assert "Circuit breaker is open" in str(error)
        assert "external_api" in str(error)
        assert error.details["system_name"] == "external_api"
        
    def test_system_unavailable_error(self):
        """Test SystemUnavailableError with reason."""
        error = SystemUnavailableError("database", "Connection timeout")
        
        assert "database" in str(error)
        assert "unavailable" in str(error)
        assert "Connection timeout" in str(error)
        assert error.details["system_name"] == "database"
        assert error.details["reason"] == "Connection timeout"
        
    def test_ambiguity_resolution_error(self):
        """Test AmbiguityResolutionError with context."""
        error = AmbiguityResolutionError("prompt", "User query too vague")
        
        assert "prompt" in str(error)
        assert "User query too vague" in str(error)
        assert error.details["ambiguity_type"] == "prompt"
        assert error.details["context"] == "User query too vague"
        
    def test_adapter_connection_error(self):
        """Test AdapterConnectionError with adapter details."""
        error = AdapterConnectionError("langchain", "Invalid credentials")
        
        assert "langchain" in str(error)
        assert "Invalid credentials" in str(error)
        assert error.details["adapter_name"] == "langchain"
        assert error.details["reason"] == "Invalid credentials"
        
    def test_missing_configuration_error(self):
        """Test MissingConfigurationError with config key."""
        error = MissingConfigurationError("api_key")
        
        assert "api_key" in str(error)
        assert "Missing required configuration" in str(error)
        assert error.details["config_key"] == "api_key"
        
    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError with details."""
        error = InvalidConfigurationError("timeout", "must be positive number")
        
        assert "timeout" in str(error)
        assert "must be positive number" in str(error)
        assert error.details["config_key"] == "timeout"
        assert error.details["reason"] == "must be positive number"
        
    def test_api_error(self):
        """Test APIError with service and status code."""
        error = APIError("openai", status_code=503)
        
        assert "openai" in str(error)
        assert "503" in str(error)
        assert error.details["service"] == "openai"
        assert error.details["status_code"] == 503
        
    def test_rate_limit_error(self):
        """Test RateLimitError with retry information."""
        error = RateLimitError("anthropic", retry_after=60.0)
        
        assert "Rate limit exceeded" in str(error)
        assert "anthropic" in str(error)
        assert "60" in str(error)
        assert error.details["retry_after"] == 60.0
        
    def test_authentication_error(self):
        """Test AuthenticationError with service."""
        error = AuthenticationError("google")
        
        assert "Authentication failed" in str(error)
        assert "google" in str(error)
        assert error.details["status_code"] == 401
        
    def test_timeout_error(self):
        """Test TimeoutError with operation details."""
        error = TimeoutError("model_inference", 30.0)
        
        assert "model_inference" in str(error)
        assert "30" in str(error)
        assert error.details["operation"] == "model_inference"
        assert error.details["timeout"] == 30.0


class TestErrorHandlingEdgeCases:
    """Test edge cases and error handling scenarios."""
    
    def test_error_with_none_details(self):
        """Test that None details are converted to empty dict."""
        error = OrchestratorError("Test", details=None)
        assert error.details == {}
        
    def test_error_inheritance_chain(self):
        """Test that errors maintain proper inheritance chain."""
        error = RateLimitError("service")
        
        # Check full inheritance chain
        assert isinstance(error, RateLimitError)
        assert isinstance(error, APIError)
        assert isinstance(error, NetworkError)
        assert isinstance(error, OrchestratorError)
        assert isinstance(error, Exception)
        
    def test_error_with_additional_kwargs(self):
        """Test that additional kwargs are handled properly."""
        error = ModelNotFoundError("model1", error_code="M001")
        assert error.error_code == "M001"
        
    def test_error_serialization(self):
        """Test that errors can be serialized properly."""
        import json
        
        error = TaskExecutionError("task1", "Test failure", error_code="T001")
        error_dict = error.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(error_dict)
        loaded = json.loads(json_str)
        
        assert loaded["error_type"] == "TaskExecutionError"
        assert loaded["message"] == "Task 'task1' failed: Test failure"
        assert loaded["error_code"] == "T001"
        
    def test_raising_and_catching_errors(self):
        """Test raising and catching errors in realistic scenarios."""
        # Test catching specific error
        with pytest.raises(ModelNotFoundError) as exc_info:
            raise ModelNotFoundError("gpt-5")
        
        assert "gpt-5" in str(exc_info.value)
        
        # Test catching by parent class
        with pytest.raises(ModelError):
            raise NoEligibleModelsError({"tasks": ["unknown"]})
            
        # Test catching by base class
        with pytest.raises(OrchestratorError):
            raise CircularDependencyError(["a", "b", "a"])
            
    def test_error_context_preservation(self):
        """Test that error context is preserved through re-raising."""
        original_error = ValueError("Original error")
        
        try:
            try:
                raise original_error
            except ValueError as e:
                raise TaskExecutionError("task1", "Wrapped error") from e
        except TaskExecutionError as task_error:
            assert task_error.__cause__ is original_error
            
    def test_custom_error_messages(self):
        """Test that custom error messages work correctly."""
        # Test error with custom formatting
        cycle = ["step1", "step2", "step3", "step1"]
        error = CircularDependencyError(cycle, error_code="CYCLE001")
        
        error_str = str(error)
        assert error_str.startswith("[CYCLE001]")
        assert "step1 -> step2 -> step3 -> step1" in error_str
        
    def test_error_equality(self):
        """Test error equality based on type and attributes."""
        error1 = ModelNotFoundError("model1")
        error2 = ModelNotFoundError("model1")
        error3 = ModelNotFoundError("model2")
        
        # Same type and message
        assert type(error1) == type(error2)
        assert error1.message == error2.message
        
        # Different message
        assert error1.message != error3.message


class TestRealWorldErrorScenarios:
    """Test error usage in real-world scenarios."""
    
    def test_pipeline_execution_error_flow(self):
        """Test error flow in pipeline execution."""
        # Simulate pipeline execution with errors
        def execute_pipeline():
            try:
                # Simulate task failure
                raise TaskExecutionError("data_processing", "Invalid input format")
            except TaskError as e:
                # Wrap in pipeline error
                raise PipelineExecutionError(
                    "Pipeline failed during data processing",
                    details={"failed_task": e.details["task_id"]},
                    error_code="PIPE001"
                ) from e
                
        with pytest.raises(PipelineExecutionError) as exc_info:
            execute_pipeline()
            
        error = exc_info.value
        assert error.error_code == "PIPE001"
        assert error.details["failed_task"] == "data_processing"
        assert isinstance(error.__cause__, TaskExecutionError)
        
    def test_model_selection_error_flow(self):
        """Test error flow in model selection."""
        requirements = {
            "tasks": ["generate"],
            "min_context": 1000000,  # Unrealistic requirement
            "supports_tools": True
        }
        
        def select_model(reqs):
            # Simulate no models meeting requirements
            raise NoEligibleModelsError(reqs)
            
        with pytest.raises(NoEligibleModelsError) as exc_info:
            select_model(requirements)
            
        error = exc_info.value
        assert error.details["requirements"] == requirements
        
    def test_api_error_handling_with_retry(self):
        """Test API error handling with retry logic."""
        retry_count = 0
        
        def call_api():
            nonlocal retry_count
            retry_count += 1
            
            if retry_count < 3:
                raise RateLimitError("openai", retry_after=1.0)
            return {"success": True}
            
        # Simulate retry logic
        for attempt in range(5):
            try:
                result = call_api()
                break
            except RateLimitError as e:
                if attempt >= 4:
                    raise
                # Would normally wait for retry_after seconds
                continue
                
        assert retry_count == 3
        assert result == {"success": True}
        
    def test_validation_error_aggregation(self):
        """Test aggregating multiple validation errors."""
        validation_errors = []
        
        # Simulate validating multiple parameters
        params = {
            "max_tokens": -1,
            "temperature": 2.5,
            "model": None
        }
        
        if params["max_tokens"] < 0:
            validation_errors.append(
                ParameterValidationError("max_tokens", "must be positive")
            )
            
        if params["temperature"] > 2.0:
            validation_errors.append(
                ParameterValidationError("temperature", "must be <= 2.0")
            )
            
        if params["model"] is None:
            validation_errors.append(
                ParameterValidationError("model", "is required")
            )
            
        # Check we collected all errors
        assert len(validation_errors) == 3
        assert all(isinstance(e, ParameterValidationError) for e in validation_errors)
        
    def test_resource_allocation_with_fallback(self):
        """Test resource allocation with fallback on error."""
        available_memory = 4.0  # GB
        
        def allocate_resources(memory_gb):
            if memory_gb > available_memory:
                raise ResourceAllocationError(
                    "memory",
                    f"{memory_gb}GB",
                    f"{available_memory}GB"
                )
            return {"allocated": memory_gb}
            
        # Try to allocate too much
        try:
            result = allocate_resources(8.0)
        except ResourceAllocationError:
            # Fall back to available amount
            result = allocate_resources(available_memory)
            
        assert result["allocated"] == 4.0
        
    def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern with errors."""
        failure_count = 0
        circuit_open = False
        
        def call_service():
            nonlocal failure_count, circuit_open
            
            if circuit_open:
                raise CircuitBreakerOpenError("external_service")
                
            # Simulate failures
            failure_count += 1
            if failure_count <= 3:
                raise SystemUnavailableError("external_service", "Timeout")
                
            return {"data": "success"}
            
        # Simulate circuit breaker logic
        result = None
        for attempt in range(5):
            try:
                result = call_service()
                break
            except SystemUnavailableError:
                if failure_count >= 3:
                    circuit_open = True
            except CircuitBreakerOpenError:
                # Circuit is open, would normally wait before retry
                break
                
        # Circuit opened after 3 failures
        assert circuit_open
        assert failure_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])