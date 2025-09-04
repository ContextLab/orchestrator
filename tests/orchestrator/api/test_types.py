"""
Comprehensive tests for API type definitions and validation.

Tests all type definitions, serialization, validation, and integration
for the orchestrator API framework type system.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock

from src.orchestrator.api.types import (
    # Enums
    APIOperation,
    ValidationLevel,
    CompilationMode,
    ExecutionMode,
    
    # Request types
    CompilationRequest,
    ExecutionRequest,
    
    # Response types
    APIResponse,
    CompilationResult,
    ExecutionResult,
    ExecutionStatusInfo,
    ProgressUpdate,
    
    # Configuration types
    APIConfiguration,
    
    # Protocol types
    PipelineCompilerProtocol,
    ExecutionManagerProtocol,
    ProgressMonitorProtocol,
    
    # TypedDict types
    PipelineCompilationDict,
    PipelineExecutionDict,
    ExecutionStatusDict,
    ValidationResult,
    ResourceUsage,
    StepSummary,
    
    # Documentation types
    APIEndpoint,
    API_DOCUMENTATION,
    
    # Response type aliases
    PipelineCompilationResponse,
    PipelineExecutionResponse,
    ExecutionStatusResponse,
    ProgressUpdateResponse,
)
from src.orchestrator.execution import (
    ExecutionStatus,
    ExecutionMetrics,
    ProgressEventType,
    StepStatus,
)
from src.orchestrator.core.pipeline import Pipeline


class TestEnums:
    """Test enum type definitions."""
    
    def test_api_operation_enum(self):
        """Test APIOperation enum values."""
        assert APIOperation.COMPILE_PIPELINE.value == "compile_pipeline"
        assert APIOperation.EXECUTE_PIPELINE.value == "execute_pipeline"
        assert APIOperation.GET_STATUS.value == "get_execution_status"
        assert APIOperation.MONITOR_EXECUTION.value == "monitor_execution"
        
        # Ensure all expected operations are present
        expected_ops = [
            "compile_pipeline", "execute_pipeline", "validate_yaml",
            "get_execution_status", "stop_execution", "list_active_executions",
            "cleanup_execution", "get_compilation_report", "get_template_variables",
            "monitor_execution", "control_execution"
        ]
        
        actual_ops = [op.value for op in APIOperation]
        for expected in expected_ops:
            assert expected in actual_ops
    
    def test_validation_level_enum(self):
        """Test ValidationLevel enum values."""
        assert ValidationLevel.STRICT.value == "strict"
        assert ValidationLevel.PERMISSIVE.value == "permissive"
        assert ValidationLevel.DEVELOPMENT.value == "development"
        assert ValidationLevel.DISABLED.value == "disabled"
    
    def test_compilation_mode_enum(self):
        """Test CompilationMode enum values."""
        assert CompilationMode.STANDARD.value == "standard"
        assert CompilationMode.FAST.value == "fast"
        assert CompilationMode.SAFE.value == "safe"
        assert CompilationMode.DEBUG.value == "debug"
    
    def test_execution_mode_enum(self):
        """Test ExecutionMode enum values."""
        assert ExecutionMode.NORMAL.value == "normal"
        assert ExecutionMode.DRY_RUN.value == "dry_run"
        assert ExecutionMode.STEP_BY_STEP.value == "step_by_step"
        assert ExecutionMode.PARALLEL.value == "parallel"
        assert ExecutionMode.RECOVERY.value == "recovery"


class TestRequestTypes:
    """Test request type definitions."""
    
    def test_compilation_request_creation(self):
        """Test creating compilation request."""
        request = CompilationRequest(
            yaml_content="steps:\n  - name: test",
            context={"var": "value"},
            validation_level=ValidationLevel.STRICT
        )
        
        assert request.yaml_content == "steps:\n  - name: test"
        assert request.context == {"var": "value"}
        assert request.validation_level == ValidationLevel.STRICT
        assert request.resolve_ambiguities is True  # Default
        assert request.validate is True  # Default
        assert isinstance(request.request_id, str)
        assert isinstance(request.timestamp, datetime)
    
    def test_compilation_request_defaults(self):
        """Test compilation request default values."""
        request = CompilationRequest(yaml_content="test content")
        
        assert request.context is None
        assert request.resolve_ambiguities is True
        assert request.validate is True
        assert request.validation_level == ValidationLevel.STRICT
        assert request.compilation_mode == CompilationMode.STANDARD
        assert request.enable_preprocessing is True
        assert request.template_strict_mode is True
        assert request.cache_result is True
        assert request.include_metadata is False
        assert request.debug_mode is False
        assert request.user_id is None
    
    def test_compilation_request_with_path(self):
        """Test compilation request with file path."""
        path = Path("/test/pipeline.yaml")
        request = CompilationRequest(yaml_content=path)
        
        assert request.yaml_content == path
    
    def test_compilation_request_to_dict(self):
        """Test converting compilation request to dictionary."""
        path = Path("/test/pipeline.yaml")
        request = CompilationRequest(
            yaml_content=path,
            context={"test": "value"},
            validation_level=ValidationLevel.DEVELOPMENT,
            user_id="test_user"
        )
        
        result = request.to_dict()
        
        assert result["yaml_content"] == str(path)
        assert result["context"] == {"test": "value"}
        assert result["validation_level"] == "development"
        assert result["user_id"] == "test_user"
        assert result["request_id"] == request.request_id
        assert "timestamp" in result
    
    def test_execution_request_creation(self):
        """Test creating execution request."""
        mock_pipeline = Mock(spec=Pipeline)
        mock_pipeline.id = "test_pipeline"
        
        request = ExecutionRequest(
            pipeline=mock_pipeline,
            context={"input": "data"},
            execution_mode=ExecutionMode.NORMAL,
            timeout=3600
        )
        
        assert request.pipeline == mock_pipeline
        assert request.context == {"input": "data"}
        assert request.execution_mode == ExecutionMode.NORMAL
        assert request.timeout == 3600
        assert isinstance(request.request_id, str)
        assert isinstance(request.timestamp, datetime)
    
    def test_execution_request_defaults(self):
        """Test execution request default values."""
        request = ExecutionRequest(pipeline="pipeline_content")
        
        assert request.pipeline == "pipeline_content"
        assert request.context is None
        assert request.execution_id is None
        assert request.execution_mode == ExecutionMode.NORMAL
        assert request.timeout is None
        assert request.max_retries == 3
        assert request.enable_recovery is True
        assert request.enable_checkpointing is True
        assert request.enable_monitoring is True
        assert request.progress_callback is None
        assert request.status_callback is None
        assert request.resource_limits is None
        assert request.environment_vars is None
        assert request.debug_mode is False
        assert request.user_id is None
    
    def test_execution_request_to_dict(self):
        """Test converting execution request to dictionary."""
        mock_pipeline = Mock(spec=Pipeline)
        mock_pipeline.id = "test_pipeline_123"
        
        request = ExecutionRequest(
            pipeline=mock_pipeline,
            execution_id="exec_123",
            timeout=1800,
            debug_mode=True
        )
        
        result = request.to_dict()
        
        assert result["pipeline"] == "test_pipeline_123"
        assert result["execution_id"] == "exec_123"
        assert result["timeout"] == 1800
        assert result["debug_mode"] is True
        assert result["execution_mode"] == "normal"


class TestResponseTypes:
    """Test response type definitions."""
    
    def test_api_response_success(self):
        """Test successful API response."""
        response = APIResponse[str](
            success=True,
            data="test_data",
            request_id="req_123",
            duration_ms=150.5
        )
        
        assert response.success is True
        assert response.data == "test_data"
        assert response.request_id == "req_123"
        assert response.duration_ms == 150.5
        assert isinstance(response.timestamp, datetime)
        assert response.error_code is None
        assert response.error_message is None
        assert response.warnings == []
        assert response.metadata == {}
    
    def test_api_response_error(self):
        """Test error API response."""
        response = APIResponse[None](
            success=False,
            error_code="VALIDATION_ERROR",
            error_message="Invalid input provided",
            error_details={"field": "yaml_content", "issue": "malformed"},
            warnings=["Deprecated feature used"]
        )
        
        assert response.success is False
        assert response.data is None
        assert response.error_code == "VALIDATION_ERROR"
        assert response.error_message == "Invalid input provided"
        assert response.error_details == {"field": "yaml_content", "issue": "malformed"}
        assert response.warnings == ["Deprecated feature used"]
    
    def test_api_response_to_dict(self):
        """Test converting API response to dictionary."""
        # Test with data object that has to_dict method
        mock_data = Mock()
        mock_data.to_dict.return_value = {"mock": "data"}
        
        response = APIResponse[Any](
            success=True,
            data=mock_data,
            request_id="req_456",
            warnings=["Warning message"]
        )
        
        result = response.to_dict()
        
        assert result["success"] is True
        assert result["data"] == {"mock": "data"}
        assert result["request_id"] == "req_456"
        assert result["warnings"] == ["Warning message"]
        assert "timestamp" in result
        
        # Test with simple data
        simple_response = APIResponse[str](
            success=True,
            data="simple_string"
        )
        
        simple_result = simple_response.to_dict()
        assert simple_result["data"] == "simple_string"
    
    def test_compilation_result_creation(self):
        """Test creating compilation result."""
        mock_pipeline = Mock(spec=Pipeline)
        mock_pipeline.id = "compiled_pipeline"
        
        result = CompilationResult(
            pipeline=mock_pipeline,
            compilation_time=timedelta(milliseconds=250),
            validation_passed=True,
            template_variables=["var1", "var2"],
            validation_warnings=["Minor issue"]
        )
        
        assert result.pipeline == mock_pipeline
        assert result.compilation_time == timedelta(milliseconds=250)
        assert result.validation_passed is True
        assert result.template_variables == ["var1", "var2"]
        assert result.validation_warnings == ["Minor issue"]
        assert isinstance(result.compilation_id, str)
        assert isinstance(result.compiled_at, datetime)
        assert result.compiler_version == "2.0.0"
    
    def test_compilation_result_to_dict(self):
        """Test converting compilation result to dictionary."""
        mock_pipeline = Mock(spec=Pipeline)
        mock_pipeline.id = "test_pipeline"
        mock_pipeline.name = "Test Pipeline"
        
        result = CompilationResult(
            pipeline=mock_pipeline,
            compilation_time=timedelta(seconds=1.5),
            validation_report={"status": "passed"},
            validation_errors=["Error 1"]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["pipeline_id"] == "test_pipeline"
        assert result_dict["pipeline_name"] == "Test Pipeline"
        assert result_dict["compilation_time_ms"] == 1500.0
        assert result_dict["validation_report"] == {"status": "passed"}
        assert result_dict["validation_errors"] == ["Error 1"]
        assert result_dict["compilation_id"] == result.compilation_id
        assert "compiled_at" in result_dict
    
    def test_execution_result_creation(self):
        """Test creating execution result."""
        start_time = datetime.now()
        
        result = ExecutionResult(
            execution_id="exec_789",
            pipeline_id="pipeline_456",
            status=ExecutionStatus.RUNNING,
            started_at=start_time,
            total_steps=5,
            monitoring_enabled=True
        )
        
        assert result.execution_id == "exec_789"
        assert result.pipeline_id == "pipeline_456"
        assert result.status == ExecutionStatus.RUNNING
        assert result.started_at == start_time
        assert result.total_steps == 5
        assert result.monitoring_enabled is True
        assert result.execution_mode == ExecutionMode.NORMAL  # Default
    
    def test_execution_result_to_dict(self):
        """Test converting execution result to dictionary."""
        start_time = datetime.now()
        estimated_duration = timedelta(minutes=5)
        
        result = ExecutionResult(
            execution_id="exec_xyz",
            pipeline_id="pipeline_abc",
            status=ExecutionStatus.PENDING,
            started_at=start_time,
            estimated_duration=estimated_duration,
            total_steps=10,
            progress_url="/progress",
            execution_mode=ExecutionMode.DEBUG
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["execution_id"] == "exec_xyz"
        assert result_dict["pipeline_id"] == "pipeline_abc"
        assert result_dict["status"] == "pending"
        assert result_dict["started_at"] == start_time.isoformat()
        assert result_dict["estimated_duration_seconds"] == 300.0
        assert result_dict["total_steps"] == 10
        assert result_dict["progress_url"] == "/progress"
        assert result_dict["execution_mode"] == "debug"


class TestStatusTypes:
    """Test status and progress type definitions."""
    
    def test_execution_status_info_creation(self):
        """Test creating execution status info."""
        start_time = datetime.now()
        update_time = datetime.now()
        
        status = ExecutionStatusInfo(
            execution_id="exec_status_test",
            pipeline_id="pipeline_status",
            status=ExecutionStatus.RUNNING,
            started_at=start_time,
            updated_at=update_time,
            current_step="step_2",
            steps_completed=2,
            steps_total=5,
            progress_percentage=40.0
        )
        
        assert status.execution_id == "exec_status_test"
        assert status.pipeline_id == "pipeline_status"
        assert status.status == ExecutionStatus.RUNNING
        assert status.started_at == start_time
        assert status.updated_at == update_time
        assert status.current_step == "step_2"
        assert status.steps_completed == 2
        assert status.steps_total == 5
        assert status.progress_percentage == 40.0
        assert status.completed_at is None
        assert status.duration is None
    
    def test_execution_status_info_to_dict(self):
        """Test converting execution status info to dictionary."""
        start_time = datetime.now()
        update_time = datetime.now()
        completed_time = datetime.now()
        duration = timedelta(minutes=2)
        
        mock_metrics = Mock(spec=ExecutionMetrics)
        mock_metrics.__dict__ = {"steps_completed": 3}
        
        status = ExecutionStatusInfo(
            execution_id="exec_dict_test",
            pipeline_id="pipeline_dict",
            status=ExecutionStatus.COMPLETED,
            started_at=start_time,
            updated_at=update_time,
            completed_at=completed_time,
            duration=duration,
            metrics=mock_metrics,
            step_statuses={"step1": StepStatus.COMPLETED, "step2": StepStatus.RUNNING},
            error_count=1,
            variables={"var1": "value1"}
        )
        
        result = status.to_dict()
        
        assert result["execution_id"] == "exec_dict_test"
        assert result["success"] == True
        assert result["started_at"] == start_time.isoformat()
        assert result["completed_at"] == completed_time.isoformat()
        assert result["duration_seconds"] == 120.0
        assert result["metrics"] == {"steps_completed": 3}
        assert result["step_statuses"] == {"step1": "completed", "step2": "running"}
        assert result["error_count"] == 1
        assert result["variables"] == {"var1": "value1"}
    
    def test_progress_update_creation(self):
        """Test creating progress update."""
        timestamp = datetime.now()
        
        update = ProgressUpdate(
            execution_id="exec_progress",
            timestamp=timestamp,
            event_type=ProgressEventType.STEP_COMPLETED,
            step_id="step_123",
            step_name="Test Step",
            message="Step completed successfully",
            step_progress=100.0,
            overall_progress=60.0,
            steps_completed=3,
            steps_total=5
        )
        
        assert update.execution_id == "exec_progress"
        assert update.timestamp == timestamp
        assert update.event_type == ProgressEventType.STEP_COMPLETED
        assert update.step_id == "step_123"
        assert update.step_name == "Test Step"
        assert update.message == "Step completed successfully"
        assert update.step_progress == 100.0
        assert update.overall_progress == 60.0
        assert update.steps_completed == 3
        assert update.steps_total == 5
    
    def test_progress_update_to_dict(self):
        """Test converting progress update to dictionary."""
        timestamp = datetime.now()
        
        update = ProgressUpdate(
            execution_id="exec_update_dict",
            timestamp=timestamp,
            event_type=ProgressEventType.STEP_STARTED,
            step_name="Processing Step",
            data={"processed_items": 10},
            metadata={"source": "executor"}
        )
        
        result = update.to_dict()
        
        assert result["execution_id"] == "exec_update_dict"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["event_type"] == "step_started"
        assert result["step_name"] == "Processing Step"
        assert result["data"] == {"processed_items": 10}
        assert result["metadata"] == {"source": "executor"}


class TestConfigurationTypes:
    """Test configuration type definitions."""
    
    def test_api_configuration_defaults(self):
        """Test API configuration default values."""
        config = APIConfiguration()
        
        assert config.model_registry_config is None
        assert config.auto_model_selection is True
        assert config.default_validation_level == ValidationLevel.STRICT
        assert config.enable_validation_caching is True
        assert config.validation_timeout == 30
        assert config.default_execution_timeout == 3600
        assert config.max_concurrent_executions == 10
        assert config.enable_execution_recovery is True
        assert config.enable_execution_checkpointing is True
        assert config.enable_compilation_caching is True
        assert config.cache_size_limit == 1000
        assert config.memory_limit_mb is None
        assert config.enable_detailed_monitoring is True
        assert config.progress_update_interval == 5
        assert config.status_cleanup_interval == 3600
        assert config.log_level == "INFO"
        assert config.log_format == "structured"
        assert config.enable_audit_logging is True
        assert config.enable_authentication is False
        assert config.api_key_required is False
        assert config.rate_limiting is None
    
    def test_api_configuration_custom(self):
        """Test API configuration with custom values."""
        config = APIConfiguration(
            default_validation_level=ValidationLevel.DEVELOPMENT,
            max_concurrent_executions=5,
            memory_limit_mb=2048,
            log_level="DEBUG",
            enable_authentication=True,
            rate_limiting={"requests_per_minute": 100}
        )
        
        assert config.default_validation_level == ValidationLevel.DEVELOPMENT
        assert config.max_concurrent_executions == 5
        assert config.memory_limit_mb == 2048
        assert config.log_level == "DEBUG"
        assert config.enable_authentication is True
        assert config.rate_limiting == {"requests_per_minute": 100}
    
    def test_api_configuration_to_dict(self):
        """Test converting API configuration to dictionary."""
        config = APIConfiguration(
            model_registry_config={"type": "local"},
            default_validation_level=ValidationLevel.PERMISSIVE,
            memory_limit_mb=4096
        )
        
        result = config.to_dict()
        
        assert result["model_registry_config"] == {"type": "local"}
        assert result["default_validation_level"] == "permissive"
        assert result["memory_limit_mb"] == 4096
        assert result["auto_model_selection"] is True  # Default


class TestProtocolTypes:
    """Test protocol type definitions."""
    
    def test_pipeline_compiler_protocol(self):
        """Test pipeline compiler protocol structure."""
        # Verify protocol methods exist
        protocol_methods = dir(PipelineCompilerProtocol)
        
        assert "compile" in protocol_methods
        assert "validate_yaml" in protocol_methods
        assert "get_template_variables" in protocol_methods
    
    def test_execution_manager_protocol(self):
        """Test execution manager protocol structure."""
        protocol_methods = dir(ExecutionManagerProtocol)
        
        assert "get_execution_status" in protocol_methods
        assert "start_execution" in protocol_methods
        assert "complete_execution" in protocol_methods
        assert "cleanup" in protocol_methods
    
    def test_progress_monitor_protocol(self):
        """Test progress monitor protocol structure."""
        protocol_methods = dir(ProgressMonitorProtocol)
        
        assert "start_monitoring" in protocol_methods
        assert "get_progress_updates" in protocol_methods
        assert "stop_monitoring" in protocol_methods


class TestTypedDictTypes:
    """Test TypedDict type definitions."""
    
    def test_pipeline_compilation_dict_structure(self):
        """Test PipelineCompilationDict structure."""
        # Create a valid compilation dict
        compilation_dict: PipelineCompilationDict = {
            "yaml_content": "steps:\n  - name: test",
            "context": {"var": "value"},
            "resolve_ambiguities": True,
            "validate": True,
            "validation_level": "strict",
            "compilation_mode": "standard",
            "request_id": "req_123"
        }
        
        assert compilation_dict["yaml_content"] == "steps:\n  - name: test"
        assert compilation_dict["context"] == {"var": "value"}
        assert compilation_dict["validation_level"] == "strict"
    
    def test_execution_status_dict_structure(self):
        """Test ExecutionStatusDict structure."""
        status_dict: ExecutionStatusDict = {
            "execution_id": "exec_123",
            "pipeline_id": "pipeline_456",
            "status": "running",
            "started_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:05:00",
            "progress_percentage": 50.0,
            "steps_completed": 2,
            "steps_total": 4,
            "step_statuses": {"step1": "completed", "step2": "running"},
            "error_count": 0,
            "variables": {"output": "result"}
        }
        
        assert status_dict["execution_id"] == "exec_123"
        assert status_dict["status"] == "running"
        assert status_dict["progress_percentage"] == 50.0
        assert status_dict["variables"] == {"output": "result"}
    
    def test_validation_result_structure(self):
        """Test ValidationResult structure."""
        validation_result: ValidationResult = {
            "valid": True,
            "errors": [],
            "warnings": ["Minor issue"],
            "info": ["Validation completed"]
        }
        
        assert validation_result["valid"] is True
        assert validation_result["errors"] == []
        assert validation_result["warnings"] == ["Minor issue"]
        assert validation_result["info"] == ["Validation completed"]
    
    def test_resource_usage_structure(self):
        """Test ResourceUsage structure."""
        resource_usage: ResourceUsage = {
            "memory_mb": 512.5,
            "cpu_percent": 25.0,
            "disk_mb": 100.0,
            "network_kb": 50.2,
            "execution_time_seconds": 120.5
        }
        
        assert resource_usage["memory_mb"] == 512.5
        assert resource_usage["cpu_percent"] == 25.0
        assert resource_usage["execution_time_seconds"] == 120.5
    
    def test_step_summary_structure(self):
        """Test StepSummary structure."""
        step_summary: StepSummary = {
            "step_id": "step_abc",
            "step_name": "Process Data",
            "step_type": "data_processor",
            "status": "completed",
            "progress": 100.0,
            "duration_seconds": 45.2,
            "error_message": None,
            "resource_usage": {
                "memory_mb": 256.0,
                "cpu_percent": 15.0,
                "disk_mb": 20.0,
                "network_kb": 10.5,
                "execution_time_seconds": 45.2
            }
        }
        
        assert step_summary["step_id"] == "step_abc"
        assert step_summary["step_name"] == "Process Data"
        assert step_summary["status"] == "completed"
        assert step_summary["progress"] == 100.0
        assert step_summary["error_message"] is None


class TestDocumentationTypes:
    """Test documentation type definitions."""
    
    def test_api_endpoint_creation(self):
        """Test creating API endpoint documentation."""
        endpoint = APIEndpoint(
            name="test_endpoint",
            method="POST",
            path="/api/test",
            description="Test endpoint for validation",
            request_type=CompilationRequest,
            response_type=CompilationResult,
            parameters=["param1: string", "param2: optional int"],
            error_codes=["VALIDATION_ERROR", "TIMEOUT_ERROR"],
            examples=[{"test": "example"}]
        )
        
        assert endpoint.name == "test_endpoint"
        assert endpoint.method == "POST"
        assert endpoint.path == "/api/test"
        assert endpoint.description == "Test endpoint for validation"
        assert endpoint.request_type == CompilationRequest
        assert endpoint.response_type == CompilationResult
        assert endpoint.parameters == ["param1: string", "param2: optional int"]
        assert endpoint.error_codes == ["VALIDATION_ERROR", "TIMEOUT_ERROR"]
        assert endpoint.examples == [{"test": "example"}]
    
    def test_api_endpoint_to_dict(self):
        """Test converting API endpoint to dictionary."""
        endpoint = APIEndpoint(
            name="dict_test",
            method="GET",
            path="/test/dict",
            description="Dictionary conversion test",
            request_schema={"field": "string"},
            response_example={"result": "success"}
        )
        
        result = endpoint.to_dict()
        
        assert result["name"] == "dict_test"
        assert result["method"] == "GET"
        assert result["path"] == "/test/dict"
        assert result["description"] == "Dictionary conversion test"
        assert result["request_schema"] == {"field": "string"}
        assert result["response_example"] == {"result": "success"}
    
    def test_api_documentation_structure(self):
        """Test API_DOCUMENTATION structure."""
        assert "title" in API_DOCUMENTATION
        assert "version" in API_DOCUMENTATION
        assert "description" in API_DOCUMENTATION
        assert "endpoints" in API_DOCUMENTATION
        
        assert API_DOCUMENTATION["title"] == "Orchestrator API Framework"
        assert API_DOCUMENTATION["version"] == "2.0.0"
        assert isinstance(API_DOCUMENTATION["endpoints"], list)
        assert len(API_DOCUMENTATION["endpoints"]) > 0
        
        # Check first endpoint structure
        first_endpoint = API_DOCUMENTATION["endpoints"][0]
        assert isinstance(first_endpoint, APIEndpoint)
        assert hasattr(first_endpoint, "name")
        assert hasattr(first_endpoint, "method")
        assert hasattr(first_endpoint, "path")
        assert hasattr(first_endpoint, "description")


class TestResponseTypeAliases:
    """Test response type aliases."""
    
    def test_pipeline_compilation_response_alias(self):
        """Test PipelineCompilationResponse type alias."""
        mock_pipeline = Mock(spec=Pipeline)
        compilation_result = CompilationResult(
            pipeline=mock_pipeline,
            compilation_time=timedelta(seconds=1)
        )
        
        response: PipelineCompilationResponse = APIResponse[CompilationResult](
            success=True,
            data=compilation_result
        )
        
        assert response.success is True
        assert response.data == compilation_result
        assert isinstance(response, APIResponse)
    
    def test_execution_status_response_alias(self):
        """Test ExecutionStatusResponse type alias."""
        status_info = ExecutionStatusInfo(
            execution_id="test_exec",
            pipeline_id="test_pipeline",
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        response: ExecutionStatusResponse = APIResponse[ExecutionStatusInfo](
            success=True,
            data=status_info
        )
        
        assert response.success is True
        assert response.data == status_info
        assert isinstance(response, APIResponse)


if __name__ == "__main__":
    pytest.main([__file__])