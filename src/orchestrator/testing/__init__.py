"""Pipeline testing infrastructure for orchestrator."""

from .pipeline_discovery import PipelineDiscovery, PipelineInfo
from .pipeline_test_suite import (
    PipelineTestSuite, 
    ExecutionResult,
    TemplateResult, 
    OrganizationResult,
    PerformanceResult,
    PipelineTestResult,
    TestResults
)
from .test_input_manager import TestInputManager
from .pipeline_validator import PipelineValidator
from .test_reporter import PipelineTestReporter

# Stream D: CI/CD Integration & Test Modes
from .test_modes import (
    TestMode,
    TestModeConfig,
    TestModeManager,
    TestSuiteComposition,
    select_pipelines_for_mode,
    get_available_test_modes
)
from .ci_cd_integration import (
    CIIntegrationManager,
    CIConfiguration,
    CITestResult,
    CITestSummary,
    CISystem,
    TestStatus,
    create_ci_config_from_environment
)
from .release_validator import (
    ReleaseValidator,
    ReleaseType,
    ValidationLevel,
    ReleaseValidationCriteria,
    ValidationResult,
    determine_release_type_from_version,
    create_release_validation_report
)
from .production_automation import (
    ProductionAutomationManager,
    ScheduleConfig,
    AutomationResult,
    AlertConfig,
    ScheduleType,
    AlertSeverity,
    AutomationStatus,
    create_default_production_schedules
)

__all__ = [
    # Core infrastructure (Streams A, B, C)
    'PipelineDiscovery',
    'PipelineInfo',
    'PipelineTestSuite',
    'ExecutionResult',
    'TemplateResult',
    'OrganizationResult', 
    'PerformanceResult',
    'PipelineTestResult',
    'TestResults',
    'TestInputManager',
    'PipelineValidator',
    'PipelineTestReporter',
    
    # Stream D: CI/CD Integration & Test Modes
    'TestMode',
    'TestModeConfig',
    'TestModeManager',
    'TestSuiteComposition',
    'select_pipelines_for_mode',
    'get_available_test_modes',
    'CIIntegrationManager',
    'CIConfiguration',
    'CITestResult',
    'CITestSummary',
    'CISystem',
    'TestStatus',
    'create_ci_config_from_environment',
    'ReleaseValidator',
    'ReleaseType',
    'ValidationLevel', 
    'ReleaseValidationCriteria',
    'ValidationResult',
    'determine_release_type_from_version',
    'create_release_validation_report',
    'ProductionAutomationManager',
    'ScheduleConfig',
    'AutomationResult',
    'AlertConfig',
    'ScheduleType',
    'AlertSeverity',
    'AutomationStatus',
    'create_default_production_schedules'
]