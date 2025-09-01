"""
Output Validation and Quality Control System.

This package provides automated output validation, configurable rule engines,
and real-time quality control for pipeline execution outputs. It integrates
with the execution engine to provide continuous quality assurance.

Key Components:
- validator.py: Core validation engine with execution context integration
- rules.py: Configurable validation rules for extensible quality checks  
- engine.py: Rule execution engine for flexible validation processing
- integration.py: Real-time integration with pipeline execution

The system provides:
- Real-time output validation during pipeline execution
- Configurable quality rules and thresholds
- Integration with execution context and metrics
- Extensible rule framework for custom quality checks
- Quality control reporting and analytics foundation
- Stream C compatibility for reporting and analytics
"""

from .validator import OutputQualityValidator, ValidationResult, ValidationSeverity
from .rules import (
    ValidationRule, QualityRule, RuleRegistry, RuleViolation, ValidationContext,
    RuleSeverity, RuleCategory, FileSizeRule, ContentFormatRule, 
    ContentQualityRule, PerformanceRule
)
from .engine import ValidationEngine, RuleExecutionContext, ValidationSession
from .integration import (
    ExecutionQualityMonitor, QualityControlManager, ValidationTrigger,
    create_quality_control_manager
)

__all__ = [
    # Core validation components
    "OutputQualityValidator",
    "ValidationResult", 
    "ValidationSeverity",
    
    # Rule system
    "ValidationRule",
    "QualityRule",
    "RuleRegistry",
    "RuleViolation",
    "ValidationContext",
    "RuleSeverity",
    "RuleCategory",
    
    # Built-in rules
    "FileSizeRule",
    "ContentFormatRule", 
    "ContentQualityRule",
    "PerformanceRule",
    
    # Execution engine
    "ValidationEngine",
    "RuleExecutionContext",
    "ValidationSession",
    
    # Integration and monitoring
    "ExecutionQualityMonitor",
    "QualityControlManager", 
    "ValidationTrigger",
    "create_quality_control_manager"
]