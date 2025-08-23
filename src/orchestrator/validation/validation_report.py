"""
Unified validation reporting for pipeline compilation.

This module provides comprehensive validation reporting that aggregates results from all validators:
- Template validation (TemplateValidator)
- Tool validation (ToolValidator) 
- Dependency validation (DependencyValidator)
- Model validation (ModelValidator)
- Output validation (OutputValidator)

Features:
- Structured validation report format
- Multiple output formats (text, JSON, detailed/summary)
- Color-coded CLI output by severity
- Error grouping by type and component
- Actionable fix suggestions for common issues
- Configurable validation levels (strict/permissive/development)
- Clear, user-friendly error messages

Issue #241: Enhanced Validation Reporting
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"           # All validations must pass, no bypasses
    PERMISSIVE = "permissive"   # Some validations can be warnings instead of errors
    DEVELOPMENT = "development"  # Maximum bypasses for development workflow


class OutputFormat(Enum):
    """Validation report output formats."""
    TEXT = "text"               # Human-readable text format
    JSON = "json"               # Machine-readable JSON format
    DETAILED = "detailed"       # Detailed text with full context
    SUMMARY = "summary"         # Brief summary format


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    ERROR = "error"             # Blocking issues that prevent execution
    WARNING = "warning"         # Non-blocking issues that should be addressed
    INFO = "info"               # Informational messages
    DEBUG = "debug"             # Debug-level information


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    
    severity: ValidationSeverity
    category: str               # e.g., "template", "tool", "dependency", "model"
    component: str              # e.g., task ID, parameter name, etc.
    message: str                # Human-readable description
    code: Optional[str] = None  # Machine-readable error code
    path: Optional[str] = None  # Context path (e.g., "steps[0].parameters.prompt")
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if isinstance(self.severity, str):
            self.severity = ValidationSeverity(self.severity)
    
    @property
    def is_error(self) -> bool:
        """Check if this is an error-level issue."""
        return self.severity == ValidationSeverity.ERROR
    
    @property
    def is_warning(self) -> bool:
        """Check if this is a warning-level issue."""
        return self.severity == ValidationSeverity.WARNING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "severity": self.severity.value,
            "category": self.category,
            "component": self.component,
            "message": self.message,
            "code": self.code,
            "path": self.path,
            "suggestions": self.suggestions,
            "metadata": self.metadata
        }


@dataclass
class ValidationStats:
    """Statistics about validation results."""
    
    total_issues: int = 0
    errors: int = 0
    warnings: int = 0
    infos: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    components: Dict[str, int] = field(default_factory=dict)
    
    def add_issue(self, issue: ValidationIssue):
        """Add an issue to the statistics."""
        self.total_issues += 1
        
        if issue.severity == ValidationSeverity.ERROR:
            self.errors += 1
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings += 1
        elif issue.severity == ValidationSeverity.INFO:
            self.infos += 1
        
        # Count by category
        self.categories[issue.category] = self.categories.get(issue.category, 0) + 1
        
        # Count by component
        if issue.component:
            self.components[issue.component] = self.components.get(issue.component, 0) + 1


class ValidationReport:
    """
    Comprehensive validation report that aggregates results from all validators.
    
    This class provides a unified interface for validation reporting with:
    - Structured validation results
    - Multiple output formats
    - Severity-based filtering
    - Category-based grouping
    - Actionable suggestions
    - Configurable validation levels
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        """
        Initialize validation report.
        
        Args:
            validation_level: Validation strictness level
        """
        self.validation_level = validation_level
        self.issues: List[ValidationIssue] = []
        self.stats = ValidationStats()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.pipeline_id: Optional[str] = None
        self.context: Dict[str, Any] = {}
        
        logger.debug(f"Initialized ValidationReport with level: {validation_level.value}")
    
    def start_validation(self, pipeline_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Mark the start of validation process."""
        import time
        self.start_time = time.time()
        self.pipeline_id = pipeline_id
        self.context = context or {}
        logger.info(f"Started validation for pipeline: {pipeline_id}")
    
    def end_validation(self):
        """Mark the end of validation process."""
        import time
        self.end_time = time.time()
        duration = self.duration if self.start_time else None
        logger.info(f"Completed validation in {duration:.2f}s" if duration else "Completed validation")
    
    @property
    def duration(self) -> Optional[float]:
        """Get validation duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return self.stats.errors == 0
    
    @property
    def has_errors(self) -> bool:
        """Check if there are error-level issues."""
        return self.stats.errors > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are warning-level issues."""
        return self.stats.warnings > 0
    
    @property
    def has_issues(self) -> bool:
        """Check if there are any issues."""
        return self.stats.total_issues > 0
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue to the report."""
        # Apply validation level adjustments
        adjusted_issue = self._adjust_issue_severity(issue)
        
        self.issues.append(adjusted_issue)
        self.stats.add_issue(adjusted_issue)
        
        if adjusted_issue.severity == ValidationSeverity.ERROR:
            logger.error(f"Validation error in {adjusted_issue.category}: {adjusted_issue.message}")
        elif adjusted_issue.severity == ValidationSeverity.WARNING:
            logger.warning(f"Validation warning in {adjusted_issue.category}: {adjusted_issue.message}")
    
    def add_issues(self, issues: List[ValidationIssue]):
        """Add multiple validation issues to the report."""
        for issue in issues:
            self.add_issue(issue)
    
    def _adjust_issue_severity(self, issue: ValidationIssue) -> ValidationIssue:
        """Adjust issue severity based on validation level."""
        if self.validation_level == ValidationLevel.STRICT:
            # No adjustments in strict mode
            return issue
        
        elif self.validation_level == ValidationLevel.PERMISSIVE:
            # Convert some errors to warnings in permissive mode
            if issue.severity == ValidationSeverity.ERROR and issue.category in ["tool", "model"]:
                # Tool and model validation can be more permissive
                if issue.code in ["unknown_tool", "missing_model", "type_mismatch", "format_mismatch"]:
                    adjusted = ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=issue.category,
                        component=issue.component,
                        message=f"[PERMISSIVE] {issue.message}",
                        code=issue.code,
                        path=issue.path,
                        suggestions=issue.suggestions,
                        metadata=issue.metadata
                    )
                    return adjusted
        
        elif self.validation_level == ValidationLevel.DEVELOPMENT:
            # Maximum bypasses in development mode
            if issue.severity == ValidationSeverity.ERROR:
                # Convert many error types to warnings in development mode
                bypassable_categories = {"tool", "model", "template"}
                bypassable_codes = {
                    "unknown_tool", "missing_model", "type_mismatch", "format_mismatch",
                    "undefined_variable", "unknown_parameter", "missing_dependency"
                }
                
                if issue.category in bypassable_categories or issue.code in bypassable_codes:
                    adjusted = ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=issue.category,
                        component=issue.component,
                        message=f"[DEV MODE] {issue.message}",
                        code=issue.code,
                        path=issue.path,
                        suggestions=issue.suggestions + ["Enable strict validation for production"],
                        metadata=issue.metadata
                    )
                    return adjusted
        
        return issue
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get all issues of a specific severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: str) -> List[ValidationIssue]:
        """Get all issues of a specific category."""
        return [issue for issue in self.issues if issue.category == category]
    
    def get_issues_by_component(self, component: str) -> List[ValidationIssue]:
        """Get all issues for a specific component."""
        return [issue for issue in self.issues if issue.component == component]
    
    def get_grouped_issues(self) -> Dict[str, Dict[str, List[ValidationIssue]]]:
        """Group issues by category and then by severity."""
        grouped = {}
        
        for issue in self.issues:
            if issue.category not in grouped:
                grouped[issue.category] = {
                    "errors": [],
                    "warnings": [],
                    "infos": []
                }
            
            if issue.severity == ValidationSeverity.ERROR:
                grouped[issue.category]["errors"].append(issue)
            elif issue.severity == ValidationSeverity.WARNING:
                grouped[issue.category]["warnings"].append(issue)
            elif issue.severity == ValidationSeverity.INFO:
                grouped[issue.category]["infos"].append(issue)
        
        return grouped
    
    def format_report(self, format_type: OutputFormat = OutputFormat.TEXT, 
                     include_suggestions: bool = True, include_context: bool = True,
                     max_width: int = 80) -> str:
        """
        Format the validation report in the specified format.
        
        Args:
            format_type: Output format type
            include_suggestions: Whether to include fix suggestions
            include_context: Whether to include context information
            max_width: Maximum line width for text formatting
            
        Returns:
            Formatted report string
        """
        if format_type == OutputFormat.JSON:
            return self._format_json()
        elif format_type == OutputFormat.SUMMARY:
            return self._format_summary(max_width)
        elif format_type == OutputFormat.DETAILED:
            return self._format_detailed(include_suggestions, include_context, max_width)
        else:  # TEXT
            return self._format_text(include_suggestions, max_width)
    
    def _format_json(self) -> str:
        """Format report as JSON."""
        data = {
            "pipeline_id": self.pipeline_id,
            "validation_level": self.validation_level.value,
            "is_valid": self.is_valid,
            "duration": self.duration,
            "stats": {
                "total_issues": self.stats.total_issues,
                "errors": self.stats.errors,
                "warnings": self.stats.warnings,
                "infos": self.stats.infos,
                "by_category": dict(self.stats.categories),
                "by_component": dict(self.stats.components)
            },
            "issues": [issue.to_dict() for issue in self.issues],
            "context": self.context
        }
        return json.dumps(data, indent=2, default=str)
    
    def _format_summary(self, max_width: int) -> str:
        """Format report as brief summary."""
        lines = []
        
        # Header
        status = "✓ PASSED" if self.is_valid else "✗ FAILED"
        level_str = f"({self.validation_level.value.upper()})"
        duration_str = f" in {self.duration:.2f}s" if self.duration else ""
        
        header = f"Validation {status} {level_str}{duration_str}"
        lines.append(header)
        lines.append("=" * min(len(header), max_width))
        
        # Statistics
        if self.has_issues:
            stats_parts = []
            if self.stats.errors > 0:
                stats_parts.append(f"{self.stats.errors} errors")
            if self.stats.warnings > 0:
                stats_parts.append(f"{self.stats.warnings} warnings")
            if self.stats.infos > 0:
                stats_parts.append(f"{self.stats.infos} infos")
            
            lines.append(f"Issues: {', '.join(stats_parts)}")
            
            # Category breakdown
            if self.stats.categories:
                category_parts = [f"{cat}: {count}" for cat, count in self.stats.categories.items()]
                lines.append(f"By category: {', '.join(category_parts)}")
        else:
            lines.append("No issues found")
        
        return "\n".join(lines)
    
    def _format_text(self, include_suggestions: bool, max_width: int) -> str:
        """Format report as readable text."""
        lines = []
        
        # Header with validation summary
        lines.extend(self._format_header(max_width))
        lines.append("")
        
        if not self.has_issues:
            lines.append("✓ No validation issues found")
            return "\n".join(lines)
        
        # Group issues by category
        grouped = self.get_grouped_issues()
        
        for category in sorted(grouped.keys()):
            category_issues = grouped[category]
            
            # Category header
            total_category_issues = sum(len(issues) for issues in category_issues.values())
            lines.append(f"🔍 {category.upper()} VALIDATION ({total_category_issues} issues)")
            lines.append("-" * min(max_width, len(lines[-1])))
            
            # Errors first
            if category_issues["errors"]:
                lines.append("")
                lines.append("❌ Errors:")
                for issue in category_issues["errors"]:
                    lines.extend(self._format_issue(issue, include_suggestions, max_width, "  "))
            
            # Then warnings
            if category_issues["warnings"]:
                lines.append("")
                lines.append("⚠️  Warnings:")
                for issue in category_issues["warnings"]:
                    lines.extend(self._format_issue(issue, include_suggestions, max_width, "  "))
            
            # Then infos
            if category_issues["infos"]:
                lines.append("")
                lines.append("ℹ️  Info:")
                for issue in category_issues["infos"]:
                    lines.extend(self._format_issue(issue, include_suggestions, max_width, "  "))
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_detailed(self, include_suggestions: bool, include_context: bool, max_width: int) -> str:
        """Format report with full details."""
        lines = []
        
        # Header with validation summary
        lines.extend(self._format_header(max_width))
        lines.append("")
        
        # Context information
        if include_context and self.context:
            lines.append("📋 CONTEXT")
            lines.append("-" * min(max_width, 12))
            for key, value in self.context.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        if not self.has_issues:
            lines.append("✓ No validation issues found")
            return "\n".join(lines)
        
        # Detailed issue listing
        lines.append("🔍 DETAILED ISSUES")
        lines.append("=" * min(max_width, 18))
        
        for i, issue in enumerate(self.issues, 1):
            lines.append("")
            lines.append(f"Issue #{i}")
            lines.append("-" * min(max_width, 10))
            lines.extend(self._format_issue_detailed(issue, include_suggestions, max_width))
        
        return "\n".join(lines)
    
    def _format_header(self, max_width: int) -> List[str]:
        """Format the report header."""
        lines = []
        
        # Title line
        pipeline_str = f" - {self.pipeline_id}" if self.pipeline_id else ""
        level_str = f" ({self.validation_level.value.upper()})"
        title = f"VALIDATION REPORT{pipeline_str}{level_str}"
        lines.append(title)
        lines.append("=" * min(len(title), max_width))
        
        # Status line
        if self.is_valid:
            status_line = "✅ Status: PASSED"
        else:
            status_line = "❌ Status: FAILED"
        
        if self.duration:
            status_line += f" (completed in {self.duration:.2f}s)"
        
        lines.append(status_line)
        
        # Statistics
        if self.has_issues:
            stats = []
            if self.stats.errors > 0:
                stats.append(f"❌ {self.stats.errors} errors")
            if self.stats.warnings > 0:
                stats.append(f"⚠️ {self.stats.warnings} warnings")
            if self.stats.infos > 0:
                stats.append(f"ℹ️ {self.stats.infos} infos")
            
            lines.append(f"📊 Issues: {', '.join(stats)}")
        
        return lines
    
    def _format_issue(self, issue: ValidationIssue, include_suggestions: bool, 
                     max_width: int, indent: str = "") -> List[str]:
        """Format a single issue."""
        lines = []
        
        # Main issue line
        component_str = f" [{issue.component}]" if issue.component else ""
        path_str = f" at {issue.path}" if issue.path else ""
        
        issue_line = f"{indent}• {issue.message}{component_str}{path_str}"
        lines.append(self._wrap_line(issue_line, max_width, indent))
        
        # Suggestions
        if include_suggestions and issue.suggestions:
            for suggestion in issue.suggestions:
                suggestion_line = f"{indent}  💡 {suggestion}"
                lines.append(self._wrap_line(suggestion_line, max_width, f"{indent}     "))
        
        return lines
    
    def _format_issue_detailed(self, issue: ValidationIssue, include_suggestions: bool, 
                              max_width: int) -> List[str]:
        """Format a single issue with full details."""
        lines = []
        
        # Severity and category
        severity_icon = {
            ValidationSeverity.ERROR: "❌",
            ValidationSeverity.WARNING: "⚠️",
            ValidationSeverity.INFO: "ℹ️",
            ValidationSeverity.DEBUG: "🔍"
        }.get(issue.severity, "")
        
        lines.append(f"  Severity: {severity_icon} {issue.severity.value.upper()}")
        lines.append(f"  Category: {issue.category}")
        
        if issue.component:
            lines.append(f"  Component: {issue.component}")
        
        if issue.path:
            lines.append(f"  Path: {issue.path}")
        
        if issue.code:
            lines.append(f"  Code: {issue.code}")
        
        lines.append(f"  Message: {issue.message}")
        
        # Suggestions
        if include_suggestions and issue.suggestions:
            lines.append("  Suggestions:")
            for suggestion in issue.suggestions:
                lines.append(f"    💡 {suggestion}")
        
        # Metadata
        if issue.metadata:
            lines.append("  Metadata:")
            for key, value in issue.metadata.items():
                lines.append(f"    {key}: {value}")
        
        return lines
    
    def _wrap_line(self, line: str, max_width: int, continuation_indent: str) -> str:
        """Wrap a line to maximum width with continuation indent."""
        if len(line) <= max_width:
            return line
        
        # Simple word wrapping
        words = line.split()
        wrapped_lines = []
        current_line = words[0] if words else ""
        
        for word in words[1:]:
            if len(current_line) + 1 + len(word) <= max_width:
                current_line += f" {word}"
            else:
                wrapped_lines.append(current_line)
                current_line = f"{continuation_indent}{word}"
        
        if current_line:
            wrapped_lines.append(current_line)
        
        return "\n".join(wrapped_lines)
    
    def save_to_file(self, file_path: Union[str, Path], format_type: OutputFormat = OutputFormat.JSON):
        """Save the validation report to a file."""
        file_path = Path(file_path)
        
        # Determine format from extension if not specified
        if format_type == OutputFormat.JSON or file_path.suffix.lower() == '.json':
            content = self._format_json()
        else:
            content = self.format_report(format_type)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Validation report saved to: {file_path}")
    
    def print_report(self, format_type: OutputFormat = OutputFormat.TEXT, 
                    include_suggestions: bool = True, use_colors: bool = True):
        """Print the validation report to console with optional color coding."""
        report = self.format_report(format_type, include_suggestions)
        
        if use_colors and format_type != OutputFormat.JSON:
            report = self._add_colors(report)
        
        print(report)
    
    def _add_colors(self, text: str) -> str:
        """Add color codes to text for terminal output."""
        try:
            # Try to use colorama for cross-platform color support
            from colorama import Fore, Back, Style, init
            init()
            
            # Color mappings
            color_replacements = [
                ("✅", f"{Fore.GREEN}✅{Style.RESET_ALL}"),
                ("❌", f"{Fore.RED}❌{Style.RESET_ALL}"),
                ("⚠️", f"{Fore.YELLOW}⚠️{Style.RESET_ALL}"),
                ("ℹ️", f"{Fore.BLUE}ℹ️{Style.RESET_ALL}"),
                ("🔍", f"{Fore.CYAN}🔍{Style.RESET_ALL}"),
                ("💡", f"{Fore.MAGENTA}💡{Style.RESET_ALL}"),
                ("FAILED", f"{Fore.RED}FAILED{Style.RESET_ALL}"),
                ("PASSED", f"{Fore.GREEN}PASSED{Style.RESET_ALL}"),
                ("ERRORS:", f"{Fore.RED}ERRORS:{Style.RESET_ALL}"),
                ("WARNINGS:", f"{Fore.YELLOW}WARNINGS:{Style.RESET_ALL}"),
            ]
            
            for old, new in color_replacements:
                text = text.replace(old, new)
            
            return text
            
        except ImportError:
            # Colorama not available, return text as-is
            logger.debug("Colorama not available, using plain text output")
            return text
    
    def clear(self):
        """Clear all issues and reset the report."""
        self.issues.clear()
        self.stats = ValidationStats()
        self.start_time = None
        self.end_time = None
        self.pipeline_id = None
        self.context = {}
        logger.debug("Validation report cleared")


# Convenience functions for creating validation issues

def create_template_issue(severity: ValidationSeverity, component: str, message: str,
                         path: Optional[str] = None, suggestions: Optional[List[str]] = None) -> ValidationIssue:
    """Create a template validation issue."""
    return ValidationIssue(
        severity=severity,
        category="template",
        component=component,
        message=message,
        path=path,
        suggestions=suggestions or [],
        code="template_error"
    )


def create_tool_issue(severity: ValidationSeverity, component: str, message: str,
                      tool_name: str, parameter: Optional[str] = None,
                      suggestions: Optional[List[str]] = None) -> ValidationIssue:
    """Create a tool validation issue."""
    return ValidationIssue(
        severity=severity,
        category="tool",
        component=component,
        message=message,
        path=f"tool.{tool_name}" + (f".{parameter}" if parameter else ""),
        suggestions=suggestions or [],
        code="tool_error",
        metadata={"tool_name": tool_name, "parameter": parameter}
    )


def create_dependency_issue(severity: ValidationSeverity, component: str, message: str,
                           dependency_chain: Optional[List[str]] = None,
                           suggestions: Optional[List[str]] = None) -> ValidationIssue:
    """Create a dependency validation issue."""
    return ValidationIssue(
        severity=severity,
        category="dependency",
        component=component,
        message=message,
        suggestions=suggestions or [],
        code="dependency_error",
        metadata={"dependency_chain": dependency_chain}
    )


def create_model_issue(severity: ValidationSeverity, component: str, message: str,
                       model_name: Optional[str] = None, capability: Optional[str] = None,
                       suggestions: Optional[List[str]] = None) -> ValidationIssue:
    """Create a model validation issue."""
    return ValidationIssue(
        severity=severity,
        category="model",
        component=component,
        message=message,
        path=f"model.{model_name}" if model_name else None,
        suggestions=suggestions or [],
        code="model_error",
        metadata={"model_name": model_name, "capability": capability}
    )