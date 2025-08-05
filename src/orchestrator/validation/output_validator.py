"""
Advanced validation system for output tracking and pipeline consistency.
Provides comprehensive validation rules and automatic issue detection.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..core.output_metadata import OutputMetadata, OutputInfo
from ..core.output_tracker import OutputTracker


@dataclass
class ValidationRule:
    """Represents a validation rule for output tracking."""
    
    name: str
    description: str
    severity: str = "error"  # "error", "warning", "info"
    category: str = "general"  # "consistency", "format", "dependency", etc.
    
    def validate(self, context: Dict[str, Any]) -> List[str]:
        """Validate against the rule. Override in subclasses."""
        return []


@dataclass
class ValidationResult:
    """Result of validation process."""
    
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    
    @property
    def has_issues(self) -> bool:
        """Check if there are any validation issues."""
        return bool(self.errors or self.warnings)
    
    def add_error(self, message: str) -> None:
        """Add error message."""
        self.errors.append(message)
        self.passed = False
    
    def add_warning(self, message: str) -> None:
        """Add warning message."""
        self.warnings.append(message)
    
    def add_info(self, message: str) -> None:
        """Add info message."""
        self.info.append(message)


class ConsistencyValidationRule(ValidationRule):
    """Validates consistency between output metadata and actual outputs."""
    
    def __init__(self):
        super().__init__(
            name="output_consistency",
            description="Validates consistency between expected and actual outputs",
            severity="error",
            category="consistency"
        )
    
    def validate(self, context: Dict[str, Any]) -> List[str]:
        issues = []
        output_tracker = context.get("output_tracker")
        
        if not output_tracker:
            return ["No output tracker provided"]
        
        for task_id, metadata in output_tracker.metadata.items():
            output_info = output_tracker.outputs.get(task_id)
            
            if not output_info:
                issues.append(f"Task '{task_id}' has metadata but no actual output")
                continue
            
            # Validate format consistency
            if metadata.format and output_info.format:
                if metadata.format != output_info.format:
                    issues.append(
                        f"Task '{task_id}' format mismatch: expected '{metadata.format}', "
                        f"got '{output_info.format}'"
                    )
            
            # Validate size limits
            if metadata.size_limit and output_info.file_size:
                if output_info.file_size > metadata.size_limit:
                    issues.append(
                        f"Task '{task_id}' output size {output_info.file_size} exceeds "
                        f"limit {metadata.size_limit}"
                    )
            
            # Validate location consistency
            if metadata.location and output_info.location:
                # Check if resolved location matches pattern
                if not self._location_matches_pattern(metadata.location, output_info.location):
                    issues.append(
                        f"Task '{task_id}' location mismatch: template '{metadata.location}', "
                        f"actual '{output_info.location}'"
                    )
        
        return issues
    
    def _location_matches_pattern(self, template: str, actual: str) -> bool:
        """Check if actual location matches template pattern."""
        if template == actual:
            return True
        
        # If template contains variables, we can't do exact matching
        if "{{" in template:
            # Basic check: see if the directory structure matches
            template_dir = os.path.dirname(template)
            actual_dir = os.path.dirname(actual)
            
            # Remove template variables for comparison
            clean_template_dir = re.sub(r'\{\{[^}]+\}\}', '*', template_dir)
            
            # Simple pattern matching
            return actual_dir.startswith(template_dir.split('{{')[0])
        
        return False


class FormatValidationRule(ValidationRule):
    """Validates output format specifications."""
    
    def __init__(self):
        super().__init__(
            name="format_validation",
            description="Validates output format specifications and consistency",
            severity="error",
            category="format"
        )
    
    def validate(self, context: Dict[str, Any]) -> List[str]:
        issues = []
        output_tracker = context.get("output_tracker")
        
        if not output_tracker:
            return ["No output tracker provided"]
        
        for task_id, metadata in output_tracker.metadata.items():
            # Validate format specification
            format_issues = self._validate_format_spec(task_id, metadata)
            issues.extend(format_issues)
            
            # Cross-validate with output info if available
            output_info = output_tracker.outputs.get(task_id)
            if output_info:
                cross_validation_issues = self._cross_validate_format(task_id, metadata, output_info)
                issues.extend(cross_validation_issues)
        
        return issues
    
    def _validate_format_spec(self, task_id: str, metadata: OutputMetadata) -> List[str]:
        """Validate format specification in metadata."""
        issues = []
        
        if not metadata.format and not metadata.produces:
            issues.append(f"Task '{task_id}' has no format or produces specification")
            return issues
        
        # Validate MIME type format
        if metadata.format:
            if not self._is_valid_mime_type(metadata.format):
                issues.append(f"Task '{task_id}' has invalid MIME type: '{metadata.format}'")
        
        # Validate produces field
        if metadata.produces:
            if not self._is_valid_produces_format(metadata.produces):
                issues.append(f"Task '{task_id}' has invalid produces format: '{metadata.produces}'")
        
        # Check consistency between format and location
        if metadata.format and metadata.location:
            expected_ext = self._get_extension_for_mime_type(metadata.format)
            if expected_ext and not metadata.location.endswith(expected_ext):
                issues.append(
                    f"Task '{task_id}' format '{metadata.format}' doesn't match location extension"
                )
        
        return issues
    
    def _cross_validate_format(self, task_id: str, metadata: OutputMetadata, 
                             output_info: OutputInfo) -> List[str]:
        """Cross-validate format between metadata and actual output."""
        issues = []
        
        if metadata.format and output_info.format:
            if metadata.format != output_info.format:
                issues.append(
                    f"Task '{task_id}' format mismatch: metadata='{metadata.format}', "
                    f"actual='{output_info.format}'"
                )
        
        return issues
    
    def _is_valid_mime_type(self, mime_type: str) -> bool:
        """Check if MIME type is valid."""
        # Basic MIME type validation
        return "/" in mime_type and len(mime_type.split("/")) == 2
    
    def _is_valid_produces_format(self, produces: str) -> bool:
        """Check if produces format is valid."""
        # Allow descriptive formats like "pdf-file", "json-data", etc.
        return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', produces))
    
    def _get_extension_for_mime_type(self, mime_type: str) -> Optional[str]:
        """Get expected file extension for MIME type."""
        mime_to_ext = {
            'application/pdf': '.pdf',
            'application/json': '.json',
            'text/markdown': '.md',
            'text/plain': '.txt',
            'text/html': '.html',
            'text/csv': '.csv',
            'image/png': '.png',
            'image/jpeg': '.jpg',
        }
        return mime_to_ext.get(mime_type)


class DependencyValidationRule(ValidationRule):
    """Validates output dependencies and references."""
    
    def __init__(self):
        super().__init__(
            name="dependency_validation",
            description="Validates output dependencies and reference integrity",
            severity="error",
            category="dependency"
        )
    
    def validate(self, context: Dict[str, Any]) -> List[str]:
        issues = []
        output_tracker = context.get("output_tracker")
        
        if not output_tracker:
            return ["No output tracker provided"]
        
        # Check for circular dependencies
        circular_deps = self._find_circular_dependencies(output_tracker)
        for cycle in circular_deps:
            issues.append(f"Circular dependency detected: {' -> '.join(cycle)}")
        
        # Check for missing dependencies
        missing_deps = self._find_missing_dependencies(output_tracker)
        for task_id, missing_task in missing_deps:
            issues.append(f"Task '{task_id}' references missing task '{missing_task}'")
        
        # Check for orphaned references
        orphaned_refs = self._find_orphaned_references(output_tracker)
        for task_id, ref_task in orphaned_refs:
            issues.append(f"Task '{task_id}' references task '{ref_task}' with no output")
        
        return issues
    
    def _find_circular_dependencies(self, output_tracker: OutputTracker) -> List[List[str]]:
        """Find circular dependencies in output references."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(task_id: str) -> bool:
            if task_id in rec_stack:
                # Found a cycle - extract it
                cycle_start = path.index(task_id)
                cycle = path[cycle_start:] + [task_id]
                cycles.append(cycle)
                return True
            
            if task_id in visited:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            path.append(task_id)
            
            # Check all references from this task
            for ref in output_tracker.references.get(task_id, []):
                if dfs(ref.task_id):
                    return True
            
            rec_stack.remove(task_id)
            path.pop()
            return False
        
        # Check all tasks
        all_tasks = set(output_tracker.metadata.keys()) | set(output_tracker.outputs.keys())
        for task_id in all_tasks:
            if task_id not in visited:
                dfs(task_id)
        
        return cycles
    
    def _find_missing_dependencies(self, output_tracker: OutputTracker) -> List[Tuple[str, str]]:
        """Find references to non-existent tasks."""
        missing = []
        all_tasks = set(output_tracker.metadata.keys()) | set(output_tracker.outputs.keys())
        
        for task_id, refs in output_tracker.references.items():
            for ref in refs:
                if ref.task_id not in all_tasks:
                    missing.append((task_id, ref.task_id))
        
        return missing
    
    def _find_orphaned_references(self, output_tracker: OutputTracker) -> List[Tuple[str, str]]:
        """Find references to tasks that have no output."""
        orphaned = []
        
        for task_id, refs in output_tracker.references.items():
            for ref in refs:
                if ref.task_id not in output_tracker.outputs:
                    orphaned.append((task_id, ref.task_id))
        
        return orphaned


class FileSystemValidationRule(ValidationRule):
    """Validates file system aspects of outputs."""
    
    def __init__(self):
        super().__init__(
            name="filesystem_validation",
            description="Validates file system consistency and accessibility",
            severity="warning",
            category="filesystem"
        )
    
    def validate(self, context: Dict[str, Any]) -> List[str]:
        issues = []
        output_tracker = context.get("output_tracker")
        
        if not output_tracker:
            return ["No output tracker provided"]
        
        for task_id, output_info in output_tracker.outputs.items():
            if not output_info.location:
                continue
            
            location = output_info.location
            
            # Check if file exists
            if not os.path.exists(location):
                issues.append(f"Task '{task_id}' output file does not exist: '{location}'")
                continue
            
            # Check if file is readable
            if not os.access(location, os.R_OK):
                issues.append(f"Task '{task_id}' output file is not readable: '{location}'")
            
            # Check file size consistency
            try:
                actual_size = os.path.getsize(location)
                if output_info.file_size and output_info.file_size != actual_size:
                    issues.append(
                        f"Task '{task_id}' file size mismatch: recorded={output_info.file_size}, "
                        f"actual={actual_size}"
                    )
            except OSError as e:
                issues.append(f"Task '{task_id}' cannot get file size: {e}")
        
        return issues


class OutputValidator:
    """
    Comprehensive output validation system.
    
    Orchestrates multiple validation rules to provide complete
    validation coverage for output tracking systems.
    """
    
    def __init__(self):
        """Initialize with default validation rules."""
        self.rules: List[ValidationRule] = [
            ConsistencyValidationRule(),
            FormatValidationRule(),
            DependencyValidationRule(),
            FileSystemValidationRule()
        ]
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add custom validation rule."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove validation rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                return True
        return False
    
    def validate(self, output_tracker: OutputTracker, 
                rules: Optional[List[str]] = None) -> ValidationResult:
        """
        Run comprehensive validation on output tracker.
        
        Args:
            output_tracker: OutputTracker instance to validate
            rules: Optional list of rule names to run (default: all rules)
        
        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult(passed=True)
        context = {"output_tracker": output_tracker}
        
        # Filter rules if specified
        rules_to_run = self.rules
        if rules:
            rules_to_run = [rule for rule in self.rules if rule.name in rules]
        
        # Run each validation rule
        for rule in rules_to_run:
            try:
                issues = rule.validate(context)
                
                for issue in issues:
                    if rule.severity == "error":
                        result.add_error(f"[{rule.name}] {issue}")
                    elif rule.severity == "warning":
                        result.add_warning(f"[{rule.name}] {issue}")
                    else:
                        result.add_info(f"[{rule.name}] {issue}")
                        
            except Exception as e:
                result.add_error(f"Validation rule '{rule.name}' failed: {e}")
        
        return result
    
    def validate_pipeline_spec(self, pipeline_spec: Any) -> ValidationResult:
        """Validate pipeline specification for output metadata consistency."""
        result = ValidationResult(passed=True)
        
        if not hasattr(pipeline_spec, 'steps'):
            result.add_error("Pipeline specification has no steps")
            return result
        
        # Track output types and locations
        output_types = {}
        output_locations = {}
        
        for step in pipeline_spec.steps:
            if not hasattr(step, 'id'):
                result.add_error("Step missing ID")
                continue
            
            step_id = step.id
            
            # Check output metadata if present
            if hasattr(step, 'produces') and step.produces:
                if step.produces in output_types:
                    result.add_warning(
                        f"Multiple steps produce '{step.produces}': {output_types[step.produces]}, {step_id}"
                    )
                else:
                    output_types[step.produces] = step_id
            
            if hasattr(step, 'location') and step.location:
                if step.location in output_locations:
                    result.add_error(
                        f"Multiple steps write to same location '{step.location}': "
                        f"{output_locations[step.location]}, {step_id}"
                    )
                else:
                    output_locations[step.location] = step_id
            
            # Validate individual step output consistency
            if hasattr(step, 'validate_output_consistency'):
                step_issues = step.validate_output_consistency()
                for issue in step_issues:
                    result.add_error(f"Step '{step_id}': {issue}")
        
        return result
    
    def get_validation_summary(self, result: ValidationResult) -> Dict[str, Any]:
        """Generate validation summary from result."""
        return {
            "passed": result.passed,
            "total_issues": len(result.errors) + len(result.warnings) + len(result.info),
            "errors": len(result.errors),
            "warnings": len(result.warnings),
            "info": len(result.info),
            "severity_distribution": {
                "error": len(result.errors),
                "warning": len(result.warnings),
                "info": len(result.info)
            },
            "issues": {
                "errors": result.errors,
                "warnings": result.warnings,
                "info": result.info
            }
        }