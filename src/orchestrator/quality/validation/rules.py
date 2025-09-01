"""
Configurable validation rules for quality control.

This module provides the foundation for extensible validation rules that can be
configured and combined to create comprehensive quality checks for pipeline outputs.
"""

from __future__ import annotations

import re
import os
import json
import yaml
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union, Callable, Pattern
from enum import Enum
from pathlib import Path

from ...execution.state import ExecutionContext, ExecutionMetrics


class RuleSeverity(Enum):
    """Severity levels for validation rules."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RuleCategory(Enum):
    """Categories for organizing validation rules."""
    CONTENT = "content"
    FORMAT = "format"
    SIZE = "size"
    STRUCTURE = "structure"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPLIANCE = "compliance"


@dataclass
class RuleViolation:
    """Represents a validation rule violation."""
    rule_id: str
    message: str
    severity: RuleSeverity
    category: RuleCategory
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationContext:
    """Context information for validation rules."""
    execution_context: ExecutionContext
    output_path: str
    output_content: Optional[Any] = None
    output_metadata: Dict[str, Any] = field(default_factory=dict)
    pipeline_config: Dict[str, Any] = field(default_factory=dict)


class ValidationRule(ABC):
    """Base class for validation rules."""
    
    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str,
        severity: RuleSeverity = RuleSeverity.ERROR,
        category: RuleCategory = RuleCategory.CONTENT,
        enabled: bool = True
    ):
        """Initialize validation rule."""
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.severity = severity
        self.category = category
        self.enabled = enabled
        self.config: Dict[str, Any] = {}
    
    def configure(self, **config) -> ValidationRule:
        """Configure the rule with parameters."""
        self.config.update(config)
        return self
    
    @abstractmethod
    def validate(self, context: ValidationContext) -> List[RuleViolation]:
        """Validate against the rule and return violations."""
        pass
    
    def is_applicable(self, context: ValidationContext) -> bool:
        """Check if rule is applicable to the given context."""
        return self.enabled


class QualityRule(ValidationRule):
    """Base class for quality-specific rules."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_thresholds: Dict[str, float] = {}
    
    def set_threshold(self, metric: str, threshold: float) -> QualityRule:
        """Set quality threshold for a specific metric."""
        self.quality_thresholds[metric] = threshold
        return self


class FileSizeRule(QualityRule):
    """Rule to validate output file sizes."""
    
    def __init__(self):
        super().__init__(
            rule_id="file_size_limit",
            name="File Size Limit",
            description="Validates output files don't exceed size limits",
            severity=RuleSeverity.WARNING,
            category=RuleCategory.SIZE
        )
        self.max_size_mb = 100  # Default max size in MB
    
    def configure(self, max_size_mb: float = 100, **config):
        """Configure max file size."""
        self.max_size_mb = max_size_mb
        return super().configure(**config)
    
    def validate(self, context: ValidationContext) -> List[RuleViolation]:
        """Validate file size."""
        violations = []
        
        if not os.path.exists(context.output_path):
            return violations
        
        try:
            file_size = os.path.getsize(context.output_path)
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size_mb > self.max_size_mb:
                violations.append(RuleViolation(
                    rule_id=self.rule_id,
                    message=f"File size {file_size_mb:.2f}MB exceeds limit of {self.max_size_mb}MB",
                    severity=self.severity,
                    category=self.category,
                    file_path=context.output_path,
                    metadata={"actual_size_mb": file_size_mb, "limit_mb": self.max_size_mb}
                ))
                
        except OSError as e:
            violations.append(RuleViolation(
                rule_id=self.rule_id,
                message=f"Could not check file size: {e}",
                severity=RuleSeverity.ERROR,
                category=self.category,
                file_path=context.output_path
            ))
        
        return violations


class ContentFormatRule(QualityRule):
    """Rule to validate content format and structure."""
    
    def __init__(self):
        super().__init__(
            rule_id="content_format",
            name="Content Format Validation",
            description="Validates content format matches expected structure",
            severity=RuleSeverity.ERROR,
            category=RuleCategory.FORMAT
        )
        self.expected_format: Optional[str] = None
        self.required_fields: List[str] = []
        self.format_validators: Dict[str, Callable] = {
            'json': self._validate_json,
            'yaml': self._validate_yaml,
            'csv': self._validate_csv,
            'xml': self._validate_xml
        }
    
    def configure(self, expected_format: str = None, required_fields: List[str] = None, **config):
        """Configure format validation."""
        if expected_format:
            self.expected_format = expected_format.lower()
        if required_fields:
            self.required_fields = required_fields
        return super().configure(**config)
    
    def validate(self, context: ValidationContext) -> List[RuleViolation]:
        """Validate content format."""
        violations = []
        
        if not self.expected_format:
            return violations
        
        try:
            if self.expected_format in self.format_validators:
                format_violations = self.format_validators[self.expected_format](context)
                violations.extend(format_violations)
            else:
                # Use MIME type detection
                mime_type, _ = mimetypes.guess_type(context.output_path)
                if mime_type and self.expected_format not in mime_type:
                    violations.append(RuleViolation(
                        rule_id=self.rule_id,
                        message=f"Expected format '{self.expected_format}' but detected '{mime_type}'",
                        severity=self.severity,
                        category=self.category,
                        file_path=context.output_path
                    ))
                    
        except Exception as e:
            violations.append(RuleViolation(
                rule_id=self.rule_id,
                message=f"Format validation failed: {e}",
                severity=RuleSeverity.ERROR,
                category=self.category,
                file_path=context.output_path
            ))
        
        return violations
    
    def _validate_json(self, context: ValidationContext) -> List[RuleViolation]:
        """Validate JSON format."""
        violations = []
        
        try:
            with open(context.output_path, 'r') as f:
                data = json.load(f)
            
            # Check required fields
            for field in self.required_fields:
                if field not in data:
                    violations.append(RuleViolation(
                        rule_id=self.rule_id,
                        message=f"Required field '{field}' missing from JSON",
                        severity=self.severity,
                        category=self.category,
                        file_path=context.output_path
                    ))
                    
        except json.JSONDecodeError as e:
            violations.append(RuleViolation(
                rule_id=self.rule_id,
                message=f"Invalid JSON format: {e}",
                severity=self.severity,
                category=self.category,
                file_path=context.output_path,
                line_number=e.lineno,
                column_number=e.colno
            ))
        except Exception as e:
            violations.append(RuleViolation(
                rule_id=self.rule_id,
                message=f"JSON validation error: {e}",
                severity=RuleSeverity.ERROR,
                category=self.category,
                file_path=context.output_path
            ))
            
        return violations
    
    def _validate_yaml(self, context: ValidationContext) -> List[RuleViolation]:
        """Validate YAML format."""
        violations = []
        
        try:
            with open(context.output_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Check required fields
            if isinstance(data, dict):
                for field in self.required_fields:
                    if field not in data:
                        violations.append(RuleViolation(
                            rule_id=self.rule_id,
                            message=f"Required field '{field}' missing from YAML",
                            severity=self.severity,
                            category=self.category,
                            file_path=context.output_path
                        ))
                        
        except yaml.YAMLError as e:
            violations.append(RuleViolation(
                rule_id=self.rule_id,
                message=f"Invalid YAML format: {e}",
                severity=self.severity,
                category=self.category,
                file_path=context.output_path
            ))
        except Exception as e:
            violations.append(RuleViolation(
                rule_id=self.rule_id,
                message=f"YAML validation error: {e}",
                severity=RuleSeverity.ERROR,
                category=self.category,
                file_path=context.output_path
            ))
            
        return violations
    
    def _validate_csv(self, context: ValidationContext) -> List[RuleViolation]:
        """Validate CSV format."""
        violations = []
        
        try:
            import csv
            with open(context.output_path, 'r') as f:
                csv_reader = csv.reader(f)
                headers = next(csv_reader, None)
                
                if headers:
                    # Check required fields in headers
                    for field in self.required_fields:
                        if field not in headers:
                            violations.append(RuleViolation(
                                rule_id=self.rule_id,
                                message=f"Required column '{field}' missing from CSV",
                                severity=self.severity,
                                category=self.category,
                                file_path=context.output_path
                            ))
                            
        except Exception as e:
            violations.append(RuleViolation(
                rule_id=self.rule_id,
                message=f"CSV validation error: {e}",
                severity=RuleSeverity.ERROR,
                category=self.category,
                file_path=context.output_path
            ))
            
        return violations
    
    def _validate_xml(self, context: ValidationContext) -> List[RuleViolation]:
        """Validate XML format."""
        violations = []
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(context.output_path)
            
            # Basic well-formedness check (ET.parse will raise if malformed)
            # Additional validation could be added here
            
        except ET.ParseError as e:
            violations.append(RuleViolation(
                rule_id=self.rule_id,
                message=f"Invalid XML format: {e}",
                severity=self.severity,
                category=self.category,
                file_path=context.output_path
            ))
        except Exception as e:
            violations.append(RuleViolation(
                rule_id=self.rule_id,
                message=f"XML validation error: {e}",
                severity=RuleSeverity.ERROR,
                category=self.category,
                file_path=context.output_path
            ))
            
        return violations


class ContentQualityRule(QualityRule):
    """Rule to validate content quality metrics."""
    
    def __init__(self):
        super().__init__(
            rule_id="content_quality",
            name="Content Quality Check",
            description="Validates content quality against defined criteria",
            severity=RuleSeverity.WARNING,
            category=RuleCategory.CONTENT
        )
        self.patterns: Dict[str, Pattern] = {}
        self.prohibited_patterns: List[Pattern] = []
        self.required_patterns: List[Pattern] = []
        self.min_length = 0
        self.max_length = 0
    
    def configure(
        self,
        prohibited_patterns: List[str] = None,
        required_patterns: List[str] = None,
        min_length: int = 0,
        max_length: int = 0,
        **config
    ):
        """Configure content quality checks."""
        if prohibited_patterns:
            self.prohibited_patterns = [re.compile(p) for p in prohibited_patterns]
        if required_patterns:
            self.required_patterns = [re.compile(p) for p in required_patterns]
        
        self.min_length = min_length
        self.max_length = max_length
        return super().configure(**config)
    
    def validate(self, context: ValidationContext) -> List[RuleViolation]:
        """Validate content quality."""
        violations = []
        
        try:
            # Read content
            with open(context.output_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Length checks
            if self.min_length > 0 and len(content) < self.min_length:
                violations.append(RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Content length {len(content)} below minimum {self.min_length}",
                    severity=self.severity,
                    category=self.category,
                    file_path=context.output_path,
                    metadata={"actual_length": len(content), "min_length": self.min_length}
                ))
            
            if self.max_length > 0 and len(content) > self.max_length:
                violations.append(RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Content length {len(content)} exceeds maximum {self.max_length}",
                    severity=self.severity,
                    category=self.category,
                    file_path=context.output_path,
                    metadata={"actual_length": len(content), "max_length": self.max_length}
                ))
            
            # Pattern checks
            for pattern in self.prohibited_patterns:
                matches = pattern.findall(content)
                if matches:
                    violations.append(RuleViolation(
                        rule_id=self.rule_id,
                        message=f"Prohibited pattern found: {pattern.pattern}",
                        severity=self.severity,
                        category=self.category,
                        file_path=context.output_path,
                        metadata={"pattern": pattern.pattern, "matches": matches[:5]}  # Limit matches
                    ))
            
            for pattern in self.required_patterns:
                if not pattern.search(content):
                    violations.append(RuleViolation(
                        rule_id=self.rule_id,
                        message=f"Required pattern missing: {pattern.pattern}",
                        severity=self.severity,
                        category=self.category,
                        file_path=context.output_path,
                        metadata={"pattern": pattern.pattern}
                    ))
                    
        except Exception as e:
            violations.append(RuleViolation(
                rule_id=self.rule_id,
                message=f"Content quality check failed: {e}",
                severity=RuleSeverity.ERROR,
                category=self.category,
                file_path=context.output_path
            ))
        
        return violations


class PerformanceRule(QualityRule):
    """Rule to validate performance-related quality metrics."""
    
    def __init__(self):
        super().__init__(
            rule_id="performance_metrics",
            name="Performance Quality Check",
            description="Validates performance metrics meet quality thresholds",
            severity=RuleSeverity.WARNING,
            category=RuleCategory.PERFORMANCE
        )
        self.max_execution_time_seconds = 300  # 5 minutes default
        self.max_memory_mb = 1024  # 1GB default
    
    def configure(
        self,
        max_execution_time_seconds: float = 300,
        max_memory_mb: float = 1024,
        **config
    ):
        """Configure performance thresholds."""
        self.max_execution_time_seconds = max_execution_time_seconds
        self.max_memory_mb = max_memory_mb
        return super().configure(**config)
    
    def validate(self, context: ValidationContext) -> List[RuleViolation]:
        """Validate performance metrics."""
        violations = []
        
        try:
            metrics = context.execution_context.metrics
            
            # Check execution time
            if metrics.duration:
                execution_seconds = metrics.duration.total_seconds()
                if execution_seconds > self.max_execution_time_seconds:
                    violations.append(RuleViolation(
                        rule_id=self.rule_id,
                        message=f"Execution time {execution_seconds:.2f}s exceeds limit {self.max_execution_time_seconds}s",
                        severity=self.severity,
                        category=self.category,
                        metadata={
                            "execution_time_seconds": execution_seconds,
                            "limit_seconds": self.max_execution_time_seconds
                        }
                    ))
            
            # Check memory usage
            if metrics.memory_peak_mb > self.max_memory_mb:
                violations.append(RuleViolation(
                    rule_id=self.rule_id,
                    message=f"Memory usage {metrics.memory_peak_mb:.2f}MB exceeds limit {self.max_memory_mb}MB",
                    severity=self.severity,
                    category=self.category,
                    metadata={
                        "memory_usage_mb": metrics.memory_peak_mb,
                        "limit_mb": self.max_memory_mb
                    }
                ))
                
        except Exception as e:
            violations.append(RuleViolation(
                rule_id=self.rule_id,
                message=f"Performance validation failed: {e}",
                severity=RuleSeverity.ERROR,
                category=self.category
            ))
        
        return violations


class RuleRegistry:
    """Registry for managing validation rules."""
    
    def __init__(self):
        """Initialize rule registry."""
        self.rules: Dict[str, ValidationRule] = {}
        self.rule_categories: Dict[RuleCategory, Set[str]] = {
            category: set() for category in RuleCategory
        }
        
        # Register built-in rules
        self.register_builtin_rules()
    
    def register_builtin_rules(self):
        """Register built-in validation rules."""
        builtin_rules = [
            FileSizeRule(),
            ContentFormatRule(),
            ContentQualityRule(),
            PerformanceRule()
        ]
        
        for rule in builtin_rules:
            self.register_rule(rule)
    
    def register_rule(self, rule: ValidationRule) -> None:
        """Register a validation rule."""
        self.rules[rule.rule_id] = rule
        self.rule_categories[rule.category].add(rule.rule_id)
    
    def unregister_rule(self, rule_id: str) -> bool:
        """Unregister a validation rule."""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            self.rule_categories[rule.category].discard(rule_id)
            del self.rules[rule_id]
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[ValidationRule]:
        """Get a specific rule by ID."""
        return self.rules.get(rule_id)
    
    def get_rules_by_category(self, category: RuleCategory) -> List[ValidationRule]:
        """Get all rules in a specific category."""
        rule_ids = self.rule_categories.get(category, set())
        return [self.rules[rule_id] for rule_id in rule_ids if rule_id in self.rules]
    
    def get_enabled_rules(self) -> List[ValidationRule]:
        """Get all enabled rules."""
        return [rule for rule in self.rules.values() if rule.enabled]
    
    def configure_rule(self, rule_id: str, **config) -> bool:
        """Configure a specific rule."""
        rule = self.rules.get(rule_id)
        if rule:
            rule.configure(**config)
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a specific rule."""
        rule = self.rules.get(rule_id)
        if rule:
            rule.enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a specific rule."""
        rule = self.rules.get(rule_id)
        if rule:
            rule.enabled = False
            return True
        return False
    
    def list_rules(self) -> Dict[str, Dict[str, Any]]:
        """List all registered rules with their metadata."""
        return {
            rule_id: {
                "name": rule.name,
                "description": rule.description,
                "severity": rule.severity.value,
                "category": rule.category.value,
                "enabled": rule.enabled
            }
            for rule_id, rule in self.rules.items()
        }
    
    def load_rules_from_config(self, config_path: Union[str, Path]) -> None:
        """Load rule configurations from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'rules' in config:
                for rule_config in config['rules']:
                    rule_id = rule_config.get('rule_id')
                    if rule_id and rule_id in self.rules:
                        # Configure existing rule
                        rule_params = rule_config.get('config', {})
                        self.configure_rule(rule_id, **rule_params)
                        
                        # Set enabled state
                        if 'enabled' in rule_config:
                            if rule_config['enabled']:
                                self.enable_rule(rule_id)
                            else:
                                self.disable_rule(rule_id)
                                
        except Exception as e:
            raise RuntimeError(f"Failed to load rules configuration: {e}")