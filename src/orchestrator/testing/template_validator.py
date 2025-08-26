"""Advanced template validation using enhanced template resolution system.

Integrates with UnifiedTemplateResolver from Issue #275 to provide comprehensive
template validation, including $variable syntax validation, cross-step reference
checking, and template artifact detection.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Import template resolution system from Issue #275
try:
    from orchestrator.core.unified_template_resolver import (
        UnifiedTemplateResolver, TemplateResolutionContext
    )
    HAS_UNIFIED_RESOLVER = True
except ImportError:
    HAS_UNIFIED_RESOLVER = False

from orchestrator.core.template_manager import TemplateManager

logger = logging.getLogger(__name__)


@dataclass
class TemplateValidationIssue:
    """A single template validation issue."""
    
    severity: str  # 'critical', 'major', 'minor'
    category: str  # 'syntax', 'resolution', 'artifact', 'reference'
    description: str
    template_text: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class TemplateValidationResult:
    """Result of comprehensive template validation."""
    
    # Overall assessment
    all_templates_resolved: bool = False
    syntax_valid: bool = False
    references_valid: bool = False
    
    # Issue categorization
    critical_issues: List[TemplateValidationIssue] = field(default_factory=list)
    major_issues: List[TemplateValidationIssue] = field(default_factory=list)
    minor_issues: List[TemplateValidationIssue] = field(default_factory=list)
    
    # Template analysis
    templates_found: List[str] = field(default_factory=list)
    unresolved_templates: List[str] = field(default_factory=list)
    dollar_variables: List[str] = field(default_factory=list)
    cross_step_references: List[str] = field(default_factory=list)
    
    # Files processed
    files_analyzed: List[str] = field(default_factory=list)
    output_artifacts_found: List[str] = field(default_factory=list)
    
    # Validation metadata
    validation_method: str = "enhanced"  # 'enhanced', 'basic', 'rule-based'
    resolver_available: bool = False
    
    @property
    def total_issues(self) -> int:
        """Total number of template issues found."""
        return len(self.critical_issues) + len(self.major_issues) + len(self.minor_issues)
    
    @property
    def has_critical_issues(self) -> bool:
        """Whether critical template issues were found."""
        return len(self.critical_issues) > 0
    
    @property
    def template_score(self) -> float:
        """Template quality score (0-100)."""
        if self.has_critical_issues:
            return 0.0
        
        base_score = 100.0
        base_score -= len(self.major_issues) * 15.0
        base_score -= len(self.minor_issues) * 5.0
        
        return max(0.0, base_score)


class TemplateValidator:
    """
    Advanced template validation using enhanced template resolution system.
    
    Provides comprehensive template validation including:
    - Integration with UnifiedTemplateResolver for advanced validation
    - $variable syntax validation and preprocessing
    - Cross-step reference validation
    - Template artifact detection in outputs
    - Comprehensive debugging and issue reporting
    """
    
    def __init__(self, 
                 template_resolver: Optional[UnifiedTemplateResolver] = None,
                 template_manager: Optional[TemplateManager] = None):
        """
        Initialize template validator.
        
        Args:
            template_resolver: Enhanced template resolver from Issue #275
            template_manager: Template manager for context
        """
        self.template_resolver = template_resolver
        self.template_manager = template_manager
        
        # Initialize resolver if available and not provided
        if not self.template_resolver and HAS_UNIFIED_RESOLVER:
            try:
                self.template_resolver = UnifiedTemplateResolver(
                    template_manager=self.template_manager
                )
                logger.info("Initialized UnifiedTemplateResolver for validation")
            except Exception as e:
                logger.warning(f"Failed to initialize UnifiedTemplateResolver: {e}")
        
        # Validation configuration
        self.enable_enhanced_validation = HAS_UNIFIED_RESOLVER and self.template_resolver is not None
        self.check_output_artifacts = True
        self.validate_cross_references = True
        self.validate_dollar_syntax = True
        
        logger.info(f"Template validator initialized (Enhanced: {self.enable_enhanced_validation})")
    
    def validate_pipeline_templates(self, 
                                  pipeline_path: Path,
                                  output_directory: Optional[Path] = None,
                                  context: Optional[Dict[str, Any]] = None) -> TemplateValidationResult:
        """
        Comprehensive template validation for a pipeline.
        
        Args:
            pipeline_path: Path to pipeline YAML file
            output_directory: Directory containing pipeline outputs
            context: Template context for validation
            
        Returns:
            TemplateValidationResult: Comprehensive validation results
        """
        logger.info(f"Starting template validation for: {pipeline_path}")
        
        result = TemplateValidationResult(
            resolver_available=self.enable_enhanced_validation,
            files_analyzed=[str(pipeline_path)]
        )
        
        try:
            # Validate pipeline YAML templates
            if pipeline_path.exists():
                pipeline_issues = self._validate_pipeline_yaml(pipeline_path, context)
                self._categorize_issues(result, pipeline_issues)
            else:
                result.critical_issues.append(TemplateValidationIssue(
                    severity='critical',
                    category='file',
                    description=f"Pipeline file not found: {pipeline_path}",
                    template_text="",
                    suggestion="Verify pipeline file path is correct"
                ))
            
            # Validate output artifacts if directory exists
            if output_directory and output_directory.exists() and self.check_output_artifacts:
                output_issues = self._validate_output_artifacts(output_directory)
                self._categorize_issues(result, output_issues)
                result.files_analyzed.extend([str(f) for f in output_directory.rglob("*") if f.is_file()])
            
            # Finalize validation assessment
            self._finalize_validation_result(result)
            
        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            result.critical_issues.append(TemplateValidationIssue(
                severity='critical',
                category='validation',
                description=f"Validation process failed: {e}",
                template_text="",
                suggestion="Check template validator configuration and inputs"
            ))
        
        logger.info(f"Template validation completed: {result.total_issues} issues found, "
                   f"Score: {result.template_score:.1f}")
        
        return result
    
    def _validate_pipeline_yaml(self, 
                              pipeline_path: Path,
                              context: Optional[Dict[str, Any]] = None) -> List[TemplateValidationIssue]:
        """Validate templates in pipeline YAML file."""
        issues = []
        
        try:
            content = pipeline_path.read_text(encoding='utf-8')
            
            # Find all template patterns
            templates = self._extract_templates(content)
            
            for template in templates:
                template_issues = self._validate_single_template(
                    template, str(pipeline_path), context
                )
                issues.extend(template_issues)
            
        except Exception as e:
            issues.append(TemplateValidationIssue(
                severity='major',
                category='file',
                description=f"Failed to read pipeline file: {e}",
                template_text="",
                file_path=str(pipeline_path),
                suggestion="Check file encoding and permissions"
            ))
        
        return issues
    
    def _validate_output_artifacts(self, output_directory: Path) -> List[TemplateValidationIssue]:
        """Validate template artifacts in output files."""
        issues = []
        
        try:
            # Scan all text files for template artifacts
            text_extensions = {'.md', '.txt', '.csv', '.json', '.html', '.yaml', '.yml'}
            
            for file_path in output_directory.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                    file_issues = self._validate_output_file(file_path)
                    issues.extend(file_issues)
        
        except Exception as e:
            issues.append(TemplateValidationIssue(
                severity='major',
                category='output',
                description=f"Failed to scan output directory: {e}",
                template_text="",
                suggestion="Check output directory permissions and structure"
            ))
        
        return issues
    
    def _validate_output_file(self, file_path: Path) -> List[TemplateValidationIssue]:
        """Validate template artifacts in a single output file."""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Look for unresolved template artifacts
            artifacts = self._find_template_artifacts(content)
            
            for artifact in artifacts:
                issues.append(TemplateValidationIssue(
                    severity='critical',
                    category='artifact',
                    description=f"Unresolved template artifact in output",
                    template_text=artifact,
                    file_path=str(file_path),
                    suggestion="Ensure all template variables are resolved before output generation"
                ))
        
        except Exception as e:
            logger.warning(f"Failed to validate output file {file_path}: {e}")
        
        return issues
    
    def _validate_single_template(self, 
                                template: str,
                                file_path: str,
                                context: Optional[Dict[str, Any]] = None) -> List[TemplateValidationIssue]:
        """Validate a single template string."""
        issues = []
        
        # Basic syntax validation
        syntax_issues = self._validate_template_syntax(template, file_path)
        issues.extend(syntax_issues)
        
        # Enhanced validation using UnifiedTemplateResolver
        if self.enable_enhanced_validation:
            enhanced_issues = self._validate_with_resolver(template, file_path, context)
            issues.extend(enhanced_issues)
        
        # Specific pattern validation
        pattern_issues = self._validate_template_patterns(template, file_path)
        issues.extend(pattern_issues)
        
        return issues
    
    def _validate_template_syntax(self, template: str, file_path: str) -> List[TemplateValidationIssue]:
        """Validate basic template syntax."""
        issues = []
        
        # Check for balanced braces
        if template.count('{{') != template.count('}}'):
            issues.append(TemplateValidationIssue(
                severity='critical',
                category='syntax',
                description="Unbalanced template braces",
                template_text=template,
                file_path=file_path,
                suggestion="Ensure all {{ have matching }}"
            ))
        
        # Check for empty templates
        template_content = template.replace('{{', '').replace('}}', '').strip()
        if not template_content:
            issues.append(TemplateValidationIssue(
                severity='major',
                category='syntax',
                description="Empty template found",
                template_text=template,
                file_path=file_path,
                suggestion="Remove empty templates or add variable name"
            ))
        
        return issues
    
    def _validate_with_resolver(self, 
                              template: str,
                              file_path: str,
                              context: Optional[Dict[str, Any]] = None) -> List[TemplateValidationIssue]:
        """Validate template using UnifiedTemplateResolver."""
        issues = []
        
        if not self.template_resolver:
            return issues
        
        try:
            # Create template context
            resolution_context = TemplateResolutionContext(
                pipeline_inputs=context or {},
                step_results=context or {},
                additional_context=context or {}
            )
            
            # Check if template uses $variable syntax
            if '$' in template and self.validate_dollar_syntax:
                # Test $variable preprocessing
                preprocessed = self.template_resolver._preprocess_dollar_variables(template)
                if preprocessed != template:
                    logger.debug(f"$variable preprocessing applied: {template} -> {preprocessed}")
            
            # Test template resolution (if resolver has debug capabilities)
            if hasattr(self.template_resolver, 'get_unresolved_variables'):
                unresolved = self.template_resolver.get_unresolved_variables(
                    template, resolution_context.to_flat_dict()
                )
                
                if unresolved:
                    issues.append(TemplateValidationIssue(
                        severity='major',
                        category='resolution',
                        description=f"Template variables may not resolve: {', '.join(unresolved)}",
                        template_text=template,
                        file_path=file_path,
                        suggestion="Ensure all variables are available in execution context"
                    ))
        
        except Exception as e:
            issues.append(TemplateValidationIssue(
                severity='minor',
                category='resolution',
                description=f"Template resolution validation failed: {e}",
                template_text=template,
                file_path=file_path,
                suggestion="Template may still work during execution"
            ))
        
        return issues
    
    def _validate_template_patterns(self, template: str, file_path: str) -> List[TemplateValidationIssue]:
        """Validate specific template patterns."""
        issues = []
        
        # Check for $variable patterns
        if '$' in template:
            dollar_vars = re.findall(r'\$\w+', template)
            for var in dollar_vars:
                if not self._is_valid_dollar_variable(var):
                    issues.append(TemplateValidationIssue(
                        severity='minor',
                        category='syntax',
                        description=f"Unusual $variable pattern: {var}",
                        template_text=template,
                        file_path=file_path,
                        suggestion="Verify $variable follows expected naming conventions"
                    ))
        
        # Check for cross-step references
        if '.' in template and self.validate_cross_references:
            dot_refs = re.findall(r'\w+\.\w+', template)
            for ref in dot_refs:
                if not self._is_valid_cross_reference(ref):
                    issues.append(TemplateValidationIssue(
                        severity='minor',
                        category='reference',
                        description=f"Cross-step reference may not resolve: {ref}",
                        template_text=template,
                        file_path=file_path,
                        suggestion="Ensure referenced step produces the expected output"
                    ))
        
        # Check for complex template expressions
        if '|' in template or 'if' in template or 'for' in template:
            # These are Jinja2 control structures - validate basic syntax
            if not self._validate_jinja2_syntax(template):
                issues.append(TemplateValidationIssue(
                    severity='major',
                    category='syntax',
                    description="Complex template expression may have syntax issues",
                    template_text=template,
                    file_path=file_path,
                    suggestion="Verify Jinja2 syntax for filters, conditionals, and loops"
                ))
        
        return issues
    
    def _extract_templates(self, content: str) -> List[str]:
        """Extract all template patterns from content."""
        # Find all {{ }} patterns
        template_pattern = r'\{\{([^}]+)\}\}'
        matches = re.findall(template_pattern, content)
        
        # Return full template strings
        return [f"{{{{{match.strip()}}}}}" for match in matches]
    
    def _find_template_artifacts(self, content: str) -> List[str]:
        """Find unresolved template artifacts in content."""
        artifacts = []
        
        # Look for {{ }} patterns that shouldn't be in final output
        template_pattern = r'\{\{[^}]+\}\}'
        matches = re.findall(template_pattern, content)
        
        for match in matches:
            # Filter out intentional templates (like in documentation)
            if not self._is_intentional_template(match):
                artifacts.append(match)
        
        return artifacts
    
    def _is_valid_dollar_variable(self, variable: str) -> bool:
        """Check if $variable follows valid naming conventions."""
        # Common valid $variables in loop contexts
        valid_patterns = [
            '$item', '$index', '$is_first', '$is_last',
            '$count', '$total', '$key', '$value'
        ]
        
        return (variable in valid_patterns or
                variable.startswith('$item.') or
                re.match(r'^\$[a-zA-Z_][a-zA-Z0-9_]*$', variable))
    
    def _is_valid_cross_reference(self, reference: str) -> bool:
        """Check if cross-step reference follows valid patterns."""
        # Common valid patterns for cross-step references
        valid_patterns = [
            r'^\w+\.(content|result|output|data)$',
            r'^\w+\.(text|response|analysis)$',
            r'^\w+\.[a-zA-Z_][a-zA-Z0-9_]*$'
        ]
        
        return any(re.match(pattern, reference) for pattern in valid_patterns)
    
    def _validate_jinja2_syntax(self, template: str) -> bool:
        """Basic Jinja2 syntax validation."""
        try:
            from jinja2 import Template
            # Try to create a Jinja2 template - will raise exception if invalid
            Template(template)
            return True
        except Exception:
            return False
    
    def _is_intentional_template(self, template_str: str) -> bool:
        """Check if template is intentionally left in output (e.g., documentation)."""
        # Templates that might legitimately appear in documentation
        intentional_patterns = [
            r'\{\{\s*example\s*\}\}',
            r'\{\{\s*placeholder\s*\}\}',
            r'\{\{\s*template\s*\}\}',
            r'\{\{\s*variable\s*\}\}'
        ]
        
        return any(re.match(pattern, template_str, re.IGNORECASE) for pattern in intentional_patterns)
    
    def _categorize_issues(self, result: TemplateValidationResult, issues: List[TemplateValidationIssue]):
        """Categorize issues by severity."""
        for issue in issues:
            if issue.severity == 'critical':
                result.critical_issues.append(issue)
            elif issue.severity == 'major':
                result.major_issues.append(issue)
            else:
                result.minor_issues.append(issue)
            
            # Track template patterns
            if issue.template_text:
                result.templates_found.append(issue.template_text)
                
                if issue.category == 'artifact':
                    result.unresolved_templates.append(issue.template_text)
                
                if '$' in issue.template_text:
                    dollar_vars = re.findall(r'\$\w+', issue.template_text)
                    result.dollar_variables.extend(dollar_vars)
                
                if '.' in issue.template_text:
                    cross_refs = re.findall(r'\w+\.\w+', issue.template_text)
                    result.cross_step_references.extend(cross_refs)
    
    def _finalize_validation_result(self, result: TemplateValidationResult):
        """Finalize validation result with overall assessments."""
        result.all_templates_resolved = len(result.unresolved_templates) == 0
        result.syntax_valid = len([i for i in result.critical_issues + result.major_issues 
                                 if i.category == 'syntax']) == 0
        result.references_valid = len([i for i in result.major_issues 
                                     if i.category == 'reference']) == 0
        
        # Set validation method
        if self.enable_enhanced_validation:
            result.validation_method = "enhanced"
        else:
            result.validation_method = "rule-based"
        
        # Remove duplicates from tracked patterns
        result.templates_found = list(set(result.templates_found))
        result.unresolved_templates = list(set(result.unresolved_templates))
        result.dollar_variables = list(set(result.dollar_variables))
        result.cross_step_references = list(set(result.cross_step_references))
    
    def supports_enhanced_validation(self) -> bool:
        """Check if enhanced validation with UnifiedTemplateResolver is available."""
        return self.enable_enhanced_validation
    
    def get_validation_capabilities(self) -> Dict[str, bool]:
        """Get available validation capabilities."""
        return {
            "enhanced_resolution": self.enable_enhanced_validation,
            "output_artifact_checking": self.check_output_artifacts,
            "cross_reference_validation": self.validate_cross_references,
            "dollar_syntax_validation": self.validate_dollar_syntax,
            "jinja2_syntax_validation": True
        }