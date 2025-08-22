"""Template validation for compile-time template checking.

This module provides comprehensive template validation to detect issues
before pipeline execution, including:
- Template syntax validation
- Variable reference checking against available context
- Undefined variable detection
- Clear error messages and suggestions

Issue #229: Compile-time template validation
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass
from jinja2 import Environment, TemplateSyntaxError, meta
from jinja2.sandbox import SandboxedEnvironment

logger = logging.getLogger(__name__)


@dataclass
class TemplateValidationError:
    """Represents a template validation error."""
    
    template: str
    error_type: str
    message: str
    context_path: Optional[str] = None
    suggestions: List[str] = None
    severity: str = "error"  # error, warning, info
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []
    
    def __str__(self) -> str:
        """String representation of the validation error."""
        path_str = f" (at {self.context_path})" if self.context_path else ""
        result = f"{self.severity.upper()}{path_str}: {self.message}"
        if self.suggestions:
            result += f"\nSuggestions: {', '.join(self.suggestions)}"
        return result


@dataclass
class TemplateValidationResult:
    """Result of template validation."""
    
    is_valid: bool
    errors: List[TemplateValidationError]
    warnings: List[TemplateValidationError]
    available_variables: Set[str]
    used_variables: Set[str]
    undefined_variables: Set[str]
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    def summary(self) -> str:
        """Get a summary of validation results."""
        if self.is_valid and not self.has_warnings:
            return "Template validation passed"
        
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warnings")
            
        return f"Template validation: {', '.join(parts)}"


class TemplateValidator:
    """Validates templates at compile time to prevent runtime errors.
    
    This validator checks:
    1. Template syntax correctness
    2. Variable references against available context
    3. Loop variable usage patterns
    4. Filter and function usage
    5. Control structure syntax
    """
    
    def __init__(self, debug_mode: bool = False):
        """Initialize the template validator.
        
        Args:
            debug_mode: Enable debug logging
        """
        self.debug_mode = debug_mode
        
        # Use sandboxed environment for safety
        self.env = SandboxedEnvironment()
        self._register_custom_filters()
        
        # Patterns for different types of templates
        self.variable_pattern = re.compile(r'{{\s*([^}]+)\s*}}')
        self.control_pattern = re.compile(r'{%\s*([^%]+)\s*%}')
        self.comment_pattern = re.compile(r'{#\s*([^#]+)\s*#}')
        
        # Loop variable patterns
        self.loop_vars = {'$item', '$index', '$is_first', '$is_last', '$iteration', '$loop'}
        self.step_result_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\.(result|results|output|outputs|content|data)')
        
        logger.info("TemplateValidator initialized")
    
    def validate_template(
        self,
        template: str,
        available_context: Optional[Dict[str, Any]] = None,
        context_path: Optional[str] = None,
        step_ids: Optional[List[str]] = None,
        in_loop_context: bool = False
    ) -> TemplateValidationResult:
        """Validate a single template string.
        
        Args:
            template: Template string to validate
            available_context: Context variables available at compile time
            context_path: Path to this template (for error reporting)
            step_ids: List of step IDs in the pipeline
            in_loop_context: Whether this template is inside a loop
            
        Returns:
            TemplateValidationResult with validation details
        """
        if available_context is None:
            available_context = {}
        if step_ids is None:
            step_ids = []
            
        errors = []
        warnings = []
        available_variables = set(available_context.keys())
        used_variables = set()
        undefined_variables = set()
        
        # Skip validation for empty or non-string templates
        if not isinstance(template, str) or not template.strip():
            return TemplateValidationResult(
                is_valid=True,
                errors=errors,
                warnings=warnings,
                available_variables=available_variables,
                used_variables=used_variables,
                undefined_variables=undefined_variables
            )
        
        if self.debug_mode:
            logger.debug(f"Validating template: {template[:100]}...")
            logger.debug(f"Available context: {list(available_context.keys())}")
        
        # 1. Check template syntax
        syntax_errors = self._validate_syntax(template, context_path)
        errors.extend(syntax_errors)
        
        # If syntax errors exist, can't proceed with further validation
        if syntax_errors:
            return TemplateValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                available_variables=available_variables,
                used_variables=used_variables,
                undefined_variables=undefined_variables
            )
        
        # 2. Extract and validate variable references
        var_results = self._validate_variables(
            template, available_context, context_path, step_ids, in_loop_context
        )
        errors.extend(var_results['errors'])
        warnings.extend(var_results['warnings'])
        used_variables.update(var_results['used_variables'])
        undefined_variables.update(var_results['undefined_variables'])
        
        # 3. Validate control structures
        control_results = self._validate_control_structures(template, context_path)
        errors.extend(control_results['errors'])
        warnings.extend(control_results['warnings'])
        
        # 4. Validate filters and functions
        filter_results = self._validate_filters(template, context_path)
        errors.extend(filter_results['errors'])
        warnings.extend(filter_results['warnings'])
        
        is_valid = len(errors) == 0
        
        if self.debug_mode:
            logger.debug(f"Validation result: valid={is_valid}, errors={len(errors)}, warnings={len(warnings)}")
        
        return TemplateValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            available_variables=available_variables,
            used_variables=used_variables,
            undefined_variables=undefined_variables
        )
    
    def validate_pipeline_templates(
        self,
        pipeline_def: Dict[str, Any],
        compile_context: Optional[Dict[str, Any]] = None
    ) -> TemplateValidationResult:
        """Validate all templates in a pipeline definition.
        
        Args:
            pipeline_def: Pipeline definition dictionary
            compile_context: Context available at compile time
            
        Returns:
            Combined validation result for all templates
        """
        if compile_context is None:
            compile_context = {}
            
        # Collect all step IDs
        step_ids = []
        if "steps" in pipeline_def:
            for step in pipeline_def["steps"]:
                if isinstance(step, dict) and "id" in step:
                    step_ids.append(step["id"])
        
        # Add pipeline inputs and parameters to context
        full_context = compile_context.copy()
        
        # Add inputs
        if "inputs" in pipeline_def:
            inputs = pipeline_def["inputs"]
            for input_name, input_spec in inputs.items():
                if isinstance(input_spec, dict) and "default" in input_spec:
                    full_context[input_name] = input_spec["default"]
                elif not isinstance(input_spec, dict):
                    full_context[input_name] = input_spec
        
        # Add parameters
        if "parameters" in pipeline_def:
            params = pipeline_def["parameters"]
            for param_name, param_spec in params.items():
                if isinstance(param_spec, dict) and "default" in param_spec:
                    full_context[param_name] = param_spec["default"]
                elif not isinstance(param_spec, dict):
                    full_context[param_name] = param_spec
        
        # Validate all templates in pipeline
        all_errors = []
        all_warnings = []
        all_used_variables = set()
        all_undefined_variables = set()
        
        self._validate_object_templates(
            pipeline_def, full_context, step_ids, "", 
            all_errors, all_warnings, all_used_variables, all_undefined_variables
        )
        
        is_valid = len(all_errors) == 0
        
        return TemplateValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            available_variables=set(full_context.keys()),
            used_variables=all_used_variables,
            undefined_variables=all_undefined_variables
        )
    
    def _validate_syntax(self, template: str, context_path: Optional[str]) -> List[TemplateValidationError]:
        """Validate Jinja2 template syntax."""
        errors = []
        
        try:
            # Try to parse the template
            self.env.parse(template)
        except TemplateSyntaxError as e:
            errors.append(TemplateValidationError(
                template=template,
                error_type="syntax_error",
                message=f"Template syntax error: {e.message}",
                context_path=context_path,
                suggestions=self._suggest_syntax_fixes(str(e))
            ))
        except Exception as e:
            errors.append(TemplateValidationError(
                template=template,
                error_type="parse_error",
                message=f"Template parsing failed: {str(e)}",
                context_path=context_path
            ))
        
        return errors
    
    def _validate_variables(
        self,
        template: str,
        available_context: Dict[str, Any],
        context_path: Optional[str],
        step_ids: List[str],
        in_loop_context: bool
    ) -> Dict[str, Any]:
        """Validate variable references in template."""
        errors = []
        warnings = []
        used_variables = set()
        undefined_variables = set()
        
        try:
            # Parse template to get AST
            ast = self.env.parse(template)
            
            # Find all variable references
            var_names = meta.find_undeclared_variables(ast)
            
            # Also look for loop variables manually (since they start with $)
            loop_var_matches = []
            for loop_var in self.loop_vars:
                if loop_var in template:
                    loop_var_matches.append(loop_var)
            
            # Combine both sets of variables
            all_var_names = set(var_names) | set(loop_var_matches)
            
            for var_name in all_var_names:
                used_variables.add(var_name)
                
                # Check if it's a loop variable
                if var_name in self.loop_vars:
                    if not in_loop_context:
                        errors.append(TemplateValidationError(
                            template=template,
                            error_type="loop_variable_outside_loop",
                            message=f"Loop variable '{var_name}' used outside of loop context",
                            context_path=context_path,
                            suggestions=["Move this template inside a for_each loop"]
                        ))
                    continue
                
                # Check if it's a step result reference
                if self._is_step_result_reference(var_name, step_ids):
                    # This is valid - step results are runtime variables
                    warnings.append(TemplateValidationError(
                        template=template,
                        error_type="runtime_variable",
                        message=f"Variable '{var_name}' references step results - will be resolved at runtime",
                        context_path=context_path,
                        severity="info"
                    ))
                    continue
                
                # Check if variable is available in context
                if var_name not in available_context:
                    undefined_variables.add(var_name)
                    
                    # Generate suggestions
                    suggestions = self._suggest_variable_alternatives(var_name, available_context, step_ids)
                    
                    errors.append(TemplateValidationError(
                        template=template,
                        error_type="undefined_variable",
                        message=f"Undefined variable: '{var_name}'",
                        context_path=context_path,
                        suggestions=suggestions
                    ))
        
        except Exception as e:
            errors.append(TemplateValidationError(
                template=template,
                error_type="variable_analysis_error",
                message=f"Failed to analyze variables: {str(e)}",
                context_path=context_path
            ))
        
        return {
            'errors': errors,
            'warnings': warnings,
            'used_variables': used_variables,
            'undefined_variables': undefined_variables
        }
    
    def _validate_control_structures(self, template: str, context_path: Optional[str]) -> Dict[str, List]:
        """Validate Jinja2 control structures."""
        errors = []
        warnings = []
        
        # Find all control structures
        controls = self.control_pattern.findall(template)
        
        if self.debug_mode:
            logger.debug(f"Found controls: {controls}")
        
        for control in controls:
            control = control.strip()
            
            # Check for common control structure issues
            if control.startswith('for '):
                # Validate for loop syntax
                if ' in ' not in control:
                    errors.append(TemplateValidationError(
                        template=template,
                        error_type="invalid_for_loop",
                        message=f"Invalid for loop syntax: '{control}'",
                        context_path=context_path,
                        suggestions=["Use format: 'for item in items'"]
                    ))
            
            elif control.startswith('if '):
                # Validate if condition syntax
                if len(control.split()) < 2:
                    errors.append(TemplateValidationError(
                        template=template,
                        error_type="invalid_if_statement",
                        message=f"Invalid if statement syntax: '{control}'",
                        context_path=context_path,
                        suggestions=["Provide a condition after 'if'"]
                    ))
            
            elif control.startswith('set '):
                # Validate set statement syntax
                if '=' not in control:
                    errors.append(TemplateValidationError(
                        template=template,
                        error_type="invalid_set_statement",
                        message=f"Invalid set statement syntax: '{control}'",
                        context_path=context_path,
                        suggestions=["Use format: 'set variable = value'"]
                    ))
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_filters(self, template: str, context_path: Optional[str]) -> Dict[str, List]:
        """Validate filter usage in template."""
        errors = []
        warnings = []
        
        # Find all variable expressions with filters
        var_matches = self.variable_pattern.findall(template)
        
        for var_expr in var_matches:
            if '|' in var_expr:
                # Extract filters
                parts = var_expr.split('|')
                for i, part in enumerate(parts[1:], 1):  # Skip variable name
                    filter_name = part.strip().split('(')[0].strip()
                    
                    # Check if filter exists
                    if filter_name not in self.env.filters:
                        errors.append(TemplateValidationError(
                            template=template,
                            error_type="unknown_filter",
                            message=f"Unknown filter: '{filter_name}'",
                            context_path=context_path,
                            suggestions=self._suggest_filter_alternatives(filter_name)
                        ))
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_object_templates(
        self,
        obj: Any,
        context: Dict[str, Any],
        step_ids: List[str],
        path: str,
        errors: List,
        warnings: List,
        used_variables: Set,
        undefined_variables: Set,
        in_loop_context: bool = False
    ):
        """Recursively validate templates in an object."""
        if isinstance(obj, str):
            # Check if this contains templates
            if '{{' in obj or '{%' in obj:
                result = self.validate_template(
                    obj, context, path, step_ids, in_loop_context
                )
                errors.extend(result.errors)
                warnings.extend(result.warnings)
                used_variables.update(result.used_variables)
                undefined_variables.update(result.undefined_variables)
        
        elif isinstance(obj, dict):
            # Check if we're entering a loop context
            is_loop = 'for_each' in obj or 'while' in obj
            
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                self._validate_object_templates(
                    value, context, step_ids, new_path,
                    errors, warnings, used_variables, undefined_variables,
                    in_loop_context or is_loop
                )
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                self._validate_object_templates(
                    item, context, step_ids, new_path,
                    errors, warnings, used_variables, undefined_variables,
                    in_loop_context
                )
    
    def _is_step_result_reference(self, var_name: str, step_ids: List[str]) -> bool:
        """Check if a variable name references step results."""
        # Check direct step ID references
        if var_name in step_ids:
            return True
        
        # Check step.property references
        parts = var_name.split('.')
        if len(parts) >= 2 and parts[0] in step_ids:
            return True
        
        # Check for common result patterns
        if self.step_result_pattern.match(var_name):
            return True
        
        return False
    
    def _suggest_variable_alternatives(
        self,
        var_name: str,
        available_context: Dict[str, Any],
        step_ids: List[str]
    ) -> List[str]:
        """Suggest alternative variable names for undefined variables."""
        suggestions = []
        
        # Look for similar names in context
        for ctx_var in available_context.keys():
            if self._similar_strings(var_name, ctx_var):
                suggestions.append(f"Did you mean '{ctx_var}'?")
        
        # Look for similar step IDs
        for step_id in step_ids:
            if self._similar_strings(var_name, step_id):
                suggestions.append(f"Did you mean '{step_id}' (step result)?")
        
        # Common patterns
        if var_name.endswith('_text'):
            suggestions.append("Consider using 'text' or 'content'")
        elif var_name.endswith('_data'):
            suggestions.append("Consider using 'data' or 'result'")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _suggest_syntax_fixes(self, error_msg: str) -> List[str]:
        """Suggest fixes for syntax errors."""
        suggestions = []
        
        if 'unexpected' in error_msg.lower():
            suggestions.append("Check for unmatched brackets or quotes")
        
        if 'expected' in error_msg.lower():
            suggestions.append("Check template syntax - missing closing tags?")
        
        if 'filter' in error_msg.lower():
            suggestions.append("Check filter syntax: {{ variable | filter_name }}")
        
        return suggestions
    
    def _suggest_filter_alternatives(self, filter_name: str) -> List[str]:
        """Suggest alternative filter names."""
        suggestions = []
        
        # Common filter alternatives
        filter_alternatives = {
            'lower': ['lower'],
            'upper': ['upper'],
            'title': ['title'],
            'capitalize': ['capitalize'],
            'default': ['default'],
            'length': ['length', 'count'],
            'first': ['first'],
            'last': ['last'],
            'join': ['join'],
            'replace': ['replace'],
            'split': ['split'],
            'format': ['format']
        }
        
        for known_filter in self.env.filters.keys():
            if self._similar_strings(filter_name, known_filter):
                suggestions.append(f"Did you mean '{known_filter}'?")
        
        return suggestions[:3]
    
    def _similar_strings(self, s1: str, s2: str, threshold: float = 0.6) -> bool:
        """Check if two strings are similar using simple edit distance."""
        if len(s1) == 0 or len(s2) == 0:
            return False
        
        s1, s2 = s1.lower(), s2.lower()
        
        # Check for partial matches first
        if s1 in s2 or s2 in s1:
            return True
        
        # Simple character-by-character similarity
        max_len = max(len(s1), len(s2))
        min_len = min(len(s1), len(s2))
        
        if max_len == 0:
            return True
        
        # Count matching characters at same positions
        matches = sum(1 for i, (a, b) in enumerate(zip(s1, s2)) if a == b)
        
        # Also check if the strings have similar length and many shared characters
        shared_chars = len(set(s1) & set(s2))
        total_chars = len(set(s1) | set(s2))
        
        position_similarity = matches / min_len if min_len > 0 else 0
        char_similarity = shared_chars / total_chars if total_chars > 0 else 0
        
        return position_similarity >= threshold or char_similarity >= threshold
    
    def _register_custom_filters(self):
        """Register custom filters that might be used in templates."""
        # Add common filters that might be missing
        
        def safe_default(value, default_value=""):
            """Safe default filter."""
            return value if value is not None else default_value
        
        def safe_length(value):
            """Safe length filter."""
            try:
                return len(value) if value is not None else 0
            except TypeError:
                return 0
        
        def safe_json(value, indent=None):
            """Safe JSON serialization."""
            import json
            try:
                return json.dumps(value, indent=indent, default=str)
            except Exception:
                return str(value)
        
        # Register filters
        self.env.filters['default'] = safe_default
        self.env.filters['length'] = safe_length
        self.env.filters['json'] = safe_json
        self.env.filters['to_json'] = safe_json
        
        # Add other commonly used filters
        self.env.filters['lower'] = lambda x: str(x).lower()
        self.env.filters['upper'] = lambda x: str(x).upper()
        self.env.filters['replace'] = lambda x, old, new: str(x).replace(old, new)
        
        if self.debug_mode:
            logger.debug(f"Registered {len(self.env.filters)} template filters")
    
    def get_available_filters(self) -> List[str]:
        """Get list of available template filters."""
        return list(self.env.filters.keys())
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the validator state."""
        return {
            "debug_mode": self.debug_mode,
            "available_filters": len(self.env.filters),
            "filter_names": list(self.env.filters.keys())
        }