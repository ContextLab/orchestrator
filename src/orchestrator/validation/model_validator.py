"""Model requirements validation for compile-time model checking.

This module provides comprehensive model validation to detect issues
before pipeline execution, including:
- Model availability checking against model registry
- Context window requirements validation
- Capability requirements validation (generate, generate_structured, etc.)
- Model-specific parameter validation
- Clear error messages and suggestions

Issue #241 Stream 3: Model Requirements Validation
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelValidationError:
    """Represents a model validation error."""
    
    task_id: str
    model_requirement: str
    error_type: str
    message: str
    context_path: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    severity: str = "error"  # error, warning, info
    
    def __str__(self) -> str:
        """String representation of the validation error."""
        path_str = f" (at {self.context_path})" if self.context_path else ""
        task_str = f"Task '{self.task_id}'" if self.task_id else "Pipeline"
        result = f"{self.severity.upper()} - {task_str}{path_str}: {self.message}"
        if self.suggestions:
            result += f"\nSuggestions: {', '.join(self.suggestions)}"
        return result


@dataclass
class ModelValidationResult:
    """Result of model validation."""
    
    is_valid: bool
    errors: List[ModelValidationError]
    warnings: List[ModelValidationError]
    validated_models: Set[str]
    missing_models: Set[str]
    capability_mismatches: Dict[str, List[str]]
    
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
            return "Model validation passed"
        
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warnings")
            
        return f"Model validation: {', '.join(parts)}"


class ModelValidator:
    """Validates model requirements at compile time to prevent runtime errors.
    
    This validator checks:
    1. Model availability against model registry
    2. Context window requirements vs model capabilities
    3. Required capabilities (generate, generate_structured, function calling)
    4. Model-specific parameters
    5. Provider-specific requirements
    """
    
    def __init__(
        self, 
        model_registry=None, 
        development_mode: bool = False, 
        debug_mode: bool = False
    ):
        """Initialize the model validator.
        
        Args:
            model_registry: Model registry to check availability
            development_mode: Enable development mode (allows some validation bypasses)
            debug_mode: Enable debug logging
        """
        self.model_registry = model_registry
        self.development_mode = development_mode
        self.debug_mode = debug_mode
        
        # Cache for performance
        self._model_cache = {}
        self._capability_cache = {}
        
        if debug_mode:
            logger.setLevel(logging.DEBUG)
    
    def validate_pipeline_models(self, pipeline_def: Dict[str, Any]) -> ModelValidationResult:
        """Validate all model requirements in a pipeline definition.
        
        Args:
            pipeline_def: Pipeline definition dictionary
            
        Returns:
            ModelValidationResult with validation results
        """
        errors = []
        warnings = []
        validated_models = set()
        missing_models = set()
        capability_mismatches = {}
        
        # Check global model specification
        global_model = pipeline_def.get("model")
        if global_model:
            result = self._validate_model_specification(
                global_model, "global", "pipeline.model"
            )
            errors.extend(result["errors"])
            warnings.extend(result["warnings"])
            if result["model_name"]:
                validated_models.add(result["model_name"])
                if result["missing"]:
                    missing_models.add(result["model_name"])
                if result["capability_issues"]:
                    capability_mismatches[result["model_name"]] = result["capability_issues"]
        
        # Check model specifications in each step
        steps = pipeline_def.get("steps", [])
        for step in steps:
            if not isinstance(step, dict):
                continue
                
            step_id = step.get("id", "unknown")
            
            # Check step-level model
            step_model = step.get("model")
            if step_model:
                result = self._validate_model_specification(
                    step_model, step_id, f"steps[{step_id}].model"
                )
                errors.extend(result["errors"])
                warnings.extend(result["warnings"])
                if result["model_name"]:
                    validated_models.add(result["model_name"])
                    if result["missing"]:
                        missing_models.add(result["model_name"])
                    if result["capability_issues"]:
                        capability_mismatches[result["model_name"]] = result["capability_issues"]
            
            # Check model requirements in parameters
            parameters = step.get("parameters", {})
            if isinstance(parameters, dict):
                self._validate_parameters_models(
                    parameters, step_id, f"steps[{step_id}].parameters", 
                    errors, warnings, validated_models, missing_models, capability_mismatches
                )
            
            # Check model requirements based on action type
            action = step.get("action", step.get("tool"))
            if action:
                self._validate_action_model_requirements(
                    action, step_id, step, errors, warnings
                )
        
        # Additional validation if we have a model registry
        if self.model_registry and not self.development_mode:
            self._validate_against_registry(
                validated_models, errors, warnings, capability_mismatches
            )
        
        is_valid = len(errors) == 0
        
        return ModelValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validated_models=validated_models,
            missing_models=missing_models,
            capability_mismatches=capability_mismatches
        )
    
    def validate_task_model(
        self, 
        task_def: Dict[str, Any], 
        global_model: Optional[str] = None
    ) -> ModelValidationResult:
        """Validate model requirements for a single task.
        
        Args:
            task_def: Task definition dictionary
            global_model: Global model specification (if any)
            
        Returns:
            ModelValidationResult with validation results
        """
        errors = []
        warnings = []
        validated_models = set()
        missing_models = set()
        capability_mismatches = {}
        
        task_id = task_def.get("id", "unknown")
        
        # Check task-level model specification
        task_model = task_def.get("model")
        effective_model = task_model or global_model
        
        if effective_model:
            result = self._validate_model_specification(
                effective_model, task_id, f"task[{task_id}].model"
            )
            errors.extend(result["errors"])
            warnings.extend(result["warnings"])
            if result["model_name"]:
                validated_models.add(result["model_name"])
                if result["missing"]:
                    missing_models.add(result["model_name"])
                if result["capability_issues"]:
                    capability_mismatches[result["model_name"]] = result["capability_issues"]
        
        # Check parameters for model references
        parameters = task_def.get("parameters", {})
        if isinstance(parameters, dict):
            self._validate_parameters_models(
                parameters, task_id, f"task[{task_id}].parameters",
                errors, warnings, validated_models, missing_models, capability_mismatches
            )
        
        # Validate against registry if available
        if self.model_registry and not self.development_mode:
            self._validate_against_registry(
                validated_models, errors, warnings, capability_mismatches
            )
        
        is_valid = len(errors) == 0
        
        return ModelValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validated_models=validated_models,
            missing_models=missing_models,
            capability_mismatches=capability_mismatches
        )
    
    def _validate_model_specification(
        self, 
        model_spec: Union[str, Dict[str, Any]], 
        task_id: str, 
        context_path: str
    ) -> Dict[str, Any]:
        """Validate a model specification.
        
        Args:
            model_spec: Model specification (string or dict)
            task_id: ID of the task being validated
            context_path: Path context for error reporting
            
        Returns:
            Dict with validation results
        """
        errors = []
        warnings = []
        model_name = None
        missing = False
        capability_issues = []
        
        if isinstance(model_spec, str):
            model_name = model_spec
            # Basic model name validation
            if not model_name.strip():
                errors.append(ModelValidationError(
                    task_id=task_id,
                    model_requirement=model_name,
                    error_type="empty_model_name",
                    message="Model name cannot be empty",
                    context_path=context_path,
                    suggestions=["Specify a valid model name like 'openai/gpt-4' or 'anthropic/claude-3-sonnet'"]
                ))
            else:
                # Validate model name format
                if self._is_template_string(model_name):
                    # Template strings are validated separately
                    pass
                elif not self._is_valid_model_name(model_name):
                    warnings.append(ModelValidationError(
                        task_id=task_id,
                        model_requirement=model_name,
                        error_type="invalid_model_format",
                        message=f"Model name '{model_name}' may not be in correct format",
                        context_path=context_path,
                        severity="warning",
                        suggestions=["Use format 'provider/model' like 'openai/gpt-4' or 'anthropic/claude-3-sonnet'"]
                    ))
        
        elif isinstance(model_spec, dict):
            # Handle model specification as dictionary
            model_name = model_spec.get("name") or model_spec.get("model")
            if not model_name:
                errors.append(ModelValidationError(
                    task_id=task_id,
                    model_requirement=str(model_spec),
                    error_type="missing_model_name",
                    message="Model specification must include 'name' or 'model' field",
                    context_path=context_path,
                    suggestions=["Add 'name': 'openai/gpt-4' to the model specification"]
                ))
            else:
                # Validate dictionary-based model spec
                self._validate_model_dict_spec(
                    model_spec, task_id, context_path, errors, warnings
                )
        
        else:
            errors.append(ModelValidationError(
                task_id=task_id,
                model_requirement=str(model_spec),
                error_type="invalid_model_type",
                message=f"Model specification must be string or dict, got {type(model_spec).__name__}",
                context_path=context_path,
                suggestions=["Use a string like 'openai/gpt-4' or a dict with 'name' field"]
            ))
        
        return {
            "errors": errors,
            "warnings": warnings,
            "model_name": model_name,
            "missing": missing,
            "capability_issues": capability_issues
        }
    
    def _validate_model_dict_spec(
        self, 
        model_spec: Dict[str, Any], 
        task_id: str, 
        context_path: str, 
        errors: List[ModelValidationError], 
        warnings: List[ModelValidationError]
    ):
        """Validate dictionary-based model specification.
        
        Args:
            model_spec: Model specification dictionary
            task_id: Task ID
            context_path: Context path for errors
            errors: List to append errors to
            warnings: List to append warnings to
        """
        # Check for common parameters
        temperature = model_spec.get("temperature")
        if temperature is not None:
            if not isinstance(temperature, (int, float)):
                errors.append(ModelValidationError(
                    task_id=task_id,
                    model_requirement=str(model_spec),
                    error_type="invalid_temperature_type",
                    message=f"Temperature must be a number, got {type(temperature).__name__}",
                    context_path=f"{context_path}.temperature",
                    suggestions=["Use a number between 0.0 and 2.0"]
                ))
            elif not 0.0 <= temperature <= 2.0:
                warnings.append(ModelValidationError(
                    task_id=task_id,
                    model_requirement=str(model_spec),
                    error_type="temperature_out_of_range",
                    message=f"Temperature {temperature} is outside typical range 0.0-2.0",
                    context_path=f"{context_path}.temperature",
                    severity="warning",
                    suggestions=["Use temperature between 0.0 (deterministic) and 2.0 (very creative)"]
                ))
        
        max_tokens = model_spec.get("max_tokens")
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                errors.append(ModelValidationError(
                    task_id=task_id,
                    model_requirement=str(model_spec),
                    error_type="invalid_max_tokens",
                    message=f"max_tokens must be a positive integer, got {max_tokens}",
                    context_path=f"{context_path}.max_tokens",
                    suggestions=["Use a positive integer like 1000, 2000, 4000"]
                ))
        
        # Check context window requirements
        context_window = model_spec.get("context_window")
        if context_window is not None:
            if not isinstance(context_window, int) or context_window <= 0:
                errors.append(ModelValidationError(
                    task_id=task_id,
                    model_requirement=str(model_spec),
                    error_type="invalid_context_window",
                    message=f"context_window must be a positive integer, got {context_window}",
                    context_path=f"{context_path}.context_window",
                    suggestions=["Use a positive integer like 4096, 8192, 32768"]
                ))
        
        # Check capability requirements
        capabilities = model_spec.get("capabilities", model_spec.get("requires", []))
        if capabilities:
            if not isinstance(capabilities, list):
                errors.append(ModelValidationError(
                    task_id=task_id,
                    model_requirement=str(model_spec),
                    error_type="invalid_capabilities_type",
                    message="capabilities must be a list",
                    context_path=f"{context_path}.capabilities",
                    suggestions=["Use a list like ['generate', 'generate_structured', 'function_calling']"]
                ))
            else:
                self._validate_capability_requirements(
                    capabilities, task_id, f"{context_path}.capabilities", errors, warnings
                )
    
    def _validate_capability_requirements(
        self, 
        capabilities: List[str], 
        task_id: str, 
        context_path: str, 
        errors: List[ModelValidationError], 
        warnings: List[ModelValidationError]
    ):
        """Validate capability requirements.
        
        Args:
            capabilities: List of required capabilities
            task_id: Task ID
            context_path: Context path for errors
            errors: List to append errors to
            warnings: List to append warnings to
        """
        valid_capabilities = {
            "generate", "generate_structured", "function_calling", 
            "vision", "multimodal", "streaming", "json_mode",
            "code_specialized", "supports_tools"
        }
        
        for capability in capabilities:
            if not isinstance(capability, str):
                errors.append(ModelValidationError(
                    task_id=task_id,
                    model_requirement=str(capabilities),
                    error_type="invalid_capability_type",
                    message=f"Capability must be string, got {type(capability).__name__}",
                    context_path=context_path,
                    suggestions=["Use string capability names"]
                ))
            elif capability not in valid_capabilities:
                warnings.append(ModelValidationError(
                    task_id=task_id,
                    model_requirement=capability,
                    error_type="unknown_capability",
                    message=f"Unknown capability '{capability}'",
                    context_path=context_path,
                    severity="warning",
                    suggestions=[f"Valid capabilities: {', '.join(sorted(valid_capabilities))}"]
                ))
    
    def _validate_parameters_models(
        self, 
        parameters: Dict[str, Any], 
        task_id: str, 
        context_path: str,
        errors: List[ModelValidationError], 
        warnings: List[ModelValidationError],
        validated_models: Set[str],
        missing_models: Set[str],
        capability_mismatches: Dict[str, List[str]]
    ):
        """Recursively validate model references in parameters.
        
        Args:
            parameters: Parameters dictionary
            task_id: Task ID
            context_path: Context path
            errors: List to append errors to
            warnings: List to append warnings to
            validated_models: Set to add validated models to
            missing_models: Set to add missing models to
            capability_mismatches: Dict to add capability mismatches to
        """
        for key, value in parameters.items():
            param_path = f"{context_path}.{key}"
            
            if key in ["model", "llm", "language_model"] and isinstance(value, str):
                # Direct model reference
                if value and not self._is_template_string(value):
                    validated_models.add(value)
                    
            elif isinstance(value, dict):
                # Check if this looks like a model specification
                if "model" in value or "name" in value:
                    result = self._validate_model_specification(
                        value, task_id, param_path
                    )
                    errors.extend(result["errors"])
                    warnings.extend(result["warnings"])
                    if result["model_name"]:
                        validated_models.add(result["model_name"])
                
                # Recurse into nested dictionaries
                self._validate_parameters_models(
                    value, task_id, param_path, errors, warnings,
                    validated_models, missing_models, capability_mismatches
                )
                
            elif isinstance(value, list):
                # Check list items for model references
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._validate_parameters_models(
                            item, task_id, f"{param_path}[{i}]", errors, warnings,
                            validated_models, missing_models, capability_mismatches
                        )
    
    def _validate_action_model_requirements(
        self, 
        action: str, 
        task_id: str, 
        task_def: Dict[str, Any],
        errors: List[ModelValidationError], 
        warnings: List[ModelValidationError]
    ):
        """Validate model requirements based on action type.
        
        Args:
            action: Action type
            task_id: Task ID
            task_def: Full task definition
            errors: List to append errors to
            warnings: List to append warnings to
        """
        # Actions that typically require LLM models
        llm_actions = {
            "llm", "generate", "chat", "complete", "summarize", 
            "analyze", "extract", "classify", "translate"
        }
        
        # Actions that require structured output capability
        structured_actions = {
            "extract_structured", "classify_structured", "analyze_structured"
        }
        
        # Actions that benefit from function calling
        function_calling_actions = {
            "tool_use", "function_call", "agent"
        }
        
        # Actions that require vision capabilities
        vision_actions = {
            "image_analyze", "image_describe", "vision", "ocr"
        }
        
        if action in llm_actions:
            # Check if a model is specified
            model = task_def.get("model")
            parameters = task_def.get("parameters", {})
            param_model = parameters.get("model") if isinstance(parameters, dict) else None
            
            if not model and not param_model:
                warnings.append(ModelValidationError(
                    task_id=task_id,
                    model_requirement="",
                    error_type="missing_model_for_llm_action",
                    message=f"Action '{action}' typically requires a model specification",
                    context_path=f"task[{task_id}]",
                    severity="warning",
                    suggestions=[
                        "Add 'model': 'openai/gpt-4' to the task",
                        "Add 'model': 'anthropic/claude-3-sonnet' to the task",
                        "Specify model in parameters"
                    ]
                ))
        
        if action in structured_actions:
            warnings.append(ModelValidationError(
                task_id=task_id,
                model_requirement="",
                error_type="structured_output_requirement",
                message=f"Action '{action}' requires model with structured output capability",
                context_path=f"task[{task_id}]",
                severity="warning",
                suggestions=[
                    "Ensure chosen model supports structured output",
                    "Consider models like 'openai/gpt-4' or 'anthropic/claude-3-sonnet'"
                ]
            ))
        
        if action in function_calling_actions:
            warnings.append(ModelValidationError(
                task_id=task_id,
                model_requirement="",
                error_type="function_calling_requirement",
                message=f"Action '{action}' requires model with function calling capability",
                context_path=f"task[{task_id}]",
                severity="warning",
                suggestions=[
                    "Ensure chosen model supports function calling",
                    "Consider models like 'openai/gpt-4' or 'anthropic/claude-3-sonnet'"
                ]
            ))
        
        if action in vision_actions:
            warnings.append(ModelValidationError(
                task_id=task_id,
                model_requirement="",
                error_type="vision_capability_requirement",
                message=f"Action '{action}' requires model with vision capabilities",
                context_path=f"task[{task_id}]",
                severity="warning",
                suggestions=[
                    "Ensure chosen model supports vision/multimodal input",
                    "Consider models like 'openai/gpt-4-vision' or 'anthropic/claude-3-sonnet'"
                ]
            ))
    
    def _validate_against_registry(
        self, 
        model_names: Set[str], 
        errors: List[ModelValidationError], 
        warnings: List[ModelValidationError],
        capability_mismatches: Dict[str, List[str]]
    ):
        """Validate models against the model registry.
        
        Args:
            model_names: Set of model names to validate
            errors: List to append errors to
            warnings: List to append warnings to
            capability_mismatches: Dict to add capability mismatches to
        """
        if not self.model_registry:
            return
            
        for model_name in model_names:
            if self._is_template_string(model_name):
                # Skip template strings
                continue
                
            try:
                # Try to get the model from registry
                model = self.model_registry.get_model(model_name)
                
                # Model found - validate its capabilities
                if self.debug_mode:
                    logger.debug(f"Found model '{model_name}' in registry")
                
                # Check for common capability issues
                capabilities = model.capabilities
                issues = []
                
                # Check context window (warn if very small)
                if capabilities.context_window < 4096:
                    issues.append(f"Small context window ({capabilities.context_window} tokens)")
                
                # Check if model supports common requirements
                if not capabilities.supports_structured_output and not capabilities.supports_json_mode:
                    issues.append("No structured output support")
                
                if issues:
                    capability_mismatches[model_name] = issues
                    if self.debug_mode:
                        warnings.append(ModelValidationError(
                            task_id="",
                            model_requirement=model_name,
                            error_type="capability_limitations",
                            message=f"Model '{model_name}' has limitations: {', '.join(issues)}",
                            severity="warning",
                            suggestions=[
                                "Consider using a model with better capabilities",
                                "Check if these limitations affect your use case"
                            ]
                        ))
                
            except Exception as e:
                # Model not found or other error
                error_msg = str(e)
                if "not found" in error_msg.lower():
                    errors.append(ModelValidationError(
                        task_id="",
                        model_requirement=model_name,
                        error_type="model_not_found",
                        message=f"Model '{model_name}' not found in registry",
                        suggestions=[
                            "Check model name spelling",
                            "Ensure model is properly registered",
                            "Check available models with list_models()",
                            f"Available providers: {', '.join(self.model_registry.list_providers()) if hasattr(self.model_registry, 'list_providers') else 'unknown'}"
                        ]
                    ))
                else:
                    warnings.append(ModelValidationError(
                        task_id="",
                        model_requirement=model_name,
                        error_type="model_registry_error",
                        message=f"Error checking model '{model_name}': {error_msg}",
                        severity="warning",
                        suggestions=["Check model registry configuration"]
                    ))
    
    def _is_template_string(self, value: str) -> bool:
        """Check if a string contains template expressions.
        
        Args:
            value: String to check
            
        Returns:
            True if string contains templates
        """
        return "{{" in value or "{%" in value or "<AUTO>" in value
    
    def _is_valid_model_name(self, model_name: str) -> bool:
        """Check if model name follows valid format.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            True if format is valid
        """
        # Accept various formats:
        # - provider/model (e.g., openai/gpt-4)
        # - provider:model (e.g., openai:gpt-4) 
        # - just model name (e.g., gpt-4)
        # - model names with versions (e.g., gpt-4-turbo-preview)
        
        if not model_name or not isinstance(model_name, str):
            return False
            
        # Basic validation - model name should be reasonable
        if len(model_name.strip()) == 0:
            return False
            
        # Check for obvious problems
        if model_name.count("/") > 1 or model_name.count(":") > 1:
            return False
            
        # If it has provider prefix, validate format
        if "/" in model_name or ":" in model_name:
            separator = "/" if "/" in model_name else ":"
            parts = model_name.split(separator)
            if len(parts) != 2:
                return False
            provider, model = parts
            if not provider.strip() or not model.strip():
                return False
                
        return True
    
    def get_registry_models(self) -> List[str]:
        """Get list of available models from registry.
        
        Returns:
            List of available model names
        """
        if not self.model_registry:
            return []
            
        try:
            if hasattr(self.model_registry, 'list_models'):
                return self.model_registry.list_models()
            else:
                return list(self.model_registry.models.keys()) if hasattr(self.model_registry, 'models') else []
        except Exception as e:
            logger.warning(f"Error getting registry models: {e}")
            return []
    
    def suggest_alternative_models(self, failed_model: str) -> List[str]:
        """Suggest alternative models when validation fails.
        
        Args:
            failed_model: Model that failed validation
            
        Returns:
            List of suggested alternative models
        """
        suggestions = []
        
        if not self.model_registry:
            # Generic suggestions
            suggestions = [
                "openai/gpt-4",
                "openai/gpt-3.5-turbo", 
                "anthropic/claude-3-sonnet",
                "anthropic/claude-3-haiku"
            ]
        else:
            # Get suggestions from registry
            available_models = self.get_registry_models()
            
            if failed_model:
                # Try to suggest similar models
                failed_lower = failed_model.lower()
                
                # Look for provider match
                if "/" in failed_model:
                    provider = failed_model.split("/")[0].lower()
                    provider_models = [m for m in available_models if m.lower().startswith(f"{provider}/")]
                    suggestions.extend(provider_models[:3])
                
                # Look for name similarity
                model_part = failed_model.split("/")[-1].lower() if "/" in failed_model else failed_lower
                similar_models = [
                    m for m in available_models 
                    if model_part in m.lower() or m.lower().split("/")[-1] in model_part
                ]
                suggestions.extend(similar_models[:3])
            
            # Add some popular models if we don't have enough suggestions
            if len(suggestions) < 3:
                popular_models = [m for m in available_models if any(
                    pop in m.lower() for pop in ["gpt-4", "gpt-3.5", "claude", "llama"]
                )]
                suggestions.extend(popular_models[:5])
        
        # Remove duplicates and limit
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
            if len(unique_suggestions) >= 5:
                break
                
        return unique_suggestions