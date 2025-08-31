"""
Advanced Pipeline Operations for the Orchestrator Framework.

This module provides specialized pipeline compilation methods with enhanced YAML
specification integration, advanced validation, and comprehensive error handling.
It builds upon the core API interface to provide detailed pipeline management.
"""

from __future__ import annotations

import logging
import tempfile
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

# Import foundation components
from ..compiler import YAMLCompiler, YAMLCompilerError
from ..core.pipeline import Pipeline
from ..core.task import Task
from ..models import ModelRegistry, get_model_registry

logger = logging.getLogger(__name__)


class PipelineCompilerError(Exception):
    """Exception raised during advanced pipeline compilation."""
    pass


class PipelineValidationError(PipelineCompilerError):
    """Exception raised during pipeline validation."""
    pass


class AdvancedPipelineCompiler:
    """
    Advanced pipeline compilation with comprehensive YAML integration.
    
    Provides specialized compilation methods that enhance the basic YAML compiler
    with advanced features like template preprocessing, dependency resolution,
    multi-stage compilation, and comprehensive validation reporting.
    
    Example:
        >>> compiler = AdvancedPipelineCompiler()
        >>> pipeline, report = await compiler.compile_with_validation(yaml_content)
        >>> dependencies = compiler.analyze_dependencies(yaml_content)
    """
    
    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        development_mode: bool = False,
        enable_preprocessing: bool = True,
        validate_dependencies: bool = True,
        cache_compilations: bool = True
    ):
        """
        Initialize the advanced pipeline compiler.
        
        Args:
            model_registry: Optional model registry for AUTO tag resolution
            development_mode: Enable development mode features
            enable_preprocessing: Enable YAML preprocessing features
            validate_dependencies: Enable dependency validation
            cache_compilations: Enable compilation result caching
        """
        self.model_registry = model_registry or get_model_registry()
        self.development_mode = development_mode
        self.enable_preprocessing = enable_preprocessing
        self.validate_dependencies = validate_dependencies
        self.cache_compilations = cache_compilations
        
        # Initialize core YAML compiler
        self.compiler = YAMLCompiler(
            model_registry=self.model_registry,
            development_mode=development_mode,
            validation_level="development" if development_mode else "strict"
        )
        
        # Compilation cache for performance
        self._compilation_cache: Dict[str, Tuple[Pipeline, Dict[str, Any]]] = {}
        
        logger.info(f"AdvancedPipelineCompiler initialized with preprocessing={enable_preprocessing}")
    
    async def compile_with_validation(
        self,
        yaml_content: Union[str, Path],
        context: Optional[Dict[str, Any]] = None,
        preprocess: bool = True,
        validate_models: bool = True,
        validate_templates: bool = True,
        validate_tools: bool = True
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Compile pipeline with comprehensive validation reporting.
        
        Args:
            yaml_content: YAML content as string or path to YAML file
            context: Template context variables for compilation
            preprocess: Whether to preprocess YAML before compilation
            validate_models: Whether to validate model configurations
            validate_templates: Whether to validate template usage
            validate_tools: Whether to validate tool configurations
            
        Returns:
            Tuple of (compiled Pipeline, validation report dictionary)
            
        Raises:
            PipelineCompilerError: If compilation fails
            PipelineValidationError: If validation fails with errors
        """
        try:
            # Handle file path input
            if isinstance(yaml_content, (str, Path)) and Path(yaml_content).exists():
                yaml_file = Path(yaml_content)
                logger.info(f"Loading YAML pipeline from file: {yaml_file}")
                yaml_content = yaml_file.read_text(encoding='utf-8')
            elif isinstance(yaml_content, Path):
                raise FileNotFoundError(f"YAML file not found: {yaml_content}")
            
            # Generate cache key
            cache_key = self._generate_cache_key(yaml_content, context or {})
            
            # Check cache if enabled
            if self.cache_compilations and cache_key in self._compilation_cache:
                logger.info("Using cached compilation result")
                return self._compilation_cache[cache_key]
            
            # Preprocess YAML if enabled
            if preprocess and self.enable_preprocessing:
                logger.info("Preprocessing YAML content")
                yaml_content = await self._preprocess_yaml(yaml_content, context or {})
            
            logger.info("Starting advanced pipeline compilation with validation")
            
            # Compile pipeline with enhanced validation
            pipeline = await self.compiler.compile(
                yaml_content=yaml_content,
                context=context or {},
                resolve_ambiguities=True
            )
            
            # Generate comprehensive validation report
            validation_report = self._generate_validation_report(
                pipeline=pipeline,
                yaml_content=yaml_content,
                validate_models=validate_models,
                validate_templates=validate_templates,
                validate_tools=validate_tools
            )
            
            # Cache result if enabled
            if self.cache_compilations:
                self._compilation_cache[cache_key] = (pipeline, validation_report)
            
            logger.info(f"Pipeline '{pipeline.id}' compiled successfully with validation")
            
            # Check for validation errors
            if validation_report.get("has_errors", False):
                error_count = validation_report.get("stats", {}).get("errors", 0)
                raise PipelineValidationError(
                    f"Pipeline validation failed with {error_count} errors. "
                    f"Check validation report for details."
                )
            
            return pipeline, validation_report
            
        except YAMLCompilerError as e:
            logger.error(f"Advanced pipeline compilation failed: {e}")
            raise PipelineCompilerError(f"Failed to compile pipeline: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during advanced compilation: {e}")
            raise PipelineCompilerError(f"Unexpected compilation error: {e}") from e
    
    async def analyze_dependencies(
        self,
        yaml_content: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Analyze pipeline dependencies without full compilation.
        
        Args:
            yaml_content: YAML content as string or path to YAML file
            
        Returns:
            Dictionary containing dependency analysis results
        """
        try:
            # Handle file path input
            if isinstance(yaml_content, (str, Path)) and Path(yaml_content).exists():
                yaml_file = Path(yaml_content)
                yaml_content = yaml_file.read_text(encoding='utf-8')
            elif isinstance(yaml_content, Path):
                raise FileNotFoundError(f"YAML file not found: {yaml_content}")
            
            logger.info("Analyzing pipeline dependencies")
            
            # Parse YAML to extract dependency information
            yaml_data = yaml.safe_load(yaml_content)
            
            analysis = {
                "models_required": [],
                "tools_required": [],
                "templates_used": [],
                "external_dependencies": [],
                "task_dependencies": {},
                "validation_requirements": []
            }
            
            # Analyze models
            if "tasks" in yaml_data:
                for task_id, task_config in yaml_data["tasks"].items():
                    if "model" in task_config:
                        model_spec = task_config["model"]
                        analysis["models_required"].append({
                            "task": task_id,
                            "model": model_spec,
                            "type": "explicit" if model_spec != "AUTO" else "auto"
                        })
                    
                    # Analyze tools
                    if "tools" in task_config:
                        tools = task_config["tools"]
                        if isinstance(tools, list):
                            for tool in tools:
                                analysis["tools_required"].append({
                                    "task": task_id,
                                    "tool": tool
                                })
                    
                    # Analyze task dependencies
                    if "depends_on" in task_config:
                        deps = task_config["depends_on"]
                        if isinstance(deps, (list, str)):
                            analysis["task_dependencies"][task_id] = deps if isinstance(deps, list) else [deps]
            
            # Extract template variables
            template_vars = self.compiler.get_template_variables(yaml_content)
            analysis["templates_used"] = template_vars
            
            logger.info(f"Dependency analysis complete: {len(analysis['models_required'])} models, "
                       f"{len(analysis['tools_required'])} tools, {len(template_vars)} templates")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            return {
                "error": str(e),
                "models_required": [],
                "tools_required": [],
                "templates_used": [],
                "external_dependencies": [],
                "task_dependencies": {},
                "validation_requirements": []
            }
    
    async def validate_pipeline_spec(
        self,
        yaml_content: Union[str, Path],
        strict_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Validate pipeline YAML specification without compilation.
        
        Args:
            yaml_content: YAML content as string or path to YAML file
            strict_mode: Enable strict validation mode
            
        Returns:
            Dictionary containing validation results
        """
        try:
            # Handle file path input
            if isinstance(yaml_content, (str, Path)) and Path(yaml_content).exists():
                yaml_file = Path(yaml_content)
                yaml_content = yaml_file.read_text(encoding='utf-8')
            elif isinstance(yaml_content, Path):
                return {
                    "valid": False,
                    "errors": [f"YAML file not found: {yaml_content}"],
                    "warnings": [],
                    "info": []
                }
            
            logger.info(f"Validating pipeline specification (strict_mode={strict_mode})")
            
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "info": [],
                "schema_valid": False,
                "syntax_valid": False,
                "dependencies_valid": False
            }
            
            # Check YAML syntax
            try:
                yaml_data = yaml.safe_load(yaml_content)
                validation_result["syntax_valid"] = True
                validation_result["info"].append("YAML syntax is valid")
            except yaml.YAMLError as e:
                validation_result["valid"] = False
                validation_result["errors"].append(f"YAML syntax error: {e}")
                return validation_result
            
            # Validate basic schema
            required_fields = ["name", "tasks"]
            for field in required_fields:
                if field not in yaml_data:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Required field missing: {field}")
            
            if validation_result["errors"]:
                return validation_result
            
            validation_result["schema_valid"] = True
            
            # Validate tasks structure
            if not isinstance(yaml_data.get("tasks"), dict):
                validation_result["valid"] = False
                validation_result["errors"].append("'tasks' must be a dictionary")
                return validation_result
            
            # Validate individual tasks
            for task_id, task_config in yaml_data["tasks"].items():
                if not isinstance(task_config, dict):
                    validation_result["errors"].append(f"Task '{task_id}' must be a dictionary")
                    continue
                
                # Check required task fields
                if "prompt" not in task_config and "template" not in task_config:
                    if strict_mode:
                        validation_result["errors"].append(f"Task '{task_id}' missing prompt or template")
                    else:
                        validation_result["warnings"].append(f"Task '{task_id}' missing prompt or template")
                
                # Check model specification
                if "model" in task_config:
                    model_spec = task_config["model"]
                    if not isinstance(model_spec, str):
                        validation_result["errors"].append(f"Task '{task_id}' model must be a string")
            
            # Validate dependencies if requested
            if self.validate_dependencies:
                dependency_issues = await self._validate_task_dependencies(yaml_data)
                validation_result["errors"].extend(dependency_issues.get("errors", []))
                validation_result["warnings"].extend(dependency_issues.get("warnings", []))
                validation_result["dependencies_valid"] = len(dependency_issues.get("errors", [])) == 0
            
            # Final validation status
            validation_result["valid"] = len(validation_result["errors"]) == 0
            
            logger.info(f"Pipeline validation complete: valid={validation_result['valid']}, "
                       f"errors={len(validation_result['errors'])}, warnings={len(validation_result['warnings'])}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Pipeline validation failed: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {e}"],
                "warnings": [],
                "info": [],
                "schema_valid": False,
                "syntax_valid": False,
                "dependencies_valid": False
            }
    
    def get_template_context_requirements(
        self,
        yaml_content: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Extract template context requirements from YAML specification.
        
        Args:
            yaml_content: YAML content as string or path to YAML file
            
        Returns:
            Dictionary containing template requirements analysis
        """
        try:
            # Handle file path input
            if isinstance(yaml_content, (str, Path)) and Path(yaml_content).exists():
                yaml_file = Path(yaml_content)
                yaml_content = yaml_file.read_text(encoding='utf-8')
            elif isinstance(yaml_content, Path):
                return {"error": f"YAML file not found: {yaml_content}"}
            
            logger.info("Analyzing template context requirements")
            
            # Extract template variables
            template_vars = self.compiler.get_template_variables(yaml_content)
            
            # Analyze variable usage patterns
            requirements = {
                "required_variables": template_vars,
                "variable_usage": {},
                "optional_variables": [],
                "default_values": {},
                "validation_patterns": {}
            }
            
            # Parse YAML to analyze variable usage
            yaml_data = yaml.safe_load(yaml_content)
            
            # Check for default values in YAML
            if "defaults" in yaml_data:
                requirements["default_values"] = yaml_data["defaults"]
                for var in template_vars:
                    if var in yaml_data["defaults"]:
                        requirements["optional_variables"].append(var)
            
            # Analyze variable usage in tasks
            if "tasks" in yaml_data:
                for task_id, task_config in yaml_data["tasks"].items():
                    task_content = str(task_config)
                    for var in template_vars:
                        if f"{{{var}}}" in task_content or f"{{{{ {var} }}}}" in task_content:
                            if var not in requirements["variable_usage"]:
                                requirements["variable_usage"][var] = []
                            requirements["variable_usage"][var].append(task_id)
            
            logger.info(f"Template analysis complete: {len(template_vars)} variables identified")
            
            return requirements
            
        except Exception as e:
            logger.error(f"Template context analysis failed: {e}")
            return {"error": str(e)}
    
    async def _preprocess_yaml(
        self,
        yaml_content: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Preprocess YAML content with advanced features.
        
        Args:
            yaml_content: Raw YAML content
            context: Template context for preprocessing
            
        Returns:
            Preprocessed YAML content
        """
        # For now, return content as-is
        # Future enhancements could include:
        # - Include file processing
        # - Macro expansion
        # - Conditional compilation
        return yaml_content
    
    def _generate_validation_report(
        self,
        pipeline: Pipeline,
        yaml_content: str,
        validate_models: bool = True,
        validate_templates: bool = True,
        validate_tools: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for compiled pipeline.
        
        Args:
            pipeline: Compiled pipeline object
            yaml_content: Original YAML content
            validate_models: Whether to validate models
            validate_templates: Whether to validate templates
            validate_tools: Whether to validate tools
            
        Returns:
            Dictionary containing validation report
        """
        report = {
            "pipeline_id": pipeline.id,
            "pipeline_name": pipeline.name,
            "validation_timestamp": datetime.now().isoformat(),
            "stats": {
                "total_tasks": len(pipeline.tasks),
                "errors": 0,
                "warnings": 0,
                "info": 0
            },
            "task_validations": {},
            "model_validations": {},
            "template_validations": {},
            "tool_validations": {},
            "has_errors": False,
            "has_warnings": False
        }
        
        # Get compiler validation report if available
        compiler_report = self.compiler.get_validation_report()
        if compiler_report:
            try:
                report["stats"]["errors"] = compiler_report.stats.errors
                report["stats"]["warnings"] = compiler_report.stats.warnings
                report["stats"]["info"] = compiler_report.stats.info
                report["has_errors"] = compiler_report.has_errors
                report["has_warnings"] = compiler_report.has_warnings
            except AttributeError:
                # Fallback if stats not available
                pass
        
        # Validate individual tasks
        for task in pipeline.tasks:
            task_validation = {
                "task_id": task.id,
                "valid": True,
                "issues": []
            }
            
            # Basic task validation
            if not task.prompt and not task.template:
                task_validation["valid"] = False
                task_validation["issues"].append("Task has no prompt or template")
            
            report["task_validations"][task.id] = task_validation
        
        return report
    
    def _generate_cache_key(
        self,
        yaml_content: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate cache key for compilation results.
        
        Args:
            yaml_content: YAML content
            context: Template context
            
        Returns:
            Cache key string
        """
        import hashlib
        content_hash = hashlib.md5(yaml_content.encode()).hexdigest()
        context_hash = hashlib.md5(str(sorted(context.items())).encode()).hexdigest()
        return f"{content_hash}_{context_hash}"
    
    async def _validate_task_dependencies(
        self,
        yaml_data: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Validate task dependency relationships.
        
        Args:
            yaml_data: Parsed YAML data
            
        Returns:
            Dictionary containing validation issues
        """
        issues = {
            "errors": [],
            "warnings": []
        }
        
        if "tasks" not in yaml_data:
            return issues
        
        tasks = yaml_data["tasks"]
        task_ids = set(tasks.keys())
        
        for task_id, task_config in tasks.items():
            if "depends_on" in task_config:
                dependencies = task_config["depends_on"]
                if isinstance(dependencies, str):
                    dependencies = [dependencies]
                elif not isinstance(dependencies, list):
                    issues["errors"].append(f"Task '{task_id}' depends_on must be string or list")
                    continue
                
                for dep in dependencies:
                    if dep not in task_ids:
                        issues["errors"].append(f"Task '{task_id}' depends on non-existent task '{dep}'")
        
        # Check for circular dependencies (basic check)
        for task_id in task_ids:
            if self._has_circular_dependency(task_id, tasks, set()):
                issues["errors"].append(f"Circular dependency detected involving task '{task_id}'")
        
        return issues
    
    def _has_circular_dependency(
        self,
        task_id: str,
        tasks: Dict[str, Any],
        visited: set
    ) -> bool:
        """
        Check for circular dependencies in task graph.
        
        Args:
            task_id: Current task ID
            tasks: All tasks dictionary
            visited: Set of visited task IDs
            
        Returns:
            True if circular dependency detected
        """
        if task_id in visited:
            return True
        
        visited.add(task_id)
        
        task_config = tasks.get(task_id, {})
        dependencies = task_config.get("depends_on", [])
        
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        elif not isinstance(dependencies, list):
            dependencies = []
        
        for dep in dependencies:
            if dep in tasks and self._has_circular_dependency(dep, tasks, visited.copy()):
                return True
        
        return False
    
    def clear_cache(self) -> None:
        """Clear the compilation cache."""
        self._compilation_cache.clear()
        logger.info("Compilation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get compilation cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        return {
            "cache_size": len(self._compilation_cache),
            "cache_enabled": self.cache_compilations,
            "cache_keys": list(self._compilation_cache.keys()) if self.development_mode else []
        }


def create_advanced_pipeline_compiler(
    model_registry: Optional[ModelRegistry] = None,
    development_mode: bool = False,
    enable_preprocessing: bool = True,
    validate_dependencies: bool = True,
    cache_compilations: bool = True
) -> AdvancedPipelineCompiler:
    """
    Create an AdvancedPipelineCompiler instance with the specified configuration.
    
    Args:
        model_registry: Optional model registry for AUTO tag resolution
        development_mode: Enable development mode features
        enable_preprocessing: Enable YAML preprocessing features
        validate_dependencies: Enable dependency validation
        cache_compilations: Enable compilation result caching
        
    Returns:
        Configured AdvancedPipelineCompiler instance
    """
    return AdvancedPipelineCompiler(
        model_registry=model_registry,
        development_mode=development_mode,
        enable_preprocessing=enable_preprocessing,
        validate_dependencies=validate_dependencies,
        cache_compilations=cache_compilations
    )