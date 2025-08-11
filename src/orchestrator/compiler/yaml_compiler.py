"""YAML parser and compiler with AUTO tag resolution."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, StrictUndefined

from ..core.pipeline import Pipeline
from ..core.task import Task
from ..core.template_metadata import TemplateMetadata
from ..core.exceptions import YAMLCompilerError
from ..core.error_handling import ErrorHandler
from ..core.file_inclusion import FileInclusionProcessor, FileInclusionError
from .ambiguity_resolver import AmbiguityResolver
from .auto_tag_yaml_parser import AutoTagYAMLParser
from .schema_validator import SchemaValidator
from .error_handler_schema import ErrorHandlerSchemaValidator

logger = logging.getLogger(__name__)


class AutoTagNotFoundError(YAMLCompilerError):
    """Raised when AUTO tag resolution fails."""

    pass


class TemplateRenderError(YAMLCompilerError):
    """Raised when template rendering fails."""

    pass


class YAMLCompiler:
    """
    Compiles YAML definitions into executable pipelines.

    The compiler handles:
    - YAML parsing and validation
    - Template processing with Jinja2
    - AUTO tag detection and resolution
    - Pipeline object construction
    """

    def __init__(
        self,
        schema_validator: Optional[SchemaValidator] = None,
        ambiguity_resolver: Optional[AmbiguityResolver] = None,
        model_registry: Optional[Any] = None,
        error_handler_validator: Optional[ErrorHandlerSchemaValidator] = None,
        file_inclusion_processor: Optional[FileInclusionProcessor] = None,
    ) -> None:
        """
        Initialize YAML compiler.

        Args:
            schema_validator: Schema validator instance
            ambiguity_resolver: Ambiguity resolver instance
            model_registry: Model registry for ambiguity resolution
            error_handler_validator: Error handler validator instance
            file_inclusion_processor: File inclusion processor instance
        """
        self.schema_validator = schema_validator or SchemaValidator()
        self.error_handler_validator = error_handler_validator or ErrorHandlerSchemaValidator()
        self.file_inclusion_processor = file_inclusion_processor or FileInclusionProcessor()

        # Create ambiguity resolver - optional for compilation without resolution
        if ambiguity_resolver:
            self.ambiguity_resolver = ambiguity_resolver
        elif model_registry:
            try:
                # Try structured resolver first
                from .structured_ambiguity_resolver import StructuredAmbiguityResolver

                self.ambiguity_resolver = StructuredAmbiguityResolver(
                    model_registry=model_registry
                )
            except (ValueError, ImportError):
                # Fall back to regular resolver
                self.ambiguity_resolver = AmbiguityResolver(
                    model_registry=model_registry
                )
        else:
            # No resolver - AUTO tags will be preserved
            self.ambiguity_resolver = None
            logger.info("No model registry provided - AUTO tags will be preserved")

        self.template_engine = Environment(undefined=StrictUndefined)

        # Add custom filters to Jinja2 environment
        self._register_custom_filters()

        # Regex pattern for AUTO tags
        self.auto_tag_pattern = re.compile(r"<AUTO>(.*?)</AUTO>", re.DOTALL)

    async def compile(
        self,
        yaml_content: str,
        context: Optional[Dict[str, Any]] = None,
        resolve_ambiguities: bool = True,
    ) -> Pipeline:
        """
        Compile YAML content to Pipeline object.

        Args:
            yaml_content: YAML content as string
            context: Template context variables
            resolve_ambiguities: Whether to resolve AUTO tags

        Returns:
            Compiled Pipeline object

        Raises:
            YAMLCompilerError: If compilation fails
        """
        try:
            # Step 1: Process file inclusions in YAML content
            processed_yaml_content = await self._process_file_inclusions(yaml_content)

            # Step 2: Parse YAML safely
            raw_pipeline = self._parse_yaml(processed_yaml_content)

            # Step 3: Validate against schema
            self.schema_validator.validate(raw_pipeline)
            
            # Step 4: Validate error handling configurations
            error_issues = self.error_handler_validator.validate_pipeline_error_handling(raw_pipeline)
            if error_issues:
                raise YAMLCompilerError(f"Error handler validation failed: {'; '.join(error_issues)}")

            # Step 5: Merge default values with context
            merged_context = self._merge_defaults_with_context(
                raw_pipeline, context or {}
            )

            # Step 6: Process templates
            processed = self._process_templates(raw_pipeline, merged_context)

            # Step 7: Detect and resolve ambiguities
            if resolve_ambiguities:
                resolved = await self._resolve_ambiguities(processed)
            else:
                resolved = processed

            # Step 8: Build pipeline object with context
            return self._build_pipeline(resolved, merged_context)

        except Exception as e:
            raise YAMLCompilerError(f"Failed to compile YAML: {e}") from e

    def _merge_defaults_with_context(
        self, pipeline_def: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge default input values with provided context.

        Args:
            pipeline_def: Pipeline definition with inputs section
            context: User-provided context

        Returns:
            Merged context with defaults applied
        """
        merged = context.copy()

        # Check if inputs section exists
        if "inputs" in pipeline_def:
            inputs_def = pipeline_def["inputs"]

            # Process each input definition
            for input_name, input_spec in inputs_def.items():
                # Skip if input already provided in context
                if input_name in merged:
                    continue

                # Apply default value if specified
                if isinstance(input_spec, dict) and "default" in input_spec:
                    merged[input_name] = input_spec["default"]
                elif isinstance(input_spec, dict) and not any(
                    key in input_spec
                    for key in ["type", "description", "required", "default"]
                ):
                    # If it's a dict but not an input definition (no type/description/etc),
                    # treat it as a nested value structure
                    merged[input_name] = input_spec
                elif not isinstance(input_spec, dict):
                    # Direct value (e.g., batch_size: 100)
                    merged[input_name] = input_spec

        # Also merge parameters section (similar to inputs)
        if "parameters" in pipeline_def:
            params_def = pipeline_def["parameters"]

            # Process each parameter definition
            for param_name, param_spec in params_def.items():
                # Skip if parameter already provided in context
                if param_name in merged:
                    continue

                # Apply default value if specified
                if isinstance(param_spec, dict) and "default" in param_spec:
                    merged[param_name] = param_spec["default"]
                elif isinstance(param_spec, dict) and not any(
                    key in param_spec
                    for key in ["type", "description", "required", "default"]
                ):
                    # If it's a dict but not a parameter definition (no type/description/etc),
                    # treat it as a nested value structure
                    merged[param_name] = param_spec
                elif not isinstance(param_spec, dict):
                    # Direct value (e.g., max_results: 10)
                    merged[param_name] = param_spec

        return merged

    async def _process_file_inclusions(self, yaml_content: str) -> str:
        """
        Process file inclusion directives in YAML content.
        
        Args:
            yaml_content: Raw YAML content that may contain file inclusions
            
        Returns:
            YAML content with file inclusions processed
            
        Raises:
            YAMLCompilerError: If file inclusion processing fails
        """
        try:
            logger.debug("Processing file inclusions in YAML content")
            processed_content = await self.file_inclusion_processor.process_content(yaml_content)
            
            # Log if any inclusions were processed
            if processed_content != yaml_content:
                logger.info(f"File inclusions processed: {len(yaml_content)} -> {len(processed_content)} characters")
            
            return processed_content
            
        except FileInclusionError as e:
            raise YAMLCompilerError(f"File inclusion processing failed: {e}") from e
        except Exception as e:
            raise YAMLCompilerError(f"Unexpected error during file inclusion processing: {e}") from e

    def _parse_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """
        Parse YAML content safely, handling AUTO tags properly.

        Args:
            yaml_content: YAML content as string

        Returns:
            Parsed YAML as dictionary

        Raises:
            YAMLCompilerError: If YAML parsing fails
        """
        try:
            # Use AUTO tag parser to handle special AUTO tags
            parser = AutoTagYAMLParser()
            return parser.parse(yaml_content)
        except (yaml.YAMLError, ValueError) as e:
            raise YAMLCompilerError(f"Invalid YAML: {e}") from e

    def _process_templates(
        self, pipeline_def: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process Jinja2 templates in the pipeline definition.

        Args:
            pipeline_def: Pipeline definition
            context: Template context variables

        Returns:
            Processed pipeline definition

        Raises:
            TemplateRenderError: If template rendering fails
        """
        
        # First, extract all step IDs from the pipeline
        step_ids = []
        if "steps" in pipeline_def:
            for step in pipeline_def["steps"]:
                if isinstance(step, dict) and "id" in step:
                    step_ids.append(step["id"])

        def process_value(value: Any, path: List[str] = None) -> Any:
            if path is None:
                path = []
                
            if isinstance(value, str):
                # Skip processing special variables that start with $
                if any(
                    var in value
                    for var in [
                        "$item",
                        "$index",
                        "$is_first",
                        "$is_last",
                        "$iteration",
                        "$loop",
                    ]
                ):
                    return value  # Keep as-is for control flow handling

                # Skip processing prompts and content fields that may reference step results
                # These should be rendered at runtime, not compile time
                current_key = path[-1] if path else ""
                parent_key = path[-2] if len(path) >= 2 else ""
                
                # Skip template processing for:
                # 1. prompt parameters in steps
                # 2. content parameters in filesystem operations
                # 3. Any field that references step results
                # 4. Any field that contains Jinja2 control structures
                # 5. URL parameters that might reference step results
                # 6. Conditions - ALWAYS skip processing conditions
                # 7. Parallel queue fields that contain AUTO tags or references
                # This includes: if, condition, while (loop conditions), for_each conditions, until, on (parallel queue)
                if current_key in ["condition", "if", "while", "until", "on"]:
                    # Conditions and queue generation should NEVER be processed at compile time
                    # They must be evaluated at runtime when step results are available
                    # Debug logging
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Skipping condition/queue template processing for key '{current_key}' at path {path}: {value}")
                    return value
                    
                if (current_key in ["prompt", "content", "url", "text"] and parent_key == "parameters"):
                    # Check if it references any actual step IDs from this pipeline
                    if any(f"{step_id}." in value for step_id in step_ids) or \
                       any(ctrl in value for ctrl in ["{% for", "{% if", "{% set", "{% endfor", "{% endif"]) or \
                       any(attr in value for attr in [".results", ".result", ".content", ".output", ".data"]):
                        return value  # Keep as-is for runtime rendering
                
                # If the string contains templates, process them individually
                if "{{" in value and "}}" in value:
                    return self._process_mixed_templates(value, context, step_ids)
                else:
                    return value
            elif isinstance(value, dict):
                return {k: process_value(v, path + [k]) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item, path) for item in value]
            return value

        return process_value(pipeline_def)

    async def _resolve_ambiguities(
        self, pipeline_def: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect and resolve AUTO tags.

        Args:
            pipeline_def: Pipeline definition with AUTO tags

        Returns:
            Pipeline definition with resolved AUTO tags
        """

        async def process_auto_tags(obj: Any, path: str = "") -> Any:
            if isinstance(obj, str):
                # Skip AUTO tag resolution for runtime fields
                runtime_fields = [".goto", "['goto']", '["goto"]', ".for_each", "['for_each']", '["for_each"]']
                if any(path.endswith(field) for field in runtime_fields):
                    logger.info(f"Skipping AUTO tag resolution for runtime field: {path}")
                    return obj
                    
                # Check if string contains AUTO tags
                if self.auto_tag_pattern.search(obj):
                    return await self._resolve_auto_string(obj, path)
                return obj
            elif isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    result[key] = await process_auto_tags(value, new_path)
                return result
            elif isinstance(obj, list):
                return [
                    await process_auto_tags(item, f"{path}[{i}]")
                    for i, item in enumerate(obj)
                ]
            return obj

        return await process_auto_tags(pipeline_def)

    async def _resolve_auto_string(self, content: str, path: str) -> Any:
        """
        Resolve AUTO tags in a string.

        Args:
            content: String containing AUTO tags
            path: Context path for debugging

        Returns:
            Resolved content
        """
        # If no resolver, preserve AUTO tags
        if not self.ambiguity_resolver:
            logger.debug(f"No resolver available - preserving AUTO tag at {path}")
            return content
            
        # Find all AUTO tags
        matches = self.auto_tag_pattern.findall(content)

        if not matches:
            return content

        # If the entire string is a single AUTO tag, resolve it directly
        if len(matches) == 1 and content.strip() == f"<AUTO>{matches[0]}</AUTO>":
            return await self.ambiguity_resolver.resolve(matches[0].strip(), path)

        # Otherwise, resolve each AUTO tag and substitute
        resolved_content = content
        for match in matches:
            resolved_value = await self.ambiguity_resolver.resolve(match.strip(), path)
            # Convert resolved value to string for substitution
            resolved_str = (
                str(resolved_value)
                if not isinstance(resolved_value, str)
                else resolved_value
            )
            resolved_content = resolved_content.replace(
                f"<AUTO>{match}</AUTO>", resolved_str
            )

        return resolved_content

    def _build_pipeline(self, pipeline_def: Dict[str, Any], compile_context: Dict[str, Any] = None) -> Pipeline:
        """
        Build Pipeline object from definition.

        Args:
            pipeline_def: Processed pipeline definition
            compile_context: Context used during compilation (includes inputs)

        Returns:
            Pipeline object
        """
        # Extract pipeline metadata
        pipeline_id = pipeline_def.get("id", pipeline_def.get("name", "unnamed"))
        pipeline_name = pipeline_def.get("name", pipeline_id)
        version = pipeline_def.get("version", "1.0.0")
        description = pipeline_def.get("description")
        # Use compile_context if provided, otherwise fallback to pipeline_def context
        context = compile_context or pipeline_def.get("context", {})
        metadata = pipeline_def.get("metadata", {})

        # Include inputs and outputs in metadata for runtime access
        if "inputs" in pipeline_def:
            metadata["inputs"] = pipeline_def["inputs"]
        if "outputs" in pipeline_def:
            metadata["outputs"] = pipeline_def["outputs"]
        # Include model specification if present
        if "model" in pipeline_def:
            metadata["model"] = pipeline_def["model"]

        # Debug: log what's in context
        if "topic" in context:
            logger.warning(f"YAML Compiler: Creating pipeline with topic='{context['topic']}'")
        else:
            logger.warning(f"YAML Compiler: Creating pipeline WITHOUT topic. Context keys: {list(context.keys())}")
        
        # Create pipeline
        pipeline = Pipeline(
            id=pipeline_id,
            name=pipeline_name,
            context=context,
            metadata=metadata,
            version=version,
            description=description,
        )

        # Build tasks
        steps = pipeline_def.get("steps", [])
        
        # First pass: collect all step IDs
        available_steps = []
        for step_def in steps:
            if "id" in step_def:
                available_steps.append(step_def["id"])
        
        # Second pass: build tasks with template analysis
        for step_def in steps:
            task = self._build_task(step_def, available_steps)
            pipeline.add_task(task)

        return pipeline
    
    def _analyze_template(self, template_str: str, available_steps: List[str], 
                          parameter_path: Optional[str] = None) -> TemplateMetadata:
        """
        Analyze a template string to extract dependencies and requirements.
        
        Args:
            template_str: The template string to analyze
            available_steps: List of step IDs in the pipeline
            parameter_path: Path to this parameter (e.g., "parameters.url")
            
        Returns:
            TemplateMetadata object with analysis results
        """
        dependencies = set()
        context_requirements = set()
        
        # Extract step references (e.g., step_id.result, step_id.outputs.data)
        if available_steps:
            # Build pattern to match step references
            # Match: step_id.property, step_id.nested.property, etc.
            step_pattern = r'\b(' + '|'.join(re.escape(step) for step in available_steps) + r')\.'
            for match in re.finditer(step_pattern, template_str):
                dependencies.add(match.group(1))
        
        # Extract loop variables
        loop_vars = ['$item', '$index', '$is_first', '$is_last', '$iteration', '$loop']
        for var in loop_vars:
            if var in template_str:
                context_requirements.add(var)
        
        # Check for Jinja2 control structures that indicate runtime rendering
        has_control_structures = any(
            ctrl in template_str 
            for ctrl in ['{% for', '{% if', '{% set', '{% endfor', '{% endif', '{% else']
        )
        
        # Determine if runtime-only
        is_runtime_only = bool(dependencies) or bool(context_requirements) or has_control_structures
        is_compile_time = not is_runtime_only
        
        return TemplateMetadata(
            original_template=template_str,
            dependencies=dependencies,
            context_requirements=context_requirements,
            is_runtime_only=is_runtime_only,
            is_compile_time=is_compile_time,
            parameter_path=parameter_path
        )
    
    def _analyze_parameter_templates(self, params: Dict[str, Any], available_steps: List[str], 
                                   path_prefix: str = "") -> Dict[str, TemplateMetadata]:
        """
        Recursively analyze all templates in parameters.
        
        Args:
            params: Parameters dictionary
            available_steps: List of available step IDs
            path_prefix: Path prefix for nested parameters
            
        Returns:
            Dict mapping parameter paths to their template metadata
        """
        template_metadata = {}
        
        def analyze_value(value: Any, path: str) -> None:
            if isinstance(value, str) and ('{{' in value or '{%' in value):
                # This is a template string
                metadata = self._analyze_template(value, available_steps, path)
                if metadata.is_runtime_only:
                    template_metadata[path] = metadata
            elif isinstance(value, dict):
                # Recursively analyze dict
                for key, subvalue in value.items():
                    subpath = f"{path}.{key}" if path else key
                    analyze_value(subvalue, subpath)
            elif isinstance(value, list):
                # Analyze list items
                for i, item in enumerate(value):
                    analyze_value(item, f"{path}[{i}]")
        
        # Start analysis
        analyze_value(params, path_prefix)
        return template_metadata

    def _build_task(self, task_def: Dict[str, Any], available_steps: List[str]) -> Task:
        """
        Build Task object from definition with template analysis.

        Args:
            task_def: Task definition
            available_steps: List of all step IDs in the pipeline

        Returns:
            Task object with template metadata
        """
        # Extract task properties
        task_id = task_def["id"]
        task_name = task_def.get("name", task_id)

        # Handle both 'action' and 'tool' fields
        action = task_def.get("action")
        if not action and "tool" in task_def:
            # Map tool to action for compatibility
            action = task_def["tool"]
        if not action:
            # Check for parallel queue syntax
            if "create_parallel_queue" in task_def:
                action = "create_parallel_queue"
            # Check for action loop syntax (Issue #188)
            elif "action_loop" in task_def:
                action = "action_loop"
            # Check for control flow steps
            elif any(key in task_def for key in ["for_each", "while", "if", "condition"]):
                # This is a control flow step, will be handled separately
                action = "control_flow"
            else:
                raise ValueError(f"Step {task_id} missing 'action' field")

        parameters = task_def.get("parameters", {})

        # Handle dependencies which may be string or array
        # Support both 'dependencies' and 'depends_on' for backward compatibility
        dependencies = task_def.get("dependencies", task_def.get("depends_on", []))
        if isinstance(dependencies, str):
            # Handle single dependency as string or comma-separated list
            if "," in dependencies:
                dependencies = [dep.strip() for dep in dependencies.split(",")]
            else:
                dependencies = [dependencies.strip()] if dependencies.strip() else []

        timeout = task_def.get("timeout")
        max_retries = task_def.get("max_retries", 3)
        metadata = task_def.get("metadata", {})

        # Add additional metadata from task definition
        if "on_failure" in task_def:
            metadata["on_failure"] = task_def["on_failure"]
        if "requires_model" in task_def:
            metadata["requires_model"] = task_def["requires_model"]
        if "tool" in task_def:
            metadata["tool"] = task_def["tool"]
        
        # Process error handling configurations
        error_handlers = self._process_error_handlers(task_def, task_id)
        if error_handlers:
            metadata["error_handlers"] = error_handlers

        # Handle output metadata fields
        produces = task_def.get("produces")
        location = task_def.get("location")
        format_type = task_def.get("format")
        output_schema = task_def.get("output_schema", task_def.get("schema"))
        size_limit = task_def.get("size_limit")
        output_description = task_def.get("output_description")
        
        # Validate output metadata consistency
        if produces or location or format_type:
            from ..core.output_metadata import validate_output_specification
            validation_issues = validate_output_specification(
                produces=produces,
                location=location,
                format=format_type
            )
            if validation_issues:
                raise YAMLCompilerError(
                    f"Task '{task_id}' has invalid output specification: {'; '.join(validation_issues)}"
                )

        # Add control flow metadata
        for cf_key in [
            "for_each",
            "while",
            "if",
            "condition",
            "steps",
            "max_iterations",
            "max_parallel",
            "foreach",
            "parallel",
            "goto",  # Add goto to the list of control flow keys
        ]:
            if cf_key in task_def:
                metadata[cf_key] = task_def[cf_key]
                # Remove condition-related keys from task_def to prevent rendering
                # They should only be evaluated at runtime
                if cf_key in ["if", "condition"]:
                    task_def.pop(cf_key, None)
        
        # Special handling for while loops
        if "while" in task_def:
            metadata["is_while_loop"] = True
            metadata["while_condition"] = task_def["while"]

        # Analyze templates in parameters
        template_metadata = self._analyze_parameter_templates(parameters, available_steps)
        
        # Also analyze templates in action if it's a string
        if isinstance(action, str) and ('{{' in action or '{%' in action):
            action_metadata = self._analyze_template(action, available_steps, "action")
            if action_metadata.is_runtime_only:
                template_metadata["action"] = action_metadata

        # Analyze templates in output location if present
        if location and ('{{' in location or '{%' in location):
            location_metadata = self._analyze_template(location, available_steps, "location")
            if location_metadata.is_runtime_only:
                template_metadata["location"] = location_metadata
        
        # Analyze condition if present
        if "condition" in task_def:
            condition = task_def["condition"]
            if isinstance(condition, str) and ('{{' in condition or '{%' in condition):
                condition_metadata = self._analyze_template(condition, available_steps, "condition")
                if condition_metadata.is_runtime_only:
                    template_metadata["condition"] = condition_metadata
        
        # Create task with template metadata
        if action == "create_parallel_queue":
            # Import here to avoid circular dependencies
            from ..core.parallel_queue_task import ParallelQueueTask
            task = ParallelQueueTask.from_task_definition(task_def)
        elif action == "action_loop":
            # Import here to avoid circular dependencies
            from ..core.action_loop_task import ActionLoopTask
            task = ActionLoopTask.from_task_definition(task_def)
        else:
            task = Task(
                id=task_id,
                name=task_name,
                action=action,
                parameters=parameters,
                dependencies=dependencies,
                timeout=timeout,
                max_retries=max_retries,
                metadata=metadata,
            )
        
        # Set template metadata on task
        task.template_metadata = template_metadata
        
        # Set output metadata if provided
        if produces or location or format_type:
            from ..core.output_metadata import create_output_metadata
            task.set_output_metadata(
                produces=produces,
                location=location,
                format=format_type,
                schema=output_schema,
                size_limit=size_limit,
                description=output_description
            )
        
        # Set error handlers on task if available
        if error_handlers:
            # Add error handlers to task metadata for runtime access
            if not hasattr(task, 'error_handlers'):
                task.error_handlers = error_handlers
            else:
                task.error_handlers.extend(error_handlers)
        
        return task

    def detect_auto_tags(self, content: str) -> List[str]:
        """
        Detect AUTO tags in content.

        Args:
            content: Content to search

        Returns:
            List of AUTO tag contents
        """
        if isinstance(content, str):
            return self.auto_tag_pattern.findall(content)
        return []

    def has_auto_tags(self, obj: Any) -> bool:
        """
        Check if object contains AUTO tags.

        Args:
            obj: Object to check

        Returns:
            True if AUTO tags are found
        """
        if isinstance(obj, str):
            return bool(self.auto_tag_pattern.search(obj))
        elif isinstance(obj, dict):
            return any(self.has_auto_tags(value) for value in obj.values())
        elif isinstance(obj, list):
            return any(self.has_auto_tags(item) for item in obj)
        return False

    def validate_yaml(self, yaml_content: str) -> bool:
        """
        Validate YAML content without compilation.

        Args:
            yaml_content: YAML content to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            pipeline_def = self._parse_yaml(yaml_content)
            self.schema_validator.validate(pipeline_def)
            return True
        except Exception:
            return False

    def _register_custom_filters(self) -> None:
        """Register custom Jinja2 filters."""
        import re as regex_module

        def regex_search(value, pattern, group=None):
            """Search for regex pattern in value."""
            if not isinstance(value, str):
                value = str(value)
            match = regex_module.search(pattern, value, regex_module.DOTALL)
            if match:
                if group is not None:
                    try:
                        # Handle numeric group references
                        if isinstance(group, str) and group.startswith("\\"):
                            group_num = int(group[1:])
                            return (
                                match.group(group_num)
                                if group_num <= match.lastindex
                                else ""
                            )
                        return match.group(group)
                    except (IndexError, ValueError):
                        return ""
                return match.group(0)
            return ""

        # Register filters
        self.template_engine.filters["regex_search"] = regex_search

        # Also add other commonly used filters that might be missing
        self.template_engine.filters["default"] = lambda v, d="": v if v else d
        self.template_engine.filters["lower"] = lambda v: str(v).lower()
        self.template_engine.filters["upper"] = lambda v: str(v).upper()
        self.template_engine.filters["replace"] = lambda v, old, new: str(v).replace(
            old, new
        )

        # Add missing filters
        import json
        from datetime import datetime
        import re as re_module

        # Slugify filter
        def slugify(value):
            """Convert string to slug format."""
            value = str(value).lower()
            # Replace spaces and underscores with hyphens
            value = re_module.sub(r"[\s_]+", "-", value)
            # Remove non-alphanumeric characters except hyphens
            value = re_module.sub(r"[^a-z0-9-]", "", value)
            # Remove multiple consecutive hyphens
            value = re_module.sub(r"-+", "-", value)
            # Strip hyphens from start and end
            return value.strip("-")

        # Date filter
        def date_filter(value, format="%Y-%m-%d"):
            """Format datetime value."""
            if isinstance(value, str):
                # Parse ISO format
                try:
                    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except Exception:
                    value = datetime.now()
            elif not isinstance(value, datetime):
                value = datetime.now()

            # Handle special format strings
            format = format.replace("Y", "%Y").replace("m", "%m").replace("d", "%d")
            format = format.replace("H", "%H").replace("i", "%M").replace("s", "%S")
            return value.strftime(format)

        # JSON filter
        def json_filter(value, indent=None):
            """Convert value to JSON string."""
            return json.dumps(value, indent=indent, default=str)

        # Now function for templates
        def now():
            """Return current datetime."""
            return datetime.now()

        # from_json filter
        def from_json(value):
            """Parse JSON string to object."""
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except Exception:
                    return value
            return value

        # Basename filter
        def basename(value):
            """Get the basename of a path."""
            import os
            return os.path.basename(str(value))
        
        self.template_engine.filters["slugify"] = slugify
        self.template_engine.filters["date"] = date_filter
        self.template_engine.filters["json"] = json_filter
        self.template_engine.filters["from_json"] = from_json
        self.template_engine.filters["to_json"] = json_filter  # Alias for json
        self.template_engine.filters["basename"] = basename
        self.template_engine.globals["now"] = now

        # Add special variables that should not be processed as regular templates
        # These will be handled by the control flow system
        self.special_vars = {"$item", "$index", "$is_first", "$is_last"}

    def get_template_variables(self, yaml_content: str) -> List[str]:
        """
        Extract template variables from YAML content.

        Args:
            yaml_content: YAML content

        Returns:
            List of template variable names
        """
        # Simple regex to find {{ variable }} patterns
        variable_pattern = re.compile(r"\{\{\s*([^}]+)\s*\}\}")
        matches = variable_pattern.findall(yaml_content)

        # Extract variable names (handle dot notation)
        variables = []
        for match in matches:
            # Take the first part before any dots or filters
            var_name = match.split(".")[0].split("|")[0].strip()
            if var_name not in variables:
                variables.append(var_name)

        return variables

    def _process_mixed_templates(self, value: str, context: Dict[str, Any], step_ids: List[str] = None) -> str:
        """
        Process a string that may contain both compile-time and runtime templates.

        This method processes each template individually, resolving compile-time
        templates while preserving runtime templates.

        Args:
            value: String containing templates
            context: Template context

        Returns:
            String with compile-time templates resolved
        """
        import re

        # Find all templates in the string
        template_pattern = re.compile(r"\{\{[^}]+\}\}")
        templates = template_pattern.findall(value)

        if not templates:
            return value

        # Process each template individually
        result = value
        for template in templates:
            try:
                # Extract variable name from template
                var_match = re.match(r"\{\{\s*([^|}\s]+)", template)
                if var_match:
                    var_name = var_match.group(1).strip()
                    # Skip templates that reference step results
                    if step_ids and any(f"{step_id}." in var_name for step_id in step_ids):
                        continue  # Skip this template, keep it as-is
                    
                    # Also skip if template contains any attribute access that might be runtime
                    if "." in var_name and var_name.split(".")[0] not in context:
                        continue  # Skip this template, keep it as-is
                
                # Try to render this specific template
                template_engine = self.template_engine.from_string(template)
                rendered = template_engine.render(**context)
                result = result.replace(template, rendered, 1)
            except Exception as e:
                # If rendering fails, check if it's a runtime reference
                error_str = str(e).lower()

                # Check if this template contains runtime references
                # Any template with a dot notation that failed to render is likely a runtime reference
                if "." in template:
                    # Keep this template for runtime resolution
                    continue
                
                # Also check for loop variables
                if any(ref in template for ref in ["$item", "$index", "$iteration", "$loop"]):
                    # Keep this template for runtime resolution
                    continue

                # Also check for undefined step references
                if "undefined" in error_str or "has no attribute" in error_str:
                    # Extract the variable name from the template
                    var_match = re.match(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)", template)
                    if var_match:
                        var_name = var_match.group(1)
                        # If it's not in context and looks like a step reference, preserve it
                        if var_name not in context:
                            continue

                # If it's not a runtime reference and still fails, it's an error
                raise TemplateRenderError(
                    f"Failed to render template '{template}': {e}"
                ) from e

        return result
    
    def _process_error_handlers(self, task_def: Dict[str, Any], task_id: str) -> List[ErrorHandler]:
        """Process error handler configurations from task definition."""
        error_handlers = []
        
        # Process on_error field (supports both legacy and new formats)
        on_error = task_def.get("on_error")
        if on_error:
            if isinstance(on_error, list):
                # New list format - each item is a handler configuration
                for i, handler_config in enumerate(on_error):
                    try:
                        handler = self._create_error_handler(handler_config, f"{task_id}_handler_{i}")
                        error_handlers.append(handler)
                    except Exception as e:
                        logger.warning(f"Failed to create error handler {i} for task {task_id}: {e}")
            else:
                # Legacy format or single handler dict
                try:
                    handler = self._create_error_handler(on_error, f"{task_id}_legacy_handler")
                    error_handlers.append(handler)
                except Exception as e:
                    logger.warning(f"Failed to create legacy error handler for task {task_id}: {e}")
        
        # Process separate error_handlers field if present
        explicit_handlers = task_def.get("error_handlers")
        if explicit_handlers and isinstance(explicit_handlers, list):
            for i, handler_config in enumerate(explicit_handlers):
                try:
                    handler = self._create_error_handler(handler_config, f"{task_id}_explicit_{i}")
                    error_handlers.append(handler)
                except Exception as e:
                    logger.warning(f"Failed to create explicit error handler {i} for task {task_id}: {e}")
        
        return error_handlers
    
    def _create_error_handler(self, handler_config: Any, handler_id: str) -> ErrorHandler:
        """Create an ErrorHandler from configuration."""
        if isinstance(handler_config, str):
            # Simple string action
            return ErrorHandler(
                handler_action=handler_config,
                error_types=["*"],  # Catch all errors by default
                retry_with_handler=False,  # Legacy behavior
                continue_on_handler_failure=True  # Legacy behavior
            )
        
        elif isinstance(handler_config, dict):
            # Check if this is a legacy ErrorHandling format
            if "action" in handler_config and "retry_count" in handler_config:
                # Convert legacy ErrorHandling to ErrorHandler
                return ErrorHandler(
                    handler_action=handler_config.get("action"),
                    error_types=["*"],
                    retry_with_handler=handler_config.get("retry_count", 0) > 0,
                    max_handler_retries=handler_config.get("retry_count", 0),
                    continue_on_handler_failure=handler_config.get("continue_on_error", False),
                    fallback_value=handler_config.get("fallback_value"),
                    priority=1000  # Lower priority for legacy handlers
                )
            else:
                # New ErrorHandler format - validate and create
                # Apply defaults for missing fields
                config = handler_config.copy()
                
                # Set default values
                if "error_types" not in config:
                    config["error_types"] = ["*"]
                if "retry_with_handler" not in config:
                    config["retry_with_handler"] = True
                if "priority" not in config:
                    config["priority"] = 100
                if "enabled" not in config:
                    config["enabled"] = True
                
                return ErrorHandler(**config)
        
        else:
            raise ValueError(f"Invalid error handler configuration: {type(handler_config)}")
