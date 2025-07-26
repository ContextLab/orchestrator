"""YAML parser and compiler with AUTO tag resolution."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, StrictUndefined

from ..core.pipeline import Pipeline
from ..core.task import Task
from .ambiguity_resolver import AmbiguityResolver
from .auto_tag_yaml_parser import AutoTagYAMLParser
from .schema_validator import SchemaValidator


class YAMLCompilerError(Exception):
    """Base exception for YAML compiler errors."""

    pass


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
    ) -> None:
        """
        Initialize YAML compiler.

        Args:
            schema_validator: Schema validator instance
            ambiguity_resolver: Ambiguity resolver instance
            model_registry: Model registry for ambiguity resolution
        """
        self.schema_validator = schema_validator or SchemaValidator()

        # Create ambiguity resolver - require real model
        if ambiguity_resolver:
            self.ambiguity_resolver = ambiguity_resolver
        else:
            if not model_registry:
                raise ValueError(
                    "No model registry provided for ambiguity resolution. "
                    "AUTO tags require an AI model to resolve ambiguities."
                )
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
            # Step 1: Parse YAML safely
            raw_pipeline = self._parse_yaml(yaml_content)

            # Step 2: Validate against schema
            self.schema_validator.validate(raw_pipeline)

            # Step 3: Merge default values with context
            merged_context = self._merge_defaults_with_context(
                raw_pipeline, context or {}
            )

            # Step 4: Process templates
            processed = self._process_templates(raw_pipeline, merged_context)

            # Step 5: Detect and resolve ambiguities
            if resolve_ambiguities:
                resolved = await self._resolve_ambiguities(processed)
            else:
                resolved = processed

            # Step 6: Build pipeline object
            return self._build_pipeline(resolved)

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

        return merged

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

        def process_value(value: Any) -> Any:
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

                # If the string contains templates, process them individually
                if "{{" in value and "}}" in value:
                    return self._process_mixed_templates(value, context)
                else:
                    return value
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
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

    def _build_pipeline(self, pipeline_def: Dict[str, Any]) -> Pipeline:
        """
        Build Pipeline object from definition.

        Args:
            pipeline_def: Processed pipeline definition

        Returns:
            Pipeline object
        """
        # Extract pipeline metadata
        pipeline_id = pipeline_def.get("id", pipeline_def.get("name", "unnamed"))
        pipeline_name = pipeline_def.get("name", pipeline_id)
        version = pipeline_def.get("version", "1.0.0")
        description = pipeline_def.get("description")
        context = pipeline_def.get("context", {})
        metadata = pipeline_def.get("metadata", {})

        # Include inputs and outputs in metadata for runtime access
        if "inputs" in pipeline_def:
            metadata["inputs"] = pipeline_def["inputs"]
        if "outputs" in pipeline_def:
            metadata["outputs"] = pipeline_def["outputs"]
        # Include model specification if present
        if "model" in pipeline_def:
            metadata["model"] = pipeline_def["model"]

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
        for step_def in steps:
            task = self._build_task(step_def)
            pipeline.add_task(task)

        return pipeline

    def _build_task(self, task_def: Dict[str, Any]) -> Task:
        """
        Build Task object from definition.

        Args:
            task_def: Task definition

        Returns:
            Task object
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
            # Check for control flow steps
            if any(key in task_def for key in ["for_each", "while", "if", "condition"]):
                # This is a control flow step, will be handled separately
                action = "control_flow"
            else:
                raise ValueError(f"Step {task_id} missing 'action' field")

        parameters = task_def.get("parameters", {})

        # Handle dependencies which may be string or array
        dependencies = task_def.get("depends_on", [])
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
        ]:
            if cf_key in task_def:
                metadata[cf_key] = task_def[cf_key]

        return Task(
            id=task_id,
            name=task_name,
            action=action,
            parameters=parameters,
            dependencies=dependencies,
            timeout=timeout,
            max_retries=max_retries,
            metadata=metadata,
        )

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

        self.template_engine.filters["slugify"] = slugify
        self.template_engine.filters["date"] = date_filter
        self.template_engine.filters["json"] = json_filter
        self.template_engine.filters["from_json"] = from_json
        self.template_engine.filters["to_json"] = json_filter  # Alias for json
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

    def _process_mixed_templates(self, value: str, context: Dict[str, Any]) -> str:
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
                # Try to render this specific template
                template_engine = self.template_engine.from_string(template)
                rendered = template_engine.render(**context)
                result = result.replace(template, rendered, 1)
            except Exception as e:
                # If rendering fails, check if it's a runtime reference
                error_str = str(e).lower()

                # Runtime patterns that should be preserved
                runtime_patterns = [
                    "outputs.",
                    "$results.",
                    "steps.",
                    ".result",
                    ".output",
                    ".value",
                    ".data",
                    "$item",
                    "$index",
                    "$iteration",
                    "$loop",
                ]

                # Check if this template contains runtime references
                if any(ref in template for ref in runtime_patterns):
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
