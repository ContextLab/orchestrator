"""YAML parser and compiler with AUTO tag resolution."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

import yaml
from jinja2 import Environment, StrictUndefined

from ..core.model import Model
from ..core.pipeline import Pipeline
from ..core.task import Task
from .ambiguity_resolver import AmbiguityResolver
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
    ) -> None:
        """
        Initialize YAML compiler.
        
        Args:
            schema_validator: Schema validator instance
            ambiguity_resolver: Ambiguity resolver instance
        """
        self.schema_validator = schema_validator or SchemaValidator()
        self.ambiguity_resolver = ambiguity_resolver or AmbiguityResolver()
        self.template_engine = Environment(undefined=StrictUndefined)
        
        # Regex pattern for AUTO tags
        self.auto_tag_pattern = re.compile(r'<AUTO>(.*?)</AUTO>', re.DOTALL)
    
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
            
            # Step 3: Process templates
            processed = self._process_templates(raw_pipeline, context or {})
            
            # Step 4: Detect and resolve ambiguities
            if resolve_ambiguities:
                resolved = await self._resolve_ambiguities(processed)
            else:
                resolved = processed
            
            # Step 5: Build pipeline object
            return self._build_pipeline(resolved)
            
        except Exception as e:
            raise YAMLCompilerError(f"Failed to compile YAML: {e}") from e
    
    def _parse_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """
        Parse YAML content safely.
        
        Args:
            yaml_content: YAML content as string
            
        Returns:
            Parsed YAML as dictionary
            
        Raises:
            YAMLCompilerError: If YAML parsing fails
        """
        try:
            return yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise YAMLCompilerError(f"Invalid YAML: {e}") from e
    
    def _process_templates(
        self, 
        pipeline_def: Dict[str, Any], 
        context: Dict[str, Any]
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
                try:
                    template = self.template_engine.from_string(value)
                    return template.render(**context)
                except Exception as e:
                    raise TemplateRenderError(f"Failed to render template '{value}': {e}") from e
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            return value
        
        return process_value(pipeline_def)
    
    async def _resolve_ambiguities(self, pipeline_def: Dict[str, Any]) -> Dict[str, Any]:
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
            resolved_str = str(resolved_value) if not isinstance(resolved_value, str) else resolved_value
            resolved_content = resolved_content.replace(f"<AUTO>{match}</AUTO>", resolved_str)
        
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
        action = task_def["action"]
        parameters = task_def.get("parameters", {})
        dependencies = task_def.get("dependencies", [])
        timeout = task_def.get("timeout")
        max_retries = task_def.get("max_retries", 3)
        metadata = task_def.get("metadata", {})
        
        # Add additional metadata from task definition
        if "on_failure" in task_def:
            metadata["on_failure"] = task_def["on_failure"]
        if "requires_model" in task_def:
            metadata["requires_model"] = task_def["requires_model"]
        
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
    
    def get_template_variables(self, yaml_content: str) -> List[str]:
        """
        Extract template variables from YAML content.
        
        Args:
            yaml_content: YAML content
            
        Returns:
            List of template variable names
        """
        # Simple regex to find {{ variable }} patterns
        variable_pattern = re.compile(r'\{\{\s*([^}]+)\s*\}\}')
        matches = variable_pattern.findall(yaml_content)
        
        # Extract variable names (handle dot notation)
        variables = []
        for match in matches:
            # Take the first part before any dots or filters
            var_name = match.split('.')[0].split('|')[0].strip()
            if var_name not in variables:
                variables.append(var_name)
        
        return variables