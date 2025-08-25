"""
Template resolution utilities for output tracking.
Provides advanced template resolution capabilities for cross-task output references.
Enhanced with Microsoft POML integration for structured markup templates.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Set, Union
from enum import Enum

from .output_metadata import OutputReference
from .output_tracker import OutputTracker

# POML integration - optional import to maintain backward compatibility
try:
    import poml
    from poml import Prompt
    POML_AVAILABLE = True
except ImportError:
    POML_AVAILABLE = False
    poml = None
    Prompt = None

logger = logging.getLogger(__name__)


class TemplateFormat(Enum):
    """Enumeration of supported template formats."""
    JINJA2 = "jinja2"
    POML = "poml"
    HYBRID = "hybrid"
    PLAIN = "plain"


class TemplateFormatDetector:
    """
    Detects template format automatically based on content patterns.
    """
    
    def __init__(self):
        self.jinja_pattern = re.compile(r'{{\s*[^}]+\s*}}|\{\%\s*[^%]+\s*\%\}')
        self.poml_patterns = [
            re.compile(r'<(role|task|example|examples|hint|output-format|document|table|img|poml)[\s>]'),
            re.compile(r'</(role|task|example|examples|hint|output-format|document|table|img|poml)>'),
        ]
    
    def detect_format(self, template_content: str) -> TemplateFormat:
        """
        Detect the format of a template string.
        
        Args:
            template_content: The template string to analyze
            
        Returns:
            TemplateFormat enum indicating the detected format
        """
        if not isinstance(template_content, str) or not template_content.strip():
            return TemplateFormat.PLAIN
        
        has_jinja = bool(self.jinja_pattern.search(template_content))
        has_poml = any(pattern.search(template_content) for pattern in self.poml_patterns)
        
        if has_poml and has_jinja:
            return TemplateFormat.HYBRID
        elif has_poml:
            return TemplateFormat.POML
        elif has_jinja:
            return TemplateFormat.JINJA2
        else:
            return TemplateFormat.PLAIN
    
    def is_poml_compatible(self, template_content: str) -> bool:
        """Check if template content is compatible with POML processing."""
        format_type = self.detect_format(template_content)
        return format_type in [TemplateFormat.POML, TemplateFormat.HYBRID]


class POMLIntegrationError(Exception):
    """Exception raised when POML integration fails."""
    pass


class POMLTemplateProcessor:
    """
    Processes POML templates and integrates them with the existing template system.
    """
    
    def __init__(self, output_tracker: Optional[OutputTracker] = None):
        """
        Initialize POML template processor.
        
        Args:
            output_tracker: Optional output tracker for context integration
        """
        if not POML_AVAILABLE:
            raise POMLIntegrationError("POML SDK is not available. Install with: pip install poml")
        
        self.output_tracker = output_tracker
        self.format_detector = TemplateFormatDetector()
    
    def render_poml_template(self, template_content: str, context: Dict[str, Any]) -> str:
        """
        Render a POML template with context data.
        
        Args:
            template_content: POML template content
            context: Context dictionary for template rendering
            
        Returns:
            Rendered template string
        """
        try:
            # Create POML prompt programmatically from template content
            prompt = self._build_poml_from_template(template_content, context)
            
            # Render to plain text format (compatible with existing system)
            result = prompt.render(chat=False)
            
            return result
            
        except Exception as e:
            logger.error(f"POML template rendering failed: {e}")
            raise POMLIntegrationError(f"Failed to render POML template: {e}")
    
    def _build_poml_from_template(self, template_content: str, context: Dict[str, Any]) -> 'Prompt':
        """
        Build a POML Prompt object from template content and context.
        
        This method handles the conversion between our template system and POML's
        programmatic API.
        """
        prompt = Prompt()
        
        with prompt:
            # Parse the template content and build POML structure
            self._parse_and_build_poml(prompt, template_content, context)
        
        return prompt
    
    def _parse_and_build_poml(self, prompt: 'Prompt', content: str, context: Dict[str, Any]):
        """
        Parse template content and build POML structure programmatically.
        
        This is a simplified parser that handles basic POML tags.
        For complex templates, we may need more sophisticated parsing.
        """
        # Simple regex-based parsing for basic POML tags
        # This handles most common use cases
        
        # Extract role content
        role_match = re.search(r'<role(?:[^>]*)>(.*?)</role>', content, re.DOTALL | re.IGNORECASE)
        if role_match:
            role_content = role_match.group(1).strip()
            # Replace Jinja2 variables with context values
            role_content = self._replace_variables(role_content, context)
            with prompt.role():
                prompt.text(role_content)
        
        # Extract task content
        task_match = re.search(r'<task(?:[^>]*)>(.*?)</task>', content, re.DOTALL | re.IGNORECASE)
        if task_match:
            task_content = task_match.group(1).strip()
            task_content = self._replace_variables(task_content, context)
            with prompt.task():
                prompt.text(task_content)
        
        # Extract examples
        examples_matches = re.finditer(r'<example(?:[^>]*)>(.*?)</example>', content, re.DOTALL | re.IGNORECASE)
        for example_match in examples_matches:
            example_content = example_match.group(1).strip()
            
            # Look for input/output within example
            input_match = re.search(r'<input(?:[^>]*)>(.*?)</input>', example_content, re.DOTALL | re.IGNORECASE)
            output_match = re.search(r'<output(?:[^>]*)>(.*?)</output>', example_content, re.DOTALL | re.IGNORECASE)
            
            with prompt.example():
                if input_match:
                    input_content = self._replace_variables(input_match.group(1).strip(), context)
                    with prompt.example_input():
                        prompt.text(input_content)
                
                if output_match:
                    output_content = self._replace_variables(output_match.group(1).strip(), context)
                    with prompt.example_output():
                        prompt.text(output_content)
                elif not input_match:
                    # If no input/output tags, use entire content
                    example_content = self._replace_variables(example_content, context)
                    prompt.text(example_content)
        
        # Extract hints
        hint_matches = re.finditer(r'<hint(?:[^>]*)>(.*?)</hint>', content, re.DOTALL | re.IGNORECASE)
        for hint_match in hint_matches:
            hint_content = hint_match.group(1).strip()
            hint_content = self._replace_variables(hint_content, context)
            with prompt.hint():
                prompt.text(hint_content)
        
        # Extract output format
        output_format_match = re.search(r'<output-format(?:[^>]*)>(.*?)</output-format>', content, re.DOTALL | re.IGNORECASE)
        if output_format_match:
            format_content = output_format_match.group(1).strip()
            format_content = self._replace_variables(format_content, context)
            with prompt.output_format():
                prompt.text(format_content)
    
    def _replace_variables(self, content: str, context: Dict[str, Any]) -> str:
        """
        Replace Jinja2-style variables in content with context values.
        
        This provides compatibility between our existing template system
        and POML's programmatic approach.
        """
        def replace_var(match):
            var_name = match.group(1).strip()
            
            # Handle nested variable access (e.g., task.result)
            if '.' in var_name:
                parts = var_name.split('.')
                value = context
                try:
                    for part in parts:
                        if hasattr(value, part):
                            value = getattr(value, part)
                        elif isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            return match.group(0)  # Return original if can't resolve
                    return str(value) if value is not None else ''
                except (AttributeError, KeyError, TypeError):
                    return match.group(0)
            else:
                # Simple variable
                if var_name in context:
                    value = context[var_name]
                    return str(value) if value is not None else ''
                return match.group(0)  # Return original if not found
        
        # Replace {{ variable }} patterns
        pattern = re.compile(r'{{\s*([^}]+)\s*}}')
        return pattern.sub(replace_var, content)
    
    def extract_data_components(self, template_content: str) -> List[Dict[str, str]]:
        """
        Extract data integration components from POML template.
        
        Returns list of components like documents, tables, images that need
        special handling for data integration.
        """
        components = []
        
        # Extract document components
        doc_matches = re.finditer(r'<document([^>]*)(?:/>|>(.*?)</document>)', template_content, re.DOTALL | re.IGNORECASE)
        for match in doc_matches:
            attrs_str = match.group(1)
            content = match.group(2) if match.group(2) else ""
            
            # Parse attributes
            attrs = self._parse_attributes(attrs_str)
            components.append({
                'type': 'document',
                'attributes': attrs,
                'content': content.strip() if content else ""
            })
        
        # Extract table components
        table_matches = re.finditer(r'<table([^>]*)(?:/>|>(.*?)</table>)', template_content, re.DOTALL | re.IGNORECASE)
        for match in table_matches:
            attrs_str = match.group(1)
            content = match.group(2) if match.group(2) else ""
            
            attrs = self._parse_attributes(attrs_str)
            components.append({
                'type': 'table',
                'attributes': attrs,
                'content': content.strip() if content else ""
            })
        
        return components
    
    def _parse_attributes(self, attrs_str: str) -> Dict[str, str]:
        """Parse HTML-style attributes from a string."""
        attrs = {}
        if not attrs_str:
            return attrs
        
        # Simple attribute parsing - handles src="value" and src=value
        attr_pattern = re.compile(r'(\w+)\s*=\s*["\']?([^"\'\s>]+)["\']?')
        matches = attr_pattern.findall(attrs_str)
        
        for name, value in matches:
            attrs[name] = value
        
        return attrs


class TemplateResolver:
    """
    Advanced template resolver for output tracking system.
    
    Handles complex template patterns, cross-task references, and validation
    of template dependencies. Enhanced with POML integration support.
    """
    
    def __init__(self, output_tracker: OutputTracker, enable_poml: bool = True):
        """
        Initialize with output tracker and optional POML support.
        
        Args:
            output_tracker: Output tracker for cross-task references
            enable_poml: Whether to enable POML template processing
        """
        self.output_tracker = output_tracker
        
        # Template patterns for Jinja2 compatibility
        self.reference_pattern = re.compile(r'{{\s*([^}]+)\s*}}')
        self.field_pattern = re.compile(r'(\w+)\.(\w+(?:\.\w+)*)')
        
        # POML integration components
        self.format_detector = TemplateFormatDetector()
        self.poml_processor = None
        
        if enable_poml and POML_AVAILABLE:
            try:
                self.poml_processor = POMLTemplateProcessor(output_tracker)
                logger.info("POML integration enabled")
            except POMLIntegrationError as e:
                logger.warning(f"POML integration failed: {e}")
                self.poml_processor = None
        elif enable_poml and not POML_AVAILABLE:
            logger.warning("POML integration requested but POML SDK not available")
    
    def extract_references(self, template: str) -> List[OutputReference]:
        """Extract all output references from a template string."""
        references = []
        
        matches = self.reference_pattern.findall(template)
        for match in matches:
            # Parse the reference (e.g., "task_id.field" or "task_id.nested.field")
            match = match.strip()
            
            if '.' in match:
                parts = match.split('.', 1)
                task_id = parts[0]
                field = parts[1]
            else:
                task_id = match
                field = None
            
            references.append(OutputReference(
                task_id=task_id,
                field=field
            ))
        
        return references
    
    def validate_template(self, template: str, available_tasks: Set[str]) -> List[str]:
        """Validate that all references in template are valid."""
        issues = []
        references = self.extract_references(template)
        
        for ref in references:
            if ref.task_id not in available_tasks:
                issues.append(f"Reference to undefined task: {ref.task_id}")
        
        return issues
    
    def resolve_template(self, template: str, default_values: Optional[Dict[str, Any]] = None) -> str:
        """
        Resolve template with output values, supporting both Jinja2 and POML formats.
        
        Args:
            template: Template string to resolve
            default_values: Default values for unresolved references
            
        Returns:
            Resolved template string
        """
        if default_values is None:
            default_values = {}
        
        # Detect template format and handle accordingly
        template_format = self.format_detector.detect_format(template)
        
        if template_format == TemplateFormat.POML and self.poml_processor:
            return self._resolve_poml_template(template, default_values)
        elif template_format == TemplateFormat.HYBRID and self.poml_processor:
            return self._resolve_hybrid_template(template, default_values)
        else:
            # Default Jinja2 resolution (backward compatibility)
            return self._resolve_jinja2_template(template, default_values)
    
    def _resolve_poml_template(self, template: str, context: Dict[str, Any]) -> str:
        """Resolve a pure POML template."""
        try:
            # Enhance context with output tracker data
            enhanced_context = self._build_enhanced_context(context)
            return self.poml_processor.render_poml_template(template, enhanced_context)
        except POMLIntegrationError as e:
            logger.warning(f"POML template resolution failed, falling back to Jinja2: {e}")
            return self._resolve_jinja2_template(template, context)
    
    def _resolve_hybrid_template(self, template: str, context: Dict[str, Any]) -> str:
        """Resolve a hybrid template with both POML and Jinja2 syntax."""
        try:
            # First, resolve Jinja2 variables in the template
            jinja2_resolved = self._resolve_jinja2_template(template, context)
            
            # Then process as POML if it still contains POML tags
            if self.format_detector.is_poml_compatible(jinja2_resolved):
                enhanced_context = self._build_enhanced_context(context)
                return self.poml_processor.render_poml_template(jinja2_resolved, enhanced_context)
            else:
                return jinja2_resolved
                
        except (POMLIntegrationError, Exception) as e:
            logger.warning(f"Hybrid template resolution failed, using Jinja2 only: {e}")
            return self._resolve_jinja2_template(template, context)
    
    def _resolve_jinja2_template(self, template: str, default_values: Dict[str, Any]) -> str:
        """Resolve template using original Jinja2 method (backward compatibility)."""
        def replace_reference(match):
            reference_str = match.group(1).strip()
            
            # Parse reference
            if '.' in reference_str:
                parts = reference_str.split('.', 1)
                task_id = parts[0]
                field = parts[1]
            else:
                task_id = reference_str
                field = None
            
            # Try to resolve from output tracker
            try:
                if field:
                    # Try to get the base output first, then navigate the field path
                    base_value = self.output_tracker.get_output(task_id)
                    current = base_value
                    
                    # Navigate through the field path
                    for part in field.split('.'):
                        if hasattr(current, part):
                            current = getattr(current, part)
                        elif isinstance(current, dict) and part in current:
                            current = current[part]
                        else:
                            # Field not found, try the output tracker's get_output method
                            try:
                                current = self.output_tracker.get_output(task_id, field)
                                break
                            except (KeyError, AttributeError):
                                return f'{{{{{reference_str}}}}}'  # Keep original if can't resolve
                    
                    return str(current) if current is not None else ''
                else:
                    value = self.output_tracker.get_output(task_id)
                    return str(value) if value is not None else ''
                    
            except (KeyError, AttributeError):
                # Fall back to default values
                if task_id in default_values:
                    if field and isinstance(default_values[task_id], dict):
                        nested_value = default_values[task_id]
                        for part in field.split('.'):
                            if isinstance(nested_value, dict) and part in nested_value:
                                nested_value = nested_value[part]
                            else:
                                return f'{{{{{reference_str}}}}}'  # Keep original if can't resolve
                        return str(nested_value)
                    elif not field:
                        return str(default_values[task_id])
                
                # Can't resolve - keep original reference
                return f'{{{{{reference_str}}}}}'
        
        return self.reference_pattern.sub(replace_reference, template)
    
    def _build_enhanced_context(self, base_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build enhanced context by combining base context with output tracker data.
        
        This provides POML templates with access to all pipeline outputs and context.
        """
        enhanced_context = dict(base_context)
        
        # Add output tracker data if available
        if self.output_tracker:
            # Add all tracked outputs to context
            for task_id in self.output_tracker.outputs.keys():
                try:
                    output = self.output_tracker.get_output(task_id)
                    enhanced_context[task_id] = output
                except (KeyError, AttributeError):
                    pass  # Skip if output not available
        
        return enhanced_context
    
    def get_template_dependencies(self, template: str) -> Set[str]:
        """Get set of task IDs that this template depends on."""
        dependencies = set()
        references = self.extract_references(template)
        
        for ref in references:
            dependencies.add(ref.task_id)
        
        return dependencies
    
    def is_template_resolvable(self, template: str) -> bool:
        """Check if template can be fully resolved with current outputs."""
        references = self.extract_references(template)
        
        for ref in references:
            if not self.output_tracker.has_output(ref.task_id):
                return False
        
        return True
    
    def get_pending_dependencies(self, template: str) -> Set[str]:
        """Get set of task IDs that need to complete before template can be resolved."""
        pending = set()
        references = self.extract_references(template)
        
        for ref in references:
            if not self.output_tracker.has_output(ref.task_id):
                pending.add(ref.task_id)
        
        return pending
    
    def register_template_references(self, task_id: str, template: str) -> None:
        """Register template references for dependency tracking."""
        references = self.extract_references(template)
        
        for ref in references:
            self.output_tracker.add_reference(task_id, ref)
    
    def batch_resolve_templates(self, templates: Dict[str, str]) -> Dict[str, str]:
        """Resolve multiple templates efficiently."""
        resolved = {}
        
        for key, template in templates.items():
            try:
                resolved[key] = self.resolve_template(template)
            except Exception as e:
                # Keep original template if resolution fails
                resolved[key] = template
        
        return resolved
    
    def create_output_location_template(self, task_id: str, base_dir: str, 
                                      extension: str = None) -> str:
        """Create a standard output location template."""
        if extension:
            return f"{base_dir}/{{{{{task_id}.result}}}}.{extension}"
        else:
            return f"{base_dir}/{{{{{task_id}.result}}}}"
    
    def resolve_file_path(self, location_template: str, ensure_dir: bool = True) -> str:
        """Resolve a file path template and optionally ensure directory exists."""
        resolved_path = self.resolve_template(location_template)
        
        if ensure_dir:
            import os
            dir_path = os.path.dirname(resolved_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
        
        return resolved_path
    
    # POML-specific utility methods
    
    def get_template_format(self, template: str) -> TemplateFormat:
        """Get the detected format of a template."""
        return self.format_detector.detect_format(template)
    
    def is_poml_available(self) -> bool:
        """Check if POML processing is available."""
        return self.poml_processor is not None
    
    def extract_poml_data_components(self, template: str) -> List[Dict[str, str]]:
        """
        Extract data integration components from POML template.
        
        Returns list of components that need special data handling.
        """
        if not self.poml_processor:
            return []
        
        return self.poml_processor.extract_data_components(template)
    
    def validate_poml_template(self, template: str) -> List[str]:
        """
        Validate POML template and return list of issues.
        
        Args:
            template: POML template to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        if not self.poml_processor:
            if self.format_detector.is_poml_compatible(template):
                issues.append("POML template detected but POML processor not available")
            return issues
        
        template_format = self.get_template_format(template)
        
        if template_format == TemplateFormat.POML:
            # Basic POML structure validation
            if not re.search(r'<(role|task)', template, re.IGNORECASE):
                issues.append("POML template should contain at least a <role> or <task> element")
        
        # Check for data components that might need file validation
        data_components = self.extract_poml_data_components(template)
        for component in data_components:
            if component['type'] == 'document' and 'src' in component['attributes']:
                src_path = component['attributes']['src']
                # Check if it's a template variable or actual file path
                if not self.reference_pattern.search(src_path):
                    # It's a literal path - we could validate existence here if needed
                    pass
        
        return issues
    
    def create_poml_template_from_components(self, 
                                           role: Optional[str] = None,
                                           task: Optional[str] = None, 
                                           examples: Optional[List[Dict[str, str]]] = None,
                                           hints: Optional[List[str]] = None,
                                           output_format: Optional[str] = None) -> str:
        """
        Create a POML template programmatically from components.
        
        This is useful for migration from existing templates or creating
        new POML templates programmatically.
        
        Args:
            role: Role description
            task: Task description  
            examples: List of examples with 'input' and 'output' keys
            hints: List of hint strings
            output_format: Output format description
            
        Returns:
            POML template string
        """
        template_parts = []
        
        if role:
            template_parts.append(f"<role>{role}</role>")
        
        if task:
            template_parts.append(f"<task>{task}</task>")
        
        if examples:
            for example in examples:
                template_parts.append("<example>")
                if 'input' in example:
                    template_parts.append(f"  <input>{example['input']}</input>")
                if 'output' in example:
                    template_parts.append(f"  <output>{example['output']}</output>")
                template_parts.append("</example>")
        
        if hints:
            for hint in hints:
                template_parts.append(f"<hint>{hint}</hint>")
        
        if output_format:
            template_parts.append(f"<output-format>{output_format}</output-format>")
        
        return "\n".join(template_parts)