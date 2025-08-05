"""
Template resolution utilities for output tracking.
Provides advanced template resolution capabilities for cross-task output references.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from .output_metadata import OutputReference
from .output_tracker import OutputTracker


class TemplateResolver:
    """
    Advanced template resolver for output tracking system.
    
    Handles complex template patterns, cross-task references, and validation
    of template dependencies.
    """
    
    def __init__(self, output_tracker: OutputTracker):
        """Initialize with output tracker."""
        self.output_tracker = output_tracker
        
        # Template patterns
        self.reference_pattern = re.compile(r'{{\s*([^}]+)\s*}}')
        self.field_pattern = re.compile(r'(\w+)\.(\w+(?:\.\w+)*)')
    
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
        """Resolve template with output values."""
        if default_values is None:
            default_values = {}
        
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
                    value = self.output_tracker.get_output(task_id, field)
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