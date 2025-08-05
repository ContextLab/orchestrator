"""
Centralized output tracking system for pipeline execution.
Manages output metadata and provides cross-task output references.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .output_metadata import OutputInfo, OutputMetadata, OutputReference


@dataclass
class OutputTracker:
    """
    Centralized tracker for pipeline outputs.
    
    Manages output metadata, provides cross-task references, and tracks
    output dependencies across the entire pipeline execution.
    """
    
    # Core tracking data
    outputs: Dict[str, OutputInfo] = field(default_factory=dict)  # task_id -> OutputInfo
    metadata: Dict[str, OutputMetadata] = field(default_factory=dict)  # task_id -> OutputMetadata
    references: Dict[str, List[OutputReference]] = field(default_factory=lambda: defaultdict(list))  # referencing_task -> [references]
    
    # State tracking
    pipeline_id: Optional[str] = None
    execution_id: Optional[str] = None
    base_output_dir: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    # Validation and consistency
    validation_errors: List[str] = field(default_factory=list)
    consistency_checks: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize tracker after creation."""
        if self.base_output_dir:
            # Ensure base output directory exists
            os.makedirs(self.base_output_dir, exist_ok=True)
    
    def register_task_metadata(self, task_id: str, output_metadata: OutputMetadata) -> None:
        """Register output metadata for a task."""
        if task_id in self.metadata:
            raise ValueError(f"Output metadata for task '{task_id}' already registered")
        
        self.metadata[task_id] = output_metadata
        
        # Validate metadata consistency
        issues = output_metadata.validate_consistency()
        if issues:
            self.validation_errors.extend([f"Task {task_id}: {issue}" for issue in issues])
    
    def register_output(self, task_id: str, result: Any, location: Optional[str] = None,
                       format: Optional[str] = None, **kwargs) -> OutputInfo:
        """Register actual output from a task execution."""
        # Get metadata if available
        metadata = self.metadata.get(task_id)
        
        # Create output info
        output_info = OutputInfo(
            task_id=task_id,
            output_type=metadata.produces if metadata else None,
            location=location,
            format=format,
            result=result,
            **kwargs
        )
        
        # Validate against metadata if available
        if metadata:
            is_valid = output_info.validate_against_metadata(metadata)
            if not is_valid:
                self.validation_errors.extend([
                    f"Task {task_id}: {error}" for error in output_info.validation_errors
                ])
        
        # Store output info
        self.outputs[task_id] = output_info
        self.consistency_checks[task_id] = len(output_info.validation_errors) == 0
        
        return output_info
    
    def add_reference(self, referencing_task: str, reference: OutputReference) -> None:
        """Add an output reference from one task to another."""
        self.references[referencing_task].append(reference)
        
        # Validate that referenced task exists or will exist
        if reference.task_id not in self.metadata and reference.task_id not in self.outputs:
            self.validation_errors.append(
                f"Task '{referencing_task}' references non-existent task '{reference.task_id}'"
            )
    
    def resolve_reference(self, reference: OutputReference) -> Any:
        """Resolve an output reference to its actual value."""
        if reference.task_id not in self.outputs:
            return reference.default_value
        
        output_info = self.outputs[reference.task_id]
        output_info.mark_accessed()  # Track access
        
        return reference.resolve(output_info)
    
    def get_output(self, task_id: str, field: Optional[str] = None) -> Any:
        """Get output from a specific task."""
        if task_id not in self.outputs:
            raise KeyError(f"No output found for task '{task_id}'")
        
        output_info = self.outputs[task_id]
        output_info.mark_accessed()
        
        if field is None:
            return output_info.result
        
        # Handle nested field access
        current = output_info
        for part in field.split('.'):
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise AttributeError(f"Field '{field}' not found in output for task '{task_id}'")
        
        return current
    
    def has_output(self, task_id: str) -> bool:
        """Check if task has registered output."""
        return task_id in self.outputs
    
    def get_dependencies(self, task_id: str) -> Set[str]:
        """Get set of task IDs that this task depends on for outputs."""
        dependencies = set()
        if task_id in self.references:
            for ref in self.references[task_id]:
                dependencies.add(ref.task_id)
        return dependencies
    
    def get_dependents(self, task_id: str) -> Set[str]:
        """Get set of task IDs that depend on this task's output."""
        dependents = set()
        for referring_task, refs in self.references.items():
            for ref in refs:
                if ref.task_id == task_id:
                    dependents.add(referring_task)
        return dependents
    
    def get_output_graph(self) -> Dict[str, Dict[str, Any]]:
        """Get complete output dependency graph."""
        graph = {}
        
        for task_id in set(self.metadata.keys()) | set(self.outputs.keys()):
            graph[task_id] = {
                'produces': self.metadata.get(task_id),
                'output': self.outputs.get(task_id),
                'dependencies': list(self.get_dependencies(task_id)),
                'dependents': list(self.get_dependents(task_id)),
                'has_output': self.has_output(task_id),
                'validation_status': self.consistency_checks.get(task_id, None)
            }
        
        return graph
    
    def validate_all_references(self) -> List[str]:
        """Validate all output references in the tracker."""
        issues = []
        
        for referencing_task, refs in self.references.items():
            for ref in refs:
                # Check if referenced task exists
                if ref.task_id not in self.metadata and ref.task_id not in self.outputs:
                    issues.append(
                        f"Task '{referencing_task}' references undefined task '{ref.task_id}'"
                    )
                    continue
                
                # Check if output is available when needed
                if ref.task_id in self.outputs:
                    try:
                        self.resolve_reference(ref)
                    except Exception as e:
                        issues.append(
                            f"Failed to resolve reference from '{referencing_task}' to '{ref.task_id}.{ref.field}': {e}"
                        )
        
        return issues
    
    def get_file_outputs(self) -> Dict[str, str]:
        """Get mapping of task IDs to their file output locations."""
        file_outputs = {}
        
        for task_id, output_info in self.outputs.items():
            if output_info.location and os.path.exists(output_info.location):
                file_outputs[task_id] = output_info.location
        
        return file_outputs
    
    def get_outputs_by_type(self, output_type: str) -> List[OutputInfo]:
        """Get all outputs of a specific type."""
        matching_outputs = []
        
        for output_info in self.outputs.values():
            if output_info.output_type == output_type:
                matching_outputs.append(output_info)
        
        return matching_outputs
    
    def cleanup_outputs(self, task_ids: Optional[List[str]] = None) -> None:
        """Clean up output files for specified tasks (or all tasks)."""
        target_tasks = task_ids if task_ids else list(self.outputs.keys())
        
        for task_id in target_tasks:
            if task_id in self.outputs:
                output_info = self.outputs[task_id]
                if output_info.location and os.path.exists(output_info.location):
                    try:
                        os.remove(output_info.location)
                    except OSError as e:
                        self.validation_errors.append(
                            f"Failed to clean up output file for task '{task_id}': {e}"
                        )
    
    def export_summary(self) -> Dict[str, Any]:
        """Export comprehensive summary of all tracked outputs."""
        summary = {
            'pipeline_id': self.pipeline_id,
            'execution_id': self.execution_id,
            'created_at': self.created_at.isoformat(),
            'total_tasks': len(set(self.metadata.keys()) | set(self.outputs.keys())),
            'tasks_with_metadata': len(self.metadata),
            'tasks_with_outputs': len(self.outputs),
            'total_references': sum(len(refs) for refs in self.references.values()),
            'validation_errors': len(self.validation_errors),
            'file_outputs': len(self.get_file_outputs()),
            'output_types': {},
            'dependency_graph': self.get_output_graph(),
            'errors': self.validation_errors
        }
        
        # Count outputs by type
        type_counts = defaultdict(int)
        for output_info in self.outputs.values():
            if output_info.output_type:
                type_counts[output_info.output_type] += 1
        summary['output_types'] = dict(type_counts)
        
        return summary
    
    def get_template_variables(self) -> Dict[str, Any]:
        """Get template variables for all tracked outputs."""
        variables = {}
        
        for task_id, output_info in self.outputs.items():
            variables[task_id] = {
                'result': output_info.result,
                'location': output_info.location,
                'format': output_info.format,
                'output_type': output_info.output_type,
                'size': output_info.file_size,
                'created_at': output_info.created_at.isoformat() if output_info.created_at else None
            }
        
        return variables
    
    def resolve_template_string(self, template: str) -> str:
        """Resolve template string with tracked output values."""
        from jinja2 import Environment, StrictUndefined
        
        env = Environment(undefined=StrictUndefined)
        template_obj = env.from_string(template)
        
        try:
            return template_obj.render(**self.get_template_variables())
        except Exception as e:
            raise ValueError(f"Failed to resolve template '{template}': {e}") from e
    
    def __len__(self) -> int:
        """Return number of tracked outputs."""
        return len(self.outputs)
    
    def __contains__(self, task_id: str) -> bool:
        """Check if task has tracked output."""
        return task_id in self.outputs
    
    def __getitem__(self, task_id: str) -> OutputInfo:
        """Get output info by task ID."""
        if task_id not in self.outputs:
            raise KeyError(f"No output for task '{task_id}'")
        return self.outputs[task_id]