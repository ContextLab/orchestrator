"""Template metadata for tracking dependencies and rendering requirements."""

from dataclasses import dataclass
from typing import Set, Optional


@dataclass
class TemplateMetadata:
    """
    Metadata about a template string to track dependencies and rendering requirements.
    
    This class is used to analyze templates during compilation and determine when
    they should be rendered during execution.
    """
    
    # The original template string
    original_template: str
    
    # Set of step IDs this template depends on (e.g., "search_topic", "analyze_data")
    dependencies: Set[str]
    
    # Special context requirements (e.g., "$item", "$index" for loops)
    context_requirements: Set[str]
    
    # True if this template references step results and must be rendered at runtime
    is_runtime_only: bool
    
    # True if this template only uses input parameters and can be rendered at compile time
    is_compile_time: bool
    
    # The parameter path this template belongs to (e.g., ["parameters", "url"])
    parameter_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate metadata consistency."""
        # Runtime and compile time are mutually exclusive
        if self.is_runtime_only and self.is_compile_time:
            raise ValueError("Template cannot be both runtime-only and compile-time")
        
        # If there are dependencies or context requirements, it must be runtime
        if (self.dependencies or self.context_requirements) and self.is_compile_time:
            raise ValueError(
                "Template with dependencies or context requirements must be runtime-only"
            )
    
    def can_render_with_context(self, available_steps: Set[str], 
                                available_contexts: Set[str]) -> bool:
        """
        Check if this template can be rendered given available context.
        
        Args:
            available_steps: Set of step IDs whose results are available
            available_contexts: Set of special contexts available (e.g., "$item")
            
        Returns:
            True if all dependencies are satisfied
        """
        # Check step dependencies
        if not self.dependencies.issubset(available_steps):
            return False
        
        # Check context requirements
        if not self.context_requirements.issubset(available_contexts):
            return False
        
        return True
    
    def get_missing_dependencies(self, available_steps: Set[str]) -> Set[str]:
        """Get the set of missing step dependencies."""
        return self.dependencies - available_steps
    
    def get_missing_contexts(self, available_contexts: Set[str]) -> Set[str]:
        """Get the set of missing context requirements."""
        return self.context_requirements - available_contexts