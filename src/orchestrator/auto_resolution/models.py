"""Data models for AUTO tag resolution system."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime

from ..core.pipeline import Pipeline


class ResolutionError(Exception):
    """Base exception for AUTO tag resolution errors."""
    pass


class AutoTagResolutionError(ResolutionError):
    """Raised when AUTO tag resolution fails after all retries."""
    pass


class AutoTagNestingError(ResolutionError):
    """Raised when AUTO tag nesting exceeds maximum depth."""
    pass


class ParseError(ResolutionError):
    """Raised when structured output parsing fails."""
    pass


class ValidationError(ResolutionError):
    """Raised when resolved value fails validation."""
    pass


@dataclass
class PassTimeouts:
    """Timeout configuration for each resolution pass."""
    
    requirements_analysis: int = 1200  # 20 minutes
    prompt_construction: int = 1200    # 20 minutes
    resolution_execution: int = 1200   # 20 minutes
    action_determination: int = 1200   # 20 minutes


@dataclass
class AutoTagConfig:
    """Configuration for AUTO tag resolution."""
    
    # Model escalation order
    model_escalation: List[str] = field(default_factory=lambda: [
        "gpt-4o-mini",
        "gpt-4o", 
        "claude-sonnet-4-20250514"
    ])
    
    # Per-pass timeouts (seconds)
    pass_timeouts: PassTimeouts = field(default_factory=PassTimeouts)
    
    # Logging configuration
    log_resolution_details: bool = True
    checkpoint_resolutions: bool = True
    
    # Resolution limits
    max_nesting_depth: int = 5
    max_retries_per_model: int = 3


@dataclass
class AutoTagContext:
    """Complete context for AUTO tag resolution."""
    
    # Full pipeline definition
    pipeline: Pipeline
    current_task_id: str
    tag_location: str  # Path to the AUTO tag (e.g., "steps[2].action")
    
    # Runtime context
    variables: Dict[str, Any]  # All variables
    step_results: Dict[str, Any]  # All step results so far
    loop_context: Optional[Dict[str, Any]] = None  # Loop variables if in loop
    
    # Resolution metadata
    resolution_depth: int = 0  # For nested AUTO tags
    parent_resolutions: List['AutoTagResolution'] = field(default_factory=list)
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    
    def get_full_context(self) -> Dict[str, Any]:
        """Get complete context dictionary for resolution."""
        context = {
            "variables": self.variables,
            "step_results": self.step_results,
            "current_task": self.current_task_id,
            "pipeline_id": self.pipeline.id,
        }
        
        if self.loop_context:
            context["loop"] = self.loop_context
            
        return context


@dataclass
class RequirementsAnalysis:
    """Result of requirements analysis pass."""
    
    tools_needed: List[str] = field(default_factory=list)
    output_format: Optional[str] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    data_dependencies: List[str] = field(default_factory=list)
    model_requirements: Dict[str, Any] = field(default_factory=dict)
    expected_output_type: str = "string"
    
    # Metadata
    analysis_time_ms: int = 0
    model_used: str = ""


@dataclass
class PromptConstruction:
    """Result of prompt construction pass."""
    
    prompt: str
    target_model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    output_schema: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    
    # Context data used
    resolved_context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    construction_time_ms: int = 0
    model_used: str = ""


@dataclass
class ActionPlan:
    """Action plan for handling resolved value."""
    
    action_type: Literal[
        "return_value", 
        "call_tool", 
        "save_file", 
        "update_context", 
        "chain_resolution"
    ]
    
    # Action-specific parameters
    tool_name: Optional[str] = None
    tool_parameters: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    context_updates: Optional[Dict[str, Any]] = None
    next_auto_tag: Optional[str] = None
    
    # Metadata
    determination_time_ms: int = 0
    model_used: str = ""


@dataclass
class AutoTagResolution:
    """Complete result of AUTO tag resolution."""
    
    # Original AUTO tag
    original_tag: str
    tag_location: str
    
    # Pass 1: Requirements analysis
    requirements: RequirementsAnalysis
    
    # Pass 2: Prompt construction
    prompt_construction: PromptConstruction
    
    # Pass 3: Resolution
    resolved_value: Any
    
    # Post-resolution: Action determination
    action_plan: ActionPlan
    
    # Overall metadata
    resolution_time_ms: int = 0
    total_time_ms: int = 0
    retry_count: int = 0
    models_attempted: List[str] = field(default_factory=list)
    final_model_used: str = ""
    
    # Errors encountered
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_checkpoint_data(self) -> Dict[str, Any]:
        """Convert to data suitable for checkpointing."""
        return {
            "original_tag": self.original_tag,
            "tag_location": self.tag_location,
            "resolved_value": self.resolved_value,
            "total_time_ms": self.total_time_ms,
            "retry_count": self.retry_count,
            "models_attempted": self.models_attempted,
            "final_model_used": self.final_model_used,
            "action_type": self.action_plan.action_type,
            "timestamp": datetime.now().isoformat()
        }