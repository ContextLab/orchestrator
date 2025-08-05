"""Lazy runtime AUTO tag resolution system with multi-pass LLM orchestration."""

from .models import (
    AutoTagContext,
    AutoTagResolution,
    AutoTagConfig,
    PassTimeouts,
    ResolutionError,
    AutoTagResolutionError,
    AutoTagNestingError,
    ParseError,
    ValidationError,
    RequirementsAnalysis,
    PromptConstruction,
    ActionPlan,
)
from .resolver import LazyAutoTagResolver
from .requirements_analyzer import RequirementsAnalyzer
from .prompt_constructor import PromptConstructor
from .resolution_executor import ResolutionExecutor
from .action_determiner import ActionDeterminer
from .nested_handler import NestedAutoTagHandler
from .resolution_logger import ResolutionLogger

__all__ = [
    # Models
    "AutoTagContext",
    "AutoTagResolution",
    "AutoTagConfig",
    "PassTimeouts",
    "ResolutionError",
    "AutoTagResolutionError",
    "AutoTagNestingError",
    "ParseError",
    "ValidationError",
    "RequirementsAnalysis",
    "PromptConstruction",
    "ActionPlan",
    # Core components
    "LazyAutoTagResolver",
    "RequirementsAnalyzer",
    "PromptConstructor",
    "ResolutionExecutor",
    "ActionDeterminer",
    "NestedAutoTagHandler",
    "ResolutionLogger",
]