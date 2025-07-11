"""
Orchestrator: AI pipeline orchestration framework with intelligent ambiguity resolution.

This package provides a unified interface for executing AI pipelines defined in YAML
with automatic ambiguity resolution using LLMs.
"""

from .core.task import Task, TaskStatus
from .core.pipeline import Pipeline
from .core.model import Model, ModelCapabilities, ModelRequirements
from .core.control_system import ControlSystem
from .compiler.yaml_compiler import YAMLCompiler
from .models.model_registry import ModelRegistry
from .state.state_manager import StateManager
from .orchestrator import Orchestrator

__version__ = "0.1.0"
__author__ = "Contextual Dynamics Lab"
__email__ = "contextualdynamics@gmail.com"

__all__ = [
    "Task",
    "TaskStatus", 
    "Pipeline",
    "Model",
    "ModelCapabilities",
    "ModelRequirements",
    "ControlSystem",
    "YAMLCompiler",
    "ModelRegistry",
    "StateManager",
    "Orchestrator",
]