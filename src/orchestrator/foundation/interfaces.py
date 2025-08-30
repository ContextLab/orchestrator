"""
Core interfaces for the refactored orchestrator architecture.

This module defines the abstract interfaces that all components must implement,
providing clear contracts for the foundational architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from .pipeline_spec import PipelineSpecification
from .result import PipelineResult, StepResult


class PipelineCompilerInterface(ABC):
    """
    Interface for pipeline compilation from YAML to executable format.
    
    The pipeline compiler is responsible for:
    - Parsing YAML pipeline definitions
    - Validating pipeline structure and dependencies
    - Compiling to LangGraph StateGraph representation
    - Resolving model and tool dependencies
    """
    
    @abstractmethod
    async def compile(self, yaml_content: str, context: Optional[Dict[str, Any]] = None) -> PipelineSpecification:
        """
        Compile YAML content to pipeline specification.
        
        Args:
            yaml_content: Raw YAML pipeline definition
            context: Optional compilation context
            
        Returns:
            Compiled pipeline specification
            
        Raises:
            CompilationError: If compilation fails
        """
        pass
    
    @abstractmethod
    async def validate(self, spec: PipelineSpecification) -> List[str]:
        """
        Validate compiled pipeline specification.
        
        Args:
            spec: Pipeline specification to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        pass


class ExecutionEngineInterface(ABC):
    """
    Interface for pipeline execution using LangGraph StateGraphs.
    
    The execution engine is responsible for:
    - Running compiled pipeline specifications
    - Managing execution state and variable flow
    - Handling step orchestration and dependencies
    - Providing progress monitoring and error recovery
    """
    
    @abstractmethod
    async def execute(self, spec: PipelineSpecification, inputs: Dict[str, Any]) -> PipelineResult:
        """
        Execute a compiled pipeline specification.
        
        Args:
            spec: Compiled pipeline specification
            inputs: Input parameters for execution
            
        Returns:
            Pipeline execution result
            
        Raises:
            ExecutionError: If execution fails
        """
        pass
    
    @abstractmethod
    async def execute_step(self, step_id: str, context: Dict[str, Any]) -> StepResult:
        """
        Execute a single pipeline step.
        
        Args:
            step_id: Identifier of step to execute
            context: Execution context and variables
            
        Returns:
            Step execution result
        """
        pass
    
    @abstractmethod
    def get_execution_progress(self) -> Dict[str, Any]:
        """
        Get current execution progress information.
        
        Returns:
            Progress information including completed steps, current step, etc.
        """
        pass


class ModelManagerInterface(ABC):
    """
    Interface for multi-provider model management.
    
    The model manager is responsible for:
    - Abstracting different model providers (OpenAI, Anthropic, etc.)
    - Intelligent model selection based on requirements
    - Managing model lifecycles and resource allocation
    - Providing unified interfaces for model interaction
    """
    
    @abstractmethod
    async def select_model(self, requirements: Dict[str, Any]) -> str:
        """
        Select best model for given requirements.
        
        Args:
            requirements: Model requirements (capabilities, cost, performance, etc.)
            
        Returns:
            Selected model identifier
            
        Raises:
            ModelError: If no suitable model found
        """
        pass
    
    @abstractmethod
    async def invoke_model(self, model_id: str, prompt: str, **kwargs) -> str:
        """
        Invoke a model with given prompt.
        
        Args:
            model_id: Model identifier
            prompt: Input prompt
            **kwargs: Additional model parameters
            
        Returns:
            Model response
        """
        pass
    
    @abstractmethod
    def list_available_models(self) -> List[str]:
        """
        List all available models.
        
        Returns:
            List of available model identifiers
        """
        pass
    
    @abstractmethod
    async def get_model_capabilities(self, model_id: str) -> Dict[str, Any]:
        """
        Get capabilities for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model capabilities information
        """
        pass


class ToolRegistryInterface(ABC):
    """
    Interface for tool management and registration.
    
    The tool registry is responsible for:
    - Managing available tools and their capabilities
    - Automatic tool discovery and initialization
    - Providing unified interfaces for tool execution
    - Handling tool dependencies and setup
    """
    
    @abstractmethod
    async def register_tool(self, tool_name: str, tool_config: Dict[str, Any]) -> None:
        """
        Register a tool with the registry.
        
        Args:
            tool_name: Name of the tool
            tool_config: Tool configuration and capabilities
        """
        pass
    
    @abstractmethod
    async def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get tool instance by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        pass
    
    @abstractmethod
    def list_available_tools(self) -> List[str]:
        """
        List all available tools.
        
        Returns:
            List of available tool names
        """
        pass
    
    @abstractmethod
    async def ensure_tool_available(self, tool_name: str) -> bool:
        """
        Ensure a tool is available and ready for use.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if tool is available and ready
        """
        pass


class QualityControlInterface(ABC):
    """
    Interface for automated quality control and assessment.
    
    The quality control system is responsible for:
    - Automated output quality assessment
    - Generating quality control reports
    - Providing improvement recommendations
    - Monitoring execution quality metrics
    """
    
    @abstractmethod
    async def assess_output(self, output: Any, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess quality of pipeline or step output.
        
        Args:
            output: Output to assess
            criteria: Assessment criteria
            
        Returns:
            Quality assessment results
        """
        pass
    
    @abstractmethod
    async def generate_qc_report(self, result: PipelineResult) -> Dict[str, Any]:
        """
        Generate comprehensive quality control report.
        
        Args:
            result: Pipeline execution result
            
        Returns:
            Quality control report
        """
        pass
    
    @abstractmethod
    async def get_improvement_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """
        Get recommendations for improving output quality.
        
        Args:
            assessment: Quality assessment results
            
        Returns:
            List of improvement recommendations
        """
        pass


@dataclass
class FoundationConfig:
    """Configuration for foundation components."""
    
    # Model Management
    default_model: Optional[str] = None
    model_selection_strategy: str = "balanced"  # "cost", "performance", "balanced"
    
    # Execution
    max_concurrent_steps: int = 5
    execution_timeout: int = 3600  # seconds
    
    # Tool Registry
    auto_install_tools: bool = True
    tool_timeout: int = 300  # seconds
    
    # Quality Control
    enable_quality_checks: bool = True
    quality_threshold: float = 0.7
    
    # LangGraph Integration
    enable_persistence: bool = False
    storage_backend: str = "memory"  # "memory", "sqlite", "postgres"
    database_url: Optional[str] = None
    
    # Progress Monitoring
    show_progress_bars: bool = True
    log_level: str = "INFO"


# Type aliases for common patterns
PipelineDict = Dict[str, Any]
ModelSpec = Dict[str, Any]
ToolSpec = Dict[str, Any]
ExecutionContext = Dict[str, Any]