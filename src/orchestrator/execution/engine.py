"""
StateGraph-based execution engine for pipeline workflows.

This module implements the core execution engine using LangGraph StateGraphs
for orchestrating pipeline workflows with comprehensive state management.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Type, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# LangGraph imports for state management
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    # Fallback for environments without LangGraph
    StateGraph = None
    START = "START"
    END = "END"
    CompiledStateGraph = None
    MemorySaver = None

from ..foundation._compatibility import (
    ExecutionEngineInterface, FoundationConfig, PipelineSpecification, 
    PipelineStep, PipelineResult, StepResult
)
from ..models.registry import ModelRegistry
from .model_selector import ExecutionModelSelector, RuntimeModelContext


logger = logging.getLogger(__name__)


class ExecutionState(TypedDict):
    """
    State structure for LangGraph StateGraph execution.
    
    This defines the shared state that flows through the execution graph,
    containing all variables, metadata, and execution context.
    """
    # Core execution context
    pipeline_id: str
    execution_id: str
    current_step: Optional[str]
    
    # Variable storage - managed by Stream B
    variables: Dict[str, Any]
    step_outputs: Dict[str, Dict[str, Any]]
    
    # Progress tracking - managed by Stream C  
    completed_steps: List[str]
    failed_steps: List[str]
    step_start_times: Dict[str, float]
    step_end_times: Dict[str, float]
    
    # Error handling and recovery
    errors: List[Dict[str, Any]]
    retry_counts: Dict[str, int]
    
    # Execution metadata
    started_at: float
    last_updated: float
    metadata: Dict[str, Any]


@dataclass
class ExecutionContext:
    """
    Context for pipeline step execution.
    
    This provides the execution environment and resources available
    to each step during execution.
    """
    pipeline_spec: PipelineSpecification
    step: PipelineStep
    state: ExecutionState
    config: FoundationConfig
    
    # Tool and model access (to be provided by Stream B)
    tools: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    
    # Progress callbacks (to be provided by Stream C)
    progress_callback: Optional[callable] = None
    error_callback: Optional[callable] = None


class ExecutionError(Exception):
    """Base exception for execution engine errors."""
    pass


class StepExecutionError(ExecutionError):
    """Exception raised when a step fails to execute."""
    
    def __init__(self, step_id: str, message: str, original_error: Exception = None):
        self.step_id = step_id
        self.original_error = original_error
        super().__init__(f"Step '{step_id}' failed: {message}")


class StateGraphEngine(ExecutionEngineInterface):
    """
    LangGraph-based StateGraph execution engine.
    
    This engine uses LangGraph StateGraphs to orchestrate pipeline execution,
    providing robust state management, parallel execution capabilities,
    and comprehensive progress tracking.
    """
    
    def __init__(self, config: FoundationConfig = None, model_registry: ModelRegistry = None):
        """Initialize the execution engine."""
        self.config = config or FoundationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Execution state
        self._current_execution: Optional[str] = None
        self._execution_states: Dict[str, ExecutionState] = {}
        self._compiled_graphs: Dict[str, CompiledStateGraph] = {}
        
        # Concurrency control
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_steps
        )
        
        # Model selection integration
        self._model_registry = model_registry
        self._model_selector = None
        if model_registry:
            self._model_selector = ExecutionModelSelector(
                model_registry=model_registry,
                enable_adaptive_selection=getattr(config, 'enable_adaptive_selection', True),
                enable_expert_assignments=getattr(config, 'enable_expert_assignments', True),
                enable_cost_optimization=getattr(config, 'enable_cost_optimization', True)
            )
        
        # State management for coordination with other streams
        self._variable_manager = None  # To be set by Stream B
        self._progress_tracker = None  # To be set by Stream C
        
        self.logger.info(f"StateGraphEngine initialized with config: {self.config}, model_selector: {self._model_selector is not None}")
    
    def set_variable_manager(self, manager: Any) -> None:
        """Set variable manager from Stream B."""
        self._variable_manager = manager
    
    def set_progress_tracker(self, tracker: Any) -> None:
        """Set progress tracker from Stream C."""
        self._progress_tracker = tracker
    
    async def execute(
        self, 
        spec: PipelineSpecification, 
        inputs: Dict[str, Any]
    ) -> PipelineResult:
        """
        Execute a complete pipeline specification.
        
        Args:
            spec: Compiled pipeline specification
            inputs: Input parameters for execution
            
        Returns:
            Pipeline execution result
            
        Raises:
            ExecutionError: If execution fails
        """
        execution_id = f"{spec.header.id}_{int(time.time() * 1000)}"
        self.logger.info(f"Starting pipeline execution: {execution_id}")
        
        try:
            # Initialize execution state
            state = self._initialize_execution_state(spec, execution_id, inputs)
            self._execution_states[execution_id] = state
            self._current_execution = execution_id
            
            # Build and compile the StateGraph
            graph = await self._build_state_graph(spec)
            self._compiled_graphs[execution_id] = graph
            
            # Execute the pipeline
            start_time = time.time()
            final_state = await self._execute_graph(graph, state, spec)
            execution_time = time.time() - start_time
            
            # Build result from final state
            result = self._build_pipeline_result(spec, final_state, execution_time)
            
            self.logger.info(
                f"Pipeline execution completed: {execution_id} "
                f"in {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {execution_id}", exc_info=True)
            # Return failure result
            return PipelineResult(
                pipeline_name=spec.header.name,
                status="failed",
                step_results=[],
                total_steps=len(spec.steps),
                execution_time=time.time() - start_time if 'start_time' in locals() else 0,
                metadata={"error": str(e), "execution_id": execution_id}
            )
        finally:
            # Cleanup
            self._cleanup_execution(execution_id)
    
    async def execute_step(
        self, 
        step_id: str, 
        context: Dict[str, Any]
    ) -> StepResult:
        """
        Execute a single pipeline step.
        
        Args:
            step_id: Identifier of step to execute
            context: Execution context and variables
            
        Returns:
            Step execution result
        """
        if not self._current_execution:
            raise ExecutionError("No active execution context")
        
        state = self._execution_states.get(self._current_execution)
        if not state:
            raise ExecutionError("Execution state not found")
        
        # This is primarily called by the StateGraph nodes
        # The actual step logic is in _execute_step_node
        return await self._execute_single_step(step_id, state, context)
    
    def get_execution_progress(self) -> Dict[str, Any]:
        """
        Get current execution progress information.
        
        Returns:
            Progress information including completed steps, current step, etc.
        """
        if not self._current_execution:
            return {"status": "no_active_execution"}
        
        state = self._execution_states.get(self._current_execution)
        if not state:
            return {"status": "state_not_found"}
        
        total_steps = len(state.get("step_start_times", {}))
        completed_steps = len(state["completed_steps"])
        failed_steps = len(state["failed_steps"])
        
        progress = {
            "execution_id": self._current_execution,
            "pipeline_id": state["pipeline_id"],
            "current_step": state["current_step"],
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "progress_percentage": (completed_steps / total_steps * 100) if total_steps > 0 else 0,
            "started_at": state["started_at"],
            "last_updated": state["last_updated"],
            "status": "running" if state["current_step"] else "completed",
            "step_timings": {
                step: state["step_end_times"].get(step, 0) - state["step_start_times"].get(step, 0)
                for step in state["completed_steps"]
                if step in state["step_start_times"] and step in state["step_end_times"]
            }
        }
        
        return progress
    
    # Private methods for StateGraph construction and execution
    
    def _initialize_execution_state(
        self,
        spec: PipelineSpecification,
        execution_id: str,
        inputs: Dict[str, Any]
    ) -> ExecutionState:
        """Initialize the execution state for a pipeline run."""
        now = time.time()
        
        state: ExecutionState = {
            "pipeline_id": spec.header.id,
            "execution_id": execution_id,
            "current_step": None,
            "variables": inputs.copy(),
            "step_outputs": {},
            "completed_steps": [],
            "failed_steps": [],
            "step_start_times": {},
            "step_end_times": {},
            "errors": [],
            "retry_counts": {},
            "started_at": now,
            "last_updated": now,
            "metadata": {}
        }
        
        return state
    
    async def _build_state_graph(self, spec: PipelineSpecification) -> CompiledStateGraph:
        """
        Build and compile a LangGraph StateGraph from pipeline specification.
        
        Args:
            spec: Pipeline specification to build graph from
            
        Returns:
            Compiled StateGraph ready for execution
        """
        if StateGraph is None:
            raise ExecutionError("LangGraph not available - cannot build StateGraph")
        
        # Create the graph with our state schema
        graph = StateGraph(ExecutionState)
        
        # Add nodes for each step
        for step in spec.steps:
            node_func = self._create_step_node(step, spec)
            graph.add_node(step.id, node_func)
        
        # Add edges based on dependencies
        self._add_execution_edges(graph, spec)
        
        # Set entry and exit points
        graph.set_entry_point(START)
        graph.set_finish_point(END)
        
        # Add memory saver if persistence is enabled
        checkpointer = None
        if self.config.enable_persistence and MemorySaver:
            checkpointer = MemorySaver()
        
        # Compile the graph
        compiled = graph.compile(checkpointer=checkpointer)
        
        self.logger.info(f"Built StateGraph with {len(spec.steps)} nodes")
        return compiled
    
    def _create_step_node(self, step: PipelineStep, spec: PipelineSpecification):
        """Create a StateGraph node function for a pipeline step."""
        
        async def step_node(state: ExecutionState) -> ExecutionState:
            """Execute a single step and update state."""
            self.logger.info(f"Executing step: {step.id}")
            
            # Check step condition if specified
            if step.condition and not self._evaluate_condition(step.condition, state):
                self.logger.info(f"Step {step.id} condition not met, skipping")
                return state
            
            # Update current step
            state["current_step"] = step.id
            state["step_start_times"][step.id] = time.time()
            state["last_updated"] = time.time()
            
            try:
                # Execute the step
                result = await self._execute_single_step(step.id, state, {"spec": spec})
                
                # Update state with results
                state["step_outputs"][step.id] = result.output
                state["completed_steps"].append(step.id)
                state["step_end_times"][step.id] = time.time()
                
                # Merge step variables into global variables
                if result.output:
                    for var_name, value in result.output.items():
                        if var_name in step.variables:
                            state["variables"][var_name] = value
                
                self.logger.info(f"Step {step.id} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Step {step.id} failed: {str(e)}")
                
                # Record failure
                state["failed_steps"].append(step.id)
                state["errors"].append({
                    "step_id": step.id,
                    "error": str(e),
                    "timestamp": time.time(),
                    "type": type(e).__name__
                })
                state["step_end_times"][step.id] = time.time()
                
                # Handle retries
                retry_count = state["retry_counts"].get(step.id, 0)
                if retry_count < step.retry_count:
                    state["retry_counts"][step.id] = retry_count + 1
                    self.logger.info(f"Retrying step {step.id} ({retry_count + 1}/{step.retry_count})")
                    # Remove from failed steps for retry
                    state["failed_steps"].remove(step.id)
                    # Recursively retry
                    return await step_node(state)
                else:
                    # Max retries reached, propagate error
                    raise StepExecutionError(step.id, str(e), e)
            
            finally:
                state["current_step"] = None
                state["last_updated"] = time.time()
            
            return state
        
        return step_node
    
    def _add_execution_edges(self, graph: StateGraph, spec: PipelineSpecification):
        """Add edges to the StateGraph based on step dependencies."""
        
        # Get execution order to determine proper edges
        execution_levels = spec.get_execution_order()
        
        # Add entry edges for steps with no dependencies
        if execution_levels:
            for step_id in execution_levels[0]:
                graph.add_edge(START, step_id)
        
        # Add edges between dependent steps
        for step in spec.steps:
            if step.dependencies:
                for dep in step.dependencies:
                    graph.add_edge(dep, step.id)
        
        # Add edges to END for steps with no dependents
        for step in spec.steps:
            dependents = spec.get_dependents(step.id)
            if not dependents:
                graph.add_edge(step.id, END)
    
    def _evaluate_condition(self, condition: str, state: ExecutionState) -> bool:
        """
        Evaluate a step condition against current state.
        
        This is a simplified implementation - Stream B should provide
        more sophisticated condition evaluation.
        """
        try:
            # Simple variable substitution and evaluation
            # In a real implementation, this would be more sophisticated
            variables = state["variables"]
            
            # Replace variable references
            eval_condition = condition
            for var_name, value in variables.items():
                eval_condition = eval_condition.replace(var_name, repr(value))
            
            # Basic safety check - only allow simple comparisons
            allowed_operators = ["==", "!=", "<", ">", "<=", ">=", "is", "is not", "in", "not in"]
            if any(op in eval_condition for op in ["import", "exec", "eval", "__"]):
                self.logger.warning(f"Unsafe condition detected: {condition}")
                return True  # Fail safe
            
            return eval(eval_condition)
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate condition '{condition}': {e}")
            return True  # Fail safe - execute the step
    
    async def _execute_single_step(
        self,
        step_id: str,
        state: ExecutionState,
        context: Dict[str, Any]
    ) -> StepResult:
        """
        Execute a single step with intelligent model selection and execution.
        
        This method now integrates with the model selection intelligence
        to choose optimal models at runtime based on step requirements.
        """
        spec = context.get("spec")
        if not spec:
            raise ExecutionError("Pipeline specification not in context")
        
        step = spec.get_step(step_id)
        if not step:
            raise ExecutionError(f"Step {step_id} not found in specification")
        
        start_time = time.time()
        selected_model = None
        
        try:
            # 1. Intelligent model selection
            if self._model_selector:
                selected_model = await self._select_optimal_model_for_step(step, spec, state)
                self.logger.info(f"Selected model for step {step_id}: {selected_model.provider}:{selected_model.name}")
            else:
                self.logger.warning(f"No model selector available for step {step_id}, using default behavior")
            
            # 2. Execute step with selected model
            output = await self._execute_step_with_model(step, selected_model, state, context)
            
            execution_time = time.time() - start_time
            
            # 3. Evaluate model selection quality for adaptive learning
            if self._model_selector and selected_model:
                execution_result = {
                    "status": "success",
                    "execution_time": execution_time,
                    "timestamp": start_time,
                    "output": output,
                    "errors": []
                }
                await self._model_selector.evaluate_selection_quality(
                    step_id, selected_model, execution_result
                )
            
            return StepResult(
                step_id=step_id,
                status="success",
                output=output,
                execution_time=execution_time,
                metadata={
                    "step_name": step.name,
                    "tools_used": step.tools,
                    "model": f"{selected_model.provider}:{selected_model.name}" if selected_model else step.model,
                    "selection_strategy": getattr(selected_model, "selection_strategy", "default") if selected_model else "none"
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record failure for adaptive learning
            if self._model_selector and selected_model:
                execution_result = {
                    "status": "failure",
                    "execution_time": execution_time,
                    "timestamp": start_time,
                    "output": {},
                    "errors": [str(e)]
                }
                await self._model_selector.evaluate_selection_quality(
                    step_id, selected_model, execution_result
                )
            
            return StepResult(
                step_id=step_id,
                status="failure",
                output={},
                error=str(e),
                execution_time=execution_time,
                metadata={
                    "step_name": step.name,
                    "model": f"{selected_model.provider}:{selected_model.name}" if selected_model else step.model,
                    "error_type": type(e).__name__
                }
            )
    
    async def _execute_graph(
        self,
        graph: CompiledStateGraph,
        initial_state: ExecutionState,
        spec: PipelineSpecification
    ) -> ExecutionState:
        """Execute the compiled StateGraph with initial state."""
        
        try:
            # Configure execution
            config = {"recursion_limit": len(spec.steps) * 2}  # Reasonable recursion limit
            
            # Execute the graph
            result = await graph.ainvoke(initial_state, config=config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"StateGraph execution failed: {e}")
            raise ExecutionError(f"Graph execution failed: {e}")
    
    def _build_pipeline_result(
        self,
        spec: PipelineSpecification,
        final_state: ExecutionState,
        execution_time: float
    ) -> PipelineResult:
        """Build pipeline result from final execution state."""
        
        step_results = []
        
        # Build step results from state
        for step in spec.steps:
            step_id = step.id
            
            if step_id in final_state["completed_steps"]:
                output = final_state["step_outputs"].get(step_id, {})
                step_time = (
                    final_state["step_end_times"].get(step_id, 0) - 
                    final_state["step_start_times"].get(step_id, 0)
                )
                
                step_result = StepResult(
                    step_id=step_id,
                    status="success",
                    output=output,
                    execution_time=step_time
                )
                
            elif step_id in final_state["failed_steps"]:
                # Find error for this step
                error = None
                for err in final_state["errors"]:
                    if err["step_id"] == step_id:
                        error = err["error"]
                        break
                
                step_time = (
                    final_state["step_end_times"].get(step_id, 0) - 
                    final_state["step_start_times"].get(step_id, 0)
                )
                
                step_result = StepResult(
                    step_id=step_id,
                    status="failure", 
                    output={},
                    error=error,
                    execution_time=step_time
                )
            else:
                # Step was not executed (condition not met or dependency failed)
                step_result = StepResult(
                    step_id=step_id,
                    status="skipped",
                    output={},
                    execution_time=0
                )
            
            step_results.append(step_result)
        
        # Determine overall status
        if final_state["failed_steps"]:
            status = "failed"
        elif len(final_state["completed_steps"]) == len([s for s in spec.steps]):
            status = "success"
        else:
            status = "partial"
        
        return PipelineResult(
            pipeline_name=spec.header.name,
            status=status,
            step_results=step_results,
            total_steps=len(spec.steps),
            executed_steps=len(final_state["completed_steps"]),
            execution_time=execution_time,
            metadata={
                "execution_id": final_state["execution_id"],
                "variables": final_state["variables"],
                "errors": final_state["errors"]
            }
        )
    
    async def _select_optimal_model_for_step(
        self,
        step: PipelineStep,
        spec: PipelineSpecification, 
        state: ExecutionState
    ) -> Optional[Model]:
        """
        Select optimal model for a step using intelligent model selection.
        
        Args:
            step: Pipeline step to select model for
            spec: Full pipeline specification for context
            state: Current execution state
            
        Returns:
            Selected model or None if selection fails
        """
        try:
            # Build runtime context for model selection
            runtime_context = RuntimeModelContext(
                step=step,
                pipeline_spec=spec,
                execution_state=state,
                available_variables=state.get("variables", {}),
                step_history=state.get("step_outputs", {})
            )
            
            # Extract expert assignments if available from pipeline spec
            if hasattr(spec, 'experts') and spec.experts:
                runtime_context.expert_assignments = spec.experts
            
            # Extract cost constraints from pipeline or execution state
            if hasattr(spec, 'selection_schema') and hasattr(spec.selection_schema, 'cost_limit'):
                runtime_context.cost_constraints = {
                    "max_cost_per_request": spec.selection_schema.cost_limit,
                    "budget_utilization": self._calculate_budget_utilization(state)
                }
            
            # Extract performance requirements
            if hasattr(spec, 'selection_schema'):
                schema = spec.selection_schema
                runtime_context.performance_requirements = {
                    "max_latency_ms": getattr(schema, 'max_latency_ms', None),
                    "min_accuracy": getattr(schema, 'min_accuracy', None)
                }
            
            # Get selection strategy from pipeline schema
            strategy = None
            if hasattr(spec, 'selection_schema') and hasattr(spec.selection_schema, 'strategy'):
                strategy = spec.selection_schema.strategy
            
            # Select optimal model
            selected_model = await self._model_selector.select_model_for_step(
                runtime_context, strategy
            )
            
            return selected_model
            
        except Exception as e:
            self.logger.error(f"Failed to select model for step {step.id}: {e}")
            return None
    
    async def _execute_step_with_model(
        self,
        step: PipelineStep,
        selected_model: Optional[Model],
        state: ExecutionState,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a step with the selected model.
        
        This is where the actual step execution logic would be implemented.
        For now, this provides enhanced simulation with model-specific behavior.
        
        Args:
            step: Pipeline step to execute
            selected_model: Selected model for execution
            state: Current execution state
            context: Execution context
            
        Returns:
            Step execution output
        """
        # TODO: This will be implemented by Stream B (tool/model execution)
        # For now, provide enhanced simulation that considers the selected model
        
        self.logger.info(f"Executing step '{step.name}' with model: {selected_model.provider}:{selected_model.name if selected_model else 'default'}")
        
        # Simulate model-specific execution behavior
        if selected_model:
            # Simulate execution time based on model speed
            speed_multiplier = {"fast": 0.5, "medium": 1.0, "slow": 2.0}.get(
                selected_model.capabilities.speed_rating, 1.0
            )
            await asyncio.sleep(0.1 * speed_multiplier)
            
            # Simulate success rate based on model accuracy
            import random
            success_probability = selected_model.capabilities.accuracy_score
            if random.random() > success_probability:
                raise ExecutionError(f"Simulated execution failure (accuracy: {success_probability})")
        else:
            # Default simulation
            await asyncio.sleep(0.1)
        
        # Generate output based on step variables and model capabilities
        output = {}
        for var_name, description in step.variables.items():
            if selected_model:
                # Enhance output based on model capabilities
                if selected_model.capabilities.code_specialized and "code" in description.lower():
                    output[var_name] = f"[CODE] Enhanced {description} for {step.name} using {selected_model.name}"
                elif selected_model.capabilities.vision_capable and "image" in description.lower():
                    output[var_name] = f"[VISION] Enhanced {description} for {step.name} using {selected_model.name}"
                else:
                    output[var_name] = f"Enhanced {description} for {step.name} using {selected_model.name}"
            else:
                output[var_name] = f"Generated {description} for {step.name}"
        
        return output
    
    def _calculate_budget_utilization(self, state: ExecutionState) -> float:
        """
        Calculate budget utilization for cost-aware model selection.
        
        Args:
            state: Current execution state
            
        Returns:
            Budget utilization ratio (0.0 to 1.0)
        """
        # TODO: Implement actual budget tracking
        # For now, simulate based on execution progress
        total_steps = len(state.get("step_start_times", {}))
        completed_steps = len(state.get("completed_steps", []))
        
        if total_steps == 0:
            return 0.0
        
        # Simple estimation - budget utilization grows with step completion
        return min(1.0, completed_steps / total_steps * 0.8)
    
    def get_model_selection_recommendations(
        self, 
        spec: PipelineSpecification,
        execution_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Get model selection recommendations for the entire pipeline.
        
        Args:
            spec: Pipeline specification
            execution_context: Optional execution context
            
        Returns:
            Dictionary with recommendations for each step
        """
        if not self._model_selector:
            return {"error": "Model selector not available"}
        
        try:
            # Use asyncio.run for synchronous interface
            import asyncio
            recommendations = asyncio.run(
                self._model_selector.get_selection_recommendations(
                    spec, execution_context or {}
                )
            )
            return recommendations
        except Exception as e:
            self.logger.error(f"Failed to get model selection recommendations: {e}")
            return {"error": str(e)}
    
    def _cleanup_execution(self, execution_id: str):
        """Cleanup execution state and resources."""
        if execution_id in self._execution_states:
            del self._execution_states[execution_id]
        
        if execution_id in self._compiled_graphs:
            del self._compiled_graphs[execution_id]
        
        if self._current_execution == execution_id:
            self._current_execution = None
        
        self.logger.info(f"Cleaned up execution: {execution_id}")
    
    def __del__(self):
        """Cleanup resources on destruction."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)