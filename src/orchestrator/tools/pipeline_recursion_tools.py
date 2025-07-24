"""Pipeline recursion tools for executing sub-pipelines and managing recursion."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from .base import Tool


@dataclass
class RecursionContext:
    """Context for tracking recursive pipeline execution."""

    depth: int = 0
    max_depth: int = 10
    call_stack: List[str] = field(default_factory=list)
    execution_count: Dict[str, int] = field(default_factory=dict)
    max_executions_per_pipeline: int = 100
    shared_state: Dict[str, Any] = field(default_factory=dict)
    parent_pipeline_id: Optional[str] = None
    root_pipeline_id: Optional[str] = None


@dataclass
class PipelineExecutionResult:
    """Result of pipeline execution."""

    success: bool
    pipeline_id: str
    outputs: Dict[str, Any]
    error: Optional[str] = None
    execution_time: float = 0.0
    steps_executed: int = 0
    recursion_depth: int = 0


class PipelineExecutorTool(Tool):
    """Execute sub-pipelines with context passing and dependency management."""

    def __init__(self):
        super().__init__(
            name="pipeline-executor",
            description="Execute a sub-pipeline as part of a larger workflow",
        )
        self.add_parameter(
            "pipeline", "string", "Pipeline definition (YAML path, ID, or inline YAML)"
        )
        self.add_parameter(
            "inputs", "object", "Input parameters for the pipeline", default={}
        )
        self.add_parameter(
            "inherit_context",
            "boolean",
            "Inherit context from parent pipeline",
            default=True,
        )
        self.add_parameter(
            "wait_for_completion",
            "boolean",
            "Wait for pipeline to complete",
            default=True,
        )
        self.add_parameter(
            "timeout",
            "number",
            "Execution timeout in seconds (0 for no timeout)",
            default=0,
        )
        self.add_parameter(
            "output_mapping",
            "object",
            "Map pipeline outputs to parent context",
            default={},
        )
        self.add_parameter(
            "error_handling",
            "string",
            "Error handling: fail, continue, retry",
            default="fail",
        )
        self.add_parameter(
            "retry_count", "integer", "Number of retries on failure", default=3
        )
        self.add_parameter(
            "retry_delay", "number", "Delay between retries in seconds", default=1.0
        )

        self.logger = logging.getLogger(__name__)

        # Import orchestrator here to avoid circular imports
        self._orchestrator = None
        self._recursion_context: Optional[RecursionContext] = None

    def _get_orchestrator(self):
        """Lazy load orchestrator to avoid circular imports."""
        if self._orchestrator is None:
            from ..orchestrator import Orchestrator
            from ..models.registry_singleton import get_model_registry
            from ..control_systems.hybrid_control_system import HybridControlSystem

            # Get the global model registry
            model_registry = get_model_registry()

            # Create control system if we have models
            control_system = None
            if model_registry and model_registry.models:
                control_system = HybridControlSystem(model_registry)

            self._orchestrator = Orchestrator(
                model_registry=model_registry, control_system=control_system
            )
        return self._orchestrator

    def _resolve_pipeline(self, pipeline_spec: str) -> Dict[str, Any]:
        """Resolve pipeline from various input formats."""
        # Check if it's a file path
        if pipeline_spec.endswith((".yaml", ".yml")):
            path = Path(pipeline_spec)
            if not path.is_absolute():
                # Try relative to current directory
                if path.exists():
                    with open(path, "r") as f:
                        return yaml.safe_load(f)

                # Try relative to examples directory
                examples_path = Path("examples") / path
                if examples_path.exists():
                    with open(examples_path, "r") as f:
                        return yaml.safe_load(f)

                # Try relative to pipelines directory
                pipelines_path = Path("pipelines") / path
                if pipelines_path.exists():
                    with open(pipelines_path, "r") as f:
                        return yaml.safe_load(f)
            else:
                with open(path, "r") as f:
                    return yaml.safe_load(f)

        # Check if it's inline YAML
        if "\n" in pipeline_spec or pipeline_spec.strip().startswith("id:"):
            return yaml.safe_load(pipeline_spec)

        # Otherwise treat as pipeline ID - look in standard locations
        for directory in ["examples", "pipelines", "."]:
            for ext in [".yaml", ".yml"]:
                path = Path(directory) / f"{pipeline_spec}{ext}"
                if path.exists():
                    with open(path, "r") as f:
                        return yaml.safe_load(f)

        raise ValueError(f"Could not resolve pipeline: {pipeline_spec}")

    def _check_recursion_limits(
        self, pipeline_id: str, context: RecursionContext
    ) -> None:
        """Check if recursion limits are exceeded."""
        # Check depth
        if context.depth >= context.max_depth:
            raise RecursionError(
                f"Maximum recursion depth ({context.max_depth}) exceeded. "
                f"Call stack: {' -> '.join(context.call_stack)}"
            )

        # Check execution count for this pipeline
        exec_count = context.execution_count.get(pipeline_id, 0)
        if exec_count >= context.max_executions_per_pipeline:
            raise RecursionError(
                f"Pipeline '{pipeline_id}' exceeded maximum executions "
                f"({context.max_executions_per_pipeline})"
            )

        # Check for direct recursion (same pipeline calling itself)
        if (
            pipeline_id in context.call_stack[-3:]
        ):  # Check last 3 to allow some patterns
            self.logger.warning(
                f"Potential infinite recursion detected: '{pipeline_id}' "
                f"appears multiple times in recent call stack"
            )

    def _merge_contexts(
        self,
        parent_context: Dict[str, Any],
        child_inputs: Dict[str, Any],
        inherit: bool,
    ) -> Dict[str, Any]:
        """Merge parent context with child inputs."""
        if not inherit:
            return child_inputs

        # Start with parent context
        merged = parent_context.copy()

        # Override with child inputs
        merged.update(child_inputs)

        return merged

    def _map_outputs(
        self, pipeline_outputs: Dict[str, Any], output_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Map pipeline outputs according to mapping rules."""
        if not output_mapping:
            return pipeline_outputs

        mapped = {}
        for source_key, target_key in output_mapping.items():
            if source_key in pipeline_outputs:
                mapped[target_key] = pipeline_outputs[source_key]
            else:
                self.logger.warning(
                    f"Output key '{source_key}' not found in pipeline outputs"
                )

        return mapped

    async def _execute_with_retry(
        self,
        pipeline_def: Dict[str, Any],
        inputs: Dict[str, Any],
        retry_count: int,
        retry_delay: float,
    ) -> PipelineExecutionResult:
        """Execute pipeline with retry logic."""
        last_error = None
        pipeline_id = pipeline_def.get("id", "unknown")

        for attempt in range(retry_count):
            try:
                self.logger.info(
                    f"Executing pipeline '{pipeline_id}' (attempt {attempt + 1}/{retry_count})"
                )

                # Get orchestrator and execute
                orchestrator = self._get_orchestrator()

                # Execute the pipeline
                start_time = time.time()
                result = await orchestrator.execute_pipeline_from_dict(
                    pipeline_def, inputs=inputs, context=self._recursion_context
                )
                execution_time = time.time() - start_time

                # Success!
                return PipelineExecutionResult(
                    success=True,
                    pipeline_id=pipeline_id,
                    outputs=result.get("outputs", {}),
                    execution_time=execution_time,
                    steps_executed=result.get("steps_executed", 0),
                    recursion_depth=(
                        self._recursion_context.depth if self._recursion_context else 0
                    ),
                )

            except Exception as e:
                last_error = str(e)
                self.logger.error(
                    f"Pipeline execution failed (attempt {attempt + 1}): {e}"
                )

                if attempt < retry_count - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

        # All retries failed
        return PipelineExecutionResult(
            success=False,
            pipeline_id=pipeline_id,
            outputs={},
            error=last_error,
            recursion_depth=(
                self._recursion_context.depth if self._recursion_context else 0
            ),
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute sub-pipeline."""
        pipeline_spec = kwargs["pipeline"]
        inputs = kwargs.get("inputs", {})
        inherit_context = kwargs.get("inherit_context", True)
        wait_for_completion = kwargs.get("wait_for_completion", True)
        timeout = kwargs.get("timeout", 0)
        output_mapping = kwargs.get("output_mapping", {})
        error_handling = kwargs.get("error_handling", "fail")
        retry_count = kwargs.get("retry_count", 3)
        retry_delay = kwargs.get("retry_delay", 1.0)

        # Validate error handling
        if error_handling not in ["fail", "continue", "retry"]:
            return {
                "success": False,
                "error": f"Invalid error_handling: {error_handling}",
            }

        try:
            # Resolve pipeline definition
            pipeline_def = self._resolve_pipeline(pipeline_spec)
            pipeline_id = pipeline_def.get("id", "unknown")

            # Get or create recursion context
            if self._recursion_context is None:
                self._recursion_context = RecursionContext(root_pipeline_id=pipeline_id)

            # Check recursion limits
            self._check_recursion_limits(pipeline_id, self._recursion_context)

            # Update recursion context
            self._recursion_context.depth += 1
            self._recursion_context.call_stack.append(pipeline_id)
            self._recursion_context.execution_count[pipeline_id] = (
                self._recursion_context.execution_count.get(pipeline_id, 0) + 1
            )

            # Merge contexts if inheriting
            if inherit_context and self._recursion_context.shared_state:
                inputs = self._merge_contexts(
                    self._recursion_context.shared_state, inputs, inherit_context
                )

            # Execute with timeout if specified
            if timeout > 0 and wait_for_completion:
                try:
                    result = await asyncio.wait_for(
                        self._execute_with_retry(
                            pipeline_def, inputs, retry_count, retry_delay
                        ),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    return {
                        "success": False,
                        "error": f"Pipeline execution timed out after {timeout} seconds",
                        "pipeline_id": pipeline_id,
                        "recursion_depth": self._recursion_context.depth,
                    }
            else:
                result = await self._execute_with_retry(
                    pipeline_def, inputs, retry_count, retry_delay
                )

            # Handle execution result
            if result.success:
                # Map outputs
                mapped_outputs = self._map_outputs(result.outputs, output_mapping)

                # Update shared state
                self._recursion_context.shared_state.update(mapped_outputs)

                return {
                    "success": True,
                    "pipeline_id": result.pipeline_id,
                    "outputs": mapped_outputs,
                    "execution_time": result.execution_time,
                    "steps_executed": result.steps_executed,
                    "recursion_depth": result.recursion_depth,
                    "call_stack": self._recursion_context.call_stack.copy(),
                }
            else:
                # Handle failure based on error_handling
                if error_handling == "fail":
                    raise RuntimeError(f"Sub-pipeline failed: {result.error}")
                elif error_handling == "continue":
                    self.logger.warning(
                        f"Sub-pipeline failed, continuing: {result.error}"
                    )
                    return {
                        "success": False,
                        "pipeline_id": result.pipeline_id,
                        "error": result.error,
                        "continued": True,
                        "recursion_depth": result.recursion_depth,
                    }
                # retry is handled by _execute_with_retry

        except Exception as e:
            self.logger.error(f"Pipeline executor error: {e}")
            return {
                "success": False,
                "error": str(e),
                "recursion_depth": (
                    self._recursion_context.depth if self._recursion_context else 0
                ),
            }

        finally:
            # Clean up recursion context
            if self._recursion_context:
                self._recursion_context.depth -= 1
                if self._recursion_context.call_stack:
                    self._recursion_context.call_stack.pop()


class RecursionControlTool(Tool):
    """Manage recursive execution with termination conditions and state tracking."""

    def __init__(self):
        super().__init__(
            name="recursion-control",
            description="Control recursive pipeline execution with conditions and limits",
        )
        self.add_parameter(
            "action",
            "string",
            "Action: check_condition, update_state, get_state, reset",
        )
        self.add_parameter(
            "condition", "string", "Termination condition expression", required=False
        )
        self.add_parameter("state_key", "string", "Key for state value", required=False)
        self.add_parameter(
            "state_value", "any", "Value to set for state", required=False
        )
        self.add_parameter(
            "increment", "number", "Increment state value by amount", required=False
        )
        self.add_parameter(
            "max_iterations", "integer", "Maximum iterations allowed", default=1000
        )
        self.add_parameter(
            "depth_limit", "integer", "Maximum recursion depth", default=10
        )
        self.add_parameter("time_limit", "number", "Time limit in seconds", default=0)

        self.logger = logging.getLogger(__name__)

        # Recursion state storage
        self._recursion_states: Dict[str, RecursionContext] = {}
        self._start_times: Dict[str, float] = {}

    def _get_or_create_context(self, context_id: str) -> RecursionContext:
        """Get or create recursion context."""
        if context_id not in self._recursion_states:
            self._recursion_states[context_id] = RecursionContext()
            self._start_times[context_id] = time.time()
        return self._recursion_states[context_id]

    def _evaluate_condition(self, condition: str, context: RecursionContext) -> bool:
        """Evaluate termination condition."""
        # Create evaluation namespace
        namespace = {
            "state": context.shared_state,
            "depth": context.depth,
            "iterations": sum(context.execution_count.values()),
            "executions": context.execution_count,
            "call_stack": context.call_stack,
            "call_stack_size": len(context.call_stack),
        }

        # Add helper functions
        namespace.update(
            {
                "len": len,
                "sum": sum,
                "max": max,
                "min": min,
                "all": all,
                "any": any,
                "abs": abs,
            }
        )

        try:
            # Evaluate condition
            result = eval(condition, {"__builtins__": {}}, namespace)
            return bool(result)
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    def _check_limits(
        self,
        context: RecursionContext,
        context_id: str,
        max_iterations: int,
        depth_limit: int,
        time_limit: float,
    ) -> Dict[str, Any]:
        """Check if any limits are exceeded."""
        total_iterations = sum(context.execution_count.values())

        # Check iteration limit
        if total_iterations >= max_iterations:
            return {
                "limit_exceeded": True,
                "reason": "max_iterations",
                "value": total_iterations,
                "limit": max_iterations,
            }

        # Check depth limit
        if context.depth >= depth_limit:
            return {
                "limit_exceeded": True,
                "reason": "depth_limit",
                "value": context.depth,
                "limit": depth_limit,
            }

        # Check time limit
        if time_limit > 0:
            elapsed = time.time() - self._start_times.get(context_id, time.time())
            if elapsed >= time_limit:
                return {
                    "limit_exceeded": True,
                    "reason": "time_limit",
                    "value": elapsed,
                    "limit": time_limit,
                }

        return {"limit_exceeded": False}

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute recursion control action."""
        action = kwargs["action"]

        # Get context ID (use root pipeline ID or generate one)
        context_id = kwargs.get("context_id", "default")

        # Validate action
        valid_actions = ["check_condition", "update_state", "get_state", "reset"]
        if action not in valid_actions:
            return {
                "success": False,
                "error": f"Invalid action: {action}. Must be one of {valid_actions}",
            }

        try:
            if action == "check_condition":
                condition = kwargs.get("condition")
                if not condition:
                    return {
                        "success": False,
                        "error": "Condition required for check_condition action",
                    }

                context = self._get_or_create_context(context_id)

                # Check limits first
                limit_check = self._check_limits(
                    context,
                    context_id,
                    kwargs.get("max_iterations", 1000),
                    kwargs.get("depth_limit", 10),
                    kwargs.get("time_limit", 0),
                )

                if limit_check["limit_exceeded"]:
                    return {
                        "success": True,
                        "should_terminate": True,
                        "reason": f"{limit_check['reason']} exceeded",
                        "details": limit_check,
                        "state": context.shared_state.copy(),
                        "iterations": sum(context.execution_count.values()),
                        "depth": context.depth,
                    }

                # Evaluate condition
                should_terminate = self._evaluate_condition(condition, context)

                return {
                    "success": True,
                    "should_terminate": should_terminate,
                    "condition": condition,
                    "state": context.shared_state.copy(),
                    "iterations": sum(context.execution_count.values()),
                    "depth": context.depth,
                    "execution_counts": context.execution_count.copy(),
                }

            elif action == "update_state":
                state_key = kwargs.get("state_key")
                if not state_key:
                    return {
                        "success": False,
                        "error": "state_key required for update_state action",
                    }

                context = self._get_or_create_context(context_id)

                # Handle different update types
                if "state_value" in kwargs:
                    context.shared_state[state_key] = kwargs["state_value"]
                elif "increment" in kwargs:
                    current = context.shared_state.get(state_key, 0)
                    if isinstance(current, (int, float)):
                        context.shared_state[state_key] = current + kwargs["increment"]
                    else:
                        return {
                            "success": False,
                            "error": f"Cannot increment non-numeric value: {type(current)}",
                        }
                else:
                    return {
                        "success": False,
                        "error": "Either state_value or increment must be provided",
                    }

                return {
                    "success": True,
                    "state_key": state_key,
                    "new_value": context.shared_state[state_key],
                    "full_state": context.shared_state.copy(),
                }

            elif action == "get_state":
                context = self._get_or_create_context(context_id)
                state_key = kwargs.get("state_key")

                if state_key:
                    return {
                        "success": True,
                        "state_key": state_key,
                        "value": context.shared_state.get(state_key),
                        "exists": state_key in context.shared_state,
                    }
                else:
                    return {
                        "success": True,
                        "state": context.shared_state.copy(),
                        "iterations": sum(context.execution_count.values()),
                        "depth": context.depth,
                        "call_stack": context.call_stack.copy(),
                        "execution_counts": context.execution_count.copy(),
                    }

            elif action == "reset":
                if context_id in self._recursion_states:
                    del self._recursion_states[context_id]
                if context_id in self._start_times:
                    del self._start_times[context_id]

                return {
                    "success": True,
                    "message": f"Recursion context '{context_id}' reset",
                    "context_id": context_id,
                }

        except Exception as e:
            self.logger.error(f"Recursion control error: {e}")
            return {"success": False, "error": str(e), "action": action}

    def get_active_contexts(self) -> List[str]:
        """Get list of active recursion contexts."""
        return list(self._recursion_states.keys())

    def get_context_info(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific context."""
        if context_id not in self._recursion_states:
            return None

        context = self._recursion_states[context_id]
        elapsed = time.time() - self._start_times.get(context_id, time.time())

        return {
            "context_id": context_id,
            "depth": context.depth,
            "iterations": sum(context.execution_count.values()),
            "elapsed_time": elapsed,
            "call_stack": context.call_stack.copy(),
            "execution_counts": context.execution_count.copy(),
            "state_keys": list(context.shared_state.keys()),
        }
