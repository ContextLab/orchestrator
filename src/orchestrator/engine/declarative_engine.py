"""Main declarative pipeline engine that executes YAML pipelines with zero custom code."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from ..models.model_registry import ModelRegistry
from ..tools.base import default_registry
from .advanced_executor import AdvancedTaskExecutor
from .pipeline_spec import PipelineSpec, TaskSpec

logger = logging.getLogger(__name__)


class DeclarativePipelineEngine:
    """Executes YAML pipelines with complete automation - no custom code required."""

    def __init__(self, model_registry: Optional[ModelRegistry] = None, tool_registry=None):
        self.model_registry = model_registry
        self.tool_registry = tool_registry or default_registry
        self.task_executor = AdvancedTaskExecutor(model_registry, tool_registry)
        self.execution_history = []

    async def execute_pipeline(self, yaml_content: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete pipeline from YAML content with provided inputs."""
        logger.info("Starting declarative pipeline execution")

        try:
            # 1. Parse YAML into pipeline specification
            pipeline_spec = self._parse_yaml_to_spec(yaml_content)
            logger.info(f"Parsed pipeline: {pipeline_spec.name}")

            # 2. Validate inputs
            pipeline_spec.validate_inputs(inputs)

            # 3. Initialize execution context
            context = self._initialize_context(pipeline_spec, inputs)

            # 4. Get execution order
            execution_order = pipeline_spec.get_execution_order()
            logger.info(f"Execution order: {[step.id for step in execution_order]}")

            # 5. Execute steps in order
            results = {}
            for step in execution_order:
                # Check if step should be executed (conditions)
                if not self._should_execute_step(step, context, results):
                    logger.info(f"Skipping step {step.id} due to condition")
                    continue

                # Update context with previous results
                step_context = self._build_step_context(context, results)

                # Execute step
                step_result = await self.task_executor.execute_task(step, step_context)
                results[step.id] = step_result

                logger.info(f"Step {step.id} completed")

            # 6. Extract pipeline outputs
            pipeline_outputs = self._extract_outputs(pipeline_spec, results, context)

            # 7. Build final result
            final_result = {
                "success": True,
                "pipeline": pipeline_spec.name,
                "execution_time": datetime.now().isoformat(),
                "steps_executed": list(results.keys()),
                "outputs": pipeline_outputs,
                "step_results": results,
            }

            self.execution_history.append(final_result)
            logger.info(f"Pipeline {pipeline_spec.name} completed successfully")

            return final_result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            error_result = {
                "success": False,
                "error": str(e),
                "pipeline": (
                    getattr(pipeline_spec, "name", "unknown")
                    if "pipeline_spec" in locals()
                    else "unknown"
                ),
                "execution_time": datetime.now().isoformat(),
            }
            self.execution_history.append(error_result)
            raise

    def _parse_yaml_to_spec(self, yaml_content: str) -> PipelineSpec:
        """Parse YAML content into a PipelineSpec object."""
        try:
            yaml_data = yaml.safe_load(yaml_content)
            return PipelineSpec(**yaml_data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to parse pipeline specification: {str(e)}")

    def _initialize_context(
        self, pipeline_spec: PipelineSpec, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialize execution context with pipeline configuration and inputs."""
        context = {
            "inputs": inputs,
            "pipeline": {
                "name": pipeline_spec.name,
                "version": pipeline_spec.version,
                "description": pipeline_spec.description,
            },
            "config": pipeline_spec.config,
            "execution": {"start_time": datetime.now().isoformat(), "engine_version": "1.0.0"},
        }

        # Add input values directly to context for easy template access
        context.update(inputs)

        return context

    def _should_execute_step(
        self, step: TaskSpec, context: Dict[str, Any], results: Dict[str, Any]
    ) -> bool:
        """Check if a step should be executed based on its condition."""
        if not step.condition:
            return True

        try:
            # Build evaluation context
            eval_context = context.copy()
            eval_context["results"] = results

            # Simple condition evaluation (can be enhanced with proper expression parser)
            condition = step.condition

            # Replace template variables in condition
            import re

            def replace_var(match):
                var_path = match.group(1).strip()
                parts = var_path.split(".")
                value = eval_context

                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return "False"  # If variable not found, condition is false

                return str(value)

            resolved_condition = re.sub(r"\{\{([^}]+)\}\}", replace_var, condition)

            # Evaluate simple boolean conditions
            # TODO: Implement proper expression parser for complex conditions
            if "==" in resolved_condition:
                left, right = resolved_condition.split("==", 1)
                return left.strip().strip('"') == right.strip().strip('"')
            elif "!=" in resolved_condition:
                left, right = resolved_condition.split("!=", 1)
                return left.strip().strip('"') != right.strip().strip('"')
            elif resolved_condition.lower() in ["true", "false"]:
                return resolved_condition.lower() == "true"
            else:
                # Default to true if condition can't be evaluated
                logger.warning(f"Could not evaluate condition: {condition}")
                return True

        except Exception as e:
            logger.warning(f"Error evaluating condition '{step.condition}': {str(e)}")
            return True  # Default to executing step if condition evaluation fails

    def _build_step_context(
        self, base_context: Dict[str, Any], results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build context for step execution including previous results."""
        step_context = base_context.copy()
        step_context["results"] = results

        # Add results from individual steps for easy access
        for step_id, result in results.items():
            if isinstance(result, dict):
                # Extract common result fields
                if "result" in result:
                    step_context[f"{step_id}_result"] = result["result"]
                if "content" in result:
                    step_context[f"{step_id}_content"] = result["content"]
                if "data" in result:
                    step_context[f"{step_id}_data"] = result["data"]

        return step_context

    def _extract_outputs(
        self, pipeline_spec: PipelineSpec, results: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract final outputs from pipeline execution results."""
        outputs = {}

        for output_name, output_spec in pipeline_spec.outputs.items():
            if isinstance(output_spec, str):
                # Simple template reference like "{{step.result}}"
                import re

                def replace_var(match):
                    var_path = match.group(1).strip()
                    parts = var_path.split(".")

                    # Look in results first
                    if parts[0] in results:
                        value = results[parts[0]]
                        for part in parts[1:]:
                            if isinstance(value, dict) and part in value:
                                value = value[part]
                            else:
                                return ""
                        return str(value)

                    # Look in context
                    value = context
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            return ""
                    return str(value)

                resolved_output = re.sub(r"\{\{([^}]+)\}\}", replace_var, output_spec)
                outputs[output_name] = resolved_output

            else:
                # Complex output specification (future enhancement)
                outputs[output_name] = output_spec

        return outputs

    async def execute_from_file(
        self, yaml_file_path: str, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute pipeline from a YAML file."""
        try:
            with open(yaml_file_path, "r", encoding="utf-8") as f:
                yaml_content = f.read()
            return await self.execute_pipeline(yaml_content, inputs)
        except FileNotFoundError:
            raise ValueError(f"Pipeline file not found: {yaml_file_path}")
        except Exception as e:
            raise ValueError(f"Failed to read pipeline file: {str(e)}")

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get history of pipeline executions."""
        return self.execution_history.copy()

    def clear_execution_history(self):
        """Clear execution history."""
        self.execution_history.clear()

    async def validate_pipeline(self, yaml_content: str) -> Dict[str, Any]:
        """Validate a pipeline without executing it."""
        try:
            pipeline_spec = self._parse_yaml_to_spec(yaml_content)

            # Check for AUTO tags that need resolution
            auto_steps = pipeline_spec.get_steps_with_auto_tags()

            # Check for required tools
            required_tools = pipeline_spec.get_required_tools()
            available_tools = self.tool_registry.list_tools()
            missing_tools = [tool for tool in required_tools if tool not in available_tools]

            validation_result = {
                "valid": True,
                "pipeline_name": pipeline_spec.name,
                "total_steps": len(pipeline_spec.steps),
                "auto_tag_steps": len(auto_steps),
                "required_tools": required_tools,
                "missing_tools": missing_tools,
                "execution_order": [step.id for step in pipeline_spec.get_execution_order()],
                "warnings": [],
            }

            # Add warnings
            if missing_tools:
                validation_result["warnings"].append(f"Missing tools: {missing_tools}")

            if auto_steps and not self.model_registry:
                validation_result["warnings"].append(
                    "AUTO tags found but no model registry available"
                )

            return validation_result

        except Exception as e:
            return {"valid": False, "error": str(e), "pipeline_name": "unknown"}
