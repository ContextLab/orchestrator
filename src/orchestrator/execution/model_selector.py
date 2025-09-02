"""
Unified model selection module for pipeline execution.

This module provides runtime model selection capabilities during pipeline execution,
integrating selection algorithms from Stream A and expert assignments from Stream B
into a cohesive system that can intelligently choose optimal models for each step.
"""

from __future__ import annotations

import logging
import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from ..models.model_selector import ModelSelector, ModelSelectionCriteria
from ..models.registry import ModelRegistry
from ..core.model import Model
from ..foundation._compatibility import PipelineStep, PipelineSpecification

logger = logging.getLogger(__name__)


@dataclass
class RuntimeModelContext:
    """
    Context information for runtime model selection.
    
    Contains all the information available at pipeline execution time
    that can influence model selection decisions.
    """
    step: PipelineStep
    pipeline_spec: PipelineSpecification
    execution_state: Dict[str, Any]
    available_variables: Dict[str, Any] = field(default_factory=dict)
    
    # Expert assignments from Stream B 
    expert_assignments: Dict[str, str] = field(default_factory=dict)
    
    # Performance requirements derived from execution context
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Budget constraints for this execution
    cost_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Previous step results for adaptive selection
    step_history: Dict[str, Any] = field(default_factory=dict)


class ExecutionModelSelector:
    """
    Runtime model selector for pipeline execution.
    
    This class coordinates model selection during pipeline execution by:
    1. Analyzing step requirements and context
    2. Applying expert assignments where specified
    3. Using intelligent selection algorithms for optimal performance
    4. Adapting selections based on execution history and constraints
    """
    
    def __init__(
        self, 
        model_registry: ModelRegistry,
        enable_adaptive_selection: bool = True,
        enable_expert_assignments: bool = True,
        enable_cost_optimization: bool = True
    ):
        """
        Initialize the execution model selector.
        
        Args:
            model_registry: Model registry for available models
            enable_adaptive_selection: Enable adaptive selection based on execution history
            enable_expert_assignments: Enable expert tool-model assignments
            enable_cost_optimization: Enable cost-aware model selection
        """
        self.model_registry = model_registry
        self.model_selector = ModelSelector(model_registry)
        
        self.enable_adaptive_selection = enable_adaptive_selection
        self.enable_expert_assignments = enable_expert_assignments
        self.enable_cost_optimization = enable_cost_optimization
        
        # Track execution metrics for adaptive selection
        self._execution_history: Dict[str, List[Dict[str, Any]]] = {}
        self._performance_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"ExecutionModelSelector initialized with features: "
                   f"adaptive={enable_adaptive_selection}, "
                   f"experts={enable_expert_assignments}, "
                   f"cost_opt={enable_cost_optimization}")
    
    async def select_model_for_step(
        self,
        context: RuntimeModelContext,
        selection_strategy: Optional[str] = None
    ) -> Model:
        """
        Select the optimal model for a pipeline step during runtime execution.
        
        Args:
            context: Runtime execution context with step and pipeline information
            selection_strategy: Optional override for selection strategy
            
        Returns:
            Selected model for the step
            
        Raises:
            RuntimeError: If no suitable model can be found
        """
        step = context.step
        logger.info(f"Selecting model for step: {step.id} ({step.name})")
        
        try:
            # 1. Check for explicit model specification
            if step.model and step.model != "AUTO":
                explicit_model = await self._resolve_explicit_model(step.model, context)
                if explicit_model:
                    logger.info(f"Using explicit model for step {step.id}: {explicit_model.provider}:{explicit_model.name}")
                    return explicit_model
            
            # 2. Check for expert assignments
            if self.enable_expert_assignments:
                expert_model = await self._check_expert_assignments(step, context)
                if expert_model:
                    logger.info(f"Using expert-assigned model for step {step.id}: {expert_model.provider}:{expert_model.name}")
                    return expert_model
            
            # 3. Build selection criteria from step requirements
            criteria = await self._build_selection_criteria(step, context, selection_strategy)
            
            # 4. Apply adaptive selection if enabled
            if self.enable_adaptive_selection:
                criteria = await self._apply_adaptive_selection(step, criteria, context)
            
            # 5. Apply cost optimization if enabled
            if self.enable_cost_optimization:
                criteria = await self._apply_cost_optimization(criteria, context)
            
            # 6. Select model using intelligent selection algorithm
            auto_description = self._extract_auto_description(step)
            selected_model = await self.model_selector.select_model(criteria, auto_description)
            
            # 7. Record selection for future adaptive decisions
            await self._record_selection(step, selected_model, criteria, context)
            
            logger.info(f"Selected model for step {step.id}: {selected_model.provider}:{selected_model.name} "
                       f"(strategy: {criteria.selection_strategy})")
            
            return selected_model
            
        except Exception as e:
            logger.error(f"Failed to select model for step {step.id}: {e}")
            raise RuntimeError(f"Model selection failed for step {step.id}: {e}") from e
    
    async def evaluate_selection_quality(
        self,
        step_id: str,
        selected_model: Model,
        execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a model selection based on execution results.
        
        This is used for adaptive selection to improve future choices.
        
        Args:
            step_id: Step identifier
            selected_model: Model that was selected
            execution_result: Results from step execution
            
        Returns:
            Dictionary containing selection quality metrics
        """
        try:
            quality_metrics = {
                "step_id": step_id,
                "model": f"{selected_model.provider}:{selected_model.name}",
                "timestamp": execution_result.get("timestamp"),
                "success": execution_result.get("status") == "success",
                "execution_time": execution_result.get("execution_time", 0),
                "error_count": len(execution_result.get("errors", [])),
                "output_quality": self._assess_output_quality(execution_result),
                "cost_efficiency": self._calculate_cost_efficiency(selected_model, execution_result),
                "performance_score": self._calculate_performance_score(execution_result)
            }
            
            # Store for adaptive selection
            if self.enable_adaptive_selection:
                self._record_performance_metrics(step_id, selected_model, quality_metrics)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate selection quality for step {step_id}: {e}")
            return {"error": str(e)}
    
    async def get_selection_recommendations(
        self,
        pipeline_spec: PipelineSpecification,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get model selection recommendations for an entire pipeline.
        
        This provides upfront analysis of optimal model choices before execution.
        
        Args:
            pipeline_spec: Pipeline specification
            execution_context: Execution context variables
            
        Returns:
            Dictionary mapping step IDs to selection recommendations
        """
        recommendations = {}
        
        try:
            for step in pipeline_spec.steps:
                context = RuntimeModelContext(
                    step=step,
                    pipeline_spec=pipeline_spec,
                    execution_state={},
                    available_variables=execution_context
                )
                
                # Get top 3 model recommendations
                criteria = await self._build_selection_criteria(step, context)
                top_models = await self._get_top_model_candidates(criteria, limit=3)
                
                step_recommendations = []
                for model, score in top_models:
                    recommendation = {
                        "model": f"{model.provider}:{model.name}",
                        "score": score,
                        "rationale": self._explain_selection(model, criteria),
                        "estimated_cost": self._estimate_step_cost(model, step),
                        "estimated_performance": self._estimate_step_performance(model, step)
                    }
                    step_recommendations.append(recommendation)
                
                recommendations[step.id] = {
                    "step_name": step.name,
                    "recommendations": step_recommendations,
                    "selection_criteria": criteria.__dict__
                }
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate selection recommendations: {e}")
            return {"error": str(e)}
    
    # Private implementation methods
    
    async def _resolve_explicit_model(
        self,
        model_spec: str,
        context: RuntimeModelContext
    ) -> Optional[Model]:
        """Resolve an explicitly specified model."""
        try:
            # Handle provider:model format
            if ":" in model_spec:
                provider, model_name = model_spec.split(":", 1)
                return await self.model_registry.get_model(provider, model_name)
            else:
                # Try to find model by name across providers
                return await self.model_registry.find_model_by_name(model_spec)
        except Exception as e:
            logger.warning(f"Failed to resolve explicit model '{model_spec}': {e}")
            return None
    
    async def _check_expert_assignments(
        self,
        step: PipelineStep,
        context: RuntimeModelContext
    ) -> Optional[Model]:
        """Check for expert tool-model assignments."""
        if not context.expert_assignments:
            return None
        
        # Check if any of the step's tools have expert assignments
        for tool_name in step.tools:
            if tool_name in context.expert_assignments:
                model_spec = context.expert_assignments[tool_name]
                expert_model = await self._resolve_explicit_model(model_spec, context)
                if expert_model:
                    return expert_model
        
        return None
    
    async def _build_selection_criteria(
        self,
        step: PipelineStep,
        context: RuntimeModelContext,
        strategy_override: Optional[str] = None
    ) -> ModelSelectionCriteria:
        """Build selection criteria from step requirements and context."""
        criteria = ModelSelectionCriteria()
        
        # Set selection strategy
        if strategy_override:
            criteria.selection_strategy = strategy_override
        elif hasattr(context.pipeline_spec, 'selection_schema'):
            criteria.selection_strategy = getattr(context.pipeline_spec.selection_schema, 'strategy', 'balanced')
        
        # Analyze step requirements
        criteria.required_tasks = self._infer_required_tasks(step)
        criteria.required_capabilities = self._infer_required_capabilities(step)
        
        # Set context requirements
        if step.context_limit:
            criteria.min_context_window = step.context_limit
        
        # Apply performance requirements from context
        if context.performance_requirements:
            if "max_latency_ms" in context.performance_requirements:
                criteria.max_latency_ms = context.performance_requirements["max_latency_ms"]
            if "min_accuracy" in context.performance_requirements:
                criteria.min_accuracy_score = context.performance_requirements["min_accuracy"]
        
        # Apply cost constraints from context
        if context.cost_constraints:
            if "max_cost_per_request" in context.cost_constraints:
                criteria.max_cost_per_request = context.cost_constraints["max_cost_per_request"]
            if "prefer_free" in context.cost_constraints:
                criteria.prefer_free_models = context.cost_constraints["prefer_free"]
        
        # Consider pipeline-level requirements
        if hasattr(context.pipeline_spec, 'requires_model'):
            pipeline_requirements = context.pipeline_spec.requires_model
            # Parse pipeline-level model requirements using existing logic
            pipeline_criteria = self.model_selector.parse_requirements_from_yaml(pipeline_requirements)
            
            # Merge pipeline criteria with step criteria
            criteria = self._merge_criteria(criteria, pipeline_criteria)
        
        return criteria
    
    async def _apply_adaptive_selection(
        self,
        step: PipelineStep,
        criteria: ModelSelectionCriteria,
        context: RuntimeModelContext
    ) -> ModelSelectionCriteria:
        """Apply adaptive selection based on execution history."""
        step_type = self._classify_step_type(step)
        
        if step_type in self._performance_cache:
            performance_data = self._performance_cache[step_type]
            
            # Adjust criteria based on historical performance
            if performance_data.get("high_accuracy_models"):
                criteria.min_accuracy_score = max(criteria.min_accuracy_score, 0.85)
            
            if performance_data.get("fast_models"):
                criteria.speed_preference = "fast"
            
            if performance_data.get("cost_effective_models"):
                criteria.selection_strategy = "cost_optimized"
        
        return criteria
    
    async def _apply_cost_optimization(
        self,
        criteria: ModelSelectionCriteria,
        context: RuntimeModelContext
    ) -> ModelSelectionCriteria:
        """Apply cost optimization to selection criteria."""
        # Check if we're approaching budget limits
        if context.cost_constraints.get("budget_utilization", 0) > 0.8:
            criteria.selection_strategy = "cost_optimized"
            criteria.prefer_free_models = True
        
        # Apply pipeline-level cost constraints
        if hasattr(context.pipeline_spec, 'selection_schema'):
            schema = context.pipeline_spec.selection_schema
            if hasattr(schema, 'cost_limit'):
                criteria.cost_limit = schema.cost_limit
            if hasattr(schema, 'budget_period'):
                criteria.budget_period = schema.budget_period
        
        return criteria
    
    def _extract_auto_description(self, step: PipelineStep) -> Optional[str]:
        """Extract AUTO description from step model specification."""
        if step.model and step.model.startswith("AUTO"):
            if ":" in step.model:
                return step.model.split(":", 1)[1]
            else:
                # Use step description or prompt as AUTO description
                return step.description or step.prompt
        return None
    
    async def _record_selection(
        self,
        step: PipelineStep,
        selected_model: Model,
        criteria: ModelSelectionCriteria,
        context: RuntimeModelContext
    ) -> None:
        """Record model selection for future adaptive decisions."""
        step_type = self._classify_step_type(step)
        
        if step_type not in self._execution_history:
            self._execution_history[step_type] = []
        
        selection_record = {
            "step_id": step.id,
            "step_type": step_type,
            "selected_model": f"{selected_model.provider}:{selected_model.name}",
            "criteria": criteria.__dict__,
            "context_hash": hash(str(sorted(context.available_variables.items()))),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        self._execution_history[step_type].append(selection_record)
        
        # Limit history size
        if len(self._execution_history[step_type]) > 100:
            self._execution_history[step_type] = self._execution_history[step_type][-50:]
    
    def _record_performance_metrics(
        self,
        step_id: str,
        model: Model,
        metrics: Dict[str, Any]
    ) -> None:
        """Record performance metrics for adaptive selection."""
        model_key = f"{model.provider}:{model.name}"
        
        if model_key not in self._performance_cache:
            self._performance_cache[model_key] = {
                "success_rate": [],
                "execution_times": [],
                "cost_efficiency": [],
                "performance_scores": []
            }
        
        cache = self._performance_cache[model_key]
        
        if metrics.get("success"):
            cache["success_rate"].append(1)
        else:
            cache["success_rate"].append(0)
        
        cache["execution_times"].append(metrics.get("execution_time", 0))
        cache["cost_efficiency"].append(metrics.get("cost_efficiency", 0))
        cache["performance_scores"].append(metrics.get("performance_score", 0))
        
        # Limit cache size
        for metric_list in cache.values():
            if len(metric_list) > 50:
                metric_list[:] = metric_list[-25:]
    
    async def _get_top_model_candidates(
        self,
        criteria: ModelSelectionCriteria,
        limit: int = 3
    ) -> List[tuple[Model, float]]:
        """Get top model candidates with scores."""
        try:
            # Get requirements dict for registry filtering
            requirements = criteria.to_requirements_dict()
            
            # Get eligible models from registry
            eligible_models = await self.model_registry._filter_by_capabilities(requirements)
            
            if not eligible_models:
                # Try fallback
                eligible_models = await self.model_selector._get_fallback_models(criteria)
            
            # Score models
            scored_models = self.model_selector._score_models(eligible_models, criteria)
            
            # Return top candidates
            return scored_models[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get top model candidates: {e}")
            return []
    
    # Utility methods
    
    def _infer_required_tasks(self, step: PipelineStep) -> List[str]:
        """Infer required tasks from step configuration."""
        tasks = []
        
        # Check step type or name for task hints
        step_name_lower = step.name.lower()
        if any(word in step_name_lower for word in ["analyze", "analysis"]):
            tasks.append("analyze")
        if any(word in step_name_lower for word in ["generate", "create"]):
            tasks.append("generate")
        if any(word in step_name_lower for word in ["code", "program", "script"]):
            tasks.append("code")
        if any(word in step_name_lower for word in ["reason", "think", "logic"]):
            tasks.append("reasoning")
        
        # Check tools for task hints
        for tool in step.tools:
            if "code" in tool.lower():
                tasks.append("code")
            elif "vision" in tool.lower() or "image" in tool.lower():
                tasks.append("vision")
        
        return list(set(tasks)) if tasks else ["general"]
    
    def _infer_required_capabilities(self, step: PipelineStep) -> List[str]:
        """Infer required capabilities from step configuration."""
        capabilities = []
        
        # Check tools for capability hints
        for tool in step.tools:
            tool_lower = tool.lower()
            if any(word in tool_lower for word in ["vision", "image", "visual"]):
                capabilities.append("vision")
            elif any(word in tool_lower for word in ["code", "program", "script"]):
                capabilities.append("code")
            elif any(word in tool_lower for word in ["function", "tool", "call"]):
                capabilities.append("tools")
        
        # Check step description
        if step.description:
            desc_lower = step.description.lower()
            if any(word in desc_lower for word in ["image", "visual", "picture"]):
                capabilities.append("vision")
            if any(word in desc_lower for word in ["code", "program", "script"]):
                capabilities.append("code")
            if any(word in desc_lower for word in ["json", "structured"]):
                capabilities.append("structured_output")
        
        return capabilities
    
    def _classify_step_type(self, step: PipelineStep) -> str:
        """Classify step type for adaptive selection."""
        # Simple classification based on step characteristics
        if any(tool for tool in step.tools if "code" in tool.lower()):
            return "code_generation"
        elif any(tool for tool in step.tools if "vision" in tool.lower()):
            return "vision_analysis"
        elif "analyze" in step.name.lower():
            return "analysis"
        elif "generate" in step.name.lower():
            return "generation"
        else:
            return "general"
    
    def _merge_criteria(
        self,
        step_criteria: ModelSelectionCriteria,
        pipeline_criteria: ModelSelectionCriteria
    ) -> ModelSelectionCriteria:
        """Merge step-level and pipeline-level criteria."""
        # Step criteria take precedence, pipeline provides defaults
        merged = ModelSelectionCriteria()
        
        # Use step values or fall back to pipeline values
        merged.selection_strategy = step_criteria.selection_strategy or pipeline_criteria.selection_strategy
        merged.required_tasks = step_criteria.required_tasks or pipeline_criteria.required_tasks
        merged.required_capabilities = list(set(step_criteria.required_capabilities + pipeline_criteria.required_capabilities))
        
        merged.min_context_window = max(step_criteria.min_context_window, pipeline_criteria.min_context_window)
        merged.max_cost_per_request = step_criteria.max_cost_per_request or pipeline_criteria.max_cost_per_request
        merged.prefer_free_models = step_criteria.prefer_free_models or pipeline_criteria.prefer_free_models
        
        merged.max_latency_ms = step_criteria.max_latency_ms or pipeline_criteria.max_latency_ms
        merged.min_accuracy_score = max(step_criteria.min_accuracy_score, pipeline_criteria.min_accuracy_score)
        
        return merged
    
    def _assess_output_quality(self, execution_result: Dict[str, Any]) -> float:
        """Assess output quality from execution results."""
        # Simple quality assessment - can be enhanced
        if execution_result.get("status") == "success":
            output = execution_result.get("output", {})
            if output:
                # Basic heuristics for output quality
                if isinstance(output, dict) and len(output) > 0:
                    return 0.8
                elif isinstance(output, str) and len(output) > 10:
                    return 0.7
            return 0.5
        return 0.0
    
    def _calculate_cost_efficiency(
        self,
        model: Model,
        execution_result: Dict[str, Any]
    ) -> float:
        """Calculate cost efficiency score."""
        execution_time = execution_result.get("execution_time", 1)
        success = execution_result.get("status") == "success"
        
        if not success:
            return 0.0
        
        # Calculate cost per successful execution
        if model.cost.is_free:
            return 1.0
        
        avg_cost = (model.cost.input_cost_per_1k_tokens + model.cost.output_cost_per_1k_tokens) / 2
        
        # Higher efficiency for lower cost and faster execution
        efficiency = 1.0 / (avg_cost * execution_time + 1)
        return min(1.0, efficiency)
    
    def _calculate_performance_score(self, execution_result: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        success_score = 1.0 if execution_result.get("status") == "success" else 0.0
        speed_score = max(0.0, 1.0 - execution_result.get("execution_time", 0) / 60.0)  # Normalize to 1 minute
        quality_score = self._assess_output_quality(execution_result)
        
        return (success_score * 0.5 + speed_score * 0.3 + quality_score * 0.2)
    
    def _explain_selection(self, model: Model, criteria: ModelSelectionCriteria) -> str:
        """Generate explanation for model selection."""
        reasons = []
        
        if criteria.selection_strategy == "cost_optimized":
            reasons.append("optimized for cost efficiency")
        elif criteria.selection_strategy == "performance_optimized":
            reasons.append("optimized for performance")
        elif criteria.selection_strategy == "accuracy_optimized":
            reasons.append("optimized for accuracy")
        else:
            reasons.append("balanced selection")
        
        if model.cost.is_free:
            reasons.append("no cost")
        
        if model.capabilities.vision_capable and "vision" in criteria.required_capabilities:
            reasons.append("vision capabilities")
        
        if model.capabilities.code_specialized and "code" in criteria.required_capabilities:
            reasons.append("code specialization")
        
        return "; ".join(reasons) if reasons else "general purpose model"
    
    def _estimate_step_cost(self, model: Model, step: PipelineStep) -> float:
        """Estimate cost for executing a step with the model."""
        if model.cost.is_free:
            return 0.0
        
        # Rough estimation based on step complexity
        estimated_tokens = 1000  # Default estimation
        
        # Adjust based on step characteristics
        if step.prompt and len(step.prompt) > 500:
            estimated_tokens *= 2
        
        if len(step.tools) > 3:
            estimated_tokens *= 1.5
        
        return model.cost.calculate_cost(estimated_tokens // 2, estimated_tokens // 2)
    
    def _estimate_step_performance(self, model: Model, step: PipelineStep) -> Dict[str, Any]:
        """Estimate performance metrics for executing a step with the model."""
        return {
            "accuracy": model.capabilities.accuracy_score,
            "speed": model.capabilities.speed_rating,
            "success_probability": 0.9 if model.metrics.success_rate > 0.8 else 0.7
        }