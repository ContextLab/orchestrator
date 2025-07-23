"""Intelligent model selection engine with AUTO tag support."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..core.model import Model
from .model_registry import ModelRegistry, NoEligibleModelsError


@dataclass
class ModelSelectionCriteria:
    """Criteria for selecting a model."""

    # Task requirements
    required_tasks: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(
        default_factory=list
    )  # e.g., ["vision", "code", "tools"]
    required_domains: List[str] = field(default_factory=list)  # e.g., ["medical", "legal"]

    # Context requirements
    min_context_window: int = 0
    required_languages: List[str] = field(default_factory=list)

    # Cost constraints
    max_cost_per_request: Optional[float] = None
    max_cost_per_1k_tokens: Optional[float] = None
    prefer_free_models: bool = False

    # Performance requirements
    max_latency_ms: Optional[int] = None
    min_accuracy_score: float = 0.0
    speed_preference: Optional[str] = None  # "fast", "medium", "slow", None

    # Model preferences
    preferred_providers: List[str] = field(default_factory=list)
    excluded_providers: List[str] = field(default_factory=list)
    preferred_models: List[str] = field(default_factory=list)
    excluded_models: List[str] = field(default_factory=list)

    # Size constraints
    min_model_size: Optional[float] = None  # In billions of parameters
    max_model_size: Optional[float] = None

    # Strategy preferences
    selection_strategy: str = (
        "balanced"  # "balanced", "cost_optimized", "performance_optimized", "accuracy_optimized"
    )

    def to_requirements_dict(self) -> Dict[str, Any]:
        """Convert to requirements dictionary for compatibility."""
        requirements = {}

        if self.required_tasks:
            requirements["tasks"] = self.required_tasks
        if self.min_context_window > 0:
            requirements["context_window"] = self.min_context_window
        if self.required_languages:
            requirements["languages"] = self.required_languages

        # Add capability requirements
        for cap in self.required_capabilities:
            if cap == "vision":
                requirements["vision_capable"] = True
            elif cap == "code":
                requirements["code_specialized"] = True
            elif cap == "tools" or cap == "function_calling":
                requirements["supports_function_calling"] = True
            elif cap == "structured_output":
                requirements["supports_structured_output"] = True
            elif cap == "json_mode":
                requirements["supports_json_mode"] = True

        if self.required_domains:
            requirements["expertise"] = self.required_domains  # Use expertise for compatibility

        if self.min_model_size:
            requirements["min_size"] = str(self.min_model_size)

        return requirements


class ModelSelector:
    """
    Advanced model selection engine with AUTO tag support.

    Selects optimal models based on requirements, constraints, and strategies.
    """

    def __init__(self, registry: ModelRegistry):
        """
        Initialize model selector.

        Args:
            registry: Model registry to select from
        """
        self.registry = registry

    async def select_model(
        self,
        criteria: ModelSelectionCriteria,
        auto_description: Optional[str] = None,
    ) -> Model:
        """
        Select the best model based on criteria.

        Args:
            criteria: Selection criteria
            auto_description: Optional AUTO tag description for dynamic selection

        Returns:
            Selected model

        Raises:
            NoEligibleModelsError: If no models meet criteria
        """
        # Parse AUTO tag if provided
        if auto_description:
            criteria = self._parse_auto_tag(auto_description, criteria)

        # Convert criteria to requirements dict for registry
        requirements = criteria.to_requirements_dict()

        # Get eligible models from registry
        eligible_models = await self.registry._filter_by_capabilities(requirements)

        if not eligible_models:
            # Try without strict requirements for domain routing
            if criteria.required_domains:
                # Remove expertise requirement and try again
                requirements.pop("expertise", None)
                eligible_models = await self.registry._filter_by_capabilities(requirements)

            if not eligible_models:
                # Get all models and see why they failed
                all_models = list(self.registry.models.values())
                if all_models:
                    # Try with just the basic requirements
                    basic_req = {"context_window": requirements.get("context_window", 0)}
                    eligible_models = await self.registry._filter_by_capabilities(basic_req)

                if not eligible_models:
                    raise NoEligibleModelsError("No models meet the basic requirements")

        # Apply additional filters
        eligible_models = self._filter_by_criteria(eligible_models, criteria)

        if not eligible_models:
            raise NoEligibleModelsError("No models meet all criteria")

        # Score and rank models
        scored_models = self._score_models(eligible_models, criteria)

        # Select best model based on strategy
        selected_model = self._select_by_strategy(scored_models, criteria.selection_strategy)

        return selected_model

    def _parse_auto_tag(
        self, description: str, base_criteria: ModelSelectionCriteria
    ) -> ModelSelectionCriteria:
        """
        Parse AUTO tag description to extract selection criteria.

        Args:
            description: AUTO tag content
            base_criteria: Base criteria to extend

        Returns:
            Updated criteria
        """
        criteria = base_criteria
        description_lower = description.lower()

        # Parse task requirements
        task_keywords = {
            "analyz": ["analyze"],
            "generat": ["generate"],
            "code": ["code"],
            "reason": ["reasoning"],
            "creative": ["creative"],
            "chat": ["chat"],
            "instruct": ["instruct"],
            "transform": ["transform"],
            "vision": ["vision"],
            "image": ["vision"],
            "audio": ["audio"],
            "voice": ["audio"],
        }

        for keyword, tasks in task_keywords.items():
            if keyword in description_lower:
                criteria.required_tasks.extend(tasks)

        # Parse capability requirements
        if "vision" in description_lower or "image" in description_lower:
            if "vision" not in criteria.required_capabilities:
                criteria.required_capabilities.append("vision")

        if "code" in description_lower or "programming" in description_lower:
            if "code" not in criteria.required_capabilities:
                criteria.required_capabilities.append("code")

        if "function" in description_lower or "tool" in description_lower:
            if "tools" not in criteria.required_capabilities:
                criteria.required_capabilities.append("tools")

        # Parse domain requirements
        domain_keywords = {
            "medical": ["medical"],
            "health": ["medical"],
            "legal": ["legal"],
            "law": ["legal"],
            "creative": ["creative"],
            "art": ["creative"],
            "technical": ["technical"],
            "scientific": ["scientific"],
            "math": ["math"],
        }

        for keyword, domains in domain_keywords.items():
            if keyword in description_lower:
                criteria.required_domains.extend(domains)

        # Parse performance preferences
        if "fast" in description_lower or "quick" in description_lower:
            criteria.speed_preference = "fast"
        elif "accurate" in description_lower or "quality" in description_lower:
            criteria.selection_strategy = "accuracy_optimized"
            criteria.min_accuracy_score = max(criteria.min_accuracy_score, 0.9)
        elif "cheap" in description_lower or "cost" in description_lower:
            criteria.selection_strategy = "cost_optimized"
            criteria.prefer_free_models = True

        # Parse size preferences
        size_match = re.search(r"(\d+\.?\d*)\s*[bB]", description)
        if size_match:
            size = float(size_match.group(1))
            criteria.min_model_size = size

        # Parse context window requirements
        context_match = re.search(r"(\d+)[kK]?\s*(tokens?|context)", description)
        if context_match:
            context_size = int(context_match.group(1))
            if "k" in description_lower[context_match.start() : context_match.end()]:
                context_size *= 1000
            criteria.min_context_window = max(criteria.min_context_window, context_size)

        return criteria

    def _filter_by_criteria(
        self, models: List[Model], criteria: ModelSelectionCriteria
    ) -> List[Model]:
        """
        Filter models by additional criteria not handled by registry.

        Args:
            models: List of models to filter
            criteria: Selection criteria

        Returns:
            Filtered list of models
        """
        filtered = []

        for model in models:
            # Check provider constraints (only apply if explicitly set)
            if criteria.preferred_providers and model.provider not in criteria.preferred_providers:
                continue
            if model.provider in criteria.excluded_providers:
                continue

            # Check model name constraints
            model_id = f"{model.provider}:{model.name}"
            if criteria.preferred_models and model_id not in criteria.preferred_models:
                continue
            if model_id in criteria.excluded_models:
                continue

            # Check capability requirements
            caps = model.capabilities
            if "vision" in criteria.required_capabilities and not caps.vision_capable:
                continue
            if "code" in criteria.required_capabilities and not caps.code_specialized:
                continue
            if "tools" in criteria.required_capabilities and not caps.supports_function_calling:
                continue
            if "json_mode" in criteria.required_capabilities and not caps.supports_json_mode:
                continue

            # Check domain requirements (more lenient - prefer but don't require)
            # Domain matching is handled in scoring instead of filtering

            # Check accuracy requirements
            if (
                criteria.min_accuracy_score > 0
                and caps.accuracy_score < criteria.min_accuracy_score
            ):
                continue

            # Check speed requirements
            if criteria.speed_preference:
                if criteria.speed_preference == "fast" and caps.speed_rating == "slow":
                    continue
                elif criteria.speed_preference == "slow" and caps.speed_rating == "fast":
                    continue

            # Check size constraints
            model_size = getattr(model, "_size_billions", 1.0)
            if criteria.min_model_size and model_size < criteria.min_model_size:
                continue
            if criteria.max_model_size and model_size > criteria.max_model_size:
                continue

            # Check cost constraints
            if criteria.max_cost_per_1k_tokens:
                avg_cost = (
                    model.cost.input_cost_per_1k_tokens + model.cost.output_cost_per_1k_tokens
                ) / 2
                if avg_cost > criteria.max_cost_per_1k_tokens:
                    continue

            if criteria.prefer_free_models and not model.cost.is_free:
                # Don't exclude paid models, but they'll score lower
                pass

            filtered.append(model)

        return filtered

    def _score_models(
        self, models: List[Model], criteria: ModelSelectionCriteria
    ) -> List[Tuple[Model, float]]:
        """
        Score models based on criteria.

        Args:
            models: List of models to score
            criteria: Selection criteria

        Returns:
            List of (model, score) tuples sorted by score descending
        """
        scored = []

        for model in models:
            score = 0.0

            # Base score from model metrics
            score += model.capabilities.accuracy_score * 10  # 0-10 points for accuracy
            score += model.metrics.success_rate * 5  # 0-5 points for reliability

            # Speed scoring
            speed_scores = {"fast": 3, "medium": 2, "slow": 1}
            score += speed_scores.get(model.capabilities.speed_rating, 2)

            # Cost scoring (inverse - lower cost = higher score)
            if model.cost.is_free:
                score += 5  # Bonus for free models
            else:
                avg_cost = (
                    model.cost.input_cost_per_1k_tokens + model.cost.output_cost_per_1k_tokens
                ) / 2
                if avg_cost < 0.001:  # Very cheap
                    score += 4
                elif avg_cost < 0.01:  # Cheap
                    score += 3
                elif avg_cost < 0.1:  # Moderate
                    score += 2
                elif avg_cost < 1.0:  # Expensive
                    score += 1
                # Very expensive models get 0 points

            # Capability match scoring
            if criteria.required_capabilities:
                capability_match = 0
                if "vision" in criteria.required_capabilities and model.capabilities.vision_capable:
                    capability_match += 1
                if "code" in criteria.required_capabilities and model.capabilities.code_specialized:
                    capability_match += 1
                if (
                    "tools" in criteria.required_capabilities
                    and model.capabilities.supports_function_calling
                ):
                    capability_match += 1

                score += capability_match * 2

            # Domain match scoring
            if criteria.required_domains and model.capabilities.domains:
                domain_match = len(set(criteria.required_domains) & set(model.capabilities.domains))
                score += domain_match * 3

            # Size preference scoring
            model_size = getattr(model, "_size_billions", 1.0)
            if criteria.min_model_size:
                # Prefer models closer to minimum size (not unnecessarily large)
                size_ratio = model_size / criteria.min_model_size
                if size_ratio < 2:  # Within 2x of minimum
                    score += 2
                elif size_ratio < 5:  # Within 5x of minimum
                    score += 1

            # Provider preference scoring
            if criteria.preferred_providers and model.provider in criteria.preferred_providers:
                score += 3

            # Model preference scoring
            model_id = f"{model.provider}:{model.name}"
            if criteria.preferred_models and model_id in criteria.preferred_models:
                score += 5

            scored.append((model, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _select_by_strategy(self, scored_models: List[Tuple[Model, float]], strategy: str) -> Model:
        """
        Select final model based on strategy.

        Args:
            scored_models: List of (model, score) tuples sorted by score
            strategy: Selection strategy

        Returns:
            Selected model
        """
        if not scored_models:
            raise NoEligibleModelsError("No models available after scoring")

        if strategy == "balanced":
            # Select highest scoring model
            return scored_models[0][0]

        elif strategy == "cost_optimized":
            # Find cheapest model in top 50% by score
            if len(scored_models) == 1:
                return scored_models[0][0]

            top_half = scored_models[: max(1, len(scored_models) // 2)]

            # Sort by cost
            def get_avg_cost(model_score_tuple):
                model = model_score_tuple[0]
                if model.cost.is_free:
                    return 0.0
                return (
                    model.cost.input_cost_per_1k_tokens + model.cost.output_cost_per_1k_tokens
                ) / 2

            top_half.sort(key=get_avg_cost)
            return top_half[0][0]

        elif strategy == "performance_optimized":
            # Find fastest model in top 50% by score
            if len(scored_models) == 1:
                return scored_models[0][0]

            top_half = scored_models[: max(1, len(scored_models) // 2)]

            # Sort by speed rating
            speed_order = {"fast": 0, "medium": 1, "slow": 2}
            top_half.sort(key=lambda x: speed_order.get(x[0].capabilities.speed_rating, 1))
            return top_half[0][0]

        elif strategy == "accuracy_optimized":
            # Find most accurate model regardless of other factors
            scored_models.sort(key=lambda x: x[0].capabilities.accuracy_score, reverse=True)
            return scored_models[0][0]

        else:
            # Default to balanced
            return scored_models[0][0]

    async def resolve_auto_model(self, auto_tag: str, context: Dict[str, Any]) -> str:
        """
        Resolve an AUTO tag to a specific model name.

        Args:
            auto_tag: AUTO tag content
            context: Context dictionary with task information

        Returns:
            Model identifier (provider:name)
        """
        # Create criteria from context
        criteria = ModelSelectionCriteria()

        # Extract task type from context
        if "task_type" in context:
            criteria.required_tasks = [context["task_type"]]

        # Extract other requirements from context
        if "min_context" in context:
            criteria.min_context_window = context["min_context"]

        if "require_vision" in context:
            criteria.required_capabilities.append("vision")

        if "require_code" in context:
            criteria.required_capabilities.append("code")

        if "cost_sensitive" in context and context["cost_sensitive"]:
            criteria.selection_strategy = "cost_optimized"

        # Select model
        model = await self.select_model(criteria, auto_tag)

        return f"{model.provider}:{model.name}"
