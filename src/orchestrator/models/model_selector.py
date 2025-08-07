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
    required_domains: List[str] = field(
        default_factory=list
    )  # e.g., ["medical", "legal"]

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
    min_tokens_per_second: Optional[int] = None  # Minimum throughput requirement

    # Model preferences
    preferred_providers: List[str] = field(default_factory=list)
    excluded_providers: List[str] = field(default_factory=list)
    preferred_models: List[str] = field(default_factory=list)
    excluded_models: List[str] = field(default_factory=list)

    # Size constraints
    min_model_size: Optional[float] = None  # In billions of parameters
    max_model_size: Optional[float] = None

    # Enhanced requirements for issue 194
    expertise: Optional[str] = None  # "low", "medium", "high", "very-high"
    modalities: List[str] = field(default_factory=list)  # "text", "vision", "code", "audio"
    
    # Cost constraints (enhanced)
    cost_limit: Optional[float] = None  # Per execution in USD
    budget_period: Optional[str] = None  # "per-task", "per-pipeline", "per-hour"
    fallback_strategy: str = "best_available"  # "best_available", "fail", "cheapest"

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
            requirements["expertise"] = (
                self.required_domains
            )  # Use expertise for compatibility

        # Add new expertise field (takes precedence over required_domains)
        if self.expertise:
            requirements["expertise"] = self.expertise

        # Add modality requirements
        for modality in self.modalities:
            if modality == "vision":
                requirements["vision_capable"] = True
            elif modality == "code":
                requirements["code_specialized"] = True
            elif modality == "audio":
                requirements["audio_capable"] = True
            elif modality == "text":
                # Text is default, no special requirement needed
                pass

        if self.min_model_size:
            requirements["min_size"] = str(self.min_model_size)
        if self.max_model_size:
            requirements["max_size"] = str(self.max_model_size)

        # Add performance requirements
        if self.min_tokens_per_second:
            requirements["min_tokens_per_second"] = self.min_tokens_per_second

        # Add cost constraints
        if self.cost_limit:
            requirements["cost_limit"] = self.cost_limit
        if self.budget_period:
            requirements["budget_period"] = self.budget_period

        # Add fallback strategy
        requirements["fallback_strategy"] = self.fallback_strategy

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
            # Handle fallback based on strategy
            eligible_models = await self._handle_no_models_fallback(criteria, requirements)

        # Apply additional filters
        eligible_models = self._filter_by_criteria(eligible_models, criteria)

        if not eligible_models:
            if criteria.fallback_strategy == "fail":
                raise NoEligibleModelsError("No models meet all criteria and fallback_strategy is 'fail'")
            else:
                # Final fallback - try to get any available model
                eligible_models = await self._get_fallback_models(criteria)

        # Score and rank models
        scored_models = self._score_models(eligible_models, criteria)

        # Select best model based on strategy
        selected_model = self._select_by_strategy(
            scored_models, criteria.selection_strategy
        )

        return selected_model

    async def _handle_no_models_fallback(
        self, 
        criteria: ModelSelectionCriteria, 
        requirements: Dict[str, Any]
    ) -> List[Model]:
        """
        Handle fallback when no models meet initial requirements.
        
        Args:
            criteria: Original selection criteria
            requirements: Requirements dictionary
            
        Returns:
            List of fallback models
        """
        if criteria.fallback_strategy == "fail":
            return []
        
        # Progressive relaxation of requirements
        fallback_attempts = [
            # 1. Remove cost constraints
            lambda req: {k: v for k, v in req.items() if k not in ["cost_limit", "budget_period"]},
            
            # 2. Remove performance constraints  
            lambda req: {k: v for k, v in req.items() if k not in ["cost_limit", "budget_period", "min_tokens_per_second", "max_latency_ms"]},
            
            # 3. Remove expertise requirements
            lambda req: {k: v for k, v in req.items() if k not in ["cost_limit", "budget_period", "min_tokens_per_second", "max_latency_ms", "expertise"]},
            
            # 4. Remove size constraints
            lambda req: {k: v for k, v in req.items() if k not in ["cost_limit", "budget_period", "min_tokens_per_second", "max_latency_ms", "expertise", "min_size", "max_size"]},
            
            # 5. Keep only basic capabilities
            lambda req: {k: v for k, v in req.items() if k in ["vision_capable", "code_specialized", "supports_function_calling"]},
        ]
        
        for attempt_func in fallback_attempts:
            relaxed_req = attempt_func(requirements)
            models = await self.registry._filter_by_capabilities(relaxed_req)
            if models:
                return models
        
        # Final fallback - get any available models
        all_models = list(self.registry.models.values())
        return [model for model in all_models if hasattr(model, 'is_available') and getattr(model, 'is_available', True)]

    async def _get_fallback_models(self, criteria: ModelSelectionCriteria) -> List[Model]:
        """
        Get fallback models based on fallback strategy.
        
        Args:
            criteria: Selection criteria
            
        Returns:
            List of fallback models
        """
        all_models = list(self.registry.models.values())
        available_models = [m for m in all_models if getattr(m, 'is_available', True)]
        
        if not available_models:
            raise NoEligibleModelsError("No models are available")
        
        if criteria.fallback_strategy == "cheapest":
            # Sort by cost (free models first, then by cost per token)
            def cost_key(model):
                if model.cost.is_free:
                    return (0, 0)  # Free models get priority
                # Use average of input and output costs
                avg_cost = (model.cost.input_cost_per_1k_tokens + model.cost.output_cost_per_1k_tokens) / 2
                return (1, avg_cost)
            
            available_models.sort(key=cost_key)
            return available_models[:3]  # Return top 3 cheapest
        
        else:  # "best_available"
            # Sort by a combination of accuracy and capabilities
            def quality_key(model):
                accuracy = getattr(model.capabilities, 'accuracy_score', 0.5)
                # Prefer models with more capabilities
                capability_score = len(getattr(model.capabilities, 'supported_tasks', []))
                return -(accuracy * 10 + capability_score)  # Negative for descending sort
            
            available_models.sort(key=quality_key)
            return available_models[:5]  # Return top 5 best

    def parse_requirements_from_yaml(self, requires_model: Dict[str, Any]) -> ModelSelectionCriteria:
        """
        Parse YAML requires_model section into ModelSelectionCriteria.
        
        Args:
            requires_model: Dictionary from YAML requires_model section
            
        Returns:
            ModelSelectionCriteria object with parsed requirements
        """
        criteria = ModelSelectionCriteria()
        
        # Size constraints
        if "min_size" in requires_model:
            size_str = requires_model["min_size"]
            # Parse size string to float (e.g., "7B" -> 7.0)
            criteria.min_model_size = self._parse_size_string(size_str)
        
        if "max_size" in requires_model:
            size_str = requires_model["max_size"]
            criteria.max_model_size = self._parse_size_string(size_str)
        
        # Expertise level
        if "expertise" in requires_model:
            criteria.expertise = requires_model["expertise"]
        
        # Capabilities
        if "capabilities" in requires_model:
            criteria.required_capabilities = requires_model["capabilities"]
        
        # Modalities
        if "modalities" in requires_model:
            criteria.modalities = requires_model["modalities"]
        
        # Performance constraints
        if "max_latency_ms" in requires_model:
            criteria.max_latency_ms = requires_model["max_latency_ms"]
        
        if "min_tokens_per_second" in requires_model:
            criteria.min_tokens_per_second = requires_model["min_tokens_per_second"]
        
        if "min_accuracy_score" in requires_model:
            criteria.min_accuracy_score = requires_model["min_accuracy_score"]
        
        # Cost constraints
        if "cost_limit" in requires_model:
            criteria.cost_limit = requires_model["cost_limit"]
        
        if "budget_period" in requires_model:
            criteria.budget_period = requires_model["budget_period"]
        
        # Model preferences
        if "preferred" in requires_model:
            criteria.preferred_models = requires_model["preferred"]
        
        if "excluded" in requires_model:
            criteria.excluded_models = requires_model["excluded"]
        
        # Fallback strategy
        if "fallback_strategy" in requires_model:
            criteria.fallback_strategy = requires_model["fallback_strategy"]
        
        return criteria
    
    def _parse_size_string(self, size_str: str) -> float:
        """
        Parse model size string to float in billions.
        
        Args:
            size_str: Size string like "1B", "7B", "70B", "1.5B"
            
        Returns:
            Size in billions of parameters
        """
        # Use existing utility function
        from ..utils.model_utils import parse_model_size
        return parse_model_size("", size_str)

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

        # Parse task requirements - use word boundaries for precise matching
        task_patterns = {
            r"\b(analyz|analysis|analyze)\b": ["analyze"],
            r"\b(generat|generate|generation)\b": ["generate"],
            r"\b(code|coding|programming)\b": ["code"],
            r"\b(reason|reasoning)\b": ["reasoning"],
            r"\b(creative|creativity)\b": ["creative"],
            r"\b(chat|conversation)\b": ["chat"],
            r"\b(instruct|instruction)\b": ["instruct"],
            r"\b(transform|transformation)\b": ["transform"],
            r"\b(vision|visual)\b": ["vision"],
            r"\b(image|picture|photo)\b": ["vision"],
            r"\b(audio|sound)\b": ["audio"],
            r"\b(voice|speech)\b": ["audio"],
        }

        for pattern, tasks in task_patterns.items():
            if re.search(pattern, description_lower):
                criteria.required_tasks.extend(tasks)

        # Parse capability requirements - use word boundaries for precise matching
        if re.search(r"\b(vision|image)\b", description_lower):
            if "vision" not in criteria.required_capabilities:
                criteria.required_capabilities.append("vision")

        if re.search(r"\b(code|programming|coding|script|program)\b", description_lower):
            if "code" not in criteria.required_capabilities:
                criteria.required_capabilities.append("code")

        # Only match "function" and "tool" as standalone words, not as parts of other words
        if re.search(r"\b(function calling|function\s+call|use\s+tools?|call\s+functions?)\b", description_lower):
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
            if (
                criteria.preferred_providers
                and model.provider not in criteria.preferred_providers
            ):
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
            if (
                "tools" in criteria.required_capabilities
                and not caps.supports_function_calling
            ):
                continue
            if (
                "json_mode" in criteria.required_capabilities
                and not caps.supports_json_mode
            ):
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
                elif (
                    criteria.speed_preference == "slow" and caps.speed_rating == "fast"
                ):
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
                    model.cost.input_cost_per_1k_tokens
                    + model.cost.output_cost_per_1k_tokens
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
        Enhanced scoring based on new criteria and requirements.

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

            # Enhanced cost scoring
            if model.cost.is_free:
                score += 5  # Bonus for free models
            elif criteria.cost_limit:
                # Calculate estimated cost for budget period
                estimated_cost = self._estimate_model_cost(model, criteria.budget_period or "per-task")
                if estimated_cost <= criteria.cost_limit:
                    # Reward models that fit within budget (inverse scoring)
                    cost_efficiency = (criteria.cost_limit - estimated_cost) / criteria.cost_limit
                    score += cost_efficiency * 3  # Up to 3 points for cost efficiency
                else:
                    score -= 2  # Penalty for over-budget models
            else:
                # Default cost scoring
                avg_cost = (
                    model.cost.input_cost_per_1k_tokens
                    + model.cost.output_cost_per_1k_tokens
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
                if (
                    "vision" in criteria.required_capabilities
                    and model.capabilities.vision_capable
                ):
                    capability_match += 1
                if (
                    "code" in criteria.required_capabilities
                    and model.capabilities.code_specialized
                ):
                    capability_match += 1
                if (
                    "tools" in criteria.required_capabilities
                    and model.capabilities.supports_function_calling
                ):
                    capability_match += 1

                score += capability_match * 2

            # Domain match scoring
            if criteria.required_domains and model.capabilities.domains:
                domain_match = len(
                    set(criteria.required_domains) & set(model.capabilities.domains)
                )
                score += domain_match * 3

            # Size preference scoring
            model_size = getattr(model, "_size_billions", 1.0)
            if criteria.min_model_size:
                # Prefer models closer to minimum size (not unnecessarily large)
                size_ratio = model_size / criteria.min_model_size
                if size_ratio < 2:  # Within 2x of minimum
                    score += 2
                elif size_ratio > 10:  # Very oversized
                    score -= 1

            # Expertise level scoring
            if criteria.expertise:
                expertise_bonus = self._score_expertise_match(model, criteria.expertise)
                score += expertise_bonus

            # Modality scoring
            if criteria.modalities:
                modality_bonus = self._score_modality_match(model, criteria.modalities)
                score += modality_bonus

            # Performance scoring
            if criteria.min_tokens_per_second:
                throughput = getattr(model.metrics, "throughput", 0)
                if throughput >= criteria.min_tokens_per_second:
                    score += 2  # Meets throughput requirement
                elif throughput > 0:
                    # Partial credit for some throughput data
                    ratio = throughput / criteria.min_tokens_per_second
                    score += ratio * 2

            # Preferred model bonus
            model_id = f"{model.provider}:{model.name}"
            if model_id in criteria.preferred_models:
                score += 5  # Strong preference bonus

            # Provider preference scoring
            if (
                criteria.preferred_providers
                and model.provider in criteria.preferred_providers
            ):
                score += 3

            scored.append((model, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _score_expertise_match(self, model: Model, required_expertise: str) -> float:
        """
        Score model based on expertise level match.
        
        Args:
            model: Model to score
            required_expertise: Required expertise level
            
        Returns:
            Expertise match bonus score (0-5 points)
        """
        # Use the registry's expertise level checking logic
        if self.registry._meets_expertise_level(model, required_expertise):
            # Exact level matches get higher scores
            model_expertise_attrs = getattr(model, "_expertise", ["general"])
            if model_expertise_attrs is None:
                model_expertise_attrs = ["general"]
            
            # Map required level to expected attributes
            level_attrs = {
                "very-high": ["analysis", "research"],
                "high": ["code", "reasoning", "math"],
                "medium": ["general", "chat"],
                "low": ["fast", "compact"]
            }
            
            expected_attrs = level_attrs.get(required_expertise, ["general"])
            
            # Higher score for exact attribute matches
            matches = len(set(expected_attrs) & set(model_expertise_attrs))
            if matches > 0:
                return 5.0  # Strong expertise match
            else:
                return 2.0  # Meets level but not perfect match
        else:
            return 0.0  # Doesn't meet requirement

    def _score_modality_match(self, model: Model, required_modalities: List[str]) -> float:
        """
        Score model based on modality support.
        
        Args:
            model: Model to score
            required_modalities: List of required modalities
            
        Returns:
            Modality match bonus score (0-3 per modality)
        """
        score = 0.0
        caps = model.capabilities
        
        for modality in required_modalities:
            if modality == "text":
                # All models support text
                score += 1.0
            elif modality == "vision" and caps.vision_capable:
                score += 3.0  # Vision is valuable
            elif modality == "code" and caps.code_specialized:
                score += 2.0  # Code specialization is valuable
            elif modality == "audio":
                # Check if model has audio capability (future enhancement)
                audio_capable = getattr(caps, "audio_capable", False)
                if audio_capable:
                    score += 3.0  # Audio is rare and valuable
        
        return score

    def _estimate_model_cost(self, model: Model, budget_period: str) -> float:
        """
        Estimate model cost for the given budget period.
        
        Args:
            model: Model to estimate cost for
            budget_period: Budget period ("per-task", "per-pipeline", "per-hour")
            
        Returns:
            Estimated cost in USD
        """
        if model.cost.is_free:
            return 0.0
        
        # Use the same logic as ModelRegistry._meets_cost_constraint
        if budget_period == "per-task":
            # Assume 1000 tokens average per task
            return model.cost.calculate_cost(500, 500)  # 500 input + 500 output
        elif budget_period == "per-pipeline":
            # Assume 5000 tokens average per pipeline
            return model.cost.calculate_cost(2500, 2500)
        elif budget_period == "per-hour":
            # Assume 50000 tokens per hour
            return model.cost.calculate_cost(25000, 25000)
        else:
            # Default to per-task
            return model.cost.calculate_cost(500, 500)

    def _select_by_strategy(
        self, scored_models: List[Tuple[Model, float]], strategy: str
    ) -> Model:
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
                    model.cost.input_cost_per_1k_tokens
                    + model.cost.output_cost_per_1k_tokens
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
            top_half.sort(
                key=lambda x: speed_order.get(x[0].capabilities.speed_rating, 1)
            )
            return top_half[0][0]

        elif strategy == "accuracy_optimized":
            # Find most accurate model regardless of other factors
            scored_models.sort(
                key=lambda x: x[0].capabilities.accuracy_score, reverse=True
            )
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
