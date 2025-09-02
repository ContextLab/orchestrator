"""Intelligent model selection strategies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ...core.model import Model, ModelCapabilities, ModelCost
from ..registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class TaskRequirements:
    """Requirements for a task that need to be matched by models."""
    
    task_type: str
    context_window: Optional[int] = None
    max_cost_per_1k_tokens: Optional[float] = None
    max_latency_ms: Optional[float] = None
    required_capabilities: List[str] = field(default_factory=list)
    preferred_domains: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["en"])
    accuracy_threshold: float = 0.7
    budget_limit: Optional[float] = None
    budget_period: str = "per-task"
    exclude_providers: Set[str] = field(default_factory=set)
    prefer_local: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_type": self.task_type,
            "context_window": self.context_window,
            "max_cost_per_1k_tokens": self.max_cost_per_1k_tokens,
            "max_latency_ms": self.max_latency_ms,
            "required_capabilities": self.required_capabilities,
            "preferred_domains": self.preferred_domains,
            "languages": self.languages,
            "accuracy_threshold": self.accuracy_threshold,
            "budget_limit": self.budget_limit,
            "budget_period": self.budget_period,
            "exclude_providers": list(self.exclude_providers),
            "prefer_local": self.prefer_local,
        }


@dataclass
class SelectionResult:
    """Result of model selection process."""
    
    model: Model
    provider: str
    confidence_score: float
    selection_reason: str
    alternatives: List[Tuple[Model, str, float]] = field(default_factory=list)
    estimated_cost: Optional[float] = None
    expected_latency: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model": {
                "name": self.model.name,
                "provider": self.model.provider,
            },
            "provider": self.provider,
            "confidence_score": self.confidence_score,
            "selection_reason": self.selection_reason,
            "alternatives": [
                {
                    "model": {"name": model.name, "provider": model.provider},
                    "provider": provider,
                    "confidence_score": score
                }
                for model, provider, score in self.alternatives
            ],
            "estimated_cost": self.estimated_cost,
            "expected_latency": self.expected_latency,
        }


class SelectionStrategy(ABC):
    """Abstract base class for model selection strategies."""

    def __init__(self, name: str):
        """Initialize selection strategy."""
        self.name = name

    @abstractmethod
    async def select_model(
        self,
        registry: ModelRegistry,
        requirements: TaskRequirements,
        available_models: Optional[List[Model]] = None,
    ) -> SelectionResult:
        """
        Select the best model for given requirements.
        
        Args:
            registry: Model registry to select from
            requirements: Task requirements
            available_models: Pre-filtered list of models (optional)
            
        Returns:
            Selection result with chosen model and metadata
        """
        pass

    @abstractmethod
    def score_model(self, model: Model, requirements: TaskRequirements) -> float:
        """
        Score a model against requirements.
        
        Args:
            model: Model to score
            requirements: Task requirements
            
        Returns:
            Score between 0 and 1 (higher is better)
        """
        pass


class TaskBasedStrategy(SelectionStrategy):
    """Strategy that selects models primarily based on task type and capabilities."""

    def __init__(self, name: str = "task_based"):
        super().__init__(name)
        
        # Task-specific weights for different capabilities
        self.task_weights = {
            "text_generation": {
                "context_window": 0.3,
                "supported_tasks": 0.4,
                "accuracy_score": 0.2,
                "speed_rating": 0.1,
            },
            "code_generation": {
                "code_specialized": 0.4,
                "supported_tasks": 0.3,
                "context_window": 0.2,
                "accuracy_score": 0.1,
            },
            "analysis": {
                "accuracy_score": 0.4,
                "supported_tasks": 0.3,
                "context_window": 0.2,
                "speed_rating": 0.1,
            },
            "creative_writing": {
                "supported_tasks": 0.3,
                "accuracy_score": 0.2,
                "context_window": 0.2,
                "domains": 0.3,
            },
            "multimodal": {
                "supports_multimodal": 0.5,
                "vision_capable": 0.3,
                "audio_capable": 0.1,
                "accuracy_score": 0.1,
            },
        }

    async def select_model(
        self,
        registry: ModelRegistry,
        requirements: TaskRequirements,
        available_models: Optional[List[Model]] = None,
    ) -> SelectionResult:
        """Select model based on task type and capabilities."""
        if available_models is None:
            # Get all available models from registry
            model_info = registry.list_models()
            models = []
            for model_name, info in model_info.items():
                if "error" not in info:
                    try:
                        model = await registry.get_model(model_name, info["provider"])
                        models.append(model)
                    except Exception as e:
                        logger.warning(f"Failed to load model {model_name}: {e}")
        else:
            models = available_models

        if not models:
            raise ValueError("No models available for selection")

        # Score all models
        scored_models = []
        for model in models:
            # Check basic compatibility
            if not self._is_compatible(model, requirements):
                continue
                
            score = self.score_model(model, requirements)
            scored_models.append((model, score))

        if not scored_models:
            raise ValueError(f"No compatible models found for task: {requirements.task_type}")

        # Sort by score
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        best_model, best_score = scored_models[0]
        alternatives = [(model, best_model.provider, score) for model, score in scored_models[1:5]]

        # Generate selection reason
        reason = self._generate_selection_reason(best_model, requirements, best_score)

        return SelectionResult(
            model=best_model,
            provider=best_model.provider,
            confidence_score=best_score,
            selection_reason=reason,
            alternatives=alternatives,
        )

    def score_model(self, model: Model, requirements: TaskRequirements) -> float:
        """Score model based on task-specific criteria."""
        # Get task-specific weights
        task_weights = self.task_weights.get(requirements.task_type, self.task_weights["text_generation"])
        
        score = 0.0
        
        # Context window score
        if "context_window" in task_weights and requirements.context_window:
            context_score = min(1.0, model.capabilities.context_window / requirements.context_window)
            score += task_weights["context_window"] * context_score

        # Task support score
        if "supported_tasks" in task_weights:
            task_score = 1.0 if model.capabilities.supports_task(requirements.task_type) else 0.5
            score += task_weights["supported_tasks"] * task_score

        # Accuracy score
        if "accuracy_score" in task_weights:
            accuracy_score = model.capabilities.accuracy_score
            score += task_weights["accuracy_score"] * accuracy_score

        # Speed rating score
        if "speed_rating" in task_weights:
            speed_scores = {"fast": 1.0, "medium": 0.7, "slow": 0.4}
            speed_score = speed_scores.get(model.capabilities.speed_rating, 0.5)
            score += task_weights["speed_rating"] * speed_score

        # Specialized capabilities
        if "code_specialized" in task_weights:
            code_score = 1.0 if model.capabilities.code_specialized else 0.3
            score += task_weights["code_specialized"] * code_score

        if "supports_multimodal" in task_weights:
            multimodal_score = 1.0 if model.capabilities.supports_multimodal else 0.0
            score += task_weights["supports_multimodal"] * multimodal_score

        if "vision_capable" in task_weights:
            vision_score = 1.0 if model.capabilities.vision_capable else 0.0
            score += task_weights["vision_capable"] * vision_score

        if "audio_capable" in task_weights:
            audio_score = 1.0 if model.capabilities.audio_capable else 0.0
            score += task_weights["audio_capable"] * audio_score

        # Domain preference bonus
        if "domains" in task_weights and requirements.preferred_domains:
            domain_overlap = set(model.capabilities.domains) & set(requirements.preferred_domains)
            domain_score = len(domain_overlap) / len(requirements.preferred_domains) if requirements.preferred_domains else 0
            score += task_weights["domains"] * domain_score

        return min(1.0, score)

    def _is_compatible(self, model: Model, requirements: TaskRequirements) -> bool:
        """Check if model is compatible with basic requirements."""
        # Check provider exclusions
        if model.provider in requirements.exclude_providers:
            return False

        # Check context window
        if requirements.context_window and model.capabilities.context_window < requirements.context_window:
            return False

        # Check accuracy threshold
        if model.capabilities.accuracy_score < requirements.accuracy_threshold:
            return False

        # Check required capabilities
        for capability in requirements.required_capabilities:
            if not getattr(model.capabilities, capability, False):
                return False

        # Check language support
        if not any(model.capabilities.supports_language(lang) for lang in requirements.languages):
            return False

        return True

    def _generate_selection_reason(self, model: Model, requirements: TaskRequirements, score: float) -> str:
        """Generate human-readable reason for selection."""
        reasons = []
        
        if model.capabilities.supports_task(requirements.task_type):
            reasons.append(f"specialized for {requirements.task_type}")
        
        if requirements.context_window and model.capabilities.context_window >= requirements.context_window:
            reasons.append(f"large context window ({model.capabilities.context_window:,} tokens)")
            
        if model.capabilities.accuracy_score >= 0.9:
            reasons.append("high accuracy")
        elif model.capabilities.accuracy_score >= 0.8:
            reasons.append("good accuracy")
            
        if model.capabilities.speed_rating == "fast":
            reasons.append("fast response time")
            
        if requirements.preferred_domains:
            domain_overlap = set(model.capabilities.domains) & set(requirements.preferred_domains)
            if domain_overlap:
                reasons.append(f"domain expertise in {', '.join(domain_overlap)}")

        base_reason = f"Best match for {requirements.task_type} (score: {score:.2f})"
        if reasons:
            return f"{base_reason}: {', '.join(reasons)}"
        return base_reason


class CostAwareStrategy(SelectionStrategy):
    """Strategy that prioritizes cost efficiency while meeting requirements."""

    def __init__(self, name: str = "cost_aware", cost_weight: float = 0.4):
        super().__init__(name)
        self.cost_weight = cost_weight  # Weight given to cost vs other factors

    async def select_model(
        self,
        registry: ModelRegistry,
        requirements: TaskRequirements,
        available_models: Optional[List[Model]] = None,
    ) -> SelectionResult:
        """Select most cost-effective model that meets requirements."""
        if available_models is None:
            # Get all available models from registry
            model_info = registry.list_models()
            models = []
            for model_name, info in model_info.items():
                if "error" not in info:
                    try:
                        model = await registry.get_model(model_name, info["provider"])
                        models.append(model)
                    except Exception as e:
                        logger.warning(f"Failed to load model {model_name}: {e}")
        else:
            models = available_models

        if not models:
            raise ValueError("No models available for selection")

        # Filter by budget if specified
        if requirements.budget_limit:
            budget_compatible_models = []
            for model in models:
                if model.cost.is_within_budget(requirements.budget_limit, requirements.budget_period):
                    budget_compatible_models.append(model)
            models = budget_compatible_models

        if not models:
            raise ValueError("No models found within budget constraints")

        # Score all models with cost emphasis
        scored_models = []
        for model in models:
            score = self.score_model(model, requirements)
            scored_models.append((model, score))

        if not scored_models:
            raise ValueError("No compatible models found")

        # Sort by score (higher is better)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        best_model, best_score = scored_models[0]
        alternatives = [(model, best_model.provider, score) for model, score in scored_models[1:5]]

        # Estimate cost for the request
        estimated_cost = best_model.cost.estimate_cost_for_budget_period(requirements.budget_period)

        return SelectionResult(
            model=best_model,
            provider=best_model.provider,
            confidence_score=best_score,
            selection_reason=f"Most cost-effective choice (estimated ${estimated_cost:.4f} {requirements.budget_period})",
            alternatives=alternatives,
            estimated_cost=estimated_cost,
        )

    def score_model(self, model: Model, requirements: TaskRequirements) -> float:
        """Score model with heavy emphasis on cost efficiency."""
        # Base capability score
        capability_score = 0.0
        
        # Task compatibility
        if model.capabilities.supports_task(requirements.task_type):
            capability_score += 0.3
        
        # Context window adequacy
        if requirements.context_window:
            if model.capabilities.context_window >= requirements.context_window:
                capability_score += 0.2
            else:
                # Penalize insufficient context window
                return 0.0
        else:
            capability_score += 0.2
        
        # Accuracy
        capability_score += 0.3 * model.capabilities.accuracy_score
        
        # Language support
        if any(model.capabilities.supports_language(lang) for lang in requirements.languages):
            capability_score += 0.2

        # Cost efficiency score (higher is better for lower cost)
        performance_score = capability_score
        cost_efficiency = model.cost.get_cost_efficiency_score(performance_score)
        
        # Normalize cost efficiency to 0-1 range (logarithmic scaling)
        import math
        cost_score = min(1.0, math.log10(max(1, cost_efficiency)) / 2.0)
        
        # Free models get maximum cost score
        if model.cost.is_free:
            cost_score = 1.0
        
        # Weighted combination
        final_score = (1 - self.cost_weight) * capability_score + self.cost_weight * cost_score
        
        return final_score


class PerformanceBasedStrategy(SelectionStrategy):
    """Strategy that prioritizes performance metrics (speed, accuracy)."""

    def __init__(self, name: str = "performance_based", speed_weight: float = 0.3, accuracy_weight: float = 0.7):
        super().__init__(name)
        self.speed_weight = speed_weight
        self.accuracy_weight = accuracy_weight

    async def select_model(
        self,
        registry: ModelRegistry,
        requirements: TaskRequirements,
        available_models: Optional[List[Model]] = None,
    ) -> SelectionResult:
        """Select highest performing model for the task."""
        if available_models is None:
            # Get all available models from registry
            model_info = registry.list_models()
            models = []
            for model_name, info in model_info.items():
                if "error" not in info:
                    try:
                        model = await registry.get_model(model_name, info["provider"])
                        models.append(model)
                    except Exception as e:
                        logger.warning(f"Failed to load model {model_name}: {e}")
        else:
            models = available_models

        if not models:
            raise ValueError("No models available for selection")

        # Score all models based on performance
        scored_models = []
        for model in models:
            if self._meets_requirements(model, requirements):
                score = self.score_model(model, requirements)
                scored_models.append((model, score))

        if not scored_models:
            raise ValueError("No models meet the requirements")

        # Sort by score
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        best_model, best_score = scored_models[0]
        alternatives = [(model, best_model.provider, score) for model, score in scored_models[1:5]]

        return SelectionResult(
            model=best_model,
            provider=best_model.provider,
            confidence_score=best_score,
            selection_reason=f"Highest performance model (accuracy: {best_model.capabilities.accuracy_score:.2f}, speed: {best_model.capabilities.speed_rating})",
            alternatives=alternatives,
        )

    def score_model(self, model: Model, requirements: TaskRequirements) -> float:
        """Score model based on performance metrics."""
        # Accuracy score
        accuracy_score = model.capabilities.accuracy_score
        
        # Speed score
        speed_scores = {"fast": 1.0, "medium": 0.6, "slow": 0.2}
        speed_score = speed_scores.get(model.capabilities.speed_rating, 0.4)
        
        # Apply latency constraint if specified
        if requirements.max_latency_ms:
            # Use metrics if available, otherwise estimate from speed rating
            latency_estimate = {
                "fast": 500,    # 500ms
                "medium": 2000, # 2s
                "slow": 5000    # 5s
            }.get(model.capabilities.speed_rating, 2000)
            
            if model.metrics and model.metrics.latency_p95 > 0:
                latency_estimate = model.metrics.latency_p95
            
            if latency_estimate > requirements.max_latency_ms:
                return 0.0  # Disqualify if too slow
        
        # Weighted combination
        performance_score = (
            self.accuracy_weight * accuracy_score +
            self.speed_weight * speed_score
        )
        
        # Bonus for specialized capabilities
        if requirements.task_type == "code_generation" and model.capabilities.code_specialized:
            performance_score += 0.1
        
        if requirements.required_capabilities:
            capability_bonus = 0.1 * sum(
                1 for cap in requirements.required_capabilities
                if getattr(model.capabilities, cap, False)
            ) / len(requirements.required_capabilities)
            performance_score += capability_bonus
        
        return min(1.0, performance_score)

    def _meets_requirements(self, model: Model, requirements: TaskRequirements) -> bool:
        """Check if model meets minimum requirements."""
        # Basic compatibility
        if model.provider in requirements.exclude_providers:
            return False
        
        if requirements.context_window and model.capabilities.context_window < requirements.context_window:
            return False
        
        if model.capabilities.accuracy_score < requirements.accuracy_threshold:
            return False
        
        # Required capabilities
        for capability in requirements.required_capabilities:
            if not getattr(model.capabilities, capability, False):
                return False
        
        # Language support
        if not any(model.capabilities.supports_language(lang) for lang in requirements.languages):
            return False
        
        return True


class WeightedStrategy(SelectionStrategy):
    """Strategy that allows custom weighting of different selection criteria."""

    def __init__(
        self,
        name: str = "weighted",
        task_weight: float = 0.3,
        cost_weight: float = 0.2,
        performance_weight: float = 0.3,
        capability_weight: float = 0.2,
    ):
        super().__init__(name)
        self.task_weight = task_weight
        self.cost_weight = cost_weight
        self.performance_weight = performance_weight
        self.capability_weight = capability_weight
        
        # Normalize weights
        total_weight = sum([task_weight, cost_weight, performance_weight, capability_weight])
        if total_weight > 0:
            self.task_weight /= total_weight
            self.cost_weight /= total_weight
            self.performance_weight /= total_weight
            self.capability_weight /= total_weight

    async def select_model(
        self,
        registry: ModelRegistry,
        requirements: TaskRequirements,
        available_models: Optional[List[Model]] = None,
    ) -> SelectionResult:
        """Select model using weighted combination of criteria."""
        if available_models is None:
            # Get all available models from registry
            model_info = registry.list_models()
            models = []
            for model_name, info in model_info.items():
                if "error" not in info:
                    try:
                        model = await registry.get_model(model_name, info["provider"])
                        models.append(model)
                    except Exception as e:
                        logger.warning(f"Failed to load model {model_name}: {e}")
        else:
            models = available_models

        if not models:
            raise ValueError("No models available for selection")

        # Score all models
        scored_models = []
        for model in models:
            if self._is_eligible(model, requirements):
                score = self.score_model(model, requirements)
                scored_models.append((model, score))

        if not scored_models:
            raise ValueError("No eligible models found")

        # Sort by score
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        best_model, best_score = scored_models[0]
        alternatives = [(model, best_model.provider, score) for model, score in scored_models[1:5]]

        return SelectionResult(
            model=best_model,
            provider=best_model.provider,
            confidence_score=best_score,
            selection_reason=f"Best weighted score ({best_score:.3f}) balancing task fit, cost, performance, and capabilities",
            alternatives=alternatives,
        )

    def score_model(self, model: Model, requirements: TaskRequirements) -> float:
        """Score model using weighted combination of factors."""
        # Task compatibility score
        task_score = 1.0 if model.capabilities.supports_task(requirements.task_type) else 0.5
        if requirements.preferred_domains:
            domain_overlap = set(model.capabilities.domains) & set(requirements.preferred_domains)
            if domain_overlap:
                task_score += 0.2 * (len(domain_overlap) / len(requirements.preferred_domains))
        task_score = min(1.0, task_score)

        # Cost efficiency score
        performance_score = model.capabilities.accuracy_score
        cost_efficiency = model.cost.get_cost_efficiency_score(performance_score)
        import math
        cost_score = min(1.0, math.log10(max(1, cost_efficiency)) / 2.0)
        if model.cost.is_free:
            cost_score = 1.0

        # Performance score
        accuracy_score = model.capabilities.accuracy_score
        speed_scores = {"fast": 1.0, "medium": 0.6, "slow": 0.2}
        speed_score = speed_scores.get(model.capabilities.speed_rating, 0.4)
        perf_score = 0.7 * accuracy_score + 0.3 * speed_score

        # Capability score
        capability_score = 0.0
        if requirements.context_window:
            capability_score += 0.3 * min(1.0, model.capabilities.context_window / requirements.context_window)
        else:
            capability_score += 0.3  # No specific requirement

        # Required capabilities
        if requirements.required_capabilities:
            met_capabilities = sum(
                1 for cap in requirements.required_capabilities
                if getattr(model.capabilities, cap, False)
            )
            capability_score += 0.4 * (met_capabilities / len(requirements.required_capabilities))
        else:
            capability_score += 0.4

        # Language support
        language_support = any(model.capabilities.supports_language(lang) for lang in requirements.languages)
        capability_score += 0.3 * (1.0 if language_support else 0.0)

        # Weighted final score
        final_score = (
            self.task_weight * task_score +
            self.cost_weight * cost_score +
            self.performance_weight * perf_score +
            self.capability_weight * capability_score
        )

        return final_score

    def _is_eligible(self, model: Model, requirements: TaskRequirements) -> bool:
        """Check basic eligibility criteria."""
        if model.provider in requirements.exclude_providers:
            return False
        
        if model.capabilities.accuracy_score < requirements.accuracy_threshold:
            return False
        
        # Hard requirements
        for capability in requirements.required_capabilities:
            if not getattr(model.capabilities, capability, False):
                return False
        
        return True


class FallbackStrategy(SelectionStrategy):
    """Strategy that tries multiple strategies in order until one succeeds."""

    def __init__(self, name: str = "fallback", strategies: Optional[List[SelectionStrategy]] = None):
        super().__init__(name)
        self.strategies = strategies or [
            PerformanceBasedStrategy(),
            TaskBasedStrategy(),
            CostAwareStrategy(),
            WeightedStrategy(),
        ]

    async def select_model(
        self,
        registry: ModelRegistry,
        requirements: TaskRequirements,
        available_models: Optional[List[Model]] = None,
    ) -> SelectionResult:
        """Try strategies in order until one succeeds."""
        last_error = None
        
        for strategy in self.strategies:
            try:
                result = await strategy.select_model(registry, requirements, available_models)
                # Add fallback info to selection reason
                result.selection_reason = f"[{strategy.name}] {result.selection_reason}"
                return result
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")
                last_error = e
                continue
        
        # If all strategies failed
        error_msg = f"All fallback strategies failed. Last error: {last_error}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def score_model(self, model: Model, requirements: TaskRequirements) -> float:
        """Use the first strategy's scoring method."""
        if self.strategies:
            return self.strategies[0].score_model(model, requirements)
        return 0.0


# Aliases for backward compatibility
SelectionCriteria = TaskRequirements
CostOptimizedStrategy = CostAwareStrategy
PerformanceOptimizedStrategy = PerformanceBasedStrategy
BalancedStrategy = WeightedStrategy
TaskSpecificStrategy = TaskBasedStrategy