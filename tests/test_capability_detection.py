"""
Test Category: Capability Detection Methods
Real tests for model capability detection and analysis functionality.
"""

import pytest
from unittest.mock import MagicMock
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.core.model import Model, ModelCapabilities, ModelCost, ModelMetrics

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider


class TestCapabilityDetection:
    """Test model capability detection and analysis methods."""

    @pytest.fixture
    def registry(self):
        """Create registry with various test models."""
        registry = ModelRegistry()
        
        # Free coding model
        code_model = MagicMock(spec=Model)
        code_model.name = "deepseek-coder"
        code_model.provider = "ollama"
        code_model._expertise = ["code", "reasoning", "programming"]
        code_model.capabilities = ModelCapabilities(
            supported_tasks=["code-generation", "text-generation"],
            code_specialized=True,
            accuracy_score=0.92,
            speed_rating="medium",
            context_window=16384
        )
        code_model.cost = ModelCost(is_free=True)
        code_model.metrics = ModelMetrics()
        registry.models["ollama:deepseek-coder"] = code_model
        
        # Premium analysis model
        analysis_model = MagicMock(spec=Model)
        analysis_model.name = "claude-sonnet-4"
        analysis_model.provider = "anthropic"
        analysis_model._expertise = ["analysis", "research", "reasoning"]
        analysis_model.capabilities = ModelCapabilities(
            supported_tasks=["text-generation", "analysis", "research"],
            accuracy_score=0.95,
            speed_rating="slow",
            context_window=200000
        )
        analysis_model.cost = ModelCost(
            input_cost_per_1k_tokens=0.003,
            output_cost_per_1k_tokens=0.015,
            is_free=False
        )
        analysis_model.metrics = ModelMetrics()
        registry.models["anthropic:claude-sonnet-4"] = analysis_model
        
        # Vision model
        vision_model = MagicMock(spec=Model)
        vision_model.name = "gpt-4o"
        vision_model.provider = "openai"
        vision_model._expertise = ["general", "vision"]
        vision_model.capabilities = ModelCapabilities(
            supported_tasks=["text-generation", "chat", "vision"],
            vision_capable=True,
            accuracy_score=0.88,
            speed_rating="medium",
            context_window=128000
        )
        vision_model.cost = ModelCost(
            input_cost_per_1k_tokens=0.005,
            output_cost_per_1k_tokens=0.015,
            is_free=False
        )
        vision_model.metrics = ModelMetrics()
        registry.models["openai:gpt-4o"] = vision_model
        
        # Fast budget model
        fast_model = MagicMock(spec=Model)
        fast_model.name = "gemma3-1b"
        fast_model.provider = "ollama"
        fast_model._expertise = ["fast", "compact"]
        fast_model.capabilities = ModelCapabilities(
            supported_tasks=["text-generation", "chat"],
            accuracy_score=0.75,
            speed_rating="fast",
            context_window=8192
        )
        fast_model.cost = ModelCost(is_free=True)
        fast_model.metrics = ModelMetrics()
        registry.models["ollama:gemma3-1b"] = fast_model
        
        return registry

    def test_detect_model_capabilities_basic(self, registry):
        """Test basic capability detection functionality."""
        # Test code model
        code_model = registry.models["ollama:deepseek-coder"]
        analysis = registry.detect_model_capabilities(code_model)
        
        # Check structure
        assert "basic_capabilities" in analysis
        assert "advanced_capabilities" in analysis
        assert "performance_metrics" in analysis
        assert "expertise_analysis" in analysis
        assert "cost_analysis" in analysis
        assert "suitability_scores" in analysis
        
        # Check basic capabilities
        basic = analysis["basic_capabilities"]
        assert basic["text_generation"] == True  # "text-generation" is in supported_tasks
        
        # Check advanced capabilities
        advanced = analysis["advanced_capabilities"]
        assert advanced["code"] == True
        assert advanced["vision"] == False
        
        # Check performance metrics
        perf = analysis["performance_metrics"]
        assert perf["accuracy_score"] == 0.92
        assert perf["speed_rating"] == "medium"

    def test_analyze_model_expertise(self, registry):
        """Test expertise analysis functionality."""
        # Test high expertise code model
        code_model = registry.models["ollama:deepseek-coder"]
        analysis = registry.detect_model_capabilities(code_model)
        expertise = analysis["expertise_analysis"]
        
        assert expertise["level"] == "high"  # Should be high due to "code" in expertise
        assert "code" in expertise["areas"]
        assert expertise["categories"]["coding"] == True
        assert expertise["categories"]["general"] == False
        assert expertise["specialization_score"] > 0
        
        # Test very high expertise analysis model
        analysis_model = registry.models["anthropic:claude-sonnet-4"]
        analysis = registry.detect_model_capabilities(analysis_model)
        expertise = analysis["expertise_analysis"]
        
        assert expertise["level"] == "very-high"  # Should be very-high due to "analysis"
        assert "analysis" in expertise["areas"]
        assert expertise["categories"]["analysis"] == True
        
        # Test low expertise fast model
        fast_model = registry.models["ollama:gemma3-1b"]
        analysis = registry.detect_model_capabilities(fast_model)
        expertise = analysis["expertise_analysis"]
        
        assert expertise["level"] == "low"  # Should be low due to "fast", "compact"

    def test_analyze_model_cost(self, registry):
        """Test cost analysis functionality."""
        # Test free model
        free_model = registry.models["ollama:deepseek-coder"]
        analysis = registry.detect_model_capabilities(free_model)
        cost_analysis = analysis["cost_analysis"]
        
        assert cost_analysis["type"] == "free"
        assert cost_analysis["cost_tier"] == "free"
        assert cost_analysis["efficiency_score"] == 100.0
        assert cost_analysis["budget_friendly"] == True
        assert cost_analysis["cost_per_1k_avg"] == 0.0
        
        # Test paid model
        paid_model = registry.models["anthropic:claude-sonnet-4"]
        analysis = registry.detect_model_capabilities(paid_model)
        cost_analysis = analysis["cost_analysis"]
        
        assert cost_analysis["type"] == "paid"
        assert cost_analysis["cost_tier"] in ["low", "medium", "high", "very_high"]  # Should be categorized
        assert cost_analysis["efficiency_score"] > 0
        assert cost_analysis["cost_per_1k_avg"] > 0
        assert "estimated_task_cost" in cost_analysis

    def test_calculate_suitability_scores(self, registry):
        """Test suitability score calculations."""
        # Test code model suitability
        code_model = registry.models["ollama:deepseek-coder"]
        analysis = registry.detect_model_capabilities(code_model)
        scores = analysis["suitability_scores"]
        
        # Should be highly suitable for coding
        assert scores["coding"] > 0.8
        assert scores["budget_constrained"] == 1.0  # Free model
        
        # Vision model should be suitable for vision tasks
        vision_model = registry.models["openai:gpt-4o"]
        analysis = registry.detect_model_capabilities(vision_model)
        scores = analysis["suitability_scores"]
        
        assert scores["vision"] > 0.8
        assert scores["coding"] < scores["vision"]  # Should prefer vision over coding
        
        # Fast model should be good for speed-critical tasks
        fast_model = registry.models["ollama:gemma3-1b"]
        analysis = registry.detect_model_capabilities(fast_model)
        scores = analysis["suitability_scores"]
        
        assert scores["speed_critical"] == 1.0  # Fast speed rating
        assert scores["budget_constrained"] == 1.0  # Free model

    def test_find_models_by_capability(self, registry):
        """Test finding models by specific capabilities."""
        # Find coding models
        coding_models = registry.find_models_by_capability("coding", threshold=0.7)
        assert len(coding_models) > 0
        
        # The code specialist should be in the results
        model_names = [model.name for model in coding_models]
        assert "deepseek-coder" in model_names
        
        # Find vision models
        vision_models = registry.find_models_by_capability("vision", threshold=0.7)
        assert len(vision_models) > 0
        
        vision_names = [model.name for model in vision_models]
        assert "gpt-4o" in vision_names
        
        # Find budget-constrained models
        budget_models = registry.find_models_by_capability("budget_constrained", threshold=0.9)
        assert len(budget_models) >= 2  # Should include both free models
        
        # Test with high threshold that excludes most models
        exclusive_models = registry.find_models_by_capability("analysis", threshold=0.95)
        # Should be empty or very few models

    def test_get_capability_matrix(self, registry):
        """Test capability matrix generation."""
        matrix = registry.get_capability_matrix()
        
        # Should have entries for all models
        assert len(matrix) == len(registry.models)
        
        # Check that all models have capability scores
        for model_key in registry.models.keys():
            assert model_key in matrix
            scores = matrix[model_key]
            
            # Should have standard capability categories
            expected_capabilities = ["coding", "analysis", "creative", "chat", "vision", "speed_critical", "budget_constrained"]
            for capability in expected_capabilities:
                assert capability in scores
                assert 0.0 <= scores[capability] <= 1.0

    def test_recommend_models_for_task(self, registry):
        """Test task-based model recommendations."""
        # Test coding task
        coding_recommendations = registry.recommend_models_for_task("Debug Python code", max_recommendations=2)
        
        assert len(coding_recommendations) <= 2
        for rec in coding_recommendations:
            assert "model" in rec
            assert "capability" in rec
            assert "suitability_score" in rec
            assert "reasoning" in rec
            assert "cost_analysis" in rec
            assert "expertise_level" in rec
            
            # Should be recommended for coding capability
            assert rec["capability"] == "coding"
            assert rec["suitability_score"] > 0
        
        # Test analysis task
        analysis_recommendations = registry.recommend_models_for_task("Analyze market research data", max_recommendations=3)
        
        # Should recommend models suitable for analysis
        for rec in analysis_recommendations:
            assert rec["capability"] == "analysis"
        
        # Test vision task
        vision_recommendations = registry.recommend_models_for_task("Describe what's in this image", max_recommendations=2)
        
        # Should recommend vision-capable models
        for rec in vision_recommendations:
            assert rec["capability"] == "vision"
        
        # Test general task (should default to chat)
        general_recommendations = registry.recommend_models_for_task("Help me with something", max_recommendations=2)
        
        # Should default to chat capability
        for rec in general_recommendations:
            assert rec["capability"] == "chat"

    def test_recommendation_reasoning(self, registry):
        """Test that recommendation reasoning is informative."""
        recommendations = registry.recommend_models_for_task("Debug a Python program", max_recommendations=3)
        
        for rec in recommendations:
            reasoning = rec["reasoning"]
            assert isinstance(reasoning, str)
            assert len(reasoning) > 10  # Should be a meaningful explanation
            assert "Recommended because" in reasoning
            
            # Should mention relevant factors
            if rec["cost_analysis"]["type"] == "free":
                assert "free" in reasoning.lower()
            if rec["model"].capabilities.code_specialized:
                assert "code" in reasoning.lower()

    def test_edge_cases_and_empty_registry(self):
        """Test edge cases and empty registry scenarios."""
        empty_registry = ModelRegistry()
        
        # Empty registry should return empty results
        assert empty_registry.find_models_by_capability("coding") == []
        assert empty_registry.get_capability_matrix() == {}
        assert empty_registry.recommend_models_for_task("test task") == []
        
        # Test with model that has minimal/default values
        minimal_model = MagicMock(spec=Model)
        minimal_model.name = "minimal"
        minimal_model.provider = "test"
        minimal_model._expertise = None
        minimal_model.capabilities = ModelCapabilities(supported_tasks=["text-generation"])
        minimal_model.cost = ModelCost()
        minimal_model.metrics = ModelMetrics()
        
        empty_registry.models["test:minimal"] = minimal_model
        
        # Should still work with default values
        analysis = empty_registry.detect_model_capabilities(minimal_model)
        assert analysis["expertise_analysis"]["level"] == "medium"  # Default level
        assert analysis["cost_analysis"]["type"] == "paid"  # Default ModelCost() is paid with zero costs

    def test_capability_scoring_consistency(self, registry):
        """Test that capability scoring is consistent and logical."""
        for model_key, model in registry.models.items():
            analysis = registry.detect_model_capabilities(model)
            scores = analysis["suitability_scores"]
            
            # All scores should be between 0 and 1
            for capability, score in scores.items():
                assert 0.0 <= score <= 1.0, f"Invalid score {score} for {capability} in {model_key}"
            
            # Free models should have maximum budget score
            if model.cost.is_free:
                assert scores["budget_constrained"] == 1.0
            
            # Code-specialized models should have higher coding scores
            if model.capabilities.code_specialized:
                assert scores["coding"] > 0.7
            
            # Vision-capable models should have high vision scores
            if model.capabilities.vision_capable:
                assert scores["vision"] > 0.7
            
            # Fast models should have high speed scores
            if model.capabilities.speed_rating == "fast":
                assert scores["speed_critical"] > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])