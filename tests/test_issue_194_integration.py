"""
Integration Test for Issue 194: Complex Model Requirements Specification
Comprehensive test demonstrating all enhanced functionality working together.
"""

import pytest
from unittest.mock import MagicMock
from src.orchestrator.models.model_selector import ModelSelector, ModelSelectionCriteria
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.core.model import Model, ModelCapabilities, ModelCost, ModelMetrics
from src.orchestrator.compiler.schema_validator import SchemaValidator


class TestIssue194Integration:
    """Integration tests demonstrating the complete enhanced model requirements system."""

    @pytest.fixture
    def full_registry(self):
        """Create a comprehensive registry with diverse models for testing."""
        registry = ModelRegistry()
        
        # Budget local model - fast and free
        budget_model = MagicMock(spec=Model)
        budget_model.name = "gemma3-1b"
        budget_model.provider = "ollama"
        budget_model._expertise = ["fast", "compact"]
        budget_model._size_billions = 1.0
        budget_model.meets_requirements.return_value = True
        budget_model.capabilities = ModelCapabilities(
            supported_tasks=["text-generation", "chat"],
            accuracy_score=0.72,
            speed_rating="fast",
            context_window=8192
        )
        budget_model.cost = ModelCost(is_free=True)
        budget_model.metrics = ModelMetrics(success_rate=0.85, throughput=60)
        registry.models["ollama:gemma3-1b"] = budget_model
        
        # Mid-range API model - balanced
        balanced_model = MagicMock(spec=Model)
        balanced_model.name = "gpt-4o-mini"
        balanced_model.provider = "openai"
        balanced_model._expertise = ["general", "chat"]
        balanced_model._size_billions = 8.0
        balanced_model.meets_requirements.return_value = True
        balanced_model.capabilities = ModelCapabilities(
            supported_tasks=["text-generation", "chat", "completion"],
            accuracy_score=0.87,
            speed_rating="medium",
            context_window=128000,
            supports_function_calling=True,
            supports_json_mode=True
        )
        balanced_model.cost = ModelCost(
            input_cost_per_1k_tokens=0.00015,
            output_cost_per_1k_tokens=0.0006,
            is_free=False
        )
        balanced_model.metrics = ModelMetrics(success_rate=0.92, throughput=35)
        registry.models["openai:gpt-4o-mini"] = balanced_model
        
        # Code specialist - high expertise
        code_model = MagicMock(spec=Model)
        code_model.name = "deepseek-r1-32b"
        code_model.provider = "ollama"
        code_model._expertise = ["code", "reasoning", "math"]
        code_model._size_billions = 32.0
        code_model.meets_requirements.return_value = True
        code_model.capabilities = ModelCapabilities(
            supported_tasks=["text-generation", "code-generation", "reasoning"],
            code_specialized=True,
            accuracy_score=0.93,
            speed_rating="medium",
            context_window=32768
        )
        code_model.cost = ModelCost(is_free=True)
        code_model.metrics = ModelMetrics(success_rate=0.89, throughput=25)
        registry.models["ollama:deepseek-r1-32b"] = code_model
        
        # Vision model - multimodal
        vision_model = MagicMock(spec=Model)
        vision_model.name = "gpt-4o"
        vision_model.provider = "openai"
        vision_model._expertise = ["general", "vision", "multimodal"]
        vision_model._size_billions = 200.0  # Estimated
        vision_model.meets_requirements.return_value = True
        vision_model.capabilities = ModelCapabilities(
            supported_tasks=["text-generation", "chat", "vision", "multimodal"],
            vision_capable=True,
            accuracy_score=0.91,
            speed_rating="medium",
            context_window=128000,
            supports_function_calling=True
        )
        vision_model.cost = ModelCost(
            input_cost_per_1k_tokens=0.005,
            output_cost_per_1k_tokens=0.015,
            is_free=False
        )
        vision_model.metrics = ModelMetrics(success_rate=0.94, throughput=20)
        registry.models["openai:gpt-4o"] = vision_model
        
        # Premium analysis model - very high expertise
        premium_model = MagicMock(spec=Model)
        premium_model.name = "claude-sonnet-4"
        premium_model.provider = "anthropic"
        premium_model._expertise = ["analysis", "research", "reasoning"]
        premium_model._size_billions = 200.0  # Estimated
        premium_model.meets_requirements.return_value = True
        premium_model.capabilities = ModelCapabilities(
            supported_tasks=["text-generation", "analysis", "research", "reasoning"],
            accuracy_score=0.96,
            speed_rating="slow",
            context_window=200000,
            supports_structured_output=True
        )
        premium_model.cost = ModelCost(
            input_cost_per_1k_tokens=0.003,
            output_cost_per_1k_tokens=0.015,
            base_cost_per_request=0.001,
            is_free=False
        )
        premium_model.metrics = ModelMetrics(success_rate=0.97, throughput=15)
        registry.models["anthropic:claude-sonnet-4"] = premium_model
        
        return registry

    @pytest.fixture
    def selector(self, full_registry):
        """Create model selector with full registry."""
        return ModelSelector(full_registry)

    def test_yaml_schema_validation(self):
        """Test that the enhanced YAML schema validation works correctly."""
        validator = SchemaValidator()
        
        # Valid pipeline with enhanced requires_model
        valid_pipeline = {
            "name": "Enhanced Model Requirements Test",
            "steps": [
                {
                    "id": "coding_task",
                    "action": "generate_code",
                    "requires_model": {
                        "expertise": "high",
                        "modalities": ["code"],
                        "min_size": "7B",
                        "max_size": "70B",
                        "cost_limit": 0.1,
                        "budget_period": "per-task",
                        "min_tokens_per_second": 20,
                        "preferred": ["ollama:deepseek-r1-32b"],
                        "fallback_strategy": "best_available"
                    }
                }
            ]
        }
        
        assert validator.is_valid(valid_pipeline)
        
        # Test invalid values
        invalid_pipeline = {
            "name": "Invalid Test",
            "steps": [
                {
                    "id": "invalid_task",
                    "action": "test",
                    "requires_model": {
                        "expertise": "invalid_level",  # Invalid expertise level
                        "modalities": ["invalid_modality"],  # Invalid modality
                        "budget_period": "invalid_period"  # Invalid budget period
                    }
                }
            ]
        }
        
        assert not validator.is_valid(invalid_pipeline)

    @pytest.mark.asyncio
    async def test_expertise_based_selection(self, selector):
        """Test model selection based on expertise levels."""
        # Low expertise - should prefer fast models
        low_criteria = ModelSelectionCriteria(expertise="low")
        model = await selector.select_model(low_criteria)
        assert model.name == "gemma3-1b"  # Fast, compact model
        
        # High expertise - should prefer specialized models
        high_criteria = ModelSelectionCriteria(expertise="high")
        model = await selector.select_model(high_criteria)
        assert model.name == "deepseek-r1-32b"  # Code specialist
        
        # Very high expertise - should prefer analysis models
        very_high_criteria = ModelSelectionCriteria(expertise="very-high")
        model = await selector.select_model(very_high_criteria)
        assert model.name == "claude-sonnet-4"  # Premium analysis model

    @pytest.mark.asyncio
    async def test_size_constraint_selection(self, selector):
        """Test model selection with size constraints."""
        # Small models only
        small_criteria = ModelSelectionCriteria(max_model_size=10.0)
        model = await selector.select_model(small_criteria)
        assert model._size_billions <= 10.0
        
        # Large models only
        large_criteria = ModelSelectionCriteria(min_model_size=30.0)
        model = await selector.select_model(large_criteria)
        assert model._size_billions >= 30.0
        
        # Specific size range
        range_criteria = ModelSelectionCriteria(
            min_model_size=5.0,
            max_model_size=50.0
        )
        model = await selector.select_model(range_criteria)
        assert 5.0 <= model._size_billions <= 50.0

    @pytest.mark.asyncio
    async def test_cost_constraint_selection(self, selector):
        """Test model selection with cost constraints."""
        # Very tight budget - should prefer free models
        budget_criteria = ModelSelectionCriteria(
            cost_limit=0.001,
            budget_period="per-task"
        )
        model = await selector.select_model(budget_criteria)
        assert model.cost.is_free
        
        # Higher budget - allows paid models
        higher_budget_criteria = ModelSelectionCriteria(
            cost_limit=2.0,
            budget_period="per-task"
        )
        model = await selector.select_model(higher_budget_criteria)
        # Should succeed with any model

    @pytest.mark.asyncio
    async def test_modality_selection(self, selector):
        """Test model selection based on modality requirements."""
        # Vision modality
        vision_criteria = ModelSelectionCriteria(modalities=["vision"])
        model = await selector.select_model(vision_criteria)
        assert model.capabilities.vision_capable
        
        # Code modality
        code_criteria = ModelSelectionCriteria(modalities=["code"])
        model = await selector.select_model(code_criteria)
        assert model.capabilities.code_specialized or "code" in model._expertise
        
        # Multiple modalities
        multi_criteria = ModelSelectionCriteria(modalities=["text", "code"])
        model = await selector.select_model(multi_criteria)
        # Should handle multiple requirements

    @pytest.mark.asyncio
    async def test_performance_constraints(self, selector):
        """Test model selection with performance constraints."""
        # Throughput requirement
        perf_criteria = ModelSelectionCriteria(min_tokens_per_second=30)
        model = await selector.select_model(perf_criteria)
        assert model.metrics.throughput >= 30

    @pytest.mark.asyncio
    async def test_fallback_strategies(self, selector):
        """Test different fallback strategies."""
        # Impossible requirements with cheapest fallback
        impossible_criteria = ModelSelectionCriteria(
            min_model_size=1000.0,  # No model this large
            fallback_strategy="cheapest"
        )
        model = await selector.select_model(impossible_criteria)
        assert model.cost.is_free  # Should fallback to cheapest (free)
        
        # Impossible requirements with best_available fallback
        best_criteria = ModelSelectionCriteria(
            min_model_size=1000.0,
            fallback_strategy="best_available"
        )
        model = await selector.select_model(best_criteria)
        assert model.capabilities.accuracy_score >= 0.9  # Should be high quality

    @pytest.mark.asyncio
    async def test_complex_multi_criteria_selection(self, selector):
        """Test selection with multiple complex criteria combined."""
        complex_criteria = ModelSelectionCriteria(
            expertise="high",
            modalities=["code"],
            min_model_size=10.0,
            max_model_size=100.0,
            cost_limit=1.0,
            budget_period="per-task",
            min_tokens_per_second=20,
            selection_strategy="balanced"
        )
        
        model = await selector.select_model(complex_criteria)
        
        # Verify it meets the criteria
        assert model._size_billions >= 10.0
        assert model._size_billions <= 100.0
        assert model.metrics.throughput >= 20 or model.metrics.throughput == 0  # 0 means no data
        
        # Should be code-capable
        assert model.capabilities.code_specialized or "code" in model._expertise

    def test_cost_analysis_integration(self, full_registry):
        """Test integration of enhanced cost analysis methods."""
        # Test cost efficiency scoring
        for model in full_registry.models.values():
            performance_score = model.capabilities.accuracy_score
            efficiency = model.cost.get_cost_efficiency_score(performance_score)
            assert efficiency >= 0.0
            
            # Free models should have maximum efficiency
            if model.cost.is_free:
                assert efficiency == 100.0
        
        # Test budget period estimates
        paid_models = [m for m in full_registry.models.values() if not m.cost.is_free]
        if paid_models:
            model = paid_models[0]
            
            task_cost = model.cost.estimate_cost_for_budget_period("per-task")
            pipeline_cost = model.cost.estimate_cost_for_budget_period("per-pipeline")
            hour_cost = model.cost.estimate_cost_for_budget_period("per-hour")
            
            # Costs should increase with usage
            assert task_cost <= pipeline_cost <= hour_cost

    def test_capability_detection_integration(self, full_registry):
        """Test integration of capability detection methods."""
        # Test capability analysis for all models
        for model_key, model in full_registry.models.items():
            analysis = full_registry.detect_model_capabilities(model)
            
            # Should have all analysis components
            assert "basic_capabilities" in analysis
            assert "advanced_capabilities" in analysis
            assert "performance_metrics" in analysis
            assert "expertise_analysis" in analysis
            assert "cost_analysis" in analysis
            assert "suitability_scores" in analysis
            
            # Suitability scores should be valid
            for capability, score in analysis["suitability_scores"].items():
                assert 0.0 <= score <= 1.0
        
        # Test model recommendations
        recommendations = full_registry.recommend_models_for_task(
            "Write Python code to analyze data", 
            max_recommendations=3
        )
        
        assert len(recommendations) <= 3
        for rec in recommendations:
            assert "reasoning" in rec
            assert rec["suitability_score"] > 0

    def test_yaml_integration_with_selector(self, selector):
        """Test YAML parsing integration with model selector."""
        # Simulate YAML requires_model section
        yaml_requirements = {
            "expertise": "high",
            "modalities": ["code"],
            "min_size": "7B",
            "max_size": "70B",
            "cost_limit": 0.1,
            "budget_period": "per-task",
            "fallback_strategy": "best_available"
        }
        
        # Parse requirements
        criteria = selector.parse_requirements_from_yaml(yaml_requirements)
        
        # Verify parsing
        assert criteria.expertise == "high"
        assert "code" in criteria.modalities
        assert criteria.min_model_size == 7.0
        assert criteria.max_model_size == 70.0
        assert criteria.cost_limit == 0.1
        assert criteria.budget_period == "per-task"
        assert criteria.fallback_strategy == "best_available"

    @pytest.mark.asyncio
    async def test_progressive_requirements_relaxation(self, selector):
        """Test progressive relaxation of requirements during fallback."""
        # Create requirements that start strict and should be relaxed
        strict_criteria = ModelSelectionCriteria(
            expertise="very-high",
            min_model_size=500.0,  # Impossibly large
            cost_limit=0.0001,     # Impossibly cheap
            min_tokens_per_second=1000,  # Impossibly fast
            fallback_strategy="best_available"
        )
        
        # Should still find a model through progressive relaxation
        model = await selector.select_model(strict_criteria)
        assert model is not None
        # The fallback system should have relaxed requirements to find a suitable model

    @pytest.mark.asyncio
    async def test_end_to_end_scenario(self, full_registry, selector):
        """Test complete end-to-end scenario demonstrating all functionality."""
        # Scenario: User wants to develop a Python web application with specific constraints
        
        # 1. Get model recommendations for the task
        recommendations = full_registry.recommend_models_for_task(
            "Help me develop and debug Python web application code",
            max_recommendations=3
        )
        
        assert len(recommendations) > 0
        
        # 2. Analyze capabilities of recommended models
        for rec in recommendations:
            model = rec["model"]
            analysis = full_registry.detect_model_capabilities(model)
            
            # Should be suitable for coding
            assert analysis["suitability_scores"]["coding"] >= 0.5
        
        # 3. Create specific criteria based on project needs
        project_criteria = ModelSelectionCriteria(
            expertise="high",           # Need good coding skills
            modalities=["code"],        # Code generation required
            min_model_size=7.0,         # Need reasonable model size
            cost_limit=0.5,             # Budget constraint
            budget_period="per-task",   # Per-task budget
            min_tokens_per_second=20,   # Performance requirement
            selection_strategy="balanced"  # Balanced approach
        )
        
        # 4. Select optimal model
        selected_model = await selector.select_model(project_criteria)
        
        # 5. Verify selection meets requirements
        assert selected_model._size_billions >= 7.0
        assert selected_model.capabilities.code_specialized or "code" in selected_model._expertise
        
        # 6. Analyze cost implications
        task_cost = selected_model.cost.estimate_cost_for_budget_period("per-task")
        assert task_cost <= 0.5  # Within budget
        
        # 7. Get cost breakdown for planning
        breakdown = selected_model.cost.get_cost_breakdown(1000, 500)  # Typical usage
        assert "total_cost" in breakdown
        
        print(f"Selected model: {selected_model.provider}:{selected_model.name}")
        print(f"Estimated task cost: ${task_cost:.4f}")
        print(f"Model expertise: {selected_model._expertise}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])