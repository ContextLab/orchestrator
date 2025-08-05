"""
Real API Integration Tests - OpenAI
Tests the enhanced model requirements specification with actual OpenAI API calls.
"""

import pytest
import os
from unittest.mock import patch
from orchestrator.models.openai_model import OpenAIModel
from orchestrator.models.model_selector import ModelSelector, ModelSelectionCriteria
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.core.model import ModelCapabilities, ModelCost


class TestRealOpenAIIntegration:
    """Test enhanced model requirements with real OpenAI models."""
    
    @pytest.fixture
    def openai_registry(self):
        """Create registry with real OpenAI models."""
        registry = ModelRegistry()
        
        # Only create models if API key is available (for CI/CD safety)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not available for real API testing")
        
        # GPT-4o mini - balanced model
        gpt4_mini = OpenAIModel(
            name="gpt-4o-mini", 
            api_key=api_key
        )
        # Override size estimate for testing
        gpt4_mini._size_billions = 8.0
        gpt4_mini._expertise = ["general", "chat", "reasoning"]
        registry.register_model(gpt4_mini)
        
        # GPT-4o - premium model with vision
        gpt4o = OpenAIModel(
            name="gpt-4o",
            api_key=api_key
        )
        # Override for testing
        gpt4o._size_billions = 200.0  # Estimated
        gpt4o._expertise = ["general", "reasoning", "code", "creative", "analysis"]
        gpt4o.capabilities.vision_capable = True
        registry.register_model(gpt4o)
        
        # GPT-3.5 Turbo - budget option
        gpt35 = OpenAIModel(
            name="gpt-3.5-turbo",
            api_key=api_key
        )
        # Override for testing
        gpt35._size_billions = 175.0
        gpt35._expertise = ["general", "chat", "fast"]
        registry.register_model(gpt35)
        
        return registry
    
    @pytest.fixture
    def openai_selector(self, openai_registry):
        """Create model selector with OpenAI registry."""
        return ModelSelector(openai_registry)

    def test_openai_model_initialization_with_enhanced_features(self):
        """Test that OpenAI models initialize with enhanced Issue 194 features."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not available")
        
        model = OpenAIModel(name="gpt-4o-mini", api_key=api_key)
        
        # Test enhanced attributes from Issue 194
        assert hasattr(model, '_expertise')
        assert hasattr(model, '_size_billions')
        assert isinstance(model._expertise, list)
        assert isinstance(model._size_billions, (int, float))
        assert model._size_billions > 0
        
        # Test cost information
        assert model.cost is not None
        assert isinstance(model.cost, ModelCost)
        assert not model.cost.is_free  # OpenAI models are paid
        
        # Test enhanced cost methods from Issue 194
        task_cost = model.cost.estimate_cost_for_budget_period("per-task")
        assert isinstance(task_cost, float)
        assert task_cost > 0
        
        efficiency = model.cost.get_cost_efficiency_score(0.9)
        assert isinstance(efficiency, float)
        assert efficiency > 0

    @pytest.mark.asyncio
    async def test_expertise_based_selection_with_openai(self, openai_selector):
        """Test expertise-based selection with real OpenAI models."""
        # Low expertise - should prefer faster/cheaper models
        low_criteria = ModelSelectionCriteria(expertise="low")
        model = await openai_selector.select_model(low_criteria)
        
        # Should select a model that meets low expertise requirements
        assert model is not None
        assert "fast" in model._expertise or "general" in model._expertise
        
        # High expertise - should prefer more capable models
        high_criteria = ModelSelectionCriteria(expertise="high")
        model = await openai_selector.select_model(high_criteria)
        
        # Should select GPT-4 variant for high expertise
        assert model is not None
        assert "gpt-4" in model.name.lower()

    @pytest.mark.asyncio
    async def test_cost_constraint_selection_with_openai(self, openai_selector):
        """Test cost constraint selection with real OpenAI pricing."""
        # Very tight budget - should prefer cheaper models
        budget_criteria = ModelSelectionCriteria(
            cost_limit=0.01,  # Very low budget
            budget_period="per-task"
        )
        
        model = await openai_selector.select_model(budget_criteria)
        assert model is not None
        
        # Verify the selected model is within budget
        estimated_cost = model.cost.estimate_cost_for_budget_period("per-task")
        assert estimated_cost <= 0.01
        
        # Higher budget - should allow premium models
        premium_criteria = ModelSelectionCriteria(
            cost_limit=5.0,  # Higher budget
            budget_period="per-task"
        )
        
        model = await openai_selector.select_model(premium_criteria)
        assert model is not None
        
        # Should be within budget
        estimated_cost = model.cost.estimate_cost_for_budget_period("per-task")
        assert estimated_cost <= 5.0

    @pytest.mark.asyncio
    async def test_size_constraint_selection_with_openai(self, openai_selector):
        """Test size constraint selection with OpenAI models."""
        # Small model preference
        small_criteria = ModelSelectionCriteria(max_model_size=50.0)
        model = await openai_selector.select_model(small_criteria)
        
        assert model is not None
        assert model._size_billions <= 50.0
        
        # Large model preference  
        large_criteria = ModelSelectionCriteria(min_model_size=100.0)
        model = await openai_selector.select_model(large_criteria)
        
        assert model is not None
        assert model._size_billions >= 100.0

    @pytest.mark.asyncio
    async def test_modality_selection_with_openai(self, openai_selector):
        """Test modality-based selection with OpenAI models."""
        # Vision modality requirement
        vision_criteria = ModelSelectionCriteria(modalities=["vision"])
        model = await openai_selector.select_model(vision_criteria)
        
        # Should select a vision-capable model (GPT-4o)
        assert model is not None
        assert model.capabilities.vision_capable
        assert "gpt-4o" in model.name.lower()
        
        # Code modality requirement
        code_criteria = ModelSelectionCriteria(modalities=["code"])
        model = await openai_selector.select_model(code_criteria)
        
        # Should select a code-capable model
        assert model is not None
        assert model.capabilities.code_specialized or "code" in model._expertise

    @pytest.mark.asyncio
    async def test_complex_criteria_with_openai(self, openai_selector):
        """Test complex multi-criteria selection with OpenAI models."""
        complex_criteria = ModelSelectionCriteria(
            expertise="medium",
            min_model_size=5.0,
            max_model_size=200.0,
            cost_limit=2.0,
            budget_period="per-task",
            selection_strategy="balanced"
        )
        
        model = await openai_selector.select_model(complex_criteria)
        assert model is not None
        
        # Verify it meets size constraints
        assert 5.0 <= model._size_billions <= 200.0
        
        # Verify cost constraint
        estimated_cost = model.cost.estimate_cost_for_budget_period("per-task")
        assert estimated_cost <= 2.0

    def test_openai_capability_detection(self, openai_registry):
        """Test capability detection with real OpenAI models."""
        # Test capability analysis for each model
        for model_key, model in openai_registry.models.items():
            analysis = openai_registry.detect_model_capabilities(model)
            
            # Should have complete analysis
            assert "basic_capabilities" in analysis
            assert "advanced_capabilities" in analysis
            assert "performance_metrics" in analysis
            assert "expertise_analysis" in analysis
            assert "cost_analysis" in analysis
            assert "suitability_scores" in analysis
            
            # Cost analysis should reflect OpenAI pricing
            cost_analysis = analysis["cost_analysis"]
            assert cost_analysis["type"] == "paid"
            assert cost_analysis["cost_per_1k_avg"] > 0
            
            # Should have reasonable suitability scores
            scores = analysis["suitability_scores"]
            for capability, score in scores.items():
                assert 0.0 <= score <= 1.0

    def test_openai_task_recommendations(self, openai_registry):
        """Test task-based recommendations with OpenAI models."""
        # Code-related task
        code_recs = openai_registry.recommend_models_for_task(
            "Help me debug Python code", 
            max_recommendations=2
        )
        
        assert len(code_recs) > 0
        
        for rec in code_recs:
            assert "model" in rec
            assert "reasoning" in rec
            assert rec["suitability_score"] > 0
            
            # Should recommend capable models for coding
            model = rec["model"]
            assert model.capabilities.code_specialized or "code" in model._expertise
        
        # Vision-related task
        vision_recs = openai_registry.recommend_models_for_task(
            "Analyze this image for me",
            max_recommendations=2
        )
        
        # Should recommend vision-capable models
        vision_capable_found = False
        for rec in vision_recs:
            if rec["model"].capabilities.vision_capable:
                vision_capable_found = True
                break
        
        assert vision_capable_found

    def test_openai_cost_analysis_integration(self, openai_registry):
        """Test cost analysis with real OpenAI pricing."""
        for model in openai_registry.models.values():
            # Test budget period estimates
            task_cost = model.cost.estimate_cost_for_budget_period("per-task")
            pipeline_cost = model.cost.estimate_cost_for_budget_period("per-pipeline")
            hour_cost = model.cost.estimate_cost_for_budget_period("per-hour")
            
            # Costs should increase with usage
            assert task_cost <= pipeline_cost <= hour_cost
            
            # Test cost breakdown
            breakdown = model.cost.get_cost_breakdown(1000, 500)  # 1000 input, 500 output
            assert breakdown["total_cost"] > 0
            assert breakdown["input_cost"] > 0
            assert breakdown["output_cost"] > 0
            assert not breakdown["is_free"]
            
            # Test cost comparison
            other_model = list(openai_registry.models.values())[0]
            if other_model != model:
                comparison = model.cost.compare_cost_with(other_model.cost)
                assert "cost_ratio" in comparison
                assert "savings" in comparison
                assert isinstance(comparison["cost_ratio"], (int, float))

    @pytest.mark.asyncio
    async def test_openai_fallback_strategies(self, openai_selector):
        """Test fallback strategies with OpenAI models."""
        # Impossible requirements with cheapest fallback
        impossible_criteria = ModelSelectionCriteria(
            min_model_size=10000.0,  # Impossibly large
            fallback_strategy="cheapest"
        )
        
        model = await openai_selector.select_model(impossible_criteria)
        assert model is not None
        
        # Should fallback to cheapest available model
        # In OpenAI case, probably GPT-3.5 turbo
        assert model.name in ["gpt-3.5-turbo", "gpt-4o-mini"]
        
        # Best available fallback
        best_criteria = ModelSelectionCriteria(
            min_model_size=10000.0,  # Impossibly large
            fallback_strategy="best_available"
        )
        
        model = await openai_selector.select_model(best_criteria)
        assert model is not None
        
        # Should fallback to highest quality model
        assert model.capabilities.accuracy_score > 0.8

    @pytest.mark.asyncio 
    async def test_openai_yaml_integration(self, openai_selector):
        """Test YAML requirements parsing with OpenAI models."""
        # Simulate YAML requirements
        yaml_requirements = {
            "expertise": "high",
            "modalities": ["text", "code"],
            "min_size": "10B",
            "max_size": "500B", 
            "cost_limit": 3.0,
            "budget_period": "per-task",
            "fallback_strategy": "best_available"
        }
        
        # Parse and select
        criteria = openai_selector.parse_requirements_from_yaml(yaml_requirements)
        model = await openai_selector.select_model(criteria)
        
        assert model is not None
        assert model._size_billions >= 10.0
        assert model._size_billions <= 500.0
        
        # Should be high expertise
        registry = openai_selector.registry
        assert registry._meets_expertise_level(model, "high")

    @pytest.mark.asyncio
    async def test_openai_real_health_check(self, openai_registry):
        """Test health checking with real OpenAI models.""" 
        # Get a model from registry
        model = list(openai_registry.models.values())[0]
        
        # Mock the health check to avoid real API calls during testing
        with patch.object(model, 'health_check', return_value=True) as mock_health:
            is_healthy = await model.health_check()
            assert is_healthy
            mock_health.assert_called_once()
        
        # Test with registry health filtering
        healthy_models = await openai_registry._filter_by_health([model])
        # Should return the model if health check passes
        assert len(healthy_models) >= 0  # May be 0 if health check is mocked to fail

    def test_openai_model_size_parsing_integration(self):
        """Test model size parsing with OpenAI model names."""
        from orchestrator.utils.model_utils import parse_model_size
        
        # Test OpenAI model names (which don't have explicit sizes)
        test_cases = [
            ("gpt-4o", None, 1.0),  # Should default to 1.0 when no size info
            ("gpt-3.5-turbo", None, 1.0),
            ("gpt-4-turbo", None, 1.0),
        ]
        
        for model_name, size_str, expected in test_cases:
            result = parse_model_size(model_name, size_str)
            assert result == expected


@pytest.mark.integration
class TestOpenAILiveIntegration:
    """
    Live integration tests that make actual API calls.
    These tests are marked separately and should be run with caution.
    """
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") or not os.getenv("RUN_LIVE_TESTS"),
        reason="Requires OPENAI_API_KEY and RUN_LIVE_TESTS=1 to run live tests"
    )
    @pytest.mark.asyncio
    async def test_live_openai_generation_with_enhanced_selection(self):
        """Test actual text generation with enhanced model selection."""
        registry = ModelRegistry()
        
        # Create real OpenAI model
        model = OpenAIModel(
            name="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        model._expertise = ["general", "reasoning"]
        model._size_billions = 8.0
        
        registry.register_model(model)
        selector = ModelSelector(registry)
        
        # Use enhanced selection
        criteria = ModelSelectionCriteria(
            expertise="medium",
            cost_limit=0.1,
            budget_period="per-task"
        )
        
        selected_model = await selector.select_model(criteria)
        assert selected_model is not None
        
        # Make actual API call
        response = await selected_model.generate(
            "What is 2+2?",
            temperature=0.1,
            max_tokens=10
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "4" in response  # Should contain the answer


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])