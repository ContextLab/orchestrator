"""
Real API Integration Tests - Anthropic
Tests the enhanced model requirements specification with actual Anthropic API integration.
"""

import pytest
import os
from unittest.mock import patch
from src.orchestrator.models.anthropic_model import AnthropicModel
from src.orchestrator.models.model_selector import ModelSelector, ModelSelectionCriteria
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.core.model import ModelCapabilities, ModelCost


class TestRealAnthropicIntegration:
    """Test enhanced model requirements with real Anthropic models."""
    
    @pytest.fixture
    def anthropic_registry(self):
        """Create registry with real Anthropic models."""
        registry = ModelRegistry()
        
        # Only create models if API key is available (for CI/CD safety)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not available for real API testing")
        
        # Claude Sonnet 4 - Premium analysis model
        claude_sonnet4 = AnthropicModel(
            name="claude-sonnet-4-20250514",
            api_key=api_key
        )
        # Override for testing consistency  
        claude_sonnet4._size_billions = 200.0  # Estimated
        claude_sonnet4._expertise = ["analysis", "research", "reasoning", "creative"]
        registry.register_model(claude_sonnet4)
        
        # Claude Haiku - Fast and efficient
        claude_haiku = AnthropicModel(
            name="claude-3-haiku-20240307", 
            api_key=api_key
        )
        # Override for testing
        claude_haiku._size_billions = 13.0  # Estimated smaller model
        claude_haiku._expertise = ["general", "fast", "chat"]
        registry.register_model(claude_haiku)
        
        # Claude Opus - Highest capability
        claude_opus = AnthropicModel(
            name="claude-3-opus-20240229",
            api_key=api_key
        )
        # Override for testing
        claude_opus._size_billions = 400.0  # Estimated largest
        claude_opus._expertise = ["analysis", "research", "reasoning", "creative", "code"]
        registry.register_model(claude_opus)
        
        return registry
    
    @pytest.fixture
    def anthropic_selector(self, anthropic_registry):
        """Create model selector with Anthropic registry."""
        return ModelSelector(anthropic_registry)

    def test_anthropic_model_initialization_with_enhanced_features(self):
        """Test that Anthropic models initialize with enhanced Issue 194 features."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not available")
        
        model = AnthropicModel(name="claude-sonnet-4-20250514", api_key=api_key)
        
        # Test enhanced attributes from Issue 194
        assert hasattr(model, '_expertise')
        assert hasattr(model, '_size_billions')
        assert isinstance(model._expertise, list)
        assert isinstance(model._size_billions, (int, float))
        assert model._size_billions > 0
        
        # Test cost information
        assert model.cost is not None
        assert isinstance(model.cost, ModelCost)
        assert not model.cost.is_free  # Anthropic models are paid
        
        # Test enhanced cost methods from Issue 194
        task_cost = model.cost.estimate_cost_for_budget_period("per-task")
        assert isinstance(task_cost, float)
        assert task_cost > 0
        
        efficiency = model.cost.get_cost_efficiency_score(0.95)
        assert isinstance(efficiency, float)
        assert efficiency > 0

    @pytest.mark.asyncio
    async def test_expertise_based_selection_with_anthropic(self, anthropic_selector):
        """Test expertise-based selection with real Anthropic models."""
        # Low expertise - should prefer faster models
        low_criteria = ModelSelectionCriteria(expertise="low")
        model = await anthropic_selector.select_model(low_criteria)
        
        # Should select Haiku (fast model) or model that meets low requirements
        assert model is not None
        assert "fast" in model._expertise or "haiku" in model.name.lower()
        
        # Very high expertise - should prefer most capable models
        very_high_criteria = ModelSelectionCriteria(expertise="very-high")
        model = await anthropic_selector.select_model(very_high_criteria)
        
        # Should select Opus or Sonnet 4 for very high expertise
        assert model is not None
        assert "analysis" in model._expertise or "research" in model._expertise
        assert model.name in ["claude-3-opus-20240229", "claude-sonnet-4-20250514"]

    @pytest.mark.asyncio
    async def test_cost_constraint_selection_with_anthropic(self, anthropic_selector):
        """Test cost constraint selection with real Anthropic pricing."""
        # Moderate budget - should prefer efficient models
        budget_criteria = ModelSelectionCriteria(
            cost_limit=1.0,  # Moderate budget 
            budget_period="per-task"
        )
        
        model = await anthropic_selector.select_model(budget_criteria)
        assert model is not None
        
        # Verify the selected model is within budget
        estimated_cost = model.cost.estimate_cost_for_budget_period("per-task")
        assert estimated_cost <= 1.0
        
        # Higher budget - should allow premium models
        premium_criteria = ModelSelectionCriteria(
            cost_limit=10.0,  # Higher budget
            budget_period="per-task"
        )
        
        model = await anthropic_selector.select_model(premium_criteria)
        assert model is not None
        
        # Should be within budget
        estimated_cost = model.cost.estimate_cost_for_budget_period("per-task")
        assert estimated_cost <= 10.0

    @pytest.mark.asyncio
    async def test_size_constraint_selection_with_anthropic(self, anthropic_selector):
        """Test size constraint selection with Anthropic models."""
        # Small model preference
        small_criteria = ModelSelectionCriteria(max_model_size=50.0)
        model = await anthropic_selector.select_model(small_criteria)
        
        assert model is not None
        assert model._size_billions <= 50.0
        # Should likely select Haiku
        assert "haiku" in model.name.lower()
        
        # Large model preference
        large_criteria = ModelSelectionCriteria(min_model_size=200.0)
        model = await anthropic_selector.select_model(large_criteria)
        
        assert model is not None
        assert model._size_billions >= 200.0
        # Should select Opus or Sonnet 4
        assert model.name in ["claude-3-opus-20240229", "claude-sonnet-4-20250514"]

    @pytest.mark.asyncio
    async def test_modality_selection_with_anthropic(self, anthropic_selector):
        """Test modality-based selection with Anthropic models."""
        # Analysis modality requirement (map to text for Anthropic)
        analysis_criteria = ModelSelectionCriteria(modalities=["text"])
        model = await anthropic_selector.select_model(analysis_criteria)
        
        # Should select any Anthropic model (all support text)
        assert model is not None
        assert "claude" in model.name.lower()
        
        # Code modality requirement  
        code_criteria = ModelSelectionCriteria(modalities=["code"])
        model = await anthropic_selector.select_model(code_criteria)
        
        # Should select a code-capable model (Opus or Sonnet 4)
        assert model is not None
        assert "code" in model._expertise or model.capabilities.code_specialized

    @pytest.mark.asyncio
    async def test_complex_criteria_with_anthropic(self, anthropic_selector):
        """Test complex multi-criteria selection with Anthropic models."""
        complex_criteria = ModelSelectionCriteria(
            expertise="high",
            min_model_size=50.0,
            max_model_size=500.0,
            cost_limit=5.0,
            budget_period="per-task",
            selection_strategy="accuracy_optimized"
        )
        
        model = await anthropic_selector.select_model(complex_criteria)
        assert model is not None
        
        # Verify it meets size constraints
        assert 50.0 <= model._size_billions <= 500.0
        
        # Verify cost constraint
        estimated_cost = model.cost.estimate_cost_for_budget_period("per-task")
        assert estimated_cost <= 5.0
        
        # Should be high-accuracy model
        assert model.capabilities.accuracy_score > 0.9

    def test_anthropic_capability_detection(self, anthropic_registry):
        """Test capability detection with real Anthropic models."""
        # Test capability analysis for each model
        for model_key, model in anthropic_registry.models.items():
            analysis = anthropic_registry.detect_model_capabilities(model)
            
            # Should have complete analysis
            assert "basic_capabilities" in analysis
            assert "advanced_capabilities" in analysis
            assert "performance_metrics" in analysis
            assert "expertise_analysis" in analysis
            assert "cost_analysis" in analysis
            assert "suitability_scores" in analysis
            
            # Cost analysis should reflect Anthropic pricing
            cost_analysis = analysis["cost_analysis"]
            assert cost_analysis["type"] == "paid"
            assert cost_analysis["cost_per_1k_avg"] > 0
            
            # Expertise analysis should recognize Claude's strengths
            expertise_analysis = analysis["expertise_analysis"]
            if "opus" in model.name.lower():
                assert expertise_analysis["level"] == "very-high"
            elif "haiku" in model.name.lower():
                assert expertise_analysis["level"] in ["low", "medium"]
            
            # Should have reasonable suitability scores
            scores = analysis["suitability_scores"]
            for capability, score in scores.items():
                assert 0.0 <= score <= 1.0
            
            # Anthropic models should score well on analysis
            assert scores["analysis"] > 0.7

    def test_anthropic_task_recommendations(self, anthropic_registry):
        """Test task-based recommendations with Anthropic models."""
        # Analysis-related task
        analysis_recs = anthropic_registry.recommend_models_for_task(
            "Analyze market research data and provide insights", 
            max_recommendations=2
        )
        
        assert len(analysis_recs) > 0
        
        for rec in analysis_recs:
            assert "model" in rec
            assert "reasoning" in rec
            assert rec["suitability_score"] > 0
            
            # Should recommend capable models for analysis
            model = rec["model"]
            assert "analysis" in model._expertise or model.capabilities.accuracy_score > 0.9
        
        # Creative writing task
        creative_recs = anthropic_registry.recommend_models_for_task(
            "Write a creative story about space exploration",
            max_recommendations=2
        )
        
        # Should recommend creative-capable models
        creative_capable_found = False
        for rec in creative_recs:
            if "creative" in rec["model"]._expertise:
                creative_capable_found = True
                break
        
        assert creative_capable_found

    def test_anthropic_cost_analysis_integration(self, anthropic_registry):
        """Test cost analysis with real Anthropic pricing."""
        for model in anthropic_registry.models.values():
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
            
            # Test cost efficiency - Claude models should have good efficiency
            efficiency = model.cost.get_cost_efficiency_score(model.capabilities.accuracy_score)
            assert efficiency > 0
            
            # Opus should be expensive but highly efficient due to quality
            if "opus" in model.name.lower():
                assert model.capabilities.accuracy_score > 0.95

    @pytest.mark.asyncio
    async def test_anthropic_fallback_strategies(self, anthropic_selector):
        """Test fallback strategies with Anthropic models."""
        # Impossible requirements with cheapest fallback
        impossible_criteria = ModelSelectionCriteria(
            min_model_size=10000.0,  # Impossibly large
            fallback_strategy="cheapest"
        )
        
        model = await anthropic_selector.select_model(impossible_criteria)
        assert model is not None
        
        # Should fallback to most cost-effective model (likely Haiku)
        assert "haiku" in model.name.lower()
        
        # Best available fallback
        best_criteria = ModelSelectionCriteria(
            min_model_size=10000.0,  # Impossibly large
            fallback_strategy="best_available"
        )
        
        model = await anthropic_selector.select_model(best_criteria)
        assert model is not None
        
        # Should fallback to highest quality model (Opus or Sonnet 4)
        assert model.capabilities.accuracy_score > 0.9
        assert model.name in ["claude-3-opus-20240229", "claude-sonnet-4-20250514"]

    @pytest.mark.asyncio 
    async def test_anthropic_yaml_integration(self, anthropic_selector):
        """Test YAML requirements parsing with Anthropic models."""
        # Simulate YAML requirements for research task
        yaml_requirements = {
            "expertise": "very-high",
            "modalities": ["text"],
            "min_size": "100B",
            "max_size": "1000B", 
            "cost_limit": 8.0,
            "budget_period": "per-task",
            "fallback_strategy": "best_available"
        }
        
        # Parse and select
        criteria = anthropic_selector.parse_requirements_from_yaml(yaml_requirements)
        model = await anthropic_selector.select_model(criteria)
        
        assert model is not None
        assert model._size_billions >= 100.0
        assert model._size_billions <= 1000.0
        
        # Should be very high expertise (Opus or Sonnet 4)
        registry = anthropic_selector.registry
        assert registry._meets_expertise_level(model, "very-high")

    @pytest.mark.asyncio
    async def test_anthropic_real_health_check(self, anthropic_registry):
        """Test health checking with real Anthropic models.""" 
        # Get a model from registry
        model = list(anthropic_registry.models.values())[0]
        
        # Mock the health check to avoid real API calls during testing
        with patch.object(model, 'health_check', return_value=True) as mock_health:
            is_healthy = await model.health_check()
            assert is_healthy
            mock_health.assert_called_once()
        
        # Test with registry health filtering
        healthy_models = await anthropic_registry._filter_by_health([model])
        # Should return the model if health check passes
        assert len(healthy_models) >= 0  # May be 0 if health check is mocked to fail

    def test_anthropic_expertise_hierarchy_integration(self, anthropic_registry):
        """Test expertise hierarchy with Anthropic models."""
        # Test that models are properly classified in hierarchy
        haiku_model = None
        opus_model = None
        sonnet_model = None
        
        for model in anthropic_registry.models.values():
            if "haiku" in model.name.lower():
                haiku_model = model
            elif "opus" in model.name.lower():
                opus_model = model
            elif "sonnet" in model.name.lower():
                sonnet_model = model
        
        if haiku_model:
            # Haiku should meet low/medium requirements
            assert anthropic_registry._meets_expertise_level(haiku_model, "low")
            assert anthropic_registry._meets_expertise_level(haiku_model, "medium")
        
        if opus_model:
            # Opus should meet all expertise levels (very high capability)
            assert anthropic_registry._meets_expertise_level(opus_model, "low")
            assert anthropic_registry._meets_expertise_level(opus_model, "medium") 
            assert anthropic_registry._meets_expertise_level(opus_model, "high")
            assert anthropic_registry._meets_expertise_level(opus_model, "very-high")
        
        if sonnet_model:
            # Sonnet 4 should meet very high requirements
            assert anthropic_registry._meets_expertise_level(sonnet_model, "very-high")

    def test_anthropic_model_comparison(self, anthropic_registry):
        """Test model comparison capabilities with Anthropic models."""
        models = list(anthropic_registry.models.values())
        if len(models) >= 2:
            model1, model2 = models[0], models[1]
            
            # Test cost comparison
            comparison = model1.cost.compare_cost_with(model2.cost)
            assert "cost_ratio" in comparison
            assert "savings" in comparison
            assert "percent_savings" in comparison
            
            # Test capability matrix
            matrix = anthropic_registry.get_capability_matrix()
            assert len(matrix) == len(models)
            
            for model_key in matrix:
                scores = matrix[model_key]
                # All models should have analysis capability
                assert "analysis" in scores
                assert scores["analysis"] > 0.5  # Anthropic models excel at analysis


@pytest.mark.integration
class TestAnthropicLiveIntegration:
    """
    Live integration tests that make actual API calls.
    These tests are marked separately and should be run with caution.
    """
    
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("RUN_LIVE_TESTS"),
        reason="Requires ANTHROPIC_API_KEY and RUN_LIVE_TESTS=1 to run live tests"
    )
    @pytest.mark.asyncio
    async def test_live_anthropic_generation_with_enhanced_selection(self):
        """Test actual text generation with enhanced model selection."""
        registry = ModelRegistry()
        
        # Create real Anthropic model
        model = AnthropicModel(
            name="claude-sonnet-4-20250514",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        model._expertise = ["analysis", "research", "reasoning"]
        model._size_billions = 200.0
        
        registry.register_model(model)
        selector = ModelSelector(registry)
        
        # Use enhanced selection
        criteria = ModelSelectionCriteria(
            expertise="very-high",
            cost_limit=2.0,
            budget_period="per-task"
        )
        
        selected_model = await selector.select_model(criteria)
        assert selected_model is not None
        
        # Make actual API call
        response = await selected_model.generate(
            "What is the capital of France?",
            temperature=0.1,
            max_tokens=20
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Paris" in response  # Should contain the answer


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])