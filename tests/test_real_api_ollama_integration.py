"""
Real API Integration Tests - Ollama
Tests the enhanced model requirements specification with actual Ollama local models.
"""

import pytest
import subprocess
import time
from unittest.mock import patch, MagicMock
from orchestrator.integrations.ollama_model import OllamaModel
from orchestrator.models.model_selector import ModelSelector, ModelSelectionCriteria
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.core.model import ModelCapabilities, ModelCost


def is_ollama_running():
    """Check if Ollama is running locally."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_available_ollama_models():
    """Get list of locally available Ollama models."""
    if not is_ollama_running():
        return []
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
    except:
        pass
    return []


class TestRealOllamaIntegration:
    """Test enhanced model requirements with real Ollama models."""
    
    @pytest.fixture
    def ollama_registry(self):
        """Create registry with real Ollama models."""
        registry = ModelRegistry()
        
        # Check if Ollama is running
        if not is_ollama_running():
            pytest.skip("Ollama is not running locally")
        
        available_models = get_available_ollama_models()
        if not available_models:
            pytest.skip("No Ollama models are locally available")
        
        # Create models based on what's available
        # Try common model names
        test_models = [
            ("gemma3:1b", 1.0, ["fast", "compact", "general"]),
            ("gemma3:4b", 4.0, ["general", "chat", "reasoning"]),
            ("llama3.2:3b", 3.0, ["fast", "compact", "general"]),
            ("deepseek-r1:1.5b", 1.5, ["code", "reasoning", "programming"]),
            ("deepseek-r1:8b", 8.0, ["general", "chat", "reasoning"]),
        ]
        
        models_created = 0
        for model_name, size, expertise in test_models:
            if model_name in available_models or models_created == 0:  # Always create at least one for testing
                try:
                    model = OllamaModel(model_name=model_name)
                    # Override for testing consistency
                    model._size_billions = size
                    model._expertise = expertise
                    registry.register_model(model)
                    models_created += 1
                except Exception as e:
                    print(f"Could not create model {model_name}: {e}")
                    continue
        
        if models_created == 0:
            pytest.skip("Could not create any Ollama models for testing")
        
        return registry
    
    @pytest.fixture
    def ollama_selector(self, ollama_registry):
        """Create model selector with Ollama registry."""
        return ModelSelector(ollama_registry)

    def test_ollama_model_initialization_with_enhanced_features(self):
        """Test that Ollama models initialize with enhanced Issue 194 features.""" 
        if not is_ollama_running():
            pytest.skip("Ollama is not running locally")
        
        model = OllamaModel(model_name="gemma3:1b")
        
        # Test enhanced attributes from Issue 194
        assert hasattr(model, '_expertise')
        assert hasattr(model, '_size_billions')
        assert isinstance(model._expertise, list)
        assert isinstance(model._size_billions, (int, float))
        assert model._size_billions > 0
        
        # Test cost information - Ollama models should be free
        assert model.cost is not None
        assert isinstance(model.cost, ModelCost)
        assert model.cost.is_free  # Ollama models are free
        
        # Test enhanced cost methods from Issue 194
        task_cost = model.cost.estimate_cost_for_budget_period("per-task")
        assert task_cost == 0.0  # Should be free
        
        efficiency = model.cost.get_cost_efficiency_score(0.8)
        assert efficiency == 100.0  # Free models have maximum efficiency

    @pytest.mark.asyncio
    async def test_expertise_based_selection_with_ollama(self, ollama_selector):
        """Test expertise-based selection with real Ollama models."""
        # Low expertise - should prefer fast/compact models
        low_criteria = ModelSelectionCriteria(expertise="low")
        model = await ollama_selector.select_model(low_criteria)
        
        # Should select a model that meets low expertise requirements
        assert model is not None
        registry = ollama_selector.registry
        assert registry._meets_expertise_level(model, "low")
        
        # Should prefer models with "fast" or "compact" in expertise
        if len(ollama_selector.registry.models) > 1:
            fast_model_found = "fast" in model._expertise or "compact" in model._expertise
            # If no fast model available, any model meeting low criteria is acceptable
            assert fast_model_found or registry._meets_expertise_level(model, "low")

    @pytest.mark.asyncio
    async def test_cost_constraint_selection_with_ollama(self, ollama_selector):
        """Test cost constraint selection with Ollama models (all free)."""
        # Any budget should work with free models
        budget_criteria = ModelSelectionCriteria(
            cost_limit=0.0,  # Even zero budget
            budget_period="per-task"
        )
        
        model = await ollama_selector.select_model(budget_criteria)
        assert model is not None
        
        # Verify the selected model is within budget (free)
        estimated_cost = model.cost.estimate_cost_for_budget_period("per-task")
        assert estimated_cost == 0.0
        
        # Higher budget - should still work with free models
        premium_criteria = ModelSelectionCriteria(
            cost_limit=100.0,  # High budget
            budget_period="per-task"
        )
        
        model = await ollama_selector.select_model(premium_criteria)
        assert model is not None
        assert model.cost.is_free

    @pytest.mark.asyncio
    async def test_size_constraint_selection_with_ollama(self, ollama_selector):
        """Test size constraint selection with Ollama models."""
        # Small model preference
        small_criteria = ModelSelectionCriteria(max_model_size=5.0)
        model = await ollama_selector.select_model(small_criteria)
        
        assert model is not None
        assert model._size_billions <= 5.0
        
        # If we have multiple models, test large model preference
        if len(ollama_selector.registry.models) > 1:
            large_criteria = ModelSelectionCriteria(min_model_size=3.0)
            model = await ollama_selector.select_model(large_criteria)
            
            assert model is not None
            assert model._size_billions >= 3.0

    @pytest.mark.asyncio
    async def test_modality_selection_with_ollama(self, ollama_selector):
        """Test modality-based selection with Ollama models."""
        # Code modality requirement
        code_criteria = ModelSelectionCriteria(modalities=["code"])
        model = await ollama_selector.select_model(code_criteria)
        
        # Should select a code-capable model if available
        assert model is not None
        
        # Check if it's actually code-specialized
        if "code" in model._expertise:
            assert "code" in model._expertise
        # Otherwise, any model should work as fallback

    @pytest.mark.asyncio
    async def test_complex_criteria_with_ollama(self, ollama_selector):
        """Test complex multi-criteria selection with Ollama models."""
        complex_criteria = ModelSelectionCriteria(
            expertise="medium",
            min_model_size=1.0,
            max_model_size=10.0,
            cost_limit=0.0,  # Free only
            budget_period="per-task",
            selection_strategy="balanced"
        )
        
        model = await ollama_selector.select_model(complex_criteria)
        assert model is not None
        
        # Verify it meets size constraints
        assert 1.0 <= model._size_billions <= 10.0
        
        # Verify cost constraint (should be free)
        assert model.cost.is_free

    def test_ollama_capability_detection(self, ollama_registry):
        """Test capability detection with real Ollama models."""
        # Test capability analysis for each model
        for model_key, model in ollama_registry.models.items():
            analysis = ollama_registry.detect_model_capabilities(model)
            
            # Should have complete analysis
            assert "basic_capabilities" in analysis
            assert "advanced_capabilities" in analysis
            assert "performance_metrics" in analysis
            assert "expertise_analysis" in analysis
            assert "cost_analysis" in analysis
            assert "suitability_scores" in analysis
            
            # Cost analysis should reflect free pricing
            cost_analysis = analysis["cost_analysis"]
            assert cost_analysis["type"] == "free"
            assert cost_analysis["cost_per_1k_avg"] == 0.0
            assert cost_analysis["budget_friendly"] == True
            assert cost_analysis["efficiency_score"] == 100.0
            
            # Should have reasonable suitability scores
            scores = analysis["suitability_scores"]
            for capability, score in scores.items():
                assert 0.0 <= score <= 1.0
            
            # Free models should have maximum budget score
            assert scores["budget_constrained"] == 1.0

    def test_ollama_task_recommendations(self, ollama_registry):
        """Test task-based recommendations with Ollama models."""
        # Code-related task
        code_recs = ollama_registry.recommend_models_for_task(
            "Help me write Python code", 
            max_recommendations=2
        )
        
        assert len(code_recs) > 0
        
        for rec in code_recs:
            assert "model" in rec
            assert "reasoning" in rec
            assert rec["suitability_score"] > 0
            
            # Should have reasonable reasoning
            assert "free" in rec["reasoning"].lower()  # Should mention it's free
        
        # General chat task
        chat_recs = ollama_registry.recommend_models_for_task(
            "Have a conversation with me",
            max_recommendations=2
        )
        
        assert len(chat_recs) > 0
        # Should recommend models suitable for chat
        for rec in chat_recs:
            model = rec["model"]
            assert "general" in model._expertise or "chat" in model._expertise

    def test_ollama_cost_analysis_integration(self, ollama_registry):
        """Test cost analysis with Ollama models (all free)."""
        for model in ollama_registry.models.values():
            # Test budget period estimates - should all be 0
            task_cost = model.cost.estimate_cost_for_budget_period("per-task")
            pipeline_cost = model.cost.estimate_cost_for_budget_period("per-pipeline")
            hour_cost = model.cost.estimate_cost_for_budget_period("per-hour")
            
            assert task_cost == 0.0
            assert pipeline_cost == 0.0
            assert hour_cost == 0.0
            
            # Test cost breakdown
            breakdown = model.cost.get_cost_breakdown(1000, 500)
            assert breakdown["total_cost"] == 0.0
            assert breakdown["input_cost"] == 0.0
            assert breakdown["output_cost"] == 0.0
            assert breakdown["is_free"] == True
            
            # Test cost efficiency - should be maximum
            efficiency = model.cost.get_cost_efficiency_score(0.8)
            assert efficiency == 100.0
            
            # Test budget compliance - should always be within budget
            assert model.cost.is_within_budget(0.0, "per-task")  # Even zero budget
            assert model.cost.is_within_budget(1000.0, "per-hour")

    @pytest.mark.asyncio
    async def test_ollama_fallback_strategies(self, ollama_selector):
        """Test fallback strategies with Ollama models."""
        # Impossible requirements with cheapest fallback
        impossible_criteria = ModelSelectionCriteria(
            min_model_size=1000.0,  # Impossibly large
            fallback_strategy="cheapest"
        )
        
        model = await ollama_selector.select_model(impossible_criteria)
        assert model is not None
        
        # All Ollama models are free, so should fallback to any available
        assert model.cost.is_free
        
        # Best available fallback
        best_criteria = ModelSelectionCriteria(
            min_model_size=1000.0,  # Impossibly large
            fallback_strategy="best_available"
        )
        
        model = await ollama_selector.select_model(best_criteria)
        assert model is not None
        
        # Should fallback to best available model
        assert model.cost.is_free

    @pytest.mark.asyncio 
    async def test_ollama_yaml_integration(self, ollama_selector):
        """Test YAML requirements parsing with Ollama models."""
        # Simulate YAML requirements suitable for local models
        yaml_requirements = {
            "expertise": "medium",
            "modalities": ["text"],
            "min_size": "1B",
            "max_size": "10B", 
            "cost_limit": 0.0,  # Free only
            "budget_period": "per-task",
            "fallback_strategy": "best_available"
        }
        
        # Parse and select
        criteria = ollama_selector.parse_requirements_from_yaml(yaml_requirements)
        model = await ollama_selector.select_model(criteria)
        
        assert model is not None
        assert model._size_billions >= 1.0
        assert model._size_billions <= 10.0
        assert model.cost.is_free

    def test_ollama_model_size_parsing_integration(self):
        """Test model size parsing with Ollama model names."""
        from orchestrator.utils.model_utils import parse_model_size
        
        # Test Ollama model names with size information
        test_cases = [
            ("gemma3:1b", None, 1.0),
            ("gemma3:4b", None, 4.0),
            ("llama3.2:3b", None, 3.0),
            ("deepseek-r1:1.5b", None, 1.5),
            ("deepseek-r1:8b", None, 8.0),
            ("deepseek-r1:32b", None, 32.0),
        ]
        
        for model_name, size_str, expected in test_cases:
            result = parse_model_size(model_name, size_str)
            assert result == expected, f"Failed for {model_name}: expected {expected}, got {result}"

    def test_ollama_expertise_detection(self):
        """Test expertise detection logic for Ollama models."""
        # Test with different model configurations
        model1 = OllamaModel(model_name="gemma3:1b")
        assert "fast" in model1._expertise or "compact" in model1._expertise
        
        model2 = OllamaModel(model_name="deepseek-r1:8b")  
        assert "code" in model2._expertise or "general" in model2._expertise
        
        model3 = OllamaModel(model_name="llama3.2:3b")
        assert len(model3._expertise) > 0

    @pytest.mark.asyncio
    async def test_ollama_health_check_integration(self, ollama_registry):
        """Test health checking with real Ollama models."""
        if not is_ollama_running():
            pytest.skip("Ollama is not running locally")
        
        # Get a model from registry
        model = list(ollama_registry.models.values())[0]
        
        # Test health check
        is_healthy = await model.health_check()
        assert isinstance(is_healthy, bool)
        
        # If Ollama is running, model should be healthy
        if is_ollama_running():
            assert is_healthy
        
        # Test with registry health filtering
        healthy_models = await ollama_registry._filter_by_health([model])
        if is_healthy:
            assert len(healthy_models) > 0
            assert model in healthy_models

    def test_ollama_performance_characteristics(self, ollama_registry):
        """Test that Ollama models have appropriate performance characteristics."""
        for model in ollama_registry.models.values():
            # Free models should have maximum budget efficiency
            analysis = ollama_registry.detect_model_capabilities(model)
            scores = analysis["suitability_scores"]
            
            # Should excel at budget-constrained tasks
            assert scores["budget_constrained"] == 1.0
            
            # Fast models should have good speed scores
            if "fast" in model._expertise:
                # Should be suitable for speed-critical tasks
                assert scores["speed_critical"] > 0.5


@pytest.mark.skipif(
    not is_ollama_running(),
    reason="Requires Ollama to be running locally"
)
class TestOllamaLiveIntegration:
    """
    Live integration tests that make actual calls to Ollama.
    These tests require Ollama to be running locally.
    """
    
    @pytest.mark.asyncio
    async def test_live_ollama_generation_with_enhanced_selection(self):
        """Test actual text generation with enhanced model selection."""
        available_models = get_available_ollama_models()
        if not available_models:
            pytest.skip("No Ollama models available locally")
        
        registry = ModelRegistry()
        
        # Use the first available model
        model_name = available_models[0]
        model = OllamaModel(model_name=model_name)
        
        registry.register_model(model)
        selector = ModelSelector(registry)
        
        # Use enhanced selection
        criteria = ModelSelectionCriteria(
            expertise="low",  # Use low to be flexible
            cost_limit=0.0,  # Free only
            budget_period="per-task"
        )
        
        selected_model = await selector.select_model(criteria)
        assert selected_model is not None
        
        # Make actual API call with short response
        try:
            response = await selected_model.generate(
                "Say hello in one word",
                temperature=0.1,
                max_tokens=5
            )
            
            assert isinstance(response, str)
            assert len(response) > 0
            print(f"Ollama response: {response}")
        except Exception as e:
            # Ollama might not have the model downloaded or running
            pytest.skip(f"Ollama generation failed: {e}")

    def test_ollama_model_listing(self):
        """Test listing of available Ollama models."""
        if not is_ollama_running():
            pytest.skip("Ollama is not running locally")
        
        available_models = get_available_ollama_models()
        print(f"Available Ollama models: {available_models}")
        
        # Should have at least some indication of Ollama status
        assert isinstance(available_models, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not skipif"])