"""
Test Category: Enhanced Model Cost Calculations
Real tests for enhanced ModelCost functionality including budget periods and cost analysis.
"""

import pytest
from orchestrator.core.model import ModelCost


class TestEnhancedModelCost:
    """Test enhanced model cost calculation methods."""

    @pytest.fixture
    def free_model_cost(self):
        """Create a free model cost instance."""
        return ModelCost(is_free=True)

    @pytest.fixture
    def budget_model_cost(self):
        """Create a budget model cost instance."""
        return ModelCost(
            input_cost_per_1k_tokens=0.0001,
            output_cost_per_1k_tokens=0.0003,
            base_cost_per_request=0.0,
            is_free=False
        )

    @pytest.fixture
    def premium_model_cost(self):
        """Create a premium model cost instance."""
        return ModelCost(
            input_cost_per_1k_tokens=0.003,
            output_cost_per_1k_tokens=0.015,
            base_cost_per_request=0.001,
            is_free=False
        )

    def test_estimate_cost_for_budget_period(self, budget_model_cost, premium_model_cost, free_model_cost):
        """Test cost estimation for different budget periods."""
        # Free model should always be $0
        assert free_model_cost.estimate_cost_for_budget_period("per-task") == 0.0
        assert free_model_cost.estimate_cost_for_budget_period("per-pipeline") == 0.0
        assert free_model_cost.estimate_cost_for_budget_period("per-hour") == 0.0
        
        # Budget model costs
        task_cost = budget_model_cost.estimate_cost_for_budget_period("per-task")
        pipeline_cost = budget_model_cost.estimate_cost_for_budget_period("per-pipeline")
        hour_cost = budget_model_cost.estimate_cost_for_budget_period("per-hour")
        
        # Costs should increase with usage
        assert task_cost < pipeline_cost < hour_cost
        
        # Verify specific calculations
        # per-task: 500 input + 500 output = 1000 tokens
        expected_task = (500/1000 * 0.0001) + (500/1000 * 0.0003)
        assert abs(task_cost - expected_task) < 1e-6
        
        # Premium model should be more expensive
        premium_task = premium_model_cost.estimate_cost_for_budget_period("per-task")
        assert premium_task > task_cost

    def test_get_cost_efficiency_score(self, free_model_cost, budget_model_cost, premium_model_cost):
        """Test cost efficiency scoring."""
        # Free models should have maximum efficiency
        assert free_model_cost.get_cost_efficiency_score(0.8) == 100.0
        assert free_model_cost.get_cost_efficiency_score(1.0) == 100.0
        
        # Budget model efficiency
        budget_efficiency = budget_model_cost.get_cost_efficiency_score(0.8)
        assert budget_efficiency > 0
        
        # Premium model efficiency (lower due to higher cost)
        premium_efficiency = premium_model_cost.get_cost_efficiency_score(0.9)
        assert premium_efficiency > 0
        
        # Budget model should be more efficient (higher score) for similar performance
        budget_perf_80 = budget_model_cost.get_cost_efficiency_score(0.8)
        premium_perf_80 = premium_model_cost.get_cost_efficiency_score(0.8)
        assert budget_perf_80 > premium_perf_80
        
        # Higher performance should increase efficiency
        budget_perf_90 = budget_model_cost.get_cost_efficiency_score(0.9)
        assert budget_perf_90 > budget_perf_80

    def test_compare_cost_with(self, budget_model_cost, premium_model_cost, free_model_cost):
        """Test cost comparison between models."""
        # Compare budget with premium
        comparison = budget_model_cost.compare_cost_with(premium_model_cost)
        
        assert "self_cost" in comparison
        assert "other_cost" in comparison
        assert "cost_ratio" in comparison
        assert "savings" in comparison
        assert "percent_savings" in comparison
        
        # Budget should be cheaper than premium
        assert comparison["self_cost"] < comparison["other_cost"]
        assert comparison["cost_ratio"] < 1.0
        assert comparison["savings"] > 0  # Positive savings means budget is cheaper
        assert comparison["percent_savings"] > 0
        
        # Compare with free model
        free_comparison = budget_model_cost.compare_cost_with(free_model_cost)
        assert free_comparison["other_cost"] == 0.0
        assert free_comparison["cost_ratio"] == float('inf')  # Budget costs more than free
        assert free_comparison["savings"] < 0  # Negative savings (budget costs more)
        
        # Free vs free should be equal
        free_vs_free = free_model_cost.compare_cost_with(free_model_cost)
        assert free_vs_free["cost_ratio"] == 1.0
        assert free_vs_free["savings"] == 0.0

    def test_is_within_budget(self, free_model_cost, budget_model_cost, premium_model_cost):
        """Test budget checking functionality."""
        # Free models are always within budget
        assert free_model_cost.is_within_budget(0.0, "per-task")
        assert free_model_cost.is_within_budget(0.001, "per-pipeline")
        
        # Budget model with various limits
        task_cost = budget_model_cost.estimate_cost_for_budget_period("per-task")
        
        # Should be within budget if limit is higher than cost
        assert budget_model_cost.is_within_budget(task_cost + 0.001, "per-task")
        
        # Should be outside budget if limit is lower than cost
        assert not budget_model_cost.is_within_budget(task_cost - 0.001, "per-task")
        
        # Edge case: exactly at budget
        assert budget_model_cost.is_within_budget(task_cost, "per-task")
        
        # Different budget periods
        pipeline_cost = budget_model_cost.estimate_cost_for_budget_period("per-pipeline")
        assert budget_model_cost.is_within_budget(pipeline_cost + 0.001, "per-pipeline")

    def test_get_cost_breakdown(self, free_model_cost, budget_model_cost, premium_model_cost):
        """Test detailed cost breakdown."""
        # Free model breakdown
        free_breakdown = free_model_cost.get_cost_breakdown(1000, 500)
        assert free_breakdown["is_free"] == True
        assert free_breakdown["total_cost"] == 0.0
        assert free_breakdown["input_cost"] == 0.0
        assert free_breakdown["output_cost"] == 0.0
        assert free_breakdown["base_cost"] == 0.0
        
        # Budget model breakdown
        breakdown = budget_model_cost.get_cost_breakdown(1000, 500)
        assert breakdown["is_free"] == False
        
        # Verify calculations
        expected_input = (1000/1000) * 0.0001  # 1000 tokens * $0.0001 per 1k
        expected_output = (500/1000) * 0.0003  # 500 tokens * $0.0003 per 1k
        
        assert abs(breakdown["input_cost"] - expected_input) < 1e-6
        assert abs(breakdown["output_cost"] - expected_output) < 1e-6
        assert breakdown["base_cost"] == 0.0
        assert abs(breakdown["total_cost"] - (expected_input + expected_output)) < 1e-6
        
        # Premium model with base cost
        premium_breakdown = premium_model_cost.get_cost_breakdown(1000, 500)
        assert premium_breakdown["base_cost"] == 0.001
        assert premium_breakdown["total_cost"] > breakdown["total_cost"]

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Zero token costs should still work
        zero_cost = ModelCost(
            input_cost_per_1k_tokens=0.0,
            output_cost_per_1k_tokens=0.0,
            base_cost_per_request=0.1,
            is_free=False
        )
        
        assert zero_cost.calculate_cost(1000, 1000) == 0.1  # Only base cost
        assert zero_cost.get_cost_efficiency_score(0.8) == 100.0  # Treated as free for efficiency
        
        # Unknown budget period should default to per-task
        budget_model = ModelCost(
            input_cost_per_1k_tokens=0.001,
            output_cost_per_1k_tokens=0.002,
            is_free=False
        )
        
        unknown_period_cost = budget_model.estimate_cost_for_budget_period("unknown-period")
        task_cost = budget_model.estimate_cost_for_budget_period("per-task")
        assert unknown_period_cost == task_cost

    def test_real_world_cost_scenarios(self):
        """Test with realistic model costs."""
        # GPT-4o mini (as of 2024)
        gpt4_mini = ModelCost(
            input_cost_per_1k_tokens=0.00015,
            output_cost_per_1k_tokens=0.0006,
            is_free=False
        )
        
        # Claude Sonnet (as of 2024)  
        claude_sonnet = ModelCost(
            input_cost_per_1k_tokens=0.003,
            output_cost_per_1k_tokens=0.015,
            is_free=False
        )
        
        # Ollama (free local)
        ollama_free = ModelCost(is_free=True)
        
        # Typical task: 1000 input, 500 output tokens
        gpt_cost = gpt4_mini.calculate_cost(1000, 500)
        claude_cost = claude_sonnet.calculate_cost(1000, 500)
        ollama_cost = ollama_free.calculate_cost(1000, 500)
        
        # Verify cost ordering
        assert ollama_cost == 0.0
        assert gpt_cost < claude_cost  # GPT-4o mini is cheaper
        
        # Budget analysis for $1/day
        daily_budget = 1.0
        
        # Estimate daily usage (assume 20 tasks per day)
        gpt_daily = gpt_cost * 20
        claude_daily = claude_cost * 20
        
        assert ollama_free.is_within_budget(daily_budget, "per-hour")  # Always within budget
        
        # Check which models fit budget
        if gpt_daily <= daily_budget:
            assert gpt4_mini.is_within_budget(daily_budget/20, "per-task")
        
        if claude_daily <= daily_budget:
            assert claude_sonnet.is_within_budget(daily_budget/20, "per-task")

    def test_cost_comparison_analysis(self):
        """Test comprehensive cost comparison analysis."""
        # Create realistic model costs
        models = {
            "free_local": ModelCost(is_free=True),
            "budget_api": ModelCost(
                input_cost_per_1k_tokens=0.0002,
                output_cost_per_1k_tokens=0.0006,
                is_free=False
            ),
            "premium_api": ModelCost(
                input_cost_per_1k_tokens=0.01,
                output_cost_per_1k_tokens=0.03,
                is_free=False
            )
        }
        
        # Performance scores (hypothetical)
        performance = {
            "free_local": 0.7,
            "budget_api": 0.85,
            "premium_api": 0.95
        }
        
        # Calculate efficiency scores
        efficiency_scores = {}
        for name, cost in models.items():
            efficiency_scores[name] = cost.get_cost_efficiency_score(performance[name])
        
        # Free model should have highest efficiency
        assert efficiency_scores["free_local"] == 100.0
        
        # Compare budget periods
        for period in ["per-task", "per-pipeline", "per-hour"]:
            costs = {}
            for name, cost in models.items():
                costs[name] = cost.estimate_cost_for_budget_period(period)
            
            # Free should always be $0
            assert costs["free_local"] == 0.0
            
            # Budget should be less than premium
            assert costs["budget_api"] < costs["premium_api"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])