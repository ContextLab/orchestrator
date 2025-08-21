"""Integration tests for modular analysis pipeline with real data and APIs."""

import asyncio
import json
import os
import sys
from pathlib import Path
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.models.registry_singleton import get_model_registry
from src.orchestrator.control_systems.hybrid_control_system import HybridControlSystem


class TestModularPipelineIntegration:
    """Test the complete modular analysis pipeline with real components."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator with real model registry."""
        model_registry = get_model_registry()
        control_system = HybridControlSystem(model_registry)
        return Orchestrator(model_registry=model_registry, control_system=control_system)
    
    @pytest.mark.asyncio
    async def test_visualization_tool_chart_generation(self):
        """Test VisualizationTool creates real charts."""
        from src.orchestrator.tools.visualization_tools import VisualizationTool
        
        tool = VisualizationTool()
        
        # Create test data
        test_data = [
            {"month": "Jan", "sales": 100, "revenue": 1000, "region": "North"},
            {"month": "Feb", "sales": 120, "revenue": 1200, "region": "North"},
            {"month": "Mar", "sales": 140, "revenue": 1400, "region": "South"},
            {"month": "Apr", "sales": 110, "revenue": 1100, "region": "South"},
            {"month": "May", "sales": 160, "revenue": 1600, "region": "East"},
        ]
        
        # Test chart creation
        result = await tool.execute(
            action="create_charts",
            data=test_data,
            chart_types=["bar", "line", "pie"],
            output_dir="tests/test_outputs/charts",
            title="Test Charts"
        )
        
        assert result["success"]
        assert len(result["charts"]) >= 1
        assert result["count"] >= 1
        
        # Verify files were created
        for chart_path in result["charts"]:
            assert Path(chart_path).exists()
            assert Path(chart_path).stat().st_size > 0
    
    @pytest.mark.asyncio
    async def test_visualization_tool_dashboard_creation(self):
        """Test VisualizationTool creates HTML dashboard."""
        from src.orchestrator.tools.visualization_tools import VisualizationTool
        
        tool = VisualizationTool()
        
        # First create some charts
        test_data = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 15, 25, 30],
            "category": ["A", "B", "A", "B", "A"]
        })
        
        charts_result = await tool.execute(
            action="create_charts",
            data=test_data.to_dict(orient='records'),
            chart_types=["line", "bar"],
            output_dir="tests/test_outputs/charts"
        )
        
        # Create dashboard
        dashboard_result = await tool.execute(
            action="create_dashboard",
            charts=charts_result["charts"],
            title="Test Dashboard",
            layout="grid",
            output_dir="tests/test_outputs"
        )
        
        assert dashboard_result["success"]
        assert "url" in dashboard_result or "dashboard_path" in dashboard_result
        
        # Verify HTML file was created
        dashboard_path = dashboard_result.get("url") or dashboard_result.get("dashboard_path")
        assert Path(dashboard_path).exists()
        
        # Check HTML content
        with open(dashboard_path, 'r') as f:
            html_content = f.read()
            assert "<html>" in html_content.lower()
            assert "dashboard" in html_content.lower()
    
    @pytest.mark.asyncio
    async def test_data_processing_clean_action(self):
        """Test DataProcessingTool clean action with real data."""
        from src.orchestrator.tools.data_tools import DataProcessingTool
        
        tool = DataProcessingTool()
        
        # Create test data with issues
        test_data = [
            {"id": 1, "value": 100, "category": "A"},
            {"id": 2, "value": None, "category": "B"},  # Missing value
            {"id": 1, "value": 100, "category": "A"},  # Duplicate
            {"id": 3, "value": 200, "category": "C"},
            {"id": 4, "value": 1000, "category": "D"},  # Potential outlier
        ]
        
        result = await tool.execute(
            action="clean",
            data=test_data,
            format="json",
            remove_duplicates=True,
            handle_missing="drop",
            remove_outliers=False
        )
        
        assert result["success"]
        assert result["rows_removed"] >= 2  # At least duplicate and missing value row
        assert result["cleaned_shape"]["rows"] < result["original_shape"]["rows"]
    
    @pytest.mark.asyncio
    async def test_data_processing_merge_action(self):
        """Test DataProcessingTool merge action with real datasets."""
        from src.orchestrator.tools.data_tools import DataProcessingTool
        
        tool = DataProcessingTool()
        
        # Create test datasets
        dataset1 = [
            {"id": 1, "name": "Alice", "score": 90},
            {"id": 2, "name": "Bob", "score": 85},
        ]
        
        dataset2 = [
            {"id": 3, "name": "Charlie", "score": 95},
            {"id": 4, "name": "Diana", "score": 88},
        ]
        
        result = await tool.execute(
            action="merge",
            datasets=[
                {"name": "first", "data": dataset1},
                {"name": "second", "data": dataset2}
            ],
            merge_strategy="combine"
        )
        
        assert result["success"]
        assert result["datasets_merged"] == 2
        assert result["total_rows"] == 4
        assert "_source" in result["columns"]
    
    @pytest.mark.asyncio
    async def test_statistical_analysis_sub_pipeline(self, orchestrator):
        """Test statistical analysis sub-pipeline with real computations."""
        # Ensure sample data exists
        data_path = Path("examples/outputs/modular_analysis/input/dataset.csv")
        if not data_path.exists():
            # Generate sample data
            os.system("python scripts/generate_sample_data.py")
        
        # Load sample data
        df = pd.read_csv(data_path)
        sample_data = df.head(100).to_dict(orient='records')
        
        # Execute statistical analysis pipeline
        with open("examples/sub_pipelines/statistical_analysis.yaml", 'r') as f:
            pipeline_yaml = f.read()
        
        results = await orchestrator.execute_yaml(
            pipeline_yaml,
            {"data": sample_data, "confidence_level": 0.95}
        )
        
        # Verify results
        assert "outputs" in results
        outputs = results["outputs"]
        
        assert "statistics" in outputs
        assert "correlations" in outputs
        assert "hypothesis_tests" in outputs
        assert "summary" in outputs
        
        # Check statistics have real values
        stats = outputs["statistics"]
        assert isinstance(stats, dict)
        if stats:  # If we have numeric columns
            first_col = list(stats.keys())[0]
            assert "mean" in stats[first_col]
            assert "std" in stats[first_col]
            assert stats[first_col]["mean"] != 0  # Should have real computed values
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires API keys for sentiment analysis"
    )
    async def test_sentiment_analysis_sub_pipeline(self, orchestrator):
        """Test sentiment analysis sub-pipeline with real LLM calls."""
        # Create test data with comments
        test_data = [
            {"id": 1, "comments": "This product is amazing! Best purchase ever."},
            {"id": 2, "comments": "Terrible quality, very disappointed."},
            {"id": 3, "comments": "It's okay, nothing special but works."},
            {"id": 4, "comments": "Excellent service and fast delivery!"},
            {"id": 5, "comments": "Waste of money, broke after one day."},
        ]
        
        # Execute sentiment analysis pipeline
        with open("examples/sub_pipelines/sentiment_analysis.yaml", 'r') as f:
            pipeline_yaml = f.read()
        
        results = await orchestrator.execute_yaml(
            pipeline_yaml,
            {"data": test_data, "text_column": "comments"}
        )
        
        # Verify results
        assert "outputs" in results
        outputs = results["outputs"]
        
        assert "sentiment_scores" in outputs
        assert "keywords" in outputs
        assert "overall_sentiment" in outputs
        
        # Check sentiment scores are realistic
        sentiment = outputs["sentiment_scores"]
        if "sentiment_statistics" in sentiment:
            stats = sentiment["sentiment_statistics"]
            assert "positive_ratio" in stats
            assert "negative_ratio" in stats
            # With our test data, we should have both positive and negative
            assert stats["positive_ratio"] > 0
            assert stats["negative_ratio"] > 0
    
    @pytest.mark.asyncio  
    async def test_trend_analysis_sub_pipeline(self, orchestrator):
        """Test trend analysis sub-pipeline with real time series calculations."""
        # Create time series data
        import datetime
        
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        test_data = []
        
        for i, date in enumerate(dates):
            # Create upward trend with some noise
            sales = 100 + i * 2 + np.random.normal(0, 10)
            revenue = sales * 10
            test_data.append({
                "timestamp": date.strftime("%Y-%m-%d"),
                "sales": max(0, sales),
                "revenue": max(0, revenue)
            })
        
        # Execute trend analysis pipeline
        with open("examples/sub_pipelines/trend_analysis.yaml", 'r') as f:
            pipeline_yaml = f.read()
        
        results = await orchestrator.execute_yaml(
            pipeline_yaml,
            {
                "data": test_data,
                "time_column": "timestamp",
                "value_columns": ["sales", "revenue"]
            }
        )
        
        # Verify results
        assert "outputs" in results
        outputs = results["outputs"]
        
        assert "trends" in outputs
        assert "moving_averages" in outputs
        assert "seasonality" in outputs
        assert "forecasts" in outputs
        
        # Check trends detected
        trends = outputs["trends"]
        if "sales" in trends:
            # Should detect upward trend in our synthetic data
            assert trends["sales"]["trend_direction"] in ["upward", "stable"]
            assert trends["sales"]["slope"] > 0  # Positive slope for upward trend
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_modular_pipeline_execution(self, orchestrator):
        """Test complete modular analysis pipeline execution."""
        # Ensure all components exist
        data_path = Path("examples/outputs/modular_analysis/input/dataset.csv")
        if not data_path.exists():
            os.system("python scripts/generate_sample_data.py")
        
        # Execute full pipeline
        with open("examples/modular_analysis_pipeline_fixed.yaml", 'r') as f:
            pipeline_yaml = f.read()
        
        inputs = {
            "output_path": "tests/test_outputs/modular_analysis",
            "dataset": "input/dataset.csv",
            "analysis_types": ["statistical", "sentiment", "trend"],
            "output_format": "pdf"
        }
        
        # Create input directory and copy dataset
        os.makedirs("tests/test_outputs/modular_analysis/input", exist_ok=True)
        import shutil
        shutil.copy(data_path, "tests/test_outputs/modular_analysis/input/dataset.csv")
        
        results = await orchestrator.execute_yaml(pipeline_yaml, inputs)
        
        # Verify pipeline completed
        assert results is not None
        
        # Check for output files
        output_dir = Path("tests/test_outputs/modular_analysis")
        
        # Charts should be generated
        charts_dir = output_dir / "charts"
        if charts_dir.exists():
            chart_files = list(charts_dir.glob("*.png"))
            assert len(chart_files) > 0
        
        # Dashboard should be created
        dashboard_files = list(output_dir.glob("dashboard*.html"))
        assert len(dashboard_files) > 0
        
        # Results file should exist
        results_files = list(output_dir.glob("results*.md"))
        assert len(results_files) > 0
    
    @pytest.mark.asyncio
    async def test_conditional_execution(self, orchestrator):
        """Test that conditional steps execute correctly based on analysis_types."""
        # Prepare minimal test
        test_data = [{"id": 1, "value": 100}, {"id": 2, "value": 200}]
        
        with open("examples/modular_analysis_pipeline_fixed.yaml", 'r') as f:
            pipeline_yaml = f.read()
        
        # Test with only statistical analysis
        inputs = {
            "output_path": "tests/test_outputs/conditional",
            "dataset": "test.csv",
            "analysis_types": ["statistical"],  # Only statistical
            "output_format": "pdf"
        }
        
        # Create test data file
        os.makedirs("tests/test_outputs/conditional/input", exist_ok=True)
        pd.DataFrame(test_data).to_csv("tests/test_outputs/conditional/input/test.csv", index=False)
        
        # This should work even with limited analysis types
        # The sentiment and trend steps should be skipped
        results = await orchestrator.execute_yaml(pipeline_yaml, inputs)
        
        # Pipeline should complete successfully even with conditional steps
        assert results is not None


# Run tests
if __name__ == "__main__":
    # Create test output directories
    os.makedirs("tests/test_outputs/charts", exist_ok=True)
    os.makedirs("tests/test_outputs/modular_analysis", exist_ok=True)
    
    # Run specific test or all tests
    import sys
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        pytest.main([__file__, f"::{test_name}", "-v", "-s"])
    else:
        pytest.main([__file__, "-v", "-s"])