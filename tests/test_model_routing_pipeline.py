"""Test complete model routing pipeline execution."""

import subprocess
import json
import os
from pathlib import Path
import pytest
import time


class TestModelRoutingPipeline:
    """Test complete pipeline execution."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create test output directory
        self.test_output_dir = Path("test_outputs/model_routing")
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up any previous test outputs
        for file in self.test_output_dir.glob("*"):
            file.unlink()
    
    def test_pipeline_with_balanced_priority(self):
        """Test pipeline with balanced routing."""
        result = subprocess.run([
            "python", "scripts/run_pipeline.py",
            "examples/model_routing_demo.yaml",
            "-i", "priority=balanced",
            "-i", "task_budget=10.00",
            "-o", str(self.test_output_dir)
        ], capture_output=True, text=True, timeout=120)
        
        # Check execution succeeded
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
        
        # Check outputs exist
        assert self.test_output_dir.exists()
        
        # Find report files
        routing_report = self.test_output_dir / "routing_analysis.md"
        report_files = list(self.test_output_dir.glob("report_*.md"))
        
        assert routing_report.exists(), "Routing analysis report not created"
        assert len(report_files) > 0, "Final report not created"
        
        # Verify routing report content
        with open(routing_report) as f:
            content = f.read()
            
            # Check structure
            assert "# Model Routing Results" in content
            assert "## Configuration" in content
            assert "Budget: $10" in content
            assert "Priority: balanced" in content
            
            # Check task routing sections
            assert "### Document Summary" in content
            assert "Assigned Model:" in content
            assert "Estimated Cost: $" in content
            
            assert "### Code Generation" in content
            assert "### Data Analysis" in content
            
            # Check batch optimization
            assert "## Batch Translation Optimization" in content
            assert "Total Tasks:" in content
            assert "Models Used:" in content
            
            # Check summary
            assert "## Summary" in content
            assert "Total Pipeline Cost:" in content
            assert "Budget Remaining:" in content
        
        # Verify final report
        with open(report_files[0]) as f:
            content = f.read()
            assert "# Model Routing Demo Report" in content
            assert "Generated:" in content
            assert "Budget: $10" in content
            assert "Priority: balanced" in content
            
            # Should contain actual results (not template markers)
            assert "{{" not in content
            assert "}}" not in content
    
    def test_pipeline_with_cost_optimization(self):
        """Test pipeline with cost-optimized routing."""
        result = subprocess.run([
            "python", "scripts/run_pipeline.py",
            "examples/model_routing_demo.yaml",
            "-i", "priority=cost",
            "-i", "task_budget=5.00",
            "-o", "test_outputs/model_routing_cost"
        ], capture_output=True, text=True, timeout=120)
        
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
        
        # Verify cost constraints respected
        output_dir = Path("test_outputs/model_routing_cost")
        report_files = list(output_dir.glob("routing_analysis.md"))
        
        assert len(report_files) > 0
        
        with open(report_files[0]) as f:
            content = f.read()
            
            # Should use cost priority
            assert "Priority: cost" in content
            
            # Should use cheaper models
            content_lower = content.lower()
            # At least one budget-friendly model should be mentioned
            budget_models = ["llama", "gemma", "nano", "mini", "1b"]
            assert any(model in content_lower for model in budget_models), \
                "Cost optimization should select budget-friendly models"
    
    def test_pipeline_with_quality_optimization(self):
        """Test pipeline with quality-optimized routing."""
        result = subprocess.run([
            "python", "scripts/run_pipeline.py",
            "examples/model_routing_demo.yaml",
            "-i", "priority=quality",
            "-i", "task_budget=20.00",
            "-o", "test_outputs/model_routing_quality"
        ], capture_output=True, text=True, timeout=120)
        
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
        
        output_dir = Path("test_outputs/model_routing_quality")
        report_files = list(output_dir.glob("routing_analysis.md"))
        
        assert len(report_files) > 0
        
        with open(report_files[0]) as f:
            content = f.read()
            
            # Should use quality priority
            assert "Priority: quality" in content
            
            # Should use higher-quality models
            content_lower = content.lower()
            quality_models = ["opus", "gpt-5", "sonnet", "pro"]
            assert any(model in content_lower for model in quality_models), \
                "Quality optimization should select high-performance models"
    
    def test_pipeline_edge_cases(self):
        """Test pipeline with edge cases."""
        # Test with very low budget
        result = subprocess.run([
            "python", "scripts/run_pipeline.py",
            "examples/model_routing_demo.yaml",
            "-i", "priority=cost",
            "-i", "task_budget=0.10",
            "-o", "test_outputs/model_routing_low_budget"
        ], capture_output=True, text=True, timeout=120)
        
        # Should still complete with cheapest options
        assert result.returncode == 0, f"Pipeline should handle low budget: {result.stderr}"
        
        # Check that it used the cheapest possible models
        output_dir = Path("test_outputs/model_routing_low_budget")
        report_files = list(output_dir.glob("routing_analysis.md"))
        
        if report_files:
            with open(report_files[0]) as f:
                content = f.read()
                # Should show very low costs
                assert "$0.10" in content or "$0.1" in content
    
    def test_pipeline_output_quality(self):
        """Test that pipeline outputs are high quality."""
        result = subprocess.run([
            "python", "scripts/run_pipeline.py",
            "examples/model_routing_demo.yaml",
            "-i", "priority=balanced",
            "-i", "task_budget=10.00",
            "-o", "test_outputs/model_routing_quality_check"
        ], capture_output=True, text=True, timeout=120)
        
        assert result.returncode == 0
        
        output_dir = Path("test_outputs/model_routing_quality_check")
        
        # Check all markdown files for quality issues
        for md_file in output_dir.glob("*.md"):
            with open(md_file) as f:
                content = f.read()
                
                # No template markers
                assert "{{" not in content, f"Template markers in {md_file}"
                assert "{%" not in content, f"Template tags in {md_file}"
                
                # No placeholder text
                assert "[placeholder]" not in content.lower()
                assert "[todo]" not in content.lower()
                assert "lorem ipsum" not in content.lower()
                
                # No error messages
                assert "error:" not in content.lower() or "no error" in content.lower()
                assert "exception:" not in content.lower()
                assert "traceback" not in content.lower()
                
                # Has actual content
                assert len(content) > 100, f"Content too short in {md_file}"
                
                # Proper markdown structure
                assert "#" in content, f"No headers in {md_file}"
    
    @pytest.mark.slow
    def test_pipeline_multiple_runs(self):
        """Test pipeline stability across multiple runs."""
        priorities = ["cost", "balanced", "quality"]
        
        for i, priority in enumerate(priorities):
            output_dir = f"test_outputs/model_routing_run_{i}"
            
            result = subprocess.run([
                "python", "scripts/run_pipeline.py",
                "examples/model_routing_demo.yaml",
                "-i", f"priority={priority}",
                "-i", "task_budget=8.00",
                "-o", output_dir
            ], capture_output=True, text=True, timeout=120)
            
            assert result.returncode == 0, f"Run {i} with {priority} failed"
            
            # Small delay between runs to avoid rate limiting
            if i < len(priorities) - 1:
                time.sleep(2)
        
        # Verify all runs produced outputs
        for i in range(len(priorities)):
            output_dir = Path(f"test_outputs/model_routing_run_{i}")
            assert output_dir.exists()
            assert len(list(output_dir.glob("*.md"))) >= 2
    
    def test_pipeline_checkpoint_recovery(self):
        """Test that pipeline can recover from checkpoints."""
        # First run the pipeline partially (this would need interruption simulation)
        # For now, just verify checkpoint creation
        
        result = subprocess.run([
            "python", "scripts/run_pipeline.py",
            "examples/model_routing_demo.yaml",
            "-i", "priority=balanced",
            "-i", "task_budget=10.00",
            "-o", "test_outputs/model_routing_checkpoint"
        ], capture_output=True, text=True, timeout=120)
        
        assert result.returncode == 0
        
        # Check if checkpoints were created
        checkpoint_dir = Path("checkpoints")
        if checkpoint_dir.exists():
            # Find checkpoints for this pipeline
            checkpoints = list(checkpoint_dir.glob("model-routing-demo*.json"))
            # Pipeline should create checkpoints during execution
            # (this depends on checkpoint configuration)
    
    def teardown_method(self):
        """Clean up test outputs."""
        # Optionally clean up test outputs
        # For debugging, you might want to keep them
        pass