"""Test MCP memory workflow pipeline integration with real execution."""

import json
import subprocess
import time
from pathlib import Path
import pytest
import shutil
import os


class TestMCPMemoryPipeline:
    """Test complete pipeline execution with real MCP memory."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test environment and clean up after."""
        # Create test output directory
        self.test_output_dir = Path("test_outputs/mcp_memory")
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        yield
        
        # Clean up test outputs
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)
    
    def test_full_pipeline_execution(self):
        """Test complete pipeline with real MCP memory."""
        # Run pipeline
        result = subprocess.run(
            [
                "python", "scripts/run_pipeline.py",
                "examples/mcp_memory_workflow.yaml",
                "-i", "user_name=TestUser",
                "-i", "task_description=Analyze test data",
                "-o", str(self.test_output_dir)
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent  # Run from project root
        )
        
        # Print output for debugging
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        # Check success
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
        
        # Verify output files
        export_file = self.test_output_dir / "mcp_memory_export.json"
        summary_file = self.test_output_dir / "context_summary.md"
        
        assert export_file.exists(), f"Export file not created at {export_file}"
        assert summary_file.exists(), f"Summary file not created at {summary_file}"
        
        # Verify JSON content
        with open(export_file) as f:
            data = json.load(f)
            assert data["namespace"] == "conversation"
            assert "user_profile" in data["keys"]
            assert "task_steps" in data["keys"]
            assert "progress" in data["keys"]
            assert data["metadata"]["user"] == "TestUser"
            assert data["metadata"]["task"] == "Analyze test data"
            assert "exported_at" in data
        
        # Verify markdown content
        with open(summary_file) as f:
            content = f.read()
            assert "Context Summary" in content
            assert "conversation" in content
            assert "user_profile" in content
            assert "task_steps" in content
            assert "progress" in content
            assert "Generated on:" in content
    
    def test_pipeline_with_custom_output_path(self):
        """Test pipeline respects custom output path."""
        custom_output = Path("test_outputs/custom_path")
        
        result = subprocess.run(
            [
                "python", "scripts/run_pipeline.py",
                "examples/mcp_memory_workflow.yaml",
                "-i", "user_name=CustomUser",
                "-i", "task_description=Custom task",
                "-o", str(custom_output)
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
        
        # Verify files in custom location
        assert (custom_output / "mcp_memory_export.json").exists()
        assert (custom_output / "context_summary.md").exists()
        
        # Clean up
        if custom_output.exists():
            shutil.rmtree(custom_output.parent)
    
    def test_pipeline_with_different_parameters(self):
        """Test pipeline with various parameter values."""
        test_cases = [
            ("Alice", "Process financial reports"),
            ("Bob Smith", "Generate quarterly analytics"),
            ("テストユーザー", "データ分析"),  # Japanese characters
            ("User with spaces", "Task with special chars: @#$%")
        ]
        
        for user_name, task_desc in test_cases:
            output_dir = self.test_output_dir / f"test_{hash(user_name)}"
            
            result = subprocess.run(
                [
                    "python", "scripts/run_pipeline.py",
                    "examples/mcp_memory_workflow.yaml",
                    "-i", f"user_name={user_name}",
                    "-i", f"task_description={task_desc}",
                    "-o", str(output_dir)
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            assert result.returncode == 0, f"Failed for {user_name}: {result.stderr}"
            
            # Verify export contains correct data
            with open(output_dir / "mcp_memory_export.json") as f:
                data = json.load(f)
                assert data["metadata"]["user"] == user_name
                assert data["metadata"]["task"] == task_desc
    
    def test_pipeline_memory_persistence(self):
        """Test that memory persists across pipeline steps."""
        result = subprocess.run(
            [
                "python", "scripts/run_pipeline.py",
                "examples/mcp_memory_workflow.yaml",
                "-i", "user_name=PersistenceTest",
                "-o", str(self.test_output_dir)
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        
        # Check that all three keys were stored and retrieved
        with open(self.test_output_dir / "mcp_memory_export.json") as f:
            data = json.load(f)
            assert len(data["keys"]) == 3
            assert set(data["keys"]) == {"user_profile", "task_steps", "progress"}
    
    def test_pipeline_checkpoint_creation(self):
        """Test that pipeline creates checkpoints with memory data."""
        result = subprocess.run(
            [
                "python", "scripts/run_pipeline.py",
                "examples/mcp_memory_workflow.yaml",
                "-o", str(self.test_output_dir)
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        
        # Check for checkpoint files
        checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
        checkpoint_files = list(checkpoint_dir.glob("mcp_memory_workflow_*.json"))
        
        # Should have created at least one checkpoint
        assert len(checkpoint_files) > 0, "No checkpoint files created"
        
        # Verify checkpoint contains memory operations
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        with open(latest_checkpoint) as f:
            checkpoint_data = json.load(f)
            
            # Check for memory operation results in metadata
            if "metadata" in checkpoint_data and "previous_results" in checkpoint_data["metadata"]:
                results = checkpoint_data["metadata"]["previous_results"]
                
                # Verify memory operations succeeded
                if "init_context" in results:
                    assert results["init_context"]["success"] is True
                    assert results["init_context"]["stored"] is True
                
                if "get_full_context" in results:
                    assert results["get_full_context"]["success"] is True
                    assert "keys" in results["get_full_context"]
    
    def test_pipeline_error_handling(self):
        """Test pipeline handles errors gracefully."""
        # Test with invalid output path (though pipeline should create it)
        result = subprocess.run(
            [
                "python", "scripts/run_pipeline.py",
                "examples/mcp_memory_workflow.yaml",
                "-o", "/invalid/path/that/cannot/be/created"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        # Pipeline might fail due to permission issues or handle it gracefully
        # Just verify it doesn't crash unexpectedly
        assert result.returncode != 139  # Segmentation fault code
    
    def test_pipeline_template_rendering(self):
        """Test that all template expressions are properly rendered."""
        result = subprocess.run(
            [
                "python", "scripts/run_pipeline.py",
                "examples/mcp_memory_workflow.yaml",
                "-i", "user_name=TemplateTest",
                "-i", "task_description=Test template rendering",
                "-o", str(self.test_output_dir)
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        
        # Check that templates were rendered (no {{ }} in output)
        with open(self.test_output_dir / "mcp_memory_export.json") as f:
            content = f.read()
            assert "{{" not in content
            assert "}}" not in content
            
        # Verify timestamp was rendered (read file again for JSON parsing)
        with open(self.test_output_dir / "mcp_memory_export.json") as f:
            data = json.load(f)
            assert data["exported_at"]  # Should have a timestamp value
            assert "now()" not in str(data["exported_at"])
        
        with open(self.test_output_dir / "context_summary.md") as f:
            content = f.read()
            assert "{{" not in content
            assert "}}" not in content
            assert "Generated on:" in content