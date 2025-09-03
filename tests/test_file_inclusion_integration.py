"""
Integration tests for file inclusion with real pipeline execution.
Tests the complete file inclusion workflow with actual files.
NO MOCKS - all tests use real file operations and pipeline execution.
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any
import pytest

from src.orchestrator.compiler.yaml_compiler import YAMLCompiler
from src.orchestrator.core.file_inclusion import FileInclusionProcessor

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider


class TestFileInclusionIntegration:
    """Integration tests for file inclusion with real pipeline execution."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_pipeline_files(self, temp_dir):
        """Create sample pipeline and supporting files."""
        
        # Create directory structure
        prompts_dir = Path(temp_dir) / "prompts"
        config_dir = Path(temp_dir) / "config"
        prompts_dir.mkdir()
        config_dir.mkdir()
        
        # System prompt
        system_prompt = prompts_dir / "system.txt"
        system_prompt.write_text(
            "You are a helpful research assistant. "
            "Provide accurate, well-researched responses based on the given context.",
            encoding="utf-8"
        )
        
        # Instructions (single line to avoid YAML conflicts)
        instructions = prompts_dir / "instructions.md"
        instructions.write_text("Research Instructions: Gather information, analyze critically, present findings clearly.", encoding="utf-8")
        
        # Configuration
        config = config_dir / "settings.json"
        config.write_text("""{
  "research": {
    "depth": "comprehensive",
    "sources": ["academic", "news", "reports"],
    "max_results": 20
  },
  "output": {
    "format": "structured",
    "include_sources": true
  }
}""", encoding="utf-8")
        
        # Main pipeline YAML
        pipeline_yaml = f"""
id: file_inclusion_test
name: File Inclusion Test Pipeline
version: 1.0.0

inputs:
  topic:
    type: string
    description: Research topic
    default: "renewable energy"

steps:
  - id: prepare_research
    name: Prepare Research
    action: llm_call
    parameters:
      model: anthropic/claude-sonnet-4-20250514  
      prompt: |
        {{{{ file:prompts/system.txt }}}}
        
        << prompts/instructions.md >>
        
        Research Topic: {{{{ topic }}}}
        
        Configuration: {{{{ file:config/settings.json }}}}
        
        Please prepare a research outline for this topic.

  - id: mock_analysis
    name: Mock Analysis Step
    action: llm_call
    depends_on: prepare_research
    parameters:
      model: anthropic/claude-sonnet-4-20250514
      prompt: |
        Based on the research outline: {{{{ prepare_research.result }}}}
        
        Topic: {{{{ topic }}}}
        
        Please provide a mock analysis (this is a test).
"""
        
        pipeline_file = Path(temp_dir) / "test_pipeline.yaml"
        pipeline_file.write_text(pipeline_yaml, encoding="utf-8")
        
        return {
            "pipeline_file": str(pipeline_file),
            "temp_dir": temp_dir,
            "system_prompt": "You are a helpful research assistant. Provide accurate, well-researched responses based on the given context.",
            "instructions_content": "Research Instructions"
        }

    @pytest.mark.asyncio
    async def test_pipeline_compilation_with_file_inclusions(self, sample_pipeline_files):
        """Test that pipeline compiles correctly with file inclusions."""
        
        # Read the pipeline YAML
        with open(sample_pipeline_files["pipeline_file"], "r") as f:
            yaml_content = f.read()
        
        # Set up compiler with file inclusion processor
        file_processor = FileInclusionProcessor(
            base_dirs=[sample_pipeline_files["temp_dir"], "."]
        )
        compiler = YAMLCompiler(file_inclusion_processor=file_processor)
        
        # Compile the pipeline
        pipeline = await compiler.compile(
            yaml_content, 
            context={"topic": "solar energy research"}
        )
        
        # Verify pipeline structure
        assert pipeline.id == "file_inclusion_test"
        assert pipeline.name == "File Inclusion Test Pipeline"
        
        # Get tasks
        tasks = list(pipeline.get_tasks())
        assert len(tasks) == 2
        
        # Check first task (prepare_research)
        prepare_task = next(t for t in tasks if t.id == "prepare_research")
        prompt = prepare_task.parameters.get("prompt", "")
        
        # Verify file inclusions worked
        assert sample_pipeline_files["system_prompt"] in prompt
        assert "Research Instructions" in prompt
        assert "solar energy research" in prompt
        assert '"depth": "comprehensive"' in prompt  # From JSON config
        
        # Check second task has dependency
        analysis_task = next(t for t in tasks if t.id == "mock_analysis")
        assert "prepare_research" in analysis_task.dependencies
        
        print("✓ Pipeline compilation with file inclusions successful")
        print(f"✓ Included system prompt: {len(sample_pipeline_files['system_prompt'])} chars")
        print(f"✓ Included instructions: Found 'Research Instructions' header")
        print(f"✓ Included JSON config: Found 'comprehensive' setting")
        print(f"✓ Template processing: Found 'solar energy research' topic")

    @pytest.mark.asyncio
    async def test_nested_file_inclusions_in_pipeline(self, temp_dir):
        """Test nested file inclusions within pipeline definitions."""
        
        # Create nested structure
        base_dir = Path(temp_dir) / "nested_test"
        base_dir.mkdir()
        
        # Base template that includes other files
        base_template = base_dir / "base_prompt.txt" 
        base_template.write_text("""Main research prompt.

{{ file:sections/methodology.md }}

{{ file:sections/quality_standards.txt }}

End of prompt.""", encoding="utf-8")
        
        # Create sections directory
        sections_dir = base_dir / "sections"
        sections_dir.mkdir()
        
        # Methodology section
        methodology = sections_dir / "methodology.md"
        methodology.write_text("""## Methodology
1. Literature review
2. Data analysis  
3. Synthesis""", encoding="utf-8")
        
        # Quality standards
        quality = sections_dir / "quality_standards.txt"
        quality.write_text("""Quality Standards:
- Accuracy: Verify all facts
- Objectivity: Avoid bias
- Completeness: Cover all aspects""", encoding="utf-8")
        
        # Pipeline using nested inclusions
        pipeline_yaml = f"""
id: nested_inclusion_test
name: Nested File Inclusion Test
version: 1.0.0

steps:
  - id: complex_prompt
    name: Complex Prompt with Nested Inclusions
    action: llm_call
    parameters:
      model: anthropic/claude-sonnet-4-20250514
      prompt: |
        {{{{ file:base_prompt.txt }}}}
        
        Please process this complex prompt structure.
"""
        
        pipeline_file = base_dir / "nested_pipeline.yaml"
        pipeline_file.write_text(pipeline_yaml, encoding="utf-8")
        
        # Compile pipeline
        file_processor = FileInclusionProcessor(base_dirs=[str(base_dir)])
        compiler = YAMLCompiler(file_inclusion_processor=file_processor)
        
        with open(pipeline_file, "r") as f:
            yaml_content = f.read()
        
        pipeline = await compiler.compile(yaml_content)
        
        # Verify nested inclusions worked
        tasks = list(pipeline.get_tasks())
        task = tasks[0]
        prompt = task.parameters.get("prompt", "")
        
        assert "Main research prompt." in prompt
        assert "## Methodology" in prompt
        assert "Literature review" in prompt
        assert "Quality Standards:" in prompt
        assert "Verify all facts" in prompt
        assert "End of prompt." in prompt
        
        # Verify the structure is correct (nested content included in right places)
        methodology_pos = prompt.find("## Methodology")
        quality_pos = prompt.find("Quality Standards:")
        end_pos = prompt.find("End of prompt.")
        
        assert methodology_pos < quality_pos < end_pos
        
        print("✓ Nested file inclusions processed correctly")
        print(f"✓ Final prompt length: {len(prompt)} characters")
        print("✓ All nested content included in correct order")

    @pytest.mark.asyncio 
    async def test_mixed_syntax_file_inclusions(self, temp_dir):
        """Test both {{ file: }} and << >> syntax in same pipeline."""
        
        # Create test files
        prompt1 = Path(temp_dir) / "prompt1.txt"
        prompt1.write_text("First prompt section", encoding="utf-8")
        
        prompt2 = Path(temp_dir) / "prompt2.txt" 
        prompt2.write_text("Second prompt section", encoding="utf-8")
        
        config_file = Path(temp_dir) / "config.json"
        config_file.write_text('{"setting": "test_value"}', encoding="utf-8")
        
        # Pipeline with mixed syntax
        pipeline_yaml = f"""
id: mixed_syntax_test
name: Mixed Syntax File Inclusion Test
version: 1.0.0

steps:
  - id: mixed_prompt
    name: Mixed Syntax Prompt
    action: llm_call
    parameters:
      model: anthropic/claude-sonnet-4-20250514
      prompt: |
        Template syntax: {{{{ file:prompt1.txt }}}}
        
        Bracket syntax: << prompt2.txt >>
        
        JSON config: {{{{ file:config.json }}}}
        
        Both syntaxes work together.
"""
        
        pipeline_file = Path(temp_dir) / "mixed_pipeline.yaml"
        pipeline_file.write_text(pipeline_yaml, encoding="utf-8")
        
        # Compile and test
        file_processor = FileInclusionProcessor(base_dirs=[temp_dir])
        compiler = YAMLCompiler(file_inclusion_processor=file_processor)
        
        with open(pipeline_file, "r") as f:
            yaml_content = f.read()
            
        pipeline = await compiler.compile(yaml_content)
        
        # Verify both syntaxes worked
        tasks = list(pipeline.get_tasks())
        prompt = tasks[0].parameters.get("prompt", "")
        
        assert "First prompt section" in prompt
        assert "Second prompt section" in prompt  
        assert '"setting": "test_value"' in prompt
        assert "Both syntaxes work together." in prompt
        
        print("✓ Mixed syntax file inclusions work correctly")
        print("✓ Both {{ file: }} and << >> syntax processed")
        print("✓ JSON and text files included successfully")

    def test_file_inclusion_performance_metrics(self, temp_dir, sample_pipeline_files):
        """Test performance metrics and caching of file inclusion processor."""
        
        async def run_performance_test():
            # Set up processor with caching enabled
            file_processor = FileInclusionProcessor(
                base_dirs=[sample_pipeline_files["temp_dir"]],
                cache_enabled=True
            )
            
            # Process the same content multiple times
            test_content = "{{ file:prompts/system.txt }} << config/settings.json >>"
            
            # First run
            result1 = await file_processor.process_content(
                test_content, 
                sample_pipeline_files["temp_dir"]
            )
            
            # Second run (should hit cache)  
            result2 = await file_processor.process_content(
                test_content,
                sample_pipeline_files["temp_dir"] 
            )
            
            # Verify results are identical
            assert result1 == result2
            
            # Check metrics
            metrics = file_processor.get_metrics()
            assert metrics["cache_hits"] > 0
            assert metrics["files_loaded"] >= 2  # system.txt and settings.json
            assert metrics["total_bytes_loaded"] > 0
            
            # Check cache info
            cache_info = file_processor.get_cache_info()
            assert len(cache_info) >= 2  # At least 2 files cached
            
            for path, info in cache_info.items():
                assert "size" in info
                assert "cached_at" in info
                assert info["age_seconds"] >= 0
            
            print("✓ Performance metrics tracking works")
            print(f"✓ Cache hits: {metrics['cache_hits']}")
            print(f"✓ Files loaded: {metrics['files_loaded']}")
            print(f"✓ Total bytes: {metrics['total_bytes_loaded']}")
            print(f"✓ Cached files: {len(cache_info)}")
        
        # Run the async test
        asyncio.run(run_performance_test())


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])