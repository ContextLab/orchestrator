"""
Tests for file inclusion system with real file operations.
NO MOCKS - all tests use real files and I/O operations.
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any
import pytest

from src.orchestrator.core.file_inclusion import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
    FileInclusionProcessor,
    FileIncludeDirective,
    FileIncludeResult,
    FileInclusionError,
    SecurityError,
    FileNotFoundError,
    FileSizeError,
    CircularInclusionError,
    include_file,
    process_content,
)


class TestFileInclusionProcessor:
    """Test suite for FileInclusionProcessor with real file operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def processor(self, temp_dir):
        """Create a FileInclusionProcessor with test directory."""
        return FileInclusionProcessor(
            base_dirs=[temp_dir, ".", "tests/fixtures"],
            cache_enabled=True,
            max_file_size=1024 * 1024,  # 1MB
            max_inclusion_depth=5
        )

    @pytest.fixture
    def sample_files(self, temp_dir):
        """Create sample files for testing."""
        files = {}
        
        # Basic text file
        basic_file = Path(temp_dir) / "basic.txt"
        basic_content = "This is a basic text file for testing."
        basic_file.write_text(basic_content, encoding="utf-8")
        files["basic.txt"] = basic_content
        
        # Markdown file with content
        md_file = Path(temp_dir) / "content.md"
        md_content = """# Test Document

This is a test markdown document with:
- Lists
- **Bold text**
- `Code snippets`

## Section 2
More content here.
"""
        md_file.write_text(md_content, encoding="utf-8")
        files["content.md"] = md_content
        
        # JSON configuration file
        json_file = Path(temp_dir) / "config.json"
        json_content = """{
  "database": {
    "host": "localhost",
    "port": 5432,
    "name": "testdb"
  },
  "features": {
    "logging": true,
    "caching": false
  }
}"""
        json_file.write_text(json_content, encoding="utf-8")
        files["config.json"] = json_content
        
        # File with nested inclusion
        nested_file = Path(temp_dir) / "with_inclusion.md"
        nested_content = """# Main Document

## Configuration
{{ file:config.json }}

## Basic Content
<< basic.txt >>

## End"""
        nested_file.write_text(nested_content, encoding="utf-8")
        files["with_inclusion.md"] = nested_content
        
        # File that includes the nested file (circular test)
        circular_file = Path(temp_dir) / "circular.md"
        circular_content = """# Circular Test
{{ file:with_inclusion.md }}
<< circular.md >>"""
        circular_file.write_text(circular_content, encoding="utf-8")
        files["circular.md"] = circular_content
        
        return files

    @pytest.mark.asyncio
    async def test_basic_file_inclusion_template_syntax(self, processor, sample_files, temp_dir):
        """Test basic file inclusion using {{ file:path }} syntax."""
        directive = FileIncludeDirective(
            syntax="template",
            path="basic.txt",
            base_dir=temp_dir
        )
        
        result = await processor.include_file(directive)
        
        assert isinstance(result, FileIncludeResult)
        assert result.content == sample_files["basic.txt"]
        assert result.resolved_path.endswith("basic.txt")
        assert result.size == len(sample_files["basic.txt"])
        assert result.encoding == "utf-8"
        assert not result.cached  # First load
        assert result.load_time > 0

    @pytest.mark.asyncio
    async def test_basic_file_inclusion_bracket_syntax(self, processor, sample_files, temp_dir):
        """Test basic file inclusion using << path >> syntax."""
        directive = FileIncludeDirective(
            syntax="bracket",
            path="content.md",
            base_dir=temp_dir
        )
        
        result = await processor.include_file(directive)
        
        assert result.content == sample_files["content.md"]
        assert "# Test Document" in result.content
        assert "## Section 2" in result.content

    @pytest.mark.asyncio
    async def test_process_content_with_mixed_syntax(self, processor, sample_files, temp_dir):
        """Test processing content with both {{ file: }} and << >> syntax."""
        content = f"""# Combined Test

## Template Syntax
{{{{ file:basic.txt }}}}

## Bracket Syntax  
<< content.md >>

## End of test
"""
        
        result = await processor.process_content(content, temp_dir)
        
        assert sample_files["basic.txt"] in result
        assert sample_files["content.md"] in result
        assert "# Combined Test" in result
        assert "{{ file:basic.txt }}" not in result  # Should be replaced
        assert "<< content.md >>" not in result     # Should be replaced

    @pytest.mark.asyncio
    async def test_nested_file_inclusion(self, processor, sample_files, temp_dir):
        """Test recursive file inclusion."""
        # Process the nested file that includes other files
        result = await processor.process_content(sample_files["with_inclusion.md"], temp_dir)
        
        # Should contain content from all included files
        assert sample_files["basic.txt"] in result
        assert '"database"' in result  # From config.json
        assert '"host": "localhost"' in result
        assert "# Main Document" in result
        assert len(result) > len(sample_files["with_inclusion.md"])

    @pytest.mark.asyncio
    async def test_circular_inclusion_detection(self, processor, sample_files, temp_dir):
        """Test detection and prevention of circular inclusions."""
        with pytest.raises((CircularInclusionError, FileInclusionError)) as exc_info:
            await processor.process_content(sample_files["circular.md"], temp_dir)
        
        assert "Circular inclusion detected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_file_caching(self, processor, sample_files, temp_dir):
        """Test file content caching functionality."""
        directive = FileIncludeDirective(
            syntax="template",
            path="basic.txt",
            base_dir=temp_dir
        )
        
        # First load
        result1 = await processor.include_file(directive)
        assert not result1.cache_hit
        
        # Second load should be cached
        result2 = await processor.include_file(directive)
        assert result2.cache_hit
        assert result2.content == result1.content
        
        # Verify cache metrics
        metrics = processor.get_metrics()
        assert metrics["cache_hits"] >= 1
        assert metrics["cache_misses"] >= 1

    @pytest.mark.asyncio
    async def test_file_not_found_error(self, processor, temp_dir):
        """Test handling of non-existent files."""
        directive = FileIncludeDirective(
            syntax="template",
            path="nonexistent.txt",
            base_dir=temp_dir
        )
        
        with pytest.raises(FileNotFoundError) as exc_info:
            await processor.include_file(directive)
        
        assert "File not found" in str(exc_info.value)
        assert "nonexistent.txt" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_security_path_traversal_prevention(self, processor, temp_dir):
        """Test prevention of path traversal attacks."""
        # Try to access files outside base directory
        directive = FileIncludeDirective(
            syntax="template",
            path="../../../etc/passwd",
            base_dir=temp_dir
        )
        
        with pytest.raises(FileNotFoundError):  # File should not be found due to security restrictions
            await processor.include_file(directive)

    @pytest.mark.asyncio
    async def test_file_size_limit_enforcement(self, processor, temp_dir):
        """Test enforcement of file size limits."""
        # Create a large file that exceeds the limit
        large_file = Path(temp_dir) / "large.txt"
        large_content = "x" * (2 * 1024 * 1024)  # 2MB, exceeds 1MB limit
        large_file.write_text(large_content, encoding="utf-8")
        
        directive = FileIncludeDirective(
            syntax="template",
            path="large.txt",
            base_dir=temp_dir,
            max_size=1024  # 1KB limit
        )
        
        with pytest.raises(FileSizeError) as exc_info:
            await processor.include_file(directive)
        
        assert "File too large" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_different_encodings(self, processor, temp_dir):
        """Test handling of different file encodings."""
        # Create a UTF-8 file with special characters
        utf8_file = Path(temp_dir) / "utf8.txt"
        utf8_content = "Hello 世界! 🌍 Café naïve"
        utf8_file.write_text(utf8_content, encoding="utf-8")
        
        directive = FileIncludeDirective(
            syntax="template",
            path="utf8.txt",
            base_dir=temp_dir,
            encoding="utf-8"
        )
        
        result = await processor.include_file(directive)
        assert result.content == utf8_content
        assert "世界" in result.content
        assert "🌍" in result.content

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, processor, sample_files, temp_dir):
        """Test performance metrics tracking."""
        # Initial metrics
        initial_metrics = processor.get_metrics()
        
        # Perform several operations
        await processor.process_content("{{ file:basic.txt }}", temp_dir)
        await processor.process_content("{{ file:basic.txt }}", temp_dir)  # Should hit cache
        await processor.process_content("<< content.md >>", temp_dir)
        
        # Check updated metrics
        final_metrics = processor.get_metrics()
        
        assert final_metrics["files_loaded"] > initial_metrics["files_loaded"]
        assert final_metrics["total_bytes_loaded"] > initial_metrics["total_bytes_loaded"]
        assert final_metrics["cache_hits"] > initial_metrics["cache_hits"]

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_file_modification(self, processor, temp_dir):
        """Test cache invalidation when files are modified."""
        # Create a test file
        test_file = Path(temp_dir) / "modifiable.txt"
        original_content = "Original content"
        test_file.write_text(original_content, encoding="utf-8")
        
        directive = FileIncludeDirective(
            syntax="template",
            path="modifiable.txt",
            base_dir=temp_dir
        )
        
        # First load
        result1 = await processor.include_file(directive)
        assert result1.content == original_content
        
        # Modify the file
        modified_content = "Modified content"
        test_file.write_text(modified_content, encoding="utf-8")
        
        # Second load should get fresh content (cache should be invalidated)
        result2 = await processor.include_file(directive)
        assert result2.content == modified_content
        assert result2.content != result1.content

    @pytest.mark.asyncio
    async def test_convenience_functions(self, temp_dir, sample_files):
        """Test the convenience functions include_file and process_content."""
        # Test include_file function
        content = await include_file("basic.txt", base_dir=temp_dir)
        assert content == sample_files["basic.txt"]
        
        # Test process_content function
        template = "Content: {{ file:basic.txt }}"
        result = await process_content(template, base_dir=temp_dir)
        assert sample_files["basic.txt"] in result

    @pytest.mark.asyncio
    async def test_complex_nested_structure(self, processor, temp_dir):
        """Test complex nested file inclusion scenarios."""
        # Create a hierarchy of files that include each other
        
        # Base template
        base_file = Path(temp_dir) / "base.md"
        base_content = """# Documentation Base

## Configuration
{{ file:sections/config.md }}

## Installation  
{{ file:sections/install.md }}
"""
        base_file.write_text(base_content, encoding="utf-8")
        
        # Create sections directory
        sections_dir = Path(temp_dir) / "sections"
        sections_dir.mkdir()
        
        # Config section
        config_file = sections_dir / "config.md"
        config_content = """### Configuration Settings

```json
{{ file:../examples/config.json }}
```
"""
        config_file.write_text(config_content, encoding="utf-8")
        
        # Install section
        install_file = sections_dir / "install.md"
        install_content = """### Installation Steps

1. Download the package
2. Configure settings
3. Run setup

<< ../examples/setup.sh >>
"""
        install_file.write_text(install_content, encoding="utf-8")
        
        # Create examples directory
        examples_dir = Path(temp_dir) / "examples"
        examples_dir.mkdir()
        
        # Example config
        example_config = examples_dir / "config.json"
        example_config.write_text('{"version": "1.0", "debug": true}', encoding="utf-8")
        
        # Example setup script
        setup_script = examples_dir / "setup.sh"
        setup_script.write_text("#!/bin/bash\necho 'Setting up...'\n", encoding="utf-8")
        
        # Process the base file
        result = await processor.process_content(base_content, temp_dir)
        
        # Verify all content is included
        assert "# Documentation Base" in result
        assert "### Configuration Settings" in result
        assert "### Installation Steps" in result
        assert '"version": "1.0"' in result
        assert "#!/bin/bash" in result

    def test_cache_info_and_management(self, processor, temp_dir, sample_files):
        """Test cache management functionality."""
        # Run some operations to populate cache
        async def populate_cache():
            await processor.process_content("{{ file:basic.txt }}", temp_dir)
            await processor.process_content("<< content.md >>", temp_dir)
        
        asyncio.run(populate_cache())
        
        # Check cache info
        cache_info = processor.get_cache_info()
        assert len(cache_info) > 0
        
        for path, info in cache_info.items():
            assert "size" in info
            assert "encoding" in info
            assert "cached_at" in info
            assert "age_seconds" in info
        
        # Clear cache
        processor.clear_cache()
        cache_info_after = processor.get_cache_info()
        assert len(cache_info_after) == 0

    @pytest.mark.asyncio
    async def test_error_recovery_and_optional_files(self, processor, temp_dir):
        """Test error recovery with optional file inclusions."""
        # Create content with both existing and non-existing files
        content = """# Test Document

## Required Content
{{ file:basic.txt }}

## Optional Content (this file doesn't exist)
{{ file:optional.txt }}

## More Required Content
<< content.md >>
"""
        
        # Create only some of the referenced files
        basic_file = Path(temp_dir) / "basic.txt"
        basic_file.write_text("Basic content", encoding="utf-8")
        
        content_file = Path(temp_dir) / "content.md"
        content_file.write_text("# Content", encoding="utf-8")
        
        # This should fail because optional.txt doesn't exist
        with pytest.raises(FileInclusionError):
            await processor.process_content(content, temp_dir)

    @pytest.mark.asyncio
    async def test_inclusion_depth_limit(self, processor, temp_dir):
        """Test maximum inclusion depth enforcement."""
        # Create a chain of files that include each other
        for i in range(10):  # Create more files than the depth limit (5)
            file_path = Path(temp_dir) / f"level_{i}.txt"
            if i < 9:
                content = f"Level {i}\n{{{{ file:level_{i+1}.txt }}}}"
            else:
                content = f"Level {i} (final)"
            file_path.write_text(content, encoding="utf-8")
        
        # This should fail due to depth limit
        with pytest.raises(FileInclusionError) as exc_info:
            await processor.process_content("{{ file:level_0.txt }}", temp_dir)
        
        assert "Maximum inclusion depth" in str(exc_info.value)


class TestYAMLCompilerIntegration:
    """Test integration with YAML compiler."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_yaml_compilation_with_file_inclusions(self, temp_dir):
        """Test YAML compilation with file inclusion directives."""
        from src.orchestrator.compiler.yaml_compiler import YAMLCompiler
        
        # Create included files
        prompt_file = Path(temp_dir) / "prompt.txt"
        prompt_content = "You are a helpful AI assistant. Please help the user with their request."
        prompt_file.write_text(prompt_content, encoding="utf-8")
        
        instructions_file = Path(temp_dir) / "instructions.md"
        instructions_content = """Instructions:
        
        - Be helpful and accurate
        - Provide clear explanations
        - Ask for clarification when needed"""
        instructions_file.write_text(instructions_content, encoding="utf-8")
        
        # Create YAML with file inclusions
        yaml_content = f"""
id: test_pipeline
name: Test Pipeline with File Inclusions
version: 1.0.0

inputs:
  topic: 
    type: string
    description: The topic to discuss

steps:
  - id: main_task
    name: Main Task
    action: llm_call
    parameters:
      model: anthropic/claude-sonnet-4-20250514
      prompt: |
        {{{{ file:prompt.txt }}}}
        
        {{{{ file:instructions.md }}}}
        
        Topic: {{{{ topic }}}}
"""
        
        # Set up compiler with file inclusion processor
        from src.orchestrator.core.file_inclusion import FileInclusionProcessor
        file_processor = FileInclusionProcessor(base_dirs=[temp_dir])
        compiler = YAMLCompiler(file_inclusion_processor=file_processor)
        
        # Compile pipeline
        pipeline = await compiler.compile(yaml_content, context={"topic": "AI Ethics"})
        
        # Verify compilation worked
        assert pipeline.id == "test_pipeline"
        print(f"Pipeline has {len(pipeline.tasks)} tasks")
        
        # Get task properly - it might be a list or dict
        if hasattr(pipeline, 'tasks') and len(pipeline.tasks) > 0:
            if isinstance(pipeline.tasks, list):
                task = pipeline.tasks[0]
            else:
                # It might be a dict with task IDs as keys
                task_id = list(pipeline.tasks.keys())[0]
                task = pipeline.tasks[task_id]
        else:
            # Alternative: get tasks as a list
            all_tasks = list(pipeline.get_tasks())
            assert len(all_tasks) == 1
            task = all_tasks[0]
        
        print(f"Task: {task}")
        print(f"Task type: {type(task)}")
        
        # The prompt parameter should contain the included content after processing
        if hasattr(task, 'parameters'):
            prompt_param = task.parameters.get("prompt", "")
        else:
            prompt_param = ""
            
        print(f"Prompt param: {prompt_param[:200]}...")
        
        # Note: The actual inclusion happens during YAML processing, 
        # so the prompt should contain the included content
        assert prompt_content in prompt_param
        assert "Be helpful and accurate" in prompt_param  # From instructions
        assert "AI Ethics" in prompt_param  # From context


class TestTemplateManagerIntegration:
    """Test integration with template manager."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_template_manager_file_inclusion(self, temp_dir):
        """Test template manager with file inclusion support."""
        from src.orchestrator.core.template_manager import TemplateManager
        from src.orchestrator.core.file_inclusion import FileInclusionProcessor
        
        # Create test files
        greeting_file = Path(temp_dir) / "greeting.txt"
        greeting_file.write_text("Hello, {{name}}!", encoding="utf-8")
        
        signature_file = Path(temp_dir) / "signature.txt"
        signature_file.write_text("Best regards,\nThe AI Assistant", encoding="utf-8")
        
        # Set up template manager with file inclusion
        file_processor = FileInclusionProcessor(base_dirs=[temp_dir])
        template_manager = TemplateManager(file_inclusion_processor=file_processor)
        
        # Register context
        template_manager.register_context("name", "Alice")
        
        # Template with file inclusions
        template = """
{{ file:greeting.txt }}

Thank you for your question about {{topic}}.

{{ file:signature.txt }}
"""
        
        # Use async rendering
        result = await template_manager.deep_render_async(template, {"topic": "machine learning"})
        
        # Verify result
        assert "Hello, Alice!" in result
        assert "machine learning" in result
        assert "Best regards" in result
        assert "The AI Assistant" in result


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])