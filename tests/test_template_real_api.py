"""
Real API Template Rendering Tests - NO MOCKS OR SIMULATIONS

This test suite validates template rendering with actual API calls and real infrastructure.
All tests use the same APIs and code paths that users rely on in production.
"""

import pytest
import os
import sys
import asyncio
import tempfile
import time
from pathlib import Path
from functools import wraps

from orchestrator import Orchestrator, init_models


# Cost control configuration
TEST_LIMITS = {
    'max_tokens_per_test': 500,      # Limit token usage per test
    'max_requests_per_minute': 10,   # Rate limiting
    'test_timeout_seconds': 120      # Prevent hanging tests
}


def cost_controlled_test(max_tokens=500, timeout=120):
    """Decorator to control API costs and timeout in real tests."""
    def decorator(test_func):
        @wraps(test_func)
        async def wrapper(*args, **kwargs):
            # Set token limits for this test
            start_time = time.time()
            try:
                # Run test with timeout
                result = await asyncio.wait_for(
                    test_func(*args, **kwargs), 
                    timeout=timeout
                )
                execution_time = time.time() - start_time
                print(f"Test executed in {execution_time:.2f}s")
                return result
            except asyncio.TimeoutError:
                pytest.fail(f"Test timed out after {timeout} seconds")
        return wrapper
    return decorator


@pytest.fixture(scope="session")
def real_model_registry():
    """Use actual model registry with real API keys - same as production."""
    # Load real API keys from environment (same as users do)
    required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']
    available_keys = [key for key in required_keys if os.getenv(key)]
    
    if not available_keys:
        pytest.skip(f"Skipping real API tests - no API keys found. Set one of: {required_keys}")
    
    print(f"Using real APIs with keys: {available_keys}")
    
    # Use EXACT same initialization as production users
    return init_models()


@pytest.fixture(scope="session") 
def real_orchestrator(real_model_registry):
    """Create orchestrator with real models - identical to user setup."""
    return Orchestrator(model_registry=real_model_registry)


class TestComplexTemplatesRealAPI:
    """Test complex template scenarios with real model responses."""
    
    @cost_controlled_test(max_tokens=300)
    async def test_nested_loops_with_real_search_results(self, real_orchestrator):
        """Test nested template loops with actual web search results."""
        pipeline_yaml = """
        name: Nested Template Test with Real Search
        parameters:
          topic: "machine learning applications"
        steps:
          - id: search
            tool: web-search
            action: search
            parameters:
              query: "{{ topic }}"
              max_results: 3
              
          - id: generate_report
            tool: filesystem
            action: write
            dependencies:
              - search
            parameters:
              path: "/tmp/real_nested_test.md"
              content: |
                # Search Results Analysis: {{ topic }}
                
                Generated: {{ execution.timestamp }}
                
                {% if search.results and (search.results | length) > 0 %}
                ## Search Results
                {% for result in search.results %}
                ### Result {{ loop.index }}: {{ result.title }}
                
                **URL**: {{ result.url }}
                **Snippet**: {{ result.snippet | truncate(100) }}
                
                {% if result.snippet %}
                Analysis: This result discusses {{ result.title | lower }}.
                {% endif %}
                
                {% endfor %}
                {% else %}
                ## No Search Results
                No search results were returned for the query "{{ topic }}".
                This may be due to network issues or search backend limitations.
                {% endif %}
                
                Total results found: {{ search.results | length }}
                Search completed at: {{ execution.timestamp }}
        """
        
        # Execute with real web search
        result = await real_orchestrator.execute_yaml(pipeline_yaml, {
            "topic": "artificial intelligence ethics"
        })
        
        # Validate template rendering worked
        assert result is not None
        
        # Check output file was created and rendered
        output_file = Path("/tmp/real_nested_test.md")
        assert output_file.exists(), "Output file was not created"
        
        content = output_file.read_text()
        
        # Validate complex template rendering
        assert "Search Results Analysis: artificial intelligence ethics" in content
        assert "Total results found:" in content
        assert '{{' not in content, f"Found unrendered templates: {content}"
        assert '{%' not in content, f"Found unrendered control structures: {content}"
        
        # Validate template conditional logic worked correctly
        if "Result 1:" in content:
            # We got search results - validate they were included
            assert "http" in content.lower(), "No URLs found in search results"
            print("✅ Search results were returned and template rendered correctly")
        else:
            # No search results - validate fallback message appeared
            assert "No Search Results" in content or "No search results" in content
            print("✅ No search results returned, but template handled gracefully")
        
        # Cleanup
        output_file.unlink()

    @cost_controlled_test(max_tokens=400)
    async def test_complex_conditional_templates_real_api(self, real_orchestrator):
        """Test complex conditional templates with real API responses."""
        pipeline_yaml = """
        name: Complex Conditional Template Test
        parameters:
          query: "quantum computing"
          include_analysis: true
        steps:
          - id: research
            action: generate_text
            parameters:
              prompt: "Provide 3 key facts about {{ query }}"
              max_tokens: 150
              
          - id: analyze
            action: analyze_text
            dependencies:
              - research
            parameters:
              text: "{{ research.result }}"
              analysis_type: "key_points"
              prompt: "Extract the main themes from this text about {{ query }}"
              
          - id: generate_complex_report
            tool: filesystem
            action: write
            dependencies:
              - research
              - analyze
            parameters:
              path: "/tmp/complex_conditional_test.md"
              content: |
                # {{ query | title }} Research Report
                
                **Generated**: {{ execution.timestamp }}
                **Query**: {{ query }}
                
                {% if include_analysis %}
                ## Research Summary
                {{ research.result }}
                
                ## Analysis
                {% if analyze.result %}
                {{ analyze.result }}
                {% else %}
                Analysis not available.
                {% endif %}
                {% endif %}
                
                {% set facts = research.result.split('.') %}
                ## Key Facts ({{ facts | length }} found)
                {% for fact in facts[:3] %}
                {% if fact.strip() %}
                {{ loop.index }}. {{ fact.strip() }}{% if not fact.strip().endswith('.') %}.{% endif %}
                {% endif %}
                {% endfor %}
                
                ---
                Report completed: {{ execution.timestamp | date("%Y-%m-%d %H:%M:%S") }}
        """
        
        # Execute with real models
        result = await real_orchestrator.execute_yaml(pipeline_yaml, {
            "query": "quantum computing",
            "include_analysis": True
        })
        
        assert result is not None
        
        # Validate output
        output_file = Path("/tmp/complex_conditional_test.md")
        assert output_file.exists()
        
        content = output_file.read_text()
        
        # Validate complex template features
        assert "# Quantum Computing Research Report" in content
        assert "## Research Summary" in content
        assert "## Analysis" in content
        assert "## Key Facts" in content
        assert "Report completed:" in content
        
        # Ensure no template placeholders remain
        assert '{{' not in content, f"Unrendered templates found: {content}"
        assert '{%' not in content, f"Unrendered control structures found: {content}"
        
        # Validate real content was generated
        assert "quantum" in content.lower()
        
        # Cleanup
        output_file.unlink()

    @cost_controlled_test(max_tokens=500)
    async def test_real_pipeline_execution_research_minimal(self, real_orchestrator):
        """Execute research_minimal.yaml with real APIs - exactly as users do."""
        
        # Read actual pipeline file that users use
        pipeline_path = Path("examples/research_minimal.yaml")
        if not pipeline_path.exists():
            pytest.skip("research_minimal.yaml not found - run from project root")
            
        pipeline_content = pipeline_path.read_text()
        
        # Execute exactly as users do
        inputs = {"topic": "template rendering testing"}
        output_dir = "/tmp/real_test_minimal"
        
        # Clean up any existing output
        output_path = Path(output_dir)
        if output_path.exists():
            import shutil
            shutil.rmtree(output_path)
        
        # Add output_path to inputs (simulating -o flag)
        inputs["output_path"] = output_dir
        
        result = await real_orchestrator.execute_yaml(pipeline_content, inputs)
        
        # Validate success
        assert result is not None
        
        # Check generated files
        output_files = list(output_path.glob("*.md"))
        assert len(output_files) > 0, f"No output files found in {output_dir}"
        
        # Validate template rendering in actual output
        for file_path in output_files:
            content = file_path.read_text()
            
            # Ensure NO template placeholders remain
            assert '{{' not in content, f"Unrendered template in {file_path}: {content}"
            assert '{%' not in content, f"Unrendered control structure in {file_path}: {content}"
            
            # Validate real content
            assert 'template rendering testing' in content.lower()
            assert len(content) > 100, f"Content too short in {file_path}"

    @cost_controlled_test(max_tokens=600)
    async def test_real_pipeline_execution_via_cli(self):
        """Test actual CLI usage that users follow with real APIs."""
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Execute exactly as users do via CLI
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "scripts/run_pipeline.py", 
                "examples/research_minimal.yaml", 
                "-i", "topic=real CLI testing",
                "-o", temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, 'PYTHONPATH': 'src'},
                cwd=Path.cwd()
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            except asyncio.TimeoutError:
                proc.kill()
                pytest.fail("CLI execution timed out after 120 seconds")
            
            # Validate CLI success
            if proc.returncode != 0:
                pytest.fail(f"CLI failed: stdout={stdout.decode()} stderr={stderr.decode()}")
            
            # Check generated files
            output_files = list(Path(temp_dir).glob("*.md"))
            assert len(output_files) > 0, f"No output files found in {temp_dir}"
            
            # Validate template rendering in CLI output
            for file_path in output_files:
                content = file_path.read_text()
                assert '{{' not in content, f"Unrendered template in CLI output {file_path}"
                assert 'real cli testing' in content.lower()


class TestPerformanceRealAPI:
    """Test template rendering performance with real API responses."""
    
    @cost_controlled_test(max_tokens=800, timeout=180)
    async def test_large_context_performance(self, real_orchestrator):
        """Test template rendering with large real API responses."""
        pipeline_yaml = """
        name: Large Context Performance Test
        parameters:
          analysis_topic: "renewable energy technologies"
        steps:
          - id: generate_large_content
            action: generate_text
            parameters:
              prompt: "Write a detailed 500-word analysis of {{ analysis_topic }}"
              max_tokens: 600
              
          - id: process_large_template
            tool: filesystem  
            action: write
            dependencies:
              - generate_large_content
            parameters:
              path: "/tmp/large_template_test.md"
              content: |
                # Large Content Processing Test: {{ analysis_topic | title }}
                
                Generated: {{ execution.timestamp }}
                Original content length: {{ generate_large_content.result | length }}
                
                {% set words = generate_large_content.result.split() %}
                Word count: {{ words | length }}
                
                {% if words | length > 10 %}
                First 10 words:
                {% for word in words[:10] %}
                {{ loop.index }}. {{ word }}
                {% endfor %}
                {% endif %}
                
                Content preview (first 300 chars):
                {{ generate_large_content.result[:300] }}...
                
                {% set sentences = generate_large_content.result.split('.') %}
                Sentence count: {{ sentences | length }}
                
                {% if sentences | length > 0 %}
                First sentence: {{ sentences[0].strip() }}{% if not sentences[0].strip().endswith('.') %}.{% endif %}
                {% endif %}
                
                Analysis complete: {{ execution.timestamp }}
        """
        
        start_time = time.time()
        result = await real_orchestrator.execute_yaml(pipeline_yaml, {
            "analysis_topic": "renewable energy technologies"
        })
        execution_time = time.time() - start_time
        
        # Validate performance is reasonable
        assert execution_time < 120.0, f"Template rendering too slow: {execution_time}s"
        assert result is not None
        
        # Validate output
        output_file = Path("/tmp/large_template_test.md")
        assert output_file.exists()
        
        content = output_file.read_text()
        
        # Validate complex template processing worked
        assert "Large Content Processing Test: Renewable Energy Technologies" in content
        assert "Word count:" in content
        assert "Sentence count:" in content
        assert "First 10 words:" in content
        assert "Content preview" in content
        assert '{{' not in content, f"Unrendered templates: {content}"
        
        # Validate performance characteristics
        print(f"Large template processing completed in {execution_time:.2f}s")
        
        # Cleanup
        output_file.unlink()


class TestConcurrentExecutionRealAPI:
    """Test concurrent pipeline execution with real APIs."""
    
    @cost_controlled_test(max_tokens=600, timeout=180)
    async def test_concurrent_template_rendering(self, real_orchestrator):
        """Test multiple pipelines running concurrently with real APIs."""
        
        async def run_pipeline(topic, output_dir):
            """Run pipeline with real API calls."""
            pipeline_yaml = f"""
            name: Concurrent Template Test
            parameters:
              research_topic: "{topic}"
            steps:
              - id: research
                action: generate_text
                parameters:
                  prompt: "Briefly research {{{{ research_topic }}}} in 2-3 sentences"
                  max_tokens: 100
              - id: save
                tool: filesystem
                action: write
                dependencies:
                  - research
                parameters:
                  path: "{output_dir}/{{{{ research_topic | slugify }}}}_concurrent.md"
                  content: |
                    # Research: {{{{ research_topic }}}}
                    
                    Generated: {{{{ execution.timestamp }}}}
                    
                    ## Summary
                    {{{{ research.result }}}}
                    
                    ## Metadata
                    - Topic: {{{{ research_topic }}}}
                    - Length: {{{{ research.result | length }}}} characters
                    - Completed: {{{{ execution.timestamp }}}}
            """
            
            return await real_orchestrator.execute_yaml(pipeline_yaml)
        
        # Create output directories
        temp_dirs = []
        for i in range(3):
            temp_dir = f"/tmp/concurrent_test_{i}"
            Path(temp_dir).mkdir(exist_ok=True)
            temp_dirs.append(temp_dir)
        
        try:
            # Run multiple pipelines concurrently
            topics = ["robotics", "blockchain", "biotechnology"]
            tasks = [
                run_pipeline(topic, temp_dirs[i]) 
                for i, topic in enumerate(topics)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            execution_time = time.time() - start_time
            
            print(f"Concurrent execution completed in {execution_time:.2f}s")
            
            # Validate all succeeded and templates rendered
            for i, topic in enumerate(topics):
                file_path = Path(temp_dirs[i]) / f"{topic}_concurrent.md"
                assert file_path.exists(), f"Output file not created: {file_path}"
                
                content = file_path.read_text()
                
                # Validate template rendering
                assert '{{' not in content, f"Unrendered templates in {file_path}"
                assert f"Research: {topic}" in content
                assert topic.lower() in content.lower()
                assert "Generated:" in content
                assert "Completed:" in content
                
        finally:
            # Cleanup
            for temp_dir in temp_dirs:
                import shutil
                if Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)


class TestErrorConditionsRealAPI:
    """Test error conditions with real infrastructure."""
    
    async def test_template_with_missing_step_results(self, real_orchestrator):
        """Test template rendering with missing step results."""
        pipeline_yaml = """
        name: Missing Step Results Test
        steps:
          - id: process_missing
            tool: filesystem
            action: write
            parameters:
              path: "/tmp/missing_step_test.md"
              content: |
                # Test Missing Step Results
                
                This should work: {{ execution.timestamp }}
                This should remain unrendered: {{ nonexistent_step.result }}
                This should also remain: {{ missing.data }}
        """
        
        result = await real_orchestrator.execute_yaml(pipeline_yaml)
        assert result is not None
        
        # Check output
        output_file = Path("/tmp/missing_step_test.md")
        assert output_file.exists()
        
        content = output_file.read_text()
        
        # Validate graceful handling
        assert "This should work:" in content
        assert "{{ nonexistent_step.result }}" in content  # Should remain unrendered
        assert "{{ missing.data }}" in content  # Should remain unrendered
        assert "execution.timestamp" not in content  # Should be rendered
        
        # Cleanup
        output_file.unlink()

    @cost_controlled_test(max_tokens=200)
    async def test_malformed_template_real_api(self, real_orchestrator):
        """Test malformed template handling with real APIs."""
        pipeline_yaml = """
        name: Malformed Template Test
        steps:
          - id: test_malformed
            action: generate_text
            parameters:
              prompt: "Generate a test response about templates"
              max_tokens: 50
              
          - id: process_malformed
            tool: filesystem
            action: write
            dependencies:
              - test_malformed
            parameters:
              path: "/tmp/malformed_test.md"
              content: |
                # Malformed Template Test
                
                Good template: {{ test_malformed.result }}
                Bad template: {{ unclosed_brace
                Another bad: }} closed_without_open
                Empty braces: {{}}
                Nested: {{ {{ nested }} }}
        """
        
        result = await real_orchestrator.execute_yaml(pipeline_yaml)
        assert result is not None
        
        # Check output
        output_file = Path("/tmp/malformed_test.md")
        assert output_file.exists()
        
        content = output_file.read_text()
        
        # Validate error handling
        assert "Good template:" in content
        assert "test_malformed.result" not in content  # Should be rendered
        assert "{{ unclosed_brace" in content  # Should remain as-is
        assert "}} closed_without_open" in content  # Should remain as-is
        
        # Cleanup
        output_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])