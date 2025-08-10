"""
Template Rendering Debug Tests - Step by step debugging

Debug the template rendering issues to understand what's happening.
"""

import pytest
import tempfile
from pathlib import Path
from orchestrator import Orchestrator, init_models


@pytest.fixture(scope="session")
def real_orchestrator():
    """Create orchestrator with real models for debugging."""
    import os
    
    available_keys = [k for k in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY'] if os.getenv(k)]
    if not available_keys:
        pytest.skip("No API keys available for debugging")
    
    registry = init_models()
    return Orchestrator(model_registry=registry)


class TestTemplateDebugging:
    """Debug template rendering step by step."""
    
    async def test_simple_template_rendering(self, real_orchestrator):
        """Test the simplest possible template rendering."""
        pipeline_yaml = """
        name: Simple Template Test
        parameters:
          test_param: "hello world"
        steps:
          - id: save_simple
            tool: filesystem
            action: write
            parameters:
              path: "/tmp/simple_template_test.md"
              content: |
                # Simple Test
                Parameter: {{ test_param }}
                Timestamp: {{ execution.timestamp }}
        """
        
        result = await real_orchestrator.execute_yaml(pipeline_yaml, {
            "test_param": "debugging templates"
        })
        
        assert result is not None
        
        # Check output
        output_file = Path("/tmp/simple_template_test.md")
        assert output_file.exists()
        
        content = output_file.read_text()
        print(f"Simple template output:\n{content}")
        
        # Validate template rendering
        assert "Parameter: debugging templates" in content
        assert "Timestamp:" in content and "execution.timestamp" not in content
        assert '{{' not in content, f"Unrendered templates: {content}"
        
        # Cleanup
        output_file.unlink()

    async def test_step_result_template_rendering(self, real_orchestrator):
        """Test template rendering with step results."""
        pipeline_yaml = """
        name: Step Result Template Test
        steps:
          - id: generate_content
            action: generate_text
            parameters:
              prompt: "Write exactly this: 'Test content for template debugging'"
              max_tokens: 20
              
          - id: save_with_step_result
            tool: filesystem
            action: write
            parameters:
              path: "/tmp/step_result_test.md"
              content: |
                # Step Result Test
                Generated content: {{ generate_content.result }}
                Timestamp: {{ execution.timestamp }}
            dependencies:
              - generate_content
        """
        
        result = await real_orchestrator.execute_yaml(pipeline_yaml)
        
        assert result is not None
        print(f"Pipeline result keys: {list(result.keys())}")
        
        # Check what the generate_content step produced
        if 'generate_content' in result:
            print(f"generate_content result: {result['generate_content']}")
        
        # Check output file
        output_file = Path("/tmp/step_result_test.md")
        assert output_file.exists()
        
        content = output_file.read_text()
        print(f"Step result template output:\n{content}")
        
        # Validate template rendering
        assert "Generated content:" in content
        assert "generate_content.result" not in content, "Step result template not rendered"
        assert "Test content" in content or "test content" in content.lower()
        assert '{{' not in content, f"Unrendered templates: {content}"
        
        # Cleanup
        output_file.unlink()

    async def test_web_search_debug(self, real_orchestrator):
        """Debug web search specifically."""
        pipeline_yaml = """
        name: Web Search Debug Test
        steps:
          - id: debug_search
            tool: web-search
            action: search
            parameters:
              query: "python programming"
              max_results: 2
              
          - id: debug_save
            tool: filesystem
            action: write
            parameters:
              path: "/tmp/web_search_debug.md"
              content: |
                # Web Search Debug
                
                Search completed: {{ execution.timestamp }}
                
                Raw search result (for debugging): {{ debug_search }}
                
                Result count: {{ debug_search.results | length if debug_search.results else 0 }}
                
                {% if debug_search.results %}
                First result: {{ debug_search.results[0] }}
                {% else %}
                No results found
                {% endif %}
        """
        
        result = await real_orchestrator.execute_yaml(pipeline_yaml)
        
        assert result is not None
        print(f"Web search pipeline result keys: {list(result.keys())}")
        
        # Check what the search step produced
        if 'debug_search' in result:
            search_result = result['debug_search']
            print(f"debug_search result type: {type(search_result)}")
            print(f"debug_search result: {search_result}")
            
            if hasattr(search_result, 'results'):
                print(f"Search results count: {len(search_result.results)}")
            elif isinstance(search_result, dict) and 'results' in search_result:
                print(f"Search results count (dict): {len(search_result['results'])}")
        
        # Check output file
        output_file = Path("/tmp/web_search_debug.md")
        assert output_file.exists()
        
        content = output_file.read_text()
        print(f"Web search debug output:\n{content}")
        
        # Don't assert template rendering worked - we're debugging
        # Just check that the file was created and has some content
        assert len(content) > 50  # Some reasonable content length
        
        # Cleanup
        output_file.unlink()

    async def test_template_context_debugging(self, real_orchestrator):
        """Debug what's actually in the template context."""
        # Enable debug mode in template manager
        real_orchestrator.template_manager.debug_mode = True
        
        pipeline_yaml = """
        name: Context Debug Test
        parameters:
          debug_param: "context debugging"
        steps:
          - id: first_step
            action: generate_text
            parameters:
              prompt: "Say 'first step complete'"
              max_tokens: 10
              
          - id: context_debug
            tool: filesystem
            action: write
            parameters:
              path: "/tmp/context_debug.md"
              content: |
                # Template Context Debug
                
                Parameter: {{ debug_param }}
                First step: {{ first_step.result }}
                Timestamp: {{ execution.timestamp }}
                
                Debug complete.
        """
        
        result = await real_orchestrator.execute_yaml(pipeline_yaml, {
            "debug_param": "debugging context"
        })
        
        # Reset debug mode
        real_orchestrator.template_manager.debug_mode = False
        
        assert result is not None
        print(f"Context debug pipeline result: {result}")
        
        # Check output
        output_file = Path("/tmp/context_debug.md")
        assert output_file.exists()
        
        content = output_file.read_text()
        print(f"Context debug output:\n{content}")
        
        # Cleanup
        output_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])