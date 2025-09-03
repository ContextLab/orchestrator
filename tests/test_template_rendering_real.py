"""
Test template rendering with real API calls - NO MOCKS
"""
import asyncio
import os
import sys
from pathlib import Path
import pytest
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.core.pipeline import Pipeline

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider


def create_pipeline_from_yaml(yaml_content):
    """Helper to create pipeline from YAML string"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_file = f.name
    
    pipeline = Pipeline.from_yaml(temp_file)
    os.unlink(temp_file)
    return pipeline


@pytest.mark.asyncio
async def test_basic_template_rendering():
    """Test that templates render correctly in all contexts."""
    
    yaml_content = """
id: test-template-render
parameters:
  message:
    type: string
    default: "Test message"
  output_dir:
    type: string
    default: "examples/outputs/test_template"

steps:
  - id: generate_content
    action: generate_text
    parameters:
      prompt: "Say exactly: SUCCESS"
      max_tokens: 10
      
  - id: save_result
    tool: filesystem
    action: write
    dependencies:
      - generate_content
    parameters:
      path: "{{ output_dir }}/result.txt"
      content: |
        Message: {{ message }}
        Result: {{ generate_content }}
        Done!
"""
    
    # Create pipeline and orchestrator
    pipeline = create_pipeline_from_yaml(yaml_content)
    orchestrator = create_test_orchestrator()
    
    # Execute with parameters
    result = await orchestrator.execute(
        pipeline,
        parameters={
            "message": "Testing template rendering",
            "output_dir": "examples/outputs/test_basic_template"
        }
    )
    
    # Check that execution succeeded
    assert "generate_content" in result
    assert "save_result" in result
    
    # Check the saved file
    output_file = Path("examples/outputs/test_basic_template/result.txt")
    if output_file.exists():
        content = output_file.read_text()
        
        # Verify no template placeholders remain
        assert "{{" not in content
        assert "{%" not in content
        
        # Verify our message is there
        assert "Testing template rendering" in content
        
        # Verify result is there (not the template)
        assert "{{ generate_content }}" not in content
        
        print(f"✅ Basic template rendering test passed!")
        print(f"File content:\n{content}")
    else:
        print(f"❌ Output file not created: {output_file}")
        assert False, "Output file was not created"


@pytest.mark.asyncio
async def test_loop_context_access():
    """Test that loops can access pipeline parameters."""
    
    yaml_content = """
id: test-loop-context
parameters:
  base_text:
    type: string
    default: "Artificial Intelligence"
  languages:
    type: list
    default: ["es", "fr"]

steps:
  - id: translate_all
    for_each: "{{ languages }}"
    steps:
      - id: translate
        action: generate_text
        parameters:
          prompt: "Translate '{{ base_text }}' to {{ $item }}. Reply with just the translation."
          max_tokens: 50
"""
    
    pipeline = create_pipeline_from_yaml(yaml_content)
    orchestrator = create_test_orchestrator()
    
    result = await orchestrator.execute(
        pipeline,
        parameters={
            "base_text": "Climate Change",
            "languages": ["es"]
        }
    )
    
    # Check that translation happened
    assert "translate_all" in result or any("translate" in key for key in result.keys())
    
    # The result structure might be flattened
    translation_found = False
    for key, value in result.items():
        if "translate" in key:
            translation_found = True
            # Check that it's not asking for the text
            if isinstance(value, str):
                assert "Please provide" not in value
                assert "Climate Change" not in value or "Cambio" in value  # Either not English or contains Spanish
                print(f"✅ Translation result for {key}: {value[:100]}")
    
    assert translation_found, "No translation results found"
    print("✅ Loop context access test passed!")


@pytest.mark.asyncio  
async def test_complex_nested_templates():
    """Test conditional templates and nested references."""
    
    yaml_content = """
id: test-nested
parameters:
  analyze_text:
    type: string
    default: "Technology advances rapidly"

steps:
  - id: initial_analysis
    action: generate_text
    parameters:
      prompt: "Rate the clarity of this text from 0 to 1: '{{ analyze_text }}'. Reply with just a decimal number."
      max_tokens: 10
      
  - id: enhance
    action: generate_text
    dependencies:
      - initial_analysis
    parameters:
      prompt: |
        {% if initial_analysis %}
        Based on rating {{ initial_analysis }}, improve: {{ analyze_text }}
        {% else %}
        Create new analysis for: {{ analyze_text }}
        {% endif %}
      max_tokens: 100
"""
    
    pipeline = create_pipeline_from_yaml(yaml_content)
    orchestrator = create_test_orchestrator()
    
    result = await orchestrator.execute(pipeline)
    
    # Check both steps executed
    assert "initial_analysis" in result
    assert "enhance" in result
    
    # Verify no template syntax in results
    enhance_result = str(result.get("enhance", ""))
    assert "{%" not in enhance_result
    assert "{{" not in enhance_result
    
    # The enhance step should have actual content
    assert len(enhance_result) > 20
    
    print(f"✅ Initial analysis: {result['initial_analysis']}")
    print(f"✅ Enhancement (first 200 chars): {enhance_result[:200]}")
    print("✅ Complex nested templates test passed!")


@pytest.mark.asyncio
async def test_filesystem_with_templates():
    """Test filesystem operations with template rendering."""
    
    yaml_content = """
id: test-filesystem
parameters:
  report_title:
    type: string
    default: "Test Report"
  output_dir:
    type: string
    default: "examples/outputs/test_filesystem"

steps:
  - id: analyze
    action: generate_text
    parameters:
      prompt: "Write one sentence about renewable energy"
      max_tokens: 50
      
  - id: save_report
    tool: filesystem
    action: write
    dependencies:
      - analyze
    parameters:
      path: "{{ output_dir }}/report.md"
      content: |
        # {{ report_title }}
        
        ## Analysis
        {{ analyze }}
        
        Report generated successfully.
"""
    
    pipeline = create_pipeline_from_yaml(yaml_content)
    orchestrator = create_test_orchestrator()
    
    result = await orchestrator.execute(
        pipeline,
        parameters={
            "report_title": "Energy Analysis Report",
            "output_dir": "examples/outputs/test_filesystem_render"
        }
    )
    
    # Check execution
    assert "analyze" in result
    assert "save_report" in result
    
    # Check the file
    report_file = Path("examples/outputs/test_filesystem_render/report.md")
    if report_file.exists():
        content = report_file.read_text()
        
        # No templates should remain
        assert "{{ report_title }}" not in content
        assert "{{ analyze }}" not in content
        
        # Content should be there
        assert "Energy Analysis Report" in content
        assert "Report generated successfully" in content
        
        print(f"✅ Filesystem template test passed!")
        print(f"Report content:\n{content}")
    else:
        assert False, f"Report file not created: {report_file}"


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_basic_template_rendering())
    asyncio.run(test_loop_context_access())
    asyncio.run(test_complex_nested_templates())
    asyncio.run(test_filesystem_with_templates())
    
    print("\n✅ All template rendering tests passed!")