#!/usr/bin/env python3
"""Test the AUTO tag parser with complex YAML cases."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser, parse_yaml_with_auto_tags
import yaml


def test_simple_auto_tag():
    """Test simple AUTO tag parsing."""
    yaml_content = """
    key: <AUTO>Choose a value</AUTO>
    """
    
    parser = AutoTagYAMLParser()
    result = parser.parse(yaml_content)
    
    assert result['key'] == '<AUTO>Choose a value</AUTO>'
    print("✓ Simple AUTO tag test passed")


def test_auto_tag_with_colon():
    """Test AUTO tag with colon inside."""
    yaml_content = """
    output_format: <AUTO>Choose appropriate format: bullet_points, narrative, or structured</AUTO>
    """
    
    parser = AutoTagYAMLParser()
    result = parser.parse(yaml_content)
    
    assert result['output_format'] == '<AUTO>Choose appropriate format: bullet_points, narrative, or structured</AUTO>'
    print("✓ AUTO tag with colon test passed")


def test_nested_auto_tags():
    """Test nested AUTO tags."""
    yaml_content = """
    config:
      value: <AUTO>Choose <AUTO>inner value</AUTO> for outer</AUTO>
    """
    
    parser = AutoTagYAMLParser()
    result = parser.parse(yaml_content)
    
    # Should preserve the full nested structure
    assert '<AUTO>' in result['config']['value']
    assert 'inner value' in result['config']['value']
    print("✓ Nested AUTO tags test passed")


def test_complex_pipeline_yaml():
    """Test the actual failing YAML from design.md."""
    yaml_content = """# example-pipeline.yaml
name: research_report_pipeline
version: 1.0.0
description: Generate a comprehensive research report on a given topic

context:
  timeout: 3600
  max_retries: 3
  checkpoint_strategy: adaptive

steps:
  - id: topic_analysis
    action: analyze
    parameters:
      input: "{{ topic }}"
      analysis_type: <AUTO>Determine the best analysis approach for this topic</AUTO>
      output_format: <AUTO>Choose appropriate format: bullet_points, narrative, or structured</AUTO>
    
  - id: research_planning
    action: plan
    parameters:
      topic_analysis: "{{ steps.topic_analysis.output }}"
      research_depth: <AUTO>Based on topic complexity, choose: shallow, medium, or deep</AUTO>
      sources: <AUTO>Determine number and types of sources needed</AUTO>
    dependencies: [topic_analysis]
    
  - id: web_search
    action: search
    parameters:
      queries: <AUTO>Generate search queries based on research plan</AUTO>
      num_results: <AUTO>Determine optimal number of results per query</AUTO>
    dependencies: [research_planning]
    
  - id: content_synthesis
    action: synthesize
    parameters:
      sources: "{{ steps.web_search.results }}"
      style: <AUTO>Choose writing style: academic, business, or general</AUTO>
      length: <AUTO>Determine appropriate length based on topic</AUTO>
    dependencies: [web_search]
    
  - id: report_generation
    action: generate_report
    parameters:
      content: "{{ steps.content_synthesis.output }}"
      format: markdown
      sections: <AUTO>Organize content into appropriate sections</AUTO>
    dependencies: [content_synthesis]
    on_failure: retry"""
    
    parser = AutoTagYAMLParser()
    result = parser.parse(yaml_content)
    
    # Verify structure is parsed correctly
    assert result['name'] == 'research_report_pipeline'
    assert len(result['steps']) == 5
    
    # Verify AUTO tags are preserved
    assert result['steps'][0]['parameters']['output_format'] == '<AUTO>Choose appropriate format: bullet_points, narrative, or structured</AUTO>'
    assert result['steps'][1]['parameters']['research_depth'] == '<AUTO>Based on topic complexity, choose: shallow, medium, or deep</AUTO>'
    
    print("✓ Complex pipeline YAML test passed")


def test_auto_tag_with_special_chars():
    """Test AUTO tags with various special characters."""
    yaml_content = """
    special_chars:
      quotes: <AUTO>Choose "quoted" or 'single' values</AUTO>
      braces: <AUTO>Select {option1} or [option2]</AUTO>
      symbols: <AUTO>Use @, #, $, %, &, *, or other symbols</AUTO>
      newlines: <AUTO>Multi-line
        content with
        line breaks</AUTO>
    """
    
    parser = AutoTagYAMLParser()
    result = parser.parse(yaml_content)
    
    assert '"quoted"' in result['special_chars']['quotes']
    assert '{option1}' in result['special_chars']['braces']
    assert '@, #, $' in result['special_chars']['symbols']
    assert 'Multi-line' in result['special_chars']['newlines']
    
    print("✓ AUTO tags with special characters test passed")


def test_extract_auto_tag_content():
    """Test extracting content from AUTO tags."""
    parser = AutoTagYAMLParser()
    
    # Test extraction
    content = parser.extract_auto_content('<AUTO>Choose a value</AUTO>')
    assert content == 'Choose a value'
    
    # Test non-AUTO tag
    content = parser.extract_auto_content('regular value')
    assert content is None
    
    print("✓ AUTO tag content extraction test passed")


def test_find_all_auto_tags():
    """Test finding all AUTO tags in a structure."""
    data = {
        'steps': [
            {
                'id': 'step1',
                'parameters': {
                    'value': '<AUTO>Choose value</AUTO>',
                    'nested': {
                        'deep': '<AUTO>Deep value</AUTO>'
                    }
                }
            }
        ],
        'config': '<AUTO>Config value</AUTO>'
    }
    
    parser = AutoTagYAMLParser()
    auto_tags = parser.find_auto_tags(data)
    
    assert len(auto_tags) == 3
    assert ('steps[0].parameters.value', 'Choose value') in auto_tags
    assert ('steps[0].parameters.nested.deep', 'Deep value') in auto_tags
    assert ('config', 'Config value') in auto_tags
    
    print("✓ Find all AUTO tags test passed")


def test_yaml_in_auto_tags():
    """Test AUTO tags containing YAML-like syntax."""
    yaml_content = """
    complex_auto:
      format: <AUTO>Choose format:
        - option1: value1
        - option2: value2
        key: value</AUTO>
    """
    
    parser = AutoTagYAMLParser()
    result = parser.parse(yaml_content)
    
    # Should preserve the entire AUTO tag content
    assert 'option1: value1' in result['complex_auto']['format']
    assert 'option2: value2' in result['complex_auto']['format']
    
    print("✓ YAML-like content in AUTO tags test passed")


def main():
    """Run all tests."""
    print("Testing AUTO tag parser...")
    
    tests = [
        test_simple_auto_tag,
        test_auto_tag_with_colon,
        test_nested_auto_tags,
        test_complex_pipeline_yaml,
        test_auto_tag_with_special_chars,
        test_extract_auto_tag_content,
        test_find_all_auto_tags,
        test_yaml_in_auto_tags
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("\nAll tests passed! The AUTO tag parser correctly handles complex YAML.")
    else:
        print(f"\n{failed} tests failed.")
        sys.exit(1)


if __name__ == '__main__':
    main()