"""Test the AUTO tag parser with complex YAML cases."""

from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser, parse_yaml_with_auto_tags


class TestAutoTagYAMLParser:
    """Test suite for AUTO tag YAML parser."""
    
    def test_simple_auto_tag(self):
        """Test simple AUTO tag parsing."""
        yaml_content = """
        key: <AUTO>Choose a value</AUTO>
        """
        
        parser = AutoTagYAMLParser()
        result = parser.parse(yaml_content)
        
        assert result['key'] == '<AUTO>Choose a value</AUTO>'
    
    def test_auto_tag_with_colon(self):
        """Test AUTO tag with colon inside."""
        yaml_content = """
        output_format: <AUTO>Choose appropriate format: bullet_points, narrative, or structured</AUTO>
        """
        
        parser = AutoTagYAMLParser()
        result = parser.parse(yaml_content)
        
        assert result['output_format'] == '<AUTO>Choose appropriate format: bullet_points, narrative, or structured</AUTO>'
    
    def test_nested_auto_tags(self):
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
    
    def test_complex_pipeline_yaml(self):
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
    
    def test_auto_tag_with_special_chars(self):
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
    
    def test_extract_auto_tag_content(self):
        """Test extracting content from AUTO tags."""
        parser = AutoTagYAMLParser()
        
        # Test extraction
        content = parser.extract_auto_content('<AUTO>Choose a value</AUTO>')
        assert content == 'Choose a value'
        
        # Test non-AUTO tag
        content = parser.extract_auto_content('regular value')
        assert content is None
    
    def test_find_all_auto_tags(self):
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
    
    def test_yaml_in_auto_tags(self):
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
    
    def test_empty_auto_tags(self):
        """Test handling of empty AUTO tags."""
        yaml_content = """
        empty: <AUTO></AUTO>
        whitespace: <AUTO>   </AUTO>
        """
        
        parser = AutoTagYAMLParser()
        result = parser.parse(yaml_content)
        
        assert result['empty'] == '<AUTO></AUTO>'
        assert result['whitespace'] == '<AUTO>   </AUTO>'
    
    def test_auto_tags_in_lists(self):
        """Test AUTO tags within list structures."""
        yaml_content = """
        items:
          - <AUTO>First item</AUTO>
          - regular item
          - <AUTO>Third item</AUTO>
        nested:
          - name: test
            value: <AUTO>Nested value</AUTO>
        """
        
        parser = AutoTagYAMLParser()
        result = parser.parse(yaml_content)
        
        assert result['items'][0] == '<AUTO>First item</AUTO>'
        assert result['items'][1] == 'regular item'
        assert result['items'][2] == '<AUTO>Third item</AUTO>'
        assert result['nested'][0]['value'] == '<AUTO>Nested value</AUTO>'
    
    def test_parse_yaml_with_auto_tags_function(self):
        """Test the convenience function."""
        yaml_content = """
        test: <AUTO>Test value</AUTO>
        """
        
        result = parse_yaml_with_auto_tags(yaml_content)
        assert result['test'] == '<AUTO>Test value</AUTO>'