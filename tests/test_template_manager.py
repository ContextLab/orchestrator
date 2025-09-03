"""Tests for the TemplateManager system."""

import pytest
from datetime import datetime
from src.orchestrator.core.template_manager import TemplateManager, DeferredTemplate, ChainableUndefined


class TestTemplateManager:
    """Test the TemplateManager class."""
    
    def test_basic_template_rendering(self):
        """Test basic template rendering with simple variables."""
        tm = TemplateManager()
        tm.register_context('name', 'Alice')
        tm.register_context('age', 30)
        
        result = tm.render('Hello {{name}}, you are {{age}} years old!')
        assert result == 'Hello Alice, you are 30 years old!'
    
    def test_undefined_variable_handling(self):
        """Test that undefined variables are handled gracefully."""
        tm = TemplateManager()
        tm.register_context('name', 'Alice')
        
        # Should return original template when variable is undefined
        result = tm.render('Hello {{name}}, your score is {{score}}!')
        assert '{{score}}' in result  # Undefined variable becomes placeholder
        assert 'Alice' in result  # Defined variable is rendered
    
    def test_nested_object_access(self):
        """Test accessing nested object properties."""
        tm = TemplateManager()
        
        # Test with dict
        search_result = {
            'results': [
                {'title': 'Test Article', 'url': 'https://example.com'},
                {'title': 'Another Article', 'url': 'https://example2.com'}
            ],
            'total_results': 2
        }
        tm.register_context('search', search_result)
        
        result = tm.render('Found {{search.total_results}} results. First: {{search.results[0].title}}')
        assert result == 'Found 2 results. First: Test Article'
    
    def test_string_result_wrapper(self):
        """Test that string results get .result attribute for compatibility."""
        tm = TemplateManager()
        tm.register_context('analysis', 'This is the analysis result')
        
        # Should work with .result attribute
        result = tm.render('Analysis: {{analysis.result}}')
        assert result == 'Analysis: This is the analysis result'
        
        # Should also work without .result
        result2 = tm.render('Analysis: {{analysis}}')
        assert result2 == 'Analysis: This is the analysis result'
    
    def test_custom_filters(self):
        """Test custom Jinja2 filters."""
        tm = TemplateManager()
        tm.register_context('topic', 'Machine Learning Applications!')
        tm.register_context('text', 'This is a long text that needs to be truncated because it has many words in it')
        tm.register_context('data', {'key': 'value', 'number': 42})
        
        # Test slugify
        result = tm.render('{{topic | slugify}}')
        assert result == 'machine-learning-applications'
        
        # Test truncate_words  
        result = tm.render('{{text | truncate_words(5)}}')
        assert result == 'This is a long text...'
        
        # Test to_json
        result = tm.render('{{data | to_json}}')
        assert '"key": "value"' in result
        assert '"number": 42' in result
    
    def test_date_filter(self):
        """Test date formatting filter."""
        tm = TemplateManager()
        
        # Test 'now' keyword
        result = tm.render('{{\"now\" | date(\"%Y\")}}')
        current_year = str(datetime.now().year)
        assert result == current_year
        
        # Test ISO format
        tm.register_context('timestamp', '2023-12-25T10:30:00')
        result = tm.render('{{timestamp | date(\"%B %d, %Y\")}}')
        assert result == 'December 25, 2023'
    
    def test_render_dict(self):
        """Test rendering all templates in a dictionary."""
        tm = TemplateManager()
        tm.register_context('topic', 'AI Research')
        tm.register_context('count', 5)
        
        data = {
            'title': 'Report on {{topic}}',
            'description': 'Found {{count}} articles',
            'nested': {
                'path': 'reports/{{topic | slugify}}.md'
            },
            'tags': ['{{topic}}', 'research', '{{count}} items'],
            'number': 42  # Non-string should be unchanged
        }
        
        result = tm.render_dict(data)
        
        assert result['title'] == 'Report on AI Research'
        assert result['description'] == 'Found 5 articles'
        assert result['nested']['path'] == 'reports/ai-research.md'
        assert result['tags'] == ['AI Research', 'research', '5 items']
        assert result['number'] == 42
    
    def test_register_all_results(self):
        """Test registering multiple results at once."""
        tm = TemplateManager()
        
        results = {
            'search': {'total_results': 10, 'query': 'test'},
            'analysis': 'This is the analysis',
            'summary': {'result': 'Summary text', 'word_count': 50}
        }
        
        tm.register_all_results(results)
        
        # Test individual access
        template = 'Query: {{search.query}}, Analysis: {{analysis.result}}, Summary: {{summary.result}}'
        result = tm.render(template)
        assert result == 'Query: test, Analysis: This is the analysis, Summary: Summary text'
        
        # Test previous_results access
        result2 = tm.render('Total: {{previous_results.search.total_results}}')
        assert result2 == 'Total: 10'
    
    def test_has_templates(self):
        """Test template detection."""
        tm = TemplateManager()
        
        assert tm.has_templates('Hello {{name}}!')
        assert tm.has_templates('{% for item in items %}{{item}}{% endfor %}')
        assert not tm.has_templates('Hello world!')
        assert not tm.has_templates('')
        assert not tm.has_templates(42)
    
    def test_deferred_template(self):
        """Test deferred template functionality."""
        tm = TemplateManager()
        
        # Create deferred template before context is available
        deferred = tm.defer_render('Hello {{name}}!')
        
        # Register context later
        tm.register_context('name', 'Bob')
        
        # Now render should work
        result = deferred.render()
        assert result == 'Hello Bob!'
    
    def test_debug_mode(self):
        """Test debug mode functionality."""
        tm = TemplateManager(debug_mode=True)
        tm.register_context('test', 'value')
        
        debug_info = tm.get_debug_info()
        assert debug_info['debug_mode'] is True
        assert 'test' in debug_info['context_keys']
        assert debug_info['context_types']['test'] == 'StringResultWrapper'
    
    def test_chainable_undefined(self):
        """Test ChainableUndefined behavior."""
        tm = TemplateManager()
        
        # Should handle chained access on undefined variables
        result = tm.render('{{undefined.property.subproperty}}')
        assert '{{undefined.property.subproperty}}' in result
        
        # Should handle indexing on undefined variables  
        result = tm.render('{{undefined[0].title}}')
        assert '{{undefined[0].title}}' in result
        
        # Should handle iteration
        result = tm.render('{% for item in undefined %}{{item}}{% endfor %}')
        assert result == ''  # Empty iteration
    
    def test_complex_real_world_scenario(self):
        """Test a complex real-world template scenario."""
        tm = TemplateManager()
        
        # Simulate pipeline results
        search_results = {
            'results': [
                {
                    'title': 'Machine Learning Advances in 2024',
                    'url': 'https://example.com/ml-2024',
                    'snippet': 'Recent advances in ML include...',
                    'relevance': 0.95
                },
                {
                    'title': 'AI Ethics and Safety',
                    'url': 'https://example.com/ai-ethics', 
                    'snippet': 'AI safety considerations...',
                    'relevance': 0.87
                }
            ],
            'total_results': 2,
            'query': 'machine learning 2024'
        }
        
        analysis_result = "The research shows significant progress in machine learning applications."
        
        tm.register_context('search_topic', search_results)
        tm.register_context('generate_analysis', analysis_result)
        tm.register_context('topic', 'Machine Learning')
        
        # Complex template like those used in pipelines
        template = '''# Research Report: {{topic}}

**Sources Analyzed:** {{search_topic.total_results}} sources

## Analysis
{{generate_analysis.result}}

## Sources
{% for result in search_topic.results %}
{{loop.index}}. [{{result.title}}]({{result.url}})
   Relevance: {{result.relevance}}
   Summary: {{result.snippet}}
{% endfor %}

---
*Report generated for query: "{{search_topic.query}}"*'''
        
        result = tm.render(template)
        
        # Verify key components are rendered correctly
        assert '# Research Report: Machine Learning' in result
        assert '**Sources Analyzed:** 2 sources' in result
        assert 'The research shows significant progress' in result
        assert '1. [Machine Learning Advances in 2024]' in result
        assert '2. [AI Ethics and Safety]' in result
        assert 'Relevance: 0.95' in result
        assert 'query: "machine learning 2024"' in result
    
    def test_clear_context(self):
        """Test context clearing functionality."""
        tm = TemplateManager()
        tm.register_context('test', 'value')
        
        assert 'test' in tm.context
        
        tm.clear_context()
        
        assert 'test' not in tm.context
        assert 'timestamp' in tm.context  # Base context should remain