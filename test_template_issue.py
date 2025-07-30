#!/usr/bin/env python3
"""Test template rendering issue."""

from jinja2 import Environment, StrictUndefined, UndefinedError
import logging

logging.basicConfig(level=logging.DEBUG)

# Create test context similar to what we see in the pipeline
context = {
    'topic': 'quantum-computing-breakthroughs',
    'search_topic': {
        'query': 'quantum-computing-breakthroughs latest developments 2024 2025',
        'results': [
            {
                'title': 'Test Result 1',
                'url': 'https://example.com/1',
                'snippet': 'Test snippet 1',
                'relevance': 0.9
            },
            {
                'title': 'Test Result 2', 
                'url': 'https://example.com/2',
                'snippet': 'Test snippet 2',
                'relevance': 0.8
            }
        ],
        'total_results': 2,
        'search_time': 1.5,
        'backend': 'DuckDuckGo',
        'timestamp': '2024-01-01T12:00:00'
    },
    'deep_search': {
        'query': 'quantum-computing-breakthroughs research papers',
        'results': [],
        'total_results': 0,
        'search_time': 1.0,
        'backend': 'DuckDuckGo',
        'timestamp': '2024-01-01T12:00:00'
    },
    'extract_content': {
        'url': 'https://example.com',
        'error': 'Failed to extract',
        'scrape_time': 0.5
    },
    'analyze_findings': {
        'result': 'Test analysis'
    },
    'generate_recommendations': {
        'result': 'Test recommendations'
    }
}

# Test template that's failing
template_string = """# Research Report: {{ topic }}

**Generated on:** {{ "now" | date("%Y-%m-%d %H:%M:%S") }}
**Total Sources Analyzed:** {{ search_topic.total_results + deep_search.total_results }}

## Search Results

### Primary Search Results ({{ search_topic.total_results }} found)
{% for result in search_topic.results[:10] %}
### {{ loop.index }}. {{ result.title }}
**URL:** [{{ result.url }}]({{ result.url }})
**Relevance:** {{ result.relevance }}
**Summary:** {{ result.snippet }}

{% endfor %}

## Extracted Content Analysis

{% if extract_content.success %}
**Primary Source:** {{ extract_content.title }}
**URL:** {{ extract_content.url }}
**Content Summary:** Successfully extracted {{ extract_content.word_count }} words from the primary source.
{% else %}
Content extraction was not successful or was skipped.
{% endif %}
"""

# Set up Jinja2 environment
env = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)

# Add date filter
from datetime import datetime
def date_filter(value, format_str="%Y-%m-%d %H:%M:%S"):
    if value == 'now':
        return datetime.now().strftime(format_str)
    return value

env.filters['date'] = date_filter

# Try to render
try:
    template = env.from_string(template_string)
    result = template.render(context)
    print("SUCCESS! Template rendered:")
    print(result)
except UndefinedError as e:
    print(f"UndefinedError: {e}")
    print(f"Error type: {type(e)}")
    print(f"Error message: {str(e)}")
except Exception as e:
    print(f"Other error: {e}")
    print(f"Error type: {type(e)}")