#!/usr/bin/env python3
"""Fix documentation issues causing test failures."""

import re
from pathlib import Path
import sys

def fix_await_in_docs():
    """Fix await usage in documentation to be properly wrapped in async functions."""
    
    fixes = [
        # Basic concepts - multiple await issues
        {
            'file': 'docs/getting_started/basic_concepts.rst',
            'pattern': r'(\s*)result = await orchestrator\.execute_pipeline\(pipeline\)',
            'replacement': r'\1import asyncio\n\1\n\1async def run_pipeline():\n\1    result = await orchestrator.execute_pipeline(pipeline)\n\1    return result\n\1\n\1# Run the pipeline\n\1result = asyncio.run(run_pipeline())'
        },
        # Quickstart - multiple await issues
        {
            'file': 'docs/getting_started/quickstart.rst',
            'pattern': r'(\s*)result = await orchestrator\.execute_pipeline\(pipeline\)',
            'replacement': r'\1import asyncio\n\1\n\1async def run_pipeline():\n\1    result = await orchestrator.execute_pipeline(pipeline)\n\1    return result\n\1\n\1# Run the pipeline\n\1result = asyncio.run(run_pipeline())'
        },
        # Your first pipeline
        {
            'file': 'docs/getting_started/your_first_pipeline.rst',
            'pattern': r'(\s*)result = await orchestrator\.execute_pipeline\(pipeline\)',
            'replacement': r'\1import asyncio\n\1\n\1async def run_pipeline():\n\1    result = await orchestrator.execute_pipeline(pipeline)\n\1    return result\n\1\n\1# Run the pipeline\n\1result = asyncio.run(run_pipeline())'
        },
    ]
    
    for fix in fixes:
        file_path = Path(fix['file'])
        if file_path.exists():
            content = file_path.read_text()
            # Check if already fixed
            if 'asyncio.run' not in content:
                new_content = re.sub(fix['pattern'], fix['replacement'], content)
                if new_content != content:
                    file_path.write_text(new_content)
                    print(f"Fixed await in {fix['file']}")

def fix_notebook_await_examples():
    """Mark notebook-specific await examples as notebook code."""
    notebook_readme = Path('notebooks/README.md')
    if notebook_readme.exists():
        content = notebook_readme.read_text()
        # This is valid in Jupyter notebooks, so we should mark it differently
        # For now, we'll leave it as is but handle it in the test generation
        print("Note: notebooks/README.md contains valid Jupyter notebook await examples")

def fix_installation_commands():
    """Fix installation command issues."""
    files_to_fix = [
        ('docs/getting_started/installation.rst', 'playwright install chromium', '# Install Playwright browser\nplaywright install chromium'),
        ('docs_sphinx/installation.rst', 'playwright install chromium', '# Install Playwright browser\nplaywright install chromium'),
    ]
    
    for file_path, old_cmd, new_cmd in files_to_fix:
        path = Path(file_path)
        if path.exists():
            content = path.read_text()
            if old_cmd in content and not new_cmd in content:
                content = content.replace(old_cmd, new_cmd)
                path.write_text(content)
                print(f"Fixed command in {file_path}")

def fix_yaml_template_examples():
    """Fix YAML template syntax examples."""
    yaml_pipelines = Path('docs_sphinx/yaml_pipelines.rst')
    if yaml_pipelines.exists():
        content = yaml_pipelines.read_text()
        
        # Fix the filters section that has bare template strings
        if '"{{ inputs.topic | lower }}"' in content:
            # This section should be a proper YAML example
            old_section = '''# String manipulation
"{{ inputs.topic | lower }}"
"{{ inputs.topic | upper }}"
"{{ inputs.topic | slugify }}"
"{{ inputs.topic | replace(' ', '_') }}"

# Date formatting
"{{ execution.timestamp | strftime('%Y-%m-%d') }}"

# Math operations
"{{ inputs.count * 2 }}"
"{{ inputs.value | round(2) }}"

# Conditionals
"{{ 'premium' if inputs.tier == 'gold' else 'standard' }}"

# Lists and loops
"{{ inputs.items | join(', ') }}"
"{{ inputs.sources | length }}"'''
            
            new_section = '''steps:
  - id: template_examples
    action: process_data
    parameters:
      # String manipulation
      lowercase_topic: "{{ inputs.topic | lower }}"
      uppercase_topic: "{{ inputs.topic | upper }}"
      slug_topic: "{{ inputs.topic | slugify }}"
      underscore_topic: "{{ inputs.topic | replace(' ', '_') }}"
      
      # Date formatting
      formatted_date: "{{ execution.timestamp | strftime('%Y-%m-%d') }}"
      
      # Math operations
      double_count: "{{ inputs.count * 2 }}"
      rounded_value: "{{ inputs.value | round(2) }}"
      
      # Conditionals
      tier_label: "{{ 'premium' if inputs.tier == 'gold' else 'standard' }}"
      
      # Lists and loops
      items_string: "{{ inputs.items | join(', ') }}"
      source_count: "{{ inputs.sources | length }}"'''
            
            content = content.replace(old_section, new_section)
            yaml_pipelines.write_text(content)
            print("Fixed YAML template examples in yaml_pipelines.rst")

def fix_tutorial_yaml_comments():
    """Fix tutorial YAML that only contains comments."""
    tutorial_data = Path('docs_sphinx/tutorials/tutorial_data_processing.rst')
    if tutorial_data.exists():
        content = tutorial_data.read_text()
        
        # Fix the second exercise that only has comments
        if '# Requirements:\n   # - Handle real-time market data' in content:
            old_section = '''.. code-block:: yaml

   # Requirements:
   # - Handle real-time market data
   # - Multi-step validation
   # - Risk calculations
   # - Automated reporting'''
            
            new_section = '''.. code-block:: text

   Requirements:
   - Handle real-time market data
   - Multi-step validation
   - Risk calculations
   - Automated reporting'''
            
            content = content.replace(old_section, new_section)
            tutorial_data.write_text(content)
            print("Fixed tutorial YAML comments in tutorial_data_processing.rst")

def main():
    """Run all fixes."""
    print("Fixing documentation issues...")
    fix_await_in_docs()
    fix_notebook_await_examples()
    fix_installation_commands()
    fix_yaml_template_examples()
    fix_tutorial_yaml_comments()
    print("\nDone! Re-run extract_code_snippets_improved.py and regenerate tests.")

if __name__ == '__main__':
    main()