#!/usr/bin/env python3
"""
Add file output support to YAML pipelines.
This demonstrates how to save outputs to markdown files.
"""

import yaml
import re
from pathlib import Path
from typing import Dict, Any, List


def add_file_output_step(yaml_content: str, output_dir: str = "examples/output") -> str:
    """
    Add a file output step to save results to markdown.
    
    Args:
        yaml_content: Original YAML content
        output_dir: Directory to save outputs
    
    Returns:
        Modified YAML content with file output step
    """
    # Parse YAML
    pipeline = yaml.safe_load(yaml_content)
    
    # Get pipeline name for filename
    pipeline_name = pipeline.get('name', 'pipeline')
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', pipeline_name.lower())
    
    # Create save step
    save_step = {
        'id': 'save_to_file',
        'action': f"""
Save all results to a markdown file.

Create a markdown file at: {output_dir}/{safe_name}_output.md

The file should contain:
1. Pipeline name and description
2. Execution timestamp
3. Input parameters used
4. Results from each step with proper formatting
5. Any outputs defined in the pipeline

Format the content as a well-structured markdown document with:
- H1 heading for the pipeline name
- H2 headings for each major section
- Code blocks for any code or data
- Lists and tables where appropriate

Include all relevant results from previous steps:
""",
        'depends_on': []
    }
    
    # Add references to all previous steps
    if 'steps' in pipeline:
        for step in pipeline['steps']:
            step_id = step.get('id', '')
            if step_id and step_id != 'save_to_file':
                save_step['depends_on'].append(step_id)
                save_step['action'] += f"\n- {step_id}: {{{{{step_id}.result}}}}"
    
    # Add the save step
    if 'steps' not in pipeline:
        pipeline['steps'] = []
    pipeline['steps'].append(save_step)
    
    # Convert back to YAML
    return yaml.dump(pipeline, default_flow_style=False, sort_keys=False)


def update_pipeline_with_file_output(yaml_file: str, output_dir: str = "examples/output"):
    """Update a pipeline YAML file to include file output."""
    yaml_path = Path(yaml_file)
    
    # Read original YAML
    with open(yaml_path, 'r') as f:
        original_yaml = f.read()
    
    # Add file output
    updated_yaml = add_file_output_step(original_yaml, output_dir)
    
    # Save to new file
    output_path = yaml_path.parent / f"{yaml_path.stem}_with_output.yaml"
    with open(output_path, 'w') as f:
        f.write(updated_yaml)
    
    print(f"Updated pipeline saved to: {output_path}")
    
    return output_path


# Example of how to add file writing directly in YAML
example_with_file_output = """
name: "Example Pipeline with File Output"
description: "Shows how to save outputs to files"

inputs:
  topic:
    type: string
    required: true

steps:
  - id: process_data
    action: |
      Process the topic: {{topic}}
      Generate some interesting insights and analysis.
      
  - id: create_report
    action: |
      Create a comprehensive report based on: {{process_data.result}}
      Format it as markdown with proper sections.
      
  - id: save_output
    action: |
      Write the following content to a file at examples/output/{{topic}}_report.md:
      
      # Report: {{topic}}
      
      Generated on: {{execution.timestamp}}
      
      ## Analysis Results
      
      {{process_data.result}}
      
      ## Full Report
      
      {{create_report.result}}
      
      ---
      *This report was generated automatically by the Orchestrator pipeline.*
    depends_on: [process_data, create_report]
    
outputs:
  report_content: "{{create_report.result}}"
  file_path: "examples/output/{{topic}}_report.md"
"""


if __name__ == "__main__":
    # Example usage
    print("Example YAML with file output:")
    print(example_with_file_output)
    
    # You can use this to update existing pipelines
    # update_pipeline_with_file_output("examples/research_assistant.yaml")