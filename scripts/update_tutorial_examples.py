#!/usr/bin/env python3
"""Update all tutorial examples to use the new declarative YAML framework."""

import os
import re

# Template for updated tutorial documentation
TUTORIAL_TEMPLATE = """
{title}
{'=' * len(title)}

This example demonstrates how to build {description} using the Orchestrator's declarative YAML framework. {details} - all defined in pure YAML with no custom Python code required.

.. note::
   **Level:** {level}  
   **Duration:** {duration}  
   **Prerequisites:** Orchestrator framework installed, API keys configured

Overview
--------

{overview}

**Key Features Demonstrated:**
- Declarative YAML pipeline definition
- AUTO tag resolution for natural language task descriptions
- {features}
- No Python code required

Quick Start
-----------

.. code-block:: bash

   # Set up environment variables
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   
   # Run the pipeline
   orchestrator run examples/{filename}.yaml {example_args}

Complete YAML Pipeline
----------------------

The complete pipeline is defined in ``examples/{filename}.yaml``. Here are the key sections:

**Pipeline Inputs:**

.. code-block:: yaml

   inputs:
{inputs_yaml}

**Pipeline Steps:**

.. code-block:: yaml

   steps:
{steps_yaml}

**Pipeline Outputs:**

.. code-block:: yaml

   outputs:
{outputs_yaml}

How It Works
------------

{how_it_works}

Running the Pipeline
--------------------

**Using the CLI:**

.. code-block:: bash

   # Basic usage
   orchestrator run {filename}.yaml {basic_example}

   # Advanced usage
   orchestrator run {filename}.yaml {advanced_example}

**Using Python SDK:**

.. code-block:: python

   from orchestrator import Orchestrator
   
   # Initialize orchestrator
   orchestrator = Orchestrator()
   
   # Run pipeline
   result = await orchestrator.run_pipeline(
       "{filename}.yaml",
       inputs={{
{python_inputs}
       }}
   )
   
   # Access results
{python_results}

{additional_sections}

Key Takeaways
-------------

This example demonstrates the power of Orchestrator's declarative framework:

1. **Zero Code Required**: Complete pipeline in pure YAML
2. **Natural Language Tasks**: Use AUTO tags to describe tasks naturally
3. **Automatic Tool Discovery**: Framework selects appropriate tools
4. **Advanced Control Flow**: {control_flow_features}
5. **Production Ready**: {production_features}

The declarative approach makes complex AI pipelines accessible to everyone, not just programmers.

Next Steps
----------

{next_steps}
"""

# Example configurations
EXAMPLES = {
    "data_processing_workflow": {
        "title": "Data Processing Workflow",
        "description": "a scalable data processing pipeline",
        "details": "The pipeline handles data ingestion, validation, transformation, analysis, and export",
        "level": "Advanced",
        "duration": "45-60 minutes",
        "filename": "data_processing_workflow",
        "features": "Parallel processing for large datasets\n- Automatic data validation and quality checks\n- Error recovery with intelligent retry logic\n- Real-time monitoring and metrics",
        "example_args": '--input source="data/*.csv" --input output_format="parquet"'
    },
    "multi_agent_collaboration": {
        "title": "Multi-Agent Collaboration",
        "description": "a multi-agent AI system",
        "details": "Multiple AI agents work together to solve complex problems through coordination and communication",
        "level": "Advanced", 
        "duration": "60-90 minutes",
        "filename": "multi_agent_collaboration",
        "features": "Agent coordination and task delegation\n- Inter-agent communication protocols\n- Consensus building and conflict resolution\n- Distributed problem solving"
    },
    "content_creation_pipeline": {
        "title": "Content Creation Pipeline",
        "description": "an automated content creation system",
        "details": "The pipeline generates, edits, and publishes high-quality content across multiple formats",
        "level": "Intermediate",
        "duration": "30-45 minutes",
        "filename": "content_creation_pipeline",
        "features": "Multi-format content generation (blog, social, video scripts)\n- Automatic SEO optimization\n- Style and tone adaptation\n- Content scheduling and publishing"
    },
    "code_analysis_suite": {
        "title": "Code Analysis Suite",
        "description": "a comprehensive code analysis and review system",
        "details": "The suite performs static analysis, security scanning, and automated code reviews",
        "level": "Intermediate",
        "duration": "45-60 minutes",
        "filename": "code_analysis_suite",
        "features": "Static code analysis and linting\n- Security vulnerability detection\n- Code quality metrics\n- Automated review suggestions"
    },
    "customer_support_automation": {
        "title": "Customer Support Automation",
        "description": "an intelligent customer support system",
        "details": "The system handles customer inquiries, routes tickets, and provides automated responses",
        "level": "Intermediate",
        "duration": "30-45 minutes",
        "filename": "customer_support_automation",
        "features": "Natural language understanding\n- Ticket classification and routing\n- Sentiment analysis\n- Automated response generation"
    }
}

def create_yaml_examples():
    """Create YAML files for each example."""
    for example_key, config in EXAMPLES.items():
        yaml_content = f"""name: "{config['title']}"
description: "{config['details']}"

inputs:
  # Add specific inputs based on example type
  
steps:
  # Add pipeline steps using AUTO tags
  
outputs:
  # Define pipeline outputs
"""
        
        yaml_path = f"examples/{config['filename']}.yaml"
        print(f"Creating {yaml_path}")
        # Would write the file here
        
def update_documentation():
    """Update all tutorial documentation files."""
    for example_key, config in EXAMPLES.items():
        doc_path = f"docs/tutorials/examples/{config['filename']}.rst"
        print(f"Updating {doc_path}")
        # Would update the documentation here

if __name__ == "__main__":
    print("This script would update all tutorial examples to use the declarative framework.")
    print("\nExamples to update:")
    for key, config in EXAMPLES.items():
        print(f"- {config['title']}")
    
    print("\nTo actually run the updates, the script would need to be modified to write files.")