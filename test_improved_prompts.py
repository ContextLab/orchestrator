#!/usr/bin/env python3
"""Test improved prompts for better output quality."""

import asyncio
import os
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from orchestrator.compiler.yaml_compiler import YAMLCompiler


async def test_improved_prompts():
    """Test improved prompts for better output quality."""
    
    # Set up models
    model_registry = ModelRegistry()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("No OpenAI API key found")
        return False
        
    openai_model = OpenAIModel(model_name="gpt-4o-mini")
    model_registry.register_model(openai_model)
    
    # Create control system and compiler
    control_system = ModelBasedControlSystem(model_registry)
    compiler = YAMLCompiler()
    
    # Test case with more complex requirements
    complex_example = {
        "yaml_content": """
name: "Technical Documentation Generator"
description: "Generate comprehensive technical documentation"

inputs:
  project_name:
    type: string
    description: "Name of the project"
    required: true
  project_type:
    type: string
    description: "Type of project"
    required: true
  key_features:
    type: string
    description: "Key features to document"
    required: true

steps:
  - id: analyze_requirements
    action: |
      Analyze the documentation requirements for {{project_name}}:
      Project Type: {{project_type}}
      Key Features: {{key_features}}
      
      Determine:
      1. Documentation structure needed
      2. Technical depth required
      3. Target audience (developers, users, both)
      4. Key sections to include
      5. Examples and code samples needed
      Return detailed documentation plan
    
  - id: create_overview
    action: |
      Create a comprehensive project overview for {{project_name}}:
      Documentation Plan: {{analyze_requirements.result}}
      
      Include:
      1. Project description and purpose
      2. Key features and benefits
      3. Architecture overview
      4. Technology stack
      5. Use cases and applications
      Return detailed overview section
    depends_on: [analyze_requirements]
    
  - id: write_getting_started
    action: |
      Write a detailed Getting Started guide:
      Project: {{project_name}}
      Type: {{project_type}}
      Overview: {{create_overview.result}}
      
      Include:
      1. Prerequisites and requirements
      2. Installation instructions (multiple platforms)
      3. Configuration steps
      4. Quick start example
      5. Common troubleshooting
      6. Next steps
      Return comprehensive getting started guide
    depends_on: [create_overview]
    
  - id: document_api
    action: |
      Document the API/Interface for {{project_name}}:
      Features: {{key_features}}
      Documentation Plan: {{analyze_requirements.result}}
      
      Include:
      1. API endpoints or interface methods
      2. Parameters and return types
      3. Code examples for each feature
      4. Error handling
      5. Best practices
      6. Rate limits or constraints
      Return detailed API documentation
    depends_on: [analyze_requirements]
    
  - id: create_examples
    action: |
      Create practical examples for {{project_name}}:
      Features: {{key_features}}
      API Documentation: {{document_api.result}}
      Getting Started: {{write_getting_started.result}}
      
      Create:
      1. Basic usage example
      2. Advanced use case
      3. Integration example
      4. Performance optimization example
      5. Error handling example
      Each with explanatory comments
      Return comprehensive examples section
    depends_on: [document_api, write_getting_started]
    
  - id: compile_documentation
    action: |
      Compile complete technical documentation:
      Overview: {{create_overview.result}}
      Getting Started: {{write_getting_started.result}}
      API Documentation: {{document_api.result}}
      Examples: {{create_examples.result}}
      
      Create a well-organized document with:
      1. Table of contents
      2. Consistent formatting
      3. Cross-references
      4. Version information
      5. Contributing guidelines
      6. License information
      Return complete documentation
    depends_on: [create_examples]

outputs:
  documentation: "{{compile_documentation.result}}"
  overview: "{{create_overview.result}}"
  examples: "{{create_examples.result}}"
        """,
        "inputs": {
            "project_name": "DataFlow Pipeline",
            "project_type": "Python data processing framework",
            "key_features": "async processing, schema validation, error handling, plugin system"
        }
    }
    
    print("=== Testing Improved Prompts ===")
    
    try:
        # Compile YAML to pipeline
        pipeline = await compiler.compile(
            complex_example["yaml_content"], 
            complex_example["inputs"]
        )
        
        # Execute pipeline
        result = await asyncio.wait_for(
            control_system.execute_pipeline(pipeline),
            timeout=180.0
        )
        
        # Analyze output quality
        print("\n=== Output Quality Analysis ===")
        
        for step_id, step_result in result.items():
            print(f"\n--- {step_id} ---")
            
            # Check for quality indicators
            result_str = str(step_result)
            
            # Quality metrics
            has_structure = any(marker in result_str for marker in ['#', '##', '1.', '-', '*'])
            has_examples = 'example' in result_str.lower() or 'code' in result_str.lower()
            has_details = len(result_str) > 500
            is_comprehensive = len(result_str) > 1000
            
            print(f"✓ Structured format: {has_structure}")
            print(f"✓ Contains examples: {has_examples}")
            print(f"✓ Detailed content: {has_details}")
            print(f"✓ Comprehensive: {is_comprehensive}")
            print(f"  Length: {len(result_str)} characters")
            
            # Show first 300 characters as preview
            preview = result_str[:300] + "..." if len(result_str) > 300 else result_str
            print(f"\n  Preview: {preview}")
        
        # Save the full documentation output
        output_dir = Path("example_outputs")
        output_dir.mkdir(exist_ok=True)
        
        doc_output = result.get("compile_documentation", "")
        with open(output_dir / "improved_documentation_output.md", "w") as f:
            f.write(str(doc_output))
        
        print(f"\n✅ Full documentation saved to: {output_dir / 'improved_documentation_output.md'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_improved_prompts())
    if success:
        print("\n🎉 Improved prompts test completed successfully!")
    else:
        print("\n❌ Improved prompts test failed")