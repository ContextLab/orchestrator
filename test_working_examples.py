#!/usr/bin/env python3
"""Test working examples using direct control system approach."""

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


async def test_working_examples():
    """Test working examples using direct control system approach."""
    
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
    
    # Create working example YAML files
    working_examples = {
        "research_example.yaml": {
            "yaml_content": """
name: "Research Assistant"
description: "Research and analyze information"

inputs:
  query:
    type: string
    description: "Research query"
    required: true

steps:
  - id: analyze_query
    action: |
      Analyze the research query "{{query}}" and identify:
      1. Key research objectives
      2. Types of information needed
      3. Focus areas to explore
      Return analysis with search terms
    
  - id: generate_findings
    action: |
      Based on the query analysis: {{analyze_query.result}}
      
      Generate key findings about "{{query}}" including:
      1. Main concepts and definitions
      2. Key insights
      3. Important facts
      4. Relevant examples
      Return structured findings
    depends_on: [analyze_query]
    
  - id: create_summary
    action: |
      Create a comprehensive summary based on:
      Query: {{query}}
      Analysis: {{analyze_query.result}}
      Findings: {{generate_findings.result}}
      
      Format as a professional research summary with:
      1. Executive summary
      2. Key findings
      3. Conclusions
      Return formatted summary
    depends_on: [generate_findings]

outputs:
  summary: "{{create_summary.result}}"
  findings: "{{generate_findings.result}}"
            """,
            "inputs": {
                "query": "artificial intelligence applications in healthcare"
            }
        },
        
        "writing_assistant.yaml": {
            "yaml_content": """
name: "Writing Assistant"
description: "Generate and improve written content"

inputs:
  topic:
    type: string
    description: "Writing topic"
    required: true
  style:
    type: string
    description: "Writing style"
    default: "professional"

steps:
  - id: plan_content
    action: |
      Create a content plan for "{{topic}}" in {{style}} style:
      1. Main themes to cover
      2. Key points to include
      3. Target audience considerations
      4. Content structure outline
      Return detailed content plan
    
  - id: write_content
    action: |
      Write comprehensive content about "{{topic}}" using:
      Content plan: {{plan_content.result}}
      Style: {{style}}
      
      Create engaging content that:
      1. Follows the content plan
      2. Maintains consistent style
      3. Provides valuable information
      4. Engages the target audience
      Return well-written content
    depends_on: [plan_content]
    
  - id: review_content
    action: |
      Review and improve the content:
      Original content: {{write_content.result}}
      
      Check for:
      1. Clarity and flow
      2. Grammar and style
      3. Completeness of coverage
      4. Engagement level
      Return improved content with suggestions
    depends_on: [write_content]

outputs:
  content: "{{review_content.result}}"
  plan: "{{plan_content.result}}"
            """,
            "inputs": {
                "topic": "sustainable technology trends",
                "style": "informative"
            }
        },
        
        "analysis_assistant.yaml": {
            "yaml_content": """
name: "Analysis Assistant"
description: "Analyze and interpret data or information"

inputs:
  subject:
    type: string
    description: "Subject to analyze"
    required: true
  focus:
    type: string
    description: "Analysis focus"
    default: "comprehensive"

steps:
  - id: gather_information
    action: |
      Gather information about "{{subject}}" for {{focus}} analysis:
      1. Key facts and data points
      2. Important characteristics
      3. Relevant context
      4. Current trends or patterns
      Return organized information
    
  - id: perform_analysis
    action: |
      Analyze the gathered information:
      Subject: {{subject}}
      Information: {{gather_information.result}}
      Focus: {{focus}}
      
      Provide analysis including:
      1. Key insights and patterns
      2. Strengths and weaknesses
      3. Opportunities and risks
      4. Trends and implications
      Return structured analysis
    depends_on: [gather_information]
    
  - id: generate_recommendations
    action: |
      Generate actionable recommendations based on:
      Analysis: {{perform_analysis.result}}
      
      Provide:
      1. Top 3 recommendations
      2. Implementation steps
      3. Expected outcomes
      4. Risk mitigation strategies
      Return practical recommendations
    depends_on: [perform_analysis]

outputs:
  analysis: "{{perform_analysis.result}}"
  recommendations: "{{generate_recommendations.result}}"
            """,
            "inputs": {
                "subject": "remote work productivity",
                "focus": "comprehensive"
            }
        }
    }
    
    results = []
    
    for yaml_filename, example_data in working_examples.items():
        print(f"\n=== Testing {yaml_filename} ===")
        
        yaml_content = example_data["yaml_content"]
        inputs = example_data["inputs"]
        
        try:
            # Compile YAML to pipeline
            pipeline = await compiler.compile(yaml_content, inputs)
            
            # Execute pipeline directly using control system
            result = await asyncio.wait_for(
                control_system.execute_pipeline(pipeline),
                timeout=90.0
            )
            
            step_count = len(result) if isinstance(result, dict) else 0
            print(f"  ✓ SUCCESS - {yaml_filename} completed ({step_count} steps)")
            
            # Show results
            if isinstance(result, dict) and result:
                for step_id, step_result in result.items():
                    result_str = str(step_result)[:150] + "..." if len(str(step_result)) > 150 else str(step_result)
                    print(f"    {step_id}: {result_str}")
            
            results.append((yaml_filename, True, f"{step_count} steps"))
            
        except asyncio.TimeoutError:
            print(f"  ✗ TIMEOUT - {yaml_filename} exceeded 90 seconds")
            results.append((yaml_filename, False, "Timeout"))
            
        except Exception as e:
            print(f"  ✗ FAILED - {yaml_filename}: {type(e).__name__}: {str(e)}")
            results.append((yaml_filename, False, f"{type(e).__name__}: {str(e)}"))
    
    # Summary
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\n=== SUMMARY ===")
    print(f"Successful: {successful}/{total} ({successful/total*100:.1f}%)")
    
    for name, success, details in results:
        status = "✓" if success else "✗"
        print(f"{status} {name} - {details}")
    
    return successful > 0


if __name__ == "__main__":
    success = asyncio.run(test_working_examples())
    if success:
        print("\n🎉 Working examples completed successfully!")
    else:
        print("\n❌ Working examples failed")