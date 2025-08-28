# Pipeline Tutorial: iterative_fact_checker_simple

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 45/100  
**Estimated Runtime**: 5-15 minutes  

### Purpose
This pipeline demonstrates data_flow, fact_checking, interactive_workflows and provides a practical example of orchestrator's capabilities for intermediate-level workflows.

### Use Cases
- AI-powered content generation

### Prerequisites
- Basic understanding of YAML syntax
- Familiarity with template variables and data flow
- Understanding of basic control flow concepts

### Key Concepts
- Data flow between pipeline steps
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 10 template patterns for dynamic content
- **feature_highlights**: Demonstrates 5 key orchestrator features

### Data Flow
This pipeline processes input parameters through 5 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# Simplified Iterative Fact-Checking Pipeline
# Avoids complex template logic for better compatibility
id: iterative_fact_checker_simple
name: Simple Iterative Fact Checker
description: Iteratively adds references to documents using simpler logic
version: "2.1.0"

parameters:
  input_document:
    type: string
    default: "test_climate_document.md"
    description: Path to the document to fact-check
  quality_threshold:
    type: number
    default: 0.95
    description: Minimum percentage of claims that must have references

steps:
  # Load the initial document
  - id: load_document
    tool: filesystem
    action: read
    parameters:
      path: "{{ parameters.input_document }}"
    
  # Single iteration of fact-checking (can be run multiple times manually)
  - id: extract_claims
    action: generate-structured
    dependencies:
      - load_document
    parameters:
      prompt: |
        Analyze this document and extract all factual claims and their references.
        
        Document:
        {{ load_document.content }}
        
        For each claim:
        1. Extract the claim text
        2. Note if it has a reference/citation
        3. If it has a reference, extract the reference details
      schema:
        type: object
        properties:
          claims:
            type: array
            items:
              type: object
              properties:
                text:
                  type: string
                has_reference:
                  type: boolean
                reference_url:
                  type: string
          total_claims:
            type: integer
          claims_with_references:
            type: integer
          percentage_referenced:
            type: number
      model: "claude-sonnet-4-20250514"
      max_tokens: 3000
  
  # Find citations for unreferenced claims
  - id: find_citations
    action: generate
    dependencies:
      - extract_claims
    parameters:
      prompt: |
        Find reliable sources and citations for these unreferenced claims from the document:
        
        {% if extract_claims.claims %}
        {% for claim in extract_claims.claims %}
        {% if not claim.has_reference %}
        - {{ claim.text }}
        {% endif %}
        {% endfor %}
        {% else %}
        No claims data available.
        {% endif %}
        
        For each claim, provide:
        1. A reliable source URL (use real, authoritative sources like NASA, NOAA, IPCC, Nature, Science, etc.)
        2. The source title
        3. A brief explanation of why this source supports the claim
        
        Format as a structured list with clear citations.
      model: "claude-sonnet-4-20250514"
      max_tokens: 3000
  
  # Update document with new references
  - id: update_document
    action: generate
    dependencies:
      - load_document
      - extract_claims
      - find_citations
    parameters:
      prompt: |
        Update this document by adding citations for all unreferenced claims.
        
        Original document:
        {{ load_document.content }}
        
        Claims analysis showing {{ extract_claims.total_claims }} total claims with {{ extract_claims.claims_with_references }} already referenced.
        
        New citations to add:
        {{ find_citations.result }}
        
        Instructions:
        1. Add inline citations [1], [2], etc. after each claim
        2. Add a "References" section at the end with all citations
        3. Keep the document structure and content otherwise unchanged
        4. Use consistent formatting throughout
        
        Return the complete updated document with all improvements.
      model: "claude-sonnet-4-20250514"
      max_tokens: 6000
  
  # Save the updated document
  - id: save_document
    tool: filesystem
    action: write
    dependencies:
      - update_document
    parameters:
      path: "{{ output_path }}/{{ parameters.input_document | basename | regex_replace('\\.md$', '') }}_fact_checked.md"
      content: "{{ update_document.result }}"
  
  # Generate report
  - id: generate_report
    tool: filesystem
    action: write
    dependencies:
      - extract_claims
      - save_document
    parameters:
      path: "{{ output_path }}/fact_checking_report.md"
      content: |
        # Fact-Checking Report
        
        ## Document Information
        - **Source Document**: {{ parameters.input_document }}
        - **Date Processed**: {{ timestamp }}
        
        ## Analysis Summary
        - **Total Claims**: {{ extract_claims.total_claims }}
        - **Claims with References (before)**: {{ extract_claims.claims_with_references }}
        - **Percentage Referenced (before)**: {{ extract_claims.percentage_referenced }}%
        - **Quality Threshold**: {{ parameters.quality_threshold * 100 }}%
        
        ## Claims Identified
        {% if extract_claims.claims %}
        {% for claim in extract_claims.claims %}
        {{ loop.index }}. {{ claim.text }}
           - Has reference: {{ claim.has_reference }}
        {% endfor %}
        {% endif %}
        
        ## Status
        {% if extract_claims.percentage_referenced >= parameters.quality_threshold * 100 %}
        ✅ **Document meets quality threshold**
        {% else %}
        ⚠️ **Document below quality threshold - references added**
        {% endif %}
        
        ## Output
        - Updated document: `{{ parameters.input_document | basename | regex_replace('\\.md$', '') }}_fact_checked.md`
        
        ---
        *Generated by Simple Iterative Fact Checker Pipeline v2.1*

outputs:
  updated_document: "{{ save_document.path }}"
  report: "{{ generate_report.path }}"
  claims_count: "{{ extract_claims.total_claims }}"
  reference_percentage: "{{ extract_claims.percentage_referenced }}%"
```

## Customization Guide

### Input Modifications
- Modify input parameters to match your specific data sources
- Adjust file paths and data formats as needed for your environment

### Parameter Tuning
- Adjust model parameters (temperature, max_tokens) for different output styles
- Modify prompts to change the tone and focus of generated content

### Step Modifications
- Add new steps by following the same pattern as existing ones
- Remove steps that aren't needed for your specific use case
- Reorder steps if your workflow requires different sequencing
- Replace tool actions with alternatives that provide similar functionality

### Output Customization
- Change output file paths and formats to match your requirements
- Modify output templates to customize the structure and content
- This pipeline produces Analysis results, Markdown documents, Reports - adjust output configuration accordingly

## Remixing Instructions

### Compatible Patterns
- fact_checker.yaml - for content verification
- research workflows - for information gathering

### Extension Ideas
- Add iterative processing for continuous improvement
- Implement parallel processing for better performance
- Include advanced error recovery mechanisms

### Combination Examples
- Can be combined with most other pipeline patterns

### Advanced Variations
- Scale to handle larger datasets and more complex processing
- Add real-time processing capabilities for streaming data
- Implement distributed processing across multiple systems
- Use multiple AI models for comparison and validation

## Hands-On Exercise

### Execution Instructions
- 1. Navigate to your orchestrator project directory
- 1.5. Ensure you have access to required services: Anthropic API
- 2. Run: python scripts/run_pipeline.py examples/iterative_fact_checker_simple.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated Analysis results in the specified output directory
- Generated Markdown documents in the specified output directory
- Generated Reports in the specified output directory
- Execution logs showing step-by-step progress
- Completion message with runtime statistics
- No error messages or warnings (successful execution)

### Troubleshooting
- **API Authentication Errors**: Ensure all required API keys are properly configured in your environment
- **Template Resolution Errors**: Check that all input parameters are provided and template syntax is correct
- **General Execution Errors**: Check the logs for specific error messages and verify your orchestrator installation

### Verification Steps
- Check that the pipeline completed without errors
- Verify all expected output files were created
- Review the output content for quality and accuracy
- Review generated content for relevance and quality

---

*Tutorial generated on 2025-08-27T23:40:24.396226*
