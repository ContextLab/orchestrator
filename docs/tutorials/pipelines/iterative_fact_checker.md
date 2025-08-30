# Pipeline Tutorial: iterative_fact_checker

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 50/100  
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
- **template_usage**: Uses 16 template patterns for dynamic content
- **feature_highlights**: Demonstrates 6 key orchestrator features

### Data Flow
This pipeline processes input parameters through 6 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# Iterative Fact-Checking Pipeline
# Uses while loops to iteratively verify and add references to documents
id: iterative_fact_checker
name: Iterative Fact Checker
description: Iteratively processes documents to ensure all claims have proper references
version: "2.0.0"

parameters:
  input_document:
    type: string
    default: "test_climate_document.md"
    description: Path to the document to fact-check
  quality_threshold:
    type: number
    default: 0.95
    description: Minimum percentage of claims that must have references
  max_iterations:
    type: integer
    default: 5
    description: Maximum number of improvement iterations

steps:
  # Initialize with the input document
  - id: initialize_vars
    action: generate
    parameters:
      prompt: "Initialize fact-checking process for document: {{ parameters.input_document }}"
      model: "claude-sonnet-4-20250514"
      max_tokens: 50
    
  # Load the initial document
  - id: load_initial_doc
    tool: filesystem
    action: read
    parameters:
      path: "{{ parameters.input_document }}"
    dependencies:
      - initialize_vars
    
  # Main iterative fact-checking loop
  - id: fact_check_loop
    while: "{{ quality_score | default(0) < parameters.quality_threshold }}"
    max_iterations: 3
    dependencies:
      - load_initial_doc
    steps:
      # Load current document (first iteration from initial, then from previous iteration)
      - id: load_document
        tool: filesystem
        action: read
        parameters:
          path: "{{ parameters.input_document }}"
    
      # Extract all claims and existing references
      - id: extract_claims
        action: generate
        parameters:
          prompt: |
            Analyze this document and extract all factual claims and their references.
            
            Document:
            {{ load_document.content }}
            
            For each claim, provide:
            1. The claim text
            2. Whether it has a reference/citation (true/false)
            3. If it has a reference, extract the reference URL
            
            Please respond in the following JSON format:
            {
              "claims": [
                {
                  "text": "claim text here",
                  "has_reference": true/false,
                  "reference_url": "URL if available"
                }
              ],
              "total_claims": number,
              "claims_with_references": number,
              "percentage_referenced": decimal_percentage
            }
            
            Return ONLY the JSON, no other text.
          model: "claude-sonnet-4-20250514"
          max_tokens: 2000
    
      # Verify existing references using headless browser (simplified for now)
      - id: verify_refs
        action: generate
        parameters:
          prompt: |
            Verify which of these reference URLs are valid and accessible:
            {% for claim in (extract_claims.result | from_json).claims %}
            {% if claim.has_reference and claim.reference_url %}
            - {{ claim.reference_url }}
            {% endif %}
            {% endfor %}
            
            For each URL, indicate if it's valid/accessible or broken.
          model: "claude-sonnet-4-20250514"
          max_tokens: 1000
    
      # Find citations for unreferenced claims
      - id: find_citations
        action: generate
        parameters:
          prompt: |
            Find reliable sources and citations for these unreferenced claims:
            
            {% for claim in (extract_claims.result | from_json).claims %}
            {% if not claim.has_reference %}
            - {{ claim.text }}
            {% endif %}
            {% endfor %}
            
            For each claim, provide:
            1. A reliable source URL
            2. The source title
            3. A brief explanation of why this source supports the claim
            
            Format as a list with clear citations.
          model: "claude-sonnet-4-20250514"
          max_tokens: 2000
    
      # Update document with new references and corrections
      - id: update_document
        action: generate
        parameters:
          prompt: |
            Update this document by:
            1. Adding citations for all unreferenced claims using the sources found
            2. Fixing any broken references (URLs that returned errors)
            3. Formatting all references consistently as footnotes at the end
            
            Original document:
            {{ load_document.content }}
            
            Claims analysis:
            {{ extract_claims }}
            
            Reference verification results:
            {{ verify_refs }}
            
            New citations to add:
            {{ find_citations.result }}
            
            Return the complete updated document with all improvements.
          model: "claude-sonnet-4-20250514"
          max_tokens: 5000
    
      # Save iteration document
      - id: save_iteration
        tool: filesystem
        action: write
        parameters:
          path: "{{ output_path }}/iteration_{{ $iteration }}_document.md"
          content: "{{ update_document.result }}"
      
      # Update quality score variable for loop condition
      - id: update_score
        action: generate
        parameters:
          prompt: |
            The document now has {{ (extract_claims.result | from_json).claims_with_references }} out of {{ (extract_claims.result | from_json).total_claims }} claims with references.
            This is {{ (extract_claims.result | from_json).percentage_referenced }}% referenced.
            Output just the decimal percentage (e.g., 0.95 for 95%).
          model: "claude-sonnet-4-20250514"
          max_tokens: 10
        produces: quality_score
    
  # Save the final verified document
  - id: save_final_document
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/{{ parameters.input_document | basename | regex_replace('\\.md$', '') }}_verified.md"
      content: |
        {% if fact_check_loop.iterations %}
        {{ fact_check_loop.iterations[-1].update_document.result }}
        {% else %}
        {{ load_initial_doc.content }}
        {% endif %}
    dependencies:
      - fact_check_loop
    
  # Generate comprehensive fact-checking report
  - id: generate_report
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/fact_checking_report.md"
      content: |
        # Fact-Checking Report
        
        ## Document Information
        - **Source Document**: {{ parameters.input_document }}
        - **Date Processed**: {{ timestamp }}
        
        ## Processing Summary
        - **Total Iterations**: {{ fact_check_loop.iteration_count | default(1) }}
        - **Quality Threshold**: {{ parameters.quality_threshold * 100 }}%
        - **Final Quality Score**: {% if fact_check_loop.iterations %}{{ (fact_check_loop.iterations[-1].extract_claims.result | from_json).percentage_referenced }}%{% else %}N/A{% endif %}
        
        ## Iteration Details
        {% for iteration in fact_check_loop.iterations %}
        ### Iteration {{ loop.index }}
        - Claims analyzed: {{ (iteration.extract_claims.result | from_json).total_claims }}
        - Claims with references: {{ (iteration.extract_claims.result | from_json).claims_with_references }}
        - Percentage referenced: {{ (iteration.extract_claims.result | from_json).percentage_referenced }}%
        - New citations added: {{ iteration.find_citations.result | length | default(0) }}
        {% endfor %}
        
        ## Final Status
        {% if fact_check_loop.iterations and (fact_check_loop.iterations[-1].extract_claims.result | from_json).percentage_referenced >= parameters.quality_threshold * 100 %}
        ✅ **Quality threshold met**: All or most claims now have proper references.
        {% else %}
        ⚠️ **Maximum iterations reached**: Some claims may still lack references.
        {% endif %}
        
        ## Output Files
        - Verified document: `{{ parameters.input_document | basename | regex_replace('\\.md$', '') }}_verified.md`
        - Iteration documents: `iteration_*_document.md`
        
        ---
        *Generated by Iterative Fact Checker Pipeline v2.0*
    dependencies:
      - save_final_document

outputs:
  verified_document: "{{ save_final_document.path }}"
  report: "{{ generate_report.path }}"
  iterations_performed: "{{ fact_check_loop.iteration_count | default(0) }}"
  final_quality: "{% if fact_check_loop.iterations %}{{ (fact_check_loop.iterations[-1].extract_claims.result | from_json).percentage_referenced }}%{% else %}N/A{% endif %}"
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
- This pipeline produces Analysis results, JSON data, Markdown documents, Reports - adjust output configuration accordingly

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
- 2. Run: python scripts/run_pipeline.py examples/iterative_fact_checker.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated Analysis results in the specified output directory
- Generated JSON data in the specified output directory
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

*Tutorial generated on 2025-08-27T23:40:24.396201*
