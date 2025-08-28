# Pipeline Tutorial: fact_checker

## Overview

**Complexity Level**: Advanced  
**Difficulty Score**: 65/100  
**Estimated Runtime**: 15+ minutes  

### Purpose
This pipeline demonstrates data_flow, data_validation, fact_checking and provides a practical example of orchestrator's capabilities for advanced-level workflows.

### Use Cases
- AI-powered content generation

### Prerequisites
- Basic understanding of YAML syntax
- Experience with intermediate pipeline patterns
- Understanding of error handling and system integration
- Familiarity with external APIs and tools

### Key Concepts
- Data flow between pipeline steps
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 8 template patterns for dynamic content
- **feature_highlights**: Demonstrates 7 key orchestrator features

### Data Flow
This pipeline processes input parameters through 7 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# Intelligent Fact-Checker Pipeline
# Demonstrates AUTO tags resolving to lists and runtime for_each expansion

id: intelligent-fact-checker
name: Intelligent Fact-Checker
description: Verifies claims and sources using AUTO tag list generation and runtime parallel processing
version: "3.0.0"

inputs:
  document_source:
    type: string
    description: Path to document file or URL to analyze
    required: true
  strictness:
    type: string
    description: How strict should fact-checking be (lenient/moderate/strict)
    default: "moderate"
  output_path:
    type: string
    description: Path where fact-check report should be saved
    required: false

outputs:
  fact_check_report:
    type: string
    value: "{{ output_path | default('examples/outputs/fact_checker/fact_check_report.md') }}"

steps:
  # Step 1: Load document using filesystem tool
  - id: load_document
    tool: filesystem
    action: read
    parameters:
      path: "{{ document_source }}"
    produces: text
    
  # Step 2: Extract sources as a list using AUTO tag
  - id: extract_sources_list
    dependencies:
      - load_document
    action: generate-structured
    parameters:
      prompt: |
        Analyze this document and extract all sources, citations, and references.
        
        Document content:
        {{ load_document.result }}
        
        Extract each source with its name, URL, and type (journal/report/website/other).
      schema:
        type: object
        properties:
          sources:
            type: array
            items:
              type: object
              properties:
                name:
                  type: string
                url:
                  type: string
                type:
                  type: string
                  enum: ["journal", "report", "website", "other"]
        required: [sources]
      max_completion_tokens: 1000
      model: "claude-sonnet-4-20250514"
    produces: json
    
  # Step 3: Extract claims as a list using AUTO tag
  - id: extract_claims_list
    dependencies:
      - load_document
    action: generate-structured
    parameters:
      prompt: |
        Extract all verifiable factual claims from this document.
        
        Document content:
        {{ load_document.result }}
        
        List each claim as a clear, concise statement that can be fact-checked.
      schema:
        type: object
        properties:
          claims:
            type: array
            items:
              type: string
        required: [claims]
      max_completion_tokens: 1000
      model: "claude-sonnet-4-20250514"
    produces: json
    
  # Step 4: Process sources in parallel using for_each with runtime expansion
  - id: verify_sources
    for_each: "<AUTO>list of sources to verify</AUTO>"
    max_parallel: 2
    add_completion_task: true
    steps:
      - id: verify_source
        action: generate-text
        parameters:
          prompt: |
            Verify this source for authenticity and accuracy:
            Name: {{ item.name }}
            URL: {{ item.url }}
            Type: {{ item.type }}
            
            Check:
            1. Is the source real and accessible?
            2. Is it correctly cited?
            3. Is it a reputable source for its type?
            
            Provide structured analysis:
            - SOURCE: {{ item.name }} ({{ item.url }})
            - STATUS: Valid/Invalid/Questionable
            - CREDIBILITY: High/Medium/Low
            - NOTES: Brief observations
          max_completion_tokens: 300
          model: "claude-sonnet-4-20250514"
    dependencies:
      - extract_sources_list
    
  # Step 5: Process claims in parallel using for_each with runtime expansion
  - id: verify_claims
    for_each: "<AUTO>claims that need fact-checking</AUTO>"
    max_parallel: 3
    add_completion_task: true
    steps:
      - id: verify_claim
        action: generate-text
        parameters:
          prompt: |
            Verify this specific claim: {{ item }}
            Claim number: {{ index }}
            
            Using sources from document for verification.
            
            Provide professional fact-checking analysis:
            - CLAIM {{ index }}: {{ item }}
            - SUPPORT: Yes/Partial/No
            - RELIABILITY: High/Medium/Low
            - EVIDENCE: Brief summary of supporting/contradicting evidence
            - STATUS: Verified/Unverified/Disputed
          max_completion_tokens: 400
          model: "claude-sonnet-4-20250514"
    dependencies:
      - extract_claims_list
    
  # Step 6: Generate final report
  - id: generate_report
    dependencies:
      - verify_sources  # The ForEachTask itself
      - verify_claims   # The ForEachTask itself
      - extract_sources_list
      - extract_claims_list
    action: generate-text
    parameters:
      prompt: |
        Generate a professional fact-checking report for the article "{{ document_source }}".
        
        You have:
        1. Extracted sources: {{ extract_sources_list.result.sources | length }} sources identified
        2. Extracted claims: {{ extract_claims_list.result.claims | length }} claims identified  
        3. Source verifications completed
        4. Claim verifications completed
        
        Create a comprehensive fact-checking report with these sections:
        
        ## Executive Summary
        - Brief overview of the article's topic and main assertions
        - Overall credibility assessment (High/Medium/Low)
        - Number of verified vs unverified claims
        
        ## Source Analysis
        - Credibility of cited sources
        - Any missing or questionable citations
        - Balance between peer-reviewed and industry sources
        
        ## Claim Verification
        For each major claim:
        - Statement of the claim
        - Verification status (Verified/Partially Verified/Unverified/False)
        - Supporting or contradicting evidence
        - Confidence level
        
        ## Red Flags and Concerns
        - Any misleading statements
        - Unsupported assertions
        - Potential biases
        
        ## Conclusion
        - Overall assessment of article accuracy
        - Recommendations for readers
        - Areas requiring further investigation
        
        Focus on providing actionable fact-checking insights about the healthcare AI article's claims and sources.
        Be specific about which claims are well-supported and which require scrutiny.
      max_completion_tokens: 2000
      model: "claude-sonnet-4-20250514"
    produces: text
    
  # Step 7: Save the report
  - id: save_report
    dependencies:
      - generate_report
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path | default('examples/outputs/fact_checker/fact_check_report.md') }}"
      content: |
        # Fact-Checking Report
        
        **Document:** {{ document_source }}
        **Analysis Date:** {{ execution.timestamp }}
        **Verification Standard:** {{ strictness }}
        
        ---
        
        {{ generate_report.result }}
```

## Customization Guide

### Input Modifications
- Modify input parameters to match your specific data sources
- Adjust file paths and data formats as needed for your environment

### Parameter Tuning
- Adjust model parameters (temperature, max_tokens) for different output styles
- Modify prompts to change the tone and focus of generated content
- Fine-tune performance parameters for your specific use case

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
- Build modular components for reusability
- Add performance monitoring and optimization
- Implement advanced security and access controls

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
- 2. Run: python scripts/run_pipeline.py examples/fact_checker.yaml
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
- **Complex Logic Errors**: Review the pipeline configuration and ensure all advanced features are properly configured
- **General Execution Errors**: Check the logs for specific error messages and verify your orchestrator installation

### Verification Steps
- Check that the pipeline completed without errors
- Verify all expected output files were created
- Review the output content for quality and accuracy
- Review generated content for relevance and quality

---

*Tutorial generated on 2025-08-27T23:40:24.396123*
