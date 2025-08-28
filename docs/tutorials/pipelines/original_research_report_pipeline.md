# Pipeline Tutorial: original_research_report_pipeline

## Overview

**Complexity Level**: Advanced  
**Difficulty Score**: 85/100  
**Estimated Runtime**: 30+ minutes  

### Purpose
This pipeline demonstrates how to build automated research workflows using the orchestrator toolbox. It showcases data_flow, error_handling, fact_checking and provides a foundation for building more sophisticated research applications.

### Use Cases
- AI-powered content generation
- Academic research and literature review
- Fact-checking and information verification
- Information gathering and research
- Market research and competitive analysis
- System administration and automation

### Prerequisites
- Basic understanding of YAML syntax
- Experience with intermediate pipeline patterns
- Understanding of error handling and system integration
- Familiarity with external APIs and tools
- Understanding of command-line interfaces and system security

### Key Concepts
- Data flow between pipeline steps
- Error handling and recovery
- Iterative processing with loops
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 11 template patterns for dynamic content
- **feature_highlights**: Demonstrates 9 key orchestrator features

### Data Flow
This pipeline processes input parameters through 9 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# Original Research Report Pipeline - Syntax Corrected
# This version fixes YAML syntax errors while preserving all intended functionality
# Features marked with comments are not yet implemented in the current framework

id: research-report-pipeline
name: Write Research Report
description: Generate a comprehensive research report with fact-checking and quality control

inputs:
 topic:
  type: string
  description: "Topic for the research report"
  required: true
 instructions:
  type: string
  description: "Additional instructions for report generation"
  required: true

outputs:
 pdf: 
  type: string
  value: "<AUTO>come up with an appropriate filename for the final report</AUTO>"
 tex: 
  type: string
  value: "{{ outputs.pdf[:-4] }}.tex" # Corrected template syntax

steps:
 - id: web-search
   action: "<AUTO>search the web for <AUTO>construct an appropriate web query about {{ topic }}, using these additional instructions: {{ instructions }}</AUTO></AUTO>"
   tool: headless-browser
   produces: "<AUTO>markdown file with detailed notes on each relevant result with annotated links to original sources; other relevant files like images, code, data, etc., that can be saved locally</AUTO>"
   location: "./searches/{{ outputs.pdf }}/"
   requires_model:
    min_size: "7B"
    expertise: "medium"

 - id: compile-search-results
   depends_on: [web-search]
   action: "<AUTO>create a markdown file collating the content from {{ web-search.result }} into a single cohesive document. maintain annotated links back to original sources.</AUTO>"
   produces: "markdown-file"
   location: "./searches/{{ outputs.pdf }}/compiled_results.md"
   requires_model:
    min_size: "10B"
    expertise: "medium-high"

 - id: quality-check-compilation
   depends_on: [compile-search-results]
   requires_model:
    min_size: "7B"
    expertise: "medium"
   # Complex nested control flow - create_parallel_queue with action_loop
   create_parallel_queue:
    on: "<AUTO>create a list of every source in this document: {{ compile-search-results.result }}</AUTO>"
    tool: headless-browser
    task:
     action_loop:
      - action: "<AUTO>verify the authenticity of this source ({{ $item }}) by following the web link and ensuring it is accurately described in {{ compile-search-results.result }}</AUTO>"
        id: verify-source
        produces: "<AUTO>'true' if reference was verified, '<AUTO>corrected reference</AUTO>' if reference could be fixed with minor edits, or 'false' if reference seems to be hallucinated</AUTO>"
      - action: "<AUTO>if {{ verify-source.result }} is 'false', update {{ compile-search-results.result }} to remove the reference. if {{ verify-source.result }} has a corrected reference, update {{ compile-search-results.result }} to use the corrected reference.</AUTO>"
        id: update-sources
     until: "<AUTO>all sources have been verified (or removed, if incorrect)</AUTO>"
   produces: "markdown-file"
   location: "./searches/{{ outputs.pdf }}/compiled_results_corrected.md"

 - id: draft-report
   depends_on: [quality-check-compilation]
   # File inclusion syntax (Issue #191)
   action: "<AUTO>{{ file:report_draft_prompt.md }}</AUTO>"
   produces: "markdown-file"
   location: "./searches/{{ outputs.pdf }}/draft_report.md"
   requires_model:
    min_size: "20B"
    expertise: "high"

 - id: quality-check-assumptions
   depends_on: [draft-report]
   requires_model:
    min_size: "20B"
    expertise: "high"
   # Another complex nested control flow
   create_parallel_queue:
    on: "<AUTO>create a comprehensive list of every non-trivial claim made in this document (include, for each claim, any sources or supporting evidence provided in the document): {{ draft-report.result }}</AUTO>"
    tool: headless-browser
    task:
     action_loop:
      - action: "<AUTO>verify the accuracy of this claim ({{ $item }}) by (a) doing a web search and (b) using logical reasoning and deductive inference *based only on the provided claim, sources, and supporting evidence. be sure to manually follow every source link to verify accuracy.</AUTO>"
        id: claim-check
        produces: "<AUTO>'true' if claim was verified, '<AUTO>corrected claim</AUTO>' if claim could be fixed with minor edits, or 'false' if claim seems to be hallucinated</AUTO>"
      - action: "<AUTO>if {{ claim-check.result }} is 'false', update {{ draft-report.result }} to remove the claim. if {{ claim-check.result }} has a corrected claim, update {{ draft-report.result }} to use the corrected claim.</AUTO>"
        id: update-claims
     until: "<AUTO>all claims have been verified (or removed, if inaccurate)</AUTO>"
   produces: "markdown-file"
   location: "./searches/{{ outputs.pdf }}/draft_report_corrected.md"

 - id: quality-check-full-report
   depends_on: [quality-check-assumptions]
   requires_model:
    min_size: "40B"
    expertise: "very-high"
   action: "<AUTO>do a thorough pass through this document. without adding *any* new claims or references, revise the document to improve (a) clarity, (b) logical flow, (c) grammar, and (d) writing quality: {{ quality-check-assumptions.result }}</AUTO>"
   produces: "markdown-file"
   location: "./searches/{{ outputs.pdf }}/draft_report_final.md"

 - id: compile-pdf
   depends_on: [quality-check-full-report]
   # Corrected terminal command syntax (Issue #195)
   action: execute_command
   parameters:
    command: "pandoc -o {{ location }} {{ quality-check-full-report.location }}"
   tool: terminal
   location: "./searches/{{ outputs.pdf }}/report.pdf"
   requires_model: none
   produces: "pdf"
   # Error handling syntax (Issue #192)
   on_error: debug-compilation

 - id: debug-compilation
   requires_model:
    min_size: "40B"
    expertise: "very-high"
   # Action loop for debugging (Issue #188)
   action_loop:
    - action: "<AUTO>carefully check the output logs in the current directory to see why compiling the pdf failed; use bash commands to update the document and/or run other bash commands as needed until {{ compile-pdf.location }} is a valid pdf.</AUTO>"
      tool: terminal
   until: "<AUTO>{{ compile-pdf.location }} is a valid pdf</AUTO>"
   produces: "pdf"
   location: "./searches/{{ outputs.pdf }}/report.pdf"
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
- This pipeline produces Images, Markdown documents, Reports - adjust output configuration accordingly

## Remixing Instructions

### Compatible Patterns
- fact_checker.yaml - for content verification
- research workflows - for information gathering

### Extension Ideas
- Build modular components for reusability
- Add performance monitoring and optimization
- Implement advanced security and access controls

### Combination Examples
- Combine with fact_checker.yaml to verify research claims
- Use with creative_image_pipeline.yaml to generate visual research summaries
- Integrate with data_processing.yaml to analyze research data

### Advanced Variations
- Scale to handle larger datasets and more complex processing
- Add real-time processing capabilities for streaming data
- Implement distributed processing across multiple systems
- Use multiple AI models for comparison and validation

## Hands-On Exercise

### Execution Instructions
- 1. Navigate to your orchestrator project directory
- 1.5. Ensure you have access to required services: System access, Web search APIs
- 2. Run: python scripts/run_pipeline.py examples/original_research_report_pipeline.yaml
- 3. Monitor the output for progress and any error messages
- 4. Check the output directory for generated results

### Expected Outputs
- Generated Images in the specified output directory
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

*Tutorial generated on 2025-08-27T23:40:24.396493*
