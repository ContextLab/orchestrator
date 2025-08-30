# Pipeline Tutorial: enhanced_research_pipeline

## Overview

**Complexity Level**: Advanced  
**Difficulty Score**: 100/100  
**Estimated Runtime**: 30+ minutes  

### Purpose
This pipeline demonstrates how to build automated research workflows using the orchestrator toolbox. It showcases conditional_execution, data_flow, data_transformation and provides a foundation for building more sophisticated research applications.

### Use Cases
- AI-powered content generation
- Academic research and literature review
- Fact-checking and information verification
- Information gathering and research
- Market research and competitive analysis

### Prerequisites
- Basic understanding of YAML syntax
- Experience with intermediate pipeline patterns
- Understanding of error handling and system integration
- Familiarity with external APIs and tools

### Key Concepts
- Conditional logic and branching
- Data flow between pipeline steps
- Error handling and recovery
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 19 template patterns for dynamic content
- **feature_highlights**: Demonstrates 12 key orchestrator features

### Data Flow
This pipeline processes input parameters through 12 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# Enhanced Research Pipeline - Issue #199 Declarative Syntax Demo
# This pipeline demonstrates the new enhanced YAML syntax that allows users
# to focus on WHAT they want to accomplish rather than HOW the graph should be structured

id: enhanced_research_pipeline
name: "Enhanced Research Pipeline with Type Safety"
description: "Demonstrates Issue #199 enhanced declarative syntax with type-safe inputs/outputs and intelligent control flow"
type: workflow
version: "2.0.0"

# Type-safe input definitions with validation
inputs:
  topic:
    type: string
    description: "Research topic to investigate"
    required: true
    example: "artificial intelligence in healthcare"
    validation:
      min_length: 3
      max_length: 200

  research_depth:
    type: string
    description: "Depth of research to conduct"
    enum: ["surface", "standard", "comprehensive", "expert"]
    default: "standard"

  max_sources:
    type: integer
    description: "Maximum number of sources to analyze"
    range: [5, 100]
    default: 20

  include_recent_only:
    type: boolean
    description: "Only include sources from the last 2 years"
    default: true

  output_formats:
    type: array
    description: "Desired output formats"
    default: ["markdown", "pdf"]
    validation:
      allowed_values: ["markdown", "pdf", "json", "html"]

# Type-safe output definitions with sources
outputs:
  research_report:
    type: file
    description: "Comprehensive research report"
    source: "{{ save_report.file_path }}"
    format: "markdown"
    location: "./reports/{{ inputs.topic | slugify }}_{{ execution.timestamp }}.md"

  source_analysis:
    type: json
    description: "Structured analysis of all sources"
    source: "{{ analyze_sources.structured_data }}"
    schema:
      total_sources: integer
      credibility_scores: array
      key_findings: array

  credibility_metrics:
    type: object
    description: "Overall credibility assessment"
    source: "{{ final_assessment.metrics }}"
    schema:
      overall_score: float
      confidence_interval: array
      reliability_factors: object

  generated_files:
    type: array
    description: "All generated output files"
    source: "{{ output_generation.file_paths }}"

# Enhanced declarative steps - users specify WHAT, system determines HOW
steps:
  # Step 1: Intelligent web search with auto-optimization
  - id: comprehensive_search
    tool: web-search
    description: "Conduct comprehensive web search with intelligent query optimization"
    inputs:
      query: "{{ inputs.topic }} {{ 'recent developments' if inputs.include_recent_only else 'comprehensive overview' }}"
      max_results: "{{ inputs.max_sources }}"
      depth: "{{ inputs.research_depth }}"
      date_filter: "{{ '2years' if inputs.include_recent_only else 'all' }}"
      quality_threshold: 0.7
    outputs:
      search_results:
        type: array
        description: "Comprehensive search results"
        schema:
          title: string
          url: string
          content: string
          publication_date: string
          credibility_score: float
          relevance_score: float
      search_metadata:
        type: object
        description: "Search execution metadata"
        schema:
          total_results: integer
          search_time: float
          query_optimization: array

  # Step 2: Content extraction with error handling
  - id: extract_content
    tool: headless-browser
    description: "Extract full content from top sources"
    depends_on: [comprehensive_search]
    condition: "{{ comprehensive_search.search_results | length > 0 }}"
    inputs:
      urls: "{{ comprehensive_search.search_results[:10] | map(attribute='url') | list }}"
      extract_method: "intelligent"
      timeout: 30
    outputs:
      extracted_content:
        type: array
        description: "Extracted content from sources"
        schema:
          url: string
          title: string
          content: string
          word_count: integer
          extraction_quality: float
      extraction_errors:
        type: array
        description: "URLs that failed extraction"
    continue_on_error: true
    retry_config:
      max_retries: 2
      backoff_factor: 2

  # Step 3: Advanced content analysis with model selection
  - id: analyze_sources
    action: analyze_text
    description: "Perform comprehensive analysis of extracted content"
    depends_on: [extract_content]
    inputs:
      content_list: "{{ extract_content.extracted_content }}"
      analysis_prompt: |
        Analyze these research sources about {{ inputs.topic }}:
        
        {% for item in extract_content.extracted_content %}
        Source {{ loop.index }}: {{ item.title }}
        Content: {{ item.content | truncate(2000) }}
        ---
        {% endfor %}
        
        Provide:
        1. Key findings and insights
        2. Credibility assessment for each source
        3. Identification of claims that need fact-checking
        4. Summary of consensus vs. conflicting viewpoints
      model: <AUTO task="analyze" complexity="high">Select high-capability model for comprehensive analysis</AUTO>
      max_tokens: 3000
      analysis_type: "comprehensive_research"
    outputs:
      structured_analysis:
        type: object
        description: "Structured analysis results"
        schema:
          key_findings: array
          source_credibility: array
          claims_to_verify: array
          consensus_points: array
          conflicting_viewpoints: array
      structured_data:
        type: json
        description: "Machine-readable analysis data"

# Advanced control flow steps demonstrating Issue #199 features
advanced_steps:
  # Parallel fact-checking with intelligent batching
  - id: fact_check_claims
    type: parallel_map
    description: "Fact-check identified claims in parallel"
    condition: "{{ analyze_sources.structured_analysis.claims_to_verify | length > 0 }}"
    items: "{{ analyze_sources.structured_analysis.claims_to_verify }}"
    max_parallel: 4
    depends_on: [analyze_sources]
    tool: fact-checker
    inputs:
      claim: "{{ item.claim_text }}"
      context: "{{ item.context }}"
      sources: "{{ item.supporting_sources }}"
      verification_level: "{{ inputs.research_depth }}"
    outputs:
      verification_result:
        type: string
        enum: ["verified", "disputed", "inconclusive", "false"]
      confidence_score:
        type: float
        range: [0.0, 1.0]
      supporting_evidence:
        type: array
        description: "Evidence supporting the verification"
      verification_sources:
        type: array
        description: "Sources used for verification"
    timeout: 120
    retry_config:
      max_retries: 1

  # Iterative quality improvement loop
  - id: quality_enhancement_loop
    type: loop
    description: "Iteratively improve report quality until threshold is met"
    depends_on: [fact_check_claims]
    loop_condition: "{{ current_quality_score < 0.85 }}"
    max_iterations: 3
    steps:
      - id: assess_quality
        action: analyze_text
        description: "Assess current report quality"
        inputs:
          text: "{{ current_report_draft }}"
          quality_metrics: ["clarity", "completeness", "accuracy", "coherence"]
          model: <AUTO task="analyze">Select model for quality assessment</AUTO>
        outputs:
          quality_score:
            type: float
            range: [0.0, 1.0]
          improvement_suggestions:
            type: array

      - id: improve_content
        action: generate_text
        depends_on: [assess_quality]
        condition: "{{ assess_quality.quality_score < 0.85 }}"
        description: "Improve report based on quality assessment"
        inputs:
          original_content: "{{ current_report_draft }}"
          improvements: "{{ assess_quality.improvement_suggestions }}"
          enhancement_prompt: |
            Improve this research report based on these suggestions:
            {{ assess_quality.improvement_suggestions | join('\n- ') }}
            
            Original report:
            {{ current_report_draft }}
          model: <AUTO task="generate" expertise="high">Select expert model for content improvement</AUTO>
          max_tokens: 4000
        outputs:
          improved_content:
            type: string
            description: "Enhanced report content"

  # Comprehensive report compilation
  - id: compile_final_report
    tool: report-generator
    description: "Compile comprehensive final report"
    depends_on: [quality_enhancement_loop]
    inputs:
      topic: "{{ inputs.topic }}"
      research_depth: "{{ inputs.research_depth }}"
      sources: "{{ comprehensive_search.search_results }}"
      analysis: "{{ analyze_sources.structured_analysis }}"
      fact_check_results: "{{ fact_check_claims.results }}"
      enhanced_content: "{{ quality_enhancement_loop.final_content | default(analyze_sources.structured_analysis) }}"
      output_formats: "{{ inputs.output_formats }}"
      template: "comprehensive_research_report"
    outputs:
      final_report:
        type: string
        format: markdown
        description: "Complete research report"
      credibility_assessment:
        type: object
        schema:
          overall_credibility: float
          source_reliability: array
          verification_summary: object
      report_metadata:
        type: object
        schema:
          generation_time: string
          sources_analyzed: integer
          claims_verified: integer
          quality_score: float

  # Intelligent output generation based on requested formats
  - id: output_generation
    type: parallel_map
    description: "Generate outputs in requested formats"
    items: "{{ inputs.output_formats }}"
    max_parallel: 2
    depends_on: [compile_final_report]
    dynamic_routing:
      markdown: generate_markdown_output
      pdf: generate_pdf_output
      json: generate_json_output
      html: generate_html_output
    steps:
      - id: generate_markdown_output
        condition: "{{ item == 'markdown' }}"
        tool: filesystem
        inputs:
          action: write
          path: "{{ outputs.research_report.location }}"
          content: "{{ compile_final_report.final_report }}"

      - id: generate_pdf_output
        condition: "{{ item == 'pdf' }}"
        tool: pdf-compiler
        inputs:
          markdown_content: "{{ compile_final_report.final_report }}"
          output_path: "{{ outputs.research_report.location | replace('.md', '.pdf') }}"
          title: "Research Report: {{ inputs.topic }}"
          author: "Enhanced Research Pipeline"

      - id: generate_json_output
        condition: "{{ item == 'json' }}"
        tool: filesystem
        inputs:
          action: write
          path: "{{ outputs.research_report.location | replace('.md', '.json') }}"
          content: |
            {
              "topic": "{{ inputs.topic }}",
              "analysis": {{ analyze_sources.structured_data | tojson }},
              "fact_check_results": {{ fact_check_claims.results | tojson }},
              "credibility_assessment": {{ compile_final_report.credibility_assessment | tojson }},
              "metadata": {{ compile_final_report.report_metadata | tojson }}
            }

      - id: generate_html_output
        condition: "{{ item == 'html' }}"
        tool: markdown-to-html
        inputs:
          markdown_content: "{{ compile_final_report.final_report }}"
          output_path: "{{ outputs.research_report.location | replace('.md', '.html') }}"
          css_theme: "academic"

    outputs:
      file_paths:
        type: array
        description: "Paths to all generated files"
      generation_summary:
        type: object
        description: "Summary of output generation"

  # Final assessment and metrics calculation
  - id: final_assessment
    action: analyze_text
    description: "Generate final credibility and quality metrics"
    depends_on: [output_generation]
    inputs:
      report_content: "{{ compile_final_report.final_report }}"
      source_data: "{{ comprehensive_search.search_results }}"
      verification_results: "{{ fact_check_claims.results }}"
      assessment_prompt: |
        Provide a final assessment of this research report on {{ inputs.topic }}:
        
        Report: {{ compile_final_report.final_report | truncate(3000) }}
        
        Consider:
        - Source quality and diversity
        - Fact-checking verification results
        - Completeness of coverage
        - Analytical depth
        
        Provide numerical scores and confidence intervals.
      model: <AUTO task="analyze" expertise="expert">Select expert model for final assessment</AUTO>
      max_tokens: 1000
    outputs:
      metrics:
        type: object
        description: "Final quality and credibility metrics"
        schema:
          overall_score: float
          confidence_interval: array
          reliability_factors: object
          recommendations: array

# Enhanced configuration with intelligent optimization
config:
  timeout: 7200  # 2 hours for comprehensive research
  retry_policy: "exponential_backoff"
  parallel_optimization: true
  error_handling: "continue_with_degraded_quality"
  model_fallbacks: true
  cache_intermediate_results: true
  checkpoint_frequency: "after_major_steps"

# Comprehensive metadata
metadata:
  version: "2.0.0"
  created_by: "Issue #199 Enhanced Syntax"
  syntax_version: "enhanced_declarative_v2"
  estimated_execution_time: "45-90 minutes"
  cost_estimate: "$2-8 depending on model selection"
  capabilities:
    - "Type-safe input/output validation"
    - "Intelligent parallel processing"
    - "Adaptive quality improvement loops" 
    - "Multi-format output generation"
    - "Comprehensive fact-checking"
    - "Real-time credibility assessment"
  use_cases:
    - "Academic research"
    - "Due diligence investigations"
    - "Market research"
    - "Competitive analysis"
    - "Policy research"

# Advanced features from Issue #199
error_handling:
  strategy: "graceful_degradation"
  fallback_models: true
  partial_results: "acceptable"
  critical_failures: ["fact_check_claims", "compile_final_report"]

monitoring:
  track_execution_time: true
  log_model_usage: true
  measure_quality_metrics: true
  alert_on_low_credibility: true
  dashboard_updates: "real_time"

optimization:
  auto_select_models: true
  dynamic_parallelization: true
  intelligent_caching: true
  cost_optimization: "balanced"
  quality_vs_speed: "prioritize_quality"
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
- 1.5. Ensure you have access to required services: Web search APIs
- 2. Run: python scripts/run_pipeline.py examples/enhanced_research_pipeline.yaml
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

*Tutorial generated on 2025-08-27T23:40:24.396050*
