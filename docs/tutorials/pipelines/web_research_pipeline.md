# Pipeline Tutorial: web_research_pipeline

## Overview

**Complexity Level**: Advanced  
**Difficulty Score**: 75/100  
**Estimated Runtime**: 30+ minutes  

### Purpose
This pipeline demonstrates how to build automated research workflows using the orchestrator toolbox. It showcases conditional_execution, data_flow, for_loops and provides a foundation for building more sophisticated research applications.

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
- Iterative processing with loops
- Large language model integration
- Template variable substitution

## Pipeline Breakdown

### Configuration Analysis
- **input_section**: Defines the data inputs and parameters for the pipeline
- **steps_section**: Contains the sequence of operations to be executed
- **output_section**: Specifies how results are formatted and stored
- **template_usage**: Uses 14 template patterns for dynamic content
- **feature_highlights**: Demonstrates 9 key orchestrator features

### Data Flow
This pipeline processes input parameters through 9 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
id: web-research-pipeline
# Web Research Pipeline
# Automated research workflow with search, analysis, and reporting

name: web-research-pipeline
description: Comprehensive web research automation
version: "1.0.0"

inputs:
  research_topic: "artificial intelligence in healthcare"
  max_sources: 10
  output_format: "pdf"
  research_depth: "comprehensive"  # quick, standard, comprehensive
  output_path: "examples/outputs/web_research_pipeline"  # Directory for output files

steps:
  # Step 1: Initial web search
  - id: initial_search
    tool: web-search
    action: search
    parameters:
      query: "{{ research_topic }} latest developments 2024"
      max_results: "{{ max_sources }}"
      region: "us-en"
      safesearch: "moderate"
      
  # Step 2: Extract key themes from search results
  - id: identify_themes
    action: analyze_text
    parameters:
      text: |
        Search results for "{{ research_topic }}":
        {% for result in initial_search.results %}
        {{ loop.index }}. {{ result.title }}
        {{ result.snippet }}
        URL: {{ result.url }}
        
        {% endfor %}
      analysis_type: "theme_extraction"
      prompt: |
        Analyze these search results and extract:
        1. Main themes (5-7 key topics)
        2. Key organizations mentioned
        3. Trending subtopics
        
        Format as JSON with fields: main_themes, key_organizations, trending_subtopics
      model: <AUTO>Choose a model for theme extraction</AUTO>
              
  # Step 3: Deep search on identified themes
  - id: themed_searches
    tool: web-search
    action: search
    condition: "{{ research_depth in ['standard', 'comprehensive'] }}"
    parallel: true  # Run searches in parallel
    foreach: "{{ identify_themes.result.trending_subtopics[:3] }}"
    parameters:
      query: "{{ research_topic }} {{ item }}"
      max_results: 5
      
  # Step 4: Fetch and analyze top sources
  - id: fetch_content
    tool: headless-browser
    action: extract
    foreach: "{{ initial_search.results[:5] }}"  # Top 5 results
    parameters:
      url: "{{ item.url }}"
      extract_main_content: true
      include_metadata: true
      timeout: 10000
    on_failure: continue  # Skip failed fetches
    
  # Step 5: Analyze each source
  - id: analyze_sources
    action: analyze_text
    foreach: "{{ fetch_content.results }}"
    parameters:
      text: "{{ item.content | truncate(5000) }}"
      analysis_type: "research_extraction"
      prompt: |
        Analyze this source about {{ research_topic }} and extract:
        1. Key findings (main points, discoveries, conclusions)
        2. Data points (statistics, metrics with context)
        3. Credibility score (0-1 based on sources, citations, authority)
        4. Relevant quotes (2-3 most important)
        
        Format as JSON with fields:
        - key_findings: array of strings
        - data_points: array of {metric, value, context}
        - credibility_score: number 0-1
        - relevant_quotes: array of strings
      model: <AUTO>Choose a model for research extraction</AUTO>
              
  # Step 6: Cross-reference and validate findings
  - id: validate_findings
    action: analyze_text
    condition: "{{ research_depth == 'comprehensive' }}"
    parameters:
      text: |
        Research findings from multiple sources:
        {% for analysis in analyze_sources.results %}
        Source {{ loop.index }}:
        {{ analysis.result | json_encode }}
        {% endfor %}
      analysis_type: "fact_checking"
      prompt: |
        Cross-reference these research findings:
        1. Identify corroborated facts (mentioned in multiple sources)
        2. Flag any conflicting information
        3. Rate overall confidence in findings (high/medium/low)
        4. Note any gaps or missing information
      model: <AUTO>Choose a model for fact checking</AUTO>
      
  # Step 7: Generate research summary
  - id: research_summary
    action: generate_text
    parameters:
      prompt: |
        Create a comprehensive research summary on "{{ research_topic }}" based on these findings:
        
        Main themes: {{ identify_themes.result.main_themes | join(', ') }}
        
        Key findings from sources:
        {% for analysis in analyze_sources.results %}
        {% for finding in analysis.result.key_findings %}
        - {{ finding }}
        {% endfor %}
        {% endfor %}
        
        Include sections for:
        1. Executive Summary
        2. Current State of {{ research_topic }}
        3. Key Players and Organizations  
        4. Recent Developments
        5. Data and Statistics
        6. Future Outlook
        7. Recommendations
      model: <AUTO>Choose a model for comprehensive writing</AUTO>
      temperature: 0.3
      max_tokens: 2000
      
  # Step 8: Create bibliography
  - id: create_bibliography
    tool: report-generator
    action: generate
    parameters:
      title: "References"
      format: "markdown"
      content: |
        # Bibliography
        
        ## Primary Sources
        {% for result in initial_search.results[:max_sources] %}
        {{ loop.index }}. {{ result.title }}
           - URL: {{ result.url }}
           - Accessed: {{ current_date }}
           - Relevance: {{ result.relevance_score }}
        {% endfor %}
        
        ## Additional Sources
        {% for search in themed_searches.results %}
        ### {{ search.query }}
        {% for result in search.results %}
        - {{ result.title }} ({{ result.url }})
        {% endfor %}
        {% endfor %}
        
  # Step 9: Generate final report
  - id: generate_report
    tool: report-generator
    action: generate
    parameters:
      title: "Research Report: {{ research_topic | title }}"
      format: "{{ output_format }}"
      template: |
        # Research Report: {{ research_topic | title }}
        
        **Date:** {{ current_date }}
        **Research Depth:** {{ research_depth }}
        **Sources Analyzed:** {{ analyze_sources.results | length }}
        
        ---
        
        {{ research_summary.result }}
        
        ---
        
        ## Detailed Findings by Source
        
        {% for analysis in analyze_sources.results %}
        ### Source {{ loop.index }}
        
        **Credibility Score:** {{ analysis.result.credibility_score }}/1.0
        
        **Key Findings:**
        {% for finding in analysis.result.key_findings %}
        - {{ finding }}
        {% endfor %}
        
        **Notable Quotes:**
        {% for quote in analysis.result.relevant_quotes[:2] %}
        > "{{ quote }}"
        {% endfor %}
        
        {% endfor %}
        
        ---
        
        {{ create_bibliography.content }}
        
        ---
        
        ## Appendix: Research Methodology
        
        This report was generated using automated web research with the following approach:
        1. Initial broad search for "{{ research_topic }}"
        2. Theme identification and focused searches
        3. Content extraction and analysis
        4. Cross-referencing and validation
        5. Synthesis and report generation
      metadata:
        author: "Orchestrator Research Pipeline"
        keywords: "{{ identify_themes.result.main_themes | join(', ') }}"
        research_date: "{{ current_date }}"
        
  # Step 10: Create executive brief (optional)
  - id: executive_brief
    action: generate_text
    condition: "{{ output_format == 'pdf' }}"
    parameters:
      prompt: |
        Create a one-page executive brief for "{{ research_topic }}" with:
        - 3 key takeaways
        - 2 critical statistics
        - 1 main recommendation
        
        Base it on: {{ research_summary.result | truncate(1000) }}
      model: <AUTO>Choose a model for concise summaries</AUTO>
      max_tokens: 500
      
  # Step 11: Save report to file
  - id: save_report
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/research_{{ research_topic | slugify }}.md"
      content: "{{ generate_report.content }}"
    dependencies:
      - generate_report

outputs:
  report_path: "{{ generate_report.filepath }}"
  sources_analyzed: "{{ analyze_sources.results | length }}"
  key_themes: "{{ identify_themes.result.main_themes }}"
  executive_brief: "{{ executive_brief.result if executive_brief else 'N/A' }}"
  total_sources_found: "{{ initial_search.total_results }}"
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
- 2. Run: python scripts/run_pipeline.py examples/web_research_pipeline.yaml
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

*Tutorial generated on 2025-08-27T23:40:24.396811*
