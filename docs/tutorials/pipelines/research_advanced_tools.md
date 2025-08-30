# Pipeline Tutorial: research_advanced_tools

## Overview

**Complexity Level**: Advanced  
**Difficulty Score**: 65/100  
**Estimated Runtime**: 30+ minutes  

### Purpose
This pipeline demonstrates how to build automated research workflows using the orchestrator toolbox. It showcases conditional_execution, data_flow, error_handling and provides a foundation for building more sophisticated research applications.

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
- **template_usage**: Uses 13 template patterns for dynamic content
- **feature_highlights**: Demonstrates 7 key orchestrator features

### Data Flow
This pipeline processes input parameters through 7 steps to generate the specified outputs.

### Control Flow
Follows linear execution flow from first step to last step.

### Pipeline Configuration
```yaml
# Research Pipeline with Advanced Tools
# Uses web-search tool and advanced features like content extraction and PDF generation
id: research_advanced_tools
name: Research Pipeline with Advanced Tools
description: Advanced research pipeline using web search, content extraction, and PDF generation
version: "2.0.0"

parameters:
  topic:
    type: string
    default: "quantum computing applications"
  max_results:
    type: integer
    default: 10
  compile_to_pdf:
    type: boolean
    default: true
  output_path:
    type: string
    default: "examples/outputs/research_advanced_tools"

steps:
  # Step 1: Initial Web Search
  - id: search_topic
    tool: web-search
    action: search
    parameters:
      query: "{{ topic }} latest developments"
      max_results: "{{ max_results | int }}"
      backend: "duckduckgo"
    
  # Step 2: Perform deeper specialized search
  - id: deep_search
    tool: web-search
    action: search
    parameters:
      query: "{{ topic }} research papers technical details implementation"
      max_results: "{{ max_results | int }}"
      backend: "duckduckgo"
    dependencies:
      - search_topic
      
  # Step 3: Extract content from top result using headless browser
  - id: extract_content
    tool: headless-browser
    action: scrape
    parameters:
      url: "{{ search_topic.results[0].url if search_topic.results and search_topic.results|length > 0 else deep_search.results[0].url if deep_search.results and deep_search.results|length > 0 else '' }}"
    dependencies:
      - search_topic
      - deep_search
    condition: "{{ (search_topic.results | length > 0) or (deep_search.results | length > 0) }}"
    continue_on_error: true
    
  # Step 4: Analyze and synthesize findings
  - id: analyze_findings
    action: analyze_text
    parameters:
      text: |
        Topic: {{ topic }}
        
        Primary search results ({{ search_topic.total_results }} total):
        {% for result in search_topic.results[:max_results] %}
        {{ loop.index }}. {{ result.title }}
           URL: {{ result.url }}
           Summary: {{ result.snippet }}
        {% endfor %}
        
        Technical search results ({{ deep_search.total_results }} total):
        {% for result in deep_search.results %}
        - {{ result.title }}: {{ result.snippet }}
        {% endfor %}
        
        {% if extract_content is defined and 'word_count' in extract_content and extract_content.word_count > 0 %}
        Extracted content from primary source:
        Title: {{ extract_content.title }}
        Key content: {{ extract_content.text | truncate(1500) }}...
        {% endif %}
      prompt: |
        You are an expert research analyst. Analyze the search results and extracted content for "{{ topic }}".
        
        Create a comprehensive analysis with the following structure:
        
        Start with a 2-3 paragraph overview that synthesizes the most important insights about {{ topic }}, highlighting recent developments and key themes.
        
        Then provide:
        1. Key Findings - 5-8 specific, substantive points with examples and data from the sources
        2. Technical Analysis - Deep dive into implementation details, methodologies, and specific technologies
        3. Current Trends and Future Directions - What's emerging and where the field is heading
        4. Critical Evaluation - Strengths, limitations, controversies, and gaps in current approaches
        
        Be specific, use examples from the sources, and avoid generic statements.
        Write in professional, analytical style suitable for an advanced research report.
        Do NOT include headers like "Executive Summary" or "Prepared by" statements.
        Start directly with the overview content.
      model: <AUTO task="analyze">Select model for comprehensive analysis</AUTO>
      max_tokens: 2000
      analysis_type: "comprehensive"
    dependencies:
      - search_topic
      - deep_search
      - extract_content
    
  # Step 5: Generate recommendations
  - id: generate_recommendations
    action: generate_text
    parameters:
      prompt: |
        Based on this analysis of {{ topic }}:
        
        {{ analyze_findings.result }}
        
        Create a strategic recommendations section:
        
        Start with a brief paragraph (2-3 sentences) summarizing the key opportunities and challenges that drive your recommendations.
        
        Then provide 4-6 specific, actionable recommendations that:
        - Address identified gaps or challenges
        - Leverage emerging opportunities
        - Provide clear next steps for researchers/practitioners
        - Consider both short-term and long-term perspectives
        
        Format as a numbered list with substantive explanations (3-4 sentences each).
        Be specific to {{ topic }} - avoid generic advice.
        Do NOT include any "Prepared by" statements or attribution.
      model: <AUTO task="generate">Select model for recommendations</AUTO>
      max_tokens: 1500
    dependencies:
      - analyze_findings
    
  # Step 6: Save comprehensive report directly
  - id: save_report
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/research_{{ topic | slugify }}.md"
      content: |
        # Research Report: {{ topic }}
        
        **Generated on:** {{ execution.timestamp }}
        **Total Sources Analyzed:** {{ search_topic.total_results + deep_search.total_results }}
        
        ---
        
        ## Analysis
        
        {{ analyze_findings.result }}
        
        ## Strategic Recommendations
        
        {{ generate_recommendations.result }}
        
        ## Search Results
        
        The analysis is based on {{ search_topic.total_results + deep_search.total_results }} sources discovered through systematic web searches. The primary search focused on recent developments in {{ topic }}, while the technical search targeted research papers and implementation details.
        
        ### Primary Sources (Top {{ search_topic.results[:10] | length }} of {{ search_topic.total_results }})
        {% for result in search_topic.results[:10] %}
        {{ loop.index }}. **[{{ result.title }}]({{ result.url }})**
           - {{ result.snippet }}
        {% endfor %}
        
        ### Technical Sources ({{ deep_search.total_results }} results)
        {% for result in deep_search.results[:5] %}
        {{ loop.index }}. **[{{ result.title }}]({{ result.url }})**
           - {{ result.snippet }}
        {% endfor %}
        
        ## Extracted Content Analysis
        
        {% if extract_content is defined and 'word_count' in extract_content and extract_content.word_count > 0 %}
        **Primary Source:** {{ extract_content.title }}
        **URL:** {{ extract_content.url }}
        **Content Summary:** Successfully extracted {{ extract_content.word_count }} words from the primary source.
        {% else %}
        Content extraction was not successful or was skipped.
        {% endif %}
        
        ## Methodology
        
        This comprehensive research report was generated through a multi-stage process combining automated web searches, content extraction, and AI-powered analysis. The methodology ensures broad coverage of current developments while maintaining analytical depth.
        
        ### Search Strategy
        - **Primary search**: "{{ topic }} latest developments" (yielded {{ search_topic.total_results }} results)
        - **Technical search**: "{{ topic }} research papers technical details implementation" (yielded {{ deep_search.total_results }} results)
        - **Content extraction**: Automated extraction from primary sources when accessible
        - **Analysis performed**: {{ execution.timestamp }}
        
        ## References
        
        All sources were accessed on {{ execution.date }} and are listed in order of relevance.
        
        {% for result in search_topic.results[:5] %}
        {{ loop.index }}. {{ result.title }}. Available at: {{ result.url }}
        {% endfor %}
        {% for result in deep_search.results[:5] %}
        {{ loop.index + 5 }}. {{ result.title }}. Available at: {{ result.url }}
        {% endfor %}
        
        ---
        *This report was automatically generated by the Orchestrator Advanced Research Pipeline v2.0*
    dependencies:
      - search_topic
      - deep_search
      - extract_content
      - analyze_findings
      - generate_recommendations
    
  # Step 7: Generate PDF from saved markdown file (optional)
  - id: read_report
    tool: filesystem
    action: read
    parameters:
      path: "{{ output_path }}/research_{{ topic | slugify }}.md"
    dependencies:
      - save_report
    condition: "{{ compile_to_pdf }}"
    continue_on_error: true
    
  - id: compile_pdf
    tool: pdf-compiler
    action: compile
    parameters:
      markdown_content: "{{ read_report.content }}"
      output_path: "{{ output_path }}/research_{{ topic | slugify }}.pdf"
      title: "Research Report: {{ topic }}"
      author: "AI Research Assistant"
      install_if_missing: true
    dependencies:
      - read_report
    condition: "{{ compile_to_pdf }}"
    continue_on_error: true

outputs:
  report_file: "{{ save_report.filepath }}"
  pdf_file: "{{ compile_pdf.output_path if compile_pdf.success else 'PDF generation skipped' }}"
  total_sources: "{{ search_topic.total_results + deep_search.total_results }}"
  primary_source: "{{ search_topic.results[0].url if search_topic.results else 'No results found' }}"
  analysis: "{{ analyze_findings.result }}"

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
- This pipeline produces Analysis results, Markdown documents, Reports - adjust output configuration accordingly

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
- 2. Run: python scripts/run_pipeline.py examples/research_advanced_tools.yaml
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
- **Complex Logic Errors**: Review the pipeline configuration and ensure all advanced features are properly configured
- **General Execution Errors**: Check the logs for specific error messages and verify your orchestrator installation

### Verification Steps
- Check that the pipeline completed without errors
- Verify all expected output files were created
- Review the output content for quality and accuracy
- Review generated content for relevance and quality

---

*Tutorial generated on 2025-08-27T23:40:24.396520*
