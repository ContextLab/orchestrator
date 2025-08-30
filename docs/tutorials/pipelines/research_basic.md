# Pipeline Tutorial: research_basic

## Overview

**Complexity Level**: Intermediate  
**Difficulty Score**: 45/100  
**Estimated Runtime**: 10-30 minutes  

### Purpose
This pipeline demonstrates how to build automated research workflows using the orchestrator toolbox. It showcases data_flow, interactive_workflows, llm_integration and provides a foundation for building more sophisticated research applications.

### Use Cases
- AI-powered content generation
- Academic research and literature review
- Fact-checking and information verification
- Information gathering and research
- Market research and competitive analysis

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
id: research_basic
name: Basic Research Pipeline
description: Generate research reports using only standard LLM actions (no specialized tools required)
version: "1.0.0"

parameters:
  topic:
    type: string
    required: true
    description: Research topic
  depth:
    type: string
    default: comprehensive
    description: Research depth (basic, comprehensive, or expert)
  output_path:
    type: string
    default: "examples/outputs/research_basic"
    description: Directory where output files will be saved

steps:
  - id: initial_search
    tool: web-search
    action: search
    parameters:
      query: "{{topic}} overview introduction basics"
      max_results: 10
  
  - id: deep_search
    tool: web-search
    action: search
    parameters:
      query: "{{topic}} advanced research latest developments 2024 2025"
      max_results: 10
    dependencies:
      - initial_search
  
  - id: extract_key_points
    action: analyze_text
    parameters:
      text: |
        Topic: {{topic}}
        
        Initial search results:
        {% for result in initial_search.results %}
        - {{result.title}}: {{result.snippet}}
        {% endfor %}
        
        Deep search results:
        {% for result in deep_search.results %}
        - {{result.title}}: {{result.snippet}}
        {% endfor %}
      prompt: |
        You are an expert research analyst. Extract and synthesize the key points about {{topic}} from the search results above.
        
        Your task:
        1. Identify the most important facts, concepts, and developments
        2. Organize information thematically
        3. Note any conflicting information or debates
        4. Highlight recent developments or breakthroughs
        5. Include specific examples, statistics, or case studies when mentioned
        
        Format your response as a structured analysis with clear sections and bullet points.
        Do NOT use conversational language or phrases like "Certainly!" or "I'd be happy to help."
        Write in a direct, academic style suitable for a research report.
      model: <AUTO task="analyze">Select model for information extraction</AUTO>
      analysis_type: "key_points"
    dependencies:
      - initial_search
      - deep_search
  
  - id: generate_summary
    action: generate_text
    parameters:
      prompt: |
        Task: Write an executive summary for a {{depth}} research report on "{{topic}}" based on these key points:
        
        {{extract_key_points.result}}
        
        Requirements:
        - Length: 250-400 words
        - Style: Professional, direct, and informative
        - Structure: Opening statement, 3-4 main points, closing insight
        - Do NOT use conversational phrases or filler text
        - Do NOT ask for more information or say things like "Certainly!" or "I'd be happy to"
        - Focus on the most significant findings and their implications
        - Include specific details, numbers, or examples where available
        - Start directly with the content, no meta-commentary
        
        Begin the executive summary immediately:
      model: <AUTO task="generate">Select model for summary generation</AUTO>
      max_tokens: 500
    dependencies:
      - extract_key_points
  
  - id: generate_analysis
    action: generate_text
    parameters:
      prompt: |
        Provide a {{depth}} analysis of {{topic}} based on these findings:
        
        {{extract_key_points.result}}
        
        Structure your analysis with these sections:
        1. Current State and Trends
        2. Key Challenges and Opportunities  
        3. Future Directions and Implications
        
        Requirements:
        - Use specific examples and evidence from the findings
        - Identify patterns and connections between different aspects
        - Provide balanced perspective on controversies or debates
        - Make concrete predictions or recommendations where appropriate
        - Write in analytical, academic style
        - Avoid generic statements or filler content
        
        Depth level "{{depth}}" means:
        - basic: Focus on fundamental concepts and major trends
        - comprehensive: Include detailed analysis and multiple perspectives
        - expert: Deep dive with technical details and nuanced insights
      model: <AUTO task="analyze">Select model for analysis</AUTO>
      max_tokens: 1000
    dependencies:
      - extract_key_points
  
  - id: generate_conclusion
    action: generate_text
    parameters:
      prompt: |
        Write a compelling conclusion for the research report on {{topic}} that synthesizes these elements:
        
        Executive Summary:
        {{generate_summary.result}}
        
        Key Analysis Points:
        {{generate_analysis.result}}
        
        Requirements:
        - Synthesize the most important findings into 2-3 key takeaways
        - Highlight surprising or particularly significant insights
        - Discuss broader implications for the field
        - Suggest areas for future research or action
        - End with a forward-looking statement
        - Length: 150-250 words
        - Avoid clichés like "This research examined X through web sources"
        - Make it substantive and specific to {{topic}}
        - Start directly with the conclusion content
      model: <AUTO task="generate">Select model for conclusion writing</AUTO>
      max_tokens: 300
    dependencies:
      - generate_summary
      - generate_analysis
  
  - id: save_report
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_path }}/research_{{topic | slugify}}.md"
      content: |
        # Research Report: {{topic}}
        
        **Date:** {{ execution.timestamp }}
        **Research Depth:** {{depth}}
        
        ## Executive Summary
        
        {{generate_summary.result}}
        
        ## Introduction
        
        This report presents a {{depth}} analysis of {{topic}} based on current web sources.
        
        ## Key Findings
        
        {{extract_key_points.result}}
        
        ## Analysis
        
        {{generate_analysis.result}}
        
        ## Sources
        
        ### Initial Search Results ({{initial_search.total_results}} found)
        {% for result in initial_search.results %}
        - [{{result.title}}]({{result.url}})
        {% endfor %}
        
        ### Deep Search Results ({{deep_search.total_results}} found)
        {% for result in deep_search.results %}
        - [{{result.title}}]({{result.url}})
        {% endfor %}
        
        ## Conclusion
        
        {{generate_conclusion.result}}
        
        ---
        *Report generated by Orchestrator Research Pipeline*
    dependencies:
      - generate_summary
      - generate_analysis
      - generate_conclusion

outputs:
  report_file: "{{save_report.filepath}}"
  total_sources: "{{initial_search.total_results + deep_search.total_results}}"
  key_findings: "{{extract_key_points.result}}"
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
- 2. Run: python scripts/run_pipeline.py examples/research_basic.yaml
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

*Tutorial generated on 2025-08-27T23:40:24.396545*
