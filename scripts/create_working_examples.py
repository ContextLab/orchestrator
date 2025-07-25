#!/usr/bin/env python3
"""
Create working example pipelines with correct YAML syntax.
"""

from pathlib import Path

# Create simplified working examples
examples = {
    "research_simple.yaml": '''name: "Research Assistant Simple"
description: "Simple research pipeline that works"
model: "google/gemini-1.5-flash"

inputs:
  query:
    type: string
    required: true
  depth:
    type: string
    default: "comprehensive"

steps:
  - id: analyze_query
    action: |
      Analyze the research query: {{query}}
      
      Provide:
      1. Key search terms
      2. Research objectives
      3. Expected information types
      4. Related topics
    
  - id: conduct_research
    action: |
      Research the topic: {{query}}
      Search depth: {{depth}}
      
      Based on analysis: {{analyze_query.result}}
      
      Provide comprehensive information including:
      1. Overview and definitions
      2. Current state and trends
      3. Key findings and insights
      4. Practical applications
      5. Future outlook
    depends_on: [analyze_query]
    
  - id: generate_report
    action: |
      Create a research report on: {{query}}
      
      Using research: {{conduct_research.result}}
      
      Format as a professional report with:
      1. Executive summary
      2. Introduction
      3. Main findings
      4. Analysis
      5. Conclusions
      6. Recommendations
    depends_on: [conduct_research]
    
  - id: save_report
    action: |
      Save to examples/output/research_{{query | replace(' ', '_') | lower}}.md:
      
      # Research Report: {{query}}
      
      *Generated on: {{execution.timestamp}}*
      *Model: Google Gemini 1.5 Flash*
      
      ## Executive Summary
      
      {{generate_report.result | truncate(500)}}
      
      ## Full Report
      
      {{generate_report.result}}
      
      ## Research Metadata
      - Query: {{query}}
      - Depth: {{depth}}
      - Analysis: {{analyze_query.result | truncate(200)}}
      
      ---
      *Generated by Orchestrator Research Pipeline*
    depends_on: [generate_report]

outputs:
  report: "{{generate_report.result}}"
  file_path: "examples/output/research_{{query | replace(' ', '_') | lower}}.md"
''',

    "content_simple.yaml": '''name: "Content Creator Simple"
description: "Simple content creation pipeline"
model: "huggingface/mistralai/Mistral-7B-Instruct-v0.2"

inputs:
  topic:
    type: string
    required: true
  audience:
    type: string
    default: "general"
  tone:
    type: string
    default: "professional"

steps:
  - id: research_topic
    action: |
      Research the topic: {{topic}}
      
      Identify:
      1. Key concepts
      2. Current trends
      3. Target audience interests
      4. Related topics
      
  - id: create_outline
    action: |
      Create content outline for: {{topic}}
      Target audience: {{audience}}
      
      Based on research: {{research_topic.result}}
      
      Structure:
      1. Introduction hook
      2. Main sections (3-5)
      3. Key points per section
      4. Conclusion and CTA
    depends_on: [research_topic]
    
  - id: write_content
    action: |
      Write engaging content about: {{topic}}
      
      Outline: {{create_outline.result}}
      Tone: {{tone}}
      Audience: {{audience}}
      
      Create 800-1000 words including:
      1. Compelling introduction
      2. Well-structured body
      3. Examples and data
      4. Strong conclusion
    depends_on: [create_outline]
    
  - id: optimize_content
    action: |
      Optimize the content for readability and SEO:
      
      {{write_content.result}}
      
      Improvements:
      1. Add subheadings
      2. Improve flow
      3. Add keywords naturally
      4. Ensure clarity
    depends_on: [write_content]
    
  - id: save_content
    action: |
      Save to examples/output/content_{{topic | replace(' ', '_') | lower}}.md:
      
      # {{topic}}
      
      *Created on: {{execution.timestamp}}*
      *Model: HuggingFace Mistral-7B*
      *Audience: {{audience}}*
      
      {{optimize_content.result}}
      
      ---
      
      ## Content Metadata
      - Topic: {{topic}}
      - Tone: {{tone}}
      - Target Audience: {{audience}}
      - Word Count: ~900 words
      
      *Generated by Orchestrator Content Pipeline*
    depends_on: [optimize_content]

outputs:
  content: "{{optimize_content.result}}"
  outline: "{{create_outline.result}}"
  file_path: "examples/output/content_{{topic | replace(' ', '_') | lower}}.md"
''',

    "creative_simple.yaml": '''name: "Creative Writer Simple"
description: "Simple creative writing pipeline"
model: "anthropic/claude-3-sonnet-20240229"

inputs:
  theme:
    type: string
    required: true
  genre:
    type: string
    default: "science fiction"
  length:
    type: string
    default: "flash"

steps:
  - id: develop_premise
    action: |
      Create a compelling story premise:
      Theme: {{theme}}
      Genre: {{genre}}
      
      Include:
      1. Central conflict
      2. Main character concept
      3. Setting
      4. Unique hook
      
  - id: create_characters
    action: |
      Develop main characters for:
      {{develop_premise.result}}
      
      For each character:
      1. Name and role
      2. Motivation
      3. Key traits
      4. Backstory snippet
    depends_on: [develop_premise]
    
  - id: write_story
    action: |
      Write a {{length}} fiction story (500-750 words):
      
      Premise: {{develop_premise.result}}
      Characters: {{create_characters.result}}
      
      Include:
      1. Engaging opening
      2. Character development
      3. Rising action
      4. Climax
      5. Satisfying resolution
    depends_on: [create_characters]
    
  - id: add_title
    action: |
      Create a compelling title for:
      
      {{write_story.result}}
      
      The title should be:
      1. Memorable
      2. Intriguing
      3. Related to theme
    depends_on: [write_story]
    
  - id: save_story
    action: |
      Save to examples/output/story_{{theme | replace(' ', '_') | lower}}.md:
      
      # {{add_title.result}}
      
      *A {{genre}} {{length}} fiction*
      *Generated on: {{execution.timestamp}}*
      *Model: Anthropic Claude 3 Sonnet*
      
      {{write_story.result}}
      
      ---
      
      ## Story Details
      - Theme: {{theme}}
      - Genre: {{genre}}
      - Characters: {{create_characters.result | truncate(200)}}
      
      *Generated by Orchestrator Creative Writing Pipeline*
    depends_on: [write_story, add_title]

outputs:
  title: "{{add_title.result}}"
  story: "{{write_story.result}}"
  file_path: "examples/output/story_{{theme | replace(' ', '_') | lower}}.md"
''',

    "analysis_simple.yaml": '''name: "Code Analysis Simple"
description: "Simple code analysis pipeline"
model: "openai/gpt-4-turbo-preview"

inputs:
  code_path:
    type: string
    required: true
  language:
    type: string
    default: "python"

steps:
  - id: analyze_structure
    action: |
      Analyze code structure in: {{code_path}}
      Language: {{language}}
      
      Identify:
      1. Main components
      2. Architecture patterns
      3. Dependencies
      4. Entry points
      
  - id: check_quality
    action: |
      Analyze code quality:
      
      Structure: {{analyze_structure.result}}
      
      Check for:
      1. Code style issues
      2. Common anti-patterns
      3. Complexity metrics
      4. Documentation coverage
    depends_on: [analyze_structure]
    
  - id: suggest_improvements
    action: |
      Suggest improvements based on:
      
      Quality analysis: {{check_quality.result}}
      
      Provide:
      1. Priority fixes
      2. Refactoring suggestions
      3. Best practices to adopt
      4. Performance optimizations
    depends_on: [check_quality]
    
  - id: save_analysis
    action: |
      Save to examples/output/code_analysis_{{code_path | replace('/', '_')}}.md:
      
      # Code Analysis Report: {{code_path}}
      
      *Generated on: {{execution.timestamp}}*
      *Language: {{language}}*
      *Model: OpenAI GPT-4 Turbo*
      
      ## Structure Analysis
      
      {{analyze_structure.result}}
      
      ## Quality Assessment
      
      {{check_quality.result}}
      
      ## Improvement Suggestions
      
      {{suggest_improvements.result}}
      
      ---
      *Generated by Orchestrator Code Analysis Pipeline*
    depends_on: [suggest_improvements]

outputs:
  analysis: "{{check_quality.result}}"
  suggestions: "{{suggest_improvements.result}}"
  file_path: "examples/output/code_analysis_{{code_path | replace('/', '_')}}.md"
''',

    "data_simple.yaml": '''name: "Data Processor Simple"
description: "Simple data processing pipeline"
model: "ollama/llama2"

inputs:
  data_source:
    type: string
    required: true
  process_type:
    type: string
    default: "analyze"

steps:
  - id: load_data
    action: |
      Load and describe data from: {{data_source}}
      
      Provide:
      1. Data format
      2. Number of records
      3. Column/field names
      4. Data types
      
  - id: analyze_data
    action: |
      Analyze the data:
      
      Data info: {{load_data.result}}
      Process type: {{process_type}}
      
      Perform:
      1. Statistical summary
      2. Data quality checks
      3. Pattern identification
      4. Anomaly detection
    depends_on: [load_data]
    
  - id: process_data
    action: |
      Process the data based on analysis:
      
      {{analyze_data.result}}
      
      Apply:
      1. Data cleaning
      2. Transformations
      3. Feature engineering
      4. Aggregations
    depends_on: [analyze_data]
    
  - id: generate_insights
    action: |
      Generate insights from processed data:
      
      {{process_data.result}}
      
      Include:
      1. Key findings
      2. Trends identified
      3. Recommendations
      4. Next steps
    depends_on: [process_data]
    
  - id: save_report
    action: |
      Save to examples/output/data_{{data_source | replace(' ', '_') | lower}}.md:
      
      # Data Processing Report: {{data_source}}
      
      *Generated on: {{execution.timestamp}}*
      *Model: Ollama Llama2*
      
      ## Data Overview
      
      {{load_data.result}}
      
      ## Analysis Results
      
      {{analyze_data.result}}
      
      ## Processing Summary
      
      {{process_data.result}}
      
      ## Key Insights
      
      {{generate_insights.result}}
      
      ---
      *Generated by Orchestrator Data Processing Pipeline*
    depends_on: [generate_insights]

outputs:
  insights: "{{generate_insights.result}}"
  file_path: "examples/output/data_{{data_source | replace(' ', '_') | lower}}.md"
'''
}

# Save all working examples
examples_dir = Path("examples/working")
examples_dir.mkdir(exist_ok=True)

for filename, content in examples.items():
    filepath = examples_dir / filename
    filepath.write_text(content)
    print(f"✅ Created: {filepath}")

print(f"\n✨ Created {len(examples)} working example pipelines in examples/working/")