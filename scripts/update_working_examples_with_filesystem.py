#!/usr/bin/env python3
"""
Update working examples to use explicit filesystem tool calls.
"""

from pathlib import Path

# Update the working examples to use filesystem tool
updated_examples = {
    "research_with_save.yaml": '''name: "Research Assistant with File Save"
description: "Research pipeline that saves output using filesystem tool"
model: "google/gemini-1.5-flash"
version: "1.0.0"

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
    
  - id: prepare_markdown
    action: |
      Format the report as markdown:
      
      # Research Report: {{query}}
      
      **Generated on:** {{execution.timestamp}}  
      **Model:** Google Gemini 1.5 Flash  
      **Research Depth:** {{depth}}
      
      ## Executive Summary
      
      {{generate_report.result | truncate(500)}}
      
      ## Full Report
      
      {{generate_report.result}}
      
      ## Research Analysis
      
      ### Key Terms and Objectives
      {{analyze_query.result | truncate(300)}}
      
      ### Research Findings
      {{conduct_research.result | truncate(500)}}
      
      ---
      *Generated by Orchestrator Research Pipeline*
    depends_on: [generate_report]
    
  - id: save_to_file
    action: "write"
    tool: "filesystem"
    parameters:
      path: "examples/output/research_{{query | replace(' ', '_') | lower}}.md"
      content: "{{prepare_markdown.result}}"
    depends_on: [prepare_markdown]

outputs:
  report: "{{generate_report.result}}"
  markdown: "{{prepare_markdown.result}}"
  file_path: "examples/output/research_{{query | replace(' ', '_') | lower}}.md"
  save_result: "{{save_to_file.result}}"
''',

    "content_with_save.yaml": '''name: "Content Creator with File Save"
description: "Content creation pipeline with filesystem save"
model: "huggingface/mistralai/Mistral-7B-Instruct-v0.2"
version: "1.0.0"

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
    
  - id: format_markdown
    action: |
      Format as markdown:
      
      # {{topic}}
      
      **Created on:** {{execution.timestamp}}  
      **Model:** HuggingFace Mistral-7B  
      **Audience:** {{audience}}  
      **Tone:** {{tone}}
      
      {{optimize_content.result}}
      
      ---
      
      ## Content Metadata
      - Topic: {{topic}}
      - Target Audience: {{audience}}
      - Content Tone: {{tone}}
      - Word Count: ~900 words
      
      ## Content Outline
      {{create_outline.result | truncate(300)}}
      
      *Generated by Orchestrator Content Pipeline*
    depends_on: [optimize_content]
    
  - id: save_content
    action: "write"
    tool: "filesystem"
    parameters:
      path: "examples/output/content_{{topic | replace(' ', '_') | lower}}.md"
      content: "{{format_markdown.result}}"
    depends_on: [format_markdown]

outputs:
  content: "{{optimize_content.result}}"
  outline: "{{create_outline.result}}"
  markdown: "{{format_markdown.result}}"
  file_path: "examples/output/content_{{topic | replace(' ', '_') | lower}}.md"
  save_result: "{{save_content.result}}"
''',

    "chatbot_with_save.yaml": '''name: "Interactive Chat Bot with Save"
description: "Chatbot demo that saves conversation to file"
model: "anthropic/claude-3-haiku-20240307"
version: "1.0.0"

inputs:
  conversation_topic:
    type: string
    required: true
  num_exchanges:
    type: integer
    default: 3
  user_persona:
    type: string
    default: "curious-learner"
  bot_persona:
    type: string
    default: "helpful-assistant"

steps:
  - id: user_intro
    action: |
      As a {{user_persona}}, generate an opening message about {{conversation_topic}}.
      
      The message should:
      1. Show genuine curiosity
      2. Ask an engaging question
      3. Be natural and conversational
      4. Set the tone for the discussion
    model: "openai/gpt-4o-mini"
    
  - id: bot_response_1
    action: |
      As a {{bot_persona}}, respond to this message about {{conversation_topic}}:
      
      User: {{user_intro.result}}
      
      Your response should:
      1. Be helpful and informative
      2. Match the {{bot_persona}} style
      3. Encourage further discussion
      4. Provide value to the user
    depends_on: [user_intro]
    
  - id: user_followup_1
    action: |
      As a {{user_persona}}, continue the conversation:
      
      Previous exchange:
      User: {{user_intro.result}}
      Assistant: {{bot_response_1.result}}
      
      Generate a follow-up that:
      1. Shows understanding
      2. Asks for clarification or more details
      3. Maintains the conversational flow
    model: "openai/gpt-4o-mini"
    depends_on: [bot_response_1]
    
  - id: bot_response_2
    action: |
      Continue as {{bot_persona}}:
      
      Conversation so far:
      User: {{user_intro.result}}
      Assistant: {{bot_response_1.result}}
      User: {{user_followup_1.result}}
      
      Provide a helpful response that:
      1. Addresses the user's question
      2. Adds depth to the discussion
      3. Maintains engagement
    depends_on: [user_followup_1]
    
  - id: format_conversation
    action: |
      Format the conversation as markdown:
      
      # AI Conversation: {{conversation_topic}}
      
      **Generated on:** {{execution.timestamp}}  
      **User Persona:** {{user_persona}}  
      **Bot Persona:** {{bot_persona}}
      
      ## Conversation Transcript
      
      ### User
      {{user_intro.result}}
      
      ### Assistant
      {{bot_response_1.result}}
      
      ### User
      {{user_followup_1.result}}
      
      ### Assistant
      {{bot_response_2.result}}
      
      ---
      
      ## Conversation Metadata
      - Topic: {{conversation_topic}}
      - Number of Exchanges: 2
      - User Model: OpenAI GPT-4o-mini (simulating {{user_persona}})
      - Assistant Model: Anthropic Claude 3 Haiku (as {{bot_persona}})
      
      ## How to Use with Real Users
      
      To adapt this for real users, replace the user simulation steps with:
      1. Web interface or CLI for user input
      2. Session management for conversation context
      3. Real-time response generation
      
      *Generated by Orchestrator Chatbot Pipeline*
    depends_on: [bot_response_2]
    
  - id: save_conversation
    action: "write"
    tool: "filesystem"
    parameters:
      path: "examples/output/chat_{{conversation_topic | replace(' ', '_') | lower}}.md"
      content: "{{format_conversation.result}}"
    depends_on: [format_conversation]

outputs:
  conversation: "{{format_conversation.result}}"
  file_path: "examples/output/chat_{{conversation_topic | replace(' ', '_') | lower}}.md"
  save_result: "{{save_conversation.result}}"
'''
}

# Save updated examples
working_dir = Path("examples/working")
working_dir.mkdir(exist_ok=True)

for filename, content in updated_examples.items():
    filepath = working_dir / filename
    filepath.write_text(content)
    print(f"✅ Created: {filepath}")

print(f"\n✨ Created {len(updated_examples)} updated example pipelines with filesystem saves")