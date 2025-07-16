========================
Web Research Automation
========================

This tutorial shows you how to build intelligent web research pipelines that can gather, validate, and synthesize information from multiple sources automatically.

What You'll Build
=================

By the end of this tutorial, you'll have created:

1. **Basic Web Search Pipeline** - Search and compile results
2. **Multi-Source Research System** - Combine web, news, and academic sources
3. **Fact-Checking Pipeline** - Validate information across sources
4. **Automated Report Generator** - Create comprehensive research reports

Prerequisites
=============

- Completed :doc:`tutorial_basics`
- Basic understanding of web technologies
- Internet connection for web searches

Tutorial 1: Basic Web Search
============================

Let's start with a simple pipeline that searches the web and compiles results.

Step 1: Create the Pipeline
---------------------------

Create a file called ``web_search.yaml``:

.. code-block:: yaml

   name: basic-web-search
   description: Search the web and compile results into a report
   
   inputs:
     query:
       type: string
       description: "Search query"
       required: true
     
     max_results:
       type: integer
       description: "Maximum number of results to return"
       default: 10
       validation:
         min: 1
         max: 50
   
   outputs:
     report:
       type: string
       value: "search_results_{{ inputs.query | slugify }}.md"
   
   steps:
     # Search the web
     - id: search
       action: search_web
       parameters:
         query: "{{ inputs.query }}"
         max_results: "{{ inputs.max_results }}"
         include_snippets: true
     
     # Compile into markdown report
     - id: compile_report
       action: generate_content
       parameters:
         prompt: |
           Create a well-organized markdown report from these search results:
           
           {{ results.search | json }}
           
           Include:
           - Executive summary
           - Key findings
           - Source links
           - Relevant details from each result
         
         style: "professional"
         format: "markdown"
     
     # Save the report
     - id: save_report
       action: write_file
       parameters:
         path: "{{ outputs.report }}"
         content: "$results.compile_report"

Step 2: Run the Pipeline
------------------------

.. code-block:: python

   import orchestrator as orc
   
   # Initialize
   orc.init_models()
   
   # Compile and run
   pipeline = orc.compile("web_search.yaml")
   
   # Search for different topics
   result1 = pipeline.run(
       query="artificial intelligence trends 2024",
       max_results=15
   )
   
   result2 = pipeline.run(
       query="sustainable energy solutions",
       max_results=10
   )
   
   print(f"Generated reports: {result1}, {result2}")

Step 3: Understanding the Results
---------------------------------

Your pipeline will create markdown files like:

.. code-block:: markdown

   # Search Results: Artificial Intelligence Trends 2024
   
   ## Executive Summary
   
   Recent searches reveal significant developments in AI across multiple domains...
   
   ## Key Findings
   
   1. **Large Language Models** - Continued advancement in reasoning capabilities
   2. **AI Safety** - Increased focus on alignment and control
   3. **Enterprise Adoption** - Growing integration in business processes
   
   ## Detailed Results
   
   ### 1. AI Breakthrough: New Model Achieves Human-Level Performance
   **Source**: [TechCrunch](https://techcrunch.com/...)
   **Summary**: Details about the latest AI advancement...

Tutorial 2: Multi-Source Research
=================================

Now let's build a more sophisticated pipeline that gathers information from multiple types of sources.

Step 1: Multi-Source Pipeline
-----------------------------

Create ``multi_source_research.yaml``:

.. code-block:: yaml

   name: multi-source-research
   description: Comprehensive research using web, news, and academic sources
   
   inputs:
     topic:
       type: string
       required: true
     
     depth:
       type: string
       description: "Research depth"
       default: "medium"
       validation:
         enum: ["light", "medium", "deep"]
     
     include_sources:
       type: array
       description: "Sources to include"
       default: ["web", "news", "academic"]
       validation:
         enum_items: ["web", "news", "academic", "patents"]
   
   outputs:
     comprehensive_report:
       type: string
       value: "research/{{ inputs.topic | slugify }}_comprehensive.md"
     
     data_file:
       type: string
       value: "research/{{ inputs.topic | slugify }}_data.json"
   
   # Research depth configuration
   config:
     research_params:
       light:
         web_results: 10
         news_results: 5
         academic_results: 3
       medium:
         web_results: 20
         news_results: 10
         academic_results: 8
       deep:
         web_results: 40
         news_results: 20
         academic_results: 15
   
   steps:
     # Parallel search across sources
     - id: search_sources
       parallel:
         # Web search
         - id: web_search
           condition: "'web' in inputs.include_sources"
           action: search_web
           parameters:
             query: "{{ inputs.topic }} comprehensive overview"
             max_results: "{{ config.research_params[inputs.depth].web_results }}"
             include_snippets: true
         
         # News search
         - id: news_search
           condition: "'news' in inputs.include_sources"
           action: search_news
           parameters:
             query: "{{ inputs.topic }}"
             max_results: "{{ config.research_params[inputs.depth].news_results }}"
             date_range: "last_month"
         
         # Academic search
         - id: academic_search
           condition: "'academic' in inputs.include_sources"
           action: search_academic
           parameters:
             query: "{{ inputs.topic }}"
             max_results: "{{ config.research_params[inputs.depth].academic_results }}"
             year_range: "2020-2024"
             peer_reviewed: true
     
     # Extract key information from each source
     - id: extract_information
       action: extract_information
       parameters:
         content: "$results.search_sources"
         extract:
           key_facts:
             description: "Important facts and findings"
           statistics:
             description: "Numerical data and metrics"
           expert_opinions:
             description: "Quotes and opinions from experts"
           trends:
             description: "Emerging trends and developments"
           challenges:
             description: "Problems and challenges mentioned"
           opportunities:
             description: "Opportunities and potential solutions"
   
     # Cross-validate information
     - id: validate_facts
       action: validate_data
       parameters:
         data: "$results.extract_information"
         rules:
           - name: "source_diversity"
             condition: "count(unique(sources)) >= 2"
             severity: "warning"
             message: "Information should be confirmed by multiple sources"
           
           - name: "recent_information"
             field: "date"
             condition: "date_diff(value, today()) <= 365"
             severity: "info"
             message: "Information is from the last year"
     
     # Generate comprehensive analysis
     - id: analyze_findings
       action: generate_content
       parameters:
         prompt: |
           Analyze the following research data about {{ inputs.topic }}:
           
           {{ results.extract_information | json }}
           
           Provide:
           1. Current state analysis
           2. Key trends identification
           3. Challenge assessment
           4. Future outlook
           5. Recommendations
           
           Base your analysis on the evidence provided and note any limitations.
         
         style: "analytical"
         max_tokens: 2000
   
     # Create structured data export
     - id: export_data
       action: transform_data
       parameters:
         data:
           topic: "{{ inputs.topic }}"
           research_date: "{{ execution.timestamp }}"
           depth: "{{ inputs.depth }}"
           sources_used: "{{ inputs.include_sources }}"
           extracted_info: "$results.extract_information"
           validation_results: "$results.validate_facts"
           analysis: "$results.analyze_findings"
         operations:
           - type: "convert_format"
             to_format: "json"
   
     # Save structured data
     - id: save_data
       action: write_file
       parameters:
         path: "{{ outputs.data_file }}"
         content: "$results.export_data"
   
     # Generate final report
     - id: create_report
       action: generate_content
       parameters:
         prompt: |
           Create a comprehensive research report about {{ inputs.topic }} using:
           
           Analysis: {{ results.analyze_findings }}
           
           Structure the report with:
           1. Executive Summary
           2. Methodology
           3. Current State Analysis
           4. Key Findings
           5. Trends and Developments
           6. Challenges and Limitations
           7. Future Outlook
           8. Recommendations
           9. Sources and References
           
           Include confidence levels for major claims.
         
         style: "professional"
         format: "markdown"
         max_tokens: 3000
   
     # Save final report
     - id: save_report
       action: write_file
       parameters:
         path: "{{ outputs.comprehensive_report }}"
         content: "$results.create_report"

Step 2: Run Multi-Source Research
---------------------------------

.. code-block:: python

   import orchestrator as orc
   
   # Initialize
   orc.init_models()
   
   # Compile pipeline
   pipeline = orc.compile("multi_source_research.yaml")
   
   # Run deep research on quantum computing
   result = pipeline.run(
       topic="quantum computing applications",
       depth="deep",
       include_sources=["web", "academic", "news"]
   )
   
   print(f"Research complete: {result}")
   
   # Run lighter research on emerging tech
   result2 = pipeline.run(
       topic="edge computing trends",
       depth="medium",
       include_sources=["web", "news"]
   )

Tutorial 3: Fact-Checking Pipeline
==================================

Let's create a pipeline that validates claims against multiple sources.

Step 1: Fact-Checker Pipeline
-----------------------------

Create ``fact_checker.yaml``:

.. code-block:: yaml

   name: fact-checker
   description: Verify claims against multiple reliable sources
   
   inputs:
     claims:
       type: array
       description: "Claims to verify"
       required: true
     
     confidence_threshold:
       type: float
       description: "Minimum confidence level to accept claims"
       default: 0.7
       validation:
         min: 0.0
         max: 1.0
   
   outputs:
     fact_check_report:
       type: string
       value: "fact_check_{{ execution.timestamp | strftime('%Y%m%d_%H%M') }}.md"
   
   steps:
     # Research each claim
     - id: research_claims
       for_each: "{{ inputs.claims }}"
       as: claim
       action: search_web
       parameters:
         query: "{{ claim }} verification facts evidence"
         max_results: 15
         include_snippets: true
     
     # Extract supporting/contradicting evidence
     - id: analyze_evidence
       for_each: "{{ inputs.claims }}"
       as: claim
       action: extract_information
       parameters:
         content: "$results.research_claims[loop.index0]"
         extract:
           supporting_evidence:
             description: "Evidence that supports the claim"
           contradicting_evidence:
             description: "Evidence that contradicts the claim"
           source_credibility:
             description: "Assessment of source reliability"
           expert_opinions:
             description: "Expert statements about the claim"
   
     # Assess credibility of each claim
     - id: assess_claims
       for_each: "{{ inputs.claims }}"
       as: claim
       action: generate_content
       parameters:
         prompt: |
           Assess the veracity of this claim: "{{ claim }}"
           
           Based on the evidence:
           {{ results.analyze_evidence[loop.index0] | json }}
           
           Provide:
           1. Verdict: True/False/Partially True/Insufficient Evidence
           2. Confidence level (0-1)
           3. Supporting evidence summary
           4. Contradicting evidence summary
           5. Overall assessment
           
           Be objective and cite specific sources.
         
         style: "analytical"
         format: "structured"
   
     # Compile fact-check report
     - id: create_fact_check_report
       action: generate_content
       parameters:
         prompt: |
           Create a comprehensive fact-check report based on:
           
           Claims assessed: {{ inputs.claims | json }}
           Assessment results: {{ results.assess_claims | json }}
           
           Format as a professional fact-checking article with:
           1. Summary of findings
           2. Individual claim assessments
           3. Methodology used
           4. Sources consulted
           5. Limitations and caveats
         
         style: "journalistic"
         format: "markdown"
   
     # Save report
     - id: save_fact_check
       action: write_file
       parameters:
         path: "{{ outputs.fact_check_report }}"
         content: "$results.create_fact_check_report"

Step 2: Use the Fact-Checker
----------------------------

.. code-block:: python

   import orchestrator as orc
   
   # Initialize
   orc.init_models()
   
   # Compile fact-checker
   fact_checker = orc.compile("fact_checker.yaml")
   
   # Check various claims
   result = fact_checker.run(
       claims=[
           "Electric vehicles produce zero emissions",
           "AI will replace 50% of jobs by 2030",
           "Quantum computers can break all current encryption",
           "Renewable energy is now cheaper than fossil fuels"
       ],
       confidence_threshold=0.8
   )
   
   print(f"Fact-check report: {result}")

Tutorial 4: Automated Report Generator
======================================

Let's build a system that generates professional reports automatically.

Step 1: Report Generator Pipeline
---------------------------------

Create ``report_generator.yaml``:

.. code-block:: yaml

   name: automated-report-generator
   description: Generate professional reports from research data
   
   inputs:
     topic:
       type: string
       required: true
     
     report_type:
       type: string
       description: "Type of report to generate"
       default: "standard"
       validation:
         enum: ["executive", "technical", "standard", "briefing"]
     
     target_audience:
       type: string
       description: "Primary audience for the report"
       default: "general"
       validation:
         enum: ["executives", "technical", "general", "academic"]
     
     sections:
       type: array
       description: "Sections to include in report"
       default: ["summary", "introduction", "analysis", "conclusion"]
   
   outputs:
     report_markdown:
       type: string
       value: "reports/{{ inputs.topic | slugify }}_{{ inputs.report_type }}.md"
     
     report_pdf:
       type: string
       value: "reports/{{ inputs.topic | slugify }}_{{ inputs.report_type }}.pdf"
     
     report_html:
       type: string
       value: "reports/{{ inputs.topic | slugify }}_{{ inputs.report_type }}.html"
   
   # Report templates by type
   config:
     report_templates:
       executive:
         style: "executive"
         length: "concise"
         focus: "strategic"
         sections: ["executive_summary", "key_findings", "recommendations", "appendix"]
       
       technical:
         style: "technical"
         length: "detailed"
         focus: "implementation"
         sections: ["introduction", "technical_analysis", "methodology", "results", "conclusion"]
       
       standard:
         style: "professional"
         length: "medium"
         focus: "comprehensive"
         sections: ["summary", "background", "analysis", "findings", "recommendations"]
       
       briefing:
         style: "concise"
         length: "short"
         focus: "actionable"
         sections: ["situation", "assessment", "recommendations"]
   
   steps:
     # Gather comprehensive research data
     - id: research_topic
       action: search_web
       parameters:
         query: "{{ inputs.topic }} comprehensive analysis research"
         max_results: 25
         include_snippets: true
     
     # Get recent news for current context
     - id: current_context
       action: search_news
       parameters:
         query: "{{ inputs.topic }}"
         max_results: 10
         date_range: "last_week"
     
     # Extract structured information
     - id: extract_report_data
       action: extract_information
       parameters:
         content:
           research: "$results.research_topic"
           news: "$results.current_context"
         extract:
           key_points:
             description: "Main points and findings"
           statistics:
             description: "Important numbers and data"
           trends:
             description: "Current and emerging trends"
           implications:
             description: "Implications and consequences"
           expert_views:
             description: "Expert opinions and quotes"
           future_outlook:
             description: "Predictions and future scenarios"
   
     # Generate executive summary
     - id: create_executive_summary
       condition: "'summary' in inputs.sections or 'executive_summary' in inputs.sections"
       action: generate_content
       parameters:
         prompt: |
           Create an executive summary for {{ inputs.target_audience }} audience about {{ inputs.topic }}.
           
           Based on: {{ results.extract_report_data.key_points | json }}
           
           Style: {{ config.report_templates[inputs.report_type].style }}
           Focus: {{ config.report_templates[inputs.report_type].focus }}
           
           Include the most critical points in 200-400 words.
         
         style: "{{ config.report_templates[inputs.report_type].style }}"
         max_tokens: 500
   
     # Generate introduction/background
     - id: create_introduction
       condition: "'introduction' in inputs.sections or 'background' in inputs.sections"
       action: generate_content
       parameters:
         prompt: |
           Write an introduction/background section about {{ inputs.topic }} for {{ inputs.target_audience }}.
           
           Context: {{ results.extract_report_data | json }}
           
           Provide necessary background and context for understanding the topic.
         
         style: "{{ config.report_templates[inputs.report_type].style }}"
         max_tokens: 800
   
     # Generate main analysis
     - id: create_analysis
       condition: "'analysis' in inputs.sections or 'technical_analysis' in inputs.sections"
       action: generate_content
       parameters:
         prompt: |
           Create a comprehensive analysis section about {{ inputs.topic }}.
           
           Data: {{ results.extract_report_data | json }}
           
           Style: {{ config.report_templates[inputs.report_type].style }}
           Audience: {{ inputs.target_audience }}
           
           Include:
           - Current state analysis
           - Trend analysis
           - Key factors and drivers
           - Challenges and opportunities
           
           Support points with specific data and examples.
         
         style: "{{ config.report_templates[inputs.report_type].style }}"
         max_tokens: 1500
   
     # Generate findings and implications
     - id: create_findings
       condition: "'findings' in inputs.sections or 'key_findings' in inputs.sections"
       action: generate_content
       parameters:
         prompt: |
           Summarize key findings and implications regarding {{ inputs.topic }}.
           
           Analysis: {{ results.create_analysis }}
           Supporting data: {{ results.extract_report_data.implications | json }}
           
           Present clear, actionable findings with implications.
         
         style: "{{ config.report_templates[inputs.report_type].style }}"
         max_tokens: 1000
   
     # Generate recommendations
     - id: create_recommendations
       condition: "'recommendations' in inputs.sections"
       action: generate_content
       parameters:
         prompt: |
           Develop actionable recommendations based on the analysis of {{ inputs.topic }}.
           
           Findings: {{ results.create_findings }}
           Target audience: {{ inputs.target_audience }}
           
           Provide specific, actionable recommendations with priorities and considerations.
         
         style: "{{ config.report_templates[inputs.report_type].style }}"
         max_tokens: 800
   
     # Generate conclusion
     - id: create_conclusion
       condition: "'conclusion' in inputs.sections"
       action: generate_content
       parameters:
         prompt: |
           Write a strong conclusion for the {{ inputs.topic }} report.
           
           Key findings: {{ results.create_findings }}
           Recommendations: {{ results.create_recommendations }}
           
           Synthesize the main points and end with a clear call to action.
         
         style: "{{ config.report_templates[inputs.report_type].style }}"
         max_tokens: 400
   
     # Assemble complete report
     - id: assemble_report
       action: generate_content
       parameters:
         prompt: |
           Compile a complete, professional report about {{ inputs.topic }}.
           
           Report type: {{ inputs.report_type }}
           Target audience: {{ inputs.target_audience }}
           
           Sections to include:
           {% if results.create_executive_summary %}
           Executive Summary: {{ results.create_executive_summary }}
           {% endif %}
           
           {% if results.create_introduction %}
           Introduction: {{ results.create_introduction }}
           {% endif %}
           
           {% if results.create_analysis %}
           Analysis: {{ results.create_analysis }}
           {% endif %}
           
           {% if results.create_findings %}
           Findings: {{ results.create_findings }}
           {% endif %}
           
           {% if results.create_recommendations %}
           Recommendations: {{ results.create_recommendations }}
           {% endif %}
           
           {% if results.create_conclusion %}
           Conclusion: {{ results.create_conclusion }}
           {% endif %}
           
           Format as a professional markdown document with:
           - Proper headings and structure
           - Table of contents
           - Professional formatting
           - Source citations where appropriate
         
         style: "professional"
         format: "markdown"
         max_tokens: 4000
   
     # Save markdown version
     - id: save_markdown
       action: write_file
       parameters:
         path: "{{ outputs.report_markdown }}"
         content: "$results.assemble_report"
   
     # Convert to PDF
     - id: create_pdf
       action: "!pandoc {{ outputs.report_markdown }} -o {{ outputs.report_pdf }} --pdf-engine=xelatex"
       error_handling:
         continue_on_error: true
         fallback:
           action: write_file
           parameters:
             path: "{{ outputs.report_pdf }}.txt"
             content: "PDF generation requires pandoc with xelatex"
   
     # Convert to HTML
     - id: create_html
       action: "!pandoc {{ outputs.report_markdown }} -o {{ outputs.report_html }} --standalone --css=style.css"
       error_handling:
         continue_on_error: true

Step 2: Generate Professional Reports
------------------------------------

.. code-block:: python

   import orchestrator as orc
   
   # Initialize
   orc.init_models()
   
   # Compile report generator
   generator = orc.compile("report_generator.yaml")
   
   # Generate executive report
   exec_report = generator.run(
       topic="artificial intelligence in healthcare",
       report_type="executive",
       target_audience="executives",
       sections=["executive_summary", "key_findings", "recommendations"]
   )
   
   # Generate technical report
   tech_report = generator.run(
       topic="blockchain scalability solutions",
       report_type="technical",
       target_audience="technical",
       sections=["introduction", "technical_analysis", "methodology", "results"]
   )
   
   # Generate standard briefing
   briefing = generator.run(
       topic="cybersecurity threats 2024",
       report_type="briefing",
       target_audience="general"
   )
   
   print(f"Generated reports: {exec_report}, {tech_report}, {briefing}")

Advanced Exercises
==================

Exercise 1: Industry Monitor
---------------------------

Create a pipeline that monitors a specific industry for news, updates, and trends:

.. code-block:: yaml

   # Hints for your solution:
   inputs:
     - name: industry
       type: string
       description: "Industry to monitor"  # Examples: "fintech", "biotech", "cleantech"
     - name: monitoring_period
       type: string
       description: "daily"  # Valid values: "daily", "weekly", "monthly"
     - name: alert_keywords
       type: list
       description: Important terms to watch for
   
   steps:
     - id: search_news
       action: search_web
       # Multiple search strategies
     - id: analyze_trends
       action: analyze
       # Trend analysis
     - id: generate_alerts
       action: filter
       # Alert generation
     - id: create_summary
       action: generate_text
       # Automated summaries

Exercise 2: Competitive Intelligence
-----------------------------------

Build a system that researches competitors and market positioning:

.. code-block:: yaml

   # Structure your pipeline to:
   # 1. Research multiple companies
   # 2. Compare features and positioning  
   # 3. Analyze market trends
   # 4. Generate competitive analysis

Exercise 3: Research Aggregator
------------------------------

Create a pipeline that combines multiple research pipelines for comprehensive analysis:

.. code-block:: python

   # Combine:
   # - Basic web search
   # - Multi-source research  
   # - Fact-checking
   # - Report generation
   
   # Into a single meta-pipeline

Solutions
=========

Complete solutions for all exercises are available in the ``examples/tutorials/web_research/`` directory.

Next Steps
==========

Now that you've mastered web research automation:

1. **Try** :doc:`tutorial_data_processing` to learn data handling
2. **Explore** :doc:`tutorial_content_generation` for AI-powered content
3. **Combine** web research with other tutorial techniques
4. **Build** your own research automation for specific domains

Tips for Production Use
=======================

1. **Rate Limiting**: Add delays between requests to respect websites
2. **Caching**: Store results to avoid redundant searches
3. **Error Handling**: Plan for network failures and API limits
4. **Source Diversity**: Use multiple search engines and sources
5. **Quality Control**: Validate information and check source credibility