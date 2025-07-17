Research Assistant Pipeline
===========================

This example demonstrates how to build a comprehensive research assistant using the Orchestrator's declarative YAML framework. The assistant can conduct web research, analyze findings, synthesize information, and generate detailed reports - all defined in pure YAML with no custom Python code required.

.. note::
   **Level:** Intermediate  
   **Duration:** 30-45 minutes  
   **Prerequisites:** Orchestrator framework installed, API keys configured

Overview
--------

The Research Assistant pipeline performs the following workflow:

1. **Query Analysis**: Analyze research query and generate search terms
2. **Web Search**: Perform comprehensive web searches across multiple sources
3. **Content Extraction**: Extract and clean relevant content from web pages
4. **Credibility Analysis**: Evaluate source reliability and quality
5. **Information Synthesis**: Analyze and synthesize findings from multiple sources
6. **Report Generation**: Create structured research reports with citations
7. **PDF Export**: Generate professional PDF reports
8. **Quality Assurance**: Validate findings and check for accuracy

**Key Features Demonstrated:**
- Declarative YAML pipeline definition
- AUTO tag resolution for natural language task descriptions
- Conditional execution based on previous results
- Parallel loop processing for content extraction
- Advanced error handling with retry logic
- Automatic tool discovery and execution
- No Python code required

Quick Start
-----------

.. code-block:: bash

   # Set up environment variables
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   
   # Run the research assistant
   orchestrator run examples/research_assistant.yaml \
     --input query="Latest developments in quantum computing" \
     --input context="Focus on practical applications in 2024"

Complete YAML Pipeline
----------------------

Here's the complete research assistant pipeline defined in declarative YAML:

.. code-block:: yaml

   name: "Research Assistant Pipeline"
   description: "Comprehensive research assistant using declarative YAML framework"

   inputs:
     query:
       type: string
       description: "Research question or topic"
       required: true
     
     context:
       type: string
       description: "Additional context to guide research"
       default: ""
     
     max_sources:
       type: integer
       description: "Maximum number of sources to analyze"
       default: 10
     
     quality_threshold:
       type: float
       description: "Minimum quality score for sources"
       default: 0.7

   steps:
     # Step 1: Analyze the research query
     - id: analyze_query
       action: <AUTO>analyze the research query "{{query}}" with context "{{context}}" and generate:
         1. Refined search terms for web searching
         2. Key research objectives
         3. Expected types of sources (academic, news, technical, etc.)
         4. Focus areas and subtopics to explore</AUTO>
       cache_results: true
       timeout: 10.0
       tags: ["analysis", "query-processing"]

     # Step 2: Conduct web search
     - id: web_search
       action: <AUTO>search the web for information about {{query}} using the search terms from {{analyze_query.result}} and find up to {{max_sources}} high-quality sources</AUTO>
       depends_on: [analyze_query]
       on_error:
         action: <AUTO>use alternative search strategies or reduce the number of sources</AUTO>
         continue_on_error: true
         retry_count: 2
       timeout: 30.0
       tags: ["search", "data-collection"]

     # Step 3: Extract content from sources (parallel processing)
     - id: extract_content
       action: <AUTO>extract and clean the main content from each web source</AUTO>
       depends_on: [web_search]
       condition: "{{web_search.success}} == true"
       loop:
         foreach: "{{web_search.results}}"
         parallel: true
         max_iterations: "{{max_sources}}"
         collect_results: true
       timeout: 20.0
       tags: ["extraction", "data-processing"]

     # Step 4: Analyze source credibility
     - id: analyze_credibility
       action: <AUTO>analyze the credibility and reliability of each extracted source based on:
         1. Domain authority and reputation
         2. Author credentials
         3. Publication date and relevance
         4. Content quality and depth
         5. Citations and references
         Return a credibility score (0-1) for each source</AUTO>
       depends_on: [extract_content]
       condition: "{{extract_content.iteration_count}} > 0"
       tags: ["analysis", "quality-control"]

     # Step 5: Filter reliable sources
     - id: filter_sources
       action: <AUTO>filter sources with credibility score >= {{quality_threshold}} and organize by relevance to {{query}}</AUTO>
       depends_on: [analyze_credibility]
       cache_results: true
       tags: ["filtering", "quality-control"]

     # Step 6: Synthesize information
     - id: synthesize_findings
       action: <AUTO>synthesize information from all reliable sources about {{query}} and extract:
         1. Key findings and insights
         2. Common themes and patterns
         3. Contradictions or debates
         4. Supporting evidence
         5. Knowledge gaps
         Organize findings by importance and relevance</AUTO>
       depends_on: [filter_sources]
       condition: "{{filter_sources.result}} != null"
       timeout: 60.0
       tags: ["synthesis", "analysis"]

     # Step 7: Generate research report
     - id: generate_report
       action: <AUTO>create a comprehensive research report about {{query}} including:
         1. Executive summary (2-3 paragraphs)
         2. Introduction and research objectives
         3. Key findings organized by theme
         4. Supporting evidence with proper citations
         5. Analysis and insights
         6. Limitations and knowledge gaps
         7. Recommendations for further research
         8. References in APA format
         Format as professional markdown</AUTO>
       depends_on: [synthesize_findings]
       on_error:
         action: <AUTO>generate a simplified report with available information</AUTO>
         continue_on_error: true
         fallback_value: "Unable to generate complete report - see partial results"
       timeout: 45.0
       tags: ["report", "output"]

     # Step 8: Generate PDF
     - id: export_pdf
       action: <AUTO>convert the markdown report to a professional PDF with:
         - Title: "Research Report: {{query}}"
         - Author: "Orchestrator Research Assistant"
         - Table of contents
         - Proper formatting and styling
         Save to reports/research_{{query}}_{{execution.timestamp}}.pdf</AUTO>
       depends_on: [generate_report]
       condition: "{{generate_report.success}} == true"
       tags: ["export", "output"]

     # Step 9: Quality assurance
     - id: quality_check
       action: <AUTO>evaluate the quality of the research report based on:
         1. Completeness of coverage
         2. Accuracy of information
         3. Clarity of presentation
         4. Proper citation of sources
         5. Relevance to original query
         Return quality score (0-1) and improvement suggestions</AUTO>
       depends_on: [generate_report]
       tags: ["validation", "quality-control"]

   outputs:
     report_markdown: "{{generate_report.result}}"
     pdf_path: "{{export_pdf.result}}"
     quality_score: "{{quality_check.result.score}}"
     key_findings: "{{synthesize_findings.result.key_findings}}"
     sources_analyzed: "{{filter_sources.result.count}}"
     total_sources_found: "{{web_search.result.count}}"
     search_terms_used: "{{analyze_query.result.search_terms}}"
     improvement_suggestions: "{{quality_check.result.suggestions}}"

How It Works
------------

**1. Declarative Task Definition**

Each step uses ``<AUTO>`` tags to describe what needs to be done in natural language. The framework automatically:

- Converts abstract descriptions into executable prompts
- Discovers and configures appropriate tools
- Selects optimal models for each task
- Handles errors and retries

**2. Automatic Tool Discovery**

The framework automatically identifies which tools are needed:

- Web search tasks → ``web-search`` tool
- Content extraction → ``headless-browser`` tool  
- Data analysis → ``data-processing`` tool
- Report generation → ``report-generator`` tool
- PDF creation → ``pdf-compiler`` tool

No manual tool configuration required!

**3. Advanced Control Flow**

The pipeline demonstrates advanced features:

- **Conditional Execution**: Steps only run when conditions are met
- **Parallel Loops**: Extract content from multiple sources simultaneously
- **Error Handling**: Automatic retries and fallback strategies
- **Caching**: Results cached for performance
- **Timeouts**: Protection against long-running tasks

Running the Pipeline
--------------------

**Using the CLI:**

.. code-block:: bash

   # Basic research
   orchestrator run research_assistant.yaml --input query="Climate change solutions"

   # With additional context
   orchestrator run research_assistant.yaml \
     --input query="Machine learning in healthcare" \
     --input context="Focus on diagnostic applications" \
     --input max_sources=20

   # With custom quality threshold
   orchestrator run research_assistant.yaml \
     --input query="Renewable energy trends" \
     --input quality_threshold=0.8

**Using Python SDK:**

.. code-block:: python

   from orchestrator import Orchestrator
   
   # Initialize orchestrator
   orchestrator = Orchestrator()
   
   # Run research pipeline
   result = await orchestrator.run_pipeline(
       "research_assistant.yaml",
       inputs={
           "query": "Artificial intelligence ethics",
           "context": "Focus on bias and fairness",
           "max_sources": 15
       }
   )
   
   # Access results
   print(f"Report saved to: {result['outputs']['pdf_path']}")
   print(f"Quality score: {result['outputs']['quality_score']}")
   print(f"Sources analyzed: {result['outputs']['sources_analyzed']}")

Example Output
--------------

Here's what the research assistant produces:

**1. Console Output:**

.. code-block:: text

   🔍 Research Assistant Pipeline
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✓ analyze_query: Analyzing research query... (2.3s)
   ✓ web_search: Found 47 potential sources (8.1s)
   ⟳ extract_content: Processing 10 sources in parallel...
     ✓ Source 1/10: arxiv.org (1.2s)
     ✓ Source 2/10: nature.com (2.1s)
     ✓ Source 3/10: mit.edu (1.8s)
     ...
   ✓ analyze_credibility: Evaluating source reliability (3.4s)
   ✓ filter_sources: 8 sources meet quality threshold (0.2s)
   ✓ synthesize_findings: Synthesizing information... (12.3s)
   ✓ generate_report: Creating research report... (8.7s)
   ✓ export_pdf: Generating PDF... (2.1s)
   ✓ quality_check: Quality score: 0.92/1.0 (1.8s)
   
   ✅ Pipeline completed successfully in 42.8s
   📄 Report: reports/research_quantum_computing_20240716_143022.pdf
   📊 Quality Score: 0.92/1.0
   📚 Sources Analyzed: 8/10

**2. Generated Report Structure:**

.. code-block:: markdown

   # Research Report: Quantum Computing Applications

   ## Executive Summary
   
   This report examines the latest developments in quantum computing...
   
   ## 1. Introduction
   
   ### 1.1 Research Objectives
   - Identify current state of quantum computing
   - Analyze practical applications
   - Evaluate commercial viability
   
   ## 2. Key Findings
   
   ### 2.1 Quantum Advantage Demonstrations
   Recent breakthroughs have shown...
   
   ### 2.2 Industry Applications
   Several sectors are actively exploring...
   
   ## 3. Analysis and Insights
   
   The synthesis of multiple sources reveals...
   
   ## 4. Recommendations
   
   Based on our analysis, we recommend...
   
   ## References
   
   1. Zhang, L. et al. (2024). "Quantum supremacy in..." Nature, 123(4), 567-589.
   2. Smith, J. (2024). "Commercial quantum applications..." MIT Technology Review.
   ...

Customization Options
---------------------

**1. Custom Search Strategies**

Modify the web search step to use specific search approaches:

.. code-block:: yaml

   - id: web_search
     action: <AUTO>search for {{query}} focusing on:
       - Academic papers from the last 2 years
       - Industry reports from reputable sources
       - Government publications
       - Peer-reviewed journals
       Prioritize recent, authoritative sources</AUTO>

**2. Specialized Analysis**

Add domain-specific analysis steps:

.. code-block:: yaml

   - id: technical_analysis
     action: <AUTO>perform technical analysis of findings:
       - Identify technological readiness levels
       - Assess implementation challenges
       - Evaluate cost-benefit ratios
       - Compare competing approaches</AUTO>
     depends_on: [synthesize_findings]
     condition: "{{query}} contains 'technology' or {{query}} contains 'technical'"

**3. Multi-format Output**

Generate reports in multiple formats:

.. code-block:: yaml

   - id: export_formats
     action: <AUTO>export report in multiple formats</AUTO>
     depends_on: [generate_report]
     loop:
       foreach: ["pdf", "docx", "html", "epub"]
       parallel: true

Performance Optimization
------------------------

The pipeline includes several optimizations:

**1. Caching Strategy**

- Query analysis results cached to avoid reprocessing
- Filtered sources cached for subsequent runs
- Cache TTL based on content freshness requirements

**2. Parallel Processing**

- Content extraction runs in parallel for all sources
- Multiple export formats generated simultaneously
- Independent analysis tasks executed concurrently

**3. Resource Management**

- Timeouts prevent runaway tasks
- Memory limits for large content processing
- Rate limiting for web requests

Error Handling
--------------

The pipeline handles various failure scenarios:

**1. Search Failures**

.. code-block:: yaml

   on_error:
     action: <AUTO>use alternative search strategies or reduce the number of sources</AUTO>
     continue_on_error: true
     retry_count: 2

**2. Content Extraction Issues**

- Automatic fallback to simplified extraction
- Skip inaccessible sources
- Continue with available content

**3. Report Generation Failures**

.. code-block:: yaml

   on_error:
     action: <AUTO>generate a simplified report with available information</AUTO>
     continue_on_error: true
     fallback_value: "Unable to generate complete report - see partial results"

Advanced Features
-----------------

**1. Conditional Processing**

Process sources differently based on type:

.. code-block:: yaml

   - id: process_academic
     action: <AUTO>extract citations, methodology, and findings from academic papers</AUTO>
     condition: "{{source.type}} == 'academic'"

   - id: process_news
     action: <AUTO>extract key facts, quotes, and timeline from news articles</AUTO>
     condition: "{{source.type}} == 'news'"

**2. Dynamic Tool Selection**

The framework automatically selects appropriate tools:

- Academic sources → Specialized academic parsers
- News sites → News-optimized extractors
- PDFs → PDF processing tools
- Videos → Transcript extraction tools

**3. Quality-Based Routing**

Route high-quality sources for deeper analysis:

.. code-block:: yaml

   - id: deep_analysis
     action: <AUTO>perform in-depth analysis including:
       - Cross-reference verification
       - Fact checking
       - Citation network analysis</AUTO>
     condition: "{{source.credibility_score}} >= 0.9"

Testing and Validation
----------------------

Test the pipeline with various queries:

.. code-block:: bash

   # Technical research
   orchestrator test research_assistant.yaml \
     --input query="Quantum error correction methods"

   # Business research  
   orchestrator test research_assistant.yaml \
     --input query="AI market trends 2024"

   # Scientific research
   orchestrator test research_assistant.yaml \
     --input query="CRISPR gene editing safety"

   # Policy research
   orchestrator test research_assistant.yaml \
     --input query="Climate change policy effectiveness"

Key Takeaways
-------------

This example demonstrates the power of Orchestrator's declarative framework:

1. **Zero Code Required**: Complete research pipeline in pure YAML
2. **Natural Language Tasks**: Use AUTO tags to describe tasks naturally
3. **Automatic Tool Discovery**: Framework selects appropriate tools
4. **Advanced Control Flow**: Conditions, loops, and error handling
5. **Production Ready**: Caching, timeouts, and error recovery
6. **Extensible**: Easy to add new steps or modify behavior

The declarative approach makes complex AI pipelines accessible to everyone, not just programmers.

Next Steps
----------

- Try the :doc:`data_processing_workflow` example for ETL pipelines
- Explore :doc:`multi_agent_collaboration` for complex AI systems
- Read the :doc:`../../user_guide/yaml_pipelines` guide for YAML syntax
- Check the :doc:`../../user_guide/auto_tags` guide for AUTO tag usage