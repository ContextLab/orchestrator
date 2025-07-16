"""Working tests for documentation code snippets - Batch 30."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_tutorial_data_processing_lines_1086_1091_0():
    """Test text snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 1086-1091."""
    # Description: Create a pipeline for IoT sensor data:
    content = 'Requirements:\n- Handle high-volume time series data\n- Detect sensor anomalies\n- Aggregate by time windows\n- Generate maintenance alerts'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_tutorial_data_processing_lines_1099_1104_1():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_data_processing.rst lines 1099-1104."""
    # Description: Build a social media data processing pipeline:
    import yaml
    
    content = '# Features:\n# - Extract from multiple platforms\n# - Text analysis and sentiment\n# - Trend detection\n# - Influence measurement'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_tutorial_web_research_lines_36_91_2():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 36-91."""
    # Description: Create a file called ``web_search.yaml``:
    import yaml
    
    content = 'name: basic-web-search\ndescription: Search the web and compile results into a report\n\ninputs:\n  query:\n    type: string\n    description: "Search query"\n    required: true\n\n  max_results:\n    type: integer\n    description: "Maximum number of results to return"\n    default: 10\n    validation:\n      min: 1\n      max: 50\n\noutputs:\n  report:\n    type: string\n    value: "search_results_{{ inputs.query | slugify }}.md"\n\nsteps:\n  # Search the web\n  - id: search\n    action: search_web\n    parameters:\n      query: "{{ inputs.query }}"\n      max_results: "{{ inputs.max_results }}"\n      include_snippets: true\n\n  # Compile into markdown report\n  - id: compile_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a well-organized markdown report from these search results:\n\n        {{ results.search | json }}\n\n        Include:\n        - Executive summary\n        - Key findings\n        - Source links\n        - Relevant details from each result\n\n      style: "professional"\n      format: "markdown"\n\n  # Save the report\n  - id: save_report\n    action: write_file\n    parameters:\n      path: "{{ outputs.report }}"\n      content: "$results.compile_report"'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_tutorial_web_research_lines_97_117_3():
    """Test Python snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 97-117."""
    # Description: ------------------------
    content = 'import orchestrator as orc\n\n# Initialize\norc.init_models()\n\n# Compile and run\npipeline = orc.compile("web_search.yaml")\n\n# Search for different topics\nresult1 = pipeline.run(\n    query="artificial intelligence trends 2024",\n    max_results=15\n)\n\nresult2 = pipeline.run(\n    query="sustainable energy solutions",\n    max_results=10\n)\n\nprint(f"Generated reports: {result1}, {result2}")'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
        try:
            compile(content, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_tutorial_web_research_lines_125_142_4():
    """Test markdown snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 125-142."""
    # Description: Your pipeline will create markdown files like:
    content = '# Search Results: Artificial Intelligence Trends 2024\n\n## Executive Summary\n\nRecent searches reveal significant developments in AI across multiple domains...\n\n## Key Findings\n\n1. **Large Language Models** - Continued advancement in reasoning capabilities\n2. **AI Safety** - Increased focus on alignment and control\n3. **Enterprise Adoption** - Growing integration in business processes\n\n## Detailed Results\n\n### 1. AI Breakthrough: New Model Achieves Human-Level Performance\n**Source**: [TechCrunch](https://techcrunch.com/...)\n**Summary**: Details about the latest AI advancement...'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_tutorial_web_research_lines_155_346_5():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 155-346."""
    # Description: Create ``multi_source_research.yaml``:
    import yaml
    
    content = 'name: multi-source-research\ndescription: Comprehensive research using web, news, and academic sources\n\ninputs:\n  topic:\n    type: string\n    required: true\n\n  depth:\n    type: string\n    description: "Research depth"\n    default: "medium"\n    validation:\n      enum: ["light", "medium", "deep"]\n\n  include_sources:\n    type: array\n    description: "Sources to include"\n    default: ["web", "news", "academic"]\n    validation:\n      enum_items: ["web", "news", "academic", "patents"]\n\noutputs:\n  comprehensive_report:\n    type: string\n    value: "research/{{ inputs.topic | slugify }}_comprehensive.md"\n\n  data_file:\n    type: string\n    value: "research/{{ inputs.topic | slugify }}_data.json"\n\n# Research depth configuration\nconfig:\n  research_params:\n    light:\n      web_results: 10\n      news_results: 5\n      academic_results: 3\n    medium:\n      web_results: 20\n      news_results: 10\n      academic_results: 8\n    deep:\n      web_results: 40\n      news_results: 20\n      academic_results: 15\n\nsteps:\n  # Parallel search across sources\n  - id: search_sources\n    parallel:\n      # Web search\n      - id: web_search\n        condition: "\'web\' in inputs.include_sources"\n        action: search_web\n        parameters:\n          query: "{{ inputs.topic }} comprehensive overview"\n          max_results: "{{ config.research_params[inputs.depth].web_results }}"\n          include_snippets: true\n\n      # News search\n      - id: news_search\n        condition: "\'news\' in inputs.include_sources"\n        action: search_news\n        parameters:\n          query: "{{ inputs.topic }}"\n          max_results: "{{ config.research_params[inputs.depth].news_results }}"\n          date_range: "last_month"\n\n      # Academic search\n      - id: academic_search\n        condition: "\'academic\' in inputs.include_sources"\n        action: search_academic\n        parameters:\n          query: "{{ inputs.topic }}"\n          max_results: "{{ config.research_params[inputs.depth].academic_results }}"\n          year_range: "2020-2024"\n          peer_reviewed: true\n\n  # Extract key information from each source\n  - id: extract_information\n    action: extract_information\n    parameters:\n      content: "$results.search_sources"\n      extract:\n        key_facts:\n          description: "Important facts and findings"\n        statistics:\n          description: "Numerical data and metrics"\n        expert_opinions:\n          description: "Quotes and opinions from experts"\n        trends:\n          description: "Emerging trends and developments"\n        challenges:\n          description: "Problems and challenges mentioned"\n        opportunities:\n          description: "Opportunities and potential solutions"\n\n  # Cross-validate information\n  - id: validate_facts\n    action: validate_data\n    parameters:\n      data: "$results.extract_information"\n      rules:\n        - name: "source_diversity"\n          condition: "count(unique(sources)) >= 2"\n          severity: "warning"\n          message: "Information should be confirmed by multiple sources"\n\n        - name: "recent_information"\n          field: "date"\n          condition: "date_diff(value, today()) <= 365"\n          severity: "info"\n          message: "Information is from the last year"\n\n  # Generate comprehensive analysis\n  - id: analyze_findings\n    action: generate_content\n    parameters:\n      prompt: |\n        Analyze the following research data about {{ inputs.topic }}:\n\n        {{ results.extract_information | json }}\n\n        Provide:\n        1. Current state analysis\n        2. Key trends identification\n        3. Challenge assessment\n        4. Future outlook\n        5. Recommendations\n\n        Base your analysis on the evidence provided and note any limitations.\n\n      style: "analytical"\n      max_tokens: 2000\n\n  # Create structured data export\n  - id: export_data\n    action: transform_data\n    parameters:\n      data:\n        topic: "{{ inputs.topic }}"\n        research_date: "{{ execution.timestamp }}"\n        depth: "{{ inputs.depth }}"\n        sources_used: "{{ inputs.include_sources }}"\n        extracted_info: "$results.extract_information"\n        validation_results: "$results.validate_facts"\n        analysis: "$results.analyze_findings"\n      operations:\n        - type: "convert_format"\n          to_format: "json"\n\n  # Save structured data\n  - id: save_data\n    action: write_file\n    parameters:\n      path: "{{ outputs.data_file }}"\n      content: "$results.export_data"\n\n  # Generate final report\n  - id: create_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a comprehensive research report about {{ inputs.topic }} using:\n\n        Analysis: {{ results.analyze_findings }}\n\n        Structure the report with:\n        1. Executive Summary\n        2. Methodology\n        3. Current State Analysis\n        4. Key Findings\n        5. Trends and Developments\n        6. Challenges and Limitations\n        7. Future Outlook\n        8. Recommendations\n        9. Sources and References\n\n        Include confidence levels for major claims.\n\n      style: "professional"\n      format: "markdown"\n      max_tokens: 3000\n\n  # Save final report\n  - id: save_report\n    action: write_file\n    parameters:\n      path: "{{ outputs.comprehensive_report }}"\n      content: "$results.create_report"'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_tutorial_web_research_lines_352_375_6():
    """Test Python snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 352-375."""
    # Description: ---------------------------------
    content = 'import orchestrator as orc\n\n# Initialize\norc.init_models()\n\n# Compile pipeline\npipeline = orc.compile("multi_source_research.yaml")\n\n# Run deep research on quantum computing\nresult = pipeline.run(\n    topic="quantum computing applications",\n    depth="deep",\n    include_sources=["web", "academic", "news"]\n)\n\nprint(f"Research complete: {result}")\n\n# Run lighter research on emerging tech\nresult2 = pipeline.run(\n    topic="edge computing trends",\n    depth="medium",\n    include_sources=["web", "news"]\n)'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
        try:
            compile(content, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_tutorial_web_research_lines_388_488_7():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 388-488."""
    # Description: Create ``fact_checker.yaml``:
    import yaml
    
    content = 'name: fact-checker\ndescription: Verify claims against multiple reliable sources\n\ninputs:\n  claims:\n    type: array\n    description: "Claims to verify"\n    required: true\n\n  confidence_threshold:\n    type: float\n    description: "Minimum confidence level to accept claims"\n    default: 0.7\n    validation:\n      min: 0.0\n      max: 1.0\n\noutputs:\n  fact_check_report:\n    type: string\n    value: "fact_check_{{ execution.timestamp | strftime(\'%Y%m%d_%H%M\') }}.md"\n\nsteps:\n  # Research each claim\n  - id: research_claims\n    for_each: "{{ inputs.claims }}"\n    as: claim\n    action: search_web\n    parameters:\n      query: "{{ claim }} verification facts evidence"\n      max_results: 15\n      include_snippets: true\n\n  # Extract supporting/contradicting evidence\n  - id: analyze_evidence\n    for_each: "{{ inputs.claims }}"\n    as: claim\n    action: extract_information\n    parameters:\n      content: "$results.research_claims[loop.index0]"\n      extract:\n        supporting_evidence:\n          description: "Evidence that supports the claim"\n        contradicting_evidence:\n          description: "Evidence that contradicts the claim"\n        source_credibility:\n          description: "Assessment of source reliability"\n        expert_opinions:\n          description: "Expert statements about the claim"\n\n  # Assess credibility of each claim\n  - id: assess_claims\n    for_each: "{{ inputs.claims }}"\n    as: claim\n    action: generate_content\n    parameters:\n      prompt: |\n        Assess the veracity of this claim: "{{ claim }}"\n\n        Based on the evidence:\n        {{ results.analyze_evidence[loop.index0] | json }}\n\n        Provide:\n        1. Verdict: True/False/Partially True/Insufficient Evidence\n        2. Confidence level (0-1)\n        3. Supporting evidence summary\n        4. Contradicting evidence summary\n        5. Overall assessment\n\n        Be objective and cite specific sources.\n\n      style: "analytical"\n      format: "structured"\n\n  # Compile fact-check report\n  - id: create_fact_check_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a comprehensive fact-check report based on:\n\n        Claims assessed: {{ inputs.claims | json }}\n        Assessment results: {{ results.assess_claims | json }}\n\n        Format as a professional fact-checking article with:\n        1. Summary of findings\n        2. Individual claim assessments\n        3. Methodology used\n        4. Sources consulted\n        5. Limitations and caveats\n\n      style: "journalistic"\n      format: "markdown"\n\n  # Save report\n  - id: save_fact_check\n    action: write_file\n    parameters:\n      path: "{{ outputs.fact_check_report }}"\n      content: "$results.create_fact_check_report"'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_tutorial_web_research_lines_494_514_8():
    """Test Python snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 494-514."""
    # Description: ----------------------------
    content = 'import orchestrator as orc\n\n# Initialize\norc.init_models()\n\n# Compile fact-checker\nfact_checker = orc.compile("fact_checker.yaml")\n\n# Check various claims\nresult = fact_checker.run(\n    claims=[\n        "Electric vehicles produce zero emissions",\n        "AI will replace 50% of jobs by 2030",\n        "Quantum computers can break all current encryption",\n        "Renewable energy is now cheaper than fossil fuels"\n    ],\n    confidence_threshold=0.8\n)\n\nprint(f"Fact-check report: {result}")'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
        try:
            compile(content, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_tutorial_web_research_lines_527_805_9():
    """Test YAML snippet from docs_sphinx/tutorials/tutorial_web_research.rst lines 527-805."""
    # Description: Create ``report_generator.yaml``:
    import yaml
    
    content = 'name: automated-report-generator\ndescription: Generate professional reports from research data\n\ninputs:\n  topic:\n    type: string\n    required: true\n\n  report_type:\n    type: string\n    description: "Type of report to generate"\n    default: "standard"\n    validation:\n      enum: ["executive", "technical", "standard", "briefing"]\n\n  target_audience:\n    type: string\n    description: "Primary audience for the report"\n    default: "general"\n    validation:\n      enum: ["executives", "technical", "general", "academic"]\n\n  sections:\n    type: array\n    description: "Sections to include in report"\n    default: ["summary", "introduction", "analysis", "conclusion"]\n\noutputs:\n  report_markdown:\n    type: string\n    value: "reports/{{ inputs.topic | slugify }}_{{ inputs.report_type }}.md"\n\n  report_pdf:\n    type: string\n    value: "reports/{{ inputs.topic | slugify }}_{{ inputs.report_type }}.pdf"\n\n  report_html:\n    type: string\n    value: "reports/{{ inputs.topic | slugify }}_{{ inputs.report_type }}.html"\n\n# Report templates by type\nconfig:\n  report_templates:\n    executive:\n      style: "executive"\n      length: "concise"\n      focus: "strategic"\n      sections: ["executive_summary", "key_findings", "recommendations", "appendix"]\n\n    technical:\n      style: "technical"\n      length: "detailed"\n      focus: "implementation"\n      sections: ["introduction", "technical_analysis", "methodology", "results", "conclusion"]\n\n    standard:\n      style: "professional"\n      length: "medium"\n      focus: "comprehensive"\n      sections: ["summary", "background", "analysis", "findings", "recommendations"]\n\n    briefing:\n      style: "concise"\n      length: "short"\n      focus: "actionable"\n      sections: ["situation", "assessment", "recommendations"]\n\nsteps:\n  # Gather comprehensive research data\n  - id: research_topic\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }} comprehensive analysis research"\n      max_results: 25\n      include_snippets: true\n\n  # Get recent news for current context\n  - id: current_context\n    action: search_news\n    parameters:\n      query: "{{ inputs.topic }}"\n      max_results: 10\n      date_range: "last_week"\n\n  # Extract structured information\n  - id: extract_report_data\n    action: extract_information\n    parameters:\n      content:\n        research: "$results.research_topic"\n        news: "$results.current_context"\n      extract:\n        key_points:\n          description: "Main points and findings"\n        statistics:\n          description: "Important numbers and data"\n        trends:\n          description: "Current and emerging trends"\n        implications:\n          description: "Implications and consequences"\n        expert_views:\n          description: "Expert opinions and quotes"\n        future_outlook:\n          description: "Predictions and future scenarios"\n\n  # Generate executive summary\n  - id: create_executive_summary\n    condition: "\'summary\' in inputs.sections or \'executive_summary\' in inputs.sections"\n    action: generate_content\n    parameters:\n      prompt: |\n        Create an executive summary for {{ inputs.target_audience }} audience about {{ inputs.topic }}.\n\n        Based on: {{ results.extract_report_data.key_points | json }}\n\n        Style: {{ config.report_templates[inputs.report_type].style }}\n        Focus: {{ config.report_templates[inputs.report_type].focus }}\n\n        Include the most critical points in 200-400 words.\n\n      style: "{{ config.report_templates[inputs.report_type].style }}"\n      max_tokens: 500\n\n  # Generate introduction/background\n  - id: create_introduction\n    condition: "\'introduction\' in inputs.sections or \'background\' in inputs.sections"\n    action: generate_content\n    parameters:\n      prompt: |\n        Write an introduction/background section about {{ inputs.topic }} for {{ inputs.target_audience }}.\n\n        Context: {{ results.extract_report_data | json }}\n\n        Provide necessary background and context for understanding the topic.\n\n      style: "{{ config.report_templates[inputs.report_type].style }}"\n      max_tokens: 800\n\n  # Generate main analysis\n  - id: create_analysis\n    condition: "\'analysis\' in inputs.sections or \'technical_analysis\' in inputs.sections"\n    action: generate_content\n    parameters:\n      prompt: |\n        Create a comprehensive analysis section about {{ inputs.topic }}.\n\n        Data: {{ results.extract_report_data | json }}\n\n        Style: {{ config.report_templates[inputs.report_type].style }}\n        Audience: {{ inputs.target_audience }}\n\n        Include:\n        - Current state analysis\n        - Trend analysis\n        - Key factors and drivers\n        - Challenges and opportunities\n\n        Support points with specific data and examples.\n\n      style: "{{ config.report_templates[inputs.report_type].style }}"\n      max_tokens: 1500\n\n  # Generate findings and implications\n  - id: create_findings\n    condition: "\'findings\' in inputs.sections or \'key_findings\' in inputs.sections"\n    action: generate_content\n    parameters:\n      prompt: |\n        Summarize key findings and implications regarding {{ inputs.topic }}.\n\n        Analysis: {{ results.create_analysis }}\n        Supporting data: {{ results.extract_report_data.implications | json }}\n\n        Present clear, actionable findings with implications.\n\n      style: "{{ config.report_templates[inputs.report_type].style }}"\n      max_tokens: 1000\n\n  # Generate recommendations\n  - id: create_recommendations\n    condition: "\'recommendations\' in inputs.sections"\n    action: generate_content\n    parameters:\n      prompt: |\n        Develop actionable recommendations based on the analysis of {{ inputs.topic }}.\n\n        Findings: {{ results.create_findings }}\n        Target audience: {{ inputs.target_audience }}\n\n        Provide specific, actionable recommendations with priorities and considerations.\n\n      style: "{{ config.report_templates[inputs.report_type].style }}"\n      max_tokens: 800\n\n  # Generate conclusion\n  - id: create_conclusion\n    condition: "\'conclusion\' in inputs.sections"\n    action: generate_content\n    parameters:\n      prompt: |\n        Write a strong conclusion for the {{ inputs.topic }} report.\n\n        Key findings: {{ results.create_findings }}\n        Recommendations: {{ results.create_recommendations }}\n\n        Synthesize the main points and end with a clear call to action.\n\n      style: "{{ config.report_templates[inputs.report_type].style }}"\n      max_tokens: 400\n\n  # Assemble complete report\n  - id: assemble_report\n    action: generate_content\n    parameters:\n      prompt: |\n        Compile a complete, professional report about {{ inputs.topic }}.\n\n        Report type: {{ inputs.report_type }}\n        Target audience: {{ inputs.target_audience }}\n\n        Sections to include:\n        {% if results.create_executive_summary %}\n        Executive Summary: {{ results.create_executive_summary }}\n        {% endif %}\n\n        {% if results.create_introduction %}\n        Introduction: {{ results.create_introduction }}\n        {% endif %}\n\n        {% if results.create_analysis %}\n        Analysis: {{ results.create_analysis }}\n        {% endif %}\n\n        {% if results.create_findings %}\n        Findings: {{ results.create_findings }}\n        {% endif %}\n\n        {% if results.create_recommendations %}\n        Recommendations: {{ results.create_recommendations }}\n        {% endif %}\n\n        {% if results.create_conclusion %}\n        Conclusion: {{ results.create_conclusion }}\n        {% endif %}\n\n        Format as a professional markdown document with:\n        - Proper headings and structure\n        - Table of contents\n        - Professional formatting\n        - Source citations where appropriate\n\n      style: "professional"\n      format: "markdown"\n      max_tokens: 4000\n\n  # Save markdown version\n  - id: save_markdown\n    action: write_file\n    parameters:\n      path: "{{ outputs.report_markdown }}"\n      content: "$results.assemble_report"\n\n  # Convert to PDF\n  - id: create_pdf\n    action: "!pandoc {{ outputs.report_markdown }} -o {{ outputs.report_pdf }} --pdf-engine=xelatex"\n    error_handling:\n      continue_on_error: true\n      fallback:\n        action: write_file\n        parameters:\n          path: "{{ outputs.report_pdf }}.txt"\n          content: "PDF generation requires pandoc with xelatex"\n\n  # Convert to HTML\n  - id: create_html\n    action: "!pandoc {{ outputs.report_markdown }} -o {{ outputs.report_html }} --standalone --css=style.css"\n    error_handling:\n      continue_on_error: true'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples
