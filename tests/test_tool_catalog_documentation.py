#!/usr/bin/env python3
"""Test all examples from the tool catalog documentation with real execution."""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import Orchestrator, init_models


async def setup_orchestrator():
    """Set up orchestrator with real models."""
    # Initialize models
    print("Initializing models...")
    model_registry = init_models()

    # Create orchestrator
    orchestrator = Orchestrator(model_registry=model_registry)
    return orchestrator


async def test_filesystem_tool():
    """Test FileSystemTool examples from documentation."""
    print("\n=== Testing FileSystemTool ===")

    # Create temporary test content
    (
        """# Test Report
This is a test report generated for documentation validation.
Generated at: """
        + datetime.now().isoformat()
    )

    # Test pipeline from documentation
    import tempfile

    report_path = os.path.join(tempfile.gettempdir(), "test_report.md")

    pipeline_yaml = (
        f"""
name: filesystem-tool-test
description: Test FileSystemTool from documentation

steps:
  - id: save_report
    tool: filesystem
    action: write
    parameters:
      path: "{report_path}"
      content: "# Test Report\\nThis is a test report generated for documentation validation.\\nGenerated at: """
        + datetime.now().isoformat()
        + """"
      mode: "w"
      
  - id: verify_write
    tool: filesystem
    action: exists
    parameters:
      path: "{report_path}"
      
  - id: read_report
    tool: filesystem
    action: read
    parameters:
      path: "{report_path}"
      
  - id: list_tmp
    tool: filesystem
    action: list
    parameters:
      path: "/tmp"
      pattern: "test_*.md"
"""
    )

    orchestrator = await setup_orchestrator()

    try:
        result = await orchestrator.execute_yaml(pipeline_yaml)

        # Verify results
        assert result.get("verify_write", {}).get("exists") is True, "File should exist after write"
        assert "Test Report" in result.get("read_report", {}).get(
            "content", ""
        ), "Content should match"

        print("✅ FileSystemTool test passed")
        return True

    except Exception as e:
        print(f"❌ FileSystemTool test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(report_path):
            os.unlink(report_path)


async def test_terminal_tool():
    """Test TerminalTool examples from documentation."""
    print("\n=== Testing TerminalTool ===")

    # Create test Python script
    test_script = """
import json
import sys

# Simple analysis script
data = {"result": "Analysis complete", "items": 5}
print(json.dumps(data))
"""

    import tempfile

    script_path = os.path.join(tempfile.gettempdir(), "test_analyze.py")
    with open(script_path, "w") as f:
        f.write(test_script)

    # Create test data file
    data_path = os.path.join(tempfile.gettempdir(), "test_data.csv")
    with open(data_path, "w") as f:
        f.write("name,value\nitem1,10\nitem2,20\n")

    pipeline_yaml = f"""
name: terminal-tool-test
description: Test TerminalTool from documentation

steps:
  - id: run_analysis
    tool: terminal
    action: execute
    parameters:
      command: "python {script_path} --input {data_path}"
      cwd: "{tempfile.gettempdir()}"
      timeout: 30
      
  - id: check_python_version
    tool: terminal
    action: execute
    parameters:
      command: "python --version"
      timeout: 10
"""

    orchestrator = await setup_orchestrator()

    try:
        result = await orchestrator.execute_yaml(pipeline_yaml)

        # Verify results
        analysis_output = result.get("run_analysis", {}).get("stdout", "")
        assert "Analysis complete" in analysis_output, "Analysis should complete"

        version_output = result.get("check_python_version", {}).get("stdout", "")
        assert "Python" in version_output, "Should get Python version"

        print("✅ TerminalTool test passed")
        return True

    except Exception as e:
        print(f"❌ TerminalTool test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        for path in [script_path, data_path]:
            if os.path.exists(path):
                os.unlink(path)


async def test_web_search_tool():
    """Test WebSearchTool examples from documentation."""
    print("\n=== Testing WebSearchTool ===")

    pipeline_yaml = """
name: web-search-tool-test
description: Test WebSearchTool from documentation

steps:
  - id: research_topic
    tool: web-search
    action: search
    parameters:
      query: "machine learning best practices 2024"
      max_results: 5
      
  - id: verify_results
    tool: report-generator
    action: generate
    parameters:
      title: "Search Results"
      format: "markdown"
      content: |
        # Web Search Test Results
        
        Found {{ research_topic.results | length }} results for the query.
        
        {% if research_topic.results %}
        First result title: {{ research_topic.results[0].title }}
        {% endif %}
"""

    orchestrator = await setup_orchestrator()

    try:
        result = await orchestrator.execute_yaml(pipeline_yaml)

        # Verify results
        search_results = result.get("research_topic", {}).get("results", [])
        assert len(search_results) > 0, "Should get search results"
        assert all("title" in r for r in search_results), "Results should have titles"

        print("✅ WebSearchTool test passed")
        return True

    except Exception as e:
        print(f"❌ WebSearchTool test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_data_processing_tool():
    """Test DataProcessingTool examples from documentation."""
    print("\n=== Testing DataProcessingTool ===")

    # Create test CSV data
    csv_data = """name,age,score
Alice,25,85
Bob,17,90
Charlie,30,75
David,16,95
Eve,22,88"""

    pipeline_yaml = (
        """
name: data-processing-tool-test
description: Test DataProcessingTool from documentation

inputs:
  raw_data: |
    """
        + csv_data
        + """

steps:
  - id: process_csv
    tool: data-processing
    action: transform
    parameters:
      input_data: "{{ raw_data }}"
      input_format: "csv"
      output_format: "json"
      operations:
        - type: filter
          condition: "age > 18"
        - type: sort
          by: "score"
          ascending: false
          
  - id: verify_processing
    tool: report-generator
    action: generate
    parameters:
      title: "Data Processing Results"
      format: "markdown"
      content: |
        # Processing Test Results
        
        Processed {{ process_csv.result | length }} records.
        
        {% if process_csv.result %}
        Top scorer: {{ process_csv.result[0].name }} with score {{ process_csv.result[0].score }}
        {% endif %}
"""
    )

    orchestrator = await setup_orchestrator()

    try:
        result = await orchestrator.execute_yaml(pipeline_yaml, {"raw_data": csv_data})

        # Verify results
        processed_data = result.get("process_csv", {}).get("result", [])
        assert len(processed_data) > 0, "Should have processed data"

        # Check filtering (age > 18)
        assert all(record["age"] > 18 for record in processed_data), "All ages should be > 18"

        # Check sorting (by score descending)
        scores = [record["score"] for record in processed_data]
        assert scores == sorted(scores, reverse=True), "Should be sorted by score descending"

        print("✅ DataProcessingTool test passed")
        return True

    except Exception as e:
        print(f"❌ DataProcessingTool test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_report_generator_tool():
    """Test ReportGeneratorTool examples from documentation."""
    print("\n=== Testing ReportGeneratorTool ===")

    import tempfile

    report_path = os.path.join(tempfile.gettempdir(), "test_analysis_report.md")

    pipeline_yaml = f"""
name: report-generator-tool-test
description: Test ReportGeneratorTool from documentation

inputs:
  title: "Analysis Report"
  summary: "This report summarizes the key findings from our analysis."
  analysis_results: |
    - Finding 1: Data shows positive trend
    - Finding 2: Performance improved by 25%
    - Finding 3: User satisfaction increased
  current_date: "2024-01-15"

steps:
  - id: create_report
    tool: report-generator
    action: generate
    parameters:
      title: "{{ title }}"
      format: "markdown"
      template: |
        # {{ title }}
        
        ## Summary
        {{ summary }}
        
        ## Data Analysis
        {{ analysis_results }}
      metadata:
        author: "AI Assistant"
        date: "{{ current_date }}"
        
  - id: save_report
    tool: filesystem
    action: write
    parameters:
      path: "{report_path}"
      content: "{{ create_report.content }}"
"""

    orchestrator = await setup_orchestrator()

    try:
        inputs = {
            "title": "Analysis Report",
            "summary": "This report summarizes the key findings from our analysis.",
            "analysis_results": "- Finding 1: Data shows positive trend\n- Finding 2: Performance improved by 25%\n- Finding 3: User satisfaction increased",
            "current_date": datetime.now().strftime("%Y-%m-%d"),
        }

        result = await orchestrator.execute_yaml(pipeline_yaml, inputs)

        # Verify results
        report_content = result.get("create_report", {}).get("content", "")
        assert inputs["title"] in report_content, "Report should contain title"
        assert inputs["summary"] in report_content, "Report should contain summary"
        assert "Finding 1" in report_content, "Report should contain findings"

        print("✅ ReportGeneratorTool test passed")
        return True

    except Exception as e:
        print(f"❌ ReportGeneratorTool test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(report_path):
            os.unlink(report_path)


async def test_llm_generate_tool():
    """Test LLMGenerateTool examples from documentation."""
    print("\n=== Testing LLMGenerateTool ===")

    pipeline_yaml = """
name: llm-generate-tool-test
description: Test LLMGenerateTool from documentation

inputs:
  article_content: |
    Artificial intelligence continues to transform industries worldwide. 
    Recent advances in machine learning have enabled new applications in 
    healthcare, finance, and transportation. However, challenges remain 
    in ensuring AI systems are ethical, transparent, and beneficial to all.

steps:
  - id: write_summary
    tool: llm-generate
    action: generate
    parameters:
      prompt: |
        Summarize this article in 2-3 sentences:
        {{ article_content }}
      temperature: 0.3
      max_tokens: 200
      
  - id: verify_summary
    tool: report-generator
    action: generate
    parameters:
      title: "Summary Test"
      format: "markdown"
      content: |
        # LLM Generation Test
        
        Original length: {{ article_content | length }} characters
        Summary length: {{ write_summary.result | length }} characters
        
        Summary:
        {{ write_summary.result }}
"""

    orchestrator = await setup_orchestrator()

    try:
        inputs = {
            "article_content": """Artificial intelligence continues to transform industries worldwide. 
Recent advances in machine learning have enabled new applications in 
healthcare, finance, and transportation. However, challenges remain 
in ensuring AI systems are ethical, transparent, and beneficial to all."""
        }

        result = await orchestrator.execute_yaml(pipeline_yaml, inputs)

        # Verify results
        summary = result.get("write_summary", {}).get("result", "")
        assert len(summary) > 0, "Should generate a summary"
        assert len(summary) < len(inputs["article_content"]), "Summary should be shorter"

        print("✅ LLMGenerateTool test passed")
        return True

    except Exception as e:
        print(f"❌ LLMGenerateTool test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_llm_analyze_tool():
    """Test LLMAnalyzeTool examples from documentation."""
    print("\n=== Testing LLMAnalyzeTool ===")

    pipeline_yaml = """
name: llm-analyze-tool-test
description: Test LLMAnalyzeTool from documentation

inputs:
  customer_feedback: |
    The product works great and I'm very happy with my purchase. 
    The customer service was excellent and shipping was fast. 
    I would definitely recommend this to others!

steps:
  - id: analyze_sentiment
    tool: llm-analyze
    action: analyze
    parameters:
      content: "{{ customer_feedback }}"
      analysis_type: "sentiment"
      schema:
        type: object
        properties:
          sentiment:
            type: string
            enum: ["positive", "negative", "neutral"]
          confidence:
            type: number
          key_points:
            type: array
            items:
              type: string
              
  - id: verify_analysis
    tool: report-generator
    action: generate
    parameters:
      title: "Sentiment Analysis"
      format: "markdown"
      content: |
        # Analysis Results
        
        Sentiment: {{ analyze_sentiment.result.sentiment }}
        Confidence: {{ analyze_sentiment.result.confidence }}
        
        Key Points:
        {% for point in analyze_sentiment.result.key_points %}
        - {{ point }}
        {% endfor %}
"""

    orchestrator = await setup_orchestrator()

    try:
        inputs = {
            "customer_feedback": """The product works great and I'm very happy with my purchase. 
The customer service was excellent and shipping was fast. 
I would definitely recommend this to others!"""
        }

        result = await orchestrator.execute_yaml(pipeline_yaml, inputs)

        # Verify results
        analysis = result.get("analyze_sentiment", {}).get("result", {})
        assert "sentiment" in analysis, "Should have sentiment field"
        assert analysis["sentiment"] in [
            "positive",
            "negative",
            "neutral",
        ], "Sentiment should be valid"
        assert "confidence" in analysis, "Should have confidence score"
        assert isinstance(analysis.get("key_points"), list), "Should have key points list"

        print("✅ LLMAnalyzeTool test passed")
        return True

    except Exception as e:
        print(f"❌ LLMAnalyzeTool test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_validation_tool():
    """Test ValidationTool examples from documentation."""
    print("\n=== Testing ValidationTool ===")

    pipeline_yaml = """
name: validation-tool-test
description: Test ValidationTool from documentation

inputs:
  user_input:
    email: "test@example.com"
    age: 25

steps:
  - id: validate_input
    tool: validation
    action: validate
    parameters:
      data: "{{ user_input }}"
      schema:
        type: object
        properties:
          email:
            type: string
            format: email
          age:
            type: number
            minimum: 18
        required: ["email", "age"]
      mode: "STRICT"
      
  - id: test_invalid
    tool: validation
    action: validate
    parameters:
      data:
        email: "not-an-email"
        age: 15
      schema:
        type: object
        properties:
          email:
            type: string
            format: email
          age:
            type: number
            minimum: 18
        required: ["email", "age"]
      mode: "REPORT_ONLY"
"""

    orchestrator = await setup_orchestrator()

    try:
        inputs = {"user_input": {"email": "test@example.com", "age": 25}}

        result = await orchestrator.execute_yaml(pipeline_yaml, inputs)

        # Verify results
        valid_result = result.get("validate_input", {})
        assert valid_result.get("is_valid") is True, "Valid data should pass"

        invalid_result = result.get("test_invalid", {})
        assert invalid_result.get("is_valid") is False, "Invalid data should fail"
        assert len(invalid_result.get("errors", [])) > 0, "Should report validation errors"

        print("✅ ValidationTool test passed")
        return True

    except Exception as e:
        print(f"❌ ValidationTool test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_sub_pipeline_tool():
    """Test SubPipelineTool examples from documentation."""
    print("\n=== Testing SubPipelineTool ===")

    # Create a simple sub-pipeline file
    sub_pipeline_yaml = """
name: data_analysis_pipeline
description: Simple analysis sub-pipeline

steps:
  - id: analyze
    tool: llm-generate
    action: generate
    parameters:
      prompt: |
        Analyze this data and provide a brief summary:
        {{ data }}
      max_tokens: 100
      
outputs:
  analysis_result: "{{ analyze.result }}"
"""

    import tempfile

    sub_pipeline_path = os.path.join(tempfile.gettempdir(), "test_sub_pipeline.yaml")
    with open(sub_pipeline_path, "w") as f:
        f.write(sub_pipeline_yaml)

    main_pipeline_yaml = f"""
name: sub-pipeline-tool-test
description: Test SubPipelineTool from documentation

inputs:
  processed_data: "Sample data: [10, 20, 30, 40, 50]"
  analysis_config:
    type: "statistical"

steps:
  - id: run_analysis
    tool: sub-pipeline
    action: execute
    parameters:
      pipeline_id: "{sub_pipeline_path}"
      inputs:
        data: "{{ processed_data }}"
        config: "{{ analysis_config }}"
      inherit_context: true
      
  - id: show_results
    tool: report-generator
    action: generate
    parameters:
      title: "Sub-Pipeline Results"
      format: "markdown"
      content: |
        # Analysis Complete
        
        Result: {{ run_analysis.outputs.analysis_result }}
"""

    orchestrator = await setup_orchestrator()

    try:
        inputs = {
            "processed_data": "Sample data: [10, 20, 30, 40, 50]",
            "analysis_config": {"type": "statistical"},
        }

        result = await orchestrator.execute_yaml(main_pipeline_yaml, inputs)

        # Verify results
        sub_result = result.get("run_analysis", {})
        assert "outputs" in sub_result, "Should have sub-pipeline outputs"
        assert "analysis_result" in sub_result.get("outputs", {}), "Should have analysis result"

        print("✅ SubPipelineTool test passed")
        return True

    except Exception as e:
        print(f"❌ SubPipelineTool test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists(sub_pipeline_path):
            os.unlink(sub_pipeline_path)


async def test_combined_pipeline():
    """Test the combined research pipeline example from documentation."""
    print("\n=== Testing Combined Research Pipeline ===")

    pipeline_yaml = """
name: research-pipeline-test
description: Test combined tool workflow

inputs:
  research_topic: "artificial intelligence ethics"

steps:
  # Search for information
  - id: search
    tool: web-search
    action: search
    parameters:
      query: "{{ research_topic }}"
      max_results: 3
      
  # Analyze results
  - id: analyze
    tool: llm-analyze
    action: analyze
    parameters:
      content: |
        Search results for "{{ research_topic }}":
        {% for result in search.results %}
        - {{ result.title }}: {{ result.snippet }}
        {% endfor %}
      analysis_type: "key_points"
      schema:
        type: object
        properties:
          main_themes:
            type: array
            items:
              type: string
          summary:
            type: string
      
  # Generate report
  - id: report
    tool: report-generator
    action: generate
    parameters:
      title: "Research: {{ research_topic }}"
      format: "markdown"
      content: |
        # {{ title }}
        
        ## Summary
        {{ analyze.result.summary }}
        
        ## Main Themes
        {% for theme in analyze.result.main_themes %}
        - {{ theme }}
        {% endfor %}
        
        ## Sources
        {% for result in search.results %}
        - [{{ result.title }}]({{ result.url }})
        {% endfor %}
"""

    orchestrator = await setup_orchestrator()

    try:
        inputs = {"research_topic": "artificial intelligence ethics"}

        result = await orchestrator.execute_yaml(pipeline_yaml, inputs)

        # Verify results
        assert "search" in result, "Should have search results"
        assert "analyze" in result, "Should have analysis"
        assert "report" in result, "Should have report"

        report_content = result.get("report", {}).get("content", "")
        assert inputs["research_topic"] in report_content, "Report should contain topic"

        print("✅ Combined Pipeline test passed")
        return True

    except Exception as e:
        print(f"❌ Combined Pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all tool catalog documentation tests."""
    print("🧪 TOOL CATALOG DOCUMENTATION TESTS")
    print("=" * 70)
    print("Testing all examples from docs/reference/tool_catalog.md")
    print("=" * 70)

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  WARNING: No API keys found!")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("Some tests may fail without API access.")
        print()

    # Define all tests
    tests = [
        ("FileSystemTool", test_filesystem_tool),
        ("TerminalTool", test_terminal_tool),
        ("WebSearchTool", test_web_search_tool),
        ("DataProcessingTool", test_data_processing_tool),
        ("ReportGeneratorTool", test_report_generator_tool),
        ("LLMGenerateTool", test_llm_generate_tool),
        ("LLMAnalyzeTool", test_llm_analyze_tool),
        ("ValidationTool", test_validation_tool),
        ("SubPipelineTool", test_sub_pipeline_tool),
        ("Combined Research Pipeline", test_combined_pipeline),
    ]

    # Run tests
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"Running: {test_name}")
        print("=" * 70)

        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {test_name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tool catalog examples are working correctly!")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
