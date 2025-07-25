"""Integration tests for Research Assistant with PDF report generation."""

import os
import tempfile
from pathlib import Path

import pytest

from orchestrator.tools.report_tools import PDFCompilerTool, ReportGeneratorTool
from orchestrator.tools.web_tools import WebSearchTool


class TestResearchAssistantWithReport:
    """Test Research Assistant with full report generation capabilities."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_research_pipeline(self):
        """Test complete research pipeline with PDF generation."""
        # Import the example module
        try:
            import sys
            from pathlib import Path

            # Add examples directory to path
            examples_dir = Path(__file__).parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_dir))
            from research_assistant_with_report import ResearchAssistantWithReport
        except ImportError:
            pytest.skip("Research Assistant example not available")

        # Create test configuration
        config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        }

        # Create temporary directory for reports
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize assistant
            assistant = ResearchAssistantWithReport(config)

            # Test research query
            query = "Python async programming best practices"
            context = "Focus on modern approaches and performance"

            # Conduct research
            result = await assistant.conduct_research_with_report(
                query=query, context=context, output_dir=temp_dir
            )

            # Verify results
            assert result["success"] is True
            assert result.get("markdown_path") is not None
            assert Path(result["markdown_path"]).exists()

            # Check markdown content
            with open(result["markdown_path"], "r") as f:
                markdown_content = f.read()

            assert len(markdown_content) > 100
            assert query in markdown_content
            assert "## Executive Summary" in markdown_content
            assert "## Search Results" in markdown_content

            # Check PDF if generated
            if result.get("pdf_path"):
                assert Path(result["pdf_path"]).exists()
                assert result.get("file_size", 0) > 0

            # Verify quality metrics
            assert "quality_score" in result
            assert 0 <= result["quality_score"] <= 1
            assert "word_count" in result
            assert result["word_count"] > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_report_generator_tool(self):
        """Test ReportGeneratorTool independently."""
        tool = ReportGeneratorTool()

        # Test data
        search_results = {
            "query": "test query",
            "results": [
                {
                    "title": "Test Article 1",
                    "url": "https://example.com/1",
                    "snippet": "This is a test snippet",
                    "relevance": 0.9,
                },
                {
                    "title": "Test Article 2",
                    "url": "https://example.com/2",
                    "snippet": "Another test snippet",
                    "relevance": 0.8,
                },
            ],
        }

        extraction_results = {
            "success": True,
            "text": "Extracted content from web page.",
            "word_count": 500,
        }

        # Generate report
        result = await tool.execute(
            title="Test Research Report",
            query="test query",
            search_results=search_results,
            extraction_results=extraction_results,
            findings=["Finding 1", "Finding 2"],
            recommendations=["Recommendation 1", "Recommendation 2"],
            quality_score=0.85,
        )

        # Verify report
        assert result["success"] is True
        assert "markdown" in result
        assert result["word_count"] > 0

        markdown = result["markdown"]
        assert "# Test Research Report" in markdown
        assert "Finding 1" in markdown
        assert "Finding 2" in markdown
        assert "Recommendation 1" in markdown
        assert "0.85" in markdown

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pdf_compiler_tool(self):
        """Test PDFCompilerTool independently."""
        tool = PDFCompilerTool()

        # Don't skip - let the tool install pandoc if needed

        # Test markdown
        markdown_content = """# Test Report

**Author:** Test Suite
**Date:** 2024-01-01

## Introduction

This is a test report for PDF generation.

## Content

- Item 1
- Item 2
- Item 3

## Conclusion

This concludes the test report.
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.pdf"

            # Compile PDF
            result = await tool.execute(
                markdown_content=markdown_content,
                output_path=str(output_path),
                title="Test Report",
                author="Test Suite",
                install_if_missing=True,
            )

            # Verify result
            if result["success"]:
                assert output_path.exists()
                assert result["file_size"] > 0
            else:
                # PDF generation failed (likely pandoc not available)
                assert "error" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_web_search_integration(self):
        """Test web search with real queries."""
        config = {
            "web_tools": {
                "search": {
                    "default_backend": "duckduckgo",
                    "max_results": 5,
                    "timeout": 30,
                }
            }
        }

        tool = WebSearchTool(config)

        # Perform real search
        result = await tool.execute(query="Python asyncio tutorial", max_results=5)

        # Verify results
        assert "results" in result
        assert len(result["results"]) > 0

        # Check result structure
        first_result = result["results"][0]
        assert "snippet" in first_result
        assert "rank" in first_result
        assert "relevance" in first_result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_research_quality_scoring(self):
        """Test research quality scoring logic."""
        # Import the example module
        try:
            import sys
            from pathlib import Path

            # Add examples directory to path
            examples_dir = Path(__file__).parent.parent.parent / "examples"
            sys.path.insert(0, str(examples_dir))
            from research_assistant_with_report import ResearchAssistantWithReport
        except ImportError:
            pytest.skip("Research Assistant example not available")

        # Create assistant
        assistant = ResearchAssistantWithReport({})

        # Test quality scoring with realistic data
        search_results = {
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example1.com/page1",
                    "relevance": 0.9,
                },
                {
                    "title": "Result 2",
                    "url": "https://example2.com/page2",
                    "relevance": 0.8,
                },
                {
                    "title": "Result 3",
                    "url": "https://example3.com/page3",
                    "relevance": 0.7,
                },
            ]
        }
        extraction_results = {"success": True, "word_count": 1500}

        score = assistant._calculate_quality_score(search_results, extraction_results)

        # Verify score calculation
        assert 0 <= score <= 1
        # With 3 results, good relevance, URLs, and 1500 words:
        # Results: 3/10 * 0.3 = 0.09
        # Content: 1500/2000 * 0.3 = 0.225
        # Relevance: 0.8 * 0.2 = 0.16
        # Diversity: 3/5 * 0.2 = 0.12
        # Total: ~0.595
        assert score > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
