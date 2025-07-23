"""Tests for report generation and PDF compilation tools."""

import os
import tempfile
from pathlib import Path

import pytest

from orchestrator.tools.report_tools import PDFCompilerTool, ReportGeneratorTool


class TestReportGeneratorTool:
    """Test suite for ReportGeneratorTool."""

    @pytest.fixture
    def report_generator(self):
        """Create a ReportGeneratorTool instance."""
        return ReportGeneratorTool()

    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return {
            "query": "test query",
            "total_results": 3,
            "search_time": 1.5,
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "snippet": "This is the first test result",
                    "relevance": 0.95,
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "snippet": "This is the second test result",
                    "relevance": 0.85,
                },
            ],
        }

    @pytest.fixture
    def sample_extraction_results(self):
        """Sample extraction results for testing."""
        return {
            "success": True,
            "url": "https://example.com/1",
            "title": "Test Page",
            "text": "This is the extracted content from the web page. " * 50,
            "word_count": 500,
        }

    @pytest.mark.asyncio
    async def test_basic_report_generation(
        self, report_generator, sample_search_results, sample_extraction_results
    ):
        """Test basic report generation."""
        result = await report_generator.execute(
            title="Test Report",
            query="test query",
            context="test context",
            search_results=sample_search_results,
            extraction_results=sample_extraction_results,
            quality_score=0.75,
        )

        assert result["success"] is True
        assert "markdown" in result
        assert result["word_count"] > 0
        assert "Test Report" in result["markdown"]
        assert "test query" in result["markdown"]
        assert "0.75" in result["markdown"]

    @pytest.mark.asyncio
    async def test_report_with_findings_and_recommendations(
        self, report_generator, sample_search_results
    ):
        """Test report generation with findings and recommendations."""
        findings = ["Finding 1", "Finding 2", "Finding 3"]
        recommendations = ["Recommendation 1", "Recommendation 2"]

        result = await report_generator.execute(
            title="Research Report",
            query="test query",
            search_results=sample_search_results,
            extraction_results={},
            findings=findings,
            recommendations=recommendations,
        )

        assert result["success"] is True
        markdown = result["markdown"]

        # Check findings are included
        for finding in findings:
            assert finding in markdown

        # Check recommendations are included
        for rec in recommendations:
            assert rec in markdown

    @pytest.mark.asyncio
    async def test_report_without_results(self, report_generator):
        """Test report generation with no search results."""
        result = await report_generator.execute(
            title="Empty Report",
            query="no results query",
            search_results={"results": [], "total_results": 0},
            extraction_results={},
        )

        assert result["success"] is True
        assert "No search results found" in result["markdown"]

    @pytest.mark.asyncio
    async def test_report_structure(
        self, report_generator, sample_search_results, sample_extraction_results
    ):
        """Test that report has proper structure."""
        result = await report_generator.execute(
            title="Structured Report",
            query="test query",
            search_results=sample_search_results,
            extraction_results=sample_extraction_results,
        )

        markdown = result["markdown"]

        # Check for main sections
        assert "# Structured Report" in markdown
        assert "## Executive Summary" in markdown
        assert "## Search Results" in markdown
        assert "## Methodology" in markdown
        assert "## References" in markdown


class TestPDFCompilerTool:
    """Test suite for PDFCompilerTool."""

    @pytest.fixture
    def pdf_compiler(self):
        """Create a PDFCompilerTool instance."""
        return PDFCompilerTool()

    @pytest.fixture
    def sample_markdown(self):
        """Sample markdown content for testing."""
        return """# Test Report

**Generated on:** 2024-01-01
**Author:** Test Author

## Introduction

This is a test report with **bold text** and *italic text*.

### Subsection

- Item 1
- Item 2
- Item 3

## Conclusion

This concludes the test report.
"""

    @pytest.mark.asyncio
    async def test_pandoc_detection(self, pdf_compiler):
        """Test pandoc installation detection."""
        is_installed = pdf_compiler._is_pandoc_installed()
        # This test will pass regardless of whether pandoc is installed
        assert isinstance(is_installed, bool)

    @pytest.mark.asyncio
    async def test_pdf_compilation_with_markdown(self, pdf_compiler, sample_markdown):
        """Test PDF compilation with markdown content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.pdf"

            result = await pdf_compiler.execute(
                markdown_content=sample_markdown,
                output_path=str(output_path),
                title="Test Report",
                author="Test Author",
                install_if_missing=False,  # Don't auto-install in tests
            )

            if pdf_compiler._is_pandoc_installed():
                # If pandoc is installed, PDF should be generated
                assert result["success"] is True
                assert output_path.exists()
                assert result["file_size"] > 0
            else:
                # If pandoc is not installed, should fail gracefully
                assert result["success"] is False
                assert "Pandoc is not installed" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_markdown_handling(self, pdf_compiler):
        """Test handling of empty markdown content."""
        result = await pdf_compiler.execute(
            markdown_content="", output_path="test.pdf", install_if_missing=False
        )

        assert result["success"] is False
        assert "No markdown content provided" in result["error"]

    @pytest.mark.asyncio
    async def test_command_exists_check(self, pdf_compiler):
        """Test command existence checking."""
        # Test with a command that should exist
        assert pdf_compiler._command_exists("python") is True

        # Test with a command that shouldn't exist
        assert pdf_compiler._command_exists("nonexistentcommand123") is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.path.exists("/usr/bin/pandoc") and not os.path.exists("/usr/local/bin/pandoc"),
        reason="Pandoc not installed",
    )
    async def test_pdf_with_special_characters(self, pdf_compiler):
        """Test PDF generation with special characters."""
        markdown_with_special = """# Test Report with Special Characters

This report contains:
- Unicode: 你好世界 🌍
- Math: α + β = γ
- Symbols: © ® ™ € £ ¥

## Code Example

```python
def hello():
    print("Hello, World!")
```
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "special_chars.pdf"

            result = await pdf_compiler.execute(
                markdown_content=markdown_with_special,
                output_path=str(output_path),
                title="Special Characters Test",
                install_if_missing=False,
            )

            if result["success"]:
                assert output_path.exists()
                assert result["file_size"] > 0
