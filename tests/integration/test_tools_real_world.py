"""Real-world integration tests for all orchestrator tools.

These tests validate that tools work correctly in real scenarios without mocks.
Each test:
1. Uses real resources (files, URLs, commands)
2. Validates actual output
3. Checks for edge cases and error handling
4. Ensures tools integrate properly with the orchestrator
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from orchestrator import init_models, Orchestrator
from orchestrator.control_systems.hybrid_control_system import HybridControlSystem
from orchestrator.tools.web_tools import HeadlessBrowserTool, WebSearchTool
from orchestrator.tools.system_tools import TerminalTool, FileSystemTool
from orchestrator.tools.data_tools import DataProcessingTool
from orchestrator.tools.report_tools import ReportGeneratorTool, PDFCompilerTool


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def model_registry():
    """Initialize real models for testing."""
    try:
        return init_models()
    except Exception as e:
        pytest.skip(f"Failed to initialize models: {e}")


@pytest.fixture(scope="module")
def orchestrator(model_registry):
    """Create orchestrator with real control system."""
    control_system = HybridControlSystem(model_registry)
    return Orchestrator(
        model_registry=model_registry,
        control_system=control_system
    )


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for file operations."""
    workspace = Path(tempfile.mkdtemp(prefix="orchestrator_test_"))
    yield workspace
    # Cleanup
    shutil.rmtree(workspace, ignore_errors=True)


class TestHeadlessBrowserTool:
    """Real-world tests for HeadlessBrowserTool."""
    
    @pytest.fixture
    def browser_tool(self):
        """Create browser tool instance."""
        return HeadlessBrowserTool()
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_scrape_real_website(self, browser_tool):
        """Test scraping a real website."""
        # Use example.com - it's stable and simple
        result = await browser_tool.execute(
            url="https://example.com",
            action="scrape"
        )
        
        # Check if there was an error
        if "error" in result:
            pytest.fail(f"Scraping failed: {result['error']}")
        
        # Should have scraped content
        assert "url" in result
        assert result["url"] == "https://example.com"
        assert "title" in result
        assert "Example Domain" in result["title"]
        
        # Check for text content (if extracted)
        if "text" in result:
            assert "Example Domain" in result["text"]
            assert "More information" in result["text"]
        
        # Check for other metadata
        if "status_code" in result:
            assert result["status_code"] == 200
    
    @pytest.mark.asyncio  
    @pytest.mark.timeout(30)
    async def test_verify_real_website(self, browser_tool):
        """Test verifying a real website."""
        result = await browser_tool.execute(
            url="https://example.com",
            action="verify"
        )
        
        # Check if there was an error
        if "error" in result:
            pytest.fail(f"Verification failed: {result['error']}")
        
        assert "url" in result
        assert "accessible" in result
        assert result["accessible"] is True
    
    @pytest.mark.asyncio
    async def test_invalid_url_handling(self, browser_tool):
        """Test handling of invalid URLs."""
        result = await browser_tool.execute(
            url="https://this-domain-definitely-does-not-exist-12345.com",
            action="scrape"
        )
        
        # Should have an error for invalid URL
        assert "error" in result
        assert "url" in result
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_scrape_with_javascript(self, browser_tool):
        """Test scraping with JavaScript support."""
        result = await browser_tool.execute(
            url="https://example.com",
            action="scrape_js"
        )
        
        # scrape_js might fail if playwright is not installed
        if "error" in result:
            # Check if it's a playwright error
            if "playwright" in result["error"].lower():
                pytest.skip("Playwright not installed")
            else:
                pytest.fail(f"JS scraping failed: {result['error']}")
        
        # Should have scraped content
        assert "url" in result
        assert "title" in result


class TestWebSearchTool:
    """Real-world tests for WebSearchTool."""
    
    @pytest.fixture
    def search_tool(self):
        """Create search tool instance."""
        return WebSearchTool()
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_real_web_search(self, search_tool):
        """Test real web search functionality."""
        result = await search_tool.execute(
            query="Python programming language official documentation",
            max_results=5
        )
        
        # WebSearchTool returns different format - no "success" key
        assert "results" in result
        assert "query" in result
        assert result["query"] == "Python programming language official documentation"
        
        # Check if we got results (may be empty if search fails)
        if result.get("error"):
            # Search failed, but should still have proper structure
            assert result["total_results"] == 0
        else:
            # Should have some results
            assert len(result["results"]) > 0
            assert len(result["results"]) <= 5
            
            # Check result structure
            for item in result["results"]:
                assert "title" in item
                assert "url" in item
                assert "snippet" in item
                if item["url"]:  # URL might be empty in some cases
                    assert item["url"].startswith("http")
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, search_tool):
        """Test handling of empty search query."""
        result = await search_tool.execute(
            query="",
            max_results=5
        )
        
        # Should have error for empty query
        assert "error" in result
        assert result["total_results"] == 0


class TestTerminalTool:
    """Real-world tests for TerminalTool."""
    
    @pytest.fixture
    def terminal_tool(self):
        """Create terminal tool instance."""
        return TerminalTool()
    
    @pytest.mark.asyncio
    async def test_simple_command_execution(self, terminal_tool):
        """Test executing simple shell commands."""
        # Test echo command
        result = await terminal_tool.execute(
            command="echo 'Hello from terminal tool'"
        )
        
        assert result["success"] is True
        assert result["stdout"].strip() == "Hello from terminal tool"
        assert result["return_code"] == 0
    
    @pytest.mark.asyncio
    async def test_command_with_error(self, terminal_tool):
        """Test handling command that returns error."""
        result = await terminal_tool.execute(
            command="ls /nonexistent/directory/path"
        )
        
        assert result["success"] is False
        assert result["return_code"] != 0
        assert result["stderr"] != ""  # Should have error message
    
    @pytest.mark.asyncio
    async def test_command_timeout(self, terminal_tool):
        """Test command timeout handling."""
        result = await terminal_tool.execute(
            command="sleep 10",
            timeout=1
        )
        
        assert result["success"] is False
        assert "timed out" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_working_directory(self, terminal_tool, temp_workspace):
        """Test command execution in specific directory."""
        # Create a test file in temp workspace
        test_file = temp_workspace / "test.txt"
        test_file.write_text("test content")
        
        result = await terminal_tool.execute(
            command="ls test.txt",
            working_dir=str(temp_workspace)
        )
        
        assert result["success"] is True
        assert "test.txt" in result["stdout"]


class TestFileSystemTool:
    """Real-world tests for FileSystemTool."""
    
    @pytest.fixture
    def fs_tool(self):
        """Create filesystem tool instance."""
        return FileSystemTool()
    
    @pytest.mark.asyncio
    async def test_file_operations(self, fs_tool, temp_workspace):
        """Test file read/write operations."""
        test_file = temp_workspace / "test.txt"
        test_content = "This is a test file\nWith multiple lines\n"
        
        # Test write
        write_result = await fs_tool.execute(
            action="write",
            path=str(test_file),
            content=test_content
        )
        
        assert write_result["success"] is True
        assert test_file.exists()
        
        # Test read
        read_result = await fs_tool.execute(
            action="read",
            path=str(test_file)
        )
        
        assert read_result["success"] is True
        assert read_result["content"] == test_content
        
        # Test delete
        delete_result = await fs_tool.execute(
            action="delete",
            path=str(test_file)
        )
        
        assert delete_result["success"] is True
        assert not test_file.exists()
    
    @pytest.mark.asyncio
    async def test_directory_operations(self, fs_tool, temp_workspace):
        """Test directory operations."""
        test_dir = temp_workspace / "test_dir"
        
        # Create directory using write action (it creates parent dirs)
        dummy_file = test_dir / "dummy.txt"
        create_result = await fs_tool.execute(
            action="write",
            path=str(dummy_file),
            content="test"
        )
        
        assert create_result["success"] is True
        assert test_dir.exists()
        assert test_dir.is_dir()
        
        # Test list directory
        # Create some files
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        
        list_result = await fs_tool.execute(
            action="list",
            path=str(test_dir)
        )
        
        assert list_result["success"] is True
        assert len(list_result["items"]) == 3  # dummy.txt, file1.txt, file2.txt
        file_names = [item["name"] for item in list_result["items"]]
        assert "file1.txt" in file_names
        assert "file2.txt" in file_names
        assert "dummy.txt" in file_names
    
    @pytest.mark.asyncio
    async def test_file_not_found(self, fs_tool):
        """Test handling of non-existent files."""
        result = await fs_tool.execute(
            action="read",
            path="/nonexistent/file/path.txt"
        )
        
        assert result["success"] is False
        assert "error" in result


class TestDataProcessingTool:
    """Real-world tests for DataProcessingTool."""
    
    @pytest.fixture
    def data_tool(self):
        """Create data processing tool instance."""
        return DataProcessingTool()
    
    @pytest.mark.asyncio
    async def test_json_processing(self, data_tool, temp_workspace):
        """Test JSON data processing."""
        test_data = {
            "users": [
                {"name": "Alice", "age": 30, "city": "New York"},
                {"name": "Bob", "age": 25, "city": "London"},
                {"name": "Charlie", "age": 35, "city": "Tokyo"}
            ]
        }
        
        json_file = temp_workspace / "test_data.json"
        json_file.write_text(json.dumps(test_data, indent=2))
        
        # Test filtering - the tool uses simple equality matching
        result = await data_tool.execute(
            action="filter",
            data=test_data["users"],  # Pass the users list directly
            operation={
                "criteria": {"city": "New York"}  # Simple equality filter
            }
        )
        
        assert result["success"] is True
        assert "result" in result
        assert result["original_count"] == 3
        assert result["filtered_count"] == 1
        assert len(result["result"]) == 1
        assert result["result"][0]["name"] == "Alice"
    
    @pytest.mark.asyncio
    async def test_csv_processing(self, data_tool, temp_workspace):
        """Test CSV data processing."""
        csv_content = """name,age,city
Alice,30,New York
Bob,25,London
Charlie,35,Tokyo
"""
        
        csv_file = temp_workspace / "test_data.csv"
        csv_file.write_text(csv_content)
        
        # Test format conversion
        result = await data_tool.execute(
            action="convert",
            data=str(csv_file),
            format="json"  # Convert CSV to JSON
        )
        
        assert result["success"] is True
        assert "result" in result
        assert result["target_format"] == "json"
        
        # Parse the JSON result
        converted_data = json.loads(result["result"])
        assert len(converted_data) == 3
        assert converted_data[0]["name"] == "Alice"
        assert converted_data[1]["age"] == "25"  # Note: CSV values are strings


# ValidationTool tests commented out - tool is not fully implemented
# class TestValidationTool:
#     """Real-world tests for ValidationTool."""
#     
#     @pytest.fixture
#     def validation_tool(self):
#         """Create validation tool instance."""
#         return ValidationTool()
#     
#     @pytest.mark.asyncio
#     async def test_schema_validation(self, validation_tool):
#         """Test JSON schema validation."""
#         test_data = {
#             "name": "Test User",
#             "email": "test@example.com",
#             "age": 25
#         }
#         
#         schema = {
#             "type": "object",
#             "properties": {
#                 "name": {"type": "string"},
#                 "email": {"type": "string", "format": "email"},
#                 "age": {"type": "integer", "minimum": 0}
#             },
#             "required": ["name", "email"]
#         }
#         
#         result = await validation_tool.execute(
#             data=test_data,
#             schema=schema
#         )
#         
#         assert result["success"] is True
#         assert "valid" in result
#         assert result["valid"] is True
#         assert len(result["errors"]) == 0
#     
#     @pytest.mark.asyncio
#     async def test_invalid_data_validation(self, validation_tool):
#         """Test validation of invalid data."""
#         test_data = {
#             "name": "Test User",
#             "email": "invalid-email",
#             "age": -5
#         }
#         
#         schema = {
#             "type": "object",
#             "properties": {
#                 "name": {"type": "string"},
#                 "email": {"type": "string", "format": "email"},
#                 "age": {"type": "integer", "minimum": 0}
#             }
#         }
#         
#         result = await validation_tool.execute(
#             data=test_data,
#             schema=schema
#         )
#         
#         assert result["success"] is True
#         assert "valid" in result
#         assert result["valid"] is False
#         assert len(result["errors"]) > 0


class TestReportGeneratorTool:
    """Real-world tests for ReportGeneratorTool."""
    
    @pytest.fixture
    def report_tool(self):
        """Create report generator tool instance."""
        return ReportGeneratorTool()
    
    @pytest.mark.asyncio
    async def test_markdown_report_generation(self, report_tool, temp_workspace):
        """Test generating a real markdown report."""
        search_results = {
            "query": "artificial intelligence applications",
            "results": [
                {
                    "title": "AI in Healthcare",
                    "url": "https://example.com/ai-health",
                    "snippet": "AI is revolutionizing healthcare..."
                },
                {
                    "title": "AI in Finance",
                    "url": "https://example.com/ai-finance",
                    "snippet": "Financial institutions use AI..."
                }
            ]
        }
        
        result = await report_tool.execute(
            title="AI Applications Report",
            query="artificial intelligence applications",
            search_results=search_results,
            extraction_results={},
            output_path=str(temp_workspace / "report.md")
        )
        
        assert result["success"] is True
        assert "markdown" in result
        
        # Verify content structure
        content = result["markdown"]
        assert "# AI Applications Report" in content
        assert "## Executive Summary" in content
        assert "## Search Results" in content
        assert "AI in Healthcare" in content
        assert "AI in Finance" in content


class TestPDFCompilerTool:
    """Real-world tests for PDFCompilerTool."""
    
    @pytest.fixture
    def pdf_tool(self):
        """Create PDF compiler tool instance."""
        return PDFCompilerTool()
    
    @pytest.mark.asyncio
    async def test_pdf_generation(self, pdf_tool, temp_workspace):
        """Test generating a real PDF from markdown."""
        markdown_content = """# Test Report

## Introduction

This is a test report for PDF generation.

### Key Points

1. **First Point**: Testing PDF generation
2. **Second Point**: Validating tool functionality
3. **Third Point**: Checking output quality

## Conclusion

The PDF generation tool should create a properly formatted document.
"""
        
        output_path = temp_workspace / "test_report.pdf"
        
        result = await pdf_tool.execute(
            markdown_content=markdown_content,
            output_path=str(output_path),
            title="Test Report",
            author="Test Suite"
        )
        
        if result["success"]:
            # Pandoc is installed
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            assert result["file_size"] > 0
        else:
            # Pandoc not installed - should fail gracefully
            assert "pandoc" in result["error"].lower()


class TestToolIntegration:
    """Test tools working together in orchestrator pipelines."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_web_scraping_pipeline(self, orchestrator, temp_workspace):
        """Test a pipeline that searches web, scrapes content, and generates report."""
        yaml_content = """
name: "Web Research Pipeline"
description: "Search, scrape, and report on a topic"

inputs:
  topic:
    type: string
    required: true
  output_dir:
    type: string
    required: true

steps:
  - id: search_web
    action: web_search
    parameters:
      query: "{{topic}} latest news 2024"
      max_results: 3
  
  - id: scrape_first_result
    action: scrape_page
    parameters:
      url: "{{search_web.results[0].url if search_web.results else 'https://example.com'}}"
    depends_on: [search_web]
  
  - id: generate_report
    action: generate_report
    parameters:
      title: "Research Report: {{topic}}"
      query: "{{topic}}"
      search_results: "{{search_web}}"
      extraction_results: "{{scrape_first_result}}"
      output_path: "{{output_dir}}/report.md"
    depends_on: [search_web, scrape_first_result]

outputs:
  report_path: "{{output_dir}}/report.md"
  search_count: "{{search_web.total_results}}"
"""
        
        # Execute pipeline
        context = {
            "topic": "artificial intelligence",
            "output_dir": str(temp_workspace)
        }
        
        result = await orchestrator.execute_yaml(
            yaml_content=yaml_content,
            context=context
        )
        
        # Verify results
        assert "search_web" in result
        assert result["search_web"]["success"] is True
        
        assert "generate_report" in result
        assert result["generate_report"]["success"] is True
        
        # Check report was created
        report_path = temp_workspace / "report.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "artificial intelligence" in content.lower()
    
    @pytest.mark.asyncio
    async def test_file_processing_pipeline(self, orchestrator, temp_workspace):
        """Test a pipeline that creates, processes, and validates files."""
        # Create test data file
        test_data = {
            "items": [
                {"id": 1, "name": "Item A", "price": 10.50},
                {"id": 2, "name": "Item B", "price": 25.00},
                {"id": 3, "name": "Item C", "price": 15.75}
            ]
        }
        
        input_file = temp_workspace / "input_data.json"
        input_file.write_text(json.dumps(test_data, indent=2))
        
        yaml_content = """
name: "File Processing Pipeline"
description: "Read, process, and validate JSON data"

inputs:
  input_file:
    type: string
    required: true
  output_file:
    type: string
    required: true

steps:
  - id: read_data
    action: file
    parameters:
      action: read
      path: "{{input_file}}"
  
  - id: process_data
    action: process
    parameters:
      action: transform
      data: "{{read_data.content}}"
      transform_spec:
        total_price: "sum(item['price'] for item in json.loads(data)['items'])"
        item_count: "len(json.loads(data)['items'])"
        average_price: "sum(item['price'] for item in json.loads(data)['items']) / len(json.loads(data)['items'])"
    depends_on: [read_data]
  
  - id: validate_results
    action: validate
    parameters:
      data: "{{process_data.processed_data}}"
      schema:
        type: object
        properties:
          total_price:
            type: number
          item_count:
            type: integer
          average_price:
            type: number
        required: [total_price, item_count, average_price]
    depends_on: [process_data]
  
  - id: save_results
    action: file
    parameters:
      action: write
      path: "{{output_file}}"
      content: "{{process_data.processed_data}}"
    depends_on: [validate_results]

outputs:
  validation_passed: "{{validate_results.valid}}"
  total_price: "{{process_data.processed_data.total_price}}"
"""
        
        # Execute pipeline
        context = {
            "input_file": str(input_file),
            "output_file": str(temp_workspace / "output_data.json")
        }
        
        result = await orchestrator.execute_yaml(
            yaml_content=yaml_content,
            context=context
        )
        
        # Verify results
        assert result["validate_results"]["valid"] is True
        assert (temp_workspace / "output_data.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])