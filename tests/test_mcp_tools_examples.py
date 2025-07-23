"""Tests for MCP tools configuration examples."""

import json
import yaml
from pathlib import Path


class TestMCPToolsExamples:
    """Test suite for MCP tools configuration and examples."""

    def test_mcp_config_file_exists(self):
        """Test that MCP tools config file exists in config directory."""
        config_path = Path(__file__).parent.parent / "config" / "mcp_tools_config.json"
        assert config_path.exists(), "MCP tools config file should exist"

    def test_mcp_config_structure(self):
        """Test that MCP config has correct structure."""
        config_path = Path(__file__).parent.parent / "config" / "mcp_tools_config.json"

        with open(config_path) as f:
            config = json.load(f)

        assert "tools" in config
        assert "version" in config
        assert "capabilities" in config
        assert isinstance(config["tools"], list)
        assert len(config["tools"]) > 0

    def test_tool_schemas(self):
        """Test that each tool has proper schema definition."""
        config_path = Path(__file__).parent.parent / "config" / "mcp_tools_config.json"

        with open(config_path) as f:
            config = json.load(f)

        for tool in config["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"
            assert "properties" in tool["inputSchema"]
            assert "required" in tool["inputSchema"]

    def test_validation_tool_example(self):
        """Test validation tool YAML example."""
        yaml_content = """
        - action: validate_data
          tool: validation
          parameters:
            data: "{{ results.extracted_data }}"
            schema:
              type: object
              properties:
                email:
                  type: string
                  format: email
              required: ["email"]
            rules:
              - field: age
                condition: "value >= 18"
                message: "Must be 18 or older"
        """

        data = yaml.safe_load(yaml_content)
        assert data[0]["tool"] == "validation"
        assert "schema" in data[0]["parameters"]
        assert "rules" in data[0]["parameters"]

    def test_filesystem_tool_examples(self):
        """Test filesystem tool YAML examples."""
        # Read example
        read_yaml = """
        - action: manage_files
          tool: filesystem
          parameters:
            action: read
            path: "/data/input.json"
        """

        data = yaml.safe_load(read_yaml)
        assert data[0]["tool"] == "filesystem"
        assert data[0]["parameters"]["action"] == "read"

        # Write example
        write_yaml = """
        - action: save_results
          tool: filesystem
          parameters:
            action: write
            path: "/output/results.json"
            content: "{{ results | json }}"
        """

        data = yaml.safe_load(write_yaml)
        assert data[0]["parameters"]["action"] == "write"
        assert "content" in data[0]["parameters"]

    def test_headless_browser_examples(self):
        """Test headless browser tool examples."""
        # Search example
        search_yaml = """
        - action: search_web
          tool: headless-browser
          parameters:
            action: search
            query: "latest AI research papers"
            sources: ["arxiv", "scholar", "pubmed"]
        """

        data = yaml.safe_load(search_yaml)
        assert data[0]["tool"] == "headless-browser"
        assert data[0]["parameters"]["action"] == "search"
        assert isinstance(data[0]["parameters"]["sources"], list)

        # Verify example
        verify_yaml = """
        - action: verify_links
          tool: headless-browser
          parameters:
            action: verify
            url: "https://example.com"
        """

        data = yaml.safe_load(verify_yaml)
        assert data[0]["parameters"]["action"] == "verify"
        assert "url" in data[0]["parameters"]

    def test_terminal_tool_example(self):
        """Test terminal tool example."""
        yaml_content = """
        - action: run_analysis
          tool: terminal
          parameters:
            command: "python analyze.py --input data.csv"
            working_dir: "/project"
            timeout: 300
            capture_output: true
        """

        data = yaml.safe_load(yaml_content)
        assert data[0]["tool"] == "terminal"
        assert data[0]["parameters"]["timeout"] == 300
        assert data[0]["parameters"]["capture_output"] is True

    def test_web_search_tool_example(self):
        """Test web search tool example."""
        yaml_content = """
        - action: research_topic
          tool: web-search
          parameters:
            query: "quantum computing applications"
            max_results: 20
        """

        data = yaml.safe_load(yaml_content)
        assert data[0]["tool"] == "web-search"
        assert data[0]["parameters"]["max_results"] == 20

    def test_data_processing_tool_example(self):
        """Test data processing tool example."""
        yaml_content = """
        - action: transform_data
          tool: data-processing
          parameters:
            action: convert
            data: "{{ raw_data }}"
            format: json
            operation:
              to_format: csv
        """

        data = yaml.safe_load(yaml_content)
        assert data[0]["tool"] == "data-processing"
        assert data[0]["parameters"]["action"] == "convert"
        assert data[0]["parameters"]["operation"]["to_format"] == "csv"

    def test_custom_tool_schema(self):
        """Test custom tool configuration schema."""
        custom_tool = {
            "name": "my-custom-tool",
            "description": "Custom tool for specific task",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["process", "analyze", "generate"],
                        "description": "Action to perform",
                    },
                    "input": {"type": "string", "description": "Input data or path"},
                    "options": {
                        "type": "object",
                        "description": "Additional options",
                        "properties": {
                            "format": {"type": "string", "default": "json"},
                            "verbose": {"type": "boolean", "default": False},
                        },
                    },
                },
                "required": ["action", "input"],
            },
        }

        # Validate structure
        assert custom_tool["name"] == "my-custom-tool"
        assert "enum" in custom_tool["inputSchema"]["properties"]["action"]
        assert (
            "default" in custom_tool["inputSchema"]["properties"]["options"]["properties"]["format"]
        )

    def test_complete_pipeline_example(self):
        """Test the complete pipeline example."""
        yaml_content = """
name: web-research-pipeline
description: Research a topic and generate a report

inputs:
  - name: topic
    type: string
    description: Research topic

steps:
  # Search for information
  - id: search_topic
    action: research
    tool: web-search
    parameters:
      query: "{{ topic }} latest developments 2024"
      max_results: 10

  # Verify and scrape top results
  - id: scrape_articles
    for_each: "{{ results.search_topic.results[:3] }}"
    as: article
    action: scrape
    tool: headless-browser
    parameters:
      action: scrape
      url: "{{ article.url }}"

  # Save raw data
  - id: save_raw
    action: save
    tool: filesystem
    parameters:
      action: write
      path: "research/{{ topic }}/raw_data.json"
      content: "{{ results.scrape_articles | json }}"

  # Process and analyze
  - id: analyze_content
    action: analyze
    tool: data-processing
    parameters:
      action: transform
      data: "{{ results.scrape_articles }}"
      operation:
        extract: ["title", "summary", "key_points"]
        format: structured

  # Validate results
  - id: validate_data
    action: validate
    tool: validation
    parameters:
      data: "{{ results.analyze_content }}"
      schema:
        type: array
        items:
          type: object
          required: ["title", "summary"]

  # Generate report
  - id: create_report
    action: generate
    parameters:
      template: |
        # Research Report: {{ topic }}
        
        ## Summary
        {{ results.analyze_content | summarize }}
        
        ## Key Findings
        {{ results.analyze_content | format_findings }}

  # Save final report
  - id: save_report
    action: save
    tool: filesystem
    parameters:
      action: write
      path: "research/{{ topic }}/report.md"
      content: "{{ results.create_report }}"
        """

        data = yaml.safe_load(yaml_content)

        # Validate pipeline structure
        assert data["name"] == "web-research-pipeline"
        assert len(data["steps"]) == 7

        # Validate tool usage
        tools_used = [step.get("tool") for step in data["steps"] if "tool" in step]
        assert "web-search" in tools_used
        assert "headless-browser" in tools_used
        assert "filesystem" in tools_used
        assert "data-processing" in tools_used
        assert "validation" in tools_used

        # Validate for_each structure
        scrape_step = next(s for s in data["steps"] if s["id"] == "scrape_articles")
        assert "for_each" in scrape_step
        assert "as" in scrape_step

    def test_tool_parameter_validation(self):
        """Test that tool parameters match schema requirements."""
        config_path = Path(__file__).parent.parent / "config" / "mcp_tools_config.json"

        with open(config_path) as f:
            config = json.load(f)

        # Create a map of tool schemas
        tool_schemas = {tool["name"]: tool["inputSchema"] for tool in config["tools"]}

        # Test filesystem tool parameters
        fs_schema = tool_schemas["filesystem"]
        required = fs_schema.get("required", [])
        assert "action" in required
        assert "path" in required

        # Test validation tool parameters
        val_schema = tool_schemas["validation"]
        required = val_schema.get("required", [])
        assert "data" in required
