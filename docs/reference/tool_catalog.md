# Orchestrator Tool Catalog

This comprehensive catalog documents all available tools in the Orchestrator framework. Each tool is designed for specific tasks and can be combined in pipelines for powerful automation.

## Tool Categories

1. [System Tools](#system-tools) - File system, terminal, and system operations
2. [Web Tools](#web-tools) - Web search, browsing, and content fetching
3. [Data Tools](#data-tools) - Data processing, analysis, and transformation
4. [Report Tools](#report-tools) - Report generation and document creation
5. [LLM Tools](#llm-tools) - Language model interactions and analysis
6. [User Interaction Tools](#user-interaction-tools) - User prompts and approvals
7. [Pipeline Tools](#pipeline-tools) - Pipeline recursion and modularity
8. [Multimodal Tools](#multimodal-tools) - Image, audio, and video processing
9. [MCP Tools](#mcp-tools) - Model Context Protocol integration
10. [Validation Tools](#validation-tools) - Data validation and schema checking

---

## System Tools

### FileSystemTool
**ID:** `filesystem`

Comprehensive file system operations tool.

**Actions:**
- `read` - Read file contents
- `write` - Write content to file
- `delete` - Delete file or directory
- `copy` - Copy file or directory
- `move` - Move/rename file or directory
- `list` - List directory contents
- `exists` - Check if path exists
- `mkdir` - Create directory
- `chmod` - Change file permissions

**Example:**
```yaml
- id: save_report
  tool: filesystem
  action: write
  parameters:
    path: "/tmp/report.md"
    content: "{{ report_content }}"
    mode: "w"  # write mode (w, a, x)
```

### TerminalTool
**ID:** `terminal`

Execute shell commands and scripts.

**Actions:**
- `execute` - Run a shell command
- `script` - Execute a script file

**Parameters:**
- `command` - Command to execute
- `cwd` - Working directory
- `env` - Environment variables
- `timeout` - Command timeout in seconds
- `shell` - Shell to use (bash, sh, zsh)

**Example:**
```yaml
- id: run_analysis
  tool: terminal
  action: execute
  parameters:
    command: "python analyze.py --input data.csv"
    cwd: "/project"
    timeout: 300
```

---

## Web Tools

### WebSearchTool
**ID:** `web-search`

Search the web using DuckDuckGo.

**Actions:**
- `search` - Perform web search

**Parameters:**
- `query` - Search query
- `max_results` - Maximum results (default: 10)
- `region` - Region code (e.g., 'us-en')
- `safesearch` - Safe search level ('on', 'moderate', 'off')

**Example:**
```yaml
- id: research_topic
  tool: web-search
  action: search
  parameters:
    query: "machine learning best practices 2024"
    max_results: 5
```

### HeadlessBrowserTool
**ID:** `headless-browser`

Advanced web automation using Playwright.

**Actions:**
- `navigate` - Go to URL
- `screenshot` - Capture screenshot
- `extract` - Extract content
- `click` - Click element
- `fill` - Fill form field
- `wait` - Wait for condition
- `execute_js` - Execute JavaScript

**Example:**
```yaml
- id: capture_page
  tool: headless-browser
  action: navigate
  parameters:
    url: "https://example.com"
    
- id: take_screenshot
  tool: headless-browser
  action: screenshot
  parameters:
    path: "/tmp/screenshot.png"
    full_page: true
```

---

## Data Tools

### DataProcessingTool
**ID:** `data-processing`

Process and transform data in various formats.

**Actions:**
- `transform` - Transform data format
- `filter` - Filter data
- `aggregate` - Aggregate data
- `sort` - Sort data
- `merge` - Merge datasets
- `split` - Split dataset

**Supported Formats:**
- CSV
- JSON
- XML
- Excel
- Parquet

**Example:**
```yaml
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
```

---

## Report Tools

### ReportGeneratorTool
**ID:** `report-generator`

Generate formatted reports and documents.

**Actions:**
- `generate` - Create report
- `append` - Add to existing report
- `convert` - Convert format

**Formats:**
- Markdown
- HTML
- PDF (requires wkhtmltopdf)
- DOCX
- LaTeX

**Example:**
```yaml
- id: create_report
  tool: report-generator
  action: generate
  parameters:
    title: "Analysis Report"
    format: "pdf"
    template: |
      # {{ title }}
      
      ## Summary
      {{ summary }}
      
      ## Data Analysis
      {{ analysis_results }}
    metadata:
      author: "AI Assistant"
      date: "{{ current_date }}"
```

### PDFCompilerTool
**ID:** `pdf-compiler`

Create PDFs from HTML or Markdown.

**Actions:**
- `compile` - Generate PDF
- `merge` - Merge PDFs
- `split` - Split PDF
- `extract` - Extract pages

**Example:**
```yaml
- id: compile_pdf
  tool: pdf-compiler
  action: compile
  parameters:
    content: "{{ markdown_content }}"
    output_path: "/tmp/document.pdf"
    options:
      margin_top: "2cm"
      margin_bottom: "2cm"
      page_size: "A4"
```

---

## LLM Tools

### LLMGenerateTool
**ID:** `llm-generate`

Generate text using language models.

**Actions:**
- `generate` - Generate text
- `complete` - Complete text
- `chat` - Chat conversation

**Parameters:**
- `prompt` - Input prompt
- `model` - Model to use (optional, uses routing)
- `temperature` - Creativity (0-2)
- `max_tokens` - Maximum length
- `system_prompt` - System instructions

**Example:**
```yaml
- id: write_summary
  tool: llm-generate
  action: generate
  parameters:
    prompt: |
      Summarize this article:
      {{ article_content }}
    temperature: 0.3
    max_tokens: 200
```

### LLMAnalyzeTool
**ID:** `llm-analyze`

Analyze content with structured output.

**Actions:**
- `analyze` - Analyze with schema
- `classify` - Classify content
- `extract` - Extract information

**Example:**
```yaml
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
```

### LLMRouterTool
**ID:** `llm-router`

Intelligently route requests to appropriate models.

**Actions:**
- `route` - Route based on requirements
- `select` - Select best model

**Example:**
```yaml
- id: smart_generation
  tool: llm-router
  action: route
  parameters:
    task: "complex_reasoning"
    requirements:
      accuracy: "high"
      speed: "medium"
      cost: "optimized"
    prompt: "{{ complex_prompt }}"
```

---

## User Interaction Tools

### UserPromptTool
**ID:** `user-prompt`

Get input from users.

**Actions:**
- `prompt` - Get user input

**Input Types:**
- `text` - Text input
- `number` - Numeric input
- `boolean` - Yes/no
- `choice` - Multiple choice
- `multiline` - Multi-line text
- `password` - Hidden input

**Example:**
```yaml
- id: get_user_input
  tool: user-prompt
  action: prompt
  parameters:
    prompt: "Enter the project name:"
    input_type: "text"
    validation:
      required: true
      pattern: "^[a-zA-Z0-9-]+$"
```

### ApprovalGateTool
**ID:** `approval-gate`

Require user approval to continue.

**Actions:**
- `request` - Request approval

**Example:**
```yaml
- id: confirm_deployment
  tool: approval-gate
  action: request
  parameters:
    title: "Deploy to Production?"
    content: |
      About to deploy version {{ version }} to production.
      
      Changes:
      {{ changes }}
    options: ["approve", "reject", "defer"]
    timeout: 3600  # 1 hour
```

### FeedbackCollectionTool
**ID:** `feedback-collection`

Collect structured feedback.

**Actions:**
- `collect` - Gather feedback

**Example:**
```yaml
- id: collect_feedback
  tool: feedback-collection
  action: collect
  parameters:
    title: "Pipeline Feedback"
    questions:
      - id: satisfaction
        text: "Rate your satisfaction (1-5)"
        type: rating
        scale: 5
      - id: improvements
        text: "Suggestions for improvement"
        type: text
        required: false
```

---

## Pipeline Tools

### SubPipelineTool
**ID:** `sub-pipeline`

Execute another pipeline as a sub-task.

**Actions:**
- `execute` - Run sub-pipeline

**Example:**
```yaml
- id: run_analysis
  tool: sub-pipeline
  action: execute
  parameters:
    pipeline_id: "data_analysis_pipeline"
    inputs:
      data: "{{ processed_data }}"
      config: "{{ analysis_config }}"
    inherit_context: true
```

### PipelineRecursionTool
**ID:** `pipeline-recursion`

Enable recursive pipeline execution.

**Actions:**
- `recurse` - Recursive execution
- `iterate` - Iterative execution

**Example:**
```yaml
- id: recursive_process
  tool: pipeline-recursion
  action: recurse
  parameters:
    condition: "{{ items_remaining > 0 }}"
    max_depth: 5
    inputs:
      items: "{{ remaining_items }}"
```

---

## Multimodal Tools

### ImageAnalysisTool
**ID:** `image-analysis`

Analyze images using vision models.

**Actions:**
- `analyze` - Analyze image
- `describe` - Generate description
- `extract_text` - OCR
- `detect_objects` - Object detection

**Example:**
```yaml
- id: analyze_image
  tool: image-analysis
  action: analyze
  parameters:
    image_path: "/tmp/chart.png"
    analysis_type: "detailed"
    questions:
      - "What type of chart is this?"
      - "What are the key trends?"
```

### ImageGenerationTool
**ID:** `image-generation`

Generate images using AI models.

**Actions:**
- `generate` - Create image
- `edit` - Edit existing image
- `variation` - Create variations

**Example:**
```yaml
- id: create_logo
  tool: image-generation
  action: generate
  parameters:
    prompt: "Modern minimalist logo for AI company"
    size: "512x512"
    style: "vector"
    n: 3  # Generate 3 variations
```

### AudioTranscriptionTool
**ID:** `audio-transcription`

Transcribe audio to text.

**Actions:**
- `transcribe` - Convert speech to text
- `translate` - Transcribe and translate

**Example:**
```yaml
- id: transcribe_meeting
  tool: audio-transcription
  action: transcribe
  parameters:
    audio_path: "/tmp/meeting.mp3"
    language: "en"
    include_timestamps: true
```

### VideoAnalysisTool
**ID:** `video-analysis`

Analyze video content.

**Actions:**
- `analyze` - Analyze video
- `extract_frames` - Extract key frames
- `summarize` - Create video summary

**Example:**
```yaml
- id: analyze_tutorial
  tool: video-analysis
  action: summarize
  parameters:
    video_path: "/tmp/tutorial.mp4"
    summary_type: "key_points"
    max_duration: 60  # 60 second summary
```

---

## MCP Tools

### MCPServerTool
**ID:** `mcp-server`

Interact with Model Context Protocol servers.

**Actions:**
- `connect` - Connect to server
- `execute` - Execute MCP command
- `query` - Query server state

**Example:**
```yaml
- id: mcp_query
  tool: mcp-server
  action: execute
  parameters:
    server: "knowledge-base"
    command: "search"
    args:
      query: "{{ search_term }}"
      limit: 10
```

### MCPMemoryTool
**ID:** `mcp-memory`

Persistent memory storage via MCP.

**Actions:**
- `store` - Store in memory
- `retrieve` - Retrieve from memory
- `update` - Update memory
- `delete` - Delete from memory

**Example:**
```yaml
- id: remember_context
  tool: mcp-memory
  action: store
  parameters:
    key: "user_preferences"
    value: "{{ preferences }}"
    ttl: 86400  # 24 hours
```

### MCPResourceTool
**ID:** `mcp-resource`

Access MCP-managed resources.

**Actions:**
- `list` - List resources
- `get` - Get resource
- `create` - Create resource
- `update` - Update resource
- `delete` - Delete resource

**Example:**
```yaml
- id: get_template
  tool: mcp-resource
  action: get
  parameters:
    resource_type: "template"
    resource_id: "report_template_v2"
```

---

## Validation Tools

The orchestrator framework provides comprehensive validation capabilities through multiple specialized validators.

### PipelineValidationTool
**ID:** `pipeline-validation`

Comprehensive pipeline validation using the unified validation framework.

**Actions:**
- `validate` - Full pipeline validation
- `check-templates` - Template-only validation  
- `check-dependencies` - Dependency validation
- `check-tools` - Tool configuration validation
- `check-models` - Model configuration validation
- `check-outputs` - Output validation

**Parameters:**
- `validation_level` - Validation strictness (`strict`, `permissive`, `development`)
- `output_format` - Report format (`text`, `json`, `detailed`, `summary`)
- `validators` - List of validators to run (optional, runs all by default)

**Example:**
```yaml
- id: validate_pipeline
  tool: pipeline-validation
  action: validate
  parameters:
    pipeline: "{{ pipeline_config }}"
    validation_level: "strict"
    output_format: "detailed"
    validators: ["template", "dependency", "tool"]
```

### TemplateValidationTool
**ID:** `template-validation`

Validates Jinja2 templates for syntax errors and undefined variables.

**Actions:**
- `validate` - Validate template syntax and variables
- `render` - Test render template with context
- `analyze` - Analyze template dependencies

**Example:**
```yaml
- id: check_template
  tool: template-validation
  action: validate
  parameters:
    template: "Hello {{name}}, your score is {{score}}"
    context:
      name: "Alice"
      score: 95
    strict: true
```

### OutputValidationTool  
**ID:** `output-validation`

Validates data against schemas and validation rules.

**Actions:**
- `validate` - Validate data against rules
- `extract` - Extract and validate specific fields
- `transform` - Transform data to match schema

**Validation Rules:**
- `consistency` - Check required fields, data consistency
- `format` - Validate data formats (JSON, YAML, markdown, etc.)
- `dependency` - Validate cross-references and relationships
- `filesystem` - Validate file system outputs

**Example:**
```yaml
- id: validate_output
  tool: output-validation
  action: validate
  parameters:
    data: "{{ analysis_result }}"
    rules:
      - name: "required_fields"
        type: "consistency"
        parameters:
          required: ["title", "summary", "findings"]
      - name: "format_check"
        type: "format"
        parameters:
          format: "markdown"
          schema:
            type: object
            properties:
              title: {type: string}
              summary: {type: string}
              findings: {type: array}
    mode: "STRICT"
```

### DataFlowValidationTool
**ID:** `dataflow-validation`

Validates data flow between pipeline steps.

**Actions:**
- `validate` - Check data flow consistency
- `trace` - Trace data dependencies
- `analyze` - Analyze data types and compatibility

**Example:**
```yaml
- id: check_dataflow  
  tool: dataflow-validation
  action: validate
  parameters:
    pipeline: "{{ pipeline_config }}"
    check_types: true
    strict_mode: false
```

### DependencyValidationTool
**ID:** `dependency-validation`

Validates task dependencies and execution order.

**Actions:**
- `validate` - Check dependency graph validity
- `detect-cycles` - Find circular dependencies
- `order` - Generate execution order

**Example:**
```yaml
- id: check_dependencies
  tool: dependency-validation  
  action: validate
  parameters:
    pipeline: "{{ pipeline_config }}"
    allow_optional_deps: true
    max_parallel: 5
```

### ModelValidationTool
**ID:** `model-validation`

Validates model configurations and availability.

**Actions:**
- `validate` - Validate model configuration
- `test` - Test model connectivity  
- `compatibility` - Check model compatibility

**Example:**
```yaml
- id: validate_model
  tool: model-validation
  action: validate
  parameters:
    model_config:
      name: "gpt-4o-mini"
      provider: "openai"
      parameters:
        temperature: 0.7
        max_tokens: 1000
    check_availability: true
```

### ToolValidationTool
**ID:** `tool-validation`

Validates tool configurations and parameters.

**Actions:**
- `validate` - Validate tool configuration
- `check-params` - Validate parameters
- `test` - Test tool execution

**Example:**
```yaml
- id: validate_tools
  tool: tool-validation
  action: validate
  parameters:
    tools_config: "{{ pipeline_config.steps }}"
    check_availability: true
    validate_params: true
```

---

## Tool Combinations

Tools can be combined for powerful workflows:

### Example: Research Pipeline
```yaml
steps:
  # Search for information
  - id: search
    tool: web-search
    action: search
    parameters:
      query: "{{ research_topic }}"
      
  # Analyze results
  - id: analyze
    tool: llm-analyze
    action: analyze
    parameters:
      content: "{{ search.results }}"
      analysis_type: "key_points"
      
  # Generate report
  - id: report
    tool: report-generator
    action: generate
    parameters:
      title: "Research: {{ research_topic }}"
      format: "pdf"
      content: "{{ analyze.result }}"
```

### Example: Data Processing Pipeline
```yaml
steps:
  # Read data
  - id: read_data
    tool: filesystem
    action: read
    parameters:
      path: "{{ input_file }}"
      
  # Process data
  - id: process
    tool: data-processing
    action: transform
    parameters:
      input_data: "{{ read_data.content }}"
      operations:
        - type: filter
          condition: "status == 'active'"
        - type: aggregate
          group_by: "category"
          
  # Validate results
  - id: validate
    tool: validation
    action: validate
    parameters:
      data: "{{ process.result }}"
      schema: "{{ output_schema }}"
      
  # Save results
  - id: save
    tool: filesystem
    action: write
    parameters:
      path: "{{ output_file }}"
      content: "{{ validate.result }}"
```

---

## Best Practices

1. **Tool Selection**
   - Choose the most specific tool for your task
   - Consider performance implications
   - Use validation tools for critical data

2. **Error Handling**
   - Always handle potential failures
   - Use conditional execution
   - Implement retries for network operations

3. **Security**
   - Validate all user inputs
   - Use approval gates for sensitive operations
   - Limit file system access paths

4. **Performance**
   - Cache results when possible
   - Use parallel execution for independent tasks
   - Batch operations when available

5. **Debugging**
   - Enable verbose logging for troubleshooting
   - Use validation tools to check intermediate results
   - Test with small datasets first

---

## Tool Development

To create custom tools, implement the `Tool` interface:

```python
from orchestrator.tools.base import Tool

class CustomTool(Tool):
    def __init__(self):
        super().__init__(
            id="custom-tool",
            name="Custom Tool",
            description="My custom tool"
        )
    
    async def execute(self, action: str, parameters: dict):
        # Implementation
        return result
```

See the [Custom Tools Guide](../advanced/custom_tools.md) for detailed instructions.