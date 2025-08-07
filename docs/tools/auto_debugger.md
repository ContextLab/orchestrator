# AutoDebugger Tool

## Overview

The AutoDebugger is a universal debugging tool that can analyze, fix, and validate ANY type of content or process. It implements Issue #201's requirements for self-healing pipeline execution using a three-step debugging loop: **Analyze → Execute → Validate**.

## Key Features

- **Universal Debugging**: Works with any content type (Python, JavaScript, SQL, LaTeX, YAML, JSON, HTML, etc.)
- **Real LLM Analysis**: Uses multiple specialized models (analyzer, fixer, validator)
- **Real Tool Execution**: Integrates with all orchestrator tools for actual fixes
- **Self-Healing**: Iteratively debugs until problems are resolved
- **NO MOCKS**: All functionality uses real systems or raises exceptions
- **Pattern Recognition**: Learns from debugging history within sessions
- **Comprehensive Validation**: Validates fixes before considering them complete

## Usage in Pipelines

### Basic Usage

```yaml
steps:
  - id: debug_code
    name: "Fix Python Code"
    tool: auto_debugger
    config:
      task_description: "Fix syntax errors in data processing script"
      content_to_debug: |
        def process_data(items):
            results = []
            for item in items
                if item.valid:
                    results.append(item.process())
            return results
      error_context: "SyntaxError: invalid syntax (missing colon on line 4)"
      expected_outcome: "Working Python code that processes valid items"
    outputs:
      fixed_code: "{{ result.final_content }}"
      debug_success: "{{ result.success }}"
```

### Advanced Usage with Context

```yaml
steps:
  - id: debug_with_tools
    name: "Debug API Integration"
    tool: auto_debugger
    config:
      task_description: "Fix API integration with proper error handling"
      content_to_debug: "{{ context.api_code }}"
      error_context: "No error handling, 404 errors cause crashes"
      expected_outcome: "Robust API code with comprehensive error handling"
      available_tools: ["web_tools", "validation", "filesystem"]
    outputs:
      fixed_api_code: "{{ result.final_content }}"
      debug_iterations: "{{ result.total_iterations }}"
      tools_used: "{{ result.tools_used }}"
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task_description` | string | Yes | - | Clear description of what you're trying to accomplish |
| `content_to_debug` | string | No | "" | The actual content that needs debugging (code, config, data, etc.) |
| `error_context` | string | No | "" | Error messages, failure descriptions, or problem context |
| `expected_outcome` | string | No | "" | Description of what should happen when the issue is fixed |
| `available_tools` | array | No | null | Specific tools to use for debugging (auto-detected if not specified) |

## Output Structure

The AutoDebugger returns a structured JSON object with comprehensive debugging information:

```json
{
  "success": true,
  "session_id": "uuid-string",
  "task_description": "Fix Python syntax errors...",
  "total_iterations": 2,
  "final_content": "def process_data(items):\n    results = []\n    for item in items:\n        if item.valid:\n            results.append(item.process())\n    return results",
  "validation": {
    "is_resolved": true,
    "quality_score": 0.95,
    "reasoning": "All syntax errors resolved, code structure improved"
  },
  "debug_summary": "Fixed missing colon, improved code structure",
  "modifications_made": ["Added missing colon on line 4", "Improved indentation"],
  "tools_used": ["python_execution", "code_analysis"],
  "execution_time": 12.5
}
```

### Success Fields

- `success`: Boolean indicating if debugging was successful
- `final_content`: The fixed/debugged content
- `validation`: Detailed validation results from LLM analysis
- `debug_summary`: Human-readable summary of the debugging session
- `modifications_made`: List of specific changes made
- `tools_used`: Tools that were used during debugging
- `execution_time`: Total time spent debugging (seconds)

### Failure Fields

- `error_message`: Description of why debugging failed
- `final_error`: The last error encountered

## Supported Content Types

The AutoDebugger automatically detects and handles various content types:

### Programming Languages
- **Python**: Syntax errors, runtime issues, import problems, logic errors
- **JavaScript**: Runtime errors, async/await issues, Node.js problems
- **SQL**: Query syntax, performance issues, data type problems
- **HTML/CSS**: Markup errors, styling issues, validation problems

### Configuration Files
- **YAML**: Syntax errors, indentation issues, structure problems
- **JSON**: Parse errors, schema validation, format issues
- **XML**: Well-formedness, validation, namespace issues

### Document Formats
- **LaTeX**: Compilation errors, missing packages, syntax issues
- **Markdown**: Formatting problems, link issues, structure errors

### Data Formats
- **CSV**: Format errors, delimiter issues, encoding problems
- **Data Processing**: Pandas errors, data type mismatches, missing values

## Debugging Process

The AutoDebugger uses a sophisticated three-step debugging loop:

### Step 1: Analysis
- Uses specialized LLM (Claude 3.5 Sonnet by default) to analyze the problem
- Identifies root cause and suggests specific actions
- Determines which tools to use and parameters needed
- Provides confidence score and risk assessment

### Step 2: Execution
- Executes the suggested fix using real tools and operations
- Supports multiple execution methods:
  - **Tool Execution**: Uses orchestrator tools (filesystem, terminal, etc.)
  - **LLM Generation**: Uses fixer model to generate corrected content
  - **Command Execution**: Runs system commands for compilation, testing, etc.
  - **File Operations**: Real filesystem operations for saving fixes

### Step 3: Validation
- Uses validator model (GPT-4o-mini by default) to verify the fix
- Checks if the problem is fully resolved
- Validates that the solution meets expected outcomes
- Provides quality score and identifies any remaining issues

### Iteration Logic
- Continues iterating until problem is resolved or max iterations reached
- Updates context with each iteration for learning
- Tracks modifications and tools used
- Handles failures gracefully with detailed error reporting

## Model Configuration

The AutoDebugger uses multiple specialized models with automatic fallbacks:

### Analyzer Model (Problem Analysis)
- **Default**: Claude 3.5 Sonnet (anthropic/claude-3-5-sonnet-20241022)
- **Fallbacks**: GPT-4o, Llama 3.1:70b
- **Purpose**: Deep problem analysis and solution planning

### Fixer Model (Content Generation)
- **Default**: GPT-4o (openai/gpt-4o)
- **Fallbacks**: Claude 3.5 Sonnet
- **Purpose**: Generate fixed content and corrections

### Validator Model (Solution Verification)
- **Default**: GPT-4o-mini (openai/gpt-4o-mini)
- **Fallbacks**: Claude 3 Haiku
- **Purpose**: Fast validation of fixes and quality assessment

## Real-World Examples

### Example 1: Python Syntax Debugging
```yaml
- tool: auto_debugger
  config:
    task_description: "Fix Python function with syntax errors"
    content_to_debug: |
      def fibonacci(n):
          if n <= 1:
              return n
          else
              return fibonacci(n-1) + fibonacci(n-2)
    error_context: "SyntaxError: invalid syntax (missing colon)"
    expected_outcome: "Working recursive Fibonacci function"
```

### Example 2: API Integration Debugging
```yaml
- tool: auto_debugger
  config:
    task_description: "Fix API client with error handling"
    content_to_debug: |
      import requests
      def get_user(user_id):
          response = requests.get(f"https://api.example.com/users/{user_id}")
          return response.json()
    error_context: "No error handling, fails on 404/500 responses"
    expected_outcome: "Robust API client with comprehensive error handling"
```

### Example 3: LaTeX Compilation Debugging
```yaml
- tool: auto_debugger
  config:
    task_description: "Fix LaTeX document compilation errors"
    content_to_debug: |
      \documentclass{article}
      \begin{document}
      \title{My Document}
      \author{Name}
      \maketitle
      \section{Introduction}
      This is a test document.
      \end{document
    error_context: "Missing closing brace in \\end{document}"
    expected_outcome: "LaTeX document that compiles to PDF successfully"
```

### Example 4: YAML Configuration Debugging
```yaml
- tool: auto_debugger
  config:
    task_description: "Fix YAML configuration syntax errors"
    content_to_debug: |
      name: my-service
      config:
        port: 8080
         timeout: 30
        retries 3
    error_context: "YAML indentation error, missing colon"
    expected_outcome: "Valid YAML that parses correctly"
```

## Integration with Other Tools

The AutoDebugger intelligently uses other orchestrator tools:

- **filesystem**: Reading/writing files during debugging
- **terminal**: Running compilation, testing, and validation commands
- **python-executor**: Testing Python fixes in real execution environments
- **web-tools**: Testing API integrations and web-related fixes
- **validation**: Schema validation and data verification
- **data-processing**: Fixing data format and processing issues

## Error Handling

The AutoDebugger follows a strict NO MOCKS policy:

- **Real System Failures**: If models or tools are unavailable, exceptions are raised
- **Graceful Degradation**: Uses fallback models when primary models fail
- **Comprehensive Logging**: All debugging steps are logged with context
- **Failure Analysis**: Failed debugging sessions provide detailed error analysis
- **Resource Management**: Proper cleanup of temporary files and processes

## Best Practices

### 1. Provide Clear Task Descriptions
```yaml
# Good
task_description: "Fix Python function that processes CSV data with proper error handling"

# Less effective  
task_description: "Fix this code"
```

### 2. Include Relevant Error Context
```yaml
# Good
error_context: "FileNotFoundError: 'data.csv' not found, KeyError: 'column_name' missing in some rows"

# Less effective
error_context: "It doesn't work"
```

### 3. Specify Expected Outcomes
```yaml
# Good
expected_outcome: "Function should gracefully handle missing files and columns, returning empty result instead of crashing"

# Less effective
expected_outcome: "Make it work"
```

### 4. Use Appropriate Tool Constraints
```yaml
# For system-level debugging
available_tools: ["terminal", "filesystem", "validation"]

# For data processing issues
available_tools: ["data-processing", "python-executor", "validation"]
```

## Performance Considerations

- **Iteration Limits**: Maximum 10 debugging iterations to prevent infinite loops
- **Model Timeouts**: Each model call has appropriate timeouts
- **Resource Usage**: Temporary files are cleaned up automatically
- **Concurrent Safety**: Thread-safe execution for multiple debugging sessions
- **Memory Management**: Large content is streamed when possible

## Monitoring and Logging

The AutoDebugger provides extensive monitoring capabilities:

- **Session Tracking**: Each debugging session has a unique ID
- **Performance Metrics**: Execution time, iteration count, success rates
- **Tool Usage Statistics**: Which tools are used most frequently
- **Model Performance**: Success rates and response times for each model
- **Error Classification**: Categorization of common debugging scenarios

## Troubleshooting

### Common Issues

1. **"No available models" Error**
   - Ensure API keys are configured for Anthropic, OpenAI, or Ollama
   - Check network connectivity for API calls
   - Verify model names are correct in configuration

2. **"Tool registry not available" Error**
   - Ensure orchestrator is properly initialized
   - Check that required tools are registered
   - Verify import paths are correct

3. **"Max iterations reached" Warning**
   - Review the complexity of the debugging task
   - Consider breaking down complex problems into smaller tasks
   - Check if the problem description is clear and actionable

4. **LLM Analysis Failures**
   - Verify content is not too large for model context limits
   - Check for special characters that might cause parsing issues
   - Ensure error context provides sufficient information

### Debug Mode

Enable detailed logging for debugging the AutoDebugger itself:

```python
import logging
logging.getLogger('orchestrator.tools.auto_debugger').setLevel(logging.DEBUG)
```

## Limitations

- **Context Windows**: Large files may exceed model context limits
- **Real-Time Requirements**: Not suitable for real-time debugging scenarios
- **Network Dependencies**: Requires internet connection for cloud-based models
- **Resource Intensive**: Complex debugging may require significant compute resources
- **Domain Knowledge**: Effectiveness depends on LLM training data coverage

## Future Enhancements

- **Specialized Models**: Domain-specific debugging models for different languages
- **Learning System**: Long-term learning from debugging patterns
- **Interactive Mode**: User feedback integration during debugging
- **Performance Optimization**: Caching and optimization for repeated scenarios
- **Custom Tool Integration**: User-defined debugging tools and validators