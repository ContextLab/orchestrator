# Testing Scripts

This directory contains scripts for testing pipeline execution, functionality, and integration with real models and APIs.

## Scripts

### Core Testing Scripts

- **`test_all_real_pipelines.py`** - Comprehensive execution tests with real models
  - Tests all pipelines with actual model API calls
  - Validates end-to-end pipeline functionality
  - Generates execution reports and metrics
  - Usage: `python scripts/testing/test_all_real_pipelines.py`

- **`test_all_pipelines_with_wrappers.py`** - Tests pipelines with wrapper functionality
  - Tests wrapper-based pipeline execution
  - Validates wrapper integration and functionality
  - Usage: `python scripts/testing/test_all_pipelines_with_wrappers.py`

### Demo and Example Testing

- **`quick_wrapper_validation_demo.py`** - Demonstration of wrapper validation
  - Shows wrapper functionality in action
  - Quick demo for development and presentations
  - Usage: `python scripts/testing/quick_wrapper_validation_demo.py`

### API and Integration Testing

- **`test_mcp_queries.py`** - Tests MCP (Model Control Protocol) query functionality
  - Validates MCP integration and communication
  - Tests model routing and query processing
  - Usage: `python scripts/testing/test_mcp_queries.py`

## Testing Philosophy

All testing scripts in this directory follow the "real world" testing philosophy:
- **No Mock Objects**: All tests use real API calls and models
- **Real Data**: Tests use actual data files and inputs
- **Production-Like**: Tests mimic actual user workflows
- **Comprehensive Coverage**: Tests cover edge cases and error conditions

## Usage Examples

```bash
# Test all pipelines with real model execution
python scripts/testing/test_all_real_pipelines.py

# Test wrapper functionality
python scripts/testing/test_all_pipelines_with_wrappers.py

# Quick wrapper demo
python scripts/testing/quick_wrapper_validation_demo.py

# Test MCP queries
python scripts/testing/test_mcp_queries.py
```

## Test Categories

### 1. Execution Tests
- Test pipeline execution from start to finish
- Validate outputs and file generation
- Check error handling and recovery

### 2. Integration Tests
- Test integration with external APIs
- Validate model routing and selection
- Check wrapper functionality

### 3. Performance Tests
- Measure execution times and resource usage
- Test with different model configurations
- Validate parallel execution capabilities

### 4. Quality Tests
- Validate output quality and format
- Check for hallucinations and inconsistencies
- Test prompt engineering effectiveness

## Test Data

Testing scripts use:
- Real pipeline configurations from `examples/`
- Actual data files from `examples/data/`
- Live model APIs (OpenAI, Anthropic, Google, etc.)
- Real output generation and validation

## Error Handling

Testing scripts include comprehensive error handling for:
- API rate limits and timeouts
- Model unavailability
- Network connectivity issues
- Invalid configurations
- Resource constraints

## Reporting

Most testing scripts generate detailed reports including:
- Execution summaries and statistics
- Error logs and debugging information
- Performance metrics and benchmarks
- Quality assessment scores
- Recommendations for improvements