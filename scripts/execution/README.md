# Execution Scripts

This directory contains scripts for executing pipelines and managing pipeline runtime operations.

## Scripts

### Core Pipeline Execution

- **`run_pipeline.py`** - Main pipeline execution script
  - Primary CLI tool for executing YAML pipeline files
  - Supports multiple input formats and output configurations
  - Comprehensive error handling and logging
  - Usage: `python scripts/execution/run_pipeline.py <pipeline_file> [options]`

- **`quick_run_pipelines.py`** - Quick pipeline execution for testing
  - Fast execution of pipelines for development and testing
  - Streamlined interface for rapid iteration
  - Usage: `python scripts/execution/quick_run_pipelines.py [pipeline_files...]`

## Usage

### Basic Pipeline Execution

```bash
# Run a pipeline with default parameters
python scripts/execution/run_pipeline.py examples/research_minimal.yaml

# Run with custom inputs
python scripts/execution/run_pipeline.py examples/research_minimal.yaml \
  -i topic="artificial intelligence"

# Run with input file and custom output directory
python scripts/execution/run_pipeline.py examples/research_basic.yaml \
  -f inputs.json \
  -o /tmp/output

# Quick execution for testing
python scripts/execution/quick_run_pipelines.py examples/simple_*.yaml
```

### Command-Line Options

#### run_pipeline.py Options

**Positional Arguments:**
- `pipeline` - Path to the YAML pipeline file to execute

**Optional Arguments:**
- `-h, --help` - Show help message and exit
- `-i INPUT, --input INPUT` - Specify input parameters in key=value format (can be used multiple times)
- `-f INPUT_FILE, --input-file INPUT_FILE` - Path to JSON or YAML file containing input parameters
- `-o OUTPUT_DIR, --output-dir OUTPUT_DIR` - Directory where pipeline outputs should be saved
- `--validate` - Validate pipeline configuration before execution
- `--checkpoint-location` - Specify location for execution checkpoints (default: checkpoints/)

### Advanced Usage Examples

```bash
# Multiple inputs
python scripts/execution/run_pipeline.py examples/data_processing_pipeline.yaml \
  -i input_file="data.csv" \
  -i output_format="json" \
  -i filter_threshold=0.5

# Custom output directory with timestamp
python scripts/execution/run_pipeline.py examples/research_basic.yaml \
  -o ./results/$(date +%Y%m%d_%H%M%S)

# Using configuration file
python scripts/execution/run_pipeline.py examples/complex_pipeline.yaml \
  -f production_config.yaml

# Validation mode (dry-run)
python scripts/execution/run_pipeline.py examples/pipeline.yaml \
  --validate

# Custom checkpoint location
python scripts/execution/run_pipeline.py examples/long_running_pipeline.yaml \
  --checkpoint-location ./checkpoints/custom/
```

## Pipeline Execution Features

### Input Handling
- Command-line parameter specification
- JSON and YAML input file support
- Environment variable substitution
- Default value handling
- Input validation and type checking

### Output Management
- Configurable output directories
- Automatic directory creation
- Structured output organization
- Progress tracking and logging
- Result validation and verification

### Error Handling
- Comprehensive error reporting
- Graceful degradation on failures
- Retry mechanisms for transient errors
- Checkpoint-based recovery
- Debug mode for troubleshooting

### Performance Features
- Parallel execution where possible
- Lazy model loading
- Resource optimization
- Memory management
- Progress indicators

## Integration

Execution scripts integrate with:
- Pipeline validation system
- Model management and routing
- Template resolution engine
- Output quality analysis
- Monitoring and logging systems
- Checkpoint and recovery mechanisms

## Development and Testing

For development and testing workflows:

```bash
# Quick test execution
python scripts/execution/quick_run_pipelines.py examples/test_*.yaml

# Development with custom models
ORCHESTRATOR_DEBUG=1 python scripts/execution/run_pipeline.py examples/dev_pipeline.yaml

# Testing with specific model providers
python scripts/execution/run_pipeline.py examples/pipeline.yaml \
  -i model_provider="ollama" \
  -o examples/outputs/test_run
```

## Error Scenarios and Troubleshooting

### Common Error Messages

- **File not found**: Check pipeline file path
- **Invalid YAML**: Validate YAML syntax
- **Missing inputs**: Provide required input parameters
- **Model unavailable**: Check API keys and model availability
- **Output directory**: Ensure write permissions

### Debug Mode

```bash
# Enable debug logging
ORCHESTRATOR_DEBUG=1 python scripts/execution/run_pipeline.py pipeline.yaml

# Verbose output
python scripts/execution/run_pipeline.py pipeline.yaml -v
```

## Environment Variables

Execution scripts respect these environment variables:
- `ORCHESTRATOR_DEBUG` - Enable debug mode
- `ORCHESTRATOR_LOG_LEVEL` - Set logging level
- `ORCHESTRATOR_OUTPUT_DIR` - Default output directory
- Model API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
- `HF_TOKEN` - HuggingFace token for model access