# Orchestrator Framework Examples

This directory contains sample pipeline configurations and test data for the Orchestrator Framework.

**📍 Main Examples Moved**: Most tutorial examples have been moved to `docs/tutorials/examples/` for better organization and documentation.

## Current Contents

### Pipeline Definitions (`pipelines/`)

Contains YAML pipeline configurations used for testing and demonstrations:
- `research-report-template.yaml` - Template-based research pipeline with AUTO tags
- `code_optimization.yaml` - Code analysis and optimization pipeline  
- `data_processing.yaml` - Data ingestion, cleaning, and processing pipeline
- `simple_research.yaml` - Basic research workflow
- `research_assistant.yaml` - Research assistant pipeline
- `research_report.yaml` - Report generation pipeline

### Test Data (`test_data/`)

Real data files used for integration testing:
- `sample_code.py` - Python code for analysis testing
- `sample_data.csv` - CSV data for processing pipelines
- `customers.json` - JSON data for validation testing
- `malformed_data.json` - Malformed data for error handling tests
- `small_dataset.csv` - Small dataset for quick tests

### Integration Examples

- `research_assistant_with_report.py` - Complete research assistant with PDF generation
- `simple_pipeline.yaml` - Basic pipeline for getting started
- `multi_model_pipeline.yaml` - Advanced multi-model pipeline
- `model_requirements_pipeline.yaml` - Pipeline with specific model requirements

### Control Systems

- `research_control_system.py` - Control system for research workflows
- `tool_integrated_control_system.py` - Tool-integrated control system

*Note: These control systems have been moved to `src/orchestrator/control_systems/` for proper module organization.*

## Running Examples

### From Documentation

The main tutorial examples are now in `docs/tutorials/examples/`. To run them:

```bash
cd docs/tutorials/examples/
python research_assistant_with_pdf.py
```

### Pipeline Testing

To test pipelines with real data:

```bash
# Run integration tests
pytest tests/integration/test_real_world_pipelines.py -v

# Test specific pipeline
python -m orchestrator run examples/pipelines/simple_research.yaml \
    --context topic="machine learning"
```

### Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

## Documentation

For complete tutorials and guides, see:
- [User Guide](../docs/user_guide/index.rst) - Getting started
- [Tutorial Examples](../docs/tutorials/examples.rst) - Step-by-step tutorials
- [API Reference](../docs/api/index.rst) - Detailed API documentation

## Contributing

When adding new examples:
1. Add pipeline YAML files to `pipelines/`
2. Add test data to `test_data/` if needed
3. Create comprehensive tutorials in `docs/tutorials/examples/`
4. Add integration tests in `tests/integration/`
5. Update documentation

For more information, see [CONTRIBUTING.md](../CONTRIBUTING.md).