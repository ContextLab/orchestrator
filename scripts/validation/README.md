# Validation Scripts

This directory contains scripts for validating pipeline configurations, dependencies, and functionality.

## Scripts

### Core Validation Scripts

- **`validate_all_pipelines.py`** - Comprehensive validation of all pipeline configurations
  - Validates YAML syntax, dependencies, and task configurations
  - Provides detailed validation reports with scoring
  - Usage: `python scripts/validation/validate_all_pipelines.py [pipeline_directory]`

- **`quick_validate.py`** - Fast validation of pipeline functionality
  - Quick validation of core pipelines
  - Useful for development and quick checks
  - Usage: `python scripts/validation/quick_validate.py`

- **`validate_and_run_all.py`** - Combined validation and execution
  - Validates pipelines and runs them if validation passes
  - Comprehensive testing approach
  - Usage: `python scripts/validation/validate_and_run_all.py`

### Template and Configuration Validation

- **`validate_template_resolution.py`** - Tests template resolution functionality
  - Validates Jinja2 template rendering
  - Checks variable substitution and filters
  - Usage: `python scripts/validation/validate_template_resolution.py`

### Audit and Analysis Scripts

- **`audit_pipelines.py`** - Audits pipeline configurations and structure
  - Analyzes pipeline architecture and dependencies
  - Identifies potential issues and improvements
  - Usage: `python scripts/validation/audit_pipelines.py`

### Dependency and Tool Validation

- **`verify_tool_names.py`** - Verifies tool name consistency
  - Checks for tool naming conflicts
  - Validates tool references in pipelines
  - Usage: `python scripts/validation/verify_tool_names.py`

- **`check_missing_tools.py`** - Checks for missing tool dependencies
  - Identifies missing tool implementations
  - Validates tool availability
  - Usage: `python scripts/validation/check_missing_tools.py`

### Performance Validation

- **`fast_compile_check.py`** - Quick compilation check for pipelines
  - Fast syntax and structure validation
  - Useful for CI/CD pipelines
  - Usage: `python scripts/validation/fast_compile_check.py`

## Usage Examples

```bash
# Validate all pipelines in examples directory
python scripts/validation/validate_all_pipelines.py examples/

# Quick validation of core functionality
python scripts/validation/quick_validate.py

# Comprehensive validation and execution
python scripts/validation/validate_and_run_all.py

# Check for missing tools
python scripts/validation/check_missing_tools.py

# Audit pipeline configurations
python scripts/validation/audit_pipelines.py examples/
```

## Integration

These validation scripts are designed to work together and integrate with:
- CI/CD pipelines for automated validation
- Development workflows for quality assurance
- Production deployment processes
- Testing frameworks for comprehensive coverage

## Dependencies

Most validation scripts require:
- Core orchestrator framework
- Model initialization (`orchestrator.init_models()`)
- Access to pipeline examples and configuration files
- Proper environment variable setup for model APIs