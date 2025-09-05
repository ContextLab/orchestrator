---
created: 2025-08-22T03:21:33Z
last_updated: 2025-08-22T03:21:33Z
version: 1.0
author: Claude Code PM System
---

# Project Structure

## Root Directory Organization

```
orchestrator/
├── src/                      # Main source code
│   └── orchestrator/         # Core package
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py           # Command-line interface
│       ├── actions/          # Action implementations
│       ├── adapters/         # Model adapters
│       ├── analytics/        # Analytics and monitoring
│       ├── auto_resolution/  # Automatic ambiguity resolution
│       ├── checkpointing/    # State persistence
│       ├── compiler/         # Pipeline compilation
│       ├── control_flow/     # Control flow implementations
│       ├── control_systems/  # System controls
│       ├── core/            # Core abstractions
│       ├── engine/          # Execution engine
│       ├── execution/       # Execution contexts
│       ├── executor/        # Task executors
│       ├── graph_generation/ # Pipeline graph generation
│       ├── integrations/    # External integrations
│       ├── intelligence/    # AI intelligence layer
│       ├── io/              # Input/output handling
│       ├── llm_tools/       # LLM tool integrations
│       ├── models/          # Model implementations
│       ├── monitoring/      # System monitoring
│       ├── orchestration/   # Orchestration logic
│       ├── sanitization/    # Input sanitization
│       ├── schemas/         # Data schemas
│       ├── tools/           # Tool implementations
│       ├── transformations/ # Data transformations
│       ├── utils/           # Utility functions
│       └── validation/      # Validation logic
├── tests/                   # Test suite
│   ├── integration/         # Integration tests
│   ├── fixtures/           # Test fixtures
│   ├── models/             # Model tests
│   ├── local/              # Local tests
│   └── test_helpers/       # Test utilities
├── docs/                    # Documentation
│   ├── user_guide/         # User documentation
│   ├── development/        # Developer documentation
│   ├── architecture/       # Architecture docs
│   ├── features/           # Feature documentation
│   ├── tools/              # Tool documentation
│   └── advanced/           # Advanced topics
├── config/                  # Configuration files
│   ├── models.yaml         # Model configurations
│   └── orchestrator.yaml   # System configuration
├── examples/               # Example pipelines
│   ├── outputs/           # Example outputs
│   └── pipelines/         # Example pipeline definitions
├── scripts/               # Utility scripts
│   └── run_pipeline.py    # Pipeline runner script
├── checkpoints/           # Pipeline checkpoints
└── .claude/              # Claude Code PM system
    ├── context/          # Project context
    ├── scripts/          # PM scripts
    └── rules/            # Project rules

```

## Key Directories

### Source Code (`src/orchestrator/`)
- **Core Package**: Main orchestrator implementation
- **Modular Architecture**: Clear separation of concerns
- **Tool System**: Extensive tool implementations in `tools/`
- **Model Support**: Multiple model adapters in `adapters/`
- **Control Flow**: Advanced control flow in `control_flow/`

### Testing (`tests/`)
- **Comprehensive Coverage**: Integration, unit, and fixture tests
- **Test Helpers**: Utilities for testing pipelines
- **Model Tests**: Specific tests for model implementations
- **Local Tests**: Tests that run without external dependencies

### Documentation (`docs/`)
- **User Guide**: End-user documentation
- **Development Guide**: Developer documentation
- **Architecture**: System design documentation
- **Feature Documentation**: Detailed feature descriptions

### Configuration (`config/`)
- **Model Configuration**: Model settings and parameters
- **System Configuration**: Core system settings

### Examples (`examples/`)
- **Pipeline Examples**: Sample pipeline definitions
- **Output Examples**: Example pipeline outputs

## File Naming Patterns

### Python Files
- **Snake Case**: `file_name.py`
- **Test Files**: `test_*.py` or `*_test.py`
- **Private Modules**: Leading underscore `_private.py`

### YAML Files
- **Pipeline Definitions**: `*_pipeline.yaml` or descriptive names
- **Test Pipelines**: `test_*.yaml`
- **Configuration**: `*.yaml` in config directory

### Documentation
- **Markdown**: `*.md` files
- **Kebab Case**: `feature-name.md` for documentation

## Module Organization

### Core Modules
- `core/`: Base classes and interfaces
- `engine/`: Pipeline execution engine
- `executor/`: Task execution logic
- `compiler/`: Pipeline compilation from YAML

### Feature Modules
- `actions/`: Implementable actions
- `tools/`: Executable tools
- `models/`: Model implementations
- `adapters/`: Model provider adapters

### Support Modules
- `utils/`: Utility functions
- `validation/`: Input validation
- `sanitization/`: Security and sanitization
- `monitoring/`: Performance monitoring

## Special Directories

### Checkpoints (`checkpoints/`)
- Stores pipeline execution state
- JSON format with timestamps
- Pattern: `{pipeline_name}_{timestamp}_{timestamp}.json`

### Claude PM System (`.claude/`)
- Project management integration
- Context documentation
- Custom scripts and rules

## Build and Distribution
- **Package Name**: `py-orc`
- **Build System**: setuptools with pyproject.toml
- **Python Version**: >=3.11
- **Distribution**: PyPI package