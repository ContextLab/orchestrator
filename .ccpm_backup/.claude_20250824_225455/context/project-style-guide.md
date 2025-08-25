---
created: 2025-08-22T03:21:33Z
last_updated: 2025-08-22T03:21:33Z
version: 1.0
author: Claude Code PM System
---

# Project Style Guide

## Code Style Standards

### Python Code Style

#### General Principles
- Follow PEP 8 with project-specific exceptions
- Use Python 3.11+ features where beneficial
- Prefer clarity over cleverness
- Write self-documenting code
- Keep functions focused and small

#### Naming Conventions
```python
# Classes: PascalCase
class PipelineExecutor:
    pass

# Functions and methods: snake_case
def execute_pipeline():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRY_COUNT = 3

# Private methods: leading underscore
def _internal_method():
    pass

# Module-level private: leading underscore
_private_variable = "internal"

# Type variables: PascalCase with suffix
ModelType = TypeVar('ModelType')
```

#### Import Organization
```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional

# Related third-party imports
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Local application imports
from orchestrator.core import Pipeline
from orchestrator.models import ModelFactory
from orchestrator.utils import logger
```

#### Type Hints
```python
# Always use type hints for function signatures
def process_data(
    input_data: Dict[str, Any],
    config: Optional[Config] = None
) -> ProcessedResult:
    pass

# Use modern type hints (3.10+)
def parse_items(items: list[str]) -> dict[str, int]:
    pass

# Complex types with TypeAlias
JsonDict: TypeAlias = Dict[str, Union[str, int, float, bool, None]]
```

#### Async Patterns
```python
# Async function naming
async def fetch_data():
    pass

# Async context managers
async with aiohttp.ClientSession() as session:
    response = await session.get(url)

# Gather for parallel execution
results = await asyncio.gather(
    fetch_user(1),
    fetch_user(2),
    fetch_user(3)
)
```

### YAML Pipeline Style

#### Structure
```yaml
# Pipeline metadata at top
id: example_pipeline
name: Example Pipeline
description: A well-structured example pipeline
version: 1.0

# Configuration section
config:
  max_retries: 3
  timeout: 300

# Steps with clear naming
steps:
  - id: fetch_data
    action: fetch
    parameters:
      source: api
      
  - id: process_data
    action: transform
    parameters:
      input: "{{ fetch_data.result }}"
    dependencies: [fetch_data]

# Outputs at the end
outputs:
  processed_data: "{{ process_data.result }}"
```

#### Naming Conventions
- Pipeline IDs: `snake_case`
- Step IDs: `snake_case` with verb_noun pattern
- Variables: `snake_case`
- Actions: `snake_case` verbs

### Documentation Style

#### Docstrings
```python
def complex_function(
    param1: str,
    param2: Optional[int] = None
) -> Dict[str, Any]:
    """
    Brief description of function purpose.
    
    Longer description explaining behavior, edge cases,
    and any important details users should know.
    
    Args:
        param1: Description of first parameter
        param2: Description of optional parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
        ConnectionError: When network issues occur
        
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result["status"])
        "success"
    """
    pass
```

#### Module Documentation
```python
"""
module_name.py

Brief module description.

This module provides functionality for [purpose]. It includes
classes and functions for [main features].

Classes:
    MainClass: Primary class for functionality
    HelperClass: Supporting utilities
    
Functions:
    main_function: Entry point for processing
    helper_function: Utility for data transformation
"""
```

### Comments

#### Inline Comments
```python
# Explain why, not what
result = value * 1.1  # Apply 10% markup for processing overhead

# TODO comments with context
# TODO(username): Optimize this loop for large datasets

# FIXME with issue reference
# FIXME: Issue #123 - Handle edge case for empty input

# NOTE for important information
# NOTE: This must run before database initialization
```

#### Block Comments
```python
# Complex algorithm explanation
# 
# This implements the Frobnicator algorithm for widget processing.
# The algorithm works in three phases:
# 1. Initial parsing and validation
# 2. Transformation using quantum mechanics
# 3. Output generation with error correction
```

### File Structure

#### Python Files
```python
"""Module docstring."""

# Imports (organized by PEP 8)
import standard_library
import third_party
import local_modules

# Constants
DEFAULT_TIMEOUT = 30

# Type definitions
ConfigType = Dict[str, Any]

# Classes (abstract first, then concrete)
class AbstractBase:
    pass

class ConcreteImplementation(AbstractBase):
    pass

# Functions (public then private)
def public_function():
    pass

def _private_function():
    pass

# Module initialization (if needed)
if __name__ == "__main__":
    main()
```

#### Test Files
```python
"""Test module for feature."""

import pytest
from unittest.mock import Mock, patch

from orchestrator.feature import function_to_test

class TestFeature:
    """Test class for Feature."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fixture = create_fixture()
    
    def test_normal_operation(self):
        """Test normal operation case."""
        assert function_to_test(input) == expected
    
    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Error Handling

#### Exception Style
```python
# Specific exceptions with context
class PipelineExecutionError(Exception):
    """Raised when pipeline execution fails."""
    pass

# Raising with helpful messages
raise ValueError(f"Invalid step ID: {step_id}. Must be alphanumeric.")

# Catching specific exceptions
try:
    result = risky_operation()
except ConnectionError as e:
    logger.error(f"Network error: {e}")
    raise
except ValueError as e:
    logger.warning(f"Invalid input: {e}")
    return default_value
```

### Logging Style

```python
import logging

logger = logging.getLogger(__name__)

# Log levels with appropriate context
logger.debug(f"Processing step: {step_id}")
logger.info(f"Pipeline {pipeline_id} started")
logger.warning(f"Retry attempt {attempt}/{max_retries}")
logger.error(f"Failed to execute step {step_id}: {error}")
logger.critical(f"System failure: {critical_error}")
```

### Git Commit Style

#### Commit Message Format
```
type: Brief description (max 50 chars)

Longer explanation of the change, why it was needed,
and any important details. Wrap at 72 characters.

- Bullet points for multiple changes
- Reference issue numbers: #123
- Note breaking changes with BREAKING CHANGE:

Closes #123
```

#### Commit Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### Project-Specific Conventions

#### Tool Implementation
```python
class CustomTool(BaseTool):
    """Tool implementation following base pattern."""
    
    name = "custom_tool"
    description = "Brief tool description"
    
    async def execute(self, **kwargs) -> Any:
        """Execute tool with standard interface."""
        pass
```

#### Model Adapter Pattern
```python
class ProviderAdapter(BaseAdapter):
    """Adapter for external model provider."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        super().__init__(config)
        
    async def generate(self, prompt: str) -> str:
        """Generate response using provider."""
        pass
```

#### Pipeline Step Pattern
```python
@register_action("action_name")
class ActionImplementation(BaseAction):
    """Action implementation with registration."""
    
    async def run(self, context: Context) -> Result:
        """Execute action with context."""
        pass
```

### Quality Standards

#### Code Quality
- No commented-out code in production
- No print statements (use logging)
- No hardcoded values (use constants/config)
- Handle all exceptions appropriately
- Include type hints for all functions

#### Testing Standards
- Minimum 80% code coverage
- Test both success and failure cases
- Use meaningful test names
- Mock external dependencies
- Include integration tests

#### Documentation Standards
- All public APIs must have docstrings
- Include examples in docstrings
- Keep README files updated
- Document breaking changes
- Provide migration guides