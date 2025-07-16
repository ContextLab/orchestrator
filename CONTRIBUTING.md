# Contributing to py-orc

Thank you for your interest in contributing to py-orc! We welcome contributions from the community and are excited to see what you'll bring to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully
- Prioritize the community's best interests

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/orchestrator.git
   cd orchestrator
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/ContextLab/orchestrator.git
   ```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- System information (OS, Python version, etc.)
- Relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- A clear and descriptive title
- Detailed description of the proposed feature
- Use cases and examples
- Any relevant mockups or diagrams

### Contributing Code

1. **Find an issue** to work on or create a new one
2. **Comment on the issue** to let others know you're working on it
3. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** following our style guidelines
5. **Write or update tests** as needed
6. **Update documentation** if you're changing behavior

## Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the package in development mode**:
   ```bash
   pip install -e ".[dev,docs,notebooks]"
   ```

3. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## Testing

We use pytest for testing. Please ensure all tests pass before submitting a PR.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=orchestrator

# Run specific test file
pytest tests/test_specific.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Follow the naming convention: `test_*.py`
- Write descriptive test names that explain what's being tested
- Include both positive and negative test cases
- Mock external dependencies when appropriate

Example test:

```python
def test_pipeline_execution_with_valid_yaml():
    """Test that pipeline executes successfully with valid YAML input."""
    pipeline = compile_pipeline("valid_pipeline.yaml")
    result = pipeline.run(input_data="test")
    assert result.status == "success"
    assert "output" in result.data
```

## Documentation

Documentation is crucial for the project's success. Please update documentation when:

- Adding new features
- Changing existing behavior
- Fixing bugs that affect user-facing functionality

### Building Documentation

```bash
cd docs_sphinx
make clean
make html
```

View the documentation by opening `docs_sphinx/_build/html/index.html` in your browser.

### Documentation Style

- Use clear, concise language
- Include code examples where appropriate
- Add docstrings to all public functions and classes
- Follow NumPy style for docstrings

## Pull Request Process

1. **Update your fork** with the latest upstream changes:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Rebase your feature branch** if needed:
   ```bash
   git checkout feature/your-feature-name
   git rebase main
   ```

3. **Run all checks**:
   ```bash
   # Run tests
   pytest
   
   # Run linters
   black src tests
   isort src tests
   flake8 src tests
   mypy src
   
   # Build documentation
   cd docs_sphinx && make html
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes
   - Screenshots if applicable

6. **Address review feedback** promptly and professionally

## Style Guidelines

### Python Style

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use type hints where appropriate
- Prefer f-strings for string formatting
- Use descriptive variable names

### Code Organization

```python
# Imports grouped and ordered
import standard_library
import third_party

from orchestrator import local_imports

# Constants
CONSTANT_VALUE = 42

# Classes and functions
class ExampleClass:
    """Clear docstring explaining the class purpose."""
    
    def __init__(self, param: str) -> None:
        """Initialize with clear parameter descriptions."""
        self.param = param
```

### Commit Messages

Follow the conventional commits specification:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Maintenance tasks

Example:
```
feat: Add support for custom model selection strategies

- Implement UCB algorithm for model selection
- Add configuration options for selection parameters
- Update documentation with examples
```

## Community

### Getting Help

- 📚 Check the [documentation](https://orc.readthedocs.io/)
- 🔍 Search existing [issues](https://github.com/ContextLab/orchestrator/issues)
- 💬 Start a [discussion](https://github.com/ContextLab/orchestrator/discussions)
- 📧 Email us at contextualdynamics@gmail.com

### Recognition

Contributors will be recognized in:
- The project's README
- Release notes
- Our documentation

Thank you for contributing to py-orc! Your efforts help make this project better for everyone. 🎉