---
created: 2025-08-22T03:21:33Z
last_updated: 2025-08-22T03:21:33Z
version: 1.0
author: Claude Code PM System
---

# Technology Context

## Language and Runtime
- **Primary Language**: Python
- **Python Version**: >=3.11 (supports 3.11, 3.12, 3.13)
- **Package Manager**: pip
- **Build System**: setuptools with pyproject.toml

## Core Dependencies

### AI/ML Frameworks
- **OpenAI**: `openai>=1.0.0` - GPT model integration
- **Anthropic**: `anthropic>=0.7.0` - Claude model integration
- **LangChain**: `langchain>=0.1.0`, `langchain-core>=0.1.0` - Structured outputs and tool chains
- **Google AI**: Integration for Gemini models
- **HuggingFace**: Support through transformers library
- **Ollama**: Local model support

### Web and Network
- **aiohttp**: `>=3.8.0` - Async HTTP client/server
- **requests**: `>=2.28.0` - HTTP library
- **httpx**: `>=0.25.0` - Modern HTTP client
- **beautifulsoup4**: `>=4.11.0` - Web scraping
- **playwright**: `>=1.40.0` - Browser automation
- **ddgs**: `>=9.0.0` - DuckDuckGo search
- **lxml**: `>=4.9.0` - XML/HTML processing
- **urllib3**: `>=2.0.0` - HTTP library

### Data Processing
- **numpy**: `>=1.21.0` - Numerical computing
- **pandas**: Data manipulation and analysis
- **pydantic**: `>=2.0.0` - Data validation
- **jinja2**: `>=3.0.0` - Template engine
- **pyyaml**: `>=6.0` - YAML parsing
- **jsonschema**: `>=4.0.0` - JSON schema validation

### Infrastructure
- **docker**: `>=6.0.0` - Container management
- **redis**: `>=4.0.0` - Caching and message broker
- **psycopg2-binary**: `>=2.9.0` - PostgreSQL adapter
- **asyncio-mqtt**: `>=0.13.0` - MQTT client
- **cryptography**: `>=3.4.0` - Cryptographic operations

### Multimedia Processing
- **opencv-python**: Video and image processing
- **librosa**: Audio analysis and processing
- **Pillow**: Image manipulation
- **moviepy**: Video editing

### Development Tools
- **pytest**: Testing framework
- **ruff**: Python linter and formatter
- **mypy**: Static type checker
- **coverage**: Code coverage measurement
- **flake8**: Style guide enforcement

## Optional Dependencies

### Feature Groups
```toml
[ollama] - Local Ollama model support
[cloud] - Cloud provider integrations
[dev] - Development and testing tools
[all] - Complete installation with all features
```

## API Key Management
- Secure storage in `~/.orchestrator/.env`
- File permissions: 600 (owner read/write only)
- Interactive setup via CLI: `orchestrator keys setup`
- Supports: OpenAI, Anthropic, Google, HuggingFace

## Model Support

### Cloud Models
- **OpenAI**: GPT-3.5, GPT-4, GPT-4 Turbo
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
- **Google**: Gemini Pro, Gemini Ultra
- **HuggingFace**: Various models via API

### Local Models
- **Ollama**: Llama, Mistral, Phi, others
- **HuggingFace Local**: Downloaded and cached models
- Lazy loading: Models downloaded only when needed

## Architecture Patterns

### Design Patterns
- **Factory Pattern**: Model and tool creation
- **Strategy Pattern**: Model selection
- **Observer Pattern**: Event monitoring
- **Command Pattern**: Action execution
- **Builder Pattern**: Pipeline construction

### Async Architecture
- Async/await throughout codebase
- Concurrent task execution
- Resource pooling for models
- Event-driven execution

## File Formats

### Configuration
- **YAML**: Pipeline definitions (`.yaml`)
- **JSON**: Checkpoints and state (`.json`)
- **TOML**: Project configuration (`pyproject.toml`)
- **ENV**: Environment variables (`.env`)

### Data Storage
- **JSON**: Checkpoint persistence
- **Parquet**: Efficient data storage
- **CSV**: Data import/export
- **Markdown**: Documentation and reports

## Security Features
- Sandboxed code execution
- Input sanitization
- API key encryption
- Docker isolation for untrusted code
- Secure tool execution with validation

## Performance Optimizations
- Lazy model loading
- Checkpoint-based recovery
- Parallel task execution
- Resource pooling
- Caching strategies with Redis

## Monitoring and Analytics
- Built-in performance monitoring
- Analytics pipeline for usage tracking
- Error tracking and reporting
- Resource usage monitoring
- Execution time profiling

## Testing Infrastructure
- Comprehensive test suite
- Integration tests for all providers
- Mock services for offline testing
- Fixture-based test data
- Continuous integration via GitHub Actions

## Distribution
- **Package Name**: `py-orc`
- **PyPI**: Published package
- **Version**: 0.1.0 (Alpha)
- **License**: MIT
- **Documentation**: ReadTheDocs integration