# Basic Examples

This directory contains simple, foundational pipeline examples that demonstrate core orchestrator concepts. These examples are perfect for learning the fundamentals and testing your installation.

## Examples Overview

### 🌟 [hello_world.yaml](hello_world.yaml)
**Perfect Starting Point**
- The simplest possible pipeline
- Single step text generation
- Basic parameter usage
- Great for testing your setup

```bash
# Run with default greeting
python scripts/execution/run_pipeline.py examples/basic/hello_world.yaml

# Custom greeting
python scripts/execution/run_pipeline.py examples/basic/hello_world.yaml -i name="Alice"
```

### 📝 [text_analysis.yaml](text_analysis.yaml)
**Text Processing Fundamentals**
- Analyze text for sentiment and themes
- Structured JSON output
- Multiple analysis types
- Word counting

```bash
# Analyze text sentiment
python scripts/execution/run_pipeline.py examples/basic/text_analysis.yaml \
  -i text="I love using this orchestrator system!" \
  -i analysis_type="sentiment"

# Comprehensive analysis
python scripts/execution/run_pipeline.py examples/basic/text_analysis.yaml \
  -i text="The future of AI is bright with many exciting developments ahead." \
  -i analysis_type="comprehensive"
```

### 🔍 [simple_research.yaml](simple_research.yaml)
**Multi-Step Workflows**
- Web search integration
- Step dependencies
- File output generation
- Template usage

```bash
# Research any topic
python scripts/execution/run_pipeline.py examples/basic/simple_research.yaml \
  -i topic="quantum computing" \
  -i max_sources=5

# Quick research with fewer sources
python scripts/execution/run_pipeline.py examples/basic/simple_research.yaml \
  -i topic="machine learning trends" \
  -i max_sources=3
```

### 🔄 [data_transformation.yaml](data_transformation.yaml)
**Data Processing Basics**
- Structured data manipulation
- Data validation and cleaning
- Multiple transformation types
- JSON output formatting

```bash
# Clean and standardize data
python scripts/execution/run_pipeline.py examples/basic/data_transformation.yaml \
  -i transformation_type="clean"

# Generate data summaries
python scripts/execution/run_pipeline.py examples/basic/data_transformation.yaml \
  -i transformation_type="summarize"

# Categorize records
python scripts/execution/run_pipeline.py examples/basic/data_transformation.yaml \
  -i transformation_type="categorize"
```

### 🔀 [conditional_logic.yaml](conditional_logic.yaml)
**Dynamic Workflows**
- Conditional step execution
- User-specific content generation
- Dynamic parameter usage
- Multi-path processing

```bash
# Admin user with analytics
python scripts/execution/run_pipeline.py examples/basic/conditional_logic.yaml \
  -i user_type="admin" \
  -i include_analytics=true \
  -i content_length="long"

# Guest user, short content
python scripts/execution/run_pipeline.py examples/basic/conditional_logic.yaml \
  -i user_type="guest" \
  -i include_analytics=false \
  -i content_length="short"

# Standard user, medium content
python scripts/execution/run_pipeline.py examples/basic/conditional_logic.yaml \
  -i user_type="standard" \
  -i content_length="medium"
```

## Learning Path

**Recommended order for new users:**

1. **hello_world.yaml** - Test your setup and learn basic syntax
2. **text_analysis.yaml** - Understand parameters and output handling  
3. **simple_research.yaml** - Learn multi-step workflows and dependencies
4. **data_transformation.yaml** - Explore structured data processing
5. **conditional_logic.yaml** - Master dynamic and conditional workflows

## Key Concepts Demonstrated

### 📋 **Basic Pipeline Structure**
- Pipeline metadata (id, name, description)
- Parameter definitions with types and defaults
- Step definitions with actions and parameters
- Output specifications
- Dependencies between steps

### 🔧 **Core Features**
- **AUTO tags** - Intelligent model selection
- **Template syntax** - Jinja2 templating with `{{ }}`
- **Dependencies** - Controlling step execution order
- **Conditions** - Conditional step execution
- **Tool integration** - Web search, filesystem operations
- **Structured output** - JSON formatting and validation

### 💡 **Best Practices**
- Clear, descriptive step IDs
- Meaningful parameter names and descriptions
- Proper dependency chains
- Error-resistant template usage
- Helpful metadata for documentation

## Requirements

All basic examples require:
- **Text Generation Model** - Any model capable of text generation
- **Web Search Tool** - For research examples (web-search)
- **Filesystem Tool** - For file operations (filesystem)

See the main [models configuration](../config/models.yaml) for setup details.

## Troubleshooting

### Common Issues

**Pipeline fails to start:**
- Check that models are properly initialized: `init_models()`
- Verify required tools are available
- Ensure parameter types match expected formats

**Template errors:**
- Check for typos in variable names: `{{ variabel }}` vs `{{ variable }}`
- Verify step IDs match dependency references
- Ensure proper JSON formatting for structured data

**Model selection issues:**
- Confirm at least one text generation model is available
- Check API keys for cloud models
- Consider using local models (Ollama) for testing

### Getting Help

- Check the [troubleshooting guide](../../docs/troubleshooting.md)
- Review [pipeline debugging tips](../../docs/debugging.md)  
- See [configuration documentation](../../docs/configuration.md)

## Next Steps

Once comfortable with basic examples, explore:
- **[Advanced Examples](../advanced/)** - Complex workflows and patterns
- **[Integration Examples](../integrations/)** - External service integrations
- **[Migration Examples](../migration/)** - Upgrading from older versions
- **[Platform Examples](../platform/)** - Cross-platform considerations