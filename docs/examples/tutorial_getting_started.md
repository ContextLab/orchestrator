# Pipeline Examples Tutorial: Getting Started

This tutorial guides you through running and understanding the pipeline examples, from simple data processing to advanced AI workflows.

## Prerequisites

1. **Orchestrator Installation**: Ensure the Orchestrator framework is installed
2. **API Keys**: Set up required API keys for AI services
3. **Basic Command Line**: Familiarity with terminal/command prompt
4. **Text Editor**: For viewing and modifying YAML files

## Tutorial Path: From Simple to Advanced

### Level 1: Basic Data Processing (5 minutes)

Start with the simplest data processing example to understand core concepts.

#### Step 1: Simple Data Processing
```bash
cd /path/to/orchestrator
python scripts/run_pipeline.py examples/simple_data_processing.yaml
```

**What happens:**
- Reads CSV file from `examples/data/input.csv`
- Filters records where status="active"
- Saves filtered CSV and generates markdown report
- Creates timestamped output files

**Key Concepts Learned:**
- Pipeline YAML structure
- File I/O operations
- Data filtering
- Template variables
- Report generation

**Examine the outputs:**
```bash
ls examples/outputs/simple_data_processing/
cat examples/outputs/simple_data_processing/analysis_report.md
```

#### Step 2: Understanding the Pipeline
Open `examples/simple_data_processing.yaml` to see:
- **Parameters**: Configurable inputs
- **Steps**: Sequential processing actions
- **Dependencies**: Step execution order
- **Templates**: Dynamic content generation

### Level 2: Research and AI Integration (10 minutes)

Learn how to integrate web search and AI analysis.

#### Step 3: Minimal Research
```bash
python scripts/run_pipeline.py examples/research_minimal.yaml \
  -i topic="artificial intelligence ethics"
```

**What happens:**
- Searches web for information on the topic
- Uses AI to analyze and structure findings
- Generates professional research report
- Includes source citations

**Key Concepts Learned:**
- Web search integration
- AI text generation
- JSON response processing
- Template loops and filters
- Source attribution

**Try different topics:**
```bash
python scripts/run_pipeline.py examples/research_minimal.yaml \
  -i topic="quantum computing applications"
```

#### Step 4: AUTO Tags Introduction
```bash
python scripts/run_pipeline.py examples/auto_tags_demo.yaml
```

**What happens:**
- AI makes dynamic decisions about processing
- Models are selected automatically
- Parameters are optimized based on content
- Conditional logic determines execution paths

**Key Concepts Learned:**
- AUTO tags for dynamic decisions
- AI-driven parameter optimization
- Conditional step execution
- Context-aware processing

### Level 3: Control Flow and Parallel Processing (15 minutes)

Explore advanced control structures and parallel execution.

#### Step 5: Batch Processing with For-Each
```bash
python scripts/run_pipeline.py examples/control_flow_for_loop.yaml
```

**What happens:**
- Processes multiple files in parallel
- Uses loop variables (index, is_first, is_last)
- Demonstrates dependency management
- Creates summary of batch results

**Key Concepts Learned:**
- For-each loops
- Parallel processing with max_parallel
- Loop variables and metadata
- Batch operation coordination

#### Step 6: Advanced Fact-Checking
```bash
python scripts/run_pipeline.py examples/fact_checker.yaml \
  -i document_source="examples/data/test_article.md"
```

**What happens:**
- Extracts claims and sources from documents
- Verifies information in parallel
- Uses AUTO tags for dynamic list processing
- Generates professional fact-checking report

**Key Concepts Learned:**
- Structured data extraction
- Parallel verification workflows
- AUTO tags with lists
- Professional reporting

### Level 4: Creative and Advanced Features (20 minutes)

Explore creative AI capabilities and complex workflows.

#### Step 7: Creative Image Generation
```bash
python scripts/run_pipeline.py examples/creative_image_pipeline.yaml \
  -i base_prompt="a futuristic cityscape at sunset"
```

**What happens:**
- Generates images from text prompts
- Creates multiple artistic style variations
- Analyzes generated images with AI
- Builds organized image galleries

**Key Concepts Learned:**
- AI image generation
- Style variation processing
- Image analysis integration
- Gallery documentation automation

#### Step 8: MCP Integration
```bash
python scripts/run_pipeline.py examples/mcp_integration_pipeline.yaml \
  -i search_query="machine learning frameworks"
```

**What happens:**
- Connects to Model Context Protocol servers
- Utilizes external service capabilities
- Manages persistent memory context
- Demonstrates advanced integration patterns

**Key Concepts Learned:**
- External service integration
- MCP server orchestration
- Memory and context management
- Service lifecycle management

## Common Patterns and Best Practices

### 1. Parameter Management
```yaml
parameters:
  input_file:
    type: string
    required: true
    description: Path to input data
  output_path:
    type: string
    default: "outputs/my_pipeline"
    description: Where to save results
```

### 2. Error Handling
```yaml
- id: safe_operation
  action: process_data
  on_error: "continue"  # Don't stop pipeline on failure
  retry_count: 3        # Retry failed operations
```

### 3. Template Usage
```yaml
content: |
  # Report: {{ pipeline.name }}
  Generated: {{ execution.timestamp }}
  Results: {{ step_results | length }} items processed
```

### 4. Dependency Management
```yaml
dependencies:
  - previous_step     # Wait for completion
  - another_step      # Multiple dependencies
```

## Troubleshooting Guide

### Common Issues

#### 1. API Key Errors
```
Error: API key not found
```
**Solution**: Set environment variables for required services
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

#### 2. File Not Found
```
Error: Could not read file 'data/input.csv'
```
**Solution**: Ensure input files exist in correct locations
```bash
ls examples/data/  # Check available files
```

#### 3. Permission Errors
```
Error: Cannot write to output directory
```
**Solution**: Check directory permissions
```bash
chmod 755 examples/outputs/
```

#### 4. Template Errors
```
Error: Undefined variable in template
```
**Solution**: Check variable names and step outputs
- Verify step IDs match template references
- Ensure dependencies are properly defined

### Debugging Tips

1. **Check Logs**: Pipeline execution creates detailed logs
2. **Examine Checkpoints**: Failed pipelines save state in checkpoints/
3. **Start Simple**: Test with minimal parameters first
4. **Validate YAML**: Ensure proper YAML syntax and structure

## Next Steps

### Explore More Examples
- **Error Handling**: `examples/error_handling_examples.yaml`
- **Statistical Analysis**: `examples/statistical_analysis.yaml`
- **Multimodal Processing**: `examples/multimodal_processing.yaml`
- **Interactive Workflows**: `examples/interactive_pipeline.yaml`

### Create Your Own Pipeline
1. Copy an existing example as a template
2. Modify parameters and steps for your use case
3. Test with simple data first
4. Add error handling and validation
5. Document your pipeline thoroughly

### Advanced Topics
- **Custom Tools**: Extend the framework with custom tools
- **Sub-Pipelines**: Compose complex workflows from modules
- **Production Deployment**: Scale pipelines for production use
- **Monitoring**: Add logging and metrics collection

## Resources

- **Main Documentation**: [../README.md](../README.md)
- **API Reference**: [../api/](../api/)
- **Example Documentation**: [README.md](README.md)
- **Troubleshooting**: [../advanced/troubleshooting.rst](../advanced/troubleshooting.rst)

## Community

- Share your pipelines and examples
- Report issues and contribute improvements  
- Collaborate on advanced use cases
- Help others learn pipeline development

This tutorial provides a structured path from basic concepts to advanced pipeline development. Each level builds on previous knowledge while introducing new capabilities and patterns.