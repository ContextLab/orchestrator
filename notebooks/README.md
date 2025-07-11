# Orchestrator Framework Tutorials

Welcome to the Orchestrator framework tutorial collection! These Jupyter notebooks will guide you through the core concepts and advanced features of the framework.

## 📚 Tutorial Overview

### [01_getting_started.ipynb](01_getting_started.ipynb)
**Introduction to Orchestrator Framework**

- Core concepts: Tasks, Pipelines, Models, Orchestrator
- Creating and executing your first pipeline
- Working with mock models for development
- State management and checkpointing
- Pipeline progress monitoring
- Basic serialization and deserialization

**Prerequisites:** Basic Python knowledge  
**Duration:** 30-45 minutes  
**Difficulty:** Beginner

### [02_yaml_configuration.ipynb](02_yaml_configuration.ipynb)
**YAML Configuration and Template System**

- Defining workflows declaratively in YAML
- Template variables and AUTO resolution
- Pipeline compilation from YAML
- Advanced YAML features and validation
- Best practices for YAML workflow design
- Error handling and debugging

**Prerequisites:** Complete tutorial 01  
**Duration:** 45-60 minutes  
**Difficulty:** Intermediate

### [03_advanced_model_integration.ipynb](03_advanced_model_integration.ipynb)
**Multi-Model Orchestration and Optimization**

- Model capabilities and requirements
- Intelligent model selection algorithms
- Fallback strategies and error handling
- Performance monitoring and cost analysis
- Load balancing and optimization
- Real-world integration patterns

**Prerequisites:** Complete tutorials 01-02  
**Duration:** 60-75 minutes  
**Difficulty:** Advanced

## 🚀 Getting Started

### Prerequisites

1. **Python Environment**: Python 3.8+ with Jupyter notebooks support
2. **Dependencies**: Install the orchestrator package and dependencies

```bash
# Install the package
pip install -e .

# Install Jupyter (if not already installed)
pip install jupyter

# Start Jupyter
jupyter notebook
```

### Running the Tutorials

1. **Clone the repository** and navigate to the notebooks directory
2. **Start Jupyter** in your terminal: `jupyter notebook`
3. **Open the first tutorial**: `01_getting_started.ipynb`
4. **Follow the tutorials in order** for the best learning experience

### Development Setup

If you're running from the source repository:

```python
# Add the src directory to your Python path (included in notebooks)
import sys
sys.path.insert(0, '../src')
```

## 📖 Learning Path

### Beginner Path
1. Start with **Getting Started** to understand core concepts
2. Try **YAML Configuration** to learn declarative workflow design
3. Practice with the provided examples and create your own simple workflows

### Advanced Path
1. Complete all three tutorials in order
2. Explore the **Advanced Model Integration** for production scenarios
3. Review the integration tests in `/tests/integration_*` for real API examples
4. Check out the examples in `/examples/` directory

### Production Path
1. Complete all tutorials
2. Study the integration tests for real API implementations
3. Review the framework source code for deep understanding
4. Implement custom model adapters for your specific needs

## 🛠️ Working with Real Models

The tutorials use mock models for demonstration. To work with real AI models:

### OpenAI Integration
```python
from orchestrator.integrations.openai_model import OpenAIModel

model = OpenAIModel(
    name="gpt-4",
    api_key="your-openai-api-key",
    model="gpt-4"
)
```

### Anthropic Integration
```python
from orchestrator.integrations.anthropic_model import AnthropicModel

model = AnthropicModel(
    name="claude-3-sonnet",
    api_key="your-anthropic-api-key", 
    model="claude-3-sonnet-20240229"
)
```

### Local Models
```python
from orchestrator.integrations.huggingface_model import HuggingFaceModel

model = HuggingFaceModel(
    name="llama-7b",
    model_path="meta-llama/Llama-2-7b-chat-hf"
)
```

**Note:** Real model integration requires API keys and additional dependencies. See the integration tests for complete examples.

## 📋 Tutorial Contents Summary

| Tutorial | Core Topics | Key Takeaways |
|----------|-------------|---------------|
| **Getting Started** | Basic workflow creation, task dependencies, mock models | Understanding the orchestration mindset |
| **YAML Configuration** | Declarative workflows, templates, validation | Separating logic from configuration |
| **Advanced Models** | Multi-model selection, optimization, monitoring | Production-ready orchestration |

## 🔧 Troubleshooting

### Common Issues

**Import Errors**
```python
# Make sure the src path is correctly added
import sys
sys.path.insert(0, '../src')
```

**Mock Model Responses**
```python
# Mock models require explicit response configuration
model.set_response("your prompt", "expected response")
```

**Async/Await Issues**
```python
# Use await in Jupyter notebook cells
result = await orchestrator.execute_pipeline(pipeline)
```

### Getting Help

- **Documentation**: Check the main README.md for framework overview
- **API Reference**: Explore the source code docstrings
- **Integration Tests**: See `/tests/integration_*` for real-world examples
- **Issues**: Report problems on the project GitHub repository

## 🎯 Next Steps

After completing these tutorials:

1. **Explore Examples**: Check the `/examples/` directory for more complex workflows
2. **Integration Testing**: Run the integration tests to see real API usage
3. **Custom Development**: Create your own models and adapters
4. **Production Deployment**: Study deployment patterns and monitoring
5. **Contribute**: Consider contributing improvements or new features

## 🏆 Best Practices Learned

By completing these tutorials, you'll understand:

- ✅ **Separation of Concerns**: Keep workflow logic separate from execution details
- ✅ **Model Abstraction**: Write workflows that work with multiple AI providers
- ✅ **Error Handling**: Build robust workflows with proper fallback strategies
- ✅ **Cost Optimization**: Balance performance, cost, and reliability
- ✅ **Monitoring**: Track and optimize workflow performance
- ✅ **Configuration Management**: Use YAML for maintainable workflows

Happy learning and orchestrating! 🎵🚀