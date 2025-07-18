# Model Distribution Across Examples

This document shows which AI models are used in each example pipeline to ensure diverse representation.

## Model Usage by Example

### Google Models
- **research_assistant.yaml** - Google Gemini 1.5 Flash
  - Used for comprehensive research and analysis

### OpenAI Models
- **interactive_chat_bot_demo.yaml** - GPT-4o-mini (for simulated user)
  - Simulates user interactions in chatbot demos

### Anthropic Models
- **interactive_chat_bot_demo.yaml** - Claude 3 Haiku (for bot responses)
  - Powers the chatbot's conversational abilities
- **creative_writing_assistant.yaml** - Claude 3 Sonnet
  - Handles creative writing tasks with nuanced output

### HuggingFace Models
- **content_creation_pipeline.yaml** - Mistral-7B-Instruct-v0.2
  - Generates multi-format content (blog, social, email)

### Ollama Models
- **data_processing_workflow.yaml** - Llama2
  - Analyzes and processes data workflows

## Examples Still Using Default Models

The following examples don't specify a model and will use the default:
- automated_testing_system.yaml
- code_analysis_suite.yaml
- customer_support_automation.yaml
- document_intelligence.yaml
- financial_analysis_bot.yaml
- multi_agent_collaboration.yaml
- scalable_customer_service_agent.yaml

## Recommendations

To ensure complete coverage, consider updating remaining examples:
- **automated_testing_system.yaml** → Could use OpenAI for test generation
- **code_analysis_suite.yaml** → Could use Google for code analysis
- **financial_analysis_bot.yaml** → Could use Anthropic for financial insights
- **document_intelligence.yaml** → Could use HuggingFace for document processing

## How to Specify Models

In any YAML pipeline, add the model specification at the top level:

```yaml
name: "Pipeline Name"
description: "Pipeline description"
model: "provider/model-name"
```

Or specify different models for different steps:

```yaml
steps:
  - id: step1
    action: "..."
    model: "openai/gpt-4"
    
  - id: step2
    action: "..."
    model: "anthropic/claude-3-haiku"
```

This allows mixing models within a single pipeline for optimal results.