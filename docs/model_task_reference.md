# Model Task Reference

This document defines the standardized task names that should be used when specifying model capabilities and requirements in the Orchestrator framework.

## Supported Task Names

The following task names are recognized by the model system:

### Core Tasks
- `generate` - Text generation, completion, and creative writing
- `analyze` - Analysis, summarization, and comprehension tasks  
- `transform` - Text transformation, translation, and reformatting
- `code` - Code generation, analysis, and programming tasks
- `chat` - Conversational and dialog tasks
- `reasoning` - Complex reasoning, logic, and problem-solving
- `complete` - Text completion (legacy, prefer `generate`)
- `instruct` - Instruction following (legacy, prefer `generate`)

### Task Usage in YAML Pipelines

When specifying tasks in YAML pipelines, use the `action` field with these exact names:

```yaml
steps:
  - id: generate_text
    action: generate  # NOT "generation"
    parameters:
      prompt: "Write a story"
      
  - id: analyze_data  
    action: analyze   # NOT "analysis"
    parameters:
      data: "{{input_data}}"
```

### Model Requirements

When specifying model requirements in code or YAML:

```yaml
# Correct
model_requirements:
  tasks: ["generate", "reasoning"]
  
# Incorrect  
model_requirements:
  tasks: ["generation", "reasoning"]  # Wrong: use "generate" not "generation"
```

### Common Mistakes

1. **Using noun forms instead of verb forms**
   - ❌ `generation`, `analysis`, `transformation`
   - ✅ `generate`, `analyze`, `transform`

2. **Using past tense**
   - ❌ `generated`, `analyzed`, `transformed`
   - ✅ `generate`, `analyze`, `transform`

3. **Using compound names**
   - ❌ `text-generation`, `code_generation`
   - ✅ `generate`, `code`

## Model-Specific Task Support

### OpenAI Models
- GPT-4 series: `["generate", "analyze", "transform", "code", "reasoning"]`
- GPT-3.5 series: `["generate", "analyze", "transform", "code"]`

### Anthropic Models  
- Claude 3 series: `["generate", "analyze", "transform", "code", "reasoning"]`
- Legacy models: `["generate", "chat"]`

### Google Models
- Gemini series: `["generate", "analyze", "transform", "code", "reasoning", "vision"]`

### Ollama Models
- Most models: `["generate", "chat", "reasoning", "code"]`
- Smaller models: `["generate", "chat"]`

### HuggingFace Models
- Instruct models: `["generate", "chat", "reasoning"]`
- Base models: `["generate", "complete"]`

## Checking Task Support

To verify if a model supports a specific task:

```python
from orchestrator import get_model_registry

registry = get_model_registry()
model = registry.get_model("gpt-4o-mini")

# Check single task
if model.can_handle_task("generate"):
    print("Model supports text generation")

# Check multiple tasks
required_tasks = ["generate", "reasoning"]
if all(model.can_handle_task(task) for task in required_tasks):
    print("Model supports all required tasks")
```

## Task Aliases (Deprecated)

The following aliases are deprecated and should not be used:
- `generation` → use `generate`
- `completion` → use `generate` 
- `instruct` → use `generate`
- `converse` → use `chat`