# Getting Started with Orchestrator

This tutorial will take you from zero to your first working pipeline in just a few minutes. You'll learn the core concepts of Orchestrator and build a simple but powerful pipeline.

## What You'll Build

By the end of this tutorial, you'll have created a pipeline that:
- Takes user input about a topic
- Uses an AI model to generate content about that topic
- Saves the result to a file
- Demonstrates the key concepts of Orchestrator

## Prerequisites

- Python 3.8 or higher
- Basic familiarity with YAML syntax
- Orchestrator installed (`pip install orchestrator-framework`)

## Step 1: Understanding Orchestrator Basics

Orchestrator is a framework for building AI-powered automation pipelines. Think of it as a way to chain together different tasks (like calling AI models, processing files, or running code) in a structured, reliable way.

**Key concepts:**
- **Pipeline**: A YAML file that defines a sequence of tasks
- **Task**: A single step in your pipeline (like calling an AI model)
- **Template**: Dynamic content that gets filled in at runtime
- **Variables**: Data that flows between tasks

## Step 2: Your First Pipeline

Create a file called `my_first_pipeline.yaml`:

```yaml
name: "My First Pipeline"
description: "A simple content generation pipeline"
version: "1.0.0"

# Define variables that can be passed in when running the pipeline
input_variables:
  topic:
    type: string
    description: "The topic to generate content about"
    default: "artificial intelligence"

# Define the sequence of tasks
tasks:
  - name: "generate_content"
    type: "llm_task"
    model: "gpt-3.5-turbo"  # You can change this to any model you have access to
    prompt: |
      Write a brief, informative paragraph about {{ topic }}.
      Make it engaging and accessible to a general audience.
    max_tokens: 200
    
  - name: "save_content"
    type: "python_task"
    script: |
      import os
      from datetime import datetime
      
      # Get the generated content from the previous task
      content = context.get_task_output('generate_content')
      
      # Create a filename with timestamp
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      filename = f"content_{timestamp}.txt"
      
      # Save to file
      with open(filename, 'w') as f:
          f.write(f"Topic: {{ topic }}\n\n")
          f.write(content)
          f.write(f"\n\nGenerated on: {datetime.now()}")
      
      print(f"Content saved to: {filename}")
      return {"filename": filename, "content": content}

# Define what the pipeline outputs
output:
  filename: "{{ save_content.filename }}"
  content: "{{ save_content.content }}"
```

## Step 3: Understanding the Pipeline Structure

Let's break down what each part does:

### Header Information
```yaml
name: "My First Pipeline"
description: "A simple content generation pipeline"
version: "1.0.0"
```
This metadata helps you identify and version your pipelines.

### Input Variables
```yaml
input_variables:
  topic:
    type: string
    description: "The topic to generate content about"
    default: "artificial intelligence"
```
This defines what inputs your pipeline accepts. Users can provide a topic when running the pipeline, or it will use the default.

### Tasks
The `tasks` section defines what your pipeline actually does:

1. **LLM Task**: Calls an AI model with a prompt
2. **Python Task**: Runs custom Python code to save the result

### Template Variables
Notice the `{{ topic }}` syntax - this is template substitution. The value gets filled in at runtime.

### Task Dependencies
The second task uses `context.get_task_output('generate_content')` to access the output from the first task. This creates an implicit dependency chain.

## Step 4: Running Your Pipeline

### Using the Python API

Create a file called `run_pipeline.py`:

```python
import asyncio
from orchestrator.api import PipelineAPI

async def main():
    # Create the API instance
    api = PipelineAPI(development_mode=True)
    
    # Compile the pipeline
    pipeline = await api.compile_pipeline("my_first_pipeline.yaml")
    
    # Execute with custom input
    result = await api.execute_pipeline(
        pipeline, 
        inputs={"topic": "renewable energy"}
    )
    
    print("Pipeline completed!")
    print(f"Generated file: {result['filename']}")
    print(f"Content preview: {result['content'][:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python run_pipeline.py
```

### Using the Command Line

You can also run pipelines directly:

```bash
# Run with default topic
orchestrator run my_first_pipeline.yaml

# Run with custom topic
orchestrator run my_first_pipeline.yaml --input topic="machine learning"

# Run with verbose output
orchestrator run my_first_pipeline.yaml --input topic="robotics" --verbose
```

## Step 5: Understanding the Output

When your pipeline runs, you should see:
1. Console output showing the pipeline progress
2. A new text file created with your generated content
3. Return values from the pipeline execution

Example output file (`content_20241231_143022.txt`):
```
Topic: renewable energy

Renewable energy represents one of the most promising solutions to our global climate crisis. By harnessing natural resources like sunlight, wind, and flowing water, renewable technologies provide clean electricity without the harmful emissions of fossil fuels...

Generated on: 2024-12-31 14:30:22
```

## Step 6: Exploring Further

### Modifying Your Pipeline

Try these modifications to learn more:

**Add error handling:**
```yaml
- name: "generate_content"
  type: "llm_task"
  model: "gpt-3.5-turbo"
  prompt: |
    Write a brief, informative paragraph about {{ topic }}.
    Make it engaging and accessible to a general audience.
  max_tokens: 200
  error_handling:
    retry_count: 3
    fallback_response: "Unable to generate content about {{ topic }} at this time."
```

**Add conditional logic:**
```yaml
- name: "validate_content"
  type: "python_task"
  script: |
    content = context.get_task_output('generate_content')
    word_count = len(content.split())
    
    if word_count < 10:
        raise ValueError("Generated content is too short")
    
    return {"word_count": word_count, "is_valid": True}
```

**Chain multiple AI tasks:**
```yaml
- name: "generate_summary"
  type: "llm_task" 
  model: "gpt-3.5-turbo"
  prompt: |
    Create a one-sentence summary of this content:
    
    {{ generate_content }}
  depends_on: ["generate_content"]
```

## Common Patterns You've Learned

1. **Variable templating**: Using `{{ variable_name }}` to insert dynamic content
2. **Task dependencies**: Using outputs from previous tasks
3. **Mixed task types**: Combining LLM calls with Python code
4. **Error handling**: Making pipelines robust with retries and fallbacks
5. **File I/O**: Reading and writing data within pipelines

## Next Steps

Now that you've mastered the basics:

1. **Explore more task types**: Try `web_search_task`, `file_task`, `api_task`
2. **Learn control flow**: Implement loops and conditionals
3. **Study the examples**: Check out the 43 example pipelines in `/docs/tutorials/pipelines/`
4. **Follow the learning path**: Progress through the structured learning modules
5. **Build something real**: Apply these concepts to solve an actual problem

## Troubleshooting

### Common Issues

**"Model not found" errors:**
- Make sure you have API keys configured for your chosen model
- Try using `model: "AUTO"` to let Orchestrator choose an available model

**Template errors:**
- Check that variable names in `{{ }}` match your input_variables
- Use `orchestrator validate my_first_pipeline.yaml` to check syntax

**Import errors:**
- Ensure all required packages are installed
- Check Python version compatibility (3.8+)

**File permission errors:**
- Make sure the directory is writable
- Try running from a directory where you have write permissions

## Key Takeaways

- Orchestrator pipelines are YAML files that define task sequences
- Tasks can be AI model calls, Python code, or many other types
- Variables and templates make pipelines flexible and reusable
- The Python API provides programmatic control over pipeline execution
- Error handling and validation make pipelines production-ready

You now have the foundation to build more complex pipelines. The concepts you've learned here - variables, tasks, dependencies, and templates - are the building blocks for any Orchestrator pipeline, no matter how sophisticated.

Ready for the next level? Check out [Advanced Patterns](advanced-patterns.md) to learn about loops, conditionals, and complex data flows.