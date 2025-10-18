# Quick Start - Claude Skills Orchestrator

## 5-Minute Setup

### 1. Install

```bash
git clone https://github.com/ContextLab/orchestrator.git
cd orchestrator
pip install -e .
```

### 2. Configure API Key

```bash
mkdir -p ~/.orchestrator
echo "ANTHROPIC_API_KEY=your-key-here" >> ~/.orchestrator/.env
```

### 3. Run Your First Pipeline

```bash
# Run the code review example
orchestrator run examples/claude_skills_refactor/01_simple_code_review.yaml
```

That's it! The orchestrator will:
- ✅ Automatically create required skills
- ✅ Use Claude models for processing
- ✅ Generate a code review report

---

## Your First Custom Pipeline

Create `my_pipeline.yaml`:

```yaml
id: my-first-pipeline
name: "My First Pipeline"
version: "1.0.0"

inputs:
  topic:
    type: string
    default: "artificial intelligence"

steps:
  - id: generate_summary
    action: llm_generate
    parameters:
      prompt: "Write a summary about {{ topic }}"
      model: claude-3-5-sonnet-20241022
      max_tokens: 500

  - id: expand_summary
    dependencies: [generate_summary]
    action: llm_generate
    parameters:
      prompt: "Expand on this summary: {{ generate_summary.result }}"
      model: claude-3-5-sonnet-20241022
      max_tokens: 1000
    produces: final_content
    location: "./output.md"
```

Run it:

```bash
orchestrator run my_pipeline.yaml --input topic="machine learning"
```

---

## Key Concepts

### Skills
Reusable capabilities that are automatically created when needed.

```yaml
# Reference a skill - it will be auto-created if it doesn't exist
tool: web-searcher
```

### Steps
Units of work in your pipeline.

```yaml
steps:
  - id: step1           # Unique identifier
    action: llm_generate  # What to do
    parameters:         # Configuration
      prompt: "Task description"
```

### Dependencies
Control execution order.

```yaml
steps:
  - id: step1
    action: llm_generate

  - id: step2
    dependencies: [step1]  # Waits for step1
    action: llm_generate
```

### Models
Choose the right Claude model for each task.

```yaml
model: claude-haiku-4-5        # Fast & cheap
model: claude-sonnet-4-5       # Balanced
model: claude-opus-4.1         # Highest quality
```

---

## Common Patterns

### Sequential Processing

```yaml
steps:
  - id: load_data
    action: llm_generate

  - id: transform_data
    dependencies: [load_data]
    action: llm_generate

  - id: save_results
    dependencies: [transform_data]
    action: llm_generate
```

### Parallel Processing

```yaml
steps:
  # These run in parallel
  - id: task_a
    action: llm_generate

  - id: task_b
    action: llm_generate

  # This waits for both
  - id: combine
    dependencies: [task_a, task_b]
    action: llm_generate
```

### With Skills

```yaml
steps:
  - id: search
    tool: web-searcher  # Auto-created if missing
    action: llm_generate
    parameters:
      query: "{{ search_term }}"
```

---

## Next Steps

1. **Try the examples** in `examples/claude_skills_refactor/`
2. **Read the full guide**: `docs/CLAUDE_SKILLS_USER_GUIDE.md`
3. **Explore skills**: Check `~/.orchestrator/skills/` after running pipelines
4. **Build your pipeline**: Start with a simple sequential workflow
5. **Join discussions**: GitHub issues and discussions

---

## Getting Help

- **Documentation**: `docs/CLAUDE_SKILLS_USER_GUIDE.md`
- **Examples**: `examples/claude_skills_refactor/`
- **Issues**: https://github.com/ContextLab/orchestrator/issues
- **API Reference**: `docs/api_reference.md`

---

**Pro Tip**: Start simple! Begin with a 2-3 step sequential pipeline, then add complexity as you learn the system.