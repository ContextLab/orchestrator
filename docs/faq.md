# Orchestrator Framework FAQ

## General Questions

### What is Orchestrator?
Orchestrator is an AI pipeline orchestration framework that provides a unified interface for executing AI workflows defined in YAML. It serves as a wrapper around LangGraph, MCP (Model Context Protocol), and other AI agent control systems.

### How do I get started?
1. Install the framework: `pip install orchestrator-ai`
2. Create a YAML pipeline file
3. Run it with: `orchestrator run my_pipeline.yaml`

### What models are supported?
Orchestrator supports models from:
- OpenAI (gpt-4.1, o3, gpt-03-mini)
- Anthropic (claude-4-opus, claude-4-sonnet, claude-3-7-sonnet)
- Google (gemini-2.5-pro, gemini-2.5-flash)
- Ollama (any model that can run locally)
- HuggingFace (any model that can run locally)

## Technical Questions

### How do AUTO tags work?
AUTO tags allow you to write abstract task descriptions that get resolved into executable prompts by AI models. Example:
```yaml
action: <AUTO>search web for information about {{topic}}</AUTO>
```

### Can I use conditions in pipelines?
Yes! You can use conditions to control execution:
```yaml
condition: "{{score}} > 0.8"
```

### How do I handle errors?
Use the `on_error` field in your steps:
```yaml
on_error:
  action: <AUTO>handle the error gracefully</AUTO>
  continue_on_error: true
```

## Billing Questions

### What are the costs?
The framework itself is free and open source. You only pay for:
- API calls to model providers (OpenAI, Anthropic, etc.)
- Cloud resources if deploying to cloud platforms

### How can I control costs?
- Use the `cache_results` option to avoid redundant API calls
- Choose appropriate models for each task
- Set token limits in your pipeline configuration

## Support

### Where can I get help?
- Documentation: https://orc.readthedocs.io/en/latest/
- GitHub Issues: https://github.com/ContextLab/orchestrator/issues
- GitHub Discussions: https://github.com/ContextLab/orchestrator/discussions

### How do I report bugs?
Please create an issue on our GitHub repository with:
- Your pipeline YAML
- Error messages
- Expected vs actual behavior