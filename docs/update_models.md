# Model Registry Update Tool

The `update_models` tool fetches currently available models from the provider
APIs it knows about and updates the `models.yaml` configuration file. Being
queryable here does **not** mean a provider is supported — see the provider
table in the README.

## Overview

The model update tool connects to various AI provider APIs to retrieve their current model offerings and generates a comprehensive configuration file that the orchestrator uses for model selection and management.

## Usage

### Command Line

```bash
python -m orchestrator.tools.update_models [config_path]
```

If no config path is provided, it defaults to `~/.orchestrator/models.yaml`. The
packaged default (`orchestrator/config/models.yaml`, shipped inside the
installed package) is never rewritten; pass `--config <path>` to additionally
update a checked-in copy.

### Programmatic Usage

```python
from orchestrator.tools.update_models import update_models

# Update the default config
await update_models()

# Update a custom config location
await update_models("/path/to/custom/models.yaml")
```

## Providers this tool can query

Being listed here means the tool knows how to ask; it does **not** mean the
provider is supported. See the provider table in the README and
[the product contract](adr/0001-product-contract.md) for what is actually
verified.

1. **OpenAI** - Via OpenAI API (requires `OPENAI_API_KEY`)
   - GPT-4 variants
   - GPT-3.5 variants
   - O1/O3 reasoning models
   - Embedding models

2. **Anthropic** - Via the API's own listing endpoint (`client.models.list()`),
   requires `ANTHROPIC_API_KEY`.

   This used to be a hardcoded list, on the belief that the API offered no
   model listing. It does. The hardcoded list rotted twice — first into
   retired 2024 model ids, then into invented `-latest` aliases that all
   returned 404 — so model ids are now always read from the live API. Do not
   reintroduce a static list here.

3. **Dartmouth Chat** - Via the gateway's OpenAI-compatible `/models`
   endpoint (requires `DARTMOUTH_CHAT_API_KEY`). Pricing comes from the same
   response, which is what makes free/paid classification trustworthy.

4. **Google** - Via Gemini API (requires `GOOGLE_API_KEY`)
   - Gemini Pro variants
   - Gemini Flash variants
   - PaLM models (if available)

5. **Ollama** - Comprehensive hardcoded list
   - Llama variants
   - Mistral variants
   - Code-specific models
   - Various open-source models

6. **HuggingFace** - Curated list of popular models
   - Meta Llama models
   - Microsoft Phi models
   - Qwen models
   - Code generation models

## Configuration Structure

The generated `models.yaml` file contains:

```yaml
models:
  # Model ID to configuration mapping
  openai_gpt-4:
    provider: openai
    type: openai
    size_b: 1760  # Estimated size in billions of parameters
    config:
      model_name: gpt-4
      # Provider-specific config

preferences:
  default: gpt-4o-mini  # Default model for general use
  fallback:  # Fallback chain if default unavailable
    - gpt-3.5-turbo
    - claude-sonnet-4-20250514
    - gemini-1.5-flash

cost_optimized:  # Models optimized for cost
  - gpt-4o-mini
  - claude-sonnet-4-20250514
  - gemini-1.5-flash

performance_optimized:  # Models optimized for quality
  - gpt-4
  - claude-opus-4-20250514
  - gemini-2.5-pro
```

## Model Size Estimation

The tool attempts to estimate model sizes based on:
- Known model sizes (hardcoded for popular models)
- Model name patterns (e.g., "7b", "13b", "175b")
- Mixtral pattern recognition (e.g., "8x7b" = 56B)
- Default fallback of 7B for unknown models

## Error Handling

The tool is resilient to failures:
- If a provider API fails, models from other providers are still fetched
- Missing API keys result in skipping that provider
- Network errors are logged but don't stop the entire update

## Automatic Model Registration

Models added to `models.yaml` through this tool are automatically available for use in the orchestrator. Additionally, if a model not in the configuration is requested at runtime, the orchestrator will attempt to:

1. Auto-register the model based on provider patterns
2. Test the model with a simple prompt
3. Add it to the local `models.yaml` if successful

## Best Practices

1. **Regular Updates**: Run the update tool periodically to get new models
2. **API Keys**: Ensure API keys are set for providers you want to fetch from
3. **Custom Models**: Manually add custom or private models to the generated file
4. **Version Control**: Commit the updated `models.yaml` to track changes

## Example Workflow

```bash
# 1. Set API keys (or add to ~/.orchestrator/.env)
export OPENAI_API_KEY="sk-..."  # Get from https://platform.openai.com/api-keys
export GOOGLE_API_KEY="..."      # Get from https://makersuite.google.com/app/apikey

# 2. Update models
python -m orchestrator.tools.update_models

# 3. Verify the update
cat config/models.yaml | grep "models:" -A 5

# 4. Use in orchestrator
from orchestrator import init_models
registry = init_models()  # Will load the updated configuration
```

## Customization

To add custom models not available through APIs:

1. Run the update tool first
2. Manually edit `models.yaml` to add your custom models
3. Follow the same structure as generated models

Example custom model entry:
```yaml
models:
  my_custom_model:
    provider: huggingface
    type: huggingface
    size_b: 13
    config:
      model_name: myorg/my-custom-model
      device: cuda
      load_in_8bit: true
```