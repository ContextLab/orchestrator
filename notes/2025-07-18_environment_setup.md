# Environment Setup Notes

## API Keys
At the start of each session in the orchestrator folder, set environment variables from the .env file:

```bash
export OPENAI_API_KEY="<key from .env>"
export ANTHROPIC_API_KEY="<key from .env>"
export GOOGLE_AI_API_KEY="<key from .env>"
export GOOGLE_API_KEY="<key from .env>"
```

The .env file contains the actual API keys needed for the various model integrations.

## Important Notes
- The .env file should never be committed to git
- Always load these environment variables before running pipelines that use API models
- Some libraries use GOOGLE_API_KEY while others use GOOGLE_AI_API_KEY, so we set both