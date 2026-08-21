"""Model integrations.

The provider adapters that lived here (Anthropic, OpenAI, Google, Ollama,
local HuggingFace) were retired under #430. The supported providers are
Dartmouth Chat (:mod:`orchestrator.models.providers.dartmouth_provider`) and
the HuggingFace Inference API (#484). What remains is provider-independent
support code.
"""
