"""Lazy loading wrapper for Ollama models."""

from typing import Any, Dict, Optional

from ..utils.model_utils import check_ollama_model, install_ollama_model
from .ollama_model import OllamaModel


class LazyOllamaModel(OllamaModel):
    """Ollama model that downloads on first use."""

    def __init__(self, model_name: str, **kwargs):
        """Initialize lazy Ollama model without checking availability."""
        self._model_downloaded = False
        self._download_attempted = False
        # Initialize parent without checking availability
        super().__init__(model_name=model_name, **kwargs)
        # Override availability to True initially (we'll check on first use)
        self._is_available = True
    
    def _check_ollama_availability(self) -> None:
        """Override parent's availability check - we check lazily on first use."""
        # Don't check availability during init - we'll do it on first use
        self._is_available = True
    
    def _pull_model(self) -> None:
        """Override parent's pull method - we do this lazily."""
        # Don't pull during init - we'll do it on first use
        pass

    async def _ensure_model_available(self) -> bool:
        """Ensure model is downloaded before use."""
        if self._model_downloaded:
            return True

        if self._download_attempted:
            # Already tried and failed
            return False

        self._download_attempted = True

        # Check if model is already available
        if check_ollama_model(self.model_name):
            self._model_downloaded = True
            return True

        # Try to download the model
        print(
            f">> 📥 Downloading Ollama model: {self.model_name} (this may take a while on first use)"
        )
        if install_ollama_model(self.model_name):
            self._model_downloaded = True
            self._is_available = True
            print(f">> ✅ Successfully downloaded {self.model_name}")
            return True
        else:
            self._is_available = False
            print(f">> ❌ Failed to download {self.model_name}")
            return False

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text, downloading model if needed."""
        if not await self._ensure_model_available():
            raise RuntimeError(
                f"Ollama model {self.model_name} is not available and could not be downloaded"
            )

        return await super().generate(prompt, temperature, max_tokens, **kwargs)

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate structured output, downloading model if needed."""
        if not await self._ensure_model_available():
            raise RuntimeError(
                f"Ollama model {self.model_name} is not available and could not be downloaded"
            )

        return await super().generate_structured(prompt, schema, temperature, **kwargs)

    async def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Chat with model, downloading if needed."""
        if not await self._ensure_model_available():
            raise RuntimeError(
                f"Ollama model {self.model_name} is not available and could not be downloaded"
            )

        return await super().chat(messages, temperature, **kwargs)

    async def health_check(self) -> bool:
        """Check if model is healthy (download if needed)."""
        # For health check, we just check if model can be made available
        # but don't actually download it
        if self._model_downloaded:
            return await super().health_check()

        # Check if model exists locally
        if check_ollama_model(self.model_name):
            self._model_downloaded = True
            return await super().health_check()

        # Model not downloaded yet, but could be - return True
        return True
