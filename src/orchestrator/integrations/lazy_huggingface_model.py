"""Lazy loading wrapper for HuggingFace models."""

from .huggingface_model import HuggingFaceModel


class LazyHuggingFaceModel(HuggingFaceModel):
    """HuggingFace model that downloads on first use."""

    def __init__(self, model_name: str, **kwargs):
        """Initialize lazy HuggingFace model without loading it."""
        # Initialize parent without loading the model
        super().__init__(model_name=model_name, **kwargs)
        # Override the model loading behavior
        self._model_loaded = False
        self._is_available = True  # Assume available until proven otherwise

    async def _load_model(self) -> None:
        """Load model and tokenizer if not already loaded."""
        if self._model_loaded:
            return

        print(
            f">> 📥 Downloading HuggingFace model: {self.model_name} (this may take a while on first use)"
        )

        try:
            # Call parent's load method
            await super()._load_model()
            print(f">> ✅ Successfully loaded {self.model_name}")
        except Exception as e:
            print(f">> ❌ Failed to load {self.model_name}: {str(e)}")
            raise

    async def health_check(self) -> bool:
        """Check if model is healthy without downloading."""
        # For lazy models, we assume they're healthy if they could be downloaded
        # We don't actually download during health check
        if self._model_loaded:
            return await super().health_check()

        # Model not loaded yet, but could be - return True
        return True
