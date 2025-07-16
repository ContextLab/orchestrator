"""Tests to improve coverage for integration models."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.integrations.anthropic_model import AnthropicModel
from orchestrator.integrations.google_model import GoogleModel
from orchestrator.integrations.huggingface_model import HuggingFaceModel
from orchestrator.integrations.lazy_huggingface_model import LazyHuggingFaceModel
from orchestrator.integrations.lazy_ollama_model import LazyOllamaModel
from orchestrator.integrations.ollama_model import OllamaModel
from orchestrator.integrations.openai_model import OpenAIModel


class TestAnthropicModel:
    """Test AnthropicModel integration."""

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_anthropic_model_init(self):
        """Test AnthropicModel initialization."""
        model = AnthropicModel(name="claude-3", model="claude-3")
        assert model.name == "claude-3"
        assert model.provider == "anthropic"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_anthropic_generate(self):
        """Test AnthropicModel generate method."""
        model = AnthropicModel(name="claude-3", model="claude-3")
        
        # Mock the client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated text")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        with patch.object(model, "_ensure_client", return_value=mock_client):
            result = await model.generate("Test prompt")
            assert result == "Generated text"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_anthropic_generate_structured(self):
        """Test AnthropicModel generate_structured method."""
        model = AnthropicModel(name="claude-3", model="claude-3")
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"key": "value"}')]
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        
        with patch.object(model, "_ensure_client", return_value=mock_client):
            result = await model.generate_structured("Test prompt", schema)
            assert result == {"key": "value"}

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_anthropic_chat(self):
        """Test AnthropicModel chat method."""
        model = AnthropicModel(name="claude-3", model="claude-3")
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Chat response")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(model, "_ensure_client", return_value=mock_client):
            result = await model.chat(messages)
            assert result == "Chat response"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_anthropic_analyze(self):
        """Test AnthropicModel analyze method."""
        model = AnthropicModel(name="claude-3", model="claude-3")
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Analysis result")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        with patch.object(model, "_ensure_client", return_value=mock_client):
            result = await model.analyze("content", "analysis_type")
            assert result == "Analysis result"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_anthropic_health_check(self):
        """Test AnthropicModel health_check method."""
        model = AnthropicModel(name="claude-3", model="claude-3")
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="OK")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        with patch.object(model, "_ensure_client", return_value=mock_client):
            result = await model.health_check()
            assert result is True

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_anthropic_error_handling(self):
        """Test AnthropicModel error handling."""
        model = AnthropicModel(name="claude-3", model="claude-3")
        
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))
        
        with patch.object(model, "_ensure_client", return_value=mock_client):
            with pytest.raises(Exception, match="API Error"):
                await model.generate("Test prompt")

    def test_anthropic_no_api_key(self):
        """Test AnthropicModel without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AnthropicModel(name="claude-3", model="claude-3")


class TestGoogleModel:
    """Test GoogleModel integration."""

    @patch("google.generativeai.configure")
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_google_model_init(self, mock_configure):
        """Test GoogleModel initialization."""
        model = GoogleModel(name="gemini-pro", model="gemini-pro")
        assert model.name == "gemini-pro"
        assert model.provider == "google"
        mock_configure.assert_called_once_with(api_key="test-key")

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_google_generate(self, mock_model_class, mock_configure):
        """Test GoogleModel generate method."""
        # Setup mock
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated text"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        model = GoogleModel(name="gemini-pro", model="gemini-pro")
        result = await model.generate("Test prompt")
        assert result == "Generated text"

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_google_generate_structured(self, mock_model_class, mock_configure):
        """Test GoogleModel generate_structured method."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"key": "value"}'
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        model = GoogleModel(name="gemini-pro", model="gemini-pro")
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        result = await model.generate_structured("Test prompt", schema)
        assert result == {"key": "value"}

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_google_chat(self, mock_model_class, mock_configure):
        """Test GoogleModel chat method."""
        mock_model = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Chat response"
        mock_chat.send_message.return_value = mock_response
        mock_model.start_chat.return_value = mock_chat
        mock_model_class.return_value = mock_model
        
        model = GoogleModel(name="gemini-pro", model="gemini-pro")
        messages = [{"role": "user", "content": "Hello"}]
        result = await model.chat(messages)
        assert result == "Chat response"

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_google_analyze(self, mock_model_class, mock_configure):
        """Test GoogleModel analyze method."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Analysis result"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        model = GoogleModel(name="gemini-pro", model="gemini-pro")
        result = await model.analyze("content", "analysis_type")
        assert result == "Analysis result"

    @patch("google.generativeai.configure")
    @patch("google.generativeai.GenerativeModel")
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_google_health_check(self, mock_model_class, mock_configure):
        """Test GoogleModel health_check method."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "OK"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        model = GoogleModel(name="gemini-pro", model="gemini-pro")
        result = await model.health_check()
        assert result is True

    def test_google_no_api_key(self):
        """Test GoogleModel without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                GoogleModel(name="gemini-pro", model="gemini-pro")


class TestOpenAIModel:
    """Test OpenAIModel integration."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_model_init(self):
        """Test OpenAIModel initialization."""
        model = OpenAIModel(name="gpt-4", model="gpt-4")
        assert model.name == "gpt-4"
        assert model.provider == "openai"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_openai_generate(self):
        """Test OpenAIModel generate method."""
        model = OpenAIModel(name="gpt-4", model="gpt-4")
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated text"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        with patch.object(model, "_ensure_client", return_value=mock_client):
            result = await model.generate("Test prompt")
            assert result == "Generated text"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_openai_generate_structured(self):
        """Test OpenAIModel generate_structured method."""
        model = OpenAIModel(name="gpt-4", model="gpt-4")
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"key": "value"}'))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        
        with patch.object(model, "_ensure_client", return_value=mock_client):
            result = await model.generate_structured("Test prompt", schema)
            assert result == {"key": "value"}

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_openai_chat(self):
        """Test OpenAIModel chat method."""
        model = OpenAIModel(name="gpt-4", model="gpt-4")
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Chat response"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(model, "_ensure_client", return_value=mock_client):
            result = await model.chat(messages)
            assert result == "Chat response"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @pytest.mark.asyncio
    async def test_openai_health_check(self):
        """Test OpenAIModel health_check method."""
        model = OpenAIModel(name="gpt-4", model="gpt-4")
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        with patch.object(model, "_ensure_client", return_value=mock_client):
            result = await model.health_check()
            assert result is True

    def test_openai_no_api_key(self):
        """Test OpenAIModel without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIModel(name="gpt-4", model="gpt-4")


class TestOllamaModel:
    """Test OllamaModel integration."""

    @patch("subprocess.run")
    def test_ollama_model_init(self, mock_run):
        """Test OllamaModel initialization."""
        # Mock ollama check
        mock_run.return_value = MagicMock(returncode=0, stdout="ollama version 0.1.0")
        
        model = OllamaModel(model_name="llama2:7b")
        assert model.name == "llama2:7b"
        assert model.provider == "ollama"

    @patch("subprocess.run")
    @patch("aiohttp.ClientSession")
    @pytest.mark.asyncio
    async def test_ollama_generate(self, mock_session, mock_run):
        """Test OllamaModel generate method."""
        # Mock ollama check
        mock_run.return_value = MagicMock(returncode=0, stdout="ollama version 0.1.0")
        
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"response": "Generated text"})
        
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
        
        model = OllamaModel(model_name="llama2:7b")
        result = await model.generate("Test prompt")
        assert result == "Generated text"

    @patch("subprocess.run")
    @patch("aiohttp.ClientSession")
    @pytest.mark.asyncio
    async def test_ollama_chat(self, mock_session, mock_run):
        """Test OllamaModel chat method."""
        # Mock ollama check
        mock_run.return_value = MagicMock(returncode=0, stdout="ollama version 0.1.0")
        
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"message": {"content": "Chat response"}})
        
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
        
        model = OllamaModel(model_name="llama2:7b")
        messages = [{"role": "user", "content": "Hello"}]
        result = await model.chat(messages)
        assert result == "Chat response"

    @patch("subprocess.run")
    @patch("aiohttp.ClientSession")
    @pytest.mark.asyncio
    async def test_ollama_health_check(self, mock_session, mock_run):
        """Test OllamaModel health_check method."""
        # Mock ollama check
        mock_run.return_value = MagicMock(returncode=0, stdout="ollama version 0.1.0")
        
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        
        mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
        
        model = OllamaModel(model_name="llama2:7b")
        result = await model.health_check()
        assert result is True

    @patch("subprocess.run")
    def test_ollama_not_installed(self, mock_run):
        """Test OllamaModel when Ollama is not installed."""
        # Mock ollama check failure
        mock_run.side_effect = FileNotFoundError()
        
        model = OllamaModel(model_name="llama2:7b")
        assert model._is_available is False


class TestLazyOllamaModel:
    """Test LazyOllamaModel integration."""

    @patch("orchestrator.integrations.lazy_ollama_model.check_ollama_model")
    def test_lazy_ollama_init(self, mock_check):
        """Test LazyOllamaModel initialization."""
        mock_check.return_value = False
        
        model = LazyOllamaModel(model_name="llama2:7b")
        assert model.name == "llama2:7b"
        assert model._is_available is True  # Initially assumed available

    @patch("orchestrator.integrations.lazy_ollama_model.check_ollama_model")
    @patch("orchestrator.integrations.lazy_ollama_model.install_ollama_model")
    @patch("aiohttp.ClientSession")
    @pytest.mark.asyncio
    async def test_lazy_ollama_download_on_use(self, mock_session, mock_install, mock_check):
        """Test LazyOllamaModel downloads on first use."""
        mock_check.return_value = False
        mock_install.return_value = True
        
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"response": "Generated text"})
        
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
        
        model = LazyOllamaModel(model_name="llama2:7b")
        result = await model.generate("Test prompt")
        
        # Should have attempted to install
        mock_install.assert_called_once_with("llama2:7b")
        assert result == "Generated text"

    @patch("orchestrator.integrations.lazy_ollama_model.check_ollama_model")
    @patch("orchestrator.integrations.lazy_ollama_model.install_ollama_model")
    @pytest.mark.asyncio
    async def test_lazy_ollama_download_failure(self, mock_install, mock_check):
        """Test LazyOllamaModel when download fails."""
        mock_check.return_value = False
        mock_install.return_value = False
        
        model = LazyOllamaModel(model_name="llama2:7b")
        
        with pytest.raises(RuntimeError, match="not available"):
            await model.generate("Test prompt")

    @patch("orchestrator.integrations.lazy_ollama_model.check_ollama_model")
    @pytest.mark.asyncio
    async def test_lazy_ollama_already_available(self, mock_check):
        """Test LazyOllamaModel when model is already available."""
        mock_check.return_value = True
        
        model = LazyOllamaModel(model_name="llama2:7b")
        result = await model._ensure_model_available()
        
        assert result is True
        assert model._model_downloaded is True


class TestHuggingFaceModel:
    """Test HuggingFaceModel integration."""

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_huggingface_model_init(self, mock_tokenizer, mock_model):
        """Test HuggingFaceModel initialization."""
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        model = HuggingFaceModel(model_name="gpt2")
        assert model.name == "gpt2"
        assert model.provider == "huggingface"

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @pytest.mark.asyncio
    async def test_huggingface_generate(self, mock_tokenizer_class, mock_model_class):
        """Test HuggingFaceModel generate method."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Generated text"
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_model_class.return_value = mock_model
        
        model = HuggingFaceModel(model_name="gpt2")
        
        # Override the async loading
        model._model = mock_model
        model._tokenizer = mock_tokenizer
        model._model_loaded = True
        
        result = await model.generate("Test prompt")
        assert result == "Generated text"

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @pytest.mark.asyncio
    async def test_huggingface_chat(self, mock_tokenizer_class, mock_model_class):
        """Test HuggingFaceModel chat method."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted chat"
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Chat response"
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_model_class.return_value = mock_model
        
        model = HuggingFaceModel(model_name="gpt2")
        model._model = mock_model
        model._tokenizer = mock_tokenizer
        model._model_loaded = True
        
        messages = [{"role": "user", "content": "Hello"}]
        result = await model.chat(messages)
        assert result == "Chat response"

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @pytest.mark.asyncio
    async def test_huggingface_health_check(self, mock_tokenizer_class, mock_model_class):
        """Test HuggingFaceModel health_check method."""
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        model = HuggingFaceModel(model_name="gpt2")
        model._model = mock_model
        model._tokenizer = mock_tokenizer
        model._model_loaded = True
        
        result = await model.health_check()
        assert result is True


class TestLazyHuggingFaceModel:
    """Test LazyHuggingFaceModel integration."""

    def test_lazy_huggingface_init(self):
        """Test LazyHuggingFaceModel initialization."""
        model = LazyHuggingFaceModel(model_name="gpt2")
        assert model.name == "gpt2"
        assert model._is_available is True
        assert model._model_loaded is False

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @pytest.mark.asyncio
    async def test_lazy_huggingface_load_on_use(self, mock_tokenizer_class, mock_model_class):
        """Test LazyHuggingFaceModel loads on first use."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        model = LazyHuggingFaceModel(model_name="gpt2")
        
        # Call _load_model
        await model._load_model()
        
        assert model._model_loaded is True
        mock_model_class.assert_called_once()
        mock_tokenizer_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_lazy_huggingface_health_check_without_loading(self):
        """Test LazyHuggingFaceModel health check doesn't load model."""
        model = LazyHuggingFaceModel(model_name="gpt2")
        
        result = await model.health_check()
        
        assert result is True
        assert model._model_loaded is False