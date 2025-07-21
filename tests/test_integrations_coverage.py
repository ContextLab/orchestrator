"""Tests to improve coverage for integration models."""

import os
import json
from typing import Dict, Any, List, Optional
import aiohttp

import pytest

from orchestrator.integrations.anthropic_model import AnthropicModel
from orchestrator.integrations.google_model import GoogleModel
from orchestrator.integrations.huggingface_model import HuggingFaceModel
from orchestrator.integrations.lazy_huggingface_model import LazyHuggingFaceModel
from orchestrator.integrations.lazy_ollama_model import LazyOllamaModel
from orchestrator.integrations.ollama_model import OllamaModel
from orchestrator.integrations.openai_model import OpenAIModel


class TestableAnthropicClient:
    """A testable Anthropic client for testing without real API calls."""
    
    def __init__(self):
        self.messages = TestableMessages()
        self.call_history = []
        
class TestableMessages:
    """Testable messages interface."""
    
    def __init__(self):
        self._responses = {}
        self._errors = {}
        self.call_history = []
        
    async def create(self, **kwargs):
        """Simulate message creation."""
        self.call_history.append(('create', kwargs))
        
        # Check for errors
        key = kwargs.get('model', 'default')
        if key in self._errors:
            raise self._errors[key]
            
        # Return response
        response = self._responses.get(key, TestableResponse("Generated text"))
        return response
        
    def set_response(self, model: str, text: str):
        """Set response for a model."""
        self._responses[model] = TestableResponse(text)
        
    def set_error(self, model: str, error: Exception):
        """Set error for a model."""
        self._errors[model] = error

class TestableResponse:
    """Testable response object."""
    
    def __init__(self, text: str):
        self.content = [TestableContent(text)]

class TestableContent:
    """Testable content object."""
    
    def __init__(self, text: str):
        self.text = text


class TestableAnthropicModel(AnthropicModel):
    """Testable Anthropic model."""
    
    def __init__(self, name: str = "test", model: str = "test"):
        # Skip parent init to avoid API key requirement
        self.name = name
        self.model = model
        self.provider = "anthropic"
        self._client = None
        self._test_client = TestableAnthropicClient()
        
    def _ensure_client(self):
        """Return testable client."""
        return self._test_client


class TestableGoogleModel(GoogleModel):
    """Testable Google model."""
    
    def __init__(self, name: str = "test", model: str = "test"):
        # Skip parent init to avoid API key requirement
        self.name = name
        self.model = model
        self.provider = "google"
        self._model = TestableGenerativeModel()
        self._configured = True
        
    def _ensure_model(self):
        """Return testable model."""
        return self._model


class TestableGenerativeModel:
    """Testable generative model."""
    
    def __init__(self):
        self._responses = {}
        self._errors = {}
        self.call_history = []
        
    def generate_content(self, prompt: str):
        """Generate content."""
        self.call_history.append(('generate_content', prompt))
        
        if 'error' in self._errors:
            raise self._errors['error']
            
        return TestableGenerateResponse(self._responses.get('generate', 'Generated text'))
        
    def start_chat(self, history=None):
        """Start chat."""
        return TestableChat(self._responses.get('chat', 'Chat response'))
        
    def set_response(self, key: str, text: str):
        """Set response."""
        self._responses[key] = text
        
    def set_error(self, key: str, error: Exception):
        """Set error."""
        self._errors[key] = error


class TestableGenerateResponse:
    """Testable generate response."""
    
    def __init__(self, text: str):
        self.text = text


class TestableChat:
    """Testable chat interface."""
    
    def __init__(self, response: str):
        self._response = response
        
    def send_message(self, message: str):
        """Send message."""
        return TestableGenerateResponse(self._response)


class TestableOpenAIClient:
    """Testable OpenAI client."""
    
    def __init__(self):
        self.chat = TestableCompletions()


class TestableCompletions:
    """Testable completions interface."""
    
    def __init__(self):
        self.completions = self
        self._responses = {}
        self._errors = {}
        self.call_history = []
        
    async def create(self, **kwargs):
        """Create completion."""
        self.call_history.append(('create', kwargs))
        
        model = kwargs.get('model', 'default')
        if model in self._errors:
            raise self._errors[model]
            
        text = self._responses.get(model, 'Generated text')
        return TestableOpenAIResponse(text)
        
    def set_response(self, model: str, text: str):
        """Set response."""
        self._responses[model] = text
        
    def set_error(self, model: str, error: Exception):
        """Set error."""
        self._errors[model] = error


class TestableOpenAIResponse:
    """Testable OpenAI response."""
    
    def __init__(self, text: str):
        self.choices = [TestableChoice(text)]


class TestableChoice:
    """Testable choice object."""
    
    def __init__(self, text: str):
        self.message = TestableMessage(text)


class TestableMessage:
    """Testable message object."""
    
    def __init__(self, content: str):
        self.content = content


class TestableOpenAIModel(OpenAIModel):
    """Testable OpenAI model."""
    
    def __init__(self, name: str = "test", model: str = "test"):
        # Skip parent init
        self.name = name
        self.model = model
        self.provider = "openai"
        self._client = None
        self._test_client = TestableOpenAIClient()
        
    def _ensure_client(self):
        """Return testable client."""
        return self._test_client


class TestableSubprocess:
    """Testable subprocess for Ollama checks."""
    
    def __init__(self):
        self.commands = {}
        self.call_history = []
        
    def run(self, cmd: list, **kwargs):
        """Simulate subprocess.run."""
        self.call_history.append((cmd, kwargs))
        
        cmd_str = ' '.join(cmd)
        if cmd_str in self.commands:
            return self.commands[cmd_str]
        else:
            # Default behavior
            if 'ollama --version' in cmd_str:
                return TestableProcess(0, "ollama version 0.1.0")
            else:
                raise FileNotFoundError()
                
    def set_command(self, cmd: str, returncode: int, stdout: str = "", stderr: str = ""):
        """Set command result."""
        self.commands[cmd] = TestableProcess(returncode, stdout, stderr)


class TestableProcess:
    """Testable process result."""
    
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestableHTTPResponse:
    """Testable HTTP response."""
    
    def __init__(self, status: int = 200, json_data: dict = None):
        self.status = status
        self._json_data = json_data or {}
        
    async def json(self):
        """Return JSON data."""
        return self._json_data
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, *args):
        pass


class TestableHTTPSession:
    """Testable HTTP session."""
    
    def __init__(self):
        self._responses = {}
        self.call_history = []
        
    async def post(self, url: str, **kwargs):
        """Simulate POST request."""
        self.call_history.append(('post', url, kwargs))
        
        if url in self._responses:
            return self._responses[url]
        else:
            # Default response
            return TestableHTTPResponse(200, {"response": "Generated text", "message": {"content": "Chat response"}})
            
    async def get(self, url: str, **kwargs):
        """Simulate GET request."""
        self.call_history.append(('get', url, kwargs))
        
        if url in self._responses:
            return self._responses[url]
        else:
            return TestableHTTPResponse(200)
            
    def set_response(self, url: str, response: TestableHTTPResponse):
        """Set response for URL."""
        self._responses[url] = response
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, *args):
        pass


class TestableOllamaModel(OllamaModel):
    """Testable Ollama model."""
    
    def __init__(self, model_name: str = "test"):
        # Skip parent init
        self.name = model_name
        self.model = model_name
        self.provider = "ollama"
        self._is_available = True
        self._test_session = TestableHTTPSession()
        
    async def _make_request(self, endpoint: str, data: dict = None):
        """Make testable request."""
        if endpoint == "generate":
            resp = await self._test_session.post("http://localhost:11434/api/generate", json=data)
            return await resp.json()
        elif endpoint == "chat":
            resp = await self._test_session.post("http://localhost:11434/api/chat", json=data)
            return await resp.json()
        elif endpoint == "health":
            resp = await self._test_session.get("http://localhost:11434/api/tags")
            return resp.status == 200


class TestableTransformersModel:
    """Testable transformers model."""
    
    def __init__(self):
        self._responses = {}
        self.call_history = []
        
    def generate(self, input_ids, **kwargs):
        """Generate output."""
        self.call_history.append(('generate', input_ids, kwargs))
        return [[1, 2, 3, 4, 5]]


class TestableTokenizer:
    """Testable tokenizer."""
    
    def __init__(self):
        self._responses = {}
        self.call_history = []
        
    def encode(self, text: str, **kwargs):
        """Encode text."""
        self.call_history.append(('encode', text))
        return [1, 2, 3]
        
    def decode(self, tokens: list, **kwargs):
        """Decode tokens."""
        self.call_history.append(('decode', tokens))
        return self._responses.get('decode', 'Generated text')
        
    def apply_chat_template(self, messages: list, **kwargs):
        """Apply chat template."""
        return "formatted chat"
        
    def set_response(self, key: str, text: str):
        """Set response."""
        self._responses[key] = text


class TestableHuggingFaceModel(HuggingFaceModel):
    """Testable HuggingFace model."""
    
    def __init__(self, model_name: str = "test"):
        # Skip parent init
        self.name = model_name
        self.model = model_name  
        self.provider = "huggingface"
        self._model = TestableTransformersModel()
        self._tokenizer = TestableTokenizer()
        self._model_loaded = True
        self._is_available = True
        
    async def _load_model(self):
        """Skip actual model loading."""
        pass


class TestAnthropicModel:
    """Test AnthropicModel integration."""

    def test_anthropic_model_init(self):
        """Test AnthropicModel initialization."""
        # Set test API key
        original_key = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        
        try:
            model = AnthropicModel(name="claude-3", model="claude-3")
            assert model.name == "claude-3"
            assert model.provider == "anthropic"
        finally:
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key
            else:
                os.environ.pop("ANTHROPIC_API_KEY", None)

    @pytest.mark.asyncio
    async def test_anthropic_generate(self):
        """Test AnthropicModel generate method."""
        model = TestableAnthropicModel("claude-3", "claude-3")
        model._test_client.messages.set_response("claude-3", "Generated text")
        
        result = await model.generate("Test prompt")
        assert result == "Generated text"

    @pytest.mark.asyncio
    async def test_anthropic_generate_structured(self):
        """Test AnthropicModel generate_structured method."""
        model = TestableAnthropicModel("claude-3", "claude-3")
        model._test_client.messages.set_response("claude-3", '{"key": "value"}')
        
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        
        result = await model.generate_structured("Test prompt", schema)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_anthropic_chat(self):
        """Test AnthropicModel chat method."""
        model = TestableAnthropicModel("claude-3", "claude-3")
        model._test_client.messages.set_response("claude-3", "Chat response")
        
        messages = [{"role": "user", "content": "Hello"}]
        
        result = await model.chat(messages)
        assert result == "Chat response"

    @pytest.mark.asyncio
    async def test_anthropic_analyze(self):
        """Test AnthropicModel analyze method."""
        model = TestableAnthropicModel("claude-3", "claude-3")
        model._test_client.messages.set_response("claude-3", "Analysis result")
        
        result = await model.analyze("content", "analysis_type")
        assert result == "Analysis result"

    @pytest.mark.asyncio
    async def test_anthropic_health_check(self):
        """Test AnthropicModel health_check method."""
        model = TestableAnthropicModel("claude-3", "claude-3")
        model._test_client.messages.set_response("claude-3", "OK")
        
        result = await model.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_anthropic_error_handling(self):
        """Test AnthropicModel error handling."""
        model = TestableAnthropicModel("claude-3", "claude-3")
        model._test_client.messages.set_error("claude-3", Exception("API Error"))
        
        with pytest.raises(Exception, match="API Error"):
            await model.generate("Test prompt")

    def test_anthropic_no_api_key(self):
        """Test AnthropicModel without API key."""
        # Clear API key
        original_key = os.environ.get("ANTHROPIC_API_KEY")
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]
            
        try:
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AnthropicModel(name="claude-3", model="claude-3")
        finally:
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key


class TestGoogleModel:
    """Test GoogleModel integration."""

    def test_google_model_init(self):
        """Test GoogleModel initialization."""
        original_key = os.environ.get("GOOGLE_API_KEY")
        os.environ["GOOGLE_API_KEY"] = "test-key"
        
        # Replace google.generativeai.configure temporarily
        import orchestrator.integrations.google_model
        original_configure = getattr(orchestrator.integrations.google_model, 'genai', None)
        
        class TestGenAI:
            @staticmethod
            def configure(api_key):
                pass
                
        if hasattr(orchestrator.integrations.google_model, 'genai'):
            orchestrator.integrations.google_model.genai = TestGenAI()
        
        try:
            model = GoogleModel(name="gemini-pro", model="gemini-pro")
            assert model.name == "gemini-pro"
            assert model.provider == "google"
        finally:
            if original_key:
                os.environ["GOOGLE_API_KEY"] = original_key
            else:
                os.environ.pop("GOOGLE_API_KEY", None)
            if original_configure:
                orchestrator.integrations.google_model.genai = original_configure

    @pytest.mark.asyncio
    async def test_google_generate(self):
        """Test GoogleModel generate method."""
        model = TestableGoogleModel("gemini-pro", "gemini-pro")
        model._model.set_response('generate', 'Generated text')
        
        result = await model.generate("Test prompt")
        assert result == "Generated text"

    @pytest.mark.asyncio
    async def test_google_generate_structured(self):
        """Test GoogleModel generate_structured method."""
        model = TestableGoogleModel("gemini-pro", "gemini-pro")
        model._model.set_response('generate', '{"key": "value"}')
        
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        result = await model.generate_structured("Test prompt", schema)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_google_chat(self):
        """Test GoogleModel chat method."""
        model = TestableGoogleModel("gemini-pro", "gemini-pro")
        model._model.set_response('chat', 'Chat response')
        
        messages = [{"role": "user", "content": "Hello"}]
        result = await model.chat(messages)
        assert result == "Chat response"

    @pytest.mark.asyncio
    async def test_google_analyze(self):
        """Test GoogleModel analyze method."""
        model = TestableGoogleModel("gemini-pro", "gemini-pro")
        model._model.set_response('generate', 'Analysis result')
        
        result = await model.analyze("content", "analysis_type")
        assert result == "Analysis result"

    @pytest.mark.asyncio
    async def test_google_health_check(self):
        """Test GoogleModel health_check method."""
        model = TestableGoogleModel("gemini-pro", "gemini-pro")
        model._model.set_response('generate', 'OK')
        
        result = await model.health_check()
        assert result is True

    def test_google_no_api_key(self):
        """Test GoogleModel without API key."""
        original_key = os.environ.get("GOOGLE_API_KEY")
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]
            
        try:
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                GoogleModel(name="gemini-pro", model="gemini-pro")
        finally:
            if original_key:
                os.environ["GOOGLE_API_KEY"] = original_key


class TestOpenAIModel:
    """Test OpenAIModel integration."""

    def test_openai_model_init(self):
        """Test OpenAIModel initialization with real API key."""
        from orchestrator.utils.api_keys import load_api_keys
        
        try:
            load_api_keys()  # Load real API keys
            model = OpenAIModel(name="gpt-4", model="gpt-4")
            assert model.name == "gpt-4"
            assert model.provider == "openai"
        except EnvironmentError as e:
            pytest.skip(f"Skipping test - API keys not configured: {e}")

    @pytest.mark.asyncio
    async def test_openai_generate(self):
        """Test OpenAIModel generate method."""
        model = TestableOpenAIModel("gpt-4", "gpt-4")
        model._test_client.chat.completions.set_response("gpt-4", "Generated text")
        
        result = await model.generate("Test prompt")
        assert result == "Generated text"

    @pytest.mark.asyncio
    async def test_openai_generate_structured(self):
        """Test OpenAIModel generate_structured method."""
        model = TestableOpenAIModel("gpt-4", "gpt-4")
        model._test_client.chat.completions.set_response("gpt-4", '{"key": "value"}')
        
        schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        
        result = await model.generate_structured("Test prompt", schema)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_openai_chat(self):
        """Test OpenAIModel chat method."""
        model = TestableOpenAIModel("gpt-4", "gpt-4")
        model._test_client.chat.completions.set_response("gpt-4", "Chat response")
        
        messages = [{"role": "user", "content": "Hello"}]
        
        result = await model.chat(messages)
        assert result == "Chat response"

    @pytest.mark.asyncio
    async def test_openai_health_check(self):
        """Test OpenAIModel health_check method."""
        model = TestableOpenAIModel("gpt-4", "gpt-4")
        model._test_client.chat.completions.set_response("gpt-4", "OK")
        
        result = await model.health_check()
        assert result is True

    def test_openai_no_api_key(self):
        """Test OpenAIModel without API key."""
        original_key = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
            
        try:
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIModel(name="gpt-4", model="gpt-4")
        finally:
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key


class TestOllamaModel:
    """Test OllamaModel integration."""

    def test_ollama_model_init(self):
        """Test OllamaModel initialization."""
        # Replace subprocess temporarily
        import subprocess
        original_run = subprocess.run
        test_subprocess = TestableSubprocess()
        subprocess.run = test_subprocess.run
        
        try:
            model = OllamaModel(model_name="llama2:7b")
            assert model.name == "llama2:7b"
            assert model.provider == "ollama"
        finally:
            subprocess.run = original_run

    @pytest.mark.asyncio
    async def test_ollama_generate(self):
        """Test OllamaModel generate method."""
        model = TestableOllamaModel("llama2:7b")
        
        result = await model.generate("Test prompt")
        assert result == "Generated text"

    @pytest.mark.asyncio
    async def test_ollama_chat(self):
        """Test OllamaModel chat method."""
        model = TestableOllamaModel("llama2:7b")
        
        messages = [{"role": "user", "content": "Hello"}]
        result = await model.chat(messages)
        assert result == "Chat response"

    @pytest.mark.asyncio
    async def test_ollama_health_check(self):
        """Test OllamaModel health_check method."""
        model = TestableOllamaModel("llama2:7b")
        
        result = await model.health_check()
        assert result is True

    def test_ollama_not_installed(self):
        """Test OllamaModel when Ollama is not installed."""
        import subprocess
        original_run = subprocess.run
        
        def failing_run(cmd, **kwargs):
            raise FileNotFoundError()
            
        subprocess.run = failing_run
        
        try:
            model = OllamaModel(model_name="llama2:7b")
            assert model._is_available is False
        finally:
            subprocess.run = original_run


class TestLazyOllamaModel:
    """Test LazyOllamaModel integration."""

    def test_lazy_ollama_init(self):
        """Test LazyOllamaModel initialization."""
        # Replace check function
        import orchestrator.integrations.lazy_ollama_model
        original_check = orchestrator.integrations.lazy_ollama_model.check_ollama_model
        
        orchestrator.integrations.lazy_ollama_model.check_ollama_model = lambda x: False
        
        try:
            model = LazyOllamaModel(model_name="llama2:7b")
            assert model.name == "llama2:7b"
            assert model._is_available is True  # Initially assumed available
        finally:
            orchestrator.integrations.lazy_ollama_model.check_ollama_model = original_check

    @pytest.mark.asyncio
    async def test_lazy_ollama_download_on_use(self):
        """Test LazyOllamaModel downloads on first use."""
        import orchestrator.integrations.lazy_ollama_model
        original_check = orchestrator.integrations.lazy_ollama_model.check_ollama_model
        original_install = orchestrator.integrations.lazy_ollama_model.install_ollama_model
        
        check_calls = []
        install_calls = []
        
        def track_check(model_name):
            check_calls.append(model_name)
            return False
            
        def track_install(model_name):
            install_calls.append(model_name)
            return True
            
        orchestrator.integrations.lazy_ollama_model.check_ollama_model = track_check
        orchestrator.integrations.lazy_ollama_model.install_ollama_model = track_install
        
        try:
            model = LazyOllamaModel(model_name="llama2:7b")
            # Override parent class methods
            model._test_session = TestableHTTPSession()
            
            # Replace _make_request
            async def test_make_request(endpoint, data=None):
                if endpoint == "generate":
                    return {"response": "Generated text"}
                    
            model._make_request = test_make_request
            
            result = await model.generate("Test prompt")
            
            # Should have attempted to install
            assert len(install_calls) == 1
            assert install_calls[0] == "llama2:7b"
            assert result == "Generated text"
        finally:
            orchestrator.integrations.lazy_ollama_model.check_ollama_model = original_check
            orchestrator.integrations.lazy_ollama_model.install_ollama_model = original_install

    @pytest.mark.asyncio
    async def test_lazy_ollama_download_failure(self):
        """Test LazyOllamaModel when download fails."""
        import orchestrator.integrations.lazy_ollama_model
        original_check = orchestrator.integrations.lazy_ollama_model.check_ollama_model
        original_install = orchestrator.integrations.lazy_ollama_model.install_ollama_model
        
        orchestrator.integrations.lazy_ollama_model.check_ollama_model = lambda x: False
        orchestrator.integrations.lazy_ollama_model.install_ollama_model = lambda x: False
        
        try:
            model = LazyOllamaModel(model_name="llama2:7b")
            
            with pytest.raises(RuntimeError, match="not available"):
                await model.generate("Test prompt")
        finally:
            orchestrator.integrations.lazy_ollama_model.check_ollama_model = original_check
            orchestrator.integrations.lazy_ollama_model.install_ollama_model = original_install

    @pytest.mark.asyncio
    async def test_lazy_ollama_already_available(self):
        """Test LazyOllamaModel when model is already available."""
        import orchestrator.integrations.lazy_ollama_model
        original_check = orchestrator.integrations.lazy_ollama_model.check_ollama_model
        
        orchestrator.integrations.lazy_ollama_model.check_ollama_model = lambda x: True
        
        try:
            model = LazyOllamaModel(model_name="llama2:7b")
            result = await model._ensure_model_available()
            
            assert result is True
            assert model._model_downloaded is True
        finally:
            orchestrator.integrations.lazy_ollama_model.check_ollama_model = original_check


class TestHuggingFaceModel:
    """Test HuggingFaceModel integration."""

    def test_huggingface_model_init(self):
        """Test HuggingFaceModel initialization."""
        # Replace transformers imports
        import orchestrator.integrations.huggingface_model
        
        class TestAutoModel:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                return TestableTransformersModel()
                
        class TestAutoTokenizer:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                return TestableTokenizer()
                
        original_model = getattr(orchestrator.integrations.huggingface_model, 'AutoModelForCausalLM', None)
        original_tokenizer = getattr(orchestrator.integrations.huggingface_model, 'AutoTokenizer', None)
        
        if hasattr(orchestrator.integrations.huggingface_model, 'AutoModelForCausalLM'):
            orchestrator.integrations.huggingface_model.AutoModelForCausalLM = TestAutoModel
        if hasattr(orchestrator.integrations.huggingface_model, 'AutoTokenizer'):
            orchestrator.integrations.huggingface_model.AutoTokenizer = TestAutoTokenizer
        
        try:
            model = HuggingFaceModel(model_name="gpt2")
            assert model.name == "gpt2"
            assert model.provider == "huggingface"
        finally:
            if original_model:
                orchestrator.integrations.huggingface_model.AutoModelForCausalLM = original_model
            if original_tokenizer:
                orchestrator.integrations.huggingface_model.AutoTokenizer = original_tokenizer

    @pytest.mark.asyncio
    async def test_huggingface_generate(self):
        """Test HuggingFaceModel generate method."""
        model = TestableHuggingFaceModel("gpt2")
        
        result = await model.generate("Test prompt")
        assert result == "Generated text"

    @pytest.mark.asyncio
    async def test_huggingface_chat(self):
        """Test HuggingFaceModel chat method."""
        model = TestableHuggingFaceModel("gpt2")
        model._tokenizer.set_response('decode', 'Chat response')
        
        messages = [{"role": "user", "content": "Hello"}]
        result = await model.chat(messages)
        assert result == "Chat response"

    @pytest.mark.asyncio
    async def test_huggingface_health_check(self):
        """Test HuggingFaceModel health_check method."""
        model = TestableHuggingFaceModel("gpt2")
        
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

    @pytest.mark.asyncio
    async def test_lazy_huggingface_load_on_use(self):
        """Test LazyHuggingFaceModel loads on first use."""
        # Replace transformers imports
        import orchestrator.integrations.lazy_huggingface_model
        
        load_calls = []
        
        class TestAutoModel:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                load_calls.append(('model', model_name))
                return TestableTransformersModel()
                
        class TestAutoTokenizer:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                load_calls.append(('tokenizer', model_name))
                return TestableTokenizer()
                
        original_model = getattr(orchestrator.integrations.lazy_huggingface_model, 'AutoModelForCausalLM', None)
        original_tokenizer = getattr(orchestrator.integrations.lazy_huggingface_model, 'AutoTokenizer', None)
        
        if hasattr(orchestrator.integrations.lazy_huggingface_model, 'AutoModelForCausalLM'):
            orchestrator.integrations.lazy_huggingface_model.AutoModelForCausalLM = TestAutoModel
        if hasattr(orchestrator.integrations.lazy_huggingface_model, 'AutoTokenizer'):
            orchestrator.integrations.lazy_huggingface_model.AutoTokenizer = TestAutoTokenizer
        
        try:
            model = LazyHuggingFaceModel(model_name="gpt2")
            
            # Call _load_model
            await model._load_model()
            
            assert model._model_loaded is True
            assert len(load_calls) == 2
            assert ('model', 'gpt2') in load_calls
            assert ('tokenizer', 'gpt2') in load_calls
        finally:
            if original_model:
                orchestrator.integrations.lazy_huggingface_model.AutoModelForCausalLM = original_model
            if original_tokenizer:
                orchestrator.integrations.lazy_huggingface_model.AutoTokenizer = original_tokenizer

    @pytest.mark.asyncio
    async def test_lazy_huggingface_health_check_without_loading(self):
        """Test LazyHuggingFaceModel health check doesn't load model."""
        model = LazyHuggingFaceModel(model_name="gpt2")
        
        result = await model.health_check()
        
        assert result is True
        assert model._model_loaded is False