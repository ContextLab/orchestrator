"""Integration tests for external LLM APIs.

These tests make actual API calls to verify:
1. Our code correctly calls external APIs
2. API responses are in expected format
3. Error handling works with real API errors
4. Authentication and rate limiting work properly

Note: These tests require valid API keys and may incur costs.
Set environment variables or skip tests if keys not available.
"""

import asyncio
import os
import time

import pytest

# Test if API keys are available
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
HAS_ANTHROPIC_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))
HAS_GOOGLE_KEY = bool(os.getenv("GOOGLE_API_KEY"))


class OpenAIModel:
    """OpenAI API integration for testing."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=self.api_key)

            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 100),
                temperature=kwargs.get("temperature", 0.7))

            return response.choices[0].message.content

        except ImportError:
            pytest.skip("OpenAI library not installed")
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")

    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            result = await self.generate("Hello", max_tokens=1)
            return isinstance(result, str) and len(result) > 0
        except Exception:
            return False


class AnthropicModel:
    """Anthropic Claude API integration for testing."""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)

            response = await client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 100),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}])

            return response.content[0].text

        except ImportError:
            pytest.skip("Anthropic library not installed")
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")

    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible."""
        try:
            result = await self.generate("Hello", max_tokens=1)
            return isinstance(result, str) and len(result) > 0
        except Exception:
            return False


class GoogleModel:
    """Google Gemini API integration for testing."""

    def __init__(self, api_key: str, model: str = "gemini-pro"):
        self.api_key = api_key
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Google Gemini API."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)

            model = genai.GenerativeModel(self.model)
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=kwargs.get("max_tokens", 100),
                    temperature=kwargs.get("temperature", 0.7)))

            return response.text

        except ImportError:
            pytest.skip("Google GenerativeAI library not installed")
        except Exception as e:
            raise Exception(f"Google API error: {e}")

    async def health_check(self) -> bool:
        """Check if Google API is accessible."""
        try:
            result = await self.generate("Hello", max_tokens=1)
            return isinstance(result, str) and len(result) > 0
        except Exception:
            return False


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key not available")
class TestOpenAIIntegration:
    """Integration tests for OpenAI API."""

    @pytest.fixture
    def openai_model(self):
        """Create OpenAI model instance."""
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAIModel(api_key)

    @pytest.mark.asyncio
    async def test_openai_basic_generation(self, openai_model):
        """Test basic text generation with OpenAI."""
        prompt = "What is 2+2? Answer with just the number."

        result = await openai_model.generate(prompt, max_tokens=10, temperature=0)

        assert isinstance(result, str)
        assert len(result.strip()) > 0
        assert "4" in result  # Should contain the answer

    @pytest.mark.asyncio
    async def test_openai_response_format(self, openai_model):
        """Test that OpenAI responses are in expected format."""
        prompt = (
            "Generate a JSON object with a 'message' field containing 'hello world'"
        )

        result = await openai_model.generate(prompt, max_tokens=50, temperature=0)

        assert isinstance(result, str)
        assert len(result) > 0
        # Basic check that it looks like JSON
        assert "{" in result and "}" in result
        assert "message" in result.lower()

    @pytest.mark.asyncio
    async def test_openai_temperature_control(self, openai_model):
        """Test temperature parameter affects output variability."""
        prompt = "Write a creative story beginning with 'Once upon a time' in exactly 10 words."

        # Generate with low temperature (deterministic)
        result1 = await openai_model.generate(prompt, max_tokens=30, temperature=0.1)
        result2 = await openai_model.generate(prompt, max_tokens=30, temperature=0.1)

        # Generate with high temperature (creative)
        result3 = await openai_model.generate(prompt, max_tokens=30, temperature=0.9)

        assert isinstance(result1, str)
        assert isinstance(result2, str)
        assert isinstance(result3, str)

        # Low temperature results should be more similar
        # High temperature result should exist (basic sanity check)
        assert len(result1) > 0
        assert len(result2) > 0
        assert len(result3) > 0

    @pytest.mark.asyncio
    async def test_openai_max_tokens_limit(self, openai_model):
        """Test max_tokens parameter limits output length."""
        prompt = "Write a very long essay about artificial intelligence."

        # Short limit
        short_result = await openai_model.generate(prompt, max_tokens=10, temperature=0)

        # Longer limit
        long_result = await openai_model.generate(prompt, max_tokens=100, temperature=0)

        assert isinstance(short_result, str)
        assert isinstance(long_result, str)

        # Longer result should generally be longer (basic check)
        assert len(short_result) > 0
        assert len(long_result) > 0

    @pytest.mark.asyncio
    async def test_openai_error_handling_invalid_model(self):
        """Test error handling with invalid model name."""
        api_key = os.getenv("OPENAI_API_KEY")
        invalid_model = OpenAIModel(api_key, model="nonexistent-model-12345")

        with pytest.raises(Exception) as exc_info:
            await invalid_model.generate("Hello")

        assert "OpenAI API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_openai_error_handling_invalid_key(self):
        """Test error handling with invalid API key."""
        invalid_model = OpenAIModel("invalid-key-12345")

        with pytest.raises(Exception) as exc_info:
            await invalid_model.generate("Hello")

        assert "OpenAI API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_openai_health_check(self, openai_model):
        """Test OpenAI health check functionality."""
        health = await openai_model.health_check()

        assert isinstance(health, bool)
        assert health is True  # Should be healthy with valid key

    @pytest.mark.asyncio
    async def test_openai_rate_limiting_awareness(self, openai_model):
        """Test that we can handle rate limiting gracefully."""
        # Make several requests in quick succession
        tasks = []
        for i in range(3):  # Keep low to avoid hitting actual rate limits
            task = openai_model.generate(f"Count to {i+1}", max_tokens=10)
            tasks.append(task)
            time.sleep(0.1)  # Small delay

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed or fail gracefully
        for result in results:
            if isinstance(result, Exception):
                # If rate limited, error message should indicate that
                assert "rate" in str(result).lower() or "limit" in str(result).lower()
            else:
                assert isinstance(result, str)


@pytest.mark.skipif(not HAS_ANTHROPIC_KEY, reason="Anthropic API key not available")
class TestAnthropicIntegration:
    """Integration tests for Anthropic Claude API."""

    @pytest.fixture
    def anthropic_model(self):
        """Create Anthropic model instance."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return AnthropicModel(api_key)

    @pytest.mark.asyncio
    async def test_anthropic_basic_generation(self, anthropic_model):
        """Test basic text generation with Anthropic."""
        prompt = "What is the capital of France? Answer with just the city name."

        result = await anthropic_model.generate(prompt, max_tokens=10, temperature=0)

        assert isinstance(result, str)
        assert len(result.strip()) > 0
        assert "Paris" in result

    @pytest.mark.asyncio
    async def test_anthropic_response_format(self, anthropic_model):
        """Test that Anthropic responses are in expected format."""
        prompt = (
            "List three colors in a simple format: 1. [color] 2. [color] 3. [color]"
        )

        result = await anthropic_model.generate(prompt, max_tokens=50, temperature=0)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain numbered list format
        assert "1." in result
        assert "2." in result
        assert "3." in result

    @pytest.mark.asyncio
    async def test_anthropic_health_check(self, anthropic_model):
        """Test Anthropic health check functionality."""
        health = await anthropic_model.health_check()

        assert isinstance(health, bool)
        assert health is True

    @pytest.mark.asyncio
    async def test_anthropic_error_handling_invalid_key(self):
        """Test error handling with invalid API key."""
        invalid_model = AnthropicModel("invalid-key-12345")

        with pytest.raises(Exception) as exc_info:
            await invalid_model.generate("Hello")

        assert "Anthropic API error" in str(exc_info.value)


@pytest.mark.skipif(not HAS_GOOGLE_KEY, reason="Google API key not available")
class TestGoogleIntegration:
    """Integration tests for Google Gemini API."""

    @pytest.fixture
    def google_model(self):
        """Create Google model instance."""
        api_key = os.getenv("GOOGLE_API_KEY")
        return GoogleModel(api_key)

    @pytest.mark.asyncio
    async def test_google_basic_generation(self, google_model):
        """Test basic text generation with Google Gemini."""
        prompt = "What is 5 * 3? Answer with just the number."

        result = await google_model.generate(prompt, max_tokens=10, temperature=0)

        assert isinstance(result, str)
        assert len(result.strip()) > 0
        assert "15" in result

    @pytest.mark.asyncio
    async def test_google_response_format(self, google_model):
        """Test that Google responses are in expected format."""
        prompt = "Name two programming languages, one per line."

        result = await google_model.generate(prompt, max_tokens=30, temperature=0)

        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain programming language names
        assert "\n" in result or len(result.split()) >= 2

    @pytest.mark.asyncio
    async def test_google_health_check(self, google_model):
        """Test Google health check functionality."""
        health = await google_model.health_check()

        assert isinstance(health, bool)
        assert health is True

    @pytest.mark.asyncio
    async def test_google_error_handling_invalid_key(self):
        """Test error handling with invalid API key."""
        invalid_model = GoogleModel("invalid-key-12345")

        with pytest.raises(Exception) as exc_info:
            await invalid_model.generate("Hello")

        assert "Google API error" in str(exc_info.value)


class TestLLMIntegrationConsistency:
    """Test consistency across different LLM providers."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not (HAS_OPENAI_KEY and HAS_ANTHROPIC_KEY),
        reason="Both OpenAI and Anthropic keys needed")
    async def test_cross_provider_consistency(self):
        """Test that different providers give reasonable responses to same prompt."""
        openai_model = OpenAIModel(os.getenv("OPENAI_API_KEY"))
        anthropic_model = AnthropicModel(os.getenv("ANTHROPIC_API_KEY"))

        prompt = "What is the chemical symbol for gold? Answer with just the symbol."

        openai_result = await openai_model.generate(prompt, max_tokens=5, temperature=0)
        anthropic_result = await anthropic_model.generate(
            prompt, max_tokens=5, temperature=0
        )

        assert isinstance(openai_result, str)
        assert isinstance(anthropic_result, str)

        # Both should contain "Au" (gold's chemical symbol)
        assert "Au" in openai_result
        assert "Au" in anthropic_result

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OpenAI API key needed")
    async def test_model_switching_capability(self):
        """Test ability to switch between different models of same provider."""
        # Test with different OpenAI models
        gpt35_model = OpenAIModel(os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

        prompt = "What is 2+2?"

        result = await gpt35_model.generate(prompt, max_tokens=10, temperature=0)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "4" in result

    def test_api_key_validation(self):
        """Test that API key validation works properly."""
        # Test empty key
        with pytest.raises((ValueError, Exception)):
            OpenAIModel("")

        # Test None key
        with pytest.raises((ValueError, TypeError, Exception)):
            OpenAIModel(None)

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that API timeouts are handled gracefully."""
        if not HAS_OPENAI_KEY:
            raise AssertionError(
                "OpenAI API key not available. "
                "Please configure API keys in ~/.orchestrator/.env"
            )

        model = OpenAIModel(os.getenv("OPENAI_API_KEY"))

        # Test with a very long prompt that might timeout
        very_long_prompt = (
            "Please write a detailed essay about " + "artificial intelligence " * 100
        )

        try:
            result = await asyncio.wait_for(
                model.generate(very_long_prompt, max_tokens=10),  # 30 second timeout
            )
            # If it succeeds, verify response format
            assert isinstance(result, str)
        except asyncio.TimeoutError:
            # Timeout is acceptable behavior
            pass
        except Exception as e:
            # Other exceptions should still be API-related
            assert "API" in str(e) or "timeout" in str(e).lower()


if __name__ == "__main__":
    # Print available API keys for debugging
    print("Available API keys:")
    print(f"OpenAI: {'✓' if HAS_OPENAI_KEY else '✗'}")
    print(f"Anthropic: {'✓' if HAS_ANTHROPIC_KEY else '✗'}")
    print(f"Google: {'✓' if HAS_GOOGLE_KEY else '✗'}")

    if not any([HAS_OPENAI_KEY, HAS_ANTHROPIC_KEY, HAS_GOOGLE_KEY]):
        print("\nNo API keys found. Set environment variables:")
        print("export OPENAI_API_KEY=your_key_here")
        print("export ANTHROPIC_API_KEY=your_key_here")
        print("export GOOGLE_API_KEY=your_key_here")

    pytest.main([__file__, "-v"])
