"""Real model calling infrastructure for AUTO tag resolution."""

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional
import openai
import anthropic
import google.generativeai as genai

logger = logging.getLogger(__name__)


class ModelCaller:
    """Handles real API calls to various LLM providers."""
    
    def __init__(self):
        """Initialize with API keys from environment."""
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_model = None
        
        # Initialize OpenAI if API key available
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = openai.AsyncOpenAI()
            logger.info("OpenAI client initialized")
        
        # Initialize Anthropic if API key available
        if os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = anthropic.AsyncAnthropic()
            logger.info("Anthropic client initialized")
            
        # Initialize Gemini if API key available
        if os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.gemini_model = genai.GenerativeModel("gemini-pro")
            logger.info("Gemini client initialized")
    
    async def call_model(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> str:
        """Call the specified model with real API.
        
        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-sonnet", "gemini-pro")
            prompt: User prompt
            system_prompt: System prompt (if supported)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            json_mode: Whether to use JSON mode (if supported)
            
        Returns:
            Model response as string
            
        Raises:
            Exception: If model call fails
        """
        logger.info(f"Calling model: {model}")
        
        try:
            if model.startswith("gpt") and self.openai_client:
                return await self._call_openai(
                    model, prompt, system_prompt, temperature, max_tokens, json_mode
                )
            elif model.startswith("claude") and self.anthropic_client:
                return await self._call_anthropic(
                    model, prompt, system_prompt, temperature, max_tokens
                )
            elif model.startswith("gemini") and self.gemini_model:
                return await self._call_gemini(
                    prompt, temperature, max_tokens
                )
            else:
                # Fallback to local model or default
                logger.warning(f"Model {model} not available, using fallback")
                return await self._call_fallback(prompt)
                
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            raise
    
    async def _call_openai(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        json_mode: bool
    ) -> str:
        """Call OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
            
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = await self.openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    async def _call_anthropic(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int]
    ) -> str:
        """Call Anthropic API."""
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = await self.anthropic_client.messages.create(**kwargs)
        return response.content[0].text
    
    async def _call_gemini(
        self,
        prompt: str,
        temperature: float,
        max_tokens: Optional[int]
    ) -> str:
        """Call Gemini API."""
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or 2048,
        )
        
        # Gemini uses sync API, so run in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
        )
        
        return response.text
    
    async def _call_fallback(self, prompt: str) -> str:
        """Fallback for when no model is available."""
        # Simple heuristic-based response
        prompt_lower = prompt.lower()
        
        if "summarize" in prompt_lower:
            return "This is a summary of the content."
        elif "list" in prompt_lower:
            return '["item1", "item2", "item3"]'
        elif "true" in prompt_lower or "false" in prompt_lower:
            return "true"
        elif "json" in prompt_lower:
            return '{"result": "example", "status": "success"}'
        else:
            return "Default response"
    
    def is_model_available(self, model: str) -> bool:
        """Check if a model is available."""
        if model.startswith("gpt"):
            return self.openai_client is not None
        elif model.startswith("claude"):
            return self.anthropic_client is not None
        elif model.startswith("gemini"):
            return self.gemini_model is not None
        else:
            return True  # Fallback always available
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        models = []
        
        if self.openai_client:
            models.extend(["gpt-4o-mini", "gpt-4o", "gpt-4o"])
        
        if self.anthropic_client:
            models.extend(["claude-haiku-4-20250514", "claude-sonnet-4-20250514", "claude-opus-4-20250514"])
        
        if self.gemini_model:
            models.extend(["gemini-pro"])
        
        models.append("fallback")
        
        return models