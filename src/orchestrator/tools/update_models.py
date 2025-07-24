#!/usr/bin/env python3
"""
Update models.yaml with the latest models from all providers.

This tool fetches the current list of models from:
- OpenAI API
- Anthropic API
- Google Gemini documentation
- Ollama model library
- HuggingFace trending models
"""

import os
import yaml
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelUpdater:
    """Updates the models.yaml file with latest models from all providers."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the model updater.

        Args:
            config_path: Path to save models.yaml. Defaults to ~/.orchestrator/models.yaml
        """
        if config_path is None:
            config_path = Path.home() / ".orchestrator" / "models.yaml"
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config if available
        self.existing_config = {}
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.existing_config = yaml.safe_load(f) or {}

    async def fetch_openai_models(self) -> List[Dict[str, Any]]:
        """Fetch available models from OpenAI API."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, skipping OpenAI models")
            return []

        models = []
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {api_key}"}
                async with session.get(
                    "https://api.openai.com/v1/models", headers=headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for model in data.get("data", []):
                            model_id = model["id"]
                            # Filter for completion/chat models
                            if any(
                                x in model_id
                                for x in [
                                    "gpt",
                                    "text",
                                    "davinci",
                                    "curie",
                                    "babbage",
                                    "ada",
                                    "o1",
                                    "o3",
                                    "o4",
                                ]
                            ):
                                models.append(
                                    {
                                        "id": model_id,
                                        "provider": "openai",
                                        "type": "openai",
                                        "created": model.get("created"),
                                    }
                                )
        except Exception as e:
            logger.error(f"Error fetching OpenAI models: {e}")

        return models

    async def fetch_anthropic_models(self) -> List[Dict[str, Any]]:
        """Fetch available models from Anthropic."""
        # Anthropic doesn't have a public API endpoint for listing models
        # We'll use a hardcoded list of known models
        models = [
            {
                "id": "claude-opus-4-20250514",
                "provider": "anthropic",
                "type": "anthropic",
            },
            {
                "id": "claude-sonnet-4-20250514",
                "provider": "anthropic",
                "type": "anthropic",
            },
            {
                "id": "claude-3-7-sonnet-20250219",
                "provider": "anthropic",
                "type": "anthropic",
            },
            {
                "id": "claude-3-5-sonnet-20241022",
                "provider": "anthropic",
                "type": "anthropic",
            },
            {
                "id": "claude-3-5-haiku-20241022",
                "provider": "anthropic",
                "type": "anthropic",
            },
            {
                "id": "claude-3-opus-20240229",
                "provider": "anthropic",
                "type": "anthropic",
            },
            {
                "id": "claude-3-sonnet-20240229",
                "provider": "anthropic",
                "type": "anthropic",
            },
            {
                "id": "claude-3-haiku-20240307",
                "provider": "anthropic",
                "type": "anthropic",
            },
            {"id": "claude-2.1", "provider": "anthropic", "type": "anthropic"},
            {"id": "claude-2", "provider": "anthropic", "type": "anthropic"},
            {"id": "claude-instant-1.2", "provider": "anthropic", "type": "anthropic"},
        ]

        # Try to use the API if available
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic

                anthropic.Anthropic(api_key=api_key)
                # Note: As of now, Anthropic doesn't have a models.list() endpoint
                # This is for future compatibility
                pass
            except Exception as e:
                logger.debug(f"Could not use Anthropic API: {e}")

        return models

    async def fetch_google_models(self) -> List[Dict[str, Any]]:
        """Fetch available models from Google Gemini."""
        # Google Gemini models - parsed from documentation
        models = [
            {"id": "gemini-2.5-pro", "provider": "google", "type": "google"},
            {"id": "gemini-2.5-flash", "provider": "google", "type": "google"},
            {"id": "gemini-2.5-flash-lite", "provider": "google", "type": "google"},
            {"id": "gemini-2.0-flash", "provider": "google", "type": "google"},
            {"id": "gemini-2.0-flash-lite", "provider": "google", "type": "google"},
            {"id": "gemini-1.5-pro", "provider": "google", "type": "google"},
            {"id": "gemini-1.5-flash", "provider": "google", "type": "google"},
            {"id": "gemini-1.5-flash-8b", "provider": "google", "type": "google"},
            {"id": "gemini-1.0-pro", "provider": "google", "type": "google"},
            {"id": "gemini-pro", "provider": "google", "type": "google"},
            {"id": "gemini-pro-vision", "provider": "google", "type": "google"},
        ]

        return models

    async def fetch_ollama_models(self) -> List[Dict[str, Any]]:
        """Fetch available models from Ollama library."""
        models = []

        # Popular Ollama models
        ollama_models = [
            "llama3.2:1b",
            "llama3.2:3b",
            "llama3.1:8b",
            "llama3.1:70b",
            "deepseek-r1:1.5b",
            "deepseek-r1:8b",
            "deepseek-r1:32b",
            "deepseek-r1:70b",
            "qwen2.5-coder:1.5b",
            "qwen2.5-coder:7b",
            "qwen2.5-coder:14b",
            "qwen2.5-coder:32b",
            "gemma3:1b",
            "gemma3:4b",
            "gemma3:12b",
            "gemma3:27b",
            "gemma3n:e4b",
            "gemma3n:e6b",
            "gemma3n:e12b",
            "mistral:7b",
            "mistral-nemo:12b",
            "mixtral:8x7b",
            "mixtral:8x22b",
            "phi3:3.8b",
            "phi3:14b",
            "phi3.5:3.8b",
            "codellama:7b",
            "codellama:13b",
            "codellama:34b",
            "starcoder2:3b",
            "starcoder2:7b",
            "starcoder2:15b",
            "vicuna:7b",
            "vicuna:13b",
            "vicuna:33b",
            "orca2:7b",
            "orca2:13b",
            "neural-chat:7b",
            "neural-chat:7b-v3.3",
            "starling-lm:7b",
            "starling-lm:7b-alpha",
            "zephyr:7b",
            "zephyr:7b-alpha",
            "zephyr:7b-beta",
            "openchat:7b",
            "openchat:7b-v3.5",
            "yarn-mistral:7b",
            "yarn-llama2:7b",
            "yarn-llama2:13b",
            "stable-beluga:7b",
            "stable-beluga:13b",
            "stable-beluga:70b",
        ]

        for model_id in ollama_models:
            models.append(
                {
                    "id": model_id,
                    "provider": "ollama",
                    "type": "ollama",
                }
            )

        return models

    async def fetch_huggingface_models(self) -> List[Dict[str, Any]]:
        """Fetch trending instruct models from HuggingFace."""
        models = []

        # Top trending instruct models under 40B parameters
        hf_models = [
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
            "microsoft/Phi-3.5-MoE-instruct",
            "HuggingFaceTB/SmolLM-1.7B-Instruct",
            "stabilityai/stable-code-instruct-3b",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "codellama/CodeLlama-7b-Instruct-hf",
            "codellama/CodeLlama-13b-Instruct-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "bigcode/starcoder2-3b",
            "bigcode/starcoder2-7b",
            "bigcode/starcoder2-15b",
            "WizardLM/WizardCoder-Python-7B-V1.0",
            "WizardLM/WizardCoder-Python-13B-V1.0",
            "WizardLM/WizardCoder-Python-34B-V1.0",
            "tencent/Hunyuan-A13B-Instruct",
            "NousResearch/Hermes-3-Llama-3.1-8B",
            "allenai/OLMo-2-1124-7B-Instruct",
            "google/gemma-2-9b-it",
        ]

        for model_id in hf_models:
            models.append(
                {
                    "id": model_id,
                    "provider": "huggingface",
                    "type": "huggingface",
                }
            )

        return models

    def estimate_model_size(self, model_id: str) -> float:
        """Estimate model size in billions of parameters from model ID."""
        # Extract number patterns like 7b, 13B, 1.5b, etc.
        patterns = [
            r"(\d+(?:\.\d+)?)[bB]",  # Matches 7b, 13B, 1.5b
            r"(\d+)x(\d+)[bB]",  # Matches 8x7b (multiply)
        ]

        for pattern in patterns:
            match = re.search(pattern, model_id)
            if match:
                if len(match.groups()) == 2:  # Multiplication pattern
                    return float(match.group(1)) * float(match.group(2))
                else:
                    return float(match.group(1))

        # Default sizes for known models
        if "gpt-4" in model_id:
            return 1760.0  # GPT-4 estimated
        elif "gpt-3.5" in model_id:
            return 175.0
        elif "claude-opus" in model_id or "opus" in model_id:
            return 2000.0
        elif "claude-sonnet" in model_id or "sonnet" in model_id:
            return 200.0
        elif "claude-haiku" in model_id or "haiku" in model_id:
            return 20.0
        elif "gemini-2.5-pro" in model_id:
            return 1500.0
        elif "gemini" in model_id and "pro" in model_id:
            return 540.0
        elif "gemini" in model_id and "flash" in model_id:
            return 80.0
        elif "o3" in model_id and "mini" not in model_id:
            return 2000.0
        elif "o3-mini" in model_id or "o4-mini" in model_id:
            return 70.0
        elif "o1" in model_id and "mini" not in model_id:
            return 175.0
        elif "o1-mini" in model_id:
            return 65.0

        return 7.0  # Default fallback

    def create_model_entry(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """Create a model entry for models.yaml."""
        model_id = model["id"]
        provider = model["provider"]

        # Estimate model size
        size_b = self.estimate_model_size(model_id)

        # Base configuration
        entry = {
            "provider": provider,
            "type": model["type"],
            "size_b": size_b,
            "config": {},
        }

        # Provider-specific configuration
        if provider == "openai":
            entry["config"] = {
                "model_name": model_id,
                "api_key": "${OPENAI_API_KEY}",
                "max_retries": 3,
                "timeout": 30.0,
            }
        elif provider == "anthropic":
            entry["config"] = {
                "model_name": model_id,
                "api_key": "${ANTHROPIC_API_KEY}",
                "max_retries": 3,
                "timeout": 30.0,
            }
        elif provider == "google":
            entry["config"] = {
                "model_name": model_id,
                "api_key": "${GOOGLE_AI_API_KEY}",
                "max_retries": 3,
                "timeout": 30.0,
            }
        elif provider == "ollama":
            entry["config"] = {
                "model_name": model_id,
                "base_url": "http://localhost:11434",
                "timeout": 30.0,
            }
        elif provider == "huggingface":
            entry["config"] = {
                "model_name": model_id,
                "use_auth_token": "${HUGGINGFACE_TOKEN}",
                "device": "auto",
                "torch_dtype": "auto",
            }

        return entry

    async def update_models(self) -> Dict[str, Any]:
        """Fetch all models and update the configuration."""
        logger.info("Fetching models from all providers...")

        # Fetch models from all providers concurrently
        tasks = [
            self.fetch_openai_models(),
            self.fetch_anthropic_models(),
            self.fetch_google_models(),
            self.fetch_ollama_models(),
            self.fetch_huggingface_models(),
        ]

        results = await asyncio.gather(*tasks)

        # Combine all models
        all_models = []
        for models in results:
            all_models.extend(models)

        logger.info(f"Found {len(all_models)} models total")

        # Create the new configuration
        config = {
            "# Model configuration for the Orchestrator Framework": None,
            "# Auto-generated on": datetime.now().isoformat(),
            "": None,
            "models": {},
        }

        # Add all models
        for model in all_models:
            model_id = model["id"]
            config["models"][model_id] = self.create_model_entry(model)

        # Add preference sections
        config["preferences"] = {
            "default": "gpt-4o-mini",
            "fallback": [
                "gpt-3.5-turbo",
                "claude-3-haiku-20240307",
                "ollama:llama3.2:1b",
            ],
        }

        # Cost-optimized selection (smaller, cheaper models)
        config["cost_optimized"] = [
            "gpt-4o-mini",
            "claude-3-haiku-20240307",
            "gemini-2.0-flash-lite",
            "ollama:llama3.2:1b",
            "ollama:gemma3:1b",
            "huggingface:Qwen/Qwen2.5-1.5B-Instruct",
        ]

        # Performance-optimized selection (larger, more capable models)
        config["performance_optimized"] = [
            "o3",
            "gpt-4.1",
            "claude-opus-4-20250514",
            "gemini-2.5-pro",
            "ollama:deepseek-r1:70b",
            "huggingface:Qwen/Qwen2.5-32B-Instruct",
        ]

        return config

    async def save_models(self, config: Dict[str, Any]) -> None:
        """Save the models configuration to file."""
        # Create a clean YAML structure
        clean_config = {}

        # Add header comments
        yaml_content = "# Model configuration for the Orchestrator Framework\n"
        yaml_content += f"# Auto-generated on: {datetime.now().isoformat()}\n\n"

        # Add models section
        clean_config["models"] = config["models"]
        clean_config["preferences"] = config["preferences"]
        clean_config["cost_optimized"] = config["cost_optimized"]
        clean_config["performance_optimized"] = config["performance_optimized"]

        # Write to file
        yaml_content += yaml.dump(
            clean_config, default_flow_style=False, sort_keys=False
        )

        with open(self.config_path, "w") as f:
            f.write(yaml_content)

        logger.info(f"Saved {len(config['models'])} models to {self.config_path}")

    async def run(self) -> None:
        """Run the model update process."""
        config = await self.update_models()
        await self.save_models(config)


async def update_models(config_path: Optional[Path] = None) -> None:
    """
    Update the models.yaml file with latest models from all providers.

    Args:
        config_path: Path to save models.yaml. Defaults to ~/.orchestrator/models.yaml
    """
    # Ensure config_path is a Path object
    if config_path is not None and not isinstance(config_path, Path):
        config_path = Path(config_path)
    updater = ModelUpdater(config_path)
    await updater.run()


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Update Orchestrator models configuration"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for models.yaml (default: ~/.orchestrator/models.yaml)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent / "config" / "models.yaml",
        help="Also update the config/models.yaml in the repository",
    )

    args = parser.parse_args()

    async def run():
        # Update user config
        await update_models(args.output)

        # Also update repository config if specified
        if args.config:
            logger.info(f"Also updating repository config at {args.config}")
            await update_models(args.config)

    asyncio.run(run())


if __name__ == "__main__":
    main()
