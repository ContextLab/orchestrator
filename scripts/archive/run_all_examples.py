#!/usr/bin/env python3
"""
Run all example pipelines and assess output quality.
"""

import asyncio
import os
from pathlib import Path
from datetime import datetime
import json

from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.integrations.anthropic_model import AnthropicModel
from orchestrator.integrations.google_model import GoogleModel
from orchestrator.integrations.ollama_model import OllamaModel
from orchestrator.integrations.huggingface_model import HuggingFaceModel


async def setup_model_registry():
    """Set up model registry with all available models."""
    registry = ModelRegistry()
    
    # Register models based on available API keys
    if os.getenv("OPENAI_API_KEY"):
        registry.register_model(OpenAIModel(model_name="gpt-4o-mini"))
        registry.register_model(OpenAIModel(model_name="gpt-4-turbo-preview"))
        print("✓ OpenAI models registered")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        registry.register_model(AnthropicModel(model_name="claude-3-haiku-20240307"))
        registry.register_model(AnthropicModel(model_name="claude-3-sonnet-20240229"))
        print("✓ Anthropic models registered")
    
    if os.getenv("GOOGLE_API_KEY"):
        registry.register_model(GoogleModel(model_name="gemini-1.5-flash"))
        print("✓ Google models registered")
    
    if os.getenv("HUGGINGFACE_API_KEY"):
        registry.register_model(HuggingFaceModel(model_name="mistralai/Mistral-7B-Instruct-v0.2"))
        print("✓ HuggingFace models registered")
    
    # Ollama doesn't need API key if running locally
    try:
        registry.register_model(OllamaModel(model_name="llama2"))
        print("✓ Ollama models registered")
    except:
        print("⚠️  Ollama not available (ensure it's running locally)")
    
    return registry


async def run_example(yaml_path: Path, inputs: dict, registry: ModelRegistry):
    """Run a single example pipeline."""
    print(f"\n{'='*60}")
    print(f"Running: {yaml_path.name}")
    print(f"{'='*60}")
    
    try:
        # Read YAML
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
        
        # Setup
        control_system = ModelBasedControlSystem(registry)
        compiler = YAMLCompiler()
        
        # Compile and run
        print(f"Inputs: {inputs}")
        pipeline = await compiler.compile(yaml_content, inputs)
        print(f"Pipeline compiled with {len(pipeline.tasks)} tasks")
        
        start_time = datetime.now()
        results = await control_system.execute_pipeline(pipeline)
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Completed in {duration:.1f} seconds")
        
        # Check for saved file
        for task_name, task_result in results.items():
            if 'save' in task_name and isinstance(task_result, str):
                if 'examples/output/' in task_result:
                    print(f"📄 Output saved to: {task_result}")
        
        return {
            "status": "success",
            "duration": duration,
            "results": results
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


async def run_all_examples():
    """Run all example pipelines with appropriate inputs."""
    # Ensure output directory exists
    Path("examples/output").mkdir(parents=True, exist_ok=True)
    
    # Set up models
    registry = await setup_model_registry()
    
    # Define examples with their inputs
    examples = [
        # Research Assistant
        {
            "file": "research_assistant.yaml",
            "inputs": {
                "query": "Impact of AI on software development",
                "context": "Focus on productivity and code quality improvements",
                "max_sources": 5
            }
        },
        
        # Interactive Chatbot Demo
        {
            "file": "interactive_chat_bot_demo.yaml",
            "inputs": {
                "conversation_topic": "Future of programming languages",
                "num_exchanges": 3
            }
        },
        
        # Content Creation
        {
            "file": "content_creation_pipeline.yaml",
            "inputs": {
                "topic": "Best practices for API design",
                "formats": ["blog"],
                "audience": "software developers",
                "target_length": 1000
            }
        },
        
        # Creative Writing
        {
            "file": "creative_writing_assistant.yaml",
            "inputs": {
                "genre": "science fiction",
                "length": "flash",
                "theme": "AI consciousness",
                "target_audience": "adults"
            }
        },
        
        # Financial Analysis
        {
            "file": "financial_analysis_bot.yaml",
            "inputs": {
                "symbols": ["AAPL", "GOOGL"],
                "analysis_type": "technical",
                "time_period": "1Y",
                "include_fundamentals": False
            }
        },
        
        # Data Processing
        {
            "file": "data_processing_workflow.yaml",
            "inputs": {
                "source": "sample_data.csv",
                "output_path": "processed_data",
                "output_format": "parquet",
                "chunk_size": 1000,
                "validation_rules": [
                    {"field": "id", "type": "required"},
                    {"field": "value", "type": "numeric"}
                ]
            }
        }
    ]
    
    # Run each example
    results = []
    for example in examples:
        yaml_path = Path("examples") / example["file"]
        if yaml_path.exists():
            result = await run_example(yaml_path, example["inputs"], registry)
            results.append({
                "example": example["file"],
                "status": result["status"],
                "duration": result.get("duration", 0),
                "error": result.get("error", None)
            })
        else:
            print(f"\n⚠️  Skipping {example['file']} - file not found")
    
    # Summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r["status"] == "success")
    total = len(results)
    
    print(f"\nTotal: {total} examples")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    
    print("\nResults by example:")
    for result in results:
        status = "✅" if result["status"] == "success" else "❌"
        duration = f"{result['duration']:.1f}s" if result["duration"] else "N/A"
        print(f"  {status} {result['example']} - {duration}")
        if result["error"]:
            print(f"     Error: {result['error'][:100]}...")
    
    # Save summary
    summary_path = Path("examples/output/execution_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_examples": total,
            "successful": successful,
            "failed": total - successful,
            "results": results
        }, f, indent=2)
    
    print(f"\n📊 Summary saved to: {summary_path}")
    print("\n🔍 Check examples/output/ directory for generated markdown files!")


if __name__ == "__main__":
    print("Running All Orchestrator Examples")
    print("This will test various AI models and save outputs")
    print("-" * 60)
    
    # Check API keys
    api_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GOOGLE_API_KEY": "Google",
        "HUGGINGFACE_API_KEY": "HuggingFace"
    }
    
    available = []
    missing = []
    
    for key, name in api_keys.items():
        if os.getenv(key):
            available.append(name)
        else:
            missing.append(name)
    
    print(f"\nAvailable APIs: {', '.join(available) if available else 'None'}")
    if missing:
        print(f"Missing APIs: {', '.join(missing)}")
        print("\nNote: Examples using missing APIs will fail")
        print("Set environment variables to enable more models")
    
    print("\nStarting execution...")
    asyncio.run(run_all_examples())