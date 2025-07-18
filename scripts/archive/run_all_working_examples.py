#!/usr/bin/env python3
"""
Run all working example pipelines to generate comprehensive outputs.
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


def setup_models():
    """Set up all available models."""
    registry = ModelRegistry()
    
    # Register all available models
    if os.getenv("OPENAI_API_KEY"):
        registry.register_model(OpenAIModel(model_name="gpt-4o-mini"))
        registry.register_model(OpenAIModel(model_name="gpt-4-turbo-preview"))
        registry.register_model(OpenAIModel(model_name="gpt-4"))
    
    if os.getenv("ANTHROPIC_API_KEY"):
        registry.register_model(AnthropicModel(model_name="claude-3-haiku-20240307"))
        registry.register_model(AnthropicModel(model_name="claude-3-sonnet-20240229"))
    
    if os.getenv("GOOGLE_API_KEY"):
        registry.register_model(GoogleModel(model_name="gemini-1.5-flash"))
    
    if os.getenv("HUGGINGFACE_API_KEY"):
        registry.register_model(HuggingFaceModel(model_name="mistralai/Mistral-7B-Instruct-v0.2"))
    
    try:
        registry.register_model(OllamaModel(model_name="llama2"))
    except:
        pass
    
    return registry


async def run_example(name, yaml_path, inputs, registry):
    """Run a single example and save outputs."""
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"File: {yaml_path}")
    print(f"{'='*80}")
    
    try:
        # Read YAML
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
        
        # Setup
        control_system = ModelBasedControlSystem(registry)
        compiler = YAMLCompiler()
        
        # Compile and run
        print(f"Inputs: {json.dumps(inputs, indent=2)}")
        pipeline = await compiler.compile(yaml_content, inputs)
        
        start = datetime.now()
        results = await control_system.execute_pipeline(pipeline)
        duration = (datetime.now() - start).total_seconds()
        
        print(f"✅ Success! Completed in {duration:.1f} seconds")
        
        # Check for saved files
        for task_name, result in results.items():
            if 'save' in task_name.lower() and isinstance(result, str):
                if 'examples/output/' in result:
                    print(f"📄 Output saved by pipeline")
        
        return {
            "status": "success",
            "duration": duration,
            "name": name
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "name": name
        }


async def main():
    """Run all working examples."""
    print("🚀 Running All Working Example Pipelines")
    print("="*80)
    
    # Ensure output directory exists
    Path("examples/output").mkdir(parents=True, exist_ok=True)
    
    # Set up models
    registry = setup_models()
    
    # Define all working examples
    examples = [
        # Working simplified examples
        {
            "name": "Research Pipeline (Simple)",
            "file": "examples/working/research_simple.yaml",
            "inputs": {
                "query": "The future of artificial intelligence in education",
                "depth": "comprehensive"
            }
        },
        {
            "name": "Content Creation (Simple)",
            "file": "examples/working/content_simple.yaml",
            "inputs": {
                "topic": "Best practices for microservices architecture",
                "audience": "software architects",
                "tone": "technical"
            }
        },
        {
            "name": "Creative Writing (Simple)",
            "file": "examples/working/creative_simple.yaml",
            "inputs": {
                "theme": "humanity's first contact with aliens",
                "genre": "science fiction",
                "length": "flash"
            }
        },
        {
            "name": "Code Analysis (Simple)",
            "file": "examples/working/analysis_simple.yaml",
            "inputs": {
                "code_path": "src/orchestrator/core",
                "language": "python"
            }
        },
        {
            "name": "Data Processing (Simple)",
            "file": "examples/working/data_simple.yaml",
            "inputs": {
                "data_source": "customer_behavior_dataset",
                "process_type": "analyze"
            }
        },
        
        # Original working examples
        {
            "name": "Interactive Chatbot Demo",
            "file": "examples/interactive_chat_bot_demo.yaml",
            "inputs": {
                "conversation_topic": "Machine learning algorithms explained simply",
                "num_exchanges": 3,
                "user_persona": "beginner-programmer",
                "bot_persona": "patient-teacher"
            }
        },
        {
            "name": "Multi-Model Pipeline",
            "file": "examples/multi_model_pipeline.yaml",
            "inputs": {
                "topic": "climate change solutions",
                "models_to_use": ["openai", "anthropic"]
            }
        },
        {
            "name": "Simple Pipeline",
            "file": "examples/simple_pipeline.yaml",
            "inputs": {
                "message": "Explain quantum entanglement",
                "style": "educational"
            }
        }
    ]
    
    # Run each example
    results = []
    for example in examples:
        if Path(example["file"]).exists():
            result = await run_example(
                example["name"],
                example["file"],
                example["inputs"],
                registry
            )
            results.append(result)
        else:
            print(f"\n⚠️  Skipping {example['name']} - file not found")
    
    # Summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    
    print(f"\nTotal executed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    print("\nResults:")
    for result in results:
        icon = "✅" if result["status"] == "success" else "❌"
        duration = f"{result.get('duration', 0):.1f}s" if result.get('duration') else "N/A"
        print(f"  {icon} {result['name']:<40} ({duration})")
    
    # List generated files
    print("\n📁 Generated Output Files:")
    output_files = list(Path("examples/output").glob("*.md"))
    recent_files = sorted(output_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]
    
    for file in recent_files:
        size = file.stat().st_size
        print(f"  - {file.name:<50} ({size:,} bytes)")
    
    print("\n✨ Done! Check examples/output/ for all generated content.")


if __name__ == "__main__":
    # Check API keys
    print("API Key Status:")
    apis = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "HUGGINGFACE_API_KEY"]
    for api in apis:
        status = "✅" if os.getenv(api) else "❌"
        print(f"  {api}: {status}")
    
    print("\nStarting pipeline execution...\n")
    asyncio.run(main())