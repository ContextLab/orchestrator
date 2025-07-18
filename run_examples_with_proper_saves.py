#!/usr/bin/env python3
"""
Run examples and properly save outputs to files.
"""

import asyncio
import os
from pathlib import Path
from datetime import datetime
import json
import re

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
    
    if os.getenv("OPENAI_API_KEY"):
        registry.register_model(OpenAIModel(model_name="gpt-4o-mini"))
        registry.register_model(OpenAIModel(model_name="gpt-4-turbo-preview"))
    
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


def extract_save_instructions(text):
    """Extract file save instructions from model output."""
    # Look for patterns like "Save to file.md:" or "Save the following to file.md:"
    patterns = [
        r'Save (?:the following )?(?:content )?to ([^:]+\.md):',
        r'Save to ([^:]+\.md):',
        r'Write (?:the following )?to ([^:]+\.md):',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            filepath = match.group(1).strip()
            # Extract content after the colon
            content_start = match.end()
            content = text[content_start:].strip()
            return filepath, content
    
    return None, None


async def run_example_with_save(name, yaml_path, inputs, registry):
    """Run example and handle file saves."""
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
        
        # Check for save instructions in results
        saved_files = []
        for task_name, result in results.items():
            if isinstance(result, str) and ('save' in task_name.lower() or 'file' in task_name.lower()):
                # Check if this contains save instructions
                filepath, content = extract_save_instructions(result)
                if filepath and content:
                    # Actually save the file
                    full_path = Path(filepath)
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(content)
                    saved_files.append(full_path)
                    print(f"📄 Saved output to: {full_path}")
        
        return {
            "status": "success",
            "duration": duration,
            "saved_files": saved_files
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


async def main():
    """Run examples and save outputs."""
    print("🚀 Running Examples with Proper File Saves")
    print("="*80)
    
    # Ensure output directory exists
    Path("examples/output").mkdir(parents=True, exist_ok=True)
    
    # Set up models
    registry = setup_models()
    
    # Define examples to run
    examples = [
        {
            "name": "Research Pipeline",
            "file": "examples/working/research_simple.yaml",
            "inputs": {
                "query": "Machine Learning in Healthcare",
                "depth": "comprehensive"
            }
        },
        {
            "name": "Content Creation",
            "file": "examples/working/content_simple.yaml", 
            "inputs": {
                "topic": "Cloud Native Architecture Patterns",
                "audience": "software engineers",
                "tone": "technical"
            }
        },
        {
            "name": "Creative Writing",
            "file": "examples/working/creative_simple.yaml",
            "inputs": {
                "theme": "AI gaining consciousness",
                "genre": "science fiction",
                "length": "flash"
            }
        },
        {
            "name": "Chatbot Demo",
            "file": "examples/interactive_chat_bot_demo.yaml",
            "inputs": {
                "conversation_topic": "Explaining Docker to beginners",
                "num_exchanges": 3,
                "user_persona": "non-technical-person",
                "bot_persona": "friendly-teacher"
            }
        }
    ]
    
    # Run each example
    results = []
    for example in examples:
        if Path(example["file"]).exists():
            result = await run_example_with_save(
                example["name"],
                example["file"],
                example["inputs"],
                registry
            )
            results.append({
                "name": example["name"],
                **result
            })
    
    # Summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    
    print(f"\nTotal executed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # List all saved files
    all_saved_files = []
    for result in results:
        if result.get("saved_files"):
            all_saved_files.extend(result["saved_files"])
    
    if all_saved_files:
        print(f"\n📁 Files Saved:")
        for file in all_saved_files:
            size = file.stat().st_size if file.exists() else 0
            print(f"  - {file} ({size:,} bytes)")
    
    # Additionally check for any markdown files in output directory
    print(f"\n📄 All Markdown Files in Output Directory:")
    output_files = list(Path("examples/output").glob("*.md"))
    for file in sorted(output_files)[-10:]:  # Show last 10
        size = file.stat().st_size
        print(f"  - {file.name:<50} ({size:>8,} bytes)")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    # Check API keys
    print("API Key Status:")
    apis = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "HUGGINGFACE_API_KEY"]
    for api in apis:
        status = "✅" if os.getenv(api) else "❌"
        print(f"  {api}: {status}")
    
    print("\nStarting execution...\n")
    asyncio.run(main())