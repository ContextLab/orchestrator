#!/usr/bin/env python3
"""
Run selected examples to generate outputs.
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


def setup_models():
    """Set up available models."""
    registry = ModelRegistry()
    
    if os.getenv("OPENAI_API_KEY"):
        registry.register_model(OpenAIModel(model_name="gpt-4o-mini"))
    
    if os.getenv("ANTHROPIC_API_KEY"):
        registry.register_model(AnthropicModel(model_name="claude-3-haiku-20240307"))
    
    if os.getenv("GOOGLE_API_KEY"):
        registry.register_model(GoogleModel(model_name="gemini-1.5-flash"))
    
    return registry


def save_output_from_result(result_text, default_filename="output"):
    """Extract and save markdown content from result."""
    # If result contains markdown formatting, save it
    if isinstance(result_text, str) and (result_text.strip().startswith('#') or '# ' in result_text):
        # Clean filename from any path instructions in the text
        filename = default_filename
        
        # Look for file path patterns
        path_patterns = [
            r'examples/output/([^\s]+\.md)',
            r'Save to ([^\s]+\.md)',
            r'file:\s*([^\s]+\.md)',
        ]
        
        for pattern in path_patterns:
            match = re.search(pattern, result_text)
            if match:
                filename = match.group(1)
                if '/' in filename:
                    filename = filename.split('/')[-1]
                break
        
        # Ensure .md extension
        if not filename.endswith('.md'):
            filename = f"{filename}.md"
        
        # Save the content
        filepath = Path(f"examples/output/{filename}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract just the markdown content if it's embedded in instructions
        content = result_text
        
        # If the content has instructions before the markdown, extract the markdown part
        if ":\n\n#" in content:
            parts = content.split(":\n\n#", 1)
            if len(parts) > 1:
                content = "#" + parts[1]
        elif ":\n#" in content:
            parts = content.split(":\n#", 1)
            if len(parts) > 1:
                content = "#" + parts[1]
        
        filepath.write_text(content)
        return filepath
    
    return None


async def run_example(name, yaml_path, inputs, registry):
    """Run a single example."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    try:
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
        
        control_system = ModelBasedControlSystem(registry)
        compiler = YAMLCompiler()
        
        pipeline = await compiler.compile(yaml_content, inputs)
        results = await control_system.execute_pipeline(pipeline)
        
        print(f"✅ Pipeline completed successfully")
        
        # Save outputs
        saved_files = []
        for task_id, result in results.items():
            if 'save' in task_id.lower() or 'report' in task_id.lower():
                # Create filename from inputs
                topic = inputs.get('query', inputs.get('topic', inputs.get('theme', inputs.get('conversation_topic', 'output'))))
                safe_topic = topic.replace(' ', '_').lower()[:50]
                filename = f"{task_id}_{safe_topic}"
                
                saved_file = save_output_from_result(result, filename)
                if saved_file:
                    saved_files.append(saved_file)
                    print(f"📄 Saved: {saved_file.name}")
        
        return {"status": "success", "files": saved_files}
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"status": "error", "error": str(e)}


async def main():
    """Run selected examples."""
    print("🚀 Running Selected Examples")
    print("="*60)
    
    Path("examples/output").mkdir(parents=True, exist_ok=True)
    registry = setup_models()
    
    # Run a few key examples
    examples = [
        {
            "name": "Research: AI in Education",
            "file": "examples/working/research_simple.yaml",
            "inputs": {
                "query": "Artificial Intelligence in Modern Education",
                "depth": "comprehensive"
            }
        },
        {
            "name": "Content: Microservices",
            "file": "examples/working/content_simple.yaml",
            "inputs": {
                "topic": "Microservices Architecture Best Practices",
                "audience": "software architects",
                "tone": "technical"
            }
        },
        {
            "name": "Story: First Contact",
            "file": "examples/working/creative_simple.yaml",
            "inputs": {
                "theme": "First Contact with Alien Life",
                "genre": "science fiction",
                "length": "flash"
            }
        },
        {
            "name": "Chatbot: Learning JavaScript",
            "file": "examples/interactive_chat_bot_demo.yaml",
            "inputs": {
                "conversation_topic": "Learning JavaScript from Scratch",
                "num_exchanges": 3,
                "user_persona": "complete-beginner",
                "bot_persona": "patient-teacher"
            }
        }
    ]
    
    results = []
    for example in examples:
        if Path(example["file"]).exists():
            print(f"\n📋 {example['name']}")
            result = await run_example(
                example["name"],
                example["file"],
                example["inputs"],
                registry
            )
            results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if r["status"] == "success")
    print(f"\nCompleted: {successful}/{len(results)} pipelines")
    
    # List generated files
    print("\n📁 Generated Files:")
    output_files = sorted(Path("examples/output").glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]
    
    for f in output_files:
        size = f.stat().st_size
        print(f"  - {f.name:<50} ({size:>7,} bytes)")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    asyncio.run(main())