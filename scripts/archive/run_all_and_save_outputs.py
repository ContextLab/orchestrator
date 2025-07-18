#!/usr/bin/env python3
"""
Run all pipelines and properly save their outputs by intercepting save requests.
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


def extract_and_save_content(task_name, result, inputs):
    """Extract save instructions from result and save file if found."""
    if not isinstance(result, str):
        return None
    
    # Common patterns for save instructions
    patterns = [
        # Pattern 1: "Save to path/file.md:\n[content]"
        r'(?:Save|Write)(?:\s+the\s+following)?(?:\s+content)?(?:\s+to|at)\s+([^\n:]+\.md):\s*\n+([\s\S]+)',
        # Pattern 2: "Write...to a markdown file at:\npath/file.md\n\nContent to save:\n[content]"
        r'Write.*?markdown file at:\s*\n\s*([^\n]+\.md)\s*\n+Content to save:\s*\n+([\s\S]+)',
        # Pattern 3: Direct content after mentioning file path
        r'(?:file|path):\s*([^\n]+\.md)\s*\n+#\s+(.+[\s\S]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, result, re.IGNORECASE | re.MULTILINE)
        if match:
            filepath = match.group(1).strip()
            content = match.group(2).strip()
            
            # Process the filepath to replace template variables
            filepath = process_template(filepath, inputs)
            
            # Save the file
            full_path = Path(filepath)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            
            return full_path
    
    # Check if the entire result looks like markdown content to save
    if 'save' in task_name.lower() and result.strip().startswith('#'):
        # Try to extract filepath from task name or use a default
        filename = f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        if 'output/' in result.lower():
            # Try to extract path from content
            path_match = re.search(r'examples/output/[^\s]+\.md', result)
            if path_match:
                filepath = path_match.group(0)
            else:
                filepath = f"examples/output/{filename}"
        else:
            filepath = f"examples/output/{filename}"
        
        # Save the content
        full_path = Path(filepath)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(result)
        
        return full_path
    
    return None


def process_template(text, inputs):
    """Process template variables in text."""
    # Simple template processing for common patterns
    result = text
    
    # Replace {{variable}} patterns
    for key, value in inputs.items():
        if isinstance(value, str):
            result = result.replace(f"{{{{{key}}}}}", value)
            # Also handle filters like | replace(' ', '_') | lower
            pattern = f"{{{{{key}\\s*\\|[^}}]+}}}}"
            if re.search(pattern, result):
                # Apply common filters
                processed_value = value.replace(' ', '_').lower()
                result = re.sub(pattern, processed_value, result)
    
    # Handle timestamp
    result = result.replace("{{execution.timestamp}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    return result


async def run_pipeline_and_save(name, yaml_path, inputs, registry):
    """Run a pipeline and save any outputs."""
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"File: {yaml_path}")
    print(f"{'='*80}")
    
    saved_files = []
    
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
        
        # Process results and save files
        for task_name, result in results.items():
            saved_file = extract_and_save_content(task_name, result, inputs)
            if saved_file:
                saved_files.append(saved_file)
                print(f"📄 Saved: {saved_file} ({saved_file.stat().st_size:,} bytes)")
        
        return {
            "status": "success",
            "duration": duration,
            "saved_files": saved_files,
            "name": name
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "name": name,
            "saved_files": saved_files
        }


async def main():
    """Run all pipelines and save outputs."""
    print("🚀 Running ALL Pipelines with Automatic Output Saving")
    print("="*80)
    
    # Ensure output directory exists
    Path("examples/output").mkdir(parents=True, exist_ok=True)
    
    # Set up models
    registry = setup_models()
    
    # Define all pipelines to run
    all_pipelines = [
        # Working simplified examples
        {
            "name": "Research Pipeline",
            "file": "examples/working/research_simple.yaml",
            "inputs": {
                "query": "Quantum Computing Applications in Finance",
                "depth": "comprehensive"
            }
        },
        {
            "name": "Content Creation",
            "file": "examples/working/content_simple.yaml",
            "inputs": {
                "topic": "Zero Trust Security Architecture",
                "audience": "IT professionals",
                "tone": "informative"
            }
        },
        {
            "name": "Creative Writing",
            "file": "examples/working/creative_simple.yaml",
            "inputs": {
                "theme": "Time Travel Paradoxes",
                "genre": "science fiction",
                "length": "flash"
            }
        },
        {
            "name": "Code Analysis",
            "file": "examples/working/analysis_simple.yaml",
            "inputs": {
                "code_path": "src/orchestrator/compiler",
                "language": "python"
            }
        },
        {
            "name": "Data Processing",
            "file": "examples/working/data_simple.yaml",
            "inputs": {
                "data_source": "sales_analytics_q4_2024",
                "process_type": "analyze"
            }
        },
        
        # Original working examples
        {
            "name": "Interactive Chatbot - Python Learning",
            "file": "examples/interactive_chat_bot_demo.yaml",
            "inputs": {
                "conversation_topic": "Learning Python for Data Science",
                "num_exchanges": 4,
                "user_persona": "beginner-data-analyst",
                "bot_persona": "experienced-instructor"
            }
        },
        {
            "name": "Interactive Chatbot - DevOps",
            "file": "examples/interactive_chat_bot_demo.yaml",
            "inputs": {
                "conversation_topic": "Kubernetes Best Practices",
                "num_exchanges": 3,
                "user_persona": "intermediate-developer",
                "bot_persona": "devops-expert"
            }
        }
    ]
    
    # Run each pipeline
    results = []
    for pipeline_config in all_pipelines:
        if Path(pipeline_config["file"]).exists():
            result = await run_pipeline_and_save(
                pipeline_config["name"],
                pipeline_config["file"],
                pipeline_config["inputs"],
                registry
            )
            results.append(result)
        else:
            print(f"\n⚠️  Skipping {pipeline_config['name']} - file not found")
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    
    print(f"\nTotal pipelines executed: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    
    print("\n📊 Execution Details:")
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        duration = f"{result.get('duration', 0):.1f}s" if result.get('duration') else "N/A"
        files_saved = len(result.get('saved_files', []))
        print(f"{status_icon} {result['name']:<40} ({duration}) - {files_saved} files saved")
    
    # List all saved files
    all_saved_files = []
    for result in results:
        all_saved_files.extend(result.get('saved_files', []))
    
    if all_saved_files:
        print(f"\n📁 Files Created During This Run:")
        for file in all_saved_files:
            if file.exists():
                size = file.stat().st_size
                print(f"  - {file} ({size:,} bytes)")
    
    # Show all output files
    print(f"\n📄 All Files in Output Directory:")
    output_files = list(Path("examples/output").glob("*.md"))
    recent_files = sorted(output_files, key=lambda x: x.stat().st_mtime, reverse=True)[:15]
    
    for file in recent_files:
        size = file.stat().st_size
        mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  - {file.name:<50} ({size:>8,} bytes) [{mod_time}]")
    
    print(f"\n✨ Pipeline execution complete!")
    print(f"📂 Check examples/output/ for all generated content")


if __name__ == "__main__":
    # Check API keys
    print("API Key Status:")
    apis = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "HUGGINGFACE_API_KEY"]
    for api in apis:
        status = "✅" if os.getenv(api) else "❌"
        print(f"  {api}: {status}")
    
    print("\nStarting comprehensive pipeline execution...\n")
    asyncio.run(main())