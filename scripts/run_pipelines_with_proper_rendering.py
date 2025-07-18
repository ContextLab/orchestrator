#!/usr/bin/env python3
"""
Run pipelines with proper template rendering and tool execution.
"""

import asyncio
import os
from pathlib import Path
from datetime import datetime
import json
import re
from typing import Dict, Any, List

from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.integrations.anthropic_model import AnthropicModel
from orchestrator.integrations.google_model import GoogleModel
from orchestrator.tools.system_tools import FileSystemTool
from orchestrator.tools.structured_output_handler import StructuredOutputHandler


class ToolAwareControlSystem(ModelBasedControlSystem):
    """Control system that properly handles tool calls."""
    
    def __init__(self, model_registry):
        super().__init__(model_registry)
        self.filesystem_tool = FileSystemTool()
        self.output_handler = StructuredOutputHandler()
    
    async def execute_task(self, task, context=None):
        """Execute task with proper tool handling."""
        # First try structured output parsing for tool calls
        if task.action and isinstance(task.action, str):
            # Check if this looks like a tool call
            tool_indicators = ['save', 'write', 'create', 'file', 'execute', 'run', 'analyze']
            if any(indicator in task.action.lower() for indicator in tool_indicators):
                # Use structured output handler
                structured_response = self.output_handler.ensure_tool_execution(
                    action=task.action[:100],  # First 100 chars for pattern matching
                    response=task.action
                )
                
                # If we got a file operation, handle it
                if structured_response.get('tool') == 'filesystem' and structured_response.get('action') == 'write':
                    params = structured_response.get('parameters', {})
                    filepath = params.get('path', '')
                    content = params.get('content', '')
                    
                    # Render templates
                    filepath = self._render_template(filepath, context)
                    content = self._render_template(content, context)
                    
                    # Save the file
                    try:
                        await self.filesystem_tool.execute(
                            action="write",
                            path=filepath,
                            content=content
                        )
                        return {
                            "success": True,
                            "filepath": filepath,
                            "size": len(content),
                            "message": f"File saved to {filepath}"
                        }
                    except Exception as e:
                        return {
                            "success": False,
                            "error": str(e)
                        }
        
        # Fallback to pattern matching for file save tasks
        if task.action and isinstance(task.action, str):
            # Look for file save patterns
            save_patterns = [
                r'Save.*?to.*?file.*?at:\s*([^\n]+\.md)',
                r'Create a file at:\s*([^\n]+\.md)',
                r'Write.*?to.*?markdown file.*?at:\s*([^\n]+\.md)',
            ]
            
            for pattern in save_patterns:
                match = re.search(pattern, task.action, re.IGNORECASE | re.MULTILINE)
                if match:
                    filepath = match.group(1).strip()
                    
                    # Extract content
                    content_match = re.search(r'Content to save:\s*([\s\S]+)', task.action, re.IGNORECASE)
                    if content_match:
                        content = content_match.group(1).strip()
                        
                        # Render template variables
                        content = self._render_template(content, context)
                        filepath = self._render_template(filepath, context)
                        
                        # Actually save the file
                        try:
                            await self.filesystem_tool.execute(
                                action="write",
                                path=filepath,
                                content=content
                            )
                            return {
                                "success": True,
                                "filepath": filepath,
                                "size": len(content),
                                "message": f"File saved to {filepath}"
                            }
                        except Exception as e:
                            return {
                                "success": False,
                                "error": str(e)
                            }
        
        # Otherwise use normal execution
        return await super().execute_task(task, context)
    
    def _render_template(self, text: str, context: Dict[str, Any]) -> str:
        """Render template variables in text."""
        if not text or not context:
            return text
        
        # Replace {{variable}} patterns
        def replace_var(match):
            var_expr = match.group(1).strip()
            
            # Handle simple variables
            if var_expr in context:
                return str(context[var_expr])
            
            # Handle dot notation (e.g., task.result)
            if '.' in var_expr:
                parts = var_expr.split('.')
                value = context
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return match.group(0)  # Return original if not found
                return str(value)
            
            # Handle filters (e.g., variable | filter)
            if '|' in var_expr:
                var_name, filter_expr = var_expr.split('|', 1)
                var_name = var_name.strip()
                filter_expr = filter_expr.strip()
                
                if var_name in context:
                    value = context[var_name]
                    
                    # Apply common filters
                    if filter_expr == 'lower':
                        return str(value).lower()
                    elif filter_expr == "replace(' ', '_')":
                        return str(value).replace(' ', '_')
                    elif filter_expr.startswith('truncate('):
                        length = int(filter_expr[9:-1])
                        return str(value)[:length]
                
            # Handle execution.timestamp
            if var_expr == 'execution.timestamp':
                return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return match.group(0)  # Return original if not resolved
        
        # Replace all {{...}} patterns
        text = re.sub(r'\{\{([^}]+)\}\}', replace_var, text)
        
        return text


def setup_models():
    """Set up high-quality models only."""
    registry = ModelRegistry()
    
    # Only register the best models
    if os.getenv("ANTHROPIC_API_KEY"):
        registry.register_model(AnthropicModel(model_name="claude-sonnet-4-20250514"))
        print("✓ Anthropic Claude Sonnet 4 registered")
    
    if os.getenv("OPENAI_API_KEY"):
        registry.register_model(OpenAIModel(model_name="gpt-4.1"))
        print("✓ OpenAI GPT-4.1 registered")
    
    if os.getenv("GOOGLE_API_KEY"):
        registry.register_model(GoogleModel(model_name="gemini-2.5-flash"))
        print("✓ Google Gemini 2.5 Flash registered")
    
    return registry


async def run_pipeline(name: str, yaml_path: Path, inputs: Dict[str, Any], registry):
    """Run a pipeline with proper rendering and tool execution."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"File: {yaml_path}")
    print(f"{'='*60}")
    
    try:
        # Read and compile YAML
        yaml_content = yaml_path.read_text()
        compiler = YAMLCompiler()
        
        # Create our enhanced control system
        control_system = ToolAwareControlSystem(registry)
        
        # Compile pipeline
        pipeline = await compiler.compile(yaml_content, inputs)
        print(f"✓ Compiled {len(pipeline.tasks)} tasks")
        
        # Execute pipeline
        start = datetime.now()
        results = await control_system.execute_pipeline(pipeline)
        duration = (datetime.now() - start).total_seconds()
        
        print(f"✅ Completed in {duration:.1f} seconds")
        
        # Check for saved files
        for task_id, result in results.items():
            if isinstance(result, dict) and result.get('filepath'):
                print(f"📄 Saved: {result['filepath']}")
        
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


async def main():
    """Run selected pipelines to test."""
    print("🚀 Running Pipelines with Proper Rendering")
    print("="*60)
    
    # Set up models
    registry = setup_models()
    
    # Test pipelines
    test_pipelines = [
        {
            "name": "Research Simple",
            "file": Path("examples/research_simple.yaml"),
            "inputs": {
                "query": "Future of Quantum Computing",
                "depth": "comprehensive"
            }
        },
        {
            "name": "Content Simple", 
            "file": Path("examples/content_simple.yaml"),
            "inputs": {
                "topic": "Best Practices for API Design",
                "audience": "developers",
                "tone": "technical"
            }
        },
        {
            "name": "Interactive Chat Bot Demo",
            "file": Path("examples/interactive_chat_bot_demo.yaml"),
            "inputs": {
                "conversation_topic": "Understanding Blockchain Technology",
                "num_exchanges": 3,
                "user_persona": "curious-beginner",
                "bot_persona": "expert-teacher"
            }
        }
    ]
    
    # Run each pipeline
    for pipeline_config in test_pipelines:
        if pipeline_config["file"].exists():
            await run_pipeline(
                pipeline_config["name"],
                pipeline_config["file"],
                pipeline_config["inputs"],
                registry
            )
        else:
            print(f"\n⚠️ File not found: {pipeline_config['file']}")
    
    # Check outputs
    print("\n📁 Generated Files:")
    output_dir = Path("examples/output")
    if output_dir.exists():
        for md_file in sorted(output_dir.glob("*.md"))[-10:]:
            size = md_file.stat().st_size
            print(f"  {md_file.name:<50} ({size:>7,} bytes)")


if __name__ == "__main__":
    asyncio.run(main())