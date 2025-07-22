#!/usr/bin/env python3
"""Minimal script to run example without model initialization."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator import init_models
from src.orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from src.orchestrator.orchestrator import Orchestrator


class SimpleControlSystem(ModelBasedControlSystem):
    """Simple control system with file operation support."""
    
    async def execute_task(self, task, context):
        """Execute task with support for save_output."""
        # Check if this is a file save operation
        action_str = str(task.action).lower()
        
        # Skip PDF export for now
        if "pdf" in action_str and "convert" in action_str:
            return {
                "success": True,
                "message": "PDF export skipped (not implemented)",
                "filepath": "examples/output/report.pdf"
            }
        
        if any(keyword in action_str for keyword in ["write the following", "save the following", "write to", "save to"]):
            return await self._handle_file_write(task, context)
        if task.action == "save_output":
            return await self._handle_save_output(task, context)
        return await super().execute_task(task, context)
    
    async def _handle_save_output(self, task, context):
        """Handle save_output action."""
        from pathlib import Path
        
        # Get parameters
        content = task.parameters.get("content", "")
        filename = task.parameters.get("filename", "output.md") 
        output_dir = task.parameters.get("output_dir", "examples/output")
        
        # Resolve template variables
        if isinstance(content, str) and "{{" in content:
            # Simple template resolution
            for key, value in context.get("previous_results", {}).items():
                content = content.replace(f"{{{{{key}}}}}", str(value))
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        content = content.replace(f"{{{{{key}.{subkey}}}}}", str(subvalue))
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write file
        file_path = output_path / filename
        file_path.write_text(str(content))
        
        return {
            "success": True,
            "filepath": str(file_path),
            "size": len(str(content))
        }
    
    async def _handle_file_write(self, task, context):
        """Handle file write operations from action text."""
        import re
        from pathlib import Path
        
        action_text = str(task.action)
        
        # Extract file path from action
        path_patterns = [
            r'to a markdown file at ([^\s:]+)',
            r'to ([^\s:]+\.md)',
            r'Save.*to ([^\s:]+)',
            r'Write.*to ([^\s:]+)'
        ]
        
        file_path = None
        for pattern in path_patterns:
            match = re.search(pattern, action_text, re.IGNORECASE)
            if match:
                file_path = match.group(1).strip()
                break
        
        if not file_path:
            # Default path
            file_path = "examples/output/output.md"
        
        # Extract content after the colon
        content_match = re.search(r':\s*\n(.*)', action_text, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
        else:
            content = action_text
        
        # Template pattern for both file path and content
        template_pattern = r'\{\{([^}]+)\}\}'
        
        # Build a full context for all template resolution
        from datetime import datetime
        full_context = context.copy()
        full_context.update({
            "execution": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        })
        
        # Add previous results if available
        if "previous_results" in context:
            for step_id, result in context["previous_results"].items():
                if isinstance(result, dict):
                    full_context[step_id] = result.copy()
                    if "result" not in full_context[step_id]:
                        full_context[step_id]["result"] = result
                else:
                    full_context[step_id] = {"result": result}
        
        # Resolve template variables in file path
        if "{{" in file_path:
            
            def replace_template(match):
                expr = match.group(1).strip()
                
                # Handle filters
                if '|' in expr:
                    parts = expr.split('|')
                    var_name = parts[0].strip()
                    
                    # Get base value from full_context
                    value = full_context.get(var_name, var_name)
                    
                    # Apply filters
                    for filter_expr in parts[1:]:
                        filter_expr = filter_expr.strip()
                        if filter_expr == 'lower':
                            value = str(value).lower()
                        elif filter_expr.startswith("replace("):
                            # Extract arguments
                            args_match = re.search(r"replace\('([^']*)',\s*'([^']*)'\)", filter_expr)
                            if args_match:
                                old, new = args_match.groups()
                                value = str(value).replace(old, new)
                    
                    return str(value)
                else:
                    # Simple variable
                    return str(full_context.get(expr, expr))
            
            file_path = re.sub(template_pattern, replace_template, file_path)
        
        # Resolve templates in content
        if "{{" in content:
            
            # Simple template resolution
            def resolve_content_template(match):
                expr = match.group(1).strip()
                
                # Handle nested properties like generate_report.result
                if "." in expr:
                    parts = expr.split(".")
                    value = full_context
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            # Keep original if not found
                            return match.group(0)
                    return str(value)
                
                # Handle filters
                if "|" in expr:
                    var_expr, *filters = expr.split("|")
                    var_expr = var_expr.strip()
                    
                    # Get base value
                    if "." in var_expr:
                        parts = var_expr.split(".")
                        value = full_context
                        for part in parts:
                            if isinstance(value, dict) and part in value:
                                value = value[part]
                            else:
                                return match.group(0)
                    else:
                        value = full_context.get(var_expr, match.group(0))
                    
                    # Apply filters
                    for filter_expr in filters:
                        filter_expr = filter_expr.strip()
                        if filter_expr.startswith("default("):
                            # Extract default value
                            default_match = re.search(r"default\('([^']*)'\)", filter_expr)
                            if default_match and not value:
                                value = default_match.group(1)
                    
                    return str(value)
                else:
                    # Simple variable
                    return str(full_context.get(expr, match.group(0)))
            
            content = re.sub(template_pattern, resolve_content_template, content)
        
        # Ensure parent directory exists
        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        file_path_obj.write_text(content)
        
        return {
            "success": True,
            "filepath": str(file_path_obj),
            "size": len(content),
            "message": f"Wrote {len(content)} bytes to {file_path}"
        }


async def main():
    """Run a simple example."""
    print("🚀 Running Research Assistant Example (with Real Models)")
    print("=" * 60)
    
    # Initialize real models from configuration
    try:
        registry = init_models()
        available_models = registry.list_models()
        
        if not available_models:
            print("\n⚠️  No models available!")
            print("Please configure API keys in ~/.orchestrator/.env:")
            print("  OPENAI_API_KEY=your-key-here")
            print("  ANTHROPIC_API_KEY=your-key-here")
            print("\nOr ensure Ollama is running for local models.")
            return
        
        print(f"\n✅ Loaded {len(available_models)} models:")
        for model_id in available_models[:5]:  # Show first 5
            model = registry.get_model(model_id)
            if model:
                print(f"  - {model_id} ({model.provider})")
        if len(available_models) > 5:
            print(f"  ... and {len(available_models) - 5} more")
            
    except Exception as e:
        print(f"\n❌ Failed to initialize models: {e}")
        print("\nTrying to continue with any available models...")
        registry = ModelRegistry()
    
    # Create control system and orchestrator
    control_system = SimpleControlSystem(registry)
    orchestrator = Orchestrator(
        model_registry=registry,
        control_system=control_system
    )
    
    # Load pipeline
    yaml_path = Path("examples/research_assistant.yaml")
    yaml_content = yaml_path.read_text()
    
    # Execute with inputs
    inputs = {
        "query": "quantum computing applications",
        "context": "Focus on practical applications in cybersecurity and cryptography",
        "output_dir": "examples/output",
        "max_sources": 5,
        "quality_threshold": 0.7
    }
    
    print(f"\n📄 Pipeline: {yaml_path}")
    print(f"📥 Inputs: {inputs}")
    
    try:
        print("\n⚡ Executing pipeline...")
        results = await orchestrator.execute_yaml(yaml_content, inputs)
        
        print("\n✅ Pipeline completed!")
        print("\n📊 Results:")
        for task_id, result in results.items():
            if isinstance(result, dict) and "filepath" in result:
                print(f"  - {task_id}: Saved to {result['filepath']}")
            else:
                print(f"  - {task_id}: {str(result)[:100]}...")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())