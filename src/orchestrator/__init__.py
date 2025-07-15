"""
Orchestrator: AI pipeline orchestration framework with intelligent ambiguity resolution.

This package provides a unified interface for executing AI pipelines defined in YAML
with automatic ambiguity resolution using LLMs.
"""

from .core.task import Task, TaskStatus
from .core.pipeline import Pipeline
from .core.model import Model, ModelCapabilities, ModelRequirements
from .core.control_system import ControlSystem
from .compiler.yaml_compiler import YAMLCompiler
from .models.model_registry import ModelRegistry
from .state.state_manager import StateManager
from .orchestrator import Orchestrator
from .integrations.ollama_model import OllamaModel
from .integrations.huggingface_model import HuggingFaceModel
from .tools.mcp_server import default_mcp_server, default_tool_detector
import asyncio
from pathlib import Path

__version__ = "0.1.0"
__author__ = "Contextual Dynamics Lab"
__email__ = "contextualdynamics@gmail.com"

# Global instances
_model_registry = None
_orchestrator = None


def init_models():
    """Initialize the pool of available models by reading models.yaml and environment."""
    global _model_registry
    
    print(">> Initializing model pool...")
    
    _model_registry = ModelRegistry()
    
    # Check for available models
    # Try Ollama first (prefer larger models)
    try:
        # Try gemma2:27b first
        model = OllamaModel(model_name="gemma2:27b", timeout=60)
        if model._is_available:
            _model_registry.register_model(model)
            print(f">>   ✅ Registered Ollama model: gemma2:27b")
        else:
            print(f">>   ⚠️  gemma2:27b not available")
    except Exception as e:
        print(f">>   ⚠️  Error checking gemma2:27b: {e}")
    
    try:
        # Fallback to smaller model
        model = OllamaModel(model_name="llama3.2:1b", timeout=30)
        if model._is_available:
            _model_registry.register_model(model)
            print(f">>   ✅ Registered Ollama model: llama3.2:1b")
    except:
        pass
    
    # Try HuggingFace
    try:
        hf_model = HuggingFaceModel()
        _model_registry.register_model(hf_model)
        print(f">>   ✅ Registered HuggingFace model: {hf_model.name}")
    except:
        pass
    
    if not _model_registry.list_models():
        print(">>   ⚠️  No models available - using mock fallback")
    
    return _model_registry


class OrchestratorPipeline:
    """Wrapper for compiled pipeline that can be called with keyword arguments."""
    
    def __init__(self, pipeline: Pipeline, compiler: YAMLCompiler, orchestrator: Orchestrator):
        self.pipeline = pipeline
        self.compiler = compiler
        self.orchestrator = orchestrator
        self._print_usage()
    
    def _print_usage(self):
        """Print keyword arguments as shown in README."""
        print(">> keyword arguments:")
        
        # Extract inputs from the raw pipeline definition
        inputs = self._extract_inputs()
        
        if inputs:
            for name, info in inputs.items():
                if isinstance(info, dict):
                    desc = info.get('description', 'No description')
                    type_str = info.get('type', 'String').title()
                    required = " (required)" if info.get('required', False) else ""
                    print(f">>   {name}: {desc} (type: {type_str}){required}")
                else:
                    # Simple string description
                    print(f">>   {name}: {info} (type: String)")
        else:
            # Default inputs for research report
            print(">>   topic: a word or underscore-separated phrase specifying the to-be-researched topic (type: String)")
            print(">>   instructions: detailed instructions to help guide the report, specify areas of particular interest (or areas to stay away from), etc. (type: String)")
    
    def _extract_inputs(self):
        """Extract input definitions from the compiled pipeline."""
        # The inputs are stored in the pipeline's metadata during compilation
        if hasattr(self.pipeline, 'metadata') and 'inputs' in self.pipeline.metadata:
            return self.pipeline.metadata['inputs']
        
        # If not in metadata, try to get from the original definition
        # This is a fallback - we should enhance the compilation process
        return {}
    
    def run(self, **kwargs):
        """Run the pipeline with given keyword arguments."""
        # Run pipeline asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._run_async(**kwargs))
        finally:
            loop.close()
    
    async def _run_async(self, **kwargs):
        """Async pipeline execution."""
        # Validate required inputs
        self._validate_inputs(kwargs)
        
        # Resolve outputs using AUTO tags
        outputs = await self._resolve_outputs(kwargs)
        
        # Create full context from kwargs and resolved outputs
        context = {
            'inputs': kwargs,
            'outputs': outputs
        }
        
        # Apply runtime template resolution to pipeline tasks
        resolved_pipeline = await self._resolve_runtime_templates(self.pipeline, context)
        
        # Execute pipeline
        results = await self.orchestrator.execute_pipeline(resolved_pipeline, context)
        
        # Return the final output (PDF path or report content)
        if outputs and 'pdf' in outputs:
            return outputs['pdf']
        elif 'final_report' in results:
            return results['final_report']
        else:
            return results
    
    def _validate_inputs(self, kwargs):
        """Validate that required inputs are provided."""
        inputs_def = self._extract_inputs()
        
        for name, info in inputs_def.items():
            if isinstance(info, dict) and info.get('required', False):
                if name not in kwargs:
                    raise ValueError(f"Required input '{name}' not provided")
    
    async def _resolve_outputs(self, inputs):
        """Resolve output definitions using AUTO tags."""
        outputs = {}
        outputs_def = self._extract_outputs()
        
        if outputs_def:
            from jinja2 import Template
            for name, value in outputs_def.items():
                if isinstance(value, str):
                    if value.startswith("<AUTO>") and value.endswith("</AUTO>"):
                        # Resolve AUTO tag
                        auto_content = value[6:-7]  # Remove <AUTO> tags
                        if hasattr(self.orchestrator.yaml_compiler, 'ambiguity_resolver'):
                            resolved = await self.orchestrator.yaml_compiler.ambiguity_resolver.resolve(auto_content, f"outputs.{name}")
                            outputs[name] = resolved
                        else:
                            outputs[name] = f"report_{inputs.get('topic', 'research')}.pdf"
                    else:
                        # Regular template - render with current context
                        try:
                            template = Template(value)
                            outputs[name] = template.render(inputs=inputs, outputs=outputs)
                        except Exception:
                            outputs[name] = value
                else:
                    outputs[name] = value
        
        return outputs
    
    def _extract_outputs(self):
        """Extract output definitions from the compiled pipeline."""
        if hasattr(self.pipeline, 'metadata') and 'outputs' in self.pipeline.metadata:
            return self.pipeline.metadata['outputs']
        return {}
    
    async def _resolve_runtime_templates(self, pipeline, context):
        """Resolve templates in pipeline tasks at runtime."""
        from jinja2 import Template
        import copy
        
        # Create a deep copy to avoid modifying the original
        resolved_pipeline = copy.deepcopy(pipeline)
        
        # Resolve templates in each task
        for task_id, task in resolved_pipeline.tasks.items():
            if hasattr(task, 'parameters'):
                task.parameters = await self._resolve_task_templates(task.parameters, context)
        
        return resolved_pipeline
    
    async def _resolve_task_templates(self, obj, context):
        """Recursively resolve templates in task parameters."""
        from jinja2 import Template
        
        if isinstance(obj, str):
            if "{{" in obj and "}}" in obj:
                try:
                    template = Template(obj)
                    return template.render(**context)
                except Exception as e:
                    # If template resolution fails, return original
                    return obj
            return obj
        elif isinstance(obj, dict):
            return {k: await self._resolve_task_templates(v, context) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [await self._resolve_task_templates(item, context) for item in obj]
        else:
            return obj


async def compile_async(yaml_path: str):
    """Compile a YAML pipeline file into an executable OrchestratorPipeline (async version)."""
    global _orchestrator, _model_registry
    
    # Ensure models are initialized
    if _model_registry is None:
        init_models()
    
    # Create orchestrator with mock control system (will be replaced)
    from .core.control_system import MockControlSystem
    control_system = MockControlSystem()
    _orchestrator = Orchestrator(control_system=control_system)
    
    # Set up model for ambiguity resolution
    model_keys = _model_registry.list_models() if _model_registry else []
    if model_keys:
        best_model_key = model_keys[0]  # Assume first is best (gemma2:27b)
        # Get model object directly from the models dict
        best_model = _model_registry.models[best_model_key]
        _orchestrator.yaml_compiler.ambiguity_resolver.model = best_model
        print(f">> Using model for AUTO resolution: {best_model_key}")
    
    # Load and compile YAML
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        yaml_content = f.read()
    
    # Parse YAML to detect required tools
    import yaml as yaml_lib
    pipeline_def = yaml_lib.safe_load(yaml_content)
    
    # Auto-detect and register tools
    required_tools = default_tool_detector.detect_tools_from_yaml(pipeline_def)
    if required_tools:
        print(f">> Detected required tools: {', '.join(required_tools)}")
        availability = default_tool_detector.ensure_tools_available(required_tools)
        for tool, available in availability.items():
            status = "✅" if available else "❌"
            print(f">>   {status} {tool}")
    
    # Start MCP server if tools are required
    if required_tools and any(availability.values()):
        print(">> Starting MCP tool server...")
        await default_mcp_server.start_server()
    
    # Compile pipeline
    pipeline = await _orchestrator.yaml_compiler.compile(yaml_content, {})
    
    # Return callable pipeline
    return OrchestratorPipeline(pipeline, _orchestrator.yaml_compiler, _orchestrator)


def compile(yaml_path: str):
    """Compile a YAML pipeline file into an executable OrchestratorPipeline."""
    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context, need to run in a new thread or return a coroutine
        import concurrent.futures
        import threading
        
        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(compile_async(yaml_path))
            finally:
                new_loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_loop)
            return future.result()
            
    except RuntimeError:
        # No event loop running, we can create one
        return asyncio.run(compile_async(yaml_path))


__all__ = [
    "Task",
    "TaskStatus", 
    "Pipeline",
    "Model",
    "ModelCapabilities",
    "ModelRequirements",
    "ControlSystem",
    "YAMLCompiler",
    "ModelRegistry",
    "StateManager",
    "Orchestrator",
    "init_models",
    "compile",
]