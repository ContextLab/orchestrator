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
        
        # Extract inputs from pipeline
        if hasattr(self.pipeline, 'metadata') and 'inputs' in self.pipeline.metadata:
            inputs = self.pipeline.metadata['inputs']
            if isinstance(inputs, dict):
                for name, info in inputs.items():
                    desc = info if isinstance(info, str) else info.get('description', 'No description')
                    type_str = "String"  # Default type
                    print(f">>   {name}: {desc} (type: {type_str})")
        else:
            # Default inputs for research report
            print(">>   topic: a word or underscore-separated phrase specifying the to-be-researched topic (type: String)")
            print(">>   instructions: detailed instructions to help guide the report, specify areas of particular interest (or areas to stay away from), etc. (type: String)")
    
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
        # Create context from kwargs
        context = {
            'inputs': kwargs,
            'outputs': {}
        }
        
        # Execute pipeline
        results = await self.orchestrator.execute_pipeline(self.pipeline, context)
        
        # Return the final output (PDF path or report content)
        if 'outputs' in context and 'pdf' in context['outputs']:
            return context['outputs']['pdf']
        elif 'final_report' in results:
            return results['final_report']
        else:
            return results


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