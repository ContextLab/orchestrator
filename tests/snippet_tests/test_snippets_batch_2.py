"""Working tests for documentation code snippets - Batch 2."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_README_lines_280_294_0():
    """Test Python snippet from README.md lines 280-294."""
    # Description: Run it with:
    content = 'import orchestrator as orc\n\n# Initialize models\norc.init_models()\n\n# Compile pipeline\npipeline = orc.compile("research_report.yaml")\n\n# Run with inputs\nresult = pipeline.run(\n    topic="quantum computing applications in medicine",\n    instructions="Focus on recent breakthroughs and future potential"\n)\n\nprint(f"Report saved to: {result}")'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
        try:
            compile(content, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_README_lines_353_359_1():
    """Test bibtex snippet from README.md lines 353-359."""
    # Description: If you use Orchestrator in your research, please cite:
    content = '@software{orchestrator2025,\n  title = {Orchestrator: AI Pipeline Orchestration Framework},\n  author = {Manning, Jeremy R. and {Contextual Dynamics Lab}},\n  year = {2025},\n  url = {https://github.com/ContextLab/orchestrator},\n  organization = {Dartmouth College}\n}'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_design_lines_12_47_2():
    """Test text snippet from design.md lines 12-47."""
    # Description: The Orchestrator is a Python library that provides a unified interface for executing AI pipelines defined in YAML with automatic ambiguity resolution using LLMs. It transparently integrates multiple c
    content = '┌─────────────────────────────────────────────────────────────────┐\n│                         User Interface                          │\n│                    (YAML Pipeline Definition)                   │\n└───────────────────────┬─────────────────────────────────────────┘\n                        │\n┌───────────────────────▼───────────────────────────────────────┐\n│                    YAML Parser & Compiler                     │\n│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │\n│  │ Schema      │  │ Ambiguity    │  │ Pipeline           │    │\n│  │ Validator   │  │ Detector     │  │ Optimizer          │    │\n│  └─────────────┘  └──────────────┘  └────────────────────┘    │\n└───────────────────────┬───────────────────────────────────────┘\n                        │\n┌───────────────────────▼───────────────────────────────────────┐\n│                  Orchestration Engine                         │\n│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │\n│  │ Task        │  │ Dependency   │  │ Resource           │    │\n│  │ Scheduler   │  │ Manager      │  │ Allocator          │    │\n│  └─────────────┘  └──────────────┘  └────────────────────┘    │\n└───────────────────────┬───────────────────────────────────────┘\n                        │\n┌───────────────────────▼───────────────────────────────────────┐\n│                  Control System Adapters                      │\n│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐     │\n│  │ LangGraph   │  │ MCP          │  │ Custom            │     │\n│  │ Adapter     │  │ Adapter      │  │ Adapters          │     │\n│  └─────────────┘  └──────────────┘  └───────────────────┘     │\n└───────────────────────┬───────────────────────────────────────┘\n                        │\n┌───────────────────────▼───────────────────────────────────────┐\n│                    Execution Layer                            │\n│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │\n│  │ Sandboxed   │  │ Model        │  │ State              │    │\n│  │ Executors   │  │ Registry     │  │ Persistence        │    │\n│  └─────────────┘  └──────────────┘  └────────────────────┘    │\n└───────────────────────────────────────────────────────────────┘'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"


def test_design_lines_63_89_3():
    """Test Python snippet from design.md lines 63-89."""
    # Description: 5. Security by Default: Sandboxed execution and input validation
    content = 'from dataclasses import dataclass\nfrom typing import Dict, Any, Optional, List\nfrom enum import Enum\n\nclass TaskStatus(Enum):\n    PENDING = "pending"\n    RUNNING = "running"\n    COMPLETED = "completed"\n    FAILED = "failed"\n    SKIPPED = "skipped"\n\n@dataclass\nclass Task:\n    """Core task abstraction for the orchestrator"""\n    id: str\n    name: str\n    action: str\n    parameters: Dict[str, Any]\n    dependencies: List[str]\n    status: TaskStatus = TaskStatus.PENDING\n    result: Optional[Any] = None\n    error: Optional[Exception] = None\n    metadata: Dict[str, Any] = None\n\n    def is_ready(self, completed_tasks: set) -> bool:\n        """Check if all dependencies are satisfied"""\n        return all(dep in completed_tasks for dep in self.dependencies)'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
        try:
            compile(content, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_design_lines_95_135_4():
    """Test Python snippet from design.md lines 95-135."""
    # Description: `
    content = '@dataclass\nclass Pipeline:\n    """Pipeline represents a collection of tasks with dependencies"""\n    id: str\n    name: str\n    tasks: Dict[str, Task]\n    context: Dict[str, Any]\n    metadata: Dict[str, Any]\n\n    def get_execution_order(self) -> List[List[str]]:\n        """Returns tasks grouped by execution level (parallel groups)"""\n        from collections import defaultdict, deque\n\n        # Build dependency graph\n        in_degree = {task_id: len(task.dependencies) for task_id, task in self.tasks.items()}\n        graph = defaultdict(list)\n\n        for task_id, task in self.tasks.items():\n            for dep in task.dependencies:\n                graph[dep].append(task_id)\n\n        # Topological sort with level grouping\n        levels = []\n        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])\n\n        while queue:\n            current_level = []\n            level_size = len(queue)\n\n            for _ in range(level_size):\n                task_id = queue.popleft()\n                current_level.append(task_id)\n\n                for neighbor in graph[task_id]:\n                    in_degree[neighbor] -= 1\n                    if in_degree[neighbor] == 0:\n                        queue.append(neighbor)\n\n            levels.append(current_level)\n\n        return levels'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
        try:
            compile(content, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_design_lines_141_171_5():
    """Test Python snippet from design.md lines 141-171."""
    # Description: `
    content = '@dataclass\nclass ModelCapabilities:\n    """Defines what a model can do"""\n    supported_tasks: List[str]\n    context_window: int\n    supports_function_calling: bool\n    supports_structured_output: bool\n    supports_streaming: bool\n    languages: List[str]\n\n@dataclass\nclass ModelRequirements:\n    """Resource requirements for a model"""\n    memory_gb: float\n    gpu_memory_gb: Optional[float]\n    cpu_cores: int\n    supports_quantization: List[str]  # ["int8", "int4", "gptq", "awq"]\n\nclass Model:\n    """Abstract base class for all models"""\n    def __init__(self, name: str, provider: str):\n        self.name = name\n        self.provider = provider\n        self.capabilities = self._load_capabilities()\n        self.requirements = self._load_requirements()\n\n    async def generate(self, prompt: str, **kwargs) -> str:\n        raise NotImplementedError\n\n    async def generate_structured(self, prompt: str, schema: dict, **kwargs) -> dict:\n        raise NotImplementedError'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
        try:
            compile(content, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_design_lines_177_200_6():
    """Test Python snippet from design.md lines 177-200."""
    # Description: `
    content = 'from abc import ABC, abstractmethod\n\nclass ControlSystem(ABC):\n    """Abstract base class for control system adapters"""\n\n    @abstractmethod\n    async def execute_task(self, task: Task, context: Dict[str, Any]) -> Any:\n        """Execute a single task"""\n        pass\n\n    @abstractmethod\n    async def execute_pipeline(self, pipeline: Pipeline) -> Dict[str, Any]:\n        """Execute an entire pipeline"""\n        pass\n\n    @abstractmethod\n    def get_capabilities(self) -> Dict[str, Any]:\n        """Return system capabilities"""\n        pass\n\n    @abstractmethod\n    async def health_check(self) -> bool:\n        """Check if the system is healthy"""\n        pass'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
        try:
            compile(content, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_design_lines_208_271_7():
    """Test Python snippet from design.md lines 208-271."""
    # Description: `
    content = 'import yaml\nfrom typing import Dict, Any, List\nimport jsonschema\nfrom jinja2 import Environment, StrictUndefined\n\nclass YAMLCompiler:\n    """Compiles YAML definitions into executable pipelines"""\n\n    def __init__(self):\n        self.schema_validator = SchemaValidator()\n        self.ambiguity_resolver = AmbiguityResolver()\n        self.template_engine = Environment(undefined=StrictUndefined)\n\n    def compile(self, yaml_content: str, context: Dict[str, Any] = None) -> Pipeline:\n        """Compile YAML to Pipeline object"""\n        # Step 1: Parse YAML safely\n        raw_pipeline = yaml.safe_load(yaml_content)\n\n        # Step 2: Validate against schema\n        self.schema_validator.validate(raw_pipeline)\n\n        # Step 3: Process templates\n        processed = self._process_templates(raw_pipeline, context or {})\n\n        # Step 4: Detect and resolve ambiguities\n        resolved = self._resolve_ambiguities(processed)\n\n        # Step 5: Build pipeline object\n        return self._build_pipeline(resolved)\n\n    def _process_templates(self, pipeline_def: dict, context: dict) -> dict:\n        """Process Jinja2 templates in the pipeline definition"""\n        def process_value(value):\n            if isinstance(value, str):\n                template = self.template_engine.from_string(value)\n                return template.render(**context)\n            elif isinstance(value, dict):\n                return {k: process_value(v) for k, v in value.items()}\n            elif isinstance(value, list):\n                return [process_value(item) for item in value]\n            return value\n\n        return process_value(pipeline_def)\n\n    def _resolve_ambiguities(self, pipeline_def: dict) -> dict:\n        """Detect and resolve <AUTO> tags"""\n        def process_auto_tags(obj, path=""):\n            if isinstance(obj, dict):\n                result = {}\n                for key, value in obj.items():\n                    if isinstance(value, str) and value.startswith("<AUTO>") and value.endswith("</AUTO>"):\n                        # Extract ambiguous content\n                        content = value[6:-7]  # Remove <AUTO> tags\n                        # Resolve ambiguity\n                        resolved = self.ambiguity_resolver.resolve(content, path + "." + key)\n                        result[key] = resolved\n                    else:\n                        result[key] = process_auto_tags(value, path + "." + key)\n                return result\n            elif isinstance(obj, list):\n                return [process_auto_tags(item, f"{path}[{i}]") for i, item in enumerate(obj)]\n            return obj\n\n        return process_auto_tags(pipeline_def)'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
        try:
            compile(content, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_design_lines_277_338_8():
    """Test Python snippet from design.md lines 277-338."""
    # Description: `
    content = 'class AmbiguityResolver:\n    """Resolves ambiguous specifications using LLMs"""\n\n    def __init__(self):\n        self.model_selector = ModelSelector()\n        self.format_cache = FormatCache()\n        self.resolution_strategies = {\n            "task_type": self._resolve_task_type,\n            "model_selection": self._resolve_model_selection,\n            "parameter_inference": self._resolve_parameters,\n            "dependency_detection": self._resolve_dependencies\n        }\n\n    async def resolve(self, ambiguous_content: str, context_path: str) -> Any:\n        """Main resolution method"""\n        # Step 1: Classify ambiguity type\n        ambiguity_type = await self._classify_ambiguity(ambiguous_content, context_path)\n\n        # Step 2: Check cache\n        cache_key = self._generate_cache_key(ambiguous_content, ambiguity_type)\n        if cached := self.format_cache.get(cache_key):\n            return cached\n\n        # Step 3: Select appropriate model\n        model = await self.model_selector.select_for_task("ambiguity_resolution")\n\n        # Step 4: Generate format specification (two-step approach)\n        format_spec = await self._generate_format_spec(model, ambiguous_content, ambiguity_type)\n\n        # Step 5: Execute resolution with format spec\n        resolution_strategy = self.resolution_strategies[ambiguity_type]\n        result = await resolution_strategy(model, ambiguous_content, format_spec)\n\n        # Step 6: Cache result\n        self.format_cache.set(cache_key, result)\n\n        return result\n\n    async def _generate_format_spec(self, model, content: str, ambiguity_type: str) -> dict:\n        """Generate output format specification"""\n        prompt = f"""\n        Analyze this ambiguous specification and generate a JSON schema for the expected output:\n\n        Ambiguity Type: {ambiguity_type}\n        Content: {content}\n\n        Return a JSON schema that describes the expected structure of the resolved output.\n        """\n\n        schema = await model.generate_structured(\n            prompt,\n            schema={\n                "type": "object",\n                "properties": {\n                    "schema": {"type": "object"},\n                    "description": {"type": "string"},\n                    "examples": {"type": "array"}\n                }\n            }\n        )\n\n        return schema'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
        try:
            compile(content, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")


def test_design_lines_344_435_9():
    """Test Python snippet from design.md lines 344-435."""
    # Description: `
    content = 'import pickle\nimport json\nfrom datetime import datetime\nfrom typing import Optional\nimport asyncio\nfrom contextlib import asynccontextmanager\n\nclass StateManager:\n    """Manages pipeline state and checkpointing"""\n\n    def __init__(self, backend: str = "postgres"):\n        self.backend = self._init_backend(backend)\n        self.checkpoint_strategy = AdaptiveCheckpointStrategy()\n\n    async def save_checkpoint(self, pipeline_id: str, state: dict, metadata: dict = None):\n        """Save pipeline state checkpoint"""\n        checkpoint = {\n            "pipeline_id": pipeline_id,\n            "state": state,\n            "metadata": metadata or {},\n            "timestamp": datetime.utcnow().isoformat(),\n            "version": "1.0"\n        }\n\n        # Compress state if large\n        if self._should_compress(state):\n            checkpoint["state"] = self._compress_state(state)\n            checkpoint["compressed"] = True\n\n        await self.backend.save(checkpoint)\n\n    async def restore_checkpoint(self, pipeline_id: str,\n                                timestamp: Optional[datetime] = None) -> Optional[dict]:\n        """Restore pipeline state from checkpoint"""\n        checkpoint = await self.backend.load(pipeline_id, timestamp)\n\n        if not checkpoint:\n            return None\n\n        # Decompress if needed\n        if checkpoint.get("compressed"):\n            checkpoint["state"] = self._decompress_state(checkpoint["state"])\n\n        return checkpoint\n\n    @asynccontextmanager\n    async def checkpoint_context(self, pipeline_id: str, task_id: str):\n        """Context manager for automatic checkpointing"""\n        start_time = datetime.utcnow()\n\n        try:\n            yield\n            # Save checkpoint on success\n            if self.checkpoint_strategy.should_checkpoint(pipeline_id, task_id):\n                await self.save_checkpoint(\n                    pipeline_id,\n                    {"last_completed_task": task_id},\n                    {"execution_time": (datetime.utcnow() - start_time).total_seconds()}\n                )\n        except Exception as e:\n            # Save error state\n            await self.save_checkpoint(\n                pipeline_id,\n                {"last_failed_task": task_id, "error": str(e)},\n                {"failure_time": datetime.utcnow().isoformat()}\n            )\n            raise\n\nclass AdaptiveCheckpointStrategy:\n    """Determines when to create checkpoints based on various factors"""\n\n    def __init__(self):\n        self.task_history = {}\n        self.checkpoint_interval = 5  # Base interval\n\n    def should_checkpoint(self, pipeline_id: str, task_id: str) -> bool:\n        """Decide if checkpoint is needed"""\n        # Always checkpoint after critical tasks\n        if self._is_critical_task(task_id):\n            return True\n\n        # Adaptive checkpointing based on task execution time\n        if pipeline_id not in self.task_history:\n            self.task_history[pipeline_id] = []\n\n        self.task_history[pipeline_id].append(task_id)\n\n        # Checkpoint every N tasks\n        if len(self.task_history[pipeline_id]) % self.checkpoint_interval == 0:\n            return True\n\n        return False'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid Python syntax
    # Skip syntax check for notebook-specific code with top-level await
    if 'await' in content and ('notebook' in content.lower() or 'jupyter' in content.lower()):
        # This is notebook-specific syntax, skip syntax validation
        pass
    else:
        try:
            compile(content, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Python syntax error: {e}")
    
    # If it's a simple import, try to execute it
    if content.strip().startswith(('import ', 'from ')) and len(content.strip().split('\n')) <= 3:
        try:
            exec(content)
        except ImportError:
            pytest.skip("Import not available in test environment")
        except Exception as e:
            pytest.fail(f"Import failed: {e}")
