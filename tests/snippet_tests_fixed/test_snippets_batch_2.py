"""Tests for documentation code snippets - Batch 2 (Fixed)."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set up test environment
os.environ.setdefault('ORCHESTRATOR_CONFIG', str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml"))

# Note: Set RUN_REAL_TESTS=1 to enable tests that use real models
# API keys should be set as environment variables:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY  
# - GOOGLE_AI_API_KEY


def test_design_lines_441_553_0():
    """Test Python snippet from design.md lines 441-553."""
    # `
    
    code = r"""import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ModelMetrics:
    \"""Performance metrics for a model\"""
    latency_p50: float
    latency_p95: float
    throughput: float
    accuracy: float
    cost_per_token: float

class ModelRegistry:
    \"""Central registry for all available models\"""

    def __init__(self):
        self.models: Dict[str, Model] = {}
        self.metrics: Dict[str, ModelMetrics] = {}
        self.bandit = UCBModelSelector()

    def register_model(self, model: Model, metrics: ModelMetrics):
        \"""Register a new model\"""
        self.models[model.name] = model
        self.metrics[model.name] = metrics

    async def select_model(self, requirements: Dict[str, Any]) -> Model:
        \"""Select best model for given requirements\"""
        # Step 1: Filter by capabilities
        eligible_models = self._filter_by_capabilities(requirements)

        # Step 2: Filter by available resources
        available_models = await self._filter_by_resources(eligible_models)

        # Step 3: Use multi-armed bandit for selection
        selected_model_name = self.bandit.select(
            [m.name for m in available_models],
            requirements
        )

        return self.models[selected_model_name]

    def _filter_by_capabilities(self, requirements: Dict[str, Any]) -> List[Model]:
        \"""Filter models by required capabilities\"""
        eligible = []

        for model in self.models.values():
            if self._meets_requirements(model, requirements):
                eligible.append(model)

        return eligible

    async def _filter_by_resources(self, models: List[Model]) -> List[Model]:
        \"""Filter models by available system resources\"""
        system_resources = await self._get_system_resources()
        available = []

        for model in models:
            if self._can_run_on_system(model, system_resources):
                available.append(model)

        # If no models fit, try quantized versions
        if not available:
            available = await self._find_quantized_alternatives(models, system_resources)

        return available

class UCBModelSelector:
    \"""Upper Confidence Bound algorithm for model selection\"""

    def __init__(self, exploration_factor: float = 2.0):
        self.exploration_factor = exploration_factor
        self.model_stats = {}  # Track performance per model

    def select(self, model_names: List[str], context: Dict[str, Any]) -> str:
        \"""Select model using UCB algorithm\"""
        if not model_names:
            raise ValueError("No models available")

        # Initialize stats for new models
        for name in model_names:
            if name not in self.model_stats:
                self.model_stats[name] = {
                    "successes": 0,
                    "attempts": 0,
                    "total_reward": 0.0
                }

        # Calculate UCB scores
        scores = {}
        total_attempts = sum(stats["attempts"] for stats in self.model_stats.values())

        for name in model_names:
            stats = self.model_stats[name]
            if stats["attempts"] == 0:
                scores[name] = float('inf')  # Explore untried models
            else:
                avg_reward = stats["total_reward"] / stats["attempts"]
                exploration_bonus = self.exploration_factor * np.sqrt(
                    np.log(total_attempts + 1) / stats["attempts"]
                )
                scores[name] = avg_reward + exploration_bonus

        # Select model with highest score
        return max(scores, key=scores.get)

    def update_reward(self, model_name: str, reward: float):
        \"""Update model statistics after execution\"""
        if model_name in self.model_stats:
            self.model_stats[model_name]["attempts"] += 1
            self.model_stats[model_name]["total_reward"] += reward
            if reward > 0:
                self.model_stats[model_name]["successes"] += 1"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

def test_design_lines_559_677_1():
    """Test Python snippet from design.md lines 559-677."""
    # `
    
    code = r"""from enum import Enum
from typing import Optional, Callable
import asyncio

class ErrorSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ErrorCategory(Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN = "unknown"

class ErrorHandler:
    \"""Comprehensive error handling system\"""

    def __init__(self):
        self.error_strategies = {
            ErrorCategory.RATE_LIMIT: self._handle_rate_limit,
            ErrorCategory.TIMEOUT: self._handle_timeout,
            ErrorCategory.RESOURCE_EXHAUSTION: self._handle_resource_exhaustion,
            ErrorCategory.VALIDATION_ERROR: self._handle_validation_error,
            ErrorCategory.SYSTEM_ERROR: self._handle_system_error
        }
        self.circuit_breaker = CircuitBreaker()

    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        \"""Main error handling method\"""
        # Classify error
        category = self._classify_error(error)
        severity = self._determine_severity(error, context)

        # Log error with full context
        await self._log_error(error, category, severity, context)

        # Apply circuit breaker
        if self.circuit_breaker.is_open(context.get("system_id")):
            raise SystemUnavailableError("System circuit breaker is open")

        # Execute error handling strategy
        strategy = self.error_strategies.get(category, self._handle_unknown)
        result = await strategy(error, context, severity)

        # Update circuit breaker
        if severity == ErrorSeverity.CRITICAL:
            self.circuit_breaker.record_failure(context.get("system_id"))

        return result

    async def _handle_rate_limit(self, error: Exception, context: Dict[str, Any],
                                severity: ErrorSeverity) -> Dict[str, Any]:
        \"""Handle rate limit errors\"""
        retry_after = self._extract_retry_after(error) or 60

        if severity == ErrorSeverity.LOW:
            # Wait and retry
            await asyncio.sleep(retry_after)
            return {"action": "retry", "delay": retry_after}
        else:
            # Switch to alternative system
            return {
                "action": "switch_system",
                "reason": "rate_limit_exceeded",
                "retry_after": retry_after
            }

    async def _handle_timeout(self, error: Exception, context: Dict[str, Any],
                             severity: ErrorSeverity) -> Dict[str, Any]:
        \"""Handle timeout errors\"""
        if context.get("retry_count", 0) < 3:
            # Retry with increased timeout
            new_timeout = context.get("timeout", 30) * 2
            return {
                "action": "retry",
                "timeout": new_timeout,
                "retry_count": context.get("retry_count", 0) + 1
            }
        else:
            # Mark task as failed
            return {"action": "fail", "reason": "timeout_exceeded"}

class CircuitBreaker:
    \"""Circuit breaker pattern implementation\"""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = {}
        self.last_failure_time = {}

    def is_open(self, system_id: str) -> bool:
        \"""Check if circuit breaker is open for a system\"""
        if system_id not in self.failures:
            return False

        # Check if timeout has passed
        if system_id in self.last_failure_time:
            time_since_failure = time.time() - self.last_failure_time[system_id]
            if time_since_failure > self.timeout:
                # Reset circuit breaker
                self.failures[system_id] = 0
                return False

        return self.failures[system_id] >= self.failure_threshold

    def record_failure(self, system_id: str):
        \"""Record a failure for a system\"""
        self.failures[system_id] = self.failures.get(system_id, 0) + 1
        self.last_failure_time[system_id] = time.time()

    def record_success(self, system_id: str):
        \"""Record a success for a system\"""
        if system_id in self.failures:
            self.failures[system_id] = max(0, self.failures[system_id] - 1)"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_685_740_2():
    """Test YAML pipeline from design.md lines 685-740."""
    import yaml
    import orchestrator
    
    yaml_content = r"""# schema/pipeline-schema.yaml
$schema: "http://json-schema.org/draft-07/schema#"
type: object
required:
  - name
  - version
  - steps
properties:
  name:
    type: string
    pattern: "^[a-zA-Z][a-zA-Z0-9_-]*$"
  version:
    type: string
    pattern: "^\\d+\\.\\d+\\.\\d+$"
  description:
    type: string
  metadata:
    type: object
  context:
    type: object
    properties:
      timeout:
        type: integer
        minimum: 1
      max_retries:
        type: integer
        minimum: 0
      checkpoint_strategy:
        type: string
        enum: ["adaptive", "fixed", "none"]
  steps:
    type: array
    minItems: 1
    items:
      type: object
      required:
        - id
        - action
      properties:
        id:
          type: string
          pattern: "^[a-zA-Z][a-zA-Z0-9_-]*$"
        action:
          type: string
        parameters:
          type: object
        dependencies:
          type: array
          items:
            type: string
        on_failure:
          type: string
          enum: ["continue", "fail", "retry", "skip"]
        timeout:
          type: integer
          minimum: 1"""
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip actual pipeline compilation unless we have real models
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        # Check for API keys
        if not (os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')):
            pytest.skip("No API keys available for pipeline testing")
        
        try:
            # Initialize models
            registry = orchestrator.init_models()
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
            
            # Create compiler
            from orchestrator.compiler import YAMLCompiler
            compiler = YAMLCompiler()
            compiler.set_model_registry(registry)
            
            # Compile the pipeline (but don't run it)
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
                
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_746_794_3():
    """Test YAML pipeline from design.md lines 746-794."""
    import yaml
    import orchestrator
    
    yaml_content = r"""# example-pipeline.yaml
name: research_report_pipeline
version: 1.0.0
description: Generate a comprehensive research report on a given topic

context:
  timeout: 3600
  max_retries: 3
  checkpoint_strategy: adaptive

steps:
  - id: topic_analysis
    action: analyze
    parameters:
      input: "{{ topic }}"
      analysis_type: <AUTO>Determine the best analysis approach for this topic</AUTO>
      output_format: <AUTO>Choose appropriate format: bullet_points, narrative, or structured</AUTO>

  - id: research_planning
    action: plan
    parameters:
      topic_analysis: "{{ steps.topic_analysis.output }}"
      research_depth: <AUTO>Based on topic complexity, choose: shallow, medium, or deep</AUTO>
      sources: <AUTO>Determine number and types of sources needed</AUTO>
    dependencies: [topic_analysis]

  - id: web_search
    action: search
    parameters:
      queries: <AUTO>Generate search queries based on research plan</AUTO>
      num_results: <AUTO>Determine optimal number of results per query</AUTO>
    dependencies: [research_planning]

  - id: content_synthesis
    action: synthesize
    parameters:
      sources: "{{ steps.web_search.results }}"
      style: <AUTO>Choose writing style: academic, business, or general</AUTO>
      length: <AUTO>Determine appropriate length based on topic</AUTO>
    dependencies: [web_search]

  - id: report_generation
    action: generate_report
    parameters:
      content: "{{ steps.content_synthesis.output }}"
      format: markdown
      sections: <AUTO>Organize content into appropriate sections</AUTO>
    dependencies: [content_synthesis]
    on_failure: retry"""
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip actual pipeline compilation unless we have real models
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        # Check for API keys
        if not (os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')):
            pytest.skip("No API keys available for pipeline testing")
        
        try:
            # Initialize models
            registry = orchestrator.init_models()
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
            
            # Create compiler
            from orchestrator.compiler import YAMLCompiler
            compiler = YAMLCompiler()
            compiler.set_model_registry(registry)
            
            # Compile the pipeline (but don't run it)
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
                
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_802_836_4():
    """Test orchestrator code from design.md lines 802-836."""
    # `
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        if 'hello_world.yaml' in r"""# main.py
import asyncio
from orchestrator import Orchestrator, YAMLCompiler
from orchestrator.models import ModelRegistry
from orchestrator.storage import PostgresBackend

async def main():
    # Initialize orchestrator
    orchestrator = Orchestrator(
        storage_backend=PostgresBackend("postgresql://localhost/orchestrator"),
        model_registry=ModelRegistry.from_config("models.yaml")
    )

    # Load and compile pipeline
    with open("research_pipeline.yaml") as f:
        yaml_content = f.read()

    compiler = YAMLCompiler()
    pipeline = await compiler.compile(
        yaml_content,
        context={"topic": "quantum computing applications in cryptography"}
    )

    # Execute pipeline
    try:
        result = await orchestrator.execute_pipeline(pipeline)
        print(f"Pipeline completed successfully: {result}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        # Attempt recovery
        recovered_result = await orchestrator.recover_pipeline(pipeline.id)
        print(f"Recovery result: {recovered_result}")

if __name__ == "__main__":
    asyncio.run(main())""":
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(r"""# main.py
import asyncio
from orchestrator import Orchestrator, YAMLCompiler
from orchestrator.models import ModelRegistry
from orchestrator.storage import PostgresBackend

async def main():
    # Initialize orchestrator
    orchestrator = Orchestrator(
        storage_backend=PostgresBackend("postgresql://localhost/orchestrator"),
        model_registry=ModelRegistry.from_config("models.yaml")
    )

    # Load and compile pipeline
    with open("research_pipeline.yaml") as f:
        yaml_content = f.read()

    compiler = YAMLCompiler()
    pipeline = await compiler.compile(
        yaml_content,
        context={"topic": "quantum computing applications in cryptography"}
    )

    # Execute pipeline
    try:
        result = await orchestrator.execute_pipeline(pipeline)
        print(f"Pipeline completed successfully: {result}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        # Attempt recovery
        recovered_result = await orchestrator.recover_pipeline(pipeline.id)
        print(f"Recovery result: {recovered_result}")

if __name__ == "__main__":
    asyncio.run(main())""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_842_903_5():
    """Test orchestrator code from design.md lines 842-903."""
    # `
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        if 'hello_world.yaml' in r"""# adapters/custom_adapter.py
from orchestrator.adapters import ControlSystem
from orchestrator.models import Task

class CustomMLAdapter(ControlSystem):
    \"""Example adapter for a custom ML framework\"""

    def __init__(self, config: dict):
        self.config = config
        self.ml_engine = self._init_ml_engine()

    async def execute_task(self, task: Task, context: dict) -> Any:
        \"""Execute task using custom ML framework\"""
        # Map task action to ML operation
        operation = self._map_action_to_operation(task.action)

        # Prepare inputs
        inputs = self._prepare_inputs(task.parameters, context)

        # Execute with monitoring
        async with self._monitor_execution(task.id):
            result = await self.ml_engine.run(operation, inputs)

        # Post-process results
        return self._process_results(result, task.metadata)

    async def execute_pipeline(self, pipeline: Pipeline) -> dict:
        \"""Execute entire pipeline with optimizations\"""
        # Build execution graph
        exec_graph = self._build_execution_graph(pipeline)

        # Optimize graph (fusion, parallelization)
        optimized_graph = self._optimize_graph(exec_graph)

        # Execute with checkpointing
        results = {}
        for level in optimized_graph.levels:
            # Execute tasks in parallel
            level_results = await asyncio.gather(*[
                self.execute_task(task, pipeline.context)
                for task in level
            ])

            # Store results
            for task, result in zip(level, level_results):
                results[task.id] = result

        return results

    def get_capabilities(self) -> dict:
        return {
            "supported_operations": ["train", "predict", "evaluate", "optimize"],
            "parallel_execution": True,
            "gpu_acceleration": True,
            "distributed": self.config.get("distributed", False)
        }

    async def health_check(self) -> bool:
        try:
            return await self.ml_engine.ping()
        except Exception:
            return False""":
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(r"""# adapters/custom_adapter.py
from orchestrator.adapters import ControlSystem
from orchestrator.models import Task

class CustomMLAdapter(ControlSystem):
    \"""Example adapter for a custom ML framework\"""

    def __init__(self, config: dict):
        self.config = config
        self.ml_engine = self._init_ml_engine()

    async def execute_task(self, task: Task, context: dict) -> Any:
        \"""Execute task using custom ML framework\"""
        # Map task action to ML operation
        operation = self._map_action_to_operation(task.action)

        # Prepare inputs
        inputs = self._prepare_inputs(task.parameters, context)

        # Execute with monitoring
        async with self._monitor_execution(task.id):
            result = await self.ml_engine.run(operation, inputs)

        # Post-process results
        return self._process_results(result, task.metadata)

    async def execute_pipeline(self, pipeline: Pipeline) -> dict:
        \"""Execute entire pipeline with optimizations\"""
        # Build execution graph
        exec_graph = self._build_execution_graph(pipeline)

        # Optimize graph (fusion, parallelization)
        optimized_graph = self._optimize_graph(exec_graph)

        # Execute with checkpointing
        results = {}
        for level in optimized_graph.levels:
            # Execute tasks in parallel
            level_results = await asyncio.gather(*[
                self.execute_task(task, pipeline.context)
                for task in level
            ])

            # Store results
            for task, result in zip(level, level_results):
                results[task.id] = result

        return results

    def get_capabilities(self) -> dict:
        return {
            "supported_operations": ["train", "predict", "evaluate", "optimize"],
            "parallel_execution": True,
            "gpu_acceleration": True,
            "distributed": self.config.get("distributed", False)
        }

    async def health_check(self) -> bool:
        try:
            return await self.ml_engine.ping()
        except Exception:
            return False""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

def test_design_lines_909_1012_6():
    """Test Python snippet from design.md lines 909-1012."""
    # `
    
    code = r"""# sandbox/executor.py
import docker
import asyncio
from typing import Dict, Any, Optional

class SandboxedExecutor:
    \"""Secure sandboxed code execution\"""

    def __init__(self, docker_client=None):
        self.docker = docker_client or docker.from_env()
        self.containers = {}
        self.resource_limits = {
            "memory": "1g",
            "cpu_quota": 50000,
            "pids_limit": 100
        }

    async def execute_code(
        self,
        code: str,
        language: str,
        environment: Dict[str, str],
        timeout: int = 30
    ) -> Dict[str, Any]:
        \"""Execute code in sandboxed environment\"""
        # Create container
        container = await self._create_container(language, environment)

        try:
            # Start container
            await self._start_container(container)

            # Execute code with timeout
            result = await asyncio.wait_for(
                self._run_code(container, code),
                timeout=timeout
            )

            return {
                "success": True,
                "output": result["output"],
                "errors": result.get("errors", ""),
                "execution_time": result["execution_time"]
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Execution timeout exceeded",
                "timeout": timeout
            }
        finally:
            # Cleanup
            await self._cleanup_container(container)

    async def _create_container(
        self,
        language: str,
        environment: Dict[str, str]
    ) -> docker.models.containers.Container:
        \"""Create isolated container for code execution\"""
        image = self._get_image_for_language(language)

        container = self.docker.containers.create(
            image=image,
            command="sleep infinity",  # Keep container running
            detach=True,
            mem_limit=self.resource_limits["memory"],
            cpu_quota=self.resource_limits["cpu_quota"],
            pids_limit=self.resource_limits["pids_limit"],
            network_mode="none",  # No network access
            read_only=True,
            tmpfs={"/tmp": "rw,noexec,nosuid,size=100m"},
            environment=environment,
            security_opt=["no-new-privileges:true"],
            user="nobody"  # Run as non-root
        )

        self.containers[container.id] = container
        return container

    async def _run_code(
        self,
        container: docker.models.containers.Container,
        code: str
    ) -> Dict[str, Any]:
        \"""Execute code inside container\"""
        # Write code to container
        code_file = "/tmp/code.py"
        container.exec_run(f"sh -c 'cat > {code_file}'", stdin=True).input = code.encode()

        # Execute code
        start_time = asyncio.get_event_loop().time()
        result = container.exec_run(f"python {code_file}", demux=True)
        execution_time = asyncio.get_event_loop().time() - start_time

        stdout, stderr = result.output

        return {
            "output": stdout.decode() if stdout else "",
            "errors": stderr.decode() if stderr else "",
            "exit_code": result.exit_code,
            "execution_time": execution_time
        }"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_1020_1062_7():
    """Test orchestrator code from design.md lines 1020-1062."""
    # `
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        if 'hello_world.yaml' in r"""# tests/test_ambiguity_resolver.py
import pytest
from orchestrator.compiler import AmbiguityResolver

@pytest.mark.asyncio
async def test_ambiguity_resolution():
    resolver = AmbiguityResolver()

    # Test simple ambiguity
    result = await resolver.resolve(
        "Choose the best format for displaying data",
        "steps.display.format"
    )

    assert result in ["table", "chart", "list", "json"]

    # Test complex ambiguity with context
    result = await resolver.resolve(
        "Determine optimal batch size based on available memory",
        "steps.processing.batch_size"
    )

    assert isinstance(result, int)
    assert 1 <= result <= 1000

@pytest.mark.asyncio
async def test_nested_ambiguity_resolution():
    resolver = AmbiguityResolver()

    nested_content = {
        "query": "<AUTO>Generate search query for topic</AUTO>",
        "filters": {
            "date_range": "<AUTO>Choose appropriate date range</AUTO>",
            "sources": "<AUTO>Select relevant sources</AUTO>"
        }
    }

    result = await resolver.resolve_nested(nested_content, "steps.search")

    assert "query" in result
    assert "filters" in result
    assert "date_range" in result["filters"]
    assert "sources" in result["filters"]""":
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(r"""# tests/test_ambiguity_resolver.py
import pytest
from orchestrator.compiler import AmbiguityResolver

@pytest.mark.asyncio
async def test_ambiguity_resolution():
    resolver = AmbiguityResolver()

    # Test simple ambiguity
    result = await resolver.resolve(
        "Choose the best format for displaying data",
        "steps.display.format"
    )

    assert result in ["table", "chart", "list", "json"]

    # Test complex ambiguity with context
    result = await resolver.resolve(
        "Determine optimal batch size based on available memory",
        "steps.processing.batch_size"
    )

    assert isinstance(result, int)
    assert 1 <= result <= 1000

@pytest.mark.asyncio
async def test_nested_ambiguity_resolution():
    resolver = AmbiguityResolver()

    nested_content = {
        "query": "<AUTO>Generate search query for topic</AUTO>",
        "filters": {
            "date_range": "<AUTO>Choose appropriate date range</AUTO>",
            "sources": "<AUTO>Select relevant sources</AUTO>"
        }
    }

    result = await resolver.resolve_nested(nested_content, "steps.search")

    assert "query" in result
    assert "filters" in result
    assert "date_range" in result["filters"]
    assert "sources" in result["filters"]""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_1068_1116_8():
    """Test orchestrator code from design.md lines 1068-1116."""
    # `
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        if 'hello_world.yaml' in r"""# tests/test_pipeline_execution.py
import pytest
from orchestrator import Orchestrator

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_execution():
    orchestrator = Orchestrator()

    pipeline_yaml = \"""
    name: test_pipeline
    version: 1.0.0
    steps:
      - id: step1
        action: generate
        parameters:
          prompt: "Hello, world!"
      - id: step2
        action: transform
        parameters:
          input: "{{ steps.step1.output }}"
          transformation: uppercase
        dependencies: [step1]
    \"""

    result = await orchestrator.execute_yaml(pipeline_yaml)

    assert result["status"] == "completed"
    assert "HELLO, WORLD!" in result["outputs"]["step2"]

@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_recovery():
    orchestrator = Orchestrator()

    # Simulate failure and recovery
    pipeline_id = "test_recovery_pipeline"

    # Create checkpoint
    await orchestrator.state_manager.save_checkpoint(
        pipeline_id,
        {"completed_steps": ["step1", "step2"], "failed_step": "step3"}
    )

    # Attempt recovery
    result = await orchestrator.recover_pipeline(pipeline_id)

    assert result["recovered"] == True
    assert result["resumed_from"] == "step3"""":
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(r"""# tests/test_pipeline_execution.py
import pytest
from orchestrator import Orchestrator

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_execution():
    orchestrator = Orchestrator()

    pipeline_yaml = \"""
    name: test_pipeline
    version: 1.0.0
    steps:
      - id: step1
        action: generate
        parameters:
          prompt: "Hello, world!"
      - id: step2
        action: transform
        parameters:
          input: "{{ steps.step1.output }}"
          transformation: uppercase
        dependencies: [step1]
    \"""

    result = await orchestrator.execute_yaml(pipeline_yaml)

    assert result["status"] == "completed"
    assert "HELLO, WORLD!" in result["outputs"]["step2"]

@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_recovery():
    orchestrator = Orchestrator()

    # Simulate failure and recovery
    pipeline_id = "test_recovery_pipeline"

    # Create checkpoint
    await orchestrator.state_manager.save_checkpoint(
        pipeline_id,
        {"completed_steps": ["step1", "step2"], "failed_step": "step3"}
    )

    # Attempt recovery
    result = await orchestrator.recover_pipeline(pipeline_id)

    assert result["recovered"] == True
    assert result["resumed_from"] == "step3"""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

def test_design_lines_1124_1152_9():
    """Test Python snippet from design.md lines 1124-1152."""
    # `
    
    code = r"""class NestedAmbiguityHandler:
    \"""Handles complex nested ambiguities\"""

    async def resolve_nested(self, obj: Any, path: str = "") -> Any:
        \"""Recursively resolve nested ambiguities\"""
        if isinstance(obj, dict):
            # Check for circular dependencies
            self._check_circular_deps(obj, path)

            resolved = {}
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key

                if self._is_ambiguous(value):
                    # Resolve with parent context
                    context = self._build_context(obj, key, path)
                    resolved[key] = await self._resolve_with_context(value, context)
                else:
                    resolved[key] = await self.resolve_nested(value, new_path)

            return resolved

        elif isinstance(obj, list):
            return [
                await self.resolve_nested(item, f"{path}[{i}]")
                for i, item in enumerate(obj)
            ]

        return obj"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_1158_1196_10():
    """Test orchestrator code from design.md lines 1158-1196."""
    # `
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        if 'hello_world.yaml' in r"""class DynamicModelSwitcher:
    \"""Handles model switching during pipeline execution\"""

    async def switch_model(
        self,
        current_model: str,
        reason: str,
        context: Dict[str, Any]
    ) -> str:
        \"""Switch to alternative model mid-execution\"""
        # Save current state
        checkpoint = await self._create_switching_checkpoint(
            current_model, reason, context
        )

        # Find alternative model
        alternatives = await self._find_alternatives(current_model, reason)

        if not alternatives:
            raise NoAlternativeModelError(
                f"No alternative found for {current_model}"
            )

        # Select best alternative
        new_model = await self._select_alternative(
            alternatives,
            context,
            checkpoint
        )

        # Migrate state if needed
        if self._needs_state_migration(current_model, new_model):
            context = await self._migrate_state(
                context,
                current_model,
                new_model
            )

        return new_model""":
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(r"""class DynamicModelSwitcher:
    \"""Handles model switching during pipeline execution\"""

    async def switch_model(
        self,
        current_model: str,
        reason: str,
        context: Dict[str, Any]
    ) -> str:
        \"""Switch to alternative model mid-execution\"""
        # Save current state
        checkpoint = await self._create_switching_checkpoint(
            current_model, reason, context
        )

        # Find alternative model
        alternatives = await self._find_alternatives(current_model, reason)

        if not alternatives:
            raise NoAlternativeModelError(
                f"No alternative found for {current_model}"
            )

        # Select best alternative
        new_model = await self._select_alternative(
            alternatives,
            context,
            checkpoint
        )

        # Migrate state if needed
        if self._needs_state_migration(current_model, new_model):
            context = await self._migrate_state(
                context,
                current_model,
                new_model
            )

        return new_model""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_1202_1232_11():
    """Test orchestrator code from design.md lines 1202-1232."""
    # `
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        if 'hello_world.yaml' in r"""class DependencyValidator:
    \"""Validates and resolves pipeline dependencies\"""

    def detect_cycles(self, tasks: Dict[str, Task]) -> List[List[str]]:
        \"""Detect circular dependencies using DFS\"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {task_id: WHITE for task_id in tasks}
        cycles = []

        def dfs(task_id: str, path: List[str]):
            color[task_id] = GRAY
            path.append(task_id)

            for dep in tasks[task_id].dependencies:
                if dep not in tasks:
                    raise InvalidDependencyError(f"Unknown dependency: {dep}")

                if color[dep] == GRAY:
                    # Found cycle
                    cycle_start = path.index(dep)
                    cycles.append(path[cycle_start:])
                elif color[dep] == WHITE:
                    dfs(dep, path[:])

            color[task_id] = BLACK

        for task_id in tasks:
            if color[task_id] == WHITE:
                dfs(task_id, [])

        return cycles""":
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(r"""class DependencyValidator:
    \"""Validates and resolves pipeline dependencies\"""

    def detect_cycles(self, tasks: Dict[str, Task]) -> List[List[str]]:
        \"""Detect circular dependencies using DFS\"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {task_id: WHITE for task_id in tasks}
        cycles = []

        def dfs(task_id: str, path: List[str]):
            color[task_id] = GRAY
            path.append(task_id)

            for dep in tasks[task_id].dependencies:
                if dep not in tasks:
                    raise InvalidDependencyError(f"Unknown dependency: {dep}")

                if color[dep] == GRAY:
                    # Found cycle
                    cycle_start = path.index(dep)
                    cycles.append(path[cycle_start:])
                elif color[dep] == WHITE:
                    dfs(dep, path[:])

            color[task_id] = BLACK

        for task_id in tasks:
            if color[task_id] == WHITE:
                dfs(task_id, [])

        return cycles""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_1240_1271_12():
    """Test orchestrator code from design.md lines 1240-1271."""
    # `
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        if 'hello_world.yaml' in r"""class MultiLevelCache:
    \"""Multi-level caching system for performance\"""

    def __init__(self):
        self.memory_cache = LRUCache(maxsize=1000)
        self.redis_cache = RedisCache()
        self.disk_cache = DiskCache("/var/cache/orchestrator")

    async def get(self, key: str) -> Optional[Any]:
        \"""Get from cache with fallback hierarchy\"""
        # L1: Memory cache
        if value := self.memory_cache.get(key):
            return value

        # L2: Redis cache
        if value := await self.redis_cache.get(key):
            self.memory_cache.set(key, value)
            return value

        # L3: Disk cache
        if value := await self.disk_cache.get(key):
            await self.redis_cache.set(key, value, ttl=3600)
            self.memory_cache.set(key, value)
            return value

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        \"""Set in all cache levels\"""
        self.memory_cache.set(key, value)
        await self.redis_cache.set(key, value, ttl=ttl or 3600)
        await self.disk_cache.set(key, value)""":
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(r"""class MultiLevelCache:
    \"""Multi-level caching system for performance\"""

    def __init__(self):
        self.memory_cache = LRUCache(maxsize=1000)
        self.redis_cache = RedisCache()
        self.disk_cache = DiskCache("/var/cache/orchestrator")

    async def get(self, key: str) -> Optional[Any]:
        \"""Get from cache with fallback hierarchy\"""
        # L1: Memory cache
        if value := self.memory_cache.get(key):
            return value

        # L2: Redis cache
        if value := await self.redis_cache.get(key):
            self.memory_cache.set(key, value)
            return value

        # L3: Disk cache
        if value := await self.disk_cache.get(key):
            await self.redis_cache.set(key, value, ttl=3600)
            self.memory_cache.set(key, value)
            return value

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        \"""Set in all cache levels\"""
        self.memory_cache.set(key, value)
        await self.redis_cache.set(key, value, ttl=ttl or 3600)
        await self.disk_cache.set(key, value)""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

def test_design_lines_1277_1320_13():
    """Test Python snippet from design.md lines 1277-1320."""
    # `
    
    code = r"""class ParallelExecutor:
    \"""Optimized parallel task execution\"""

    def __init__(self, max_workers: int = 10):
        self.semaphore = asyncio.Semaphore(max_workers)
        self.task_queue = asyncio.Queue()

    async def execute_level(
        self,
        tasks: List[Task],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        \"""Execute tasks in parallel with resource management\"""
        # Group tasks by resource requirements
        task_groups = self._group_by_resources(tasks)

        results = {}
        for group in task_groups:
            # Execute group with resource limits
            group_results = await self._execute_group(group, context)
            results.update(group_results)

        return results

    async def _execute_group(
        self,
        tasks: List[Task],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        \"""Execute a group of similar tasks\"""
        async def execute_with_semaphore(task):
            async with self.semaphore:
                return await self._execute_single(task, context)

        # Execute all tasks in parallel
        results = await asyncio.gather(*[
            execute_with_semaphore(task) for task in tasks
        ], return_exceptions=True)

        # Process results
        return {
            task.id: self._process_result(result)
            for task, result in zip(tasks, results)
        }"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_1328_1444_14():
    """Test YAML pipeline from design.md lines 1328-1444."""
    import yaml
    import orchestrator
    
    yaml_content = r"""# models.yaml
models:
  # Large models
  gpt-4o:
    provider: openai
    capabilities:
      tasks: [reasoning, code_generation, analysis, creative_writing]
      context_window: 128000
      supports_function_calling: true
      supports_structured_output: true
      languages: [en, es, fr, de, zh, ja, ko]
    requirements:
      memory_gb: 16
      gpu_memory_gb: 24
      cpu_cores: 8
    metrics:
      latency_p50: 2.1
      latency_p95: 4.5
      throughput: 10
      accuracy: 0.95
      cost_per_token: 0.00003

  claude-4-opus:
    provider: anthropic
    capabilities:
      tasks: [reasoning, analysis, creative_writing, code_review]
      context_window: 200000
      supports_function_calling: false
      supports_structured_output: true
      languages: [en, es, fr, de, zh, ja]
    requirements:
      memory_gb: 16
      gpu_memory_gb: 32
      cpu_cores: 8
    metrics:
      latency_p50: 2.5
      latency_p95: 5.0
      throughput: 8
      accuracy: 0.94
      cost_per_token: 0.00003

  # Medium models
  gpt-4o-mini:
    provider: openai
    capabilities:
      tasks: [general, code_generation, summarization]
      context_window: 16384
      supports_function_calling: true
      supports_structured_output: true
      languages: [en, es, fr, de]
    requirements:
      memory_gb: 8
      gpu_memory_gb: 8
      cpu_cores: 4
    metrics:
      latency_p50: 0.8
      latency_p95: 1.5
      throughput: 50
      accuracy: 0.85
      cost_per_token: 0.000002

  # Small models (local)
  llama2-7b:
    provider: local
    capabilities:
      tasks: [general, summarization]
      context_window: 4096
      supports_function_calling: false
      supports_structured_output: false
      languages: [en]
    requirements:
      memory_gb: 16
      gpu_memory_gb: 8
      cpu_cores: 4
      supports_quantization: [int8, int4, gptq]
    metrics:
      latency_p50: 0.5
      latency_p95: 1.0
      throughput: 20
      accuracy: 0.75
      cost_per_token: 0

  # Quantized versions
  llama2-7b-int4:
    provider: local
    base_model: llama2-7b
    quantization: int4
    requirements:
      memory_gb: 8
      gpu_memory_gb: 4
      cpu_cores: 4
    metrics:
      latency_p50: 0.6
      latency_p95: 1.2
      throughput: 15
      accuracy: 0.72
      cost_per_token: 0

# Model selection policies
selection_policies:
  default:
    strategy: ucb  # upper confidence bound
    exploration_factor: 2.0

  cost_optimized:
    strategy: weighted
    weights:
      cost: 0.6
      accuracy: 0.3
      latency: 0.1

  performance_optimized:
    strategy: weighted
    weights:
      latency: 0.5
      accuracy: 0.4
      cost: 0.1"""
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip actual pipeline compilation unless we have real models
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        # Check for API keys
        if not (os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')):
            pytest.skip("No API keys available for pipeline testing")
        
        try:
            # Initialize models
            registry = orchestrator.init_models()
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
            
            # Create compiler
            from orchestrator.compiler import YAMLCompiler
            compiler = YAMLCompiler()
            compiler.set_model_registry(registry)
            
            # Compile the pipeline (but don't run it)
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
                
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_1450_1527_15():
    """Test YAML pipeline from design.md lines 1450-1527."""
    import yaml
    import orchestrator
    
    yaml_content = r"""# config.yaml
orchestrator:
  # Core settings
  version: 1.0.0
  environment: production

  # Storage backend
  storage:
    backend: postgres
    connection_string: ${DATABASE_URL}
    pool_size: 20
    checkpoint_compression: true
    retention_days: 30

  # Execution settings
  execution:
    max_concurrent_pipelines: 10
    max_concurrent_tasks: 50
    default_timeout: 300
    max_retries: 3
    retry_backoff_factor: 2.0

  # Model settings
  models:
    registry_path: models.yaml
    selection_policy: default
    fallback_enabled: true
    local_models_path: /opt/models

  # Sandbox settings
  sandbox:
    enabled: true
    docker_socket: /var/run/docker.sock
    images:
      python: orchestrator/python:3.11-slim
      nodejs: orchestrator/node:18-slim
      custom: ${CUSTOM_SANDBOX_IMAGE}
    resource_limits:
      memory: 1GB
      cpu: 0.5
      disk: 100MB
      network: none

  # Security settings
  security:
    api_key_required: true
    rate_limiting:
      enabled: true
      requests_per_minute: 60
      burst_size: 10
    allowed_actions:
      - generate
      - transform
      - analyze
      - search
      - execute
    forbidden_modules:
      - os
      - subprocess
      - eval
      - exec

  # Monitoring settings
  monitoring:
    metrics_enabled: true
    metrics_port: 9090
    tracing_enabled: true
    tracing_endpoint: ${JAEGER_ENDPOINT}
    log_level: INFO
    structured_logging: true

  # Cache settings
  cache:
    enabled: true
    redis_url: ${REDIS_URL}
    memory_cache_size: 1000
    ttl_seconds: 3600
    compression_enabled: true"""
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip actual pipeline compilation unless we have real models
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        # Check for API keys
        if not (os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')):
            pytest.skip("No API keys available for pipeline testing")
        
        try:
            # Initialize models
            registry = orchestrator.init_models()
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
            
            # Create compiler
            from orchestrator.compiler import YAMLCompiler
            compiler = YAMLCompiler()
            compiler.set_model_registry(registry)
            
            # Compile the pipeline (but don't run it)
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
                
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_1535_1577_16():
    """Test YAML pipeline from design.md lines 1535-1577."""
    import yaml
    import orchestrator
    
    yaml_content = r"""# pipelines/code-review.yaml
name: automated_code_review
version: 1.0.0
description: Comprehensive code review with multiple analysis passes

steps:
  - id: code_parsing
    action: parse_code
    parameters:
      source: "{{ github_pr_url }}"
      languages: <AUTO>Detect programming languages in PR</AUTO>

  - id: security_scan
    action: security_analysis
    parameters:
      code: "{{ steps.code_parsing.parsed_code }}"
      severity_threshold: <AUTO>Based on project type, set threshold</AUTO>
      scan_depth: <AUTO>Determine scan depth based on code size</AUTO>
    dependencies: [code_parsing]

  - id: style_check
    action: style_analysis
    parameters:
      code: "{{ steps.code_parsing.parsed_code }}"
      style_guide: <AUTO>Select appropriate style guide for language</AUTO>
    dependencies: [code_parsing]

  - id: complexity_analysis
    action: analyze_complexity
    parameters:
      code: "{{ steps.code_parsing.parsed_code }}"
      metrics: <AUTO>Choose relevant complexity metrics</AUTO>
    dependencies: [code_parsing]

  - id: generate_review
    action: synthesize_review
    parameters:
      security_results: "{{ steps.security_scan.findings }}"
      style_results: "{{ steps.style_check.violations }}"
      complexity_results: "{{ steps.complexity_analysis.metrics }}"
      review_tone: <AUTO>Professional, constructive, or educational</AUTO>
      priority_order: <AUTO>Order findings by severity and impact</AUTO>
    dependencies: [security_scan, style_check, complexity_analysis]"""
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip actual pipeline compilation unless we have real models
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        # Check for API keys
        if not (os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')):
            pytest.skip("No API keys available for pipeline testing")
        
        try:
            # Initialize models
            registry = orchestrator.init_models()
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
            
            # Create compiler
            from orchestrator.compiler import YAMLCompiler
            compiler = YAMLCompiler()
            compiler.set_model_registry(registry)
            
            # Compile the pipeline (but don't run it)
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
                
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_1583_1629_17():
    """Test YAML pipeline from design.md lines 1583-1629."""
    import yaml
    import orchestrator
    
    yaml_content = r"""# pipelines/data-analysis.yaml
name: intelligent_data_analysis
version: 1.0.0
description: Automated data analysis with visualization

steps:
  - id: data_ingestion
    action: load_data
    parameters:
      source: "{{ data_source }}"
      format: <AUTO>Detect data format (csv, json, parquet, etc)</AUTO>
      sampling_strategy: <AUTO>Full load or sampling based on size</AUTO>

  - id: data_profiling
    action: profile_data
    parameters:
      data: "{{ steps.data_ingestion.data }}"
      profile_depth: <AUTO>Basic, standard, or comprehensive</AUTO>
      anomaly_detection: <AUTO>Enable based on data characteristics</AUTO>
    dependencies: [data_ingestion]

  - id: statistical_analysis
    action: analyze_statistics
    parameters:
      data: "{{ steps.data_ingestion.data }}"
      profile: "{{ steps.data_profiling.profile }}"
      tests: <AUTO>Select appropriate statistical tests</AUTO>
      confidence_level: <AUTO>Set based on data quality</AUTO>
    dependencies: [data_profiling]

  - id: visualization_planning
    action: plan_visualizations
    parameters:
      data_profile: "{{ steps.data_profiling.profile }}"
      insights: "{{ steps.statistical_analysis.insights }}"
      num_visualizations: <AUTO>Determine optimal number</AUTO>
      chart_types: <AUTO>Select appropriate chart types</AUTO>
    dependencies: [statistical_analysis]

  - id: generate_report
    action: create_analysis_report
    parameters:
      visualizations: "{{ steps.visualization_planning.charts }}"
      insights: "{{ steps.statistical_analysis.insights }}"
      executive_summary: <AUTO>Generate executive summary</AUTO>
      technical_depth: <AUTO>Adjust based on audience</AUTO>
    dependencies: [visualization_planning]"""
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip actual pipeline compilation unless we have real models
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        # Check for API keys
        if not (os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')):
            pytest.skip("No API keys available for pipeline testing")
        
        try:
            # Initialize models
            registry = orchestrator.init_models()
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
            
            # Create compiler
            from orchestrator.compiler import YAMLCompiler
            compiler = YAMLCompiler()
            compiler.set_model_registry(registry)
            
            # Compile the pipeline (but don't run it)
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
                
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_1635_1678_18():
    """Test YAML pipeline from design.md lines 1635-1678."""
    import yaml
    import orchestrator
    
    yaml_content = r"""# pipelines/ensemble-prediction.yaml
name: ensemble_prediction
version: 1.0.0
description: Ensemble multiple models for robust predictions

steps:
  - id: data_preparation
    action: prepare_data
    parameters:
      input_data: "{{ raw_data }}"
      preprocessing: <AUTO>Determine required preprocessing steps</AUTO>
      feature_engineering: <AUTO>Identify useful feature transformations</AUTO>

  - id: model_selection
    action: select_models
    parameters:
      task_type: "{{ prediction_task }}"
      num_models: <AUTO>Optimal ensemble size (3-7 models)</AUTO>
      diversity_strategy: <AUTO>Ensure model diversity</AUTO>
    dependencies: [data_preparation]

  - id: parallel_predictions
    action: batch_predict
    parameters:
      models: "{{ steps.model_selection.selected_models }}"
      data: "{{ steps.data_preparation.processed_data }}"
      execution_strategy: parallel
    dependencies: [model_selection]

  - id: ensemble_aggregation
    action: aggregate_predictions
    parameters:
      predictions: "{{ steps.parallel_predictions.results }}"
      aggregation_method: <AUTO>voting, averaging, or stacking</AUTO>
      confidence_calculation: <AUTO>Method for confidence scores</AUTO>
    dependencies: [parallel_predictions]

  - id: result_validation
    action: validate_results
    parameters:
      ensemble_predictions: "{{ steps.ensemble_aggregation.final_predictions }}"
      validation_threshold: <AUTO>Set based on task criticality</AUTO>
      fallback_strategy: <AUTO>Define fallback if validation fails</AUTO>
    dependencies: [ensemble_aggregation]"""
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip actual pipeline compilation unless we have real models
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        # Check for API keys
        if not (os.environ.get('OPENAI_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')):
            pytest.skip("No API keys available for pipeline testing")
        
        try:
            # Initialize models
            registry = orchestrator.init_models()
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
            
            # Create compiler
            from orchestrator.compiler import YAMLCompiler
            compiler = YAMLCompiler()
            compiler.set_model_registry(registry)
            
            # Compile the pipeline (but don't run it)
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
                
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_README_lines_54_68_19():
    """Test orchestrator code from examples/README.md lines 54-68."""
    # Edit config/models.yaml to customize model settings.
    
    # Check for required API keys
    missing_keys = []
    if not os.environ.get('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.environ.get('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    
    if missing_keys:
        pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
    
    # Set up test environment
    import tempfile
    import orchestrator as orc
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Create a minimal test pipeline file if referenced
        if 'hello_world.yaml' in r"""import asyncio
from orchestrator import Orchestrator

async def run_example():
    orchestrator = Orchestrator()

    # Run simple pipeline
    results = await orchestrator.execute_yaml_file(
        "examples/simple_pipeline.yaml",
        context={"input_topic": "machine learning"}
    )

    print("Pipeline results:", results)

asyncio.run(run_example())""":
            with open('hello_world.yaml', 'w') as f:
                f.write("""
id: hello_world
name: Hello World
steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello world"
outputs:
  greeting: "{{ greet.result }}"
""".strip())
        
        # Mock the compile function to avoid real file dependencies
        original_compile = getattr(orc, 'compile', None)
        
        def mock_compile(filename):
            # Return a mock pipeline object
            class MockPipeline:
                def run(self):
                    return {"greeting": "Hello World"}
            return MockPipeline()
        
        # Test the code with mocked dependencies
        try:
            # Replace compile calls temporarily
            if original_compile:
                setattr(orc, 'compile_mock', mock_compile)
                setattr(orc, 'compile', mock_compile)
            
            # Execute the code
            exec(r"""import asyncio
from orchestrator import Orchestrator

async def run_example():
    orchestrator = Orchestrator()

    # Run simple pipeline
    results = await orchestrator.execute_yaml_file(
        "examples/simple_pipeline.yaml",
        context={"input_topic": "machine learning"}
    )

    print("Pipeline results:", results)

asyncio.run(run_example())""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)
