"""Tests for documentation code snippets - Batch 2."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Set up test environment
os.environ.setdefault('ORCHESTRATOR_CONFIG', str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml"))

# Note: API keys should be set as environment variables or GitHub secrets:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY  
# - GOOGLE_AI_API_KEY


@pytest.mark.asyncio
async def test_design_lines_1158_1196_0():
    """Test Python snippet from design.md lines 1158-1196."""
    # `
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """class DynamicModelSwitcher:
    """Handles model switching during pipeline execution"""

    async def switch_model(
        self,
        current_model: str,
        reason: str,
        context: Dict[str, Any]
    ) -> str:
        """Switch to alternative model mid-execution"""
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

        return new_model"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_design_lines_1202_1232_1():
    """Test Python snippet from design.md lines 1202-1232."""
    # `
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """class DependencyValidator:
    """Validates and resolves pipeline dependencies"""

    def detect_cycles(self, tasks: Dict[str, Task]) -> List[List[str]]:
        """Detect circular dependencies using DFS"""
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

        return cycles"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_design_lines_1240_1271_2():
    """Test Python snippet from design.md lines 1240-1271."""
    # `
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """class MultiLevelCache:
    """Multi-level caching system for performance"""

    def __init__(self):
        self.memory_cache = LRUCache(maxsize=1000)
        self.redis_cache = RedisCache()
        self.disk_cache = DiskCache("/var/cache/orchestrator")

    async def get(self, key: str) -> Optional[Any]:
        """Get from cache with fallback hierarchy"""
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
        """Set in all cache levels"""
        self.memory_cache.set(key, value)
        await self.redis_cache.set(key, value, ttl=ttl or 3600)
        await self.disk_cache.set(key, value)"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_design_lines_1277_1320_3():
    """Test Python snippet from design.md lines 1277-1320."""
    # `
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """class ParallelExecutor:
    """Optimized parallel task execution"""

    def __init__(self, max_workers: int = 10):
        self.semaphore = asyncio.Semaphore(max_workers)
        self.task_queue = asyncio.Queue()

    async def execute_level(
        self,
        tasks: List[Task],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tasks in parallel with resource management"""
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
        """Execute a group of similar tasks"""
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
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_design_lines_1328_1444_4():
    """Test YAML pipeline from design.md lines 1328-1444."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# models.yaml
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
    
    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # If it's a pipeline, validate it
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        from orchestrator.compiler import YAMLCompiler
        import orchestrator
        
        # Set up environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        compiler = YAMLCompiler()
        
        # Initialize real models
        try:
            registry = orchestrator.init_models()
            compiler.set_model_registry(registry)
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
                
        except Exception as e:
            if "API key" in str(e):
                pytest.skip(f"Missing API keys for real model testing: {e}")
            else:
                raise
        
        # Compile the pipeline
        try:
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
            
        except Exception as e:
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

@pytest.mark.asyncio
async def test_design_lines_1450_1527_5():
    """Test YAML pipeline from design.md lines 1450-1527."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# config.yaml
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
    
    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # If it's a pipeline, validate it
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        from orchestrator.compiler import YAMLCompiler
        import orchestrator
        
        # Set up environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        compiler = YAMLCompiler()
        
        # Initialize real models
        try:
            registry = orchestrator.init_models()
            compiler.set_model_registry(registry)
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
                
        except Exception as e:
            if "API key" in str(e):
                pytest.skip(f"Missing API keys for real model testing: {e}")
            else:
                raise
        
        # Compile the pipeline
        try:
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
            
        except Exception as e:
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

@pytest.mark.asyncio
async def test_design_lines_1535_1577_6():
    """Test YAML pipeline from design.md lines 1535-1577."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# pipelines/code-review.yaml
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
    
    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # If it's a pipeline, validate it
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        from orchestrator.compiler import YAMLCompiler
        import orchestrator
        
        # Set up environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        compiler = YAMLCompiler()
        
        # Initialize real models
        try:
            registry = orchestrator.init_models()
            compiler.set_model_registry(registry)
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
                
        except Exception as e:
            if "API key" in str(e):
                pytest.skip(f"Missing API keys for real model testing: {e}")
            else:
                raise
        
        # Compile the pipeline
        try:
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
            
        except Exception as e:
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

@pytest.mark.asyncio
async def test_design_lines_1583_1629_7():
    """Test YAML pipeline from design.md lines 1583-1629."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# pipelines/data-analysis.yaml
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
    
    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # If it's a pipeline, validate it
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        from orchestrator.compiler import YAMLCompiler
        import orchestrator
        
        # Set up environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        compiler = YAMLCompiler()
        
        # Initialize real models
        try:
            registry = orchestrator.init_models()
            compiler.set_model_registry(registry)
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
                
        except Exception as e:
            if "API key" in str(e):
                pytest.skip(f"Missing API keys for real model testing: {e}")
            else:
                raise
        
        # Compile the pipeline
        try:
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
            
        except Exception as e:
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

@pytest.mark.asyncio
async def test_design_lines_1635_1678_8():
    """Test YAML pipeline from design.md lines 1635-1678."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """# pipelines/ensemble-prediction.yaml
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
    
    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # If it's a pipeline, validate it
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        from orchestrator.compiler import YAMLCompiler
        import orchestrator
        
        # Set up environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        compiler = YAMLCompiler()
        
        # Initialize real models
        try:
            registry = orchestrator.init_models()
            compiler.set_model_registry(registry)
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
                
        except Exception as e:
            if "API key" in str(e):
                pytest.skip(f"Missing API keys for real model testing: {e}")
            else:
                raise
        
        # Compile the pipeline
        try:
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
            
        except Exception as e:
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

def test_README_lines_54_68_9():
    """Test Python import from examples/README.md lines 54-68."""
    # Test imports
    try:
        exec("""import asyncio
from orchestrator import Orchestrator

async def run_example():
    orchestrator = Orchestrator()

    # Run simple pipeline
    results = await orchestrator.execute_yaml_file(
        "examples/simple_pipeline.yaml",
        context={"input_topic": "machine learning"}
    )

    print("Pipeline results:", results)

asyncio.run(run_example())""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_README_lines_74_80_10():
    """Test bash snippet from examples/README.md lines 74-80."""
    import subprocess
    import tempfile
    import os
    
    bash_content = r"""# Simple pipeline
python -m orchestrator run examples/simple_pipeline.yaml \
    --context input_topic="machine learning"

# Multi-model pipeline
python -m orchestrator run examples/multi_model_pipeline.yaml \
    --context dataset_url="https://example.com/sales_data.csv""""
    
    # Skip if it's a command we shouldn't run
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # Check bash syntax
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(bash_content)
            f.flush()
            
            # Check syntax only
            result = subprocess.run(['bash', '-n', f.name], 
                                  capture_output=True, text=True)
            
            os.unlink(f.name)
            
            if result.returncode != 0:
                pytest.fail(f"Bash syntax error: {result.stderr}")
                
    except FileNotFoundError:
        pytest.skip("Bash not available for testing")

def test_README_lines_89_89_11():
    """Test YAML snippet from examples/README.md lines 89-89."""
    import yaml
    
    yaml_content = """analysis_type: <AUTO>What type of analysis is most appropriate for this text?</AUTO>"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_README_lines_94_94_12():
    """Test YAML snippet from examples/README.md lines 94-94."""
    import yaml
    
    yaml_content = """format: <AUTO>Determine the best format for this data source</AUTO>"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_README_lines_99_99_13():
    """Test YAML snippet from examples/README.md lines 99-99."""
    import yaml
    
    yaml_content = """methods: <AUTO>Choose the most appropriate statistical methods</AUTO>"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_README_lines_104_104_14():
    """Test YAML snippet from examples/README.md lines 104-104."""
    import yaml
    
    yaml_content = """validation_rules: <AUTO>Generate appropriate validation rules for this dataset</AUTO>"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_README_lines_112_131_15():
    """Test YAML pipeline from examples/README.md lines 112-131."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """id: my_pipeline
name: My Custom Pipeline
description: Description of what this pipeline does
version: "1.0"

context:
  # Global variables accessible to all tasks
  variable_name: value

steps:
  - id: task1
    name: First Task
    action: generate  # or analyze, transform, etc.
    parameters:
      # Task-specific parameters
      prompt: "Your prompt here"
    metadata:
      # Optional metadata
      requires_model: true
      priority: 1.0"""
    
    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # If it's a pipeline, validate it
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        from orchestrator.compiler import YAMLCompiler
        import orchestrator
        
        # Set up environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        compiler = YAMLCompiler()
        
        # Initialize real models
        try:
            registry = orchestrator.init_models()
            compiler.set_model_registry(registry)
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
                
        except Exception as e:
            if "API key" in str(e):
                pytest.skip(f"Missing API keys for real model testing: {e}")
            else:
                raise
        
        # Compile the pipeline
        try:
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
            
        except Exception as e:
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

@pytest.mark.asyncio
async def test_README_lines_137_152_16():
    """Test YAML pipeline from examples/README.md lines 137-152."""
    import yaml
    import os
    from pathlib import Path
    
    yaml_content = """steps:
  - id: advanced_task
    name: Advanced Task
    action: analyze
    parameters:
      data: "{{ results.previous_task }}"  # Reference previous results
      method: <AUTO>Choose best method</AUTO>  # AUTO resolution
    dependencies:
      - previous_task  # Task dependencies
    metadata:
      requires_model: gpt-4  # Specific model requirement
      cpu_cores: 4  # Resource requirements
      memory_mb: 2048
      timeout: 300
      priority: 0.8
      on_failure: continue  # Error handling"""
    
    # Parse YAML
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # If it's a pipeline, validate it
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        from orchestrator.compiler import YAMLCompiler
        import orchestrator
        
        # Set up environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        compiler = YAMLCompiler()
        
        # Initialize real models
        try:
            registry = orchestrator.init_models()
            compiler.set_model_registry(registry)
            
            # Check if we have any models available
            if not registry.list_models():
                pytest.skip("No models available for testing")
                
        except Exception as e:
            if "API key" in str(e):
                pytest.skip(f"Missing API keys for real model testing: {e}")
            else:
                raise
        
        # Compile the pipeline
        try:
            pipeline = await compiler.compile(data)
            assert pipeline is not None
            assert pipeline.id
            
            # Validate pipeline structure
            if 'steps' in data:
                assert len(pipeline.tasks) == len(data['steps'])
            
        except Exception as e:
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Pipeline compilation failed: {e}")

def test_README_lines_181_184_17():
    """Test Python import from examples/README.md lines 181-184."""
    # Test imports
    try:
        exec("""import logging
logging.basicConfig(level=logging.DEBUG)

orchestrator = Orchestrator()""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_README_lines_192_193_18():
    """Test Python snippet from examples/README.md lines 192-193."""
    # Verify system health before running pipelines:
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """health = await orchestrator.health_check()
print("System health:", health["overall"])"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

def test_draft_report_lines_59_59_19():
    """Test text snippet from examples/output/readme_report/draft_report.md lines 59-59."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")

def test_draft_report_lines_68_68_20():
    """Test text snippet from examples/output/readme_report/draft_report.md lines 68-68."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")

def test_draft_report_lines_74_76_21():
    """Test text snippet from examples/output/readme_report/draft_report.md lines 74-76."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")

def test_draft_report_lines_88_95_22():
    """Test Python import from examples/output/readme_report/draft_report.md lines 88-95."""
    # Test imports
    try:
        exec("""from langchain.agents import create_react_agent
from langchain.tools import Tool

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, calculator_tool],
    prompt=agent_prompt
)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_draft_report_lines_104_107_23():
    """Test Python import from examples/output/readme_report/draft_report.md lines 104-107."""
    # Test imports
    try:
        exec("""from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user", code_execution_config=Ellipsis)""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_draft_report_lines_116_120_24():
    """Test Python import from examples/output/readme_report/draft_report.md lines 116-120."""
    # Test imports
    try:
        exec("""from crewai import Agent, Task, Crew

researcher = Agent(role="Researcher", goal="Find information")
writer = Agent(role="Writer", goal="Create content")
crew = Crew(agents=[researcher, writer], tasks=[...])""")
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_draft_report_lines_128_144_25():
    """Test Python snippet from examples/output/readme_report/draft_report.md lines 128-144."""
    # `
    
    code = """# Using LangChain
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun

tools = [
    DuckDuckGoSearchRun(),
    WikipediaQueryRun()
]

research_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = research_agent.run("What are the latest developments in quantum computing?")"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_draft_report_lines_150_166_26():
    """Test Python snippet from examples/output/readme_report/draft_report.md lines 150-166."""
    # `
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """# Using AutoGen
coding_assistant = AssistantAgent(
    "coding_assistant",
    system_message="You are a helpful AI that writes and explains code.",
    llm_config={"model": "gpt-4"}
)

user_proxy = UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config={"work_dir": "coding", "use_docker": False}
)

user_proxy.initiate_chat(
    coding_assistant,
    message="Write a Python function to calculate fibonacci numbers efficiently"
)"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

@pytest.mark.asyncio
async def test_draft_report_lines_172_206_27():
    """Test Python snippet from examples/output/readme_report/draft_report.md lines 172-206."""
    # `
    
    # Import required modules
    import os
    import tempfile
    from pathlib import Path
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set up test environment
        os.environ['ORCHESTRATOR_CONFIG'] = str(Path(__file__).parent.parent.parent / "config" / "orchestrator.yaml")
        
        # Test code snippet
        code = """# Using CrewAI
from crewai import Agent, Task, Crew, Process

# Define agents
researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI',
    backstory="You're an expert researcher with a keen eye for detail."
)

writer = Agent(
    role='Tech Content Strategist',
    goal='Create compelling content about AI developments',
    backstory="You're a skilled writer who makes complex topics accessible."
)

# Define tasks
research_task = Task(
    description='Research the latest AI developments in the past month',
    agent=researcher
)

writing_task = Task(
    description='Write a blog post about the research findings',
    agent=writer
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential
)

result = crew.kickoff()"""
        
        # Execute with real models (API keys from environment/GitHub secrets)
        try:
            # Check if required API keys are available
            missing_keys = []
            if 'openai' in code.lower() and not os.environ.get('OPENAI_API_KEY'):
                missing_keys.append('OPENAI_API_KEY')
            if 'anthropic' in code.lower() and not os.environ.get('ANTHROPIC_API_KEY'):
                missing_keys.append('ANTHROPIC_API_KEY')
            if ('gemini' in code.lower() or 'google' in code.lower()) and not os.environ.get('GOOGLE_AI_API_KEY'):
                missing_keys.append('GOOGLE_AI_API_KEY')
            
            if missing_keys:
                pytest.skip(f"Missing API keys for real model testing: {', '.join(missing_keys)}")
            
            # Execute the code with real models
            if 'await' in code or 'async' in code:
                # Handle async code
                import asyncio
                exec_globals = {'__name__': '__main__', 'asyncio': asyncio}
                exec(code, exec_globals)
                
                # If there's a main coroutine, run it
                if 'main' in exec_globals and asyncio.iscoroutinefunction(exec_globals['main']):
                    await exec_globals['main']()
            else:
                exec(code, {'__name__': '__main__'})
                
        except Exception as e:
            # Check if it's an expected error
            if "No eligible models" in str(e):
                pytest.skip(f"No eligible models available: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")

def test_final_report_lines_59_59_28():
    """Test text snippet from examples/output/readme_report/final_report.md lines 59-59."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")

def test_final_report_lines_68_68_29():
    """Test text snippet from examples/output/readme_report/final_report.md lines 68-68."""
    # Snippet type 'text' not yet supported for testing
    pytest.skip("Snippet type 'text' not yet supported")
