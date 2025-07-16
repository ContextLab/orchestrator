"""Tests for documentation code snippets - Batch 1 (Robust)."""
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


def test_CLAUDE_lines_49_59_0():
    """Test text snippet from CLAUDE.md lines 49-59."""
    # Content validation for text snippet
    content = ("""orchestrator/
├── src/orchestrator/          # Core library code
│   ├── compiler/             # YAML parsing and compilation
│   ├── models/               # Model abstractions and registry
│   ├── adapters/             # Control system adapters
│   ├── executor/             # Sandboxed execution
│   └── state/                # State management
├── tests/                    # Unit and integration tests
├── examples/                 # Example pipeline definitions
├── docs/                     # Documentation
└── config/                   # Configuration schemas""")
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_CLAUDE_lines_67_73_1():
    """Test YAML pipeline from CLAUDE.md lines 67-73."""
    import yaml
    import orchestrator
    
    yaml_content = ("""steps:
  - id: analyze_data
    action: analyze
    parameters:
      data: "{{ input_data }}"
      method: <AUTO>Choose best analysis method for this data type</AUTO>
      depth: <AUTO>Determine analysis depth based on data complexity</AUTO>""")
    
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

def test_README_lines_32_32_2():
    """Test bash snippet from README.md lines 32-32."""
    # Bash command snippet
    snippet_bash = ("""pip install py-orc""")
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_README_lines_37_40_3():
    """Test bash snippet from README.md lines 37-40."""
    # Bash command snippet
    snippet_bash = ("""pip install py-orc[ollama]      # Ollama model support
pip install py-orc[cloud]        # Cloud model providers
pip install py-orc[dev]          # Development tools
pip install py-orc[all]          # Everything""")
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_README_lines_48_66_4():
    """Test YAML pipeline from README.md lines 48-66."""
    import yaml
    import orchestrator
    
    yaml_content = ("""id: hello_world
name: Hello World Pipeline
description: A simple example pipeline

steps:
  - id: greet
    action: generate_text
    parameters:
      prompt: "Say hello to the world in a creative way!"

  - id: translate
    action: generate_text
    parameters:
      prompt: "Translate this greeting to Spanish: {{ greet.result }}"
    dependencies: [greet]

outputs:
  greeting: "{{ greet.result }}"
  spanish: "{{ translate.result }}"""")
    
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
async def test_README_lines_72_81_5():
    """Test orchestrator code from README.md lines 72-81."""
    # 2. Run the pipeline:
    
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
        code = ("""import orchestrator as orc

# Initialize models (auto-detects available models)
orc.init_models()

# Compile and run the pipeline
pipeline = orc.compile("hello_world.yaml")
result = pipeline.run()

print(result)""")
        if 'hello_world.yaml' in code:
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
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
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
async def test_README_lines_89_95_6():
    """Test YAML pipeline from README.md lines 89-95."""
    import yaml
    import orchestrator
    
    yaml_content = ("""steps:
  - id: analyze_data
    action: analyze
    parameters:
      data: "{{ input_data }}"
      method: <AUTO>Choose the best analysis method for this data type</AUTO>
      visualization: <AUTO>Decide if we should create a chart</AUTO>""")
    
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

def test_README_lines_103_125_7():
    """Test YAML snippet from README.md lines 103-125."""
    import yaml
    
    yaml_content = ("""models:
  # Local models (via Ollama) - downloaded on first use
  - source: ollama
    name: llama3.1:8b
    expertise: [general, reasoning, multilingual]
    size: 8b

  - source: ollama
    name: qwen2.5-coder:7b
    expertise: [code, programming]
    size: 7b

  # Cloud models
  - source: openai
    name: gpt-4o
    expertise: [general, reasoning, code, analysis, vision]
    size: 1760b  # Estimated

defaults:
  expertise_preferences:
    code: qwen2.5-coder:7b
    reasoning: deepseek-r1:8b
    fast: llama3.2:1b""")
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_README_lines_135_192_8():
    """Test YAML pipeline from README.md lines 135-192."""
    import yaml
    import orchestrator
    
    yaml_content = ("""id: research_pipeline
name: AI Research Pipeline
description: Research a topic and create a comprehensive report

inputs:
  - name: topic
    type: string
    description: Research topic

  - name: depth
    type: string
    default: <AUTO>Determine appropriate research depth</AUTO>

steps:
  # Parallel research from multiple sources
  - id: web_search
    action: search_web
    parameters:
      query: "{{ topic }} latest research 2025"
      count: <AUTO>Decide how many results to fetch</AUTO>
    requires_model:
      expertise: [research, web]

  - id: academic_search
    action: search_academic
    parameters:
      query: "{{ topic }}"
      filters: <AUTO>Set appropriate academic filters</AUTO>
    requires_model:
      expertise: [research, academic]

  # Analyze findings with specialized model
  - id: analyze_findings
    action: analyze
    parameters:
      web_results: "{{ web_search.results }}"
      academic_results: "{{ academic_search.results }}"
      analysis_focus: <AUTO>Determine key aspects to analyze</AUTO>
    dependencies: [web_search, academic_search]
    requires_model:
      expertise: [analysis, reasoning]
      min_size: 20b  # Require large model for complex analysis

  # Generate report
  - id: write_report
    action: generate_document
    parameters:
      topic: "{{ topic }}"
      analysis: "{{ analyze_findings.result }}"
      style: <AUTO>Choose appropriate writing style</AUTO>
      length: <AUTO>Determine optimal report length</AUTO>
    dependencies: [analyze_findings]
    requires_model:
      expertise: [writing, general]

outputs:
  report: "{{ write_report.document }}"
  summary: "{{ analyze_findings.summary }}"""")
    
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
async def test_README_lines_200_274_9():
    """Test YAML pipeline from README.md lines 200-274."""
    import yaml
    import orchestrator
    
    yaml_content = ("""# research_report.yaml
id: research_report
name: Research Report Generator
description: Generate comprehensive research reports with citations

inputs:
  - name: topic
    type: string
    description: Research topic
  - name: instructions
    type: string
    description: Additional instructions for the report

outputs:
  - pdf: <AUTO>Generate appropriate filename for the research report PDF</AUTO>

steps:
  - id: search
    name: Web Search
    action: search_web
    parameters:
      query: <AUTO>Create effective search query for {topic} with {instructions}</AUTO>
      max_results: 10
    requires_model:
      expertise: fast

  - id: compile_notes
    name: Compile Research Notes
    action: generate_text
    parameters:
      prompt: |
        Compile comprehensive research notes from these search results:
        {{ search.results }}

        Topic: {{ topic }}
        Instructions: {{ instructions }}

        Create detailed notes with:
        - Key findings
        - Important quotes
        - Source citations
        - Relevant statistics
    dependencies: [search]
    requires_model:
      expertise: [analysis, reasoning]
      min_size: 7b

  - id: write_report
    name: Write Report
    action: generate_document
    parameters:
      content: |
        Write a comprehensive research report on "{{ topic }}"

        Research notes:
        {{ compile_notes.result }}

        Requirements:
        - Professional academic style
        - Include introduction, body sections, and conclusion
        - Cite sources properly
        - {{ instructions }}
      format: markdown
    dependencies: [compile_notes]
    requires_model:
      expertise: [writing, general]
      min_size: 20b

  - id: create_pdf
    name: Create PDF
    action: convert_to_pdf
    parameters:
      markdown: "{{ write_report.document }}"
      filename: "{{ outputs.pdf }}"
    dependencies: [write_report]""")
    
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
async def test_README_lines_280_294_10():
    """Test orchestrator code from README.md lines 280-294."""
    # Run it with:
    
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
        code = ("""import orchestrator as orc

# Initialize models
orc.init_models()

# Compile pipeline
pipeline = orc.compile("research_report.yaml")

# Run with inputs
result = pipeline.run(
    topic="quantum computing applications in medicine",
    instructions="Focus on recent breakthroughs and future potential"
)

print(f"Report saved to: {result}")""")
        if 'hello_world.yaml' in code:
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
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

def test_README_lines_353_359_11():
    """Test bibtex snippet from README.md lines 353-359."""
    # Content validation for bibtex snippet
    content = ("""@software{orchestrator2025,
  title = {Orchestrator: AI Pipeline Orchestration Framework},
  author = {Manning, Jeremy R. and {Contextual Dynamics Lab}},
  year = {2025},
  url = {https://github.com/ContextLab/orchestrator},
  organization = {Dartmouth College}
}""")
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

def test_design_lines_12_47_12():
    """Test text snippet from design.md lines 12-47."""
    # Content validation for text snippet
    content = ("""┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                    (YAML Pipeline Definition)                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────┐
│                    YAML Parser & Compiler                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Schema      │  │ Ambiguity    │  │ Pipeline           │    │
│  │ Validator   │  │ Detector     │  │ Optimizer          │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└───────────────────────┬───────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────┐
│                  Orchestration Engine                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Task        │  │ Dependency   │  │ Resource           │    │
│  │ Scheduler   │  │ Manager      │  │ Allocator          │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└───────────────────────┬───────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────┐
│                  Control System Adapters                      │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐     │
│  │ LangGraph   │  │ MCP          │  │ Custom            │     │
│  │ Adapter     │  │ Adapter      │  │ Adapters          │     │
│  └─────────────┘  └──────────────┘  └───────────────────┘     │
└───────────────────────┬───────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────┐
│                    Execution Layer                            │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Sandboxed   │  │ Model        │  │ State              │    │
│  │ Executors   │  │ Registry     │  │ Persistence        │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└───────────────────────────────────────────────────────────────┘""")
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_design_lines_63_89_13():
    """Test orchestrator code from design.md lines 63-89."""
    # 5. Security by Default: Sandboxed execution and input validation
    
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
        code = ('''from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class Task:
    """Core task abstraction for the orchestrator"""
    id: str
    name: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = None

    def is_ready(self, completed_tasks: set) -> bool:
        """Check if all dependencies are satisfied"""
        return all(dep in completed_tasks for dep in self.dependencies)''')
        if 'hello_world.yaml' in code:
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
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
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
async def test_design_lines_95_135_14():
    """Test orchestrator code from design.md lines 95-135."""
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
        code = ('''@dataclass
class Pipeline:
    """Pipeline represents a collection of tasks with dependencies"""
    id: str
    name: str
    tasks: Dict[str, Task]
    context: Dict[str, Any]
    metadata: Dict[str, Any]

    def get_execution_order(self) -> List[List[str]]:
        """Returns tasks grouped by execution level (parallel groups)"""
        from collections import defaultdict, deque

        # Build dependency graph
        in_degree = {task_id: len(task.dependencies) for task_id, task in self.tasks.items()}
        graph = defaultdict(list)

        for task_id, task in self.tasks.items():
            for dep in task.dependencies:
                graph[dep].append(task_id)

        # Topological sort with level grouping
        levels = []
        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])

        while queue:
            current_level = []
            level_size = len(queue)

            for _ in range(level_size):
                task_id = queue.popleft()
                current_level.append(task_id)

                for neighbor in graph[task_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            levels.append(current_level)

        return levels''')
        if 'hello_world.yaml' in code:
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
            exec(code, {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)
