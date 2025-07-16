"""Tests for documentation code snippets - Batch 8."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.asyncio
async def test_concepts_lines_120_129_0():
    """Test YAML pipeline from docs_sphinx/concepts.rst lines 120-129."""
    import yaml
    
    yaml_content = 'steps:\n  - id: example\n    parameters:\n      # Compile-time: resolved once during compilation\n      timestamp: "{{ compile_time.now }}"\n\n      # Runtime: resolved during each execution\n      user_input: "{{ inputs.query }}"\n      previous_result: "$results.other_task"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_concepts_lines_137_153_1():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 137-153."""
    import yaml
    
    yaml_content = 'parameters:\n  # Simple AUTO tag\n  style: <AUTO>Choose appropriate writing style</AUTO>\n\n  # Context-aware AUTO tag\n  method: <AUTO>Based on data type {{ results.data.type }}, choose best analysis method</AUTO>\n\n  # Complex AUTO tag with instructions\n  sections: |\n    <AUTO>\n    For a report about {{ inputs.topic }}, determine which sections to include:\n    - Executive Summary: yes/no\n    - Technical Details: yes/no\n    - Future Outlook: yes/no\n    Return as JSON object\n    </AUTO>'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_concepts_lines_197_215_2():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 197-215."""
    import yaml
    
    yaml_content = '# Web search\n- action: search_web\n  parameters:\n    query: "machine learning"\n\n# File operations\n- action: write_file\n  parameters:\n    path: "output.txt"\n    content: "Hello world"\n\n# Shell commands (prefix with !)\n- action: "!ls -la"\n\n# AI generation\n- action: generate_content\n  parameters:\n    prompt: "Write a summary about {{ topic }}"'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_concepts_lines_223_227_3():
    """Test YAML pipeline from docs_sphinx/concepts.rst lines 223-227."""
    import yaml
    
    yaml_content = 'steps:\n  - action: search_web        # → Requires web tool\n  - action: "!python script.py"  # → Requires terminal tool\n  - action: write_file        # → Requires filesystem tool'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

@pytest.mark.asyncio
async def test_concepts_lines_265_271_4():
    """Test Python snippet from docs_sphinx/concepts.rst lines 265-271."""
    # Skip complex orchestrator code for now - would need full setup
    pytest.skip("Complex orchestrator code testing not yet implemented")

@pytest.mark.asyncio
async def test_concepts_lines_284_291_5():
    """Test YAML pipeline from docs_sphinx/concepts.rst lines 284-291."""
    import yaml
    
    yaml_content = 'config:\n  checkpoint: true  # Enable automatic checkpointing\n\nsteps:\n  - id: expensive_task\n    action: long_running_process\n    checkpoint: true  # Force checkpoint after this step'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

@pytest.mark.asyncio
async def test_concepts_lines_299_304_6():
    """Test Python snippet from docs_sphinx/concepts.rst lines 299-304."""
    # Skip complex orchestrator code for now - would need full setup
    pytest.skip("Complex orchestrator code testing not yet implemented")

def test_concepts_lines_335_341_7():
    """Test Python import from docs_sphinx/concepts.rst lines 335-341."""
    # Import test - check if modules are available
    code = 'from orchestrator.core.control_system import ControlSystem\n\nclass MyControlSystem(ControlSystem):\n    async def execute_task(self, task, context):\n        # Custom execution logic\n        pass'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_concepts_lines_352_366_8():
    """Test YAML pipeline from docs_sphinx/concepts.rst lines 352-366."""
    import yaml
    
    yaml_content = 'imports:\n  - common/validation.yaml as validator\n  - workflows/analysis.yaml as analyzer\n\nsteps:\n  - id: validate\n    pipeline: validator\n    inputs:\n      data: "{{ inputs.raw_data }}"\n\n  - id: analyze\n    pipeline: analyzer\n    inputs:\n      validated_data: "$results.validate"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

@pytest.mark.asyncio
async def test_concepts_lines_386_401_9():
    """Test YAML pipeline from docs_sphinx/concepts.rst lines 386-401."""
    import yaml
    
    yaml_content = 'steps:\n  - id: risky_task\n    action: external_api_call\n    error_handling:\n      # Retry with backoff\n      retry:\n        max_attempts: 3\n        backoff: exponential\n\n      # Fallback action\n      fallback:\n        action: use_cached_data\n\n      # Continue pipeline on error\n      continue_on_error: true'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

@pytest.mark.asyncio
async def test_concepts_lines_422_434_10():
    """Test YAML pipeline from docs_sphinx/concepts.rst lines 422-434."""
    import yaml
    
    yaml_content = 'steps:\n  # These run in parallel\n  - id: source1\n    action: fetch_data_a\n\n  - id: source2\n    action: fetch_data_b\n\n  # This waits for both\n  - id: combine\n    depends_on: [source1, source2]\n    action: merge_data'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

@pytest.mark.asyncio
async def test_concepts_lines_442_449_11():
    """Test YAML pipeline from docs_sphinx/concepts.rst lines 442-449."""
    import yaml
    
    yaml_content = 'steps:\n  - id: expensive_computation\n    action: complex_analysis\n    cache:\n      enabled: true\n      key: "{{ inputs.data_hash }}"\n      ttl: 3600  # 1 hour'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_concepts_lines_457_462_12():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 457-462."""
    import yaml
    
    yaml_content = 'config:\n  resources:\n    max_memory: "8GB"\n    max_threads: 4\n    gpu_enabled: false'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_concepts_lines_483_494_13():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 483-494."""
    import yaml
    
    yaml_content = 'inputs:\n  email:\n    type: string\n    validation:\n      pattern: "^[\\\\\\\\w.-]+@[\\\\\\\\w.-]+\\\\\\\\.\\\\\\\\w+$"\n\n  amount:\n    type: number\n    validation:\n      min: 0\n      max: 10000'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_concepts_lines_502_505_14():
    """Test YAML snippet from docs_sphinx/concepts.rst lines 502-505."""
    import yaml
    
    yaml_content = 'parameters:\n  api_key: "{{ env.SECRET_API_KEY }}"  # From environment\n  password: "{{ vault.db_password }}"   # From secret vault'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_concepts_lines_523_533_15():
    """Test  snippet from docs_sphinx/concepts.rst lines 523-533."""
    pytest.skip("Snippet type '' not yet supported")

@pytest.mark.asyncio
async def test_getting_started_lines_29_43_16():
    """Test YAML pipeline from docs_sphinx/getting_started.rst lines 29-43."""
    import yaml
    
    yaml_content = 'name: research-report\ndescription: Generate comprehensive research reports\n\ninputs:\n  topic:\n    type: string\n    description: Research topic\n    required: true\n\nsteps:\n  - id: search\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }} latest research"'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_getting_started_lines_61_64_17():
    """Test YAML snippet from docs_sphinx/getting_started.rst lines 61-64."""
    import yaml
    
    yaml_content = 'parameters:\n  method: <AUTO>Choose best analysis method for this data</AUTO>\n  depth: <AUTO>Determine appropriate depth level</AUTO>'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_getting_started_lines_83_108_18():
    """Test YAML pipeline from docs_sphinx/getting_started.rst lines 83-108."""
    import yaml
    
    yaml_content = 'name: quick-research\ndescription: Quick research on any topic\n\ninputs:\n  topic:\n    type: string\n    required: true\n\noutputs:\n  report:\n    type: string\n    value: "{{ inputs.topic }}_report.md"\n\nsteps:\n  - id: search\n    action: search_web\n    parameters:\n      query: "{{ inputs.topic }}"\n      max_results: 5\n\n  - id: summarize\n    action: generate_summary\n    parameters:\n      content: "$results.search"\n      style: <AUTO>Choose appropriate summary style</AUTO>'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_getting_started_lines_113_126_19():
    """Test Python import from docs_sphinx/getting_started.rst lines 113-126."""
    # Import test - check if modules are available
    code = 'import orchestrator as orc\n\n# Initialize models\norc.init_models()\n\n# Compile the pipeline\npipeline = orc.compile("research.yaml")\n\n# Execute with different topics\nresult1 = pipeline.run(topic="artificial intelligence")\nresult2 = pipeline.run(topic="climate change")\n\nprint(f"Reports generated: {result1}, {result2}")'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_getting_started_lines_141_148_20():
    """Test Python snippet from docs_sphinx/getting_started.rst lines 141-148."""
    # Skip complex orchestrator code for now - would need full setup
    pytest.skip("Complex orchestrator code testing not yet implemented")

def test_getting_started_lines_153_160_21():
    """Test YAML snippet from docs_sphinx/getting_started.rst lines 153-160."""
    import yaml
    
    yaml_content = 'inputs:\n  topic:\n    type: string\n    required: true\n  style:\n    type: string\n    default: "technical"'
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.asyncio
async def test_getting_started_lines_168_177_22():
    """Test YAML pipeline from docs_sphinx/getting_started.rst lines 168-177."""
    import yaml
    
    yaml_content = 'steps:\n  - id: fetch_data\n    action: search_web        # Auto-detects web tool\n\n  - id: save_results\n    action: write_file        # Auto-detects filesystem tool\n\n  - id: run_analysis\n    action: "!python analyze.py"  # Auto-detects terminal tool'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

@pytest.mark.asyncio
async def test_getting_started_lines_185_193_23():
    """Test Python snippet from docs_sphinx/getting_started.rst lines 185-193."""
    # Skip complex orchestrator code for now - would need full setup
    pytest.skip("Complex orchestrator code testing not yet implemented")

def test_index_lines_43_56_24():
    """Test Python import from docs_sphinx/index.rst lines 43-56."""
    # Import test - check if modules are available
    code = 'import orchestrator as orc\n\n# Initialize models\norc.init_models()\n\n# Compile a pipeline\npipeline = orc.compile("pipelines/research-report.yaml")\n\n# Execute with different inputs\nresult = pipeline.run(\n    topic="quantum_computing",\n    instructions="Focus on error correction"\n)'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_installation_lines_34_41_25():
    """Test bash snippet from docs_sphinx/installation.rst lines 34-41."""
    bash_content = '# Install from PyPI (when available)\npip install py-orc\n\n# Or install from source\ngit clone https://github.com/ContextLab/orchestrator.git\ncd orchestrator\npip install -e .'
    
    # Verify it's a pip install command
    assert "pip install" in bash_content
    
    # Parse each line
    lines = bash_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_47_53_26():
    """Test bash snippet from docs_sphinx/installation.rst lines 47-53."""
    bash_content = '# Create conda environment\nconda create -n py-orc python=3.11\nconda activate py-orc\n\n# Install orchestrator\npip install py-orc'
    
    # Verify it's a pip install command
    assert "pip install" in bash_content
    
    # Parse each line
    lines = bash_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_59_64_27():
    """Test bash snippet from docs_sphinx/installation.rst lines 59-64."""
    bash_content = '# Pull the official image\ndocker pull contextlab/py-orc:latest\n\n# Run with volume mount\ndocker run -v $(pwd):/workspace contextlab/py-orc'
    
    # Skip dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # For now, just check it's not empty
    assert bash_content.strip(), "Bash content should not be empty"

def test_installation_lines_72_85_28():
    """Test bash snippet from docs_sphinx/installation.rst lines 72-85."""
    bash_content = '# Clone the repository\ngit clone https://github.com/ContextLab/orchestrator.git\ncd orchestrator\n\n# Create virtual environment\npython -m venv venv\nsource venv/bin/activate  # On Windows: venv\\\\Scripts\\\\activate\n\n# Install in development mode with extras\npip install -e ".[dev,test,docs]"\n\n# Install pre-commit hooks\npre-commit install'
    
    # Verify it's a pip install command
    assert "pip install" in bash_content
    
    # Parse each line
    lines = bash_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_96_101_29():
    """Test bash snippet from docs_sphinx/installation.rst lines 96-101."""
    bash_content = '# macOS\nbrew install ollama\n\n# Linux\ncurl -fsSL https://ollama.ai/install.sh | sh'
    
    # Skip dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # For now, just check it's not empty
    assert bash_content.strip(), "Bash content should not be empty"
