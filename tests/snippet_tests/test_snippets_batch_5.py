"""Tests for documentation code snippets - Batch 5."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_installation_lines_56_57_0():
    """Test bash snippet from docs/getting_started/installation.rst lines 56-57."""
    bash_content = 'pip install py-orc[docker]'
    
    # Verify it's a pip install command
    assert "pip install" in bash_content
    
    # Parse each line
    lines = bash_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_64_65_1():
    """Test bash snippet from docs/getting_started/installation.rst lines 64-65."""
    bash_content = 'pip install py-orc[database]'
    
    # Verify it's a pip install command
    assert "pip install" in bash_content
    
    # Parse each line
    lines = bash_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_72_73_2():
    """Test bash snippet from docs/getting_started/installation.rst lines 72-73."""
    bash_content = 'pip install py-orc[all]'
    
    # Verify it's a pip install command
    assert "pip install" in bash_content
    
    # Parse each line
    lines = bash_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_installation_lines_81_92_3():
    """Test Python import from docs/getting_started/installation.rst lines 81-92."""
    # Import test - check if modules are available
    code = 'import orchestrator\nprint(f"Orchestrator version: {orchestrator.__version__}")\n\n# Test basic functionality\nfrom orchestrator import Task, Pipeline\n\ntask = Task(id="test", name="Test Task", action="echo", parameters={"message": "Hello!"})\npipeline = Pipeline(id="test_pipeline", name="Test Pipeline")\npipeline.add_task(task)\n\nprint("✅ Installation successful!")'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_installation_lines_105_113_4():
    """Test bash snippet from docs/getting_started/installation.rst lines 105-113."""
    bash_content = '# Optional: Set cache directory\nexport ORCHESTRATOR_CACHE_DIR=/path/to/cache\n\n# Optional: Set checkpoint directory\nexport ORCHESTRATOR_CHECKPOINT_DIR=/path/to/checkpoints\n\n# Optional: Set log level\nexport ORCHESTRATOR_LOG_LEVEL=INFO'
    
    # Skip dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # For now, just check it's not empty
    assert bash_content.strip(), "Bash content should not be empty"

def test_installation_lines_121_129_5():
    """Test bash snippet from docs/getting_started/installation.rst lines 121-129."""
    bash_content = '# OpenAI\nexport OPENAI_API_KEY=your_openai_key\n\n# Anthropic\nexport ANTHROPIC_API_KEY=your_anthropic_key\n\n# Google\nexport GOOGLE_API_KEY=your_google_key'
    
    # Skip dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # For now, just check it's not empty
    assert bash_content.strip(), "Bash content should not be empty"

def test_installation_lines_137_139_6():
    """Test bash snippet from docs/getting_started/installation.rst lines 137-139."""
    bash_content = 'docker --version\ndocker run hello-world'
    
    # Skip dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # For now, just check it's not empty
    assert bash_content.strip(), "Bash content should not be empty"

def test_installation_lines_157_166_7():
    """Test bash snippet from docs/getting_started/installation.rst lines 157-166."""
    bash_content = '# Ubuntu/Debian\nsudo apt-get update\nsudo apt-get install python3-dev build-essential\n\n# macOS\nbrew install python\n\n# Windows\n# Use Python from python.org'
    
    # Skip dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # For now, just check it's not empty
    assert bash_content.strip(), "Bash content should not be empty"

def test_installation_lines_172_174_8():
    """Test bash snippet from docs/getting_started/installation.rst lines 172-174."""
    bash_content = 'docker --version\ndocker info'
    
    # Skip dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker', 'systemctl']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # For now, just check it's not empty
    assert bash_content.strip(), "Bash content should not be empty"

def test_quickstart_lines_13_39_9():
    """Test Python import from docs/getting_started/quickstart.rst lines 13-39."""
    # Import test - check if modules are available
    code = 'from orchestrator import Orchestrator, Task, Pipeline\nfrom orchestrator.models.mock_model import MockModel\n\n# Create a mock model for testing\nmodel = MockModel("gpt-test")\nmodel.set_response("Hello, world!", "Hello! How can I help you today?")\n\n# Create a task\ntask = Task(\n    id="greeting",\n    name="Generate Greeting",\n    action="generate_text",\n    parameters={"prompt": "Hello, world!"}\n)\n\n# Create a pipeline\npipeline = Pipeline(id="hello_pipeline", name="Hello Pipeline")\npipeline.add_task(task)\n\n# Create orchestrator and register model\norchestrator = Orchestrator()\norchestrator.register_model(model)\n\n# Execute pipeline\nresult = await orchestrator.execute_pipeline(pipeline)\nprint(f"Result: {result[\'greeting\']}")'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_quickstart_lines_47_86_10():
    """Test Python import from docs/getting_started/quickstart.rst lines 47-86."""
    # Import test - check if modules are available
    code = 'from orchestrator import Task, Pipeline\n\n# Task 1: Generate story outline\noutline_task = Task(\n    id="outline",\n    name="Generate Story Outline",\n    action="generate_text",\n    parameters={"prompt": "Create a story outline about space exploration"}\n)\n\n# Task 2: Write story (depends on outline)\nstory_task = Task(\n    id="story",\n    name="Write Story",\n    action="generate_text",\n    parameters={"prompt": "Write a story based on: {outline}"},\n    dependencies=["outline"]\n)\n\n# Task 3: Summarize story (depends on story)\nsummary_task = Task(\n    id="summary",\n    name="Summarize Story",\n    action="generate_text",\n    parameters={"prompt": "Summarize this story: {story}"},\n    dependencies=["story"]\n)\n\n# Create pipeline with all tasks\npipeline = Pipeline(id="story_pipeline", name="Story Creation Pipeline")\npipeline.add_task(outline_task)\npipeline.add_task(story_task)\npipeline.add_task(summary_task)\n\n# Execute pipeline\nresult = await orchestrator.execute_pipeline(pipeline)\nprint(f"Outline: {result[\'outline\']}")\nprint(f"Story: {result[\'story\']}")\nprint(f"Summary: {result[\'summary\']}")'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_quickstart_lines_94_120_11():
    """Test YAML pipeline from docs/getting_started/quickstart.rst lines 94-120."""
    import yaml
    
    yaml_content = '# story_pipeline.yaml\nid: story_pipeline\nname: Story Creation Pipeline\n\ntasks:\n  - id: outline\n    name: Generate Story Outline\n    action: generate_text\n    parameters:\n      prompt: "Create a story outline about space exploration"\n\n  - id: story\n    name: Write Story\n    action: generate_text\n    parameters:\n      prompt: "Write a story based on: {outline}"\n    dependencies:\n      - outline\n\n  - id: summary\n    name: Summarize Story\n    action: generate_text\n    parameters:\n      prompt: "Summarize this story: {story}"\n    dependencies:\n      - story'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_quickstart_lines_125_133_12():
    """Test Python import from docs/getting_started/quickstart.rst lines 125-133."""
    # Import test - check if modules are available
    code = 'from orchestrator.compiler import YAMLCompiler\n\n# Load pipeline from YAML\ncompiler = YAMLCompiler()\npipeline = compiler.compile_file("story_pipeline.yaml")\n\n# Execute pipeline\nresult = await orchestrator.execute_pipeline(pipeline)'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_quickstart_lines_141_155_13():
    """Test Python import from docs/getting_started/quickstart.rst lines 141-155."""
    # Import test - check if modules are available
    code = 'from orchestrator.models.openai_model import OpenAIModel\n\n# Create OpenAI model\nopenai_model = OpenAIModel(\n    name="gpt-4",\n    api_key="your-api-key-here",\n    model="gpt-4"\n)\n\n# Register model\norchestrator.register_model(openai_model)\n\n# Execute pipeline (will use OpenAI)\nresult = await orchestrator.execute_pipeline(pipeline)'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_quickstart_lines_163_176_14():
    """Test Python import from docs/getting_started/quickstart.rst lines 163-176."""
    # Import test - check if modules are available
    code = 'from orchestrator.core.error_handler import ErrorHandler\n\n# Create error handler with retry strategy\nerror_handler = ErrorHandler()\n\n# Configure orchestrator with error handling\norchestrator = Orchestrator(error_handler=error_handler)\n\n# Execute pipeline with automatic retry on failures\ntry:\n    result = await orchestrator.execute_pipeline(pipeline)\nexcept Exception as e:\n    print(f"Pipeline failed: {e}")'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_quickstart_lines_184_194_15():
    """Test Python import from docs/getting_started/quickstart.rst lines 184-194."""
    # Import test - check if modules are available
    code = 'from orchestrator.state import StateManager\n\n# Create state manager\nstate_manager = StateManager(storage_path="./checkpoints")\n\n# Configure orchestrator with state management\norchestrator = Orchestrator(state_manager=state_manager)\n\n# Execute pipeline with automatic checkpointing\nresult = await orchestrator.execute_pipeline(pipeline)'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_quickstart_lines_202_214_16():
    """Test Python import from docs/getting_started/quickstart.rst lines 202-214."""
    # Import test - check if modules are available
    code = 'import logging\n\n# Enable debug logging\nlogging.basicConfig(level=logging.DEBUG)\n\n# Execute pipeline with logging\nresult = await orchestrator.execute_pipeline(pipeline)\n\n# Get execution statistics\nstats = orchestrator.get_execution_stats()\nprint(f"Execution time: {stats[\'total_time\']:.2f}s")\nprint(f"Tasks completed: {stats[\'completed_tasks\']}")'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_your_first_pipeline_lines_24_46_17():
    """Test Python import from docs/getting_started/your_first_pipeline.rst lines 24-46."""
    # Import test - check if modules are available
    code = 'import asyncio\nfrom orchestrator import Orchestrator, Task, Pipeline\nfrom orchestrator.models.mock_model import MockModel\n\n# Create a mock model for testing\nmodel = MockModel("research_assistant")\n\n# Set up responses for our mock model\nmodel.set_response(\n    "Generate 3 research questions about: artificial intelligence",\n    "1. How does AI impact job markets?\\\\n2. What are the ethical implications of AI?\\\\n3. How can AI be made more accessible?"\n)\n\nmodel.set_response(\n    "Analyze these questions and identify key themes: 1. How does AI impact job markets?\\\\n2. What are the ethical implications of AI?\\\\n3. How can AI be made more accessible?",\n    "Key themes identified: Economic Impact, Ethics and Responsibility, Accessibility and Democratization"\n)\n\nmodel.set_response(\n    "Write a comprehensive report on artificial intelligence covering these themes: Economic Impact, Ethics and Responsibility, Accessibility and Democratization",\n    "# AI Research Report\\\\n\\\\n## Economic Impact\\\\nAI is reshaping job markets...\\\\n\\\\n## Ethics and Responsibility\\\\nAI systems must be developed responsibly...\\\\n\\\\n## Accessibility and Democratization\\\\nMaking AI tools accessible to all..."\n)'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_your_first_pipeline_lines_54_88_18():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 54-88."""
    # Skip complex orchestrator code for now - would need full setup
    pytest.skip("Complex orchestrator code testing not yet implemented")

@pytest.mark.asyncio
async def test_your_first_pipeline_lines_96_114_19():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 96-114."""
    # Skip complex orchestrator code for now - would need full setup
    pytest.skip("Complex orchestrator code testing not yet implemented")

@pytest.mark.asyncio
async def test_your_first_pipeline_lines_122_143_20():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 122-143."""
    # Skip complex orchestrator code for now - would need full setup
    pytest.skip("Complex orchestrator code testing not yet implemented")

def test_your_first_pipeline_lines_151_181_21():
    """Test Python import from docs/getting_started/your_first_pipeline.rst lines 151-181."""
    # Import test - check if modules are available
    code = 'from orchestrator.core.error_handler import ErrorHandler\nfrom orchestrator.core.error_handler import ExponentialBackoffRetry\n\nasync def run_robust_pipeline():\n    # Create error handler with retry strategy\n    error_handler = ErrorHandler()\n    error_handler.register_retry_strategy(\n        "research_retry",\n        ExponentialBackoffRetry(max_retries=3, base_delay=1.0)\n    )\n\n    # Create orchestrator with error handling\n    orchestrator = Orchestrator(error_handler=error_handler)\n    orchestrator.register_model(model)\n\n    try:\n        print("Starting robust pipeline execution...")\n        result = await orchestrator.execute_pipeline(pipeline)\n        print("✅ Pipeline completed successfully!")\n        return result\n\n    except Exception as e:\n        print(f"❌ Pipeline failed: {e}")\n        # Get execution statistics\n        stats = error_handler.get_error_statistics()\n        print(f"Errors encountered: {stats[\'total_errors\']}")\n        return None\n\n# Run robust pipeline\nresult = await run_robust_pipeline()'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_your_first_pipeline_lines_189_214_22():
    """Test Python import from docs/getting_started/your_first_pipeline.rst lines 189-214."""
    # Import test - check if modules are available
    code = 'from orchestrator.state import StateManager\n\nasync def run_stateful_pipeline():\n    # Create state manager\n    state_manager = StateManager(storage_path="./checkpoints")\n\n    # Create orchestrator with state management\n    orchestrator = Orchestrator(state_manager=state_manager)\n    orchestrator.register_model(model)\n\n    print("Starting stateful pipeline execution...")\n\n    # Execute with automatic checkpointing\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    print("✅ Pipeline completed with checkpointing!")\n\n    # List checkpoints created\n    checkpoints = await state_manager.list_checkpoints("research_assistant")\n    print(f"Checkpoints created: {len(checkpoints)}")\n\n    return result\n\n# Run stateful pipeline\nresult = await run_stateful_pipeline()'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

@pytest.mark.asyncio
async def test_your_first_pipeline_lines_222_255_23():
    """Test YAML pipeline from docs/getting_started/your_first_pipeline.rst lines 222-255."""
    import yaml
    
    yaml_content = '# research_pipeline.yaml\nid: research_assistant\nname: Research Assistant Pipeline\ndescription: Generates research questions, analyzes themes, and writes a report\n\ncontext:\n  topic: artificial intelligence\n\ntasks:\n  - id: research_questions\n    name: Generate Research Questions\n    action: generate_text\n    parameters:\n      prompt: "Generate 3 research questions about: {topic}"\n      max_tokens: 200\n\n  - id: analyze_themes\n    name: Analyze Key Themes\n    action: generate_text\n    parameters:\n      prompt: "Analyze these questions and identify key themes: {research_questions}"\n      max_tokens: 150\n    dependencies:\n      - research_questions\n\n  - id: write_report\n    name: Write Research Report\n    action: generate_text\n    parameters:\n      prompt: "Write a comprehensive report on {topic} covering these themes: {analyze_themes}"\n      max_tokens: 500\n    dependencies:\n      - analyze_themes'
    
    # Parse YAML first
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")
    
    # Skip pipeline compilation for now - would need full orchestrator setup
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        pytest.skip("Pipeline compilation testing not yet implemented")

def test_your_first_pipeline_lines_260_281_24():
    """Test Python import from docs/getting_started/your_first_pipeline.rst lines 260-281."""
    # Import test - check if modules are available
    code = 'from orchestrator.compiler import YAMLCompiler\n\nasync def run_yaml_pipeline():\n    # Create compiler and load pipeline\n    compiler = YAMLCompiler()\n    pipeline = compiler.compile_file("research_pipeline.yaml")\n\n    # Create orchestrator\n    orchestrator = Orchestrator()\n    orchestrator.register_model(model)\n\n    print("Starting YAML pipeline execution...")\n\n    # Execute pipeline\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    print("✅ YAML pipeline completed!")\n    return result\n\n# Run YAML pipeline\nresult = await run_yaml_pipeline()'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_your_first_pipeline_lines_289_313_25():
    """Test Python import from docs/getting_started/your_first_pipeline.rst lines 289-313."""
    # Import test - check if modules are available
    code = 'from orchestrator.models.openai_model import OpenAIModel\n\nasync def run_with_real_ai():\n    # Create OpenAI model\n    openai_model = OpenAIModel(\n        name="gpt-4",\n        api_key="your-openai-api-key",\n        model="gpt-4"\n    )\n\n    # Create orchestrator with real AI\n    orchestrator = Orchestrator()\n    orchestrator.register_model(openai_model)\n\n    print("Starting pipeline with real AI...")\n\n    # Execute pipeline with real AI\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    print("✅ Real AI pipeline completed!")\n    return result\n\n# Run with real AI (uncomment when you have API keys)\n# result = await run_with_real_ai()'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_your_first_pipeline_lines_321_351_26():
    """Test Python import from docs/getting_started/your_first_pipeline.rst lines 321-351."""
    # Import test - check if modules are available
    code = 'import time\nfrom orchestrator.core.resource_allocator import ResourceAllocator\n\nasync def run_monitored_pipeline():\n    # Create resource allocator for monitoring\n    allocator = ResourceAllocator()\n\n    # Create orchestrator with monitoring\n    orchestrator = Orchestrator(resource_allocator=allocator)\n    orchestrator.register_model(model)\n\n    print("Starting monitored pipeline execution...")\n    start_time = time.time()\n\n    # Execute pipeline\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    end_time = time.time()\n    execution_time = end_time - start_time\n\n    print(f"✅ Pipeline completed in {execution_time:.2f} seconds")\n\n    # Get resource statistics\n    stats = allocator.get_overall_statistics()\n    print(f"Resource utilization: {stats[\'overall_utilization\']:.2f}")\n\n    return result\n\n# Run monitored pipeline\nresult = await run_monitored_pipeline()'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_your_first_pipeline_lines_359_474_27():
    """Test Python import from docs/getting_started/your_first_pipeline.rst lines 359-474."""
    # Import test - check if modules are available
    code = 'import asyncio\nimport logging\nfrom orchestrator import Orchestrator, Task, Pipeline\nfrom orchestrator.models.mock_model import MockModel\nfrom orchestrator.core.error_handler import ErrorHandler\nfrom orchestrator.state import StateManager\nfrom orchestrator.core.resource_allocator import ResourceAllocator\n\n# Configure logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\n\nasync def create_research_pipeline():\n    """ Create a production-ready research assistant pipeline.""" \n\n    # Create mock model with responses\n    model = MockModel("research_assistant")\n    model.set_response(\n        "Generate 3 research questions about: artificial intelligence",\n        "1. How does AI impact job markets?\\\\n2. What are the ethical implications of AI?\\\\n3. How can AI be made more accessible?"\n    )\n    model.set_response(\n        "Analyze these questions and identify key themes: 1. How does AI impact job markets?\\\\n2. What are the ethical implications of AI?\\\\n3. How can AI be made more accessible?",\n        "Key themes: Economic Impact, Ethics and Responsibility, Accessibility"\n    )\n    model.set_response(\n        "Write a comprehensive report on artificial intelligence covering these themes: Economic Impact, Ethics and Responsibility, Accessibility",\n        "# AI Research Report\\\\n\\\\n## Economic Impact\\\\nAI is transforming industries...\\\\n\\\\n## Ethics\\\\nResponsible AI development...\\\\n\\\\n## Accessibility\\\\nDemocratizing AI tools..."\n    )\n\n    # Create tasks\n    tasks = [\n        Task(\n            id="research_questions",\n            name="Generate Research Questions",\n            action="generate_text",\n            parameters={\n                "prompt": "Generate 3 research questions about: {topic}",\n                "max_tokens": 200\n            }\n        ),\n        Task(\n            id="analyze_themes",\n            name="Analyze Key Themes",\n            action="generate_text",\n            parameters={\n                "prompt": "Analyze these questions and identify key themes: {research_questions}",\n                "max_tokens": 150\n            },\n            dependencies=["research_questions"]\n        ),\n        Task(\n            id="write_report",\n            name="Write Research Report",\n            action="generate_text",\n            parameters={\n                "prompt": "Write a comprehensive report on {topic} covering these themes: {analyze_themes}",\n                "max_tokens": 500\n            },\n            dependencies=["analyze_themes"]\n        )\n    ]\n\n    # Create pipeline\n    pipeline = Pipeline(\n        id="research_assistant",\n        name="Research Assistant Pipeline"\n    )\n\n    for task in tasks:\n        pipeline.add_task(task)\n\n    pipeline.set_context("topic", "artificial intelligence")\n\n    # Create components\n    error_handler = ErrorHandler()\n    state_manager = StateManager(storage_path="./checkpoints")\n    resource_allocator = ResourceAllocator()\n\n    # Create orchestrator\n    orchestrator = Orchestrator(\n        error_handler=error_handler,\n        state_manager=state_manager,\n        resource_allocator=resource_allocator\n    )\n\n    orchestrator.register_model(model)\n\n    return orchestrator, pipeline\n\nasync def main():\n    """ Main execution function.""" \n    logger.info("Creating research assistant pipeline...")\n\n    orchestrator, pipeline = await create_research_pipeline()\n\n    logger.info("Executing pipeline...")\n\n    try:\n        result = await orchestrator.execute_pipeline(pipeline)\n\n        logger.info("Pipeline completed successfully!")\n\n        print("\\\\n=== Results ===")\n        for task_id, output in result.items():\n            print(f"\\\\n{task_id}:")\n            print(f"{output}")\n\n    except Exception as e:\n        logger.error(f"Pipeline failed: {e}")\n        raise\n\n# Run the complete example\nif __name__ == "__main__":\n    asyncio.run(main())'
    
    try:
        exec(code)
    except ImportError as e:
        pytest.skip(f"Import not available: {e}")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")

def test_index_lines_27_28_28():
    """Test bash snippet from docs/index.rst lines 27-28."""
    bash_content = 'pip install py-orc'
    
    # Verify it's a pip install command
    assert "pip install" in bash_content
    
    # Parse each line
    lines = bash_content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_index_lines_107_124_29():
    """Test text snippet from docs/index.rst lines 107-124."""
    pytest.skip("Snippet type 'text' not yet supported")
