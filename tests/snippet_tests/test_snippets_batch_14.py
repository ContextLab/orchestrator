"""Working tests for documentation code snippets - Batch 14."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_quickstart_lines_61_114_0():
    """Test Python snippet from docs/getting_started/quickstart.rst lines 61-114."""
    # Description: Let's create a more complex pipeline with multiple tasks:
    content = 'from orchestrator import Task, Pipeline\n\n# Task 1: Generate story outline\noutline_task = Task(\n    id="outline",\n    name="Generate Story Outline",\n    action="generate_text",\n    parameters={"prompt": "Create a story outline about space exploration"}\n)\n\n# Task 2: Write story (depends on outline)\nstory_task = Task(\n    id="story",\n    name="Write Story",\n    action="generate_text",\n    parameters={"prompt": "Write a story based on: {outline}"},\n    dependencies=["outline"]\n)\n\n# Task 3: Summarize story (depends on story)\nsummary_task = Task(\n    id="summary",\n    name="Summarize Story",\n    action="generate_text",\n    parameters={"prompt": "Summarize this story: {story}"},\n    dependencies=["story"]\n)\n\n# Create pipeline with all tasks\npipeline = Pipeline(id="story_pipeline", name="Story Creation Pipeline")\npipeline.add_task(outline_task)\npipeline.add_task(story_task)\npipeline.add_task(summary_task)\n\n# Execute pipeline\nimport asyncio\n\n\n\nasync def run_pipeline():\n\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    return result\n\n\n\n# Run the pipeline\n\nresult = asyncio.run(run_pipeline())\nprint(f"Outline: {result[\'outline\']}")\nprint(f"Story: {result[\'story\']}")\nprint(f"Summary: {result[\'summary\']}")'
    
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


def test_quickstart_lines_122_148_1():
    """Test YAML snippet from docs/getting_started/quickstart.rst lines 122-148."""
    # Description: You can also define pipelines in YAML:
    import yaml
    
    content = '# story_pipeline.yaml\nid: story_pipeline\nname: Story Creation Pipeline\n\ntasks:\n  - id: outline\n    name: Generate Story Outline\n    action: generate_text\n    parameters:\n      prompt: "Create a story outline about space exploration"\n\n  - id: story\n    name: Write Story\n    action: generate_text\n    parameters:\n      prompt: "Write a story based on: {outline}"\n    dependencies:\n      - outline\n\n  - id: summary\n    name: Summarize Story\n    action: generate_text\n    parameters:\n      prompt: "Summarize this story: {story}"\n    dependencies:\n      - story'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Check if it's valid YAML
    try:
        # Check if content contains AUTO tags
        if '<AUTO>' in content:
            # Use AUTO tag parser
            from orchestrator.compiler.auto_tag_yaml_parser import AutoTagYAMLParser
            parser = AutoTagYAMLParser()
            data = parser.parse(content)
        else:
            # Use standard YAML parser
            data = yaml.safe_load(content)
        # Note: data can be None for YAML with only comments
    except (yaml.YAMLError, ValueError) as e:
        pytest.fail(f"YAML parsing error: {e}")
    
    # If it looks like a pipeline, do basic structure validation
    if isinstance(data, dict) and ('steps' in data or 'tasks' in data):
        if 'steps' in data:
            assert isinstance(data['steps'], list), "Steps should be a list"
            for step in data['steps']:
                assert isinstance(step, dict), "Each step should be a dict"
                # Note: 'id' is optional in minimal examples


def test_quickstart_lines_153_175_2():
    """Test Python snippet from docs/getting_started/quickstart.rst lines 153-175."""
    # Description: Load and execute the YAML pipeline:
    content = 'from orchestrator.compiler import YAMLCompiler\n\n# Load pipeline from YAML\ncompiler = YAMLCompiler()\npipeline = compiler.compile_file("story_pipeline.yaml")\n\n# Execute pipeline\nimport asyncio\n\n\n\nasync def run_pipeline():\n\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    return result\n\n\n\n# Run the pipeline\n\nresult = asyncio.run(run_pipeline())'
    
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


def test_quickstart_lines_183_211_3():
    """Test Python snippet from docs/getting_started/quickstart.rst lines 183-211."""
    # Description: Let's use a real AI model instead of the mock:
    content = 'import os\nfrom orchestrator.models.openai_model import OpenAIModel\n\n# API key should be set in environment variable or ~/.orchestrator/.env\n# Create OpenAI model\nopenai_model = OpenAIModel(\n    name="gpt-4",\n    api_key=os.environ.get("OPENAI_API_KEY"),  # Loaded from environment\n    model="gpt-4"\n)\n\n# Register model\norchestrator.register_model(openai_model)\n\n# Execute pipeline (will use OpenAI)\nimport asyncio\n\n\n\nasync def run_pipeline():\n\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    return result\n\n\n\n# Run the pipeline\n\nresult = asyncio.run(run_pipeline())'
    
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


def test_quickstart_lines_219_246_4():
    """Test Python snippet from docs/getting_started/quickstart.rst lines 219-246."""
    # Description: Orchestrator provides built-in error handling:
    content = 'from orchestrator.core.error_handler import ErrorHandler\n\n# Create error handler with retry strategy\nerror_handler = ErrorHandler()\n\n# Configure orchestrator with error handling\norchestrator = Orchestrator(error_handler=error_handler)\n\n# Execute pipeline with automatic retry on failures\ntry:\n    import asyncio\n\n\n\n    async def run_pipeline():\n\n        result = await orchestrator.execute_pipeline(pipeline)\n\n        return result\n\n\n\n    # Run the pipeline\n\n    result = asyncio.run(run_pipeline())\nexcept Exception as e:\n    print(f"Pipeline failed: {e}")'
    
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


def test_quickstart_lines_254_278_5():
    """Test Python snippet from docs/getting_started/quickstart.rst lines 254-278."""
    # Description: Enable checkpointing for long-running pipelines:
    content = 'from orchestrator.state import StateManager\n\n# Create state manager\nstate_manager = StateManager(storage_path="./checkpoints")\n\n# Configure orchestrator with state management\norchestrator = Orchestrator(state_manager=state_manager)\n\n# Execute pipeline with automatic checkpointing\nimport asyncio\n\n\n\nasync def run_pipeline():\n\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    return result\n\n\n\n# Run the pipeline\n\nresult = asyncio.run(run_pipeline())'
    
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


def test_quickstart_lines_286_312_6():
    """Test Python snippet from docs/getting_started/quickstart.rst lines 286-312."""
    # Description: Enable monitoring to track pipeline execution:
    content = 'import logging\n\n# Enable debug logging\nlogging.basicConfig(level=logging.DEBUG)\n\n# Execute pipeline with logging\nimport asyncio\n\n\n\nasync def run_pipeline():\n\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    return result\n\n\n\n# Run the pipeline\n\nresult = asyncio.run(run_pipeline())\n\n# Get execution statistics\nstats = orchestrator.get_execution_stats()\nprint(f"Execution time: {stats[\'total_time\']:.2f}s")\nprint(f"Tasks completed: {stats[\'completed_tasks\']}")'
    
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


def test_your_first_pipeline_lines_24_46_7():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 24-46."""
    # Description: First, let's set up our environment:
    content = 'import asyncio\nimport os\nfrom orchestrator import Orchestrator, Task, Pipeline\nfrom orchestrator.models.openai_model import OpenAIModel\nfrom orchestrator.utils.api_keys import load_api_keys\n\n# Load API keys from environment\nload_api_keys()\n\n# Create a real OpenAI model\nmodel = OpenAIModel(\n    name="gpt-3.5-turbo",\n    api_key=os.environ.get("OPENAI_API_KEY"),  # Loaded from environment\n)\n\n# Note: Make sure you have set your OPENAI_API_KEY environment variable\n# You can also use AnthropicModel with ANTHROPIC_API_KEY if preferred'
    
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


def test_your_first_pipeline_lines_54_88_8():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 54-88."""
    # Description: Now let's create our three tasks:
    content = '# Task 1: Generate research questions\nresearch_task = Task(\n    id="research_questions",\n    name="Generate Research Questions",\n    action="generate_text",\n    parameters={\n        "prompt": "Generate 3 research questions about: {topic}",\n        "max_tokens": 200\n    }\n)\n\n# Task 2: Analyze questions for themes\nanalysis_task = Task(\n    id="analyze_themes",\n    name="Analyze Key Themes",\n    action="generate_text",\n    parameters={\n        "prompt": "Analyze these questions and identify key themes: {research_questions}",\n        "max_tokens": 150\n    },\n    dependencies=["research_questions"]  # Depends on research task\n)\n\n# Task 3: Write comprehensive report\nreport_task = Task(\n    id="write_report",\n    name="Write Research Report",\n    action="generate_text",\n    parameters={\n        "prompt": "Write a comprehensive report on {topic} covering these themes: {analyze_themes}",\n        "max_tokens": 500\n    },\n    dependencies=["analyze_themes"]  # Depends on analysis task\n)'
    
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


def test_your_first_pipeline_lines_96_114_9():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 96-114."""
    # Description: Combine tasks into a pipeline:
    content = '# Create pipeline\npipeline = Pipeline(\n    id="research_assistant",\n    name="Research Assistant Pipeline",\n    description="Generates research questions, analyzes themes, and writes a report"\n)\n\n# Add tasks to pipeline\npipeline.add_task(research_task)\npipeline.add_task(analysis_task)\npipeline.add_task(report_task)\n\n# Set initial context\npipeline.set_context("topic", "artificial intelligence")\n\nprint("Pipeline created successfully!")\nprint(f"Tasks: {list(pipeline.tasks.keys())}")\nprint(f"Execution order: {pipeline.get_execution_order()}")'
    
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
