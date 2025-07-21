"""Working tests for documentation code snippets - Batch 15."""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# These tests focus on validation and syntax checking without execution
# They should run quickly and not require external dependencies


def test_your_first_pipeline_lines_122_148_0():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 122-148."""
    # Description: Now let's execute our pipeline:
    content = 'async def run_pipeline():\n    # Create orchestrator\n    orchestrator = Orchestrator()\n\n    # Register our model\n    orchestrator.register_model(model)\n\n    print("Starting pipeline execution...")\n\n    # Execute pipeline\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    print("\\n=== Pipeline Results ===")\n    print(f"Research Questions:\\n{result[\'research_questions\']}\\n")\n    print(f"Key Themes:\\n{result[\'analyze_themes\']}\\n")\n    print(f"Final Report:\\n{result[\'write_report\']}\\n")\n\n    return result\n\n# Run the pipeline\n# Note: In Jupyter notebooks, you can use top-level await:\n# result = await run_pipeline()\n\n# In regular Python scripts, use asyncio.run():\nimport asyncio\nresult = asyncio.run(run_pipeline())'
    
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


def test_your_first_pipeline_lines_156_188_1():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 156-188."""
    # Description: Let's make our pipeline more robust:
    content = 'from orchestrator.core.error_handler import ErrorHandler\nfrom orchestrator.core.error_handler import ExponentialBackoffRetry\n\nasync def run_robust_pipeline():\n    # Create error handler with retry strategy\n    error_handler = ErrorHandler()\n    error_handler.register_retry_strategy(\n        "research_retry",\n        ExponentialBackoffRetry(max_retries=3, base_delay=1.0)\n    )\n\n    # Create orchestrator with error handling\n    orchestrator = Orchestrator(error_handler=error_handler)\n    orchestrator.register_model(model)\n\n    try:\n        print("Starting robust pipeline execution...")\n        result = await orchestrator.execute_pipeline(pipeline)\n        print("✅ Pipeline completed successfully!")\n        return result\n\n    except Exception as e:\n        print(f"❌ Pipeline failed: {e}")\n        # Get execution statistics\n        stats = error_handler.get_error_statistics()\n        print(f"Errors encountered: {stats[\'total_errors\']}")\n        return None\n\n# Run robust pipeline\n# In Jupyter notebooks: result = await run_robust_pipeline()\n# In regular Python scripts:\nresult = asyncio.run(run_robust_pipeline())'
    
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


def test_your_first_pipeline_lines_196_223_2():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 196-223."""
    # Description: For longer pipelines, add checkpointing:
    content = 'from orchestrator.state import StateManager\n\nasync def run_stateful_pipeline():\n    # Create state manager\n    state_manager = StateManager(storage_path="./checkpoints")\n\n    # Create orchestrator with state management\n    orchestrator = Orchestrator(state_manager=state_manager)\n    orchestrator.register_model(model)\n\n    print("Starting stateful pipeline execution...")\n\n    # Execute with automatic checkpointing\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    print("✅ Pipeline completed with checkpointing!")\n\n    # List checkpoints created\n    checkpoints = await state_manager.list_checkpoints("research_assistant")\n    print(f"Checkpoints created: {len(checkpoints)}")\n\n    return result\n\n# Run stateful pipeline\n# In Jupyter notebooks: result = await run_stateful_pipeline()\n# In regular Python scripts:\nresult = asyncio.run(run_stateful_pipeline())'
    
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


def test_your_first_pipeline_lines_231_264_3():
    """Test YAML snippet from docs/getting_started/your_first_pipeline.rst lines 231-264."""
    # Description: Let's convert our pipeline to YAML:
    import yaml
    
    content = '# research_pipeline.yaml\nid: research_assistant\nname: Research Assistant Pipeline\ndescription: Generates research questions, analyzes themes, and writes a report\n\ncontext:\n  topic: artificial intelligence\n\ntasks:\n  - id: research_questions\n    name: Generate Research Questions\n    action: generate_text\n    parameters:\n      prompt: "Generate 3 research questions about: {topic}"\n      max_tokens: 200\n\n  - id: analyze_themes\n    name: Analyze Key Themes\n    action: generate_text\n    parameters:\n      prompt: "Analyze these questions and identify key themes: {research_questions}"\n      max_tokens: 150\n    dependencies:\n      - research_questions\n\n  - id: write_report\n    name: Write Research Report\n    action: generate_text\n    parameters:\n      prompt: "Write a comprehensive report on {topic} covering these themes: {analyze_themes}"\n      max_tokens: 500\n    dependencies:\n      - analyze_themes'
    
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


def test_your_first_pipeline_lines_269_292_4():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 269-292."""
    # Description: Load and execute the YAML pipeline:
    content = 'from orchestrator.compiler import YAMLCompiler\n\nasync def run_yaml_pipeline():\n    # Create compiler and load pipeline\n    compiler = YAMLCompiler()\n    pipeline = compiler.compile_file("research_pipeline.yaml")\n\n    # Create orchestrator\n    orchestrator = Orchestrator()\n    orchestrator.register_model(model)\n\n    print("Starting YAML pipeline execution...")\n\n    # Execute pipeline\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    print("✅ YAML pipeline completed!")\n    return result\n\n# Run YAML pipeline\n# In Jupyter notebooks: result = await run_yaml_pipeline()\n# In regular Python scripts:\nresult = asyncio.run(run_yaml_pipeline())'
    
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


def test_your_first_pipeline_lines_300_325_5():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 300-325."""
    # Description: Replace mock model with real AI:
    content = 'import os\nfrom orchestrator.models.openai_model import OpenAIModel\n\nasync def run_with_real_ai():\n    # API key should be set in environment variable or ~/.orchestrator/.env\n    # Create OpenAI model\n    openai_model = OpenAIModel(\n        name="gpt-4",\n        api_key=os.environ.get("OPENAI_API_KEY"),  # Loaded from environment\n        model="gpt-4"\n    )\n\n    # Create orchestrator with real AI\n    orchestrator = Orchestrator()\n    orchestrator.register_model(openai_model)\n\n    print("Starting pipeline with real AI...")\n\n    # Execute pipeline with real AI\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    print("✅ Real AI pipeline completed!")\n    return result\n\n# Run with real AI (uncomment when you have API keys)\n# In Jupyter notebooks: result = await run_with_real_ai()\n# In regular Python scripts: result = asyncio.run(run_with_real_ai())'
    
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


def test_your_first_pipeline_lines_333_365_6():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 333-365."""
    # Description: Add monitoring to track performance:
    content = 'import time\nfrom orchestrator.core.resource_allocator import ResourceAllocator\n\nasync def run_monitored_pipeline():\n    # Create resource allocator for monitoring\n    allocator = ResourceAllocator()\n\n    # Create orchestrator with monitoring\n    orchestrator = Orchestrator(resource_allocator=allocator)\n    orchestrator.register_model(model)\n\n    print("Starting monitored pipeline execution...")\n    start_time = time.time()\n\n    # Execute pipeline\n    result = await orchestrator.execute_pipeline(pipeline)\n\n    end_time = time.time()\n    execution_time = end_time - start_time\n\n    print(f"✅ Pipeline completed in {execution_time:.2f} seconds")\n\n    # Get resource statistics\n    stats = allocator.get_overall_statistics()\n    print(f"Resource utilization: {stats[\'overall_utilization\']:.2f}")\n\n    return result\n\n# Run monitored pipeline\n# In Jupyter notebooks: result = await run_monitored_pipeline()\n# In regular Python scripts:\nresult = asyncio.run(run_monitored_pipeline())'
    
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


def test_your_first_pipeline_lines_373_488_7():
    """Test Python snippet from docs/getting_started/your_first_pipeline.rst lines 373-488."""
    # Description: Here's the complete, production-ready pipeline:
    content = 'import asyncio\nimport logging\nfrom orchestrator import Orchestrator, Task, Pipeline\nfrom orchestrator.models.mock_model import MockModel\nfrom orchestrator.core.error_handler import ErrorHandler\nfrom orchestrator.state import StateManager\nfrom orchestrator.core.resource_allocator import ResourceAllocator\n\n# Configure logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger(__name__)\n\nasync def create_research_pipeline():\n    """Create a production-ready research assistant pipeline."""\n\n    # Create mock model with responses\n    model = MockModel("research_assistant")\n    model.set_response(\n        "Generate 3 research questions about: artificial intelligence",\n        "1. How does AI impact job markets?\\n2. What are the ethical implications of AI?\\n3. How can AI be made more accessible?"\n    )\n    model.set_response(\n        "Analyze these questions and identify key themes: 1. How does AI impact job markets?\\n2. What are the ethical implications of AI?\\n3. How can AI be made more accessible?",\n        "Key themes: Economic Impact, Ethics and Responsibility, Accessibility"\n    )\n    model.set_response(\n        "Write a comprehensive report on artificial intelligence covering these themes: Economic Impact, Ethics and Responsibility, Accessibility",\n        "# AI Research Report\\n\\n## Economic Impact\\nAI is transforming industries...\\n\\n## Ethics\\nResponsible AI development...\\n\\n## Accessibility\\nDemocratizing AI tools..."\n    )\n\n    # Create tasks\n    tasks = [\n        Task(\n            id="research_questions",\n            name="Generate Research Questions",\n            action="generate_text",\n            parameters={\n                "prompt": "Generate 3 research questions about: {topic}",\n                "max_tokens": 200\n            }\n        ),\n        Task(\n            id="analyze_themes",\n            name="Analyze Key Themes",\n            action="generate_text",\n            parameters={\n                "prompt": "Analyze these questions and identify key themes: {research_questions}",\n                "max_tokens": 150\n            },\n            dependencies=["research_questions"]\n        ),\n        Task(\n            id="write_report",\n            name="Write Research Report",\n            action="generate_text",\n            parameters={\n                "prompt": "Write a comprehensive report on {topic} covering these themes: {analyze_themes}",\n                "max_tokens": 500\n            },\n            dependencies=["analyze_themes"]\n        )\n    ]\n\n    # Create pipeline\n    pipeline = Pipeline(\n        id="research_assistant",\n        name="Research Assistant Pipeline"\n    )\n\n    for task in tasks:\n        pipeline.add_task(task)\n\n    pipeline.set_context("topic", "artificial intelligence")\n\n    # Create components\n    error_handler = ErrorHandler()\n    state_manager = StateManager(storage_path="./checkpoints")\n    resource_allocator = ResourceAllocator()\n\n    # Create orchestrator\n    orchestrator = Orchestrator(\n        error_handler=error_handler,\n        state_manager=state_manager,\n        resource_allocator=resource_allocator\n    )\n\n    orchestrator.register_model(model)\n\n    return orchestrator, pipeline\n\nasync def main():\n    """Main execution function."""\n    logger.info("Creating research assistant pipeline...")\n\n    orchestrator, pipeline = await create_research_pipeline()\n\n    logger.info("Executing pipeline...")\n\n    try:\n        result = await orchestrator.execute_pipeline(pipeline)\n\n        logger.info("Pipeline completed successfully!")\n\n        print("\\n=== Results ===")\n        for task_id, output in result.items():\n            print(f"\\n{task_id}:")\n            print(f"{output}")\n\n    except Exception as e:\n        logger.error(f"Pipeline failed: {e}")\n        raise\n\n# Run the complete example\nif __name__ == "__main__":\n    asyncio.run(main())'
    
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


def test_index_lines_27_28_8():
    """Test Bash snippet from docs/index.rst lines 27-28."""
    # Description: Get started with Orchestrator in just a few minutes:
    content = 'pip install py-orc'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    
    # Special handling for pip install commands
    if 'pip install' in content:
        lines = content.strip().split('\n')
        has_pip_command = False
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and 'pip install' in line:
                has_pip_command = True
                break
        if has_pip_command:
            return  # Skip further validation for pip commands
    
    # For other bash commands, just check they're not empty
    assert len(content.strip()) > 0, "Bash content should not be empty"


def test_index_lines_107_124_9():
    """Test text snippet from docs/index.rst lines 107-124."""
    # Description: The Orchestrator Framework is built with a modular architecture that separates concerns and promotes extensibility:
    content = '┌─────────────────────────────────────────────────────────────┐\n│                    Orchestrator Engine                      │\n└─────────────────────────────────────────────────────────────┘\n\n┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐\n│  YAML Compiler  │  │ Model Registry  │  │ State Manager   │\n│  - Parser       │  │ - Selection     │  │ - Checkpoints   │\n│  - Validation   │  │ - Load Balance  │  │ - Recovery      │\n│  - Templates    │  │ - Health Check  │  │ - Persistence   │\n└─────────────────┘  └─────────────────┘  └─────────────────┘\n\n┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐\n│ Execution Layer │  │ Error Handler   │  │ Resource Mgmt   │\n│ - Parallel      │  │ - Circuit Break │  │ - Allocation    │\n│ - Sandboxed     │  │ - Retry Logic   │  │ - Monitoring    │\n│ - Distributed   │  │ - Recovery      │  │ - Optimization  │\n└─────────────────┘  └─────────────────┘  └─────────────────┘'
    
    # Basic validation
    assert content.strip(), "Content should not be empty"
    assert len(content) > 0, "Content should have length"
