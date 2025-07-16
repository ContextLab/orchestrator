"""Tests for documentation code snippets - Batch 3 (Fixed)."""
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


def test_README_lines_74_80_0():
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
    
    # Skip potentially dangerous commands
    skip_commands = ['rm -rf', 'sudo', 'docker run', 'systemctl', 'service']
    if any(cmd in bash_content for cmd in skip_commands):
        pytest.skip("Skipping potentially destructive command")
    
    # Check bash syntax only
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

def test_README_lines_89_89_1():
    """Test YAML snippet from examples/README.md lines 89-89."""
    import yaml
    
    yaml_content = r"""analysis_type: <AUTO>What type of analysis is most appropriate for this text?</AUTO>"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_README_lines_94_94_2():
    """Test YAML snippet from examples/README.md lines 94-94."""
    import yaml
    
    yaml_content = r"""format: <AUTO>Determine the best format for this data source</AUTO>"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_README_lines_99_99_3():
    """Test YAML snippet from examples/README.md lines 99-99."""
    import yaml
    
    yaml_content = r"""methods: <AUTO>Choose the most appropriate statistical methods</AUTO>"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

def test_README_lines_104_104_4():
    """Test YAML snippet from examples/README.md lines 104-104."""
    import yaml
    
    yaml_content = r"""validation_rules: <AUTO>Generate appropriate validation rules for this dataset</AUTO>"""
    
    try:
        data = yaml.safe_load(yaml_content)
        assert data is not None
    except yaml.YAMLError as e:
        pytest.fail(f"YAML parsing failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_README_lines_112_131_5():
    """Test YAML pipeline from examples/README.md lines 112-131."""
    import yaml
    import orchestrator
    
    yaml_content = r"""id: my_pipeline
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
async def test_README_lines_137_152_6():
    """Test YAML pipeline from examples/README.md lines 137-152."""
    import yaml
    import orchestrator
    
    yaml_content = r"""steps:
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
async def test_README_lines_181_184_7():
    """Test orchestrator code from examples/README.md lines 181-184."""
    # Enable debug logging to see detailed execution information:
    
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
        if 'hello_world.yaml' in r"""import logging
logging.basicConfig(level=logging.DEBUG)

orchestrator = Orchestrator()""":
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
            exec(r"""import logging
logging.basicConfig(level=logging.DEBUG)

orchestrator = Orchestrator()""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
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
async def test_README_lines_192_193_8():
    """Test orchestrator code from examples/README.md lines 192-193."""
    # Verify system health before running pipelines:
    
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
        if 'hello_world.yaml' in r"""health = await orchestrator.health_check()
print("System health:", health["overall"])""":
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
            exec(r"""health = await orchestrator.health_check()
print("System health:", health["overall"])""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)

def test_draft_report_lines_59_59_9():
    """Test text snippet from examples/output/readme_report/draft_report.md lines 59-59."""
    # Content validation for text snippet
    content = r"""Thought → Action → Observation → Thought → ..."""
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

def test_draft_report_lines_68_68_10():
    """Test text snippet from examples/output/readme_report/draft_report.md lines 68-68."""
    # Content validation for text snippet
    content = r"""Goal → Plan → Execute Steps → Evaluate → Replan if needed"""
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

def test_draft_report_lines_74_76_11():
    """Test text snippet from examples/output/readme_report/draft_report.md lines 74-76."""
    # Content validation for text snippet
    content = r"""Agent A ↔ Agent B ↔ Agent C
   ↓         ↓         ↓
Shared State/Communication"""
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

def test_draft_report_lines_88_95_12():
    """Test Python snippet from examples/output/readme_report/draft_report.md lines 88-95."""
    # - Key Features: Chains, agents, tools, memory, callbacks
    
    code = r"""from langchain.agents import create_react_agent
from langchain.tools import Tool

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, calculator_tool],
    prompt=agent_prompt
)"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

def test_draft_report_lines_104_107_13():
    """Test Python snippet from examples/output/readme_report/draft_report.md lines 104-107."""
    # - Key Features: Conversable agents, group chat, code execution
    
    code = r"""from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user", code_execution_config=Ellipsis)"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

def test_draft_report_lines_116_120_14():
    """Test Python snippet from examples/output/readme_report/draft_report.md lines 116-120."""
    # - Key Features: Crews, roles, tasks, processes
    
    code = r"""from crewai import Agent, Task, Crew

researcher = Agent(role="Researcher", goal="Find information")
writer = Agent(role="Writer", goal="Create content")
crew = Crew(agents=[researcher, writer], tasks=[...])"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

def test_draft_report_lines_128_144_15():
    """Test Python snippet from examples/output/readme_report/draft_report.md lines 128-144."""
    # `
    
    code = r"""# Using LangChain
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

def test_draft_report_lines_150_166_16():
    """Test Python snippet from examples/output/readme_report/draft_report.md lines 150-166."""
    # `
    
    code = r"""# Using AutoGen
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
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

def test_draft_report_lines_172_206_17():
    """Test Python snippet from examples/output/readme_report/draft_report.md lines 172-206."""
    # `
    
    code = r"""# Using CrewAI
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
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

def test_final_report_lines_59_59_18():
    """Test text snippet from examples/output/readme_report/final_report.md lines 59-59."""
    # Content validation for text snippet
    content = r"""Thought → Action → Observation → Thought → ..."""
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

def test_final_report_lines_68_68_19():
    """Test text snippet from examples/output/readme_report/final_report.md lines 68-68."""
    # Content validation for text snippet
    content = r"""Goal → Plan → Execute Steps → Evaluate → Replan if needed"""
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"
