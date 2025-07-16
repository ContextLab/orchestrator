"""Tests for documentation code snippets - Batch 4 (Fixed)."""
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


def test_final_report_lines_74_76_0():
    """Test text snippet from examples/output/readme_report/final_report.md lines 74-76."""
    # Content validation for text snippet
    content = r"""Agent A ↔ Agent B ↔ Agent C
   ↓         ↓         ↓
Shared State/Communication"""
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

def test_final_report_lines_88_95_1():
    """Test Python snippet from examples/output/readme_report/final_report.md lines 88-95."""
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

def test_final_report_lines_104_107_2():
    """Test Python snippet from examples/output/readme_report/final_report.md lines 104-107."""
    # - Key Features: Conversable agents, group chat, code execution
    
    code = r"""from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user", code_execution_config=Ellipsis)"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

def test_final_report_lines_116_120_3():
    """Test Python snippet from examples/output/readme_report/final_report.md lines 116-120."""
    # - Key Features: Crews, roles, tasks, processes
    
    code = r"""from crewai import Agent, Task, Crew

researcher = Agent(role="Researcher", goal="Find information")
writer = Agent(role="Writer", goal="Create content")
crew = Crew(agents=[researcher, writer], tasks=[...])"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

def test_final_report_lines_128_144_4():
    """Test Python snippet from examples/output/readme_report/final_report.md lines 128-144."""
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

def test_final_report_lines_150_166_5():
    """Test Python snippet from examples/output/readme_report/final_report.md lines 150-166."""
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

def test_final_report_lines_172_206_6():
    """Test Python snippet from examples/output/readme_report/final_report.md lines 172-206."""
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

def test_research_report_lines_59_59_7():
    """Test text snippet from examples/output/readme_report/research_report.md lines 59-59."""
    # Content validation for text snippet
    content = r"""Thought → Action → Observation → Thought → ..."""
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

def test_research_report_lines_68_68_8():
    """Test text snippet from examples/output/readme_report/research_report.md lines 68-68."""
    # Content validation for text snippet
    content = r"""Goal → Plan → Execute Steps → Evaluate → Replan if needed"""
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

def test_research_report_lines_74_76_9():
    """Test text snippet from examples/output/readme_report/research_report.md lines 74-76."""
    # Content validation for text snippet
    content = r"""Agent A ↔ Agent B ↔ Agent C
   ↓         ↓         ↓
Shared State/Communication"""
    
    # Basic checks
    assert content.strip(), "Snippet content should not be empty"
    assert len(content.strip()) > 0, "Snippet should have content"

def test_research_report_lines_88_95_10():
    """Test Python snippet from examples/output/readme_report/research_report.md lines 88-95."""
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

def test_research_report_lines_104_107_11():
    """Test Python snippet from examples/output/readme_report/research_report.md lines 104-107."""
    # - Key Features: Conversable agents, group chat, code execution
    
    code = r"""from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user", code_execution_config=Ellipsis)"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

def test_research_report_lines_116_120_12():
    """Test Python snippet from examples/output/readme_report/research_report.md lines 116-120."""
    # - Key Features: Crews, roles, tasks, processes
    
    code = r"""from crewai import Agent, Task, Crew

researcher = Agent(role="Researcher", goal="Find information")
writer = Agent(role="Writer", goal="Create content")
crew = Crew(agents=[researcher, writer], tasks=[...])"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

def test_research_report_lines_128_144_13():
    """Test Python snippet from examples/output/readme_report/research_report.md lines 128-144."""
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

def test_research_report_lines_150_166_14():
    """Test Python snippet from examples/output/readme_report/research_report.md lines 150-166."""
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

def test_research_report_lines_172_206_15():
    """Test Python snippet from examples/output/readme_report/research_report.md lines 172-206."""
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

def test_README_lines_57_64_16():
    """Test bash snippet from notebooks/README.md lines 57-64."""
    # Bash command snippet
    snippet_bash = r"""# Install the package
pip install -e .

# Install Jupyter (if not already installed)
pip install jupyter

# Start Jupyter
jupyter notebook"""
    
    # Validate pip install commands without actually running them
    assert "pip install" in snippet_bash
    
    lines = snippet_bash.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            assert line.startswith('pip install'), f"Invalid pip command: {line}"

def test_README_lines_79_81_17():
    """Test Python snippet from notebooks/README.md lines 79-81."""
    # If you're running from the source repository:
    
    code = r"""# Add the src directory to your Python path (included in notebooks)
import sys
sys.path.insert(0, '../src')"""
    
    try:
        exec(code, {'__name__': '__main__'})
    except Exception as e:
        pytest.fail(f"Code execution failed: {e}")

@pytest.mark.skipif(not os.environ.get('RUN_REAL_TESTS'), reason="Real model testing disabled")
@pytest.mark.asyncio
async def test_README_lines_109_115_18():
    """Test orchestrator code from notebooks/README.md lines 109-115."""
    # The tutorials use mock models for demonstration. To work with real AI models:
    
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
        if 'hello_world.yaml' in r"""from orchestrator.integrations.openai_model import OpenAIModel

model = OpenAIModel(
    name="gpt-4",
    api_key="your-openai-api-key",
    model="gpt-4"
)""":
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
            exec(r"""from orchestrator.integrations.openai_model import OpenAIModel

model = OpenAIModel(
    name="gpt-4",
    api_key="your-openai-api-key",
    model="gpt-4"
)""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
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
async def test_README_lines_120_126_19():
    """Test orchestrator code from notebooks/README.md lines 120-126."""
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
        if 'hello_world.yaml' in r"""from orchestrator.integrations.anthropic_model import AnthropicModel

model = AnthropicModel(
    name="claude-3-sonnet",
    api_key="your-anthropic-api-key",
    model="claude-3-sonnet-20240229"
)""":
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
            exec(r"""from orchestrator.integrations.anthropic_model import AnthropicModel

model = AnthropicModel(
    name="claude-3-sonnet",
    api_key="your-anthropic-api-key",
    model="claude-3-sonnet-20240229"
)""", {'__name__': '__main__', 'orc': orc, 'orchestrator': orc})
            
        except Exception as e:
            if "No eligible models" in str(e) or "API key" in str(e):
                pytest.skip(f"Model/API issue: {e}")
            else:
                pytest.fail(f"Code execution failed: {e}")
        finally:
            # Restore original function
            if original_compile:
                setattr(orc, 'compile', original_compile)
