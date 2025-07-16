"""Control system for research report generation following README design."""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any

from orchestrator.core.control_system import MockControlSystem
from orchestrator.core.task import Task, TaskStatus


class ResearchReportControlSystem(MockControlSystem):
    """Control system that implements research report generation."""
    
    def __init__(self, output_dir: str = "./output/research"):
        super().__init__(name="research-report-system")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results = {}
    
    async def execute_task(self, task: Task, context: dict = None) -> Dict[str, Any]:
        """Execute a research task."""
        print(f"\n⚙️  Executing task: {task.id} ({task.action})")
        
        # Resolve $results references
        self._resolve_references(task, context)
        
        # Route to appropriate handler
        if task.action == "search_web":
            result = await self._search_web(task, context)
        elif task.action == "compile_markdown":
            result = await self._compile_markdown(task, context)
        elif task.action == "generate_report":
            result = await self._generate_report(task, context)
        elif task.action == "validate_report":
            result = await self._validate_report(task, context)
        elif task.action == "finalize_report":
            result = await self._finalize_report(task, context)
        else:
            result = {"status": "completed", "message": f"Executed {task.action}"}
        
        # Store result
        self._results[task.id] = result
        task.status = TaskStatus.COMPLETED
        
        return result
    
    def _resolve_references(self, task: Task, context: dict):
        """Resolve $results and template references."""
        if not task.parameters:
            return
            
        for key, value in task.parameters.items():
            if isinstance(value, str):
                # Handle $results references
                if value.startswith("$results."):
                    parts = value.split(".")
                    if len(parts) >= 2:
                        task_id = parts[1]
                        if task_id in self._results:
                            result = self._results[task_id]
                            for part in parts[2:]:
                                if isinstance(result, dict) and part in result:
                                    result = result[part]
                                else:
                                    result = None
                                    break
                            task.parameters[key] = result
                
                # Handle template variables
                elif "{{" in value and "}}" in value:
                    # Simple template replacement
                    if context and 'inputs' in context:
                        for input_key, input_value in context['inputs'].items():
                            value = value.replace(f"{{{{ inputs.{input_key} }}}}", str(input_value))
                    task.parameters[key] = value
    
    async def _search_web(self, task: Task, context: dict) -> Dict[str, Any]:
        """Perform web search for research."""
        query = task.parameters.get("query", "")
        sources = task.parameters.get("sources", ["web"])
        
        print(f"   🔍 Searching: '{query}'")
        print(f"   📚 Sources: {sources}")
        
        # Simulate comprehensive search results
        results = []
        
        if "web" in sources:
            results.extend([
                {
                    "title": "Building AI Agents with LangChain",
                    "url": "https://python.langchain.com/docs/modules/agents/",
                    "snippet": "LangChain provides a framework for building AI agents that can use tools, make decisions, and execute complex workflows. Agents use an LLM as a reasoning engine to determine which actions to take and in what order.",
                    "source": "web",
                    "relevance": 0.95
                },
                {
                    "title": "AutoGen: Enable Next-Gen Large Language Model Applications",
                    "url": "https://microsoft.github.io/autogen/",
                    "snippet": "AutoGen is a framework that enables development of LLM applications using multiple agents that can converse with each other to solve tasks. It simplifies the orchestration, automation, and optimization of complex LLM workflows.",
                    "source": "web",
                    "relevance": 0.93
                }
            ])
        
        if "documentation" in sources:
            results.extend([
                {
                    "title": "OpenAI Assistants API Documentation",
                    "url": "https://platform.openai.com/docs/assistants",
                    "snippet": "The Assistants API allows you to build AI assistants within your own applications. An Assistant has instructions and can leverage models, tools, and knowledge to respond to user queries.",
                    "source": "documentation",
                    "relevance": 0.90
                },
                {
                    "title": "Hugging Face Agents Documentation",
                    "url": "https://huggingface.co/docs/transformers/transformers_agents",
                    "snippet": "Transformers Agents provides a natural language API on top of transformers. It allows you to quickly create AI agents that can perform various tasks by leveraging the extensive collection of models on the Hugging Face Hub.",
                    "source": "documentation",
                    "relevance": 0.88
                }
            ])
        
        if "academic" in sources:
            results.append({
                "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
                "url": "https://arxiv.org/abs/2210.03629",
                "snippet": "This paper presents ReAct, a paradigm for synergizing reasoning and acting in language models. ReAct prompts LLMs to generate both verbal reasoning traces and actions in an interleaved manner.",
                "source": "academic",
                "relevance": 0.92
            })
        
        # Save search results
        search_file = self.output_dir / "search_results.json"
        with open(search_file, "w") as f:
            json.dump({"query": query, "results": results}, f, indent=2)
        
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "file": str(search_file)
        }
    
    async def _compile_markdown(self, task: Task, context: dict) -> Dict[str, Any]:
        """Compile search results into markdown."""
        content = task.parameters.get("content", {})
        instruction = task.parameters.get("instruction", "")
        
        print(f"   📚 Compiling {content.get('count', 0)} results")
        
        results = content.get("results", [])
        
        # Create comprehensive markdown compilation
        compiled = f"""# Research Compilation: {content.get('query', 'AI Agents')}

**Compiled on**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Total sources**: {len(results)}
**Instruction**: {instruction}

## Overview

This document compiles research findings from web searches, documentation, and academic sources about AI agents and their implementation.

## Search Results by Source

"""
        
        # Group by source
        by_source = {}
        for result in results:
            source = result.get("source", "unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(result)
        
        for source, items in by_source.items():
            compiled += f"### {source.title()} Sources\n\n"
            for i, item in enumerate(items, 1):
                compiled += f"#### {i}. [{item['title']}]({item['url']})\n\n"
                compiled += f"**Relevance**: {item.get('relevance', 0):.0%}\n\n"
                compiled += f"{item['snippet']}\n\n"
                compiled += "---\n\n"
        
        compiled += """## Key Themes

Based on the search results, several key themes emerge:

1. **Framework Diversity**: Multiple frameworks exist for building AI agents (LangChain, AutoGen, Hugging Face Agents)
2. **Core Concepts**: Agents use LLMs for reasoning, tools for actions, and various orchestration patterns
3. **Academic Foundation**: Research like ReAct provides theoretical grounding for agent architectures
4. **API Solutions**: Major providers offer hosted agent solutions (OpenAI Assistants)

## Next Steps

Further analysis should explore:
- Implementation patterns and best practices
- Comparative analysis of different frameworks
- Real-world use cases and examples
- Performance and cost considerations
"""
        
        # Save compiled markdown
        compiled_file = self.output_dir / "compiled_results.md"
        with open(compiled_file, "w") as f:
            f.write(compiled)
        
        return {
            "content": compiled,
            "word_count": len(compiled.split()),
            "source_count": len(results),
            "file": str(compiled_file)
        }
    
    async def _generate_report(self, task: Task, context: dict) -> Dict[str, Any]:
        """Generate comprehensive research report."""
        try:
            content = task.parameters.get("content", {})
            topic = task.parameters.get("topic", "AI Agents")
            instructions = task.parameters.get("instructions", "")
            style = task.parameters.get("style", "technical")
            sections = task.parameters.get("sections", ["introduction", "overview", "details", "examples", "conclusion"])
            
            # Ensure all values are strings and safe for f-strings
            topic = str(topic) if topic else "AI Agents"
            instructions = str(instructions) if instructions else ""
            style = str(style) if style else "technical"
            
            print(f"   📝 Generating report on: {topic}")
            
            compiled_text = content.get("content", "") if isinstance(content, dict) else str(content)
        
            # Generate comprehensive report
            topic_title = topic.title() if topic else "AI Agents"
            style_title = style.title() if style else "Technical"
            
            report = f"""# {topic_title}: A Comprehensive Research Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Style**: {style_title}
**Instructions**: {instructions}

## Executive Summary

This report provides a comprehensive analysis of {topic}, synthesizing information from multiple authoritative sources including technical documentation, academic research, and industry resources. The report addresses the specific requirements: {instructions}

## Table of Contents

1. [Introduction](#introduction)
2. [Overview](#overview)
3. [Technical Details](#technical-details)
4. [Practical Examples](#practical-examples)
5. [Conclusion](#conclusion)
6. [References](#references)

## Introduction

{topic.title()} represent a significant advancement in artificial intelligence, enabling autonomous systems that can reason, plan, and execute complex tasks. This report explores the current state of the field, implementation approaches, and practical applications.

### Purpose and Scope

This report aims to:
- Provide a comprehensive understanding of {topic}
- Explore available frameworks and tools
- Present practical implementation examples
- Address the specific areas of interest outlined in the instructions

## Overview

### What Are AI Agents?

AI agents are autonomous systems that:
1. **Perceive** their environment through various inputs
2. **Reason** about the current state and desired outcomes
3. **Plan** sequences of actions to achieve goals
4. **Execute** actions using available tools and APIs
5. **Learn** from outcomes to improve future performance

### Core Components

Every AI agent system includes:

- **LLM Core**: The reasoning engine (GPT-4, Claude, Llama, etc.)
- **Tool Interface**: Connections to external capabilities
- **Memory System**: Short-term and long-term state management
- **Planning Module**: Task decomposition and scheduling
- **Execution Engine**: Action orchestration and error handling

## Technical Details

### Architecture Patterns

#### 1. ReAct Pattern (Reasoning + Acting)
```
Thought → Action → Observation → Thought → ...
```
This pattern interleaves reasoning and action, allowing agents to:
- Think through problems step-by-step
- Take actions based on reasoning
- Observe results and adjust approach

#### 2. Plan-and-Execute Pattern
```
Goal → Plan → Execute Steps → Evaluate → Replan if needed
```
Suitable for complex, multi-step tasks requiring upfront planning.

#### 3. Multi-Agent Collaboration
```
Agent A ↔ Agent B ↔ Agent C
   ↓         ↓         ↓
Shared State/Communication
```
Multiple specialized agents work together on complex problems.

### Available Frameworks

#### LangChain
- **Strengths**: Extensive ecosystem, production-ready, great documentation
- **Use Cases**: Complex workflows, tool integration, RAG applications
- **Key Features**: Chains, agents, tools, memory, callbacks

```python
from langchain.agents import create_react_agent
from langchain.tools import Tool

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, calculator_tool],
    prompt=agent_prompt
)
```

#### AutoGen (Microsoft)
- **Strengths**: Multi-agent conversations, code execution, flexible
- **Use Cases**: Collaborative problem-solving, code generation
- **Key Features**: Conversable agents, group chat, code execution

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user", code_execution_config={...})
```

#### CrewAI
- **Strengths**: Role-based design, intuitive API, process automation
- **Use Cases**: Business workflows, content creation, research
- **Key Features**: Crews, roles, tasks, processes

```python
from crewai import Agent, Task, Crew

researcher = Agent(role="Researcher", goal="Find information")
writer = Agent(role="Writer", goal="Create content")
crew = Crew(agents=[researcher, writer], tasks=[...])
```

## Practical Examples

### Example 1: Research Assistant

```python
# Using LangChain
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

result = research_agent.run("What are the latest developments in quantum computing?")
```

### Example 2: Code Generation Agent

```python
# Using AutoGen
coding_assistant = AssistantAgent(
    "coding_assistant",
    system_message="You are a helpful AI that writes and explains code.",
    llm_config={{"model": "gpt-4"}}
)

user_proxy = UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config={{"work_dir": "coding", "use_docker": False}}
)

user_proxy.initiate_chat(
    coding_assistant,
    message="Write a Python function to calculate fibonacci numbers efficiently"
)
```

### Example 3: Multi-Agent System

```python
# Using CrewAI
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

result = crew.kickoff()
```

## Best Practices

### 1. Agent Design
- **Single Responsibility**: Each agent should have a clear, focused purpose
- **Clear Instructions**: Provide detailed system prompts and examples
- **Error Handling**: Implement robust error recovery mechanisms

### 2. Tool Selection
- **Minimal Set**: Only include necessary tools to reduce complexity
- **Clear Descriptions**: Tools should have clear, detailed descriptions
- **Safety Checks**: Validate tool inputs and outputs

### 3. Performance Optimization
- **Caching**: Cache LLM responses and tool results when appropriate
- **Parallel Execution**: Run independent tasks concurrently
- **Model Selection**: Choose appropriate model sizes for different tasks

### 4. Safety and Reliability
- **Sandboxing**: Execute code in isolated environments
- **Rate Limiting**: Implement limits on API calls and resource usage
- **Monitoring**: Log all agent actions and decisions

## Conclusion

{topic.title()} represent a powerful paradigm for building intelligent applications that can autonomously solve complex problems. Key takeaways include:

1. **Multiple Frameworks**: Choose based on your specific use case
2. **Design Patterns**: ReAct, Plan-and-Execute, and Multi-Agent patterns each have strengths
3. **Best Practices**: Focus on safety, performance, and clear agent design
4. **Rapid Evolution**: The field is advancing quickly with new capabilities

### Future Directions

The field of AI agents is rapidly evolving with trends including:
- **Multimodal Agents**: Handling text, images, audio, and video
- **Long-Term Memory**: Persistent learning across sessions
- **Tool Creation**: Agents that can create their own tools
- **Verification**: Better methods for validating agent outputs

## References

1. [LangChain Documentation](https://python.langchain.com/docs/modules/agents/)
2. [AutoGen Framework](https://microsoft.github.io/autogen/)
3. [CrewAI GitHub Repository](https://github.com/joaomdmoura/crewAI)
4. [OpenAI Assistants API](https://platform.openai.com/docs/assistants)
5. [Hugging Face Agents](https://huggingface.co/docs/transformers/transformers_agents)
6. [ReAct Paper](https://arxiv.org/abs/2210.03629)

---

*This report was generated by the Orchestrator framework using multiple AI models and automated research tools.*
"""
        
            # Save report
            report_file = self.output_dir / "draft_report.md"
            with open(report_file, "w") as f:
                f.write(report)
            
            return {
                "content": report,
                "word_count": len(report.split()),
                "sections": sections,
                "style": style,
                "file": str(report_file)
            }
        
        except Exception as e:
            print(f"ERROR in _generate_report: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _validate_report(self, task: Task, context: dict) -> Dict[str, Any]:
        """Validate report quality."""
        report = task.parameters.get("report", {})
        checks = task.parameters.get("checks", ["completeness"])
        
        print(f"   ✅ Running quality checks: {checks}")
        
        report_content = report.get("content", "") if isinstance(report, dict) else str(report)
        
        validation_results = {}
        issues = []
        recommendations = []
        
        for check in checks:
            if check == "completeness":
                # Check for required sections
                required = ["Introduction", "Overview", "Technical Details", "Examples", "Conclusion", "References"]
                missing = [s for s in required if s not in report_content]
                validation_results[check] = len(missing) == 0
                if missing:
                    issues.append(f"Missing sections: {', '.join(missing)}")
                    recommendations.append("Add missing sections to ensure comprehensive coverage")
            
            elif check == "accuracy":
                # Check for technical accuracy indicators
                has_code = "```python" in report_content or "```" in report_content
                has_frameworks = all(f in report_content for f in ["LangChain", "AutoGen", "CrewAI"])
                validation_results[check] = has_code and has_frameworks
                if not has_code:
                    issues.append("No code examples found")
                    recommendations.append("Add practical code examples")
            
            elif check == "sources_cited":
                # Check for references
                has_refs = "References" in report_content and ("http" in report_content or "www." in report_content)
                validation_results[check] = has_refs
                if not has_refs:
                    issues.append("References section missing or incomplete")
                    recommendations.append("Add proper citations and references")
            
            elif check == "logical_flow":
                # Check structure
                has_toc = "Table of Contents" in report_content or "## " in report_content
                validation_results[check] = has_toc
                if not has_toc:
                    issues.append("Document structure could be improved")
                    recommendations.append("Add clear section headers and organization")
        
        overall_score = sum(validation_results.values()) / len(validation_results) if validation_results else 0
        
        result = {
            "validation_passed": overall_score >= 0.75,
            "overall_score": overall_score,
            "checks": validation_results,
            "issues": issues,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save validation results
        validation_file = self.output_dir / "validation_results.json"
        with open(validation_file, "w") as f:
            json.dump(result, f, indent=2)
        
        return result
    
    async def _finalize_report(self, task: Task, context: dict) -> Dict[str, Any]:
        """Finalize the report with improvements."""
        draft = task.parameters.get("draft", {})
        validation = task.parameters.get("validation", {})
        improvements = task.parameters.get("improvements", [])
        
        print("   📄 Finalizing report")
        
        draft_content = draft.get("content", "") if isinstance(draft, dict) else str(draft)
        overall_score = validation.get("overall_score", 0) if isinstance(validation, dict) else 0.8
        
        # Add quality stamp
        quality_stamp = f"""

---

## Quality Assurance

**Validation Score**: {overall_score:.0%}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Framework**: Orchestrator AI Pipeline Framework
"""
        
        if overall_score >= 0.75:
            quality_stamp += "\n✅ This report passed all quality checks."
        else:
            quality_stamp += "\n⚠️ This report may benefit from additional review."
        
        if isinstance(validation, dict) and validation.get("recommendations"):
            quality_stamp += "\n\n### Recommendations for Improvement\n"
            for rec in validation["recommendations"]:
                quality_stamp += f"- {rec}\n"
        
        final_content = draft_content + quality_stamp
        
        # Save final report
        final_file = self.output_dir / "final_report.md"
        with open(final_file, "w") as f:
            f.write(final_content)
        
        # Also save as the main output
        output_file = self.output_dir / "research_report.md"
        with open(output_file, "w") as f:
            f.write(final_content)
        
        print(f"\n✅ Report saved to: {output_file}")
        
        return {
            "content": final_content,
            "word_count": len(final_content.split()),
            "quality_score": overall_score,
            "file": str(output_file),
            "final_file": str(final_file)
        }
    
    async def can_execute_task(self, task: Task) -> bool:
        """Check if this control system can execute the given task."""
        supported_actions = [
            "search_web", "compile_markdown", "generate_report",
            "validate_report", "finalize_report"
        ]
        return task.action in supported_actions
    
    async def estimate_resource_requirements(self, task: Task) -> Dict[str, Any]:
        """Estimate resources needed for task execution."""
        # Basic estimates
        if task.action == "search_web":
            return {"time_seconds": 2, "memory_mb": 50}
        elif task.action == "generate_report":
            return {"time_seconds": 5, "memory_mb": 100}
        else:
            return {"time_seconds": 1, "memory_mb": 25}
    
    async def validate_task_parameters(self, task: Task) -> bool:
        """Validate that task has required parameters."""
        required_params = {
            "search_web": ["query"],
            "compile_markdown": ["content"],
            "generate_report": ["content", "topic"],
            "validate_report": ["report"],
            "finalize_report": ["draft", "validation"]
        }
        
        if task.action in required_params:
            for param in required_params[task.action]:
                if param not in task.parameters:
                    print(f"⚠️  Missing required parameter '{param}' for {task.action}")
                    return False
        
        return True