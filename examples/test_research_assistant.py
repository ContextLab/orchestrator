#!/usr/bin/env python3
"""Test the research assistant pipeline."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.models.model_registry import ModelRegistry
from src.orchestrator.core.model import Model


async def mock_search(task):
    """Mock search action."""
    query = task.parameters.get("query", "")
    print(f"[Web Search] Searching for: {query}")
    
    # Simulate search results
    return {
        "results": [
            "https://example.com/article1",
            "https://example.com/research2",
            "https://example.com/paper3"
        ],
        "status": "success"
    }


async def mock_extract(task):
    """Mock content extraction."""
    urls = task.parameters.get("urls", {}).get("results", [])
    print(f"[Content Extraction] Extracting from {len(urls)} URLs")
    
    # Simulate extracted content
    return {
        "text": """
        This is extracted content about the research topic.
        It contains various information and facts that need to be analyzed.
        Multiple perspectives are presented here for comprehensive analysis.
        """,
        "sources": urls,
        "status": "success"
    }


async def mock_analyze(task):
    """Mock analysis action."""
    content = task.parameters.get("content", "")
    depth = task.parameters.get("analysis_depth", "standard")
    print(f"[Analysis] Performing {depth} analysis")
    
    return {
        "key_points": [
            "Key finding 1: Important discovery",
            "Key finding 2: Significant pattern",
            "Key finding 3: Notable correlation"
        ],
        "outline": {
            "introduction": "Overview of findings",
            "methodology": "Research approach",
            "results": "Key discoveries",
            "conclusion": "Summary and implications"
        },
        "status": "success"
    }


async def mock_verify(task):
    """Mock fact checking."""
    claims = task.parameters.get("claims", [])
    print(f"[Fact Check] Verifying {len(claims)} claims")
    
    return {
        "verified": [
            {"claim": claim, "verified": True, "confidence": 0.95}
            for claim in claims
        ],
        "status": "success"
    }


async def mock_generate(task):
    """Mock report generation."""
    format_type = task.parameters.get("format", "markdown")
    print(f"[Report Generation] Generating {format_type} report")
    
    return {
        "document": """
# Research Report

## Executive Summary
This report presents findings from comprehensive research on the given topic.

## Introduction
The research was conducted using automated pipeline processing.

## Methodology
- Web search for relevant sources
- Content extraction and parsing
- Deep analysis of findings
- Fact verification
- Report synthesis

## Results
1. Key finding 1: Important discovery
2. Key finding 2: Significant pattern  
3. Key finding 3: Notable correlation

## Conclusion
The research pipeline successfully analyzed the topic and generated insights.

## References
- https://example.com/article1
- https://example.com/research2
- https://example.com/paper3
        """,
        "summary": "Research completed successfully with 3 key findings verified.",
        "status": "success"
    }


# Mock control system that routes actions to our mock functions
class MockResearchControlSystem:
    """Mock control system for research pipeline."""
    
    def __init__(self):
        self.action_handlers = {
            "search": mock_search,
            "extract": mock_extract,
            "analyze": mock_analyze,
            "verify": mock_verify,
            "generate": mock_generate
        }
    
    async def execute_action(self, action, task):
        """Execute a control action."""
        handler = self.action_handlers.get(action.name)
        if handler:
            return await handler(task)
        else:
            raise ValueError(f"Unknown action: {action.name}")
    
    async def shutdown(self):
        """Shutdown the control system."""
        pass


async def test_research_pipeline():
    """Test the research assistant pipeline."""
    print("Testing Research Assistant Pipeline")
    print("=" * 50)
    
    # Load the pipeline
    with open("pipelines/research_assistant.yaml", "r") as f:
        pipeline_yaml = f.read()
    
    # Initialize components
    orchestrator = Orchestrator()
    orchestrator.control_system = MockResearchControlSystem()
    
    # Set up a simple model for resolving <AUTO> tags
    class MockAutoModel(Model):
        def __init__(self):
            from src.orchestrator.core.model import ModelCapabilities
            capabilities = ModelCapabilities(
                supported_tasks=["reasoning", "generation"],
                context_window=4096,
                languages=["en"]
            )
            super().__init__(
                name="Mock Auto Model",
                provider="mock",
                capabilities=capabilities
            )
        
        async def generate(self, prompt, **kwargs):
            # Simple responses for AUTO resolution
            if "sources for research" in prompt:
                return "academic databases, scholarly articles, peer-reviewed journals"
            elif "number based on" in prompt:
                return "10"
            elif "extraction method" in prompt:
                return "structured_extraction"
            elif "key areas to analyze" in prompt:
                return "methodology, findings, implications, limitations"
            elif "format for" in prompt:
                return "markdown"
            return "default_value"
        
        async def validate_response(self, response, schema):
            return True
        
        def estimate_tokens(self, text):
            return len(text.split())
        
        async def generate_structured(self, prompt, schema, **kwargs):
            # For structured generation, just return a dict
            response = await self.generate(prompt, **kwargs)
            return {"value": response}
        
        def estimate_cost(self, input_tokens, output_tokens):
            # Mock cost estimation
            return 0.0
        
        async def health_check(self):
            # Always healthy
            return True
    
    # Register the mock model
    orchestrator.model_registry = ModelRegistry()
    mock_model = MockAutoModel()
    orchestrator.model_registry.register_model(mock_model)
    orchestrator.yaml_compiler.ambiguity_resolver.model = mock_model
    
    # Compile the pipeline with inputs
    pipeline = await orchestrator.yaml_compiler.compile(
        pipeline_yaml,
        context={
            "topic": "Artificial Intelligence in Healthcare",
            "depth": "comprehensive"
        }
    )
    
    print(f"\nCompiled pipeline: {pipeline.name}")
    print(f"Number of steps: {len(pipeline.tasks)}")
    print(f"Execution levels: {len(pipeline.get_execution_levels())}")
    
    # Execute the pipeline
    print("\nExecuting pipeline...")
    print("-" * 50)
    
    results = await orchestrator.execute_pipeline(pipeline)
    
    print("\n" + "-" * 50)
    print("Pipeline execution completed!")
    
    # Display results
    print("\nResults:")
    for task_id, result in results.items():
        task = pipeline.get_task(task_id)
        print(f"\n[{task.name}] Status: {result.get('status', 'unknown')}")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        elif task_id == 'report_generation':
            print(f"  Summary: {result.get('summary', 'N/A')}")
    
    # Check outputs
    outputs = pipeline.metadata.get("outputs", {})
    print("\nPipeline Outputs:")
    print(f"- Report available: {'report' in outputs}")
    print(f"- Summary available: {'summary' in outputs}")
    print(f"- Sources available: {'sources' in outputs}")
    
    return results


if __name__ == "__main__":
    # Run the test
    results = asyncio.run(test_research_pipeline())
    
    # Check if all tasks completed successfully
    all_success = all(
        r.get("status") == "success" 
        for r in results.values()
    )
    
    if all_success:
        print("\n✅ All pipeline steps completed successfully!")
    else:
        print("\n❌ Some pipeline steps failed")
        sys.exit(1)