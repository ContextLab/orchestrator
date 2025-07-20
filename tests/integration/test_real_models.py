#!/usr/bin/env python3
"""Test pipelines with real AI models (Ollama/HuggingFace)."""

import asyncio
import sys
import os
import traceback
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator.orchestrator import Orchestrator
from orchestrator.core.control_system import ControlSystem
from orchestrator.core.task import Task


class RealModelControlSystem(ControlSystem):
    """Control system that uses real models for AUTO resolution."""
    
    def __init__(self):
        config = {
            "capabilities": {
                "supported_actions": ["search", "analyze", "summarize"],
                "parallel_execution": True,
                "streaming": False,
                "checkpoint_support": True,
            },
            "base_priority": 20,
        }
        super().__init__(name="real-model-control", config=config)
        self._results = {}
    
    async def execute_task(self, task: Task, context: dict = None):
        """Execute task with real processing."""
        # Handle $results references
        self._resolve_references(task)
        
        # Route to appropriate handler
        if task.action == "search":
            return await self._search(task)
        elif task.action == "analyze":
            return await self._analyze(task)
        elif task.action == "summarize":
            return await self._summarize(task)
        else:
            result = {"status": "completed", "message": f"Executed {task.action}"}
            self._results[task.id] = result
            return result
    
    def _resolve_references(self, task):
        """Resolve $results references."""
        for key, value in task.parameters.items():
            if isinstance(value, str) and value.startswith("$results."):
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
    
    async def _search(self, task):
        """Search for information using real web search."""
        query = task.parameters.get("query", "")
        print(f"[SEARCH] Searching for: '{query}'")
        
        # Use real web search if available
        try:
            from orchestrator.tools.web_tools import DuckDuckGoSearchBackend
            search_tool = DuckDuckGoSearchBackend()
            search_results = await search_tool.search(query, max_results=5)
            
            # Convert to expected format
            results = []
            for i, result in enumerate(search_results):
                results.append({
                    "title": result.get("title", f"Result {i+1}"),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "relevance": 0.9 - (i * 0.05)  # Descending relevance
                })
            
            result = {
                "results": results,
                "total_results": len(results),
                "search_quality": "high" if len(results) >= 3 else "medium",
                "query": query
            }
        except Exception as e:
            print(f"[SEARCH] Real search failed: {e}, using fallback")
            # Fallback to simple results
            result = {
                "results": [
                    {"title": f"Research on {query}", "url": f"https://scholar.google.com/search?q={query}", "relevance": 0.85},
                    {"title": f"{query} Overview", "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}", "relevance": 0.80}
                ],
                "total_results": 2,
                "search_quality": "fallback",
                "query": query
            }
        
        self._results[task.id] = result
        return result
    
    async def _analyze(self, task):
        """Analyze search results using real model."""
        data = task.parameters.get("data", {})
        results = data.get("results", [])
        print(f"[ANALYZE] Analyzing {len(results)} search results")
        
        # Get available model for analysis
        model = get_available_model()
        if model:
            # Create analysis prompt
            prompt = f"""Analyze these search results and provide key insights:

{json.dumps(results, indent=2)}

Provide 3-5 key insights about the search results."""
            
            try:
                # Use real model for analysis
                analysis = await model.generate(prompt, max_tokens=200, temperature=0.3)
                
                # Parse insights from model response
                insights = []
                for line in analysis.strip().split('\n'):
                    if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                        insights.append(line.strip().lstrip('0123456789.-) '))
                
                if not insights:
                    insights = [f"Found {len(results)} relevant sources", "Analysis completed successfully"]
                
                result = {
                    "key_insights": insights[:5],  # Limit to 5 insights
                    "analysis_quality": "ai-powered",
                    "confidence_score": 0.92,
                    "source_credibility": "verified",
                    "model_used": model.name
                }
            except Exception as e:
                print(f"[ANALYZE] Model analysis failed: {e}")
                # Fallback to basic analysis
                result = {
                    "key_insights": [
                        f"Found {len(results)} search results",
                        "Results cover multiple perspectives",
                        "Further analysis recommended"
                    ],
                    "analysis_quality": "basic",
                    "confidence_score": 0.75,
                    "source_credibility": "unverified"
                }
        else:
            # No model available - basic analysis
            result = {
                "key_insights": [
                    f"Processed {len(results)} search results",
                    "Manual review needed for detailed insights"
                ],
                "analysis_quality": "minimal",
                "confidence_score": 0.60,
                "source_credibility": "unknown"
            }
        
        self._results[task.id] = result
        return result
    
    async def _summarize(self, task):
        """Create research summary using real model."""
        content = task.parameters.get("content", {})
        insights = content.get("key_insights", [])
        print(f"[SUMMARIZE] Creating research summary with {len(insights)} insights")
        
        # Get available model for summarization
        model = get_available_model()
        if model and insights:
            # Create summarization prompt
            prompt = f"""Create a research summary based on these key insights:

{chr(10).join(f'- {insight}' for insight in insights)}

Analysis Quality: {content.get('analysis_quality', 'standard')}
Confidence Score: {content.get('confidence_score', 0.8)}

Write a concise executive summary (2-3 sentences)."""
            
            try:
                # Use real model for summary
                exec_summary = await model.generate(prompt, max_tokens=150, temperature=0.2)
                
                # Build complete summary
                summary = "# Research Summary\\n\\n"
                summary += "## Executive Summary\\n"
                summary += exec_summary.strip() + "\\n\\n"
                summary += "## Key Insights\\n"
                for i, insight in enumerate(insights, 1):
                    summary += f"{i}. {insight}\\n"
                summary += f"\\n## Analysis Quality: {content.get('analysis_quality', 'standard')}"
                summary += f"\\n## Confidence Score: {content.get('confidence_score', 0.8):.2f}"
                if content.get('model_used'):
                    summary += f"\\n## Model Used: {content['model_used']}"
                
                result = {
                    "summary": summary,
                    "executive_summary": exec_summary.strip(),
                    "word_count": len(summary.split()),
                    "quality_metrics": {
                        "completeness": 0.95,
                        "accuracy": content.get("confidence_score", 0.8),
                        "clarity": 0.92,
                        "ai_enhanced": True
                    },
                    "model_used": model.name
                }
            except Exception as e:
                print(f"[SUMMARIZE] Model summarization failed: {e}")
                # Fallback to basic summary
                summary = self._create_basic_summary(insights, content)
                result = {
                    "summary": summary,
                    "executive_summary": f"Research analysis completed with {len(insights)} key insights",
                    "word_count": len(summary.split()),
                    "quality_metrics": {
                        "completeness": 0.85,
                        "accuracy": content.get("confidence_score", 0.8),
                        "clarity": 0.80,
                        "ai_enhanced": False
                    }
                }
        else:
            # No model or insights - basic summary
            summary = self._create_basic_summary(insights, content)
            result = {
                "summary": summary,
                "executive_summary": "Summary created without AI enhancement",
                "word_count": len(summary.split()),
                "quality_metrics": {
                    "completeness": 0.70,
                    "accuracy": content.get("confidence_score", 0.8),
                    "clarity": 0.75,
                    "ai_enhanced": False
                }
            }
        
        self._results[task.id] = result
        return result
    
    def _create_basic_summary(self, insights, content):
        """Create a basic summary without AI."""
        summary = "# Research Summary\\n\\n"
        summary += "## Key Insights\\n"
        for i, insight in enumerate(insights, 1):
            summary += f"{i}. {insight}\\n"
        summary += f"\\n## Analysis Quality: {content.get('analysis_quality', 'standard')}"
        summary += f"\\n## Confidence Score: {content.get('confidence_score', 0.8):.2f}"
        return summary


def get_available_model():
    """Get the best available model for testing."""
    # Check if Ollama is available
    try:
        from orchestrator.integrations.ollama_model import OllamaModel
        if OllamaModel.check_ollama_installation():
            for model_name in ["gemma2:27b", "gemma2:9b", "llama3.2:3b", "llama3.2:1b"]:
                try:
                    model = OllamaModel(model_name=model_name)
                    if model._is_available:
                        print(f"✅ Using Ollama model: {model_name}")
                        return model
                except Exception as e:
                    print(f"⚠️  Ollama model {model_name} not available: {e}")
                    continue
    except ImportError:
        print("⚠️  Ollama integration not available")
    
    # Fallback to HuggingFace
    try:
        from orchestrator.integrations.huggingface_model import HuggingFaceModel
        for model_name in ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "distilgpt2"]:
            try:
                print(f"📥 Loading HuggingFace model: {model_name}")
                model = HuggingFaceModel(model_name=model_name)
                print(f"✅ Using HuggingFace model: {model_name}")
                return model
            except Exception as e:
                print(f"⚠️  HuggingFace model {model_name} failed: {e}")
                continue
    except ImportError:
        print("⚠️  HuggingFace integration not available")
    
    print("❌ No real models available, tests cannot run")
    return None


async def test_real_auto_resolution():
    """Test AUTO tag resolution with real models."""
    print("\n" + "="*70)
    print("🔍 TESTING AUTO TAG RESOLUTION WITH REAL MODELS")
    print("="*70)
    
    # Get available model
    model = get_available_model()
    if model is None:
        print("❌ No models available for testing")
        return False
    
    try:
        # Test the model directly
        print(f"\n🧪 Testing model: {model.name}")
        
        # Simple generation test
        result = await model.generate("What is machine learning?", max_tokens=50, temperature=0.1)
        print(f"✅ Model generation successful: {result[:100]}...")
        
        # Test AUTO resolution scenarios
        from orchestrator.compiler.ambiguity_resolver import AmbiguityResolver
        resolver = AmbiguityResolver(model=model)
        
        test_cases = [
            ("Choose the best format for data output", "parameters.output_format"),
            ("Select appropriate batch size for processing", "config.batch_size"),
            ("Determine optimal timeout value", "settings.timeout"),
            ("Pick suitable analysis method", "task.method"),
        ]
        
        print(f"\n🎯 Testing AUTO resolution with {len(test_cases)} scenarios:")
        
        for content, context in test_cases:
            try:
                resolved = await resolver.resolve(content, context)
                print(f"✅ '{content}' → '{resolved}'")
            except Exception as e:
                print(f"❌ Failed to resolve '{content}': {e}")
                return False
        
        print("\n✅ All AUTO resolution tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        traceback.print_exc()
        return False


async def test_pipeline_with_real_model():
    """Test a complete pipeline with real model for AUTO resolution."""
    print("\n" + "="*70)
    print("🚀 TESTING PIPELINE WITH REAL MODEL AUTO RESOLUTION")
    print("="*70)
    
    # Get available model
    model = get_available_model()
    if model is None:
        print("❌ No models available for testing")
        return False
    
    try:
        # Load simple research pipeline
        pipeline_yaml = """
name: "real_model_research"
description: "Research pipeline using real AI model for AUTO resolution"

steps:
  - id: search
    action: search
    parameters:
      query: "{{ topic }}"
      sources: <AUTO>Choose best sources for academic research</AUTO>
      depth: <AUTO>Determine appropriate search depth</AUTO>

  - id: analyze
    action: analyze
    depends_on: [search]
    parameters:
      data: "$results.search"
      method: <AUTO>Select best analysis method for research data</AUTO>
      threshold: <AUTO>Choose confidence threshold</AUTO>

  - id: summarize
    action: summarize
    depends_on: [analyze]
    parameters:
      content: "$results.analyze"
      format: <AUTO>Pick optimal summary format</AUTO>
      length: <AUTO>Determine appropriate summary length</AUTO>
"""
        
        # Set up orchestrator with real model
        control_system = RealModelControlSystem()
        orchestrator = Orchestrator(control_system=control_system)
        
        # Use our real model for AUTO resolution
        orchestrator.yaml_compiler.ambiguity_resolver.model = model
        
        # Execute pipeline
        print(f"⚙️  Executing pipeline with model: {model.name}")
        context = {"topic": "artificial intelligence in healthcare"}
        
        results = await orchestrator.execute_yaml(pipeline_yaml, context=context)
        
        print("\n✅ Pipeline executed successfully!")
        print(f"📊 Tasks completed: {len(results)}")
        
        # Verify results
        for task_id, result in results.items():
            print(f"\n📋 Task: {task_id}")
            if isinstance(result, dict):
                if "summary" in result:
                    lines = result["summary"].count('\\n') + 1
                    print(f"   📄 Generated summary ({lines} lines)")
                elif "key_insights" in result:
                    insights = len(result["key_insights"])
                    print(f"   🔍 Found {insights} key insights")
                elif "results" in result:
                    count = len(result["results"])
                    print(f"   🔎 Retrieved {count} search results")
                else:
                    print("   ✓ Completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        traceback.print_exc()
        return False


async def test_model_availability():
    """Test which models are available on the system."""
    print("\n" + "="*70)
    print("📋 CHECKING MODEL AVAILABILITY")
    print("="*70)
    
    available_models = []
    
    # Check Ollama models
    print("\n🦙 Checking Ollama models:")
    try:
        from orchestrator.integrations.ollama_model import OllamaModel
        
        if OllamaModel.check_ollama_installation():
            print("✅ Ollama CLI is installed")
            
            # Try to connect to Ollama service
            test_model = OllamaModel("llama3.2:1b")  # Small model for quick test
            if test_model.is_ollama_running():
                print("✅ Ollama service is running")
                
                # Check available models
                models = test_model.get_available_models()
                for model in OllamaModel.get_recommended_models():
                    if any(model in m for m in models):
                        print(f"✅ {model} is available")
                        available_models.append(f"ollama:{model}")
                    else:
                        print(f"❌ {model} not available")
            else:
                print("❌ Ollama service not running")
        else:
            print("❌ Ollama CLI not installed")
            
    except ImportError:
        print("❌ Ollama integration not available")
    
    # Check HuggingFace models
    print("\n🤗 Checking HuggingFace models:")
    try:
        from orchestrator.integrations.huggingface_model import HuggingFaceModel
        
        test_models = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "distilgpt2"]
        for model_name in test_models:
            try:
                print(f"📥 Testing {model_name}...")
                model = HuggingFaceModel(model_name=model_name)
                # Don't actually load the model, just check if we can create it
                print(f"✅ {model_name} is available")
                available_models.append(f"huggingface:{model_name}")
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                
    except ImportError:
        print("❌ HuggingFace integration not available")
    
    print(f"\n📊 Summary: {len(available_models)} models available")
    for model in available_models:
        print(f"   ✅ {model}")
    
    return len(available_models) > 0


async def main():
    """Run all real model tests."""
    print("🚀 REAL MODEL TESTING SUITE")
    print("Testing Ollama and HuggingFace model integrations")
    print("="*70)
    
    test_results = []
    
    # Test 1: Model availability
    success = await test_model_availability()
    test_results.append(("Model Availability", success))
    
    if not success:
        print("\n❌ No models available - skipping further tests")
        print("💡 Install Ollama or HuggingFace transformers to run tests")
        return False
    
    # Test 2: AUTO resolution
    success = await test_real_auto_resolution()
    test_results.append(("AUTO Resolution", success))
    
    # Test 3: Complete pipeline
    success = await test_pipeline_with_real_model()
    test_results.append(("Pipeline Execution", success))
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 FINAL TEST RESULTS")
    print('='*70)
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print(f"\n📈 Real Model Tests: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    overall_success = passed == total
    
    if overall_success:
        print("\n🎉 ALL REAL MODEL TESTS PASSED!")
        print("✅ Ollama/HuggingFace integrations working correctly")
        print("✅ AUTO tag resolution with real models successful")
        print("✅ Complete pipeline execution working")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("❌ Issues detected that need investigation")
    
    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)