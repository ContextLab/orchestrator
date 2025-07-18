#!/usr/bin/env python3
"""
Run ALL example pipelines comprehensively and generate outputs.
"""

import asyncio
import os
from pathlib import Path
from datetime import datetime
import json
import traceback

from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.integrations.anthropic_model import AnthropicModel
from orchestrator.integrations.google_model import GoogleModel
from orchestrator.integrations.ollama_model import OllamaModel
from orchestrator.integrations.huggingface_model import HuggingFaceModel


def setup_all_models():
    """Set up model registry with all available models."""
    registry = ModelRegistry()
    models_registered = []
    
    # OpenAI Models
    if os.getenv("OPENAI_API_KEY"):
        try:
            registry.register_model(OpenAIModel(model_name="gpt-4o-mini"))
            registry.register_model(OpenAIModel(model_name="gpt-4-turbo-preview"))
            registry.register_model(OpenAIModel(model_name="gpt-4"))
            models_registered.append("OpenAI")
            print("✓ OpenAI models registered")
        except Exception as e:
            print(f"⚠️  OpenAI registration failed: {e}")
    
    # Anthropic Models
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            registry.register_model(AnthropicModel(model_name="claude-3-haiku-20240307"))
            registry.register_model(AnthropicModel(model_name="claude-3-sonnet-20240229"))
            models_registered.append("Anthropic")
            print("✓ Anthropic models registered")
        except Exception as e:
            print(f"⚠️  Anthropic registration failed: {e}")
    
    # Google Models
    if os.getenv("GOOGLE_API_KEY"):
        try:
            registry.register_model(GoogleModel(model_name="gemini-1.5-flash"))
            models_registered.append("Google")
            print("✓ Google models registered")
        except Exception as e:
            print(f"⚠️  Google registration failed: {e}")
    
    # HuggingFace Models
    if os.getenv("HUGGINGFACE_API_KEY"):
        try:
            registry.register_model(HuggingFaceModel(model_name="mistralai/Mistral-7B-Instruct-v0.2"))
            models_registered.append("HuggingFace")
            print("✓ HuggingFace models registered")
        except Exception as e:
            print(f"⚠️  HuggingFace registration failed: {e}")
    
    # Ollama Models (local)
    try:
        registry.register_model(OllamaModel(model_name="llama2"))
        models_registered.append("Ollama")
        print("✓ Ollama models registered")
    except Exception as e:
        print(f"⚠️  Ollama registration failed (is it running?): {e}")
    
    return registry, models_registered


async def run_pipeline(name, yaml_path, inputs, registry, save_outputs=True):
    """Run a single pipeline and handle all outputs."""
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"File: {yaml_path}")
    print(f"{'='*80}")
    
    result_data = {
        "pipeline": name,
        "file": str(yaml_path),
        "status": "pending",
        "start_time": datetime.now().isoformat(),
        "inputs": inputs,
        "outputs": {},
        "error": None
    }
    
    try:
        # Read YAML
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
        
        # Setup
        control_system = ModelBasedControlSystem(registry)
        compiler = YAMLCompiler()
        
        # Compile and run
        print(f"Inputs: {json.dumps(inputs, indent=2)}")
        pipeline = await compiler.compile(yaml_content, inputs)
        print(f"✓ Pipeline compiled with {len(pipeline.tasks)} tasks")
        
        start = datetime.now()
        results = await control_system.execute_pipeline(pipeline)
        duration = (datetime.now() - start).total_seconds()
        
        result_data["status"] = "success"
        result_data["duration"] = duration
        result_data["outputs"] = {k: str(v)[:500] + "..." if len(str(v)) > 500 else str(v) 
                                 for k, v in results.items()}
        
        print(f"✅ Completed in {duration:.1f} seconds")
        
        # Save outputs if requested
        if save_outputs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"examples/output/run_{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save raw results
            results_file = output_dir / f"{name.replace(' ', '_').lower()}_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "pipeline": name,
                    "inputs": inputs,
                    "results": {k: str(v) for k, v in results.items()},
                    "duration": duration,
                    "timestamp": timestamp
                }, f, indent=2)
            
            print(f"📄 Results saved to: {results_file}")
            
            # Check for file outputs in results
            for task_name, task_result in results.items():
                if isinstance(task_result, str) and 'examples/output/' in task_result:
                    print(f"📝 Pipeline saved output to: {task_result}")
        
    except Exception as e:
        result_data["status"] = "error"
        result_data["error"] = str(e)
        result_data["traceback"] = traceback.format_exc()
        print(f"❌ Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
    
    result_data["end_time"] = datetime.now().isoformat()
    return result_data


async def run_all_pipelines():
    """Run all example pipelines with appropriate test data."""
    print("="*80)
    print("COMPREHENSIVE PIPELINE EXECUTION")
    print("Running ALL example pipelines with test data")
    print("="*80)
    
    # Ensure output directory exists
    Path("examples/output").mkdir(parents=True, exist_ok=True)
    
    # Set up models
    registry, available_models = setup_all_models()
    print(f"\nAvailable model providers: {', '.join(available_models)}")
    
    # Define all pipelines with test inputs
    all_pipelines = [
        # Research and Analysis
        {
            "name": "Research Assistant",
            "file": "research_assistant.yaml",
            "inputs": {
                "query": "The impact of quantum computing on cryptography",
                "context": "Focus on current threats and future quantum-resistant algorithms",
                "max_sources": 5,
                "quality_threshold": 0.7
            }
        },
        
        # Content Creation
        {
            "name": "Content Creation Pipeline",
            "file": "content_creation_pipeline.yaml",
            "inputs": {
                "topic": "10 Best Practices for RESTful API Design",
                "formats": ["blog", "social"],
                "audience": "backend developers",
                "brand_voice": "technical but approachable",
                "target_length": 1500,
                "auto_publish": False
            }
        },
        
        # Creative Writing
        {
            "name": "Creative Writing Assistant",
            "file": "creative_writing_assistant.yaml",
            "inputs": {
                "genre": "science fiction",
                "length": "flash",
                "writing_style": "philosophical",
                "target_audience": "adults",
                "initial_premise": "An AI discovers it can dream",
                "include_worldbuilding": True,
                "chapter_count": 1
            }
        },
        
        # Financial Analysis
        {
            "name": "Financial Analysis Bot",
            "file": "financial_analysis_bot.yaml",
            "inputs": {
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "analysis_type": "technical",
                "time_period": "6M",
                "include_fundamentals": False,
                "backtest_period": "1Y"
            }
        },
        
        # Data Processing
        {
            "name": "Data Processing Workflow",
            "file": "data_processing_workflow.yaml",
            "inputs": {
                "source": "customer_feedback_data.csv",
                "output_path": "processed/customer_insights",
                "output_format": "parquet",
                "chunk_size": 1000,
                "parallel_workers": 4,
                "validation_rules": [
                    {"field": "customer_id", "type": "required"},
                    {"field": "rating", "type": "numeric", "min": 1, "max": 5},
                    {"field": "feedback", "type": "text", "min_length": 10}
                ],
                "transformations": [
                    {"type": "normalize_text", "field": "feedback"},
                    {"type": "extract_sentiment", "field": "feedback"},
                    {"type": "categorize", "field": "feedback"}
                ]
            }
        },
        
        # Chatbot Demo
        {
            "name": "Interactive Chat Bot Demo",
            "file": "interactive_chat_bot_demo.yaml",
            "inputs": {
                "conversation_topic": "The future of AI in healthcare",
                "num_exchanges": 5,
                "user_persona": "medical-professional",
                "bot_persona": "ai-researcher"
            }
        },
        
        # Code Analysis
        {
            "name": "Code Analysis Suite",
            "file": "code_analysis_suite.yaml",
            "inputs": {
                "repository_path": "./src",
                "analysis_types": ["quality", "security", "complexity"],
                "languages": ["python"],
                "output_format": "markdown",
                "fix_issues": False,
                "severity_threshold": "medium"
            }
        },
        
        # Customer Support
        {
            "name": "Customer Support Automation",
            "file": "customer_support_automation.yaml",
            "inputs": {
                "ticket_id": "TICKET-12345",
                "customer_message": "I'm having trouble logging into my account. It says my password is incorrect but I'm sure it's right.",
                "customer_history": {
                    "previous_tickets": 2,
                    "account_type": "premium",
                    "customer_since": "2021-01-15"
                },
                "escalation_threshold": 0.8,
                "auto_respond": True
            }
        },
        
        # Automated Testing
        {
            "name": "Automated Testing System",
            "file": "automated_testing_system.yaml",
            "inputs": {
                "source_dir": "./src/orchestrator",
                "test_dir": "./tests",
                "coverage_target": 80.0,
                "test_types": ["unit", "integration"],
                "test_framework": "pytest",
                "include_edge_cases": True,
                "include_performance": False
            }
        },
        
        # Document Intelligence
        {
            "name": "Document Intelligence",
            "file": "document_intelligence.yaml",
            "inputs": {
                "document_path": "sample_contract.pdf",
                "analysis_types": ["summary", "key_terms", "risks"],
                "output_format": "structured_json",
                "extract_entities": True,
                "languages": ["en"]
            }
        },
        
        # Multi-Agent Collaboration
        {
            "name": "Multi Agent Collaboration",
            "file": "multi_agent_collaboration.yaml",
            "inputs": {
                "task": "Design a sustainable smart city transportation system",
                "num_agents": 4,
                "agent_roles": ["urban_planner", "environmental_scientist", "tech_architect", "economist"],
                "collaboration_rounds": 3,
                "consensus_threshold": 0.75
            }
        },
        
        # Scalable Customer Service
        {
            "name": "Scalable Customer Service Agent",
            "file": "scalable_customer_service_agent.yaml",
            "inputs": {
                "customer_id": "CUST-789456",
                "query": "What's the status of my order #ORD-123456?",
                "channel": "chat",
                "language": "en",
                "priority": "normal",
                "account_type": "standard"
            }
        }
    ]
    
    # Run each pipeline
    execution_results = []
    successful = 0
    failed = 0
    
    for pipeline_config in all_pipelines:
        yaml_path = Path("examples") / pipeline_config["file"]
        
        if not yaml_path.exists():
            print(f"\n⚠️  Skipping {pipeline_config['name']} - file not found: {yaml_path}")
            failed += 1
            continue
        
        try:
            result = await run_pipeline(
                name=pipeline_config["name"],
                yaml_path=yaml_path,
                inputs=pipeline_config["inputs"],
                registry=registry,
                save_outputs=True
            )
            
            execution_results.append(result)
            
            if result["status"] == "success":
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"\n💥 Catastrophic failure for {pipeline_config['name']}: {e}")
            failed += 1
            execution_results.append({
                "pipeline": pipeline_config["name"],
                "status": "catastrophic_failure",
                "error": str(e)
            })
    
    # Generate comprehensive report
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    print(f"\nTotal pipelines: {len(all_pipelines)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(f"examples/output/comprehensive_run_{timestamp}.json")
    
    summary_data = {
        "execution_timestamp": timestamp,
        "total_pipelines": len(all_pipelines),
        "successful": successful,
        "failed": failed,
        "available_models": available_models,
        "results": execution_results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n📊 Comprehensive results saved to: {summary_file}")
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 80)
    
    for result in execution_results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        duration = f"{result.get('duration', 0):.1f}s" if result.get('duration') else "N/A"
        print(f"{status_icon} {result['pipeline']:<30} - {result['status']:<10} ({duration})")
        
        if result["status"] == "error" and result.get("error"):
            print(f"   Error: {result['error'][:100]}...")
    
    # List all generated output files
    print("\n📁 Generated Output Files:")
    output_files = list(Path("examples/output").rglob("*.md"))
    recent_files = sorted(output_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]
    
    for file in recent_files:
        size = file.stat().st_size
        print(f"  - {file.name:<50} ({size:>8,} bytes)")
    
    print("\n✨ Pipeline execution complete! Check examples/output/ for all generated files.")


async def main():
    """Main entry point."""
    print("🚀 Orchestrator Comprehensive Pipeline Runner")
    print("This will run ALL example pipelines and generate outputs")
    print("-" * 80)
    
    # Check API keys
    api_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GOOGLE_API_KEY": "Google",
        "HUGGINGFACE_API_KEY": "HuggingFace"
    }
    
    print("\nAPI Key Status:")
    for key, name in api_keys.items():
        status = "✅ Available" if os.getenv(key) else "❌ Missing"
        print(f"  {name}: {status}")
    
    print("\nNote: Pipelines requiring missing APIs may fail")
    print("Starting in 3 seconds...\n")
    
    await asyncio.sleep(3)
    
    await run_all_pipelines()


if __name__ == "__main__":
    asyncio.run(main())