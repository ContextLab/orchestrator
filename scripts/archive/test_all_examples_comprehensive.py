#!/usr/bin/env python3
"""Comprehensive test of all YAML examples with real models."""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import traceback

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator import Orchestrator
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.integrations.anthropic_model import AnthropicModel
from orchestrator.integrations.google_model import GoogleModel
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem


class ComprehensiveExampleTester:
    """Test all YAML examples with real models and capture detailed outputs."""
    
    def __init__(self):
        self.orchestrator = None
        self.model_registry = ModelRegistry()
        self.examples_dir = Path("examples")
        self.output_dir = Path("example_outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def setup_models(self):
        """Configure available AI models."""
        models_configured = []
        
        # Try OpenAI first (most reliable)
        if os.environ.get("OPENAI_API_KEY"):
            try:
                openai_model = OpenAIModel(model_name="gpt-4o-mini")
                self.model_registry.register_model(openai_model)
                models_configured.append("OpenAI GPT-4o-mini")
            except Exception as e:
                print(f"Failed to configure OpenAI model: {e}")
        
        # Try Google models
        if os.environ.get("GOOGLE_API_KEY"):
            try:
                gemini_model = GoogleModel(model_name="gemini-1.5-flash")
                self.model_registry.register_model(gemini_model)
                models_configured.append("Google Gemini 1.5 Flash")
            except Exception as e:
                print(f"Failed to configure Google model: {e}")
        
        # Try Anthropic (may have credit issues)
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                claude_model = AnthropicModel(model_name="claude-3-5-sonnet-20241022")
                # Check if healthy before registering
                if await claude_model.health_check():
                    self.model_registry.register_model(claude_model)
                    models_configured.append("Anthropic Claude 3.5 Sonnet")
                else:
                    print("Anthropic model failed health check - skipping")
            except Exception as e:
                print(f"Failed to configure Anthropic model: {e}")
        
        if not models_configured:
            raise Exception("No AI models configured! Please set API keys.")
        
        print(f"Configured models: {', '.join(models_configured)}")
        
        # Create control system and orchestrator
        control_system = ModelBasedControlSystem(self.model_registry)
        self.orchestrator = Orchestrator(
            model_registry=self.model_registry,
            control_system=control_system
        )
        
        return True
        
    async def test_all_examples(self):
        """Test all YAML examples."""
        await self.setup_models()
        
        # Get all YAML files
        yaml_files = sorted(self.examples_dir.glob("*.yaml"))
        
        print(f"Found {len(yaml_files)} YAML examples to test")
        
        results = {}
        
        for yaml_file in yaml_files:
            example_name = yaml_file.name
            print(f"\n=== Testing {example_name} ===")
            
            try:
                # Load YAML content
                with open(yaml_file, 'r') as f:
                    yaml_content = f.read()
                
                # Get appropriate inputs
                inputs = self.get_example_inputs(example_name)
                
                # Execute pipeline
                start_time = datetime.now()
                result = await self.orchestrator.execute_yaml(yaml_content, inputs)
                end_time = datetime.now()
                
                execution_time = (end_time - start_time).total_seconds()
                
                results[example_name] = {
                    "status": "success",
                    "execution_time": execution_time,
                    "inputs": inputs,
                    "outputs": result.get("outputs", {}) if isinstance(result, dict) else {},
                    "step_results": result.get("step_results", result) if isinstance(result, dict) else result,
                    "timestamp": start_time.isoformat()
                }
                
                print(f"✓ {example_name} completed in {execution_time:.1f}s")
                
                # Save individual result
                with open(self.output_dir / f"{example_name.replace('.yaml', '')}_result.json", 'w') as f:
                    json.dump(results[example_name], f, indent=2, default=str)
                
            except Exception as e:
                results[example_name] = {
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "inputs": inputs if 'inputs' in locals() else {},
                    "timestamp": datetime.now().isoformat()
                }
                
                print(f"✗ {example_name} failed: {str(e)}")
                
                # Save error result
                with open(self.output_dir / f"{example_name.replace('.yaml', '')}_error.json", 'w') as f:
                    json.dump(results[example_name], f, indent=2, default=str)
        
        # Save comprehensive results
        with open(self.output_dir / "comprehensive_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        await self.generate_summary_report(results)
        
        return results
    
    def get_example_inputs(self, example_name: str) -> Dict[str, Any]:
        """Get appropriate inputs for each example."""
        inputs = {
            "research_assistant.yaml": {
                "query": "latest developments in artificial intelligence 2024",
                "context": "Focus on breakthrough technologies and practical applications",
                "max_sources": 10,
                "quality_threshold": 0.8
            },
            "data_processing_workflow.yaml": {
                "source": "sample_dataset.csv",
                "output_path": "/tmp/processed_data",
                "output_format": "json",
                "quality_threshold": 0.9
            },
            "multi_agent_collaboration.yaml": {
                "problem": "Build a web application for task management",
                "num_agents": 5,
                "max_rounds": 10,
                "agent_roles": "auto"
            },
            "content_creation_pipeline.yaml": {
                "topic": "sustainable technology trends",
                "formats": ["blog", "social"],
                "audience": "technology professionals",
                "brand_voice": "professional"
            },
            "code_analysis_suite.yaml": {
                "repo_path": "/tmp/sample_project",
                "languages": ["python", "javascript"],
                "analysis_depth": "comprehensive",
                "security_scan": True,
                "performance_check": True,
                "doc_check": True
            },
            "customer_support_automation.yaml": {
                "ticket_id": "TICKET-2024-001",
                "ticketing_system": "zendesk",
                "auto_respond": True,
                "languages": ["en", "es"],
                "escalation_threshold": -0.3
            },
            "automated_testing_system.yaml": {
                "source_dir": "/tmp/sample_project",
                "test_dir": "./tests",
                "coverage_target": 85.0,
                "test_types": ["unit", "integration"],
                "test_framework": "pytest"
            },
            "creative_writing_assistant.yaml": {
                "genre": "science fiction",
                "length": "short_story",
                "writing_style": "literary",
                "target_audience": "adult readers",
                "initial_premise": "A scientist discovers parallel universes through quantum computing",
                "include_worldbuilding": True,
                "chapter_count": 3
            },
            "interactive_chat_bot.yaml": {
                "message": "I need help planning a sustainable technology conference",
                "conversation_id": "conv_2024_001",
                "persona": "helpful-assistant",
                "enable_streaming": False,
                "available_tools": ["web_search", "knowledge_base"]
            },
            "scalable_customer_service_agent.yaml": {
                "interaction_id": "INT-2024-001",
                "customer_id": "CUST-12345",
                "channel": "chat",
                "content": "I'm having trouble with my recent order and need a refund",
                "metadata": {"session_id": "sess_123", "user_agent": "web_chat"},
                "languages": ["en"],
                "sla_targets": {"first_response": 60, "resolution": 3600}
            },
            "document_intelligence.yaml": {
                "input_dir": "/tmp/sample_documents",
                "output_dir": "/tmp/processed_documents",
                "enable_ocr": True,
                "languages": ["en"],
                "extract_tables": True,
                "build_knowledge_graph": True
            },
            "financial_analysis_bot.yaml": {
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "timeframe": "1y",
                "analysis_type": "comprehensive",
                "risk_tolerance": "moderate",
                "include_predictions": True,
                "run_backtest": False
            }
        }
        
        return inputs.get(example_name, {})
    
    async def generate_summary_report(self, results: Dict[str, Any]):
        """Generate a comprehensive summary report."""
        successful = [name for name, result in results.items() if result["status"] == "success"]
        failed = [name for name, result in results.items() if result["status"] == "error"]
        
        total_execution_time = sum(
            result.get("execution_time", 0) 
            for result in results.values() 
            if result["status"] == "success"
        )
        
        report = {
            "summary": {
                "total_examples": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(results) * 100,
                "total_execution_time": total_execution_time,
                "average_execution_time": total_execution_time / len(successful) if successful else 0
            },
            "successful_examples": successful,
            "failed_examples": failed,
            "execution_details": {
                name: {
                    "execution_time": result.get("execution_time", 0),
                    "step_count": len(result.get("step_results", {})) if isinstance(result.get("step_results"), dict) else 0,
                    "output_keys": list(result.get("outputs", {}).keys())
                }
                for name, result in results.items()
                if result["status"] == "success"
            },
            "error_analysis": {
                name: {
                    "error_type": result.get("error_type", "Unknown"),
                    "error_message": result.get("error", "No error message")
                }
                for name, result in results.items()
                if result["status"] == "error"
            }
        }
        
        # Save report
        with open(self.output_dir / "summary_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n{'='*60}")
        print("COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total examples tested: {report['summary']['total_examples']}")
        print(f"Successful: {report['summary']['successful']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Success rate: {report['summary']['success_rate']:.1f}%")
        print(f"Total execution time: {report['summary']['total_execution_time']:.1f}s")
        print(f"Average execution time: {report['summary']['average_execution_time']:.1f}s")
        
        if successful:
            print(f"\nSuccessful examples:")
            for name in successful:
                exec_time = results[name].get("execution_time", 0)
                print(f"  ✓ {name} ({exec_time:.1f}s)")
        
        if failed:
            print(f"\nFailed examples:")
            for name in failed:
                error_type = results[name].get("error_type", "Unknown")
                print(f"  ✗ {name} ({error_type})")
        
        print(f"\nDetailed results saved to: {self.output_dir}")


async def main():
    """Main test function."""
    tester = ComprehensiveExampleTester()
    results = await tester.test_all_examples()
    return results


if __name__ == "__main__":
    asyncio.run(main())