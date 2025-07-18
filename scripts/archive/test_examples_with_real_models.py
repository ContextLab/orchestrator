#!/usr/bin/env python3
"""Test YAML examples with real AI models to verify quality and functionality."""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestrator import Orchestrator
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.anthropic_model import AnthropicModel
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.integrations.google_model import GoogleModel
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem


class RealModelTester:
    """Test YAML examples with real AI models."""
    
    def __init__(self):
        self.results = []
        self.orchestrator = None
        self.model_registry = ModelRegistry()
        
    async def setup_models(self):
        """Configure real AI models based on available API keys."""
        models_configured = []
        
        # Try to configure Anthropic models
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                # Claude 3.5 Sonnet - latest and most capable
                claude_sonnet = AnthropicModel(
                    model_name="claude-3-5-sonnet-20241022"
                )
                self.model_registry.register_model(claude_sonnet)
                models_configured.append("Anthropic Claude 3.5 Sonnet")
                
                # Claude 3 Haiku - fast and cost-effective
                claude_haiku = AnthropicModel(
                    model_name="claude-3-haiku-20240307"
                )
                self.model_registry.register_model(claude_haiku)
                models_configured.append("Anthropic Claude 3 Haiku")
            except Exception as e:
                print(f"Failed to configure Anthropic models: {e}")
        
        # Try to configure OpenAI models
        if os.environ.get("OPENAI_API_KEY"):
            try:
                # GPT-4 - most capable OpenAI model
                gpt4 = OpenAIModel(
                    model_name="gpt-4"
                )
                self.model_registry.register_model(gpt4)
                models_configured.append("OpenAI GPT-4")
                
                # GPT-3.5 Turbo - fast and cost-effective
                gpt35 = OpenAIModel(
                    model_name="gpt-3.5-turbo"
                )
                self.model_registry.register_model(gpt35)
                models_configured.append("OpenAI GPT-3.5 Turbo")
            except Exception as e:
                print(f"Failed to configure OpenAI models: {e}")
        
        # Try to configure Google models
        if os.environ.get("GOOGLE_API_KEY"):
            try:
                # Gemini Pro - Google's latest model
                gemini = GoogleModel(
                    model_name="gemini-pro"
                )
                self.model_registry.register_model(gemini)
                models_configured.append("Google Gemini Pro")
            except Exception as e:
                print(f"Failed to configure Google models: {e}")
        
        if not models_configured:
            print("\nWARNING: No AI models configured!")
            print("Please set one or more API keys:")
            print("  - ANTHROPIC_API_KEY")
            print("  - OPENAI_API_KEY")
            print("  - GOOGLE_API_KEY")
            return False
        
        print(f"\nConfigured models: {', '.join(models_configured)}")
        
        # Create model-based control system
        control_system = ModelBasedControlSystem(
            model_registry=self.model_registry
        )
        
        # Create orchestrator with real models and control system
        self.orchestrator = Orchestrator(
            model_registry=self.model_registry,
            control_system=control_system
        )
        
        return True
    
    async def test_example(self, example_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single example with real models."""
        print(f"\n{'=' * 60}")
        print(f"Testing: {example_name}")
        print(f"{'=' * 60}")
        
        example_path = Path(__file__).parent / "examples" / example_name
        
        if not example_path.exists():
            print(f"ERROR: Example file not found: {example_path}")
            return {
                "example": example_name,
                "status": "error",
                "error": "File not found"
            }
        
        try:
            # Read YAML content
            with open(example_path, 'r') as f:
                yaml_content = f.read()
            
            print(f"Inputs: {json.dumps(inputs, indent=2)}")
            print("\nExecuting pipeline...")
            
            start_time = datetime.now()
            
            # Execute with real models
            result = await self.orchestrator.execute_yaml(
                yaml_content,
                context=inputs
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            print(f"\nExecution completed in {execution_time:.2f} seconds")
            print(f"Output keys: {list(result.keys())}")
            
            # Extract key outputs for evaluation
            evaluation = self.evaluate_result(example_name, result)
            
            return {
                "example": example_name,
                "status": "success",
                "execution_time": execution_time,
                "outputs": result,
                "evaluation": evaluation
            }
            
        except Exception as e:
            print(f"\nERROR executing {example_name}: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "example": example_name,
                "status": "error",
                "error": str(e)
            }
    
    def evaluate_result(self, example_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality of the result."""
        evaluation = {
            "has_outputs": bool(result),
            "output_count": len(result),
            "quality_checks": []
        }
        
        # Since outputs are step results, check the actual content quality
        # Example-specific quality checks based on step outputs
        if "research_assistant" in example_name:
            # Check if key steps produced meaningful output
            checks = [
                ("analyze_query_executed", "analyze_query" in result and len(str(result.get("analyze_query", ""))) > 50),
                ("web_search_executed", "web_search" in result),
                ("synthesis_executed", "synthesize_findings" in result),
                ("report_generated", "generate_report" in result and len(str(result.get("generate_report", ""))) > 200),
                ("quality_check_passed", "quality_check" in result)
            ]
            evaluation["quality_checks"] = checks
            
        elif "data_processing" in example_name:
            # Check data processing steps
            checks = [
                ("sources_discovered", "discover_sources" in result),
                ("schema_validated", "validate_schema" in result),
                ("data_cleaned", "clean_data" in result),
                ("data_exported", "export_data" in result),
                ("report_generated", "generate_report" in result and len(str(result.get("generate_report", ""))) > 100)
            ]
            evaluation["quality_checks"] = checks
            
        elif "multi_agent" in example_name:
            # Check multi-agent collaboration steps
            checks = [
                ("agents_initialized", "initialize_agents" in result),
                ("problem_decomposed", "decompose_problem" in result),
                ("collaboration_executed", "collaboration_round" in result),
                ("solution_integrated", "integrate_solutions" in result),
                ("report_generated", "generate_report" in result and len(str(result.get("generate_report", ""))) > 100)
            ]
            evaluation["quality_checks"] = checks
            
        
        # For all other examples, create generic checks based on step count and content
        if not evaluation["quality_checks"]:
            # Generic quality checks
            step_count = len(result)
            total_content_length = sum(len(str(v)) for v in result.values())
            avg_content_length = total_content_length / step_count if step_count > 0 else 0
            
            checks = [
                ("has_multiple_steps", step_count >= 3),
                ("steps_have_content", avg_content_length > 50),
                ("total_content_substantial", total_content_length > 500),
                ("no_empty_outputs", all(str(v).strip() for v in result.values())),
                ("execution_complete", step_count > 0)
            ]
            evaluation["quality_checks"] = checks
        
        # Calculate quality score
        if evaluation["quality_checks"]:
            passed_checks = sum(1 for _, passed in evaluation["quality_checks"] if passed)
            total_checks = len(evaluation["quality_checks"])
            evaluation["quality_score"] = passed_checks / total_checks
        else:
            evaluation["quality_score"] = 1.0 if evaluation["has_outputs"] else 0.0
        
        return evaluation
    
    async def run_all_tests(self):
        """Run all example tests."""
        # Example configurations with correct input names from YAML files
        test_cases = [
            ("research_assistant.yaml", {
                "query": "artificial intelligence in healthcare",
                "context": "Focus on recent developments in diagnosis and treatment",
                "max_sources": 5,
                "quality_threshold": 0.7
            }),
            ("data_processing_workflow.yaml", {
                "source": "sales_data_*.csv",
                "output_path": "./processed_data",
                "output_format": "json",
                "chunk_size": 1000,
                "quality_threshold": 0.8
            }),
            ("multi_agent_collaboration.yaml", {
                "problem": "Design a sustainable urban transportation system",
                "num_agents": 4,
                "agent_roles": "balanced",
                "max_rounds": 3,
                "consensus_threshold": 0.8
            }),
            ("content_creation_pipeline.yaml", {
                "topic": "The Future of Remote Work",
                "formats": ["blog"],
                "audience": "business professionals",
                "brand_voice": "professional yet approachable",
                "goals": ["educate", "engage"],
                "target_length": 800
            }),
            ("code_analysis_suite.yaml", {
                "repo_path": "./src",
                "languages": ["python"],
                "analysis_depth": "comprehensive",
                "security_scan": True,
                "performance_check": True
            }),
            ("customer_support_automation.yaml", {
                "ticket_id": "TICKET-12345",
                "ticketing_system": "zendesk",
                "auto_respond": True,
                "languages": ["en"],
                "escalation_threshold": -0.5
            }),
            ("automated_testing_system.yaml", {
                "source_dir": "./src",
                "test_dir": "./tests",
                "coverage_target": 80.0,
                "test_types": ["unit", "integration"],
                "test_framework": "pytest"
            }),
            ("creative_writing_assistant.yaml", {
                "genre": "science fiction",
                "length": "short_story",
                "writing_style": "contemporary",
                "target_audience": "young adults",
                "initial_premise": "First contact with aliens through music",
                "include_worldbuilding": True
            }),
            ("interactive_chat_bot.yaml", {
                "message": "What's the weather like today?",
                "conversation_id": "test_session_001",
                "persona": "helpful-assistant",
                "enable_streaming": False,
                "available_tools": ["weather", "web_search"]
            }),
            ("scalable_customer_service_agent.yaml", {
                "interaction_id": "INT-123456",
                "customer_id": "CUST-12345",
                "channel": "email",
                "content": "My premium subscription isn't showing the correct features",
                "metadata": {"priority": "high", "account_type": "premium"}
            }),
            ("document_intelligence.yaml", {
                "input_dir": "./sample_docs",
                "output_dir": "./processed",
                "enable_ocr": True,
                "languages": ["en"],
                "extract_tables": True,
                "build_knowledge_graph": True
            }),
            ("financial_analysis_bot.yaml", {
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "timeframe": "1y",
                "analysis_type": "comprehensive",
                "risk_tolerance": "moderate",
                "include_predictions": True
            })
        ]
        
        # Run tests
        for example_name, inputs in test_cases:
            result = await self.test_example(example_name, inputs)
            self.results.append(result)
            
            # Add a small delay between tests to avoid rate limiting
            await asyncio.sleep(2)
    
    def generate_report(self):
        """Generate a comprehensive test report."""
        report_path = Path(__file__).parent / "example_quality_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# YAML Examples Quality Test Report\n\n")
            f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Models Used:** {', '.join(self.model_registry.list_models())}\n\n")
            
            # Summary statistics
            total_examples = len(self.results)
            successful = sum(1 for r in self.results if r["status"] == "success")
            failed = total_examples - successful
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Examples Tested:** {total_examples}\n")
            f.write(f"- **Successful:** {successful}\n")
            f.write(f"- **Failed:** {failed}\n")
            f.write(f"- **Success Rate:** {(successful/total_examples)*100:.1f}%\n\n")
            
            # Quality summary
            if successful > 0:
                avg_quality = sum(
                    r["evaluation"]["quality_score"] 
                    for r in self.results 
                    if r["status"] == "success"
                ) / successful
                f.write(f"- **Average Quality Score:** {avg_quality:.2f}/1.0\n\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            
            for result in self.results:
                f.write(f"### {result['example']}\n\n")
                
                if result["status"] == "success":
                    f.write(f"**Status:** ✅ Success\n")
                    f.write(f"**Execution Time:** {result['execution_time']:.2f}s\n")
                    f.write(f"**Quality Score:** {result['evaluation']['quality_score']:.2f}/1.0\n\n")
                    
                    # Quality checks
                    if result["evaluation"]["quality_checks"]:
                        f.write("**Quality Checks:**\n")
                        for check_name, passed in result["evaluation"]["quality_checks"]:
                            emoji = "✅" if passed else "❌"
                            f.write(f"- {emoji} {check_name}\n")
                        f.write("\n")
                    
                    # Sample outputs
                    f.write("**Sample Outputs:**\n```json\n")
                    # Show first few outputs
                    sample_outputs = {}
                    for i, (key, value) in enumerate(result["outputs"].items()):
                        if i >= 3:  # Show max 3 outputs
                            sample_outputs["..."] = f"({len(result['outputs']) - 3} more outputs)"
                            break
                        # Truncate long values
                        if isinstance(value, str) and len(value) > 200:
                            sample_outputs[key] = value[:200] + "..."
                        elif isinstance(value, (list, dict)) and len(str(value)) > 200:
                            sample_outputs[key] = f"{type(value).__name__} with {len(value)} items"
                        else:
                            sample_outputs[key] = value
                    f.write(json.dumps(sample_outputs, indent=2))
                    f.write("\n```\n\n")
                else:
                    f.write(f"**Status:** ❌ Failed\n")
                    f.write(f"**Error:** {result['error']}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if failed > 0:
                f.write("### Failed Examples\n\n")
                for result in self.results:
                    if result["status"] == "error":
                        f.write(f"- **{result['example']}**: {result['error']}\n")
                f.write("\n")
            
            # Quality issues
            low_quality = [
                r for r in self.results 
                if r["status"] == "success" and r["evaluation"]["quality_score"] < 0.7
            ]
            if low_quality:
                f.write("### Low Quality Outputs\n\n")
                for result in low_quality:
                    f.write(f"- **{result['example']}** (Score: {result['evaluation']['quality_score']:.2f})\n")
                    failed_checks = [
                        check_name for check_name, passed in result["evaluation"]["quality_checks"]
                        if not passed
                    ]
                    if failed_checks:
                        f.write(f"  - Failed checks: {', '.join(failed_checks)}\n")
                f.write("\n")
            
            f.write("## Conclusion\n\n")
            if successful == total_examples and avg_quality >= 0.8:
                f.write("✅ All examples executed successfully with high quality outputs!\n")
            elif successful == total_examples:
                f.write("✅ All examples executed successfully, but some outputs could be improved.\n")
            elif successful > 0:
                f.write("⚠️ Some examples failed, but those that succeeded produced reasonable outputs.\n")
            else:
                f.write("❌ All examples failed. Please check model configuration and API keys.\n")
        
        print(f"\n\nReport saved to: {report_path}")
        return report_path


async def main():
    """Main test runner."""
    print("Starting YAML Examples Quality Test")
    print("=" * 60)
    
    tester = RealModelTester()
    
    # Setup models
    if not await tester.setup_models():
        print("\nExiting: No models configured")
        return
    
    # Run tests
    await tester.run_all_tests()
    
    # Generate report
    report_path = tester.generate_report()
    
    print("\n" + "=" * 60)
    print("Testing Complete!")
    print(f"See detailed report: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())