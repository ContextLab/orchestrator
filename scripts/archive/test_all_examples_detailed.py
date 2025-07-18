#!/usr/bin/env python3
"""Test all YAML examples with real models and capture detailed outputs for analysis."""

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
from orchestrator.integrations.anthropic_model import AnthropicModel
from orchestrator.integrations.openai_model import OpenAIModel
from orchestrator.integrations.google_model import GoogleModel
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem


class DetailedExampleTester:
    """Test YAML examples with detailed output capture."""
    
    def __init__(self):
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
    
    async def test_example(self, example_name: str, inputs: Dict[str, Any], output_dir: Path):
        """Test a single example and save detailed outputs."""
        print(f"\n{'=' * 80}")
        print(f"Testing: {example_name}")
        print(f"{'=' * 80}")
        
        example_path = Path(__file__).parent / "examples" / example_name
        output_file = output_dir / f"{example_name.replace('.yaml', '')}_output.json"
        
        if not example_path.exists():
            print(f"ERROR: Example file not found: {example_path}")
            return
        
        try:
            # Read YAML content
            with open(example_path, 'r') as f:
                yaml_content = f.read()
            
            print(f"\nInputs:")
            print(json.dumps(inputs, indent=2))
            print("\nExecuting pipeline...")
            
            start_time = datetime.now()
            
            # Execute with real models
            result = await self.orchestrator.execute_yaml(
                yaml_content,
                context=inputs
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            print(f"\n✅ SUCCESS - Execution completed in {execution_time:.2f} seconds")
            print(f"Number of steps executed: {len(result)}")
            
            # Save full output
            output_data = {
                "example": example_name,
                "status": "success",
                "execution_time": execution_time,
                "inputs": inputs,
                "outputs": result,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            # Display key outputs
            print(f"\nKey Outputs (first 3 steps):")
            for i, (step_id, output) in enumerate(result.items()):
                if i >= 3:
                    print(f"... and {len(result) - 3} more steps")
                    break
                
                output_str = str(output)
                if len(output_str) > 300:
                    output_str = output_str[:300] + "..."
                print(f"\n[{step_id}]:")
                print(output_str)
            
            print(f"\nFull output saved to: {output_file}")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            traceback.print_exc()
            
            # Save error information
            output_data = {
                "example": example_name,
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "inputs": inputs,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
    
    async def run_all_tests(self):
        """Run all example tests."""
        # Create output directory
        output_dir = Path(__file__).parent / "example_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nOutput directory: {output_dir}")
        
        # Example configurations with correct input names from YAML files
        test_cases = [
            ("research_assistant.yaml", {
                "query": "What are the latest breakthroughs in quantum computing for drug discovery?",
                "context": "Focus on practical applications in pharmaceutical research from 2023-2024",
                "max_sources": 5,
                "quality_threshold": 0.7
            }),
            
            ("data_processing_workflow.yaml", {
                "source": "customer_transactions_2024_*.csv",
                "output_path": "./processed_analytics",
                "output_format": "parquet",
                "chunk_size": 5000,
                "quality_threshold": 0.85,
                "parallel_workers": 4
            }),
            
            ("multi_agent_collaboration.yaml", {
                "problem": "Design an AI-powered education platform that personalizes learning for K-12 students",
                "num_agents": 5,
                "agent_roles": "balanced",
                "max_rounds": 4,
                "consensus_threshold": 0.8
            }),
            
            ("content_creation_pipeline.yaml", {
                "topic": "The Impact of AI on Software Development in 2024",
                "formats": ["blog", "social"],
                "audience": "software developers and tech leaders",
                "brand_voice": "authoritative yet accessible",
                "goals": ["educate", "thought leadership", "engagement"],
                "target_length": 1200,
                "auto_publish": False
            }),
            
            ("code_analysis_suite.yaml", {
                "repo_path": "./src/orchestrator",
                "languages": ["python"],
                "analysis_depth": "comprehensive",
                "security_scan": True,
                "performance_check": True,
                "doc_check": True,
                "severity_threshold": "medium"
            }),
            
            ("customer_support_automation.yaml", {
                "ticket_id": "TICKET-2024-001",
                "ticketing_system": "internal",
                "auto_respond": True,
                "languages": ["en"],
                "escalation_threshold": -0.5,
                "kb_confidence_threshold": 0.8
            }),
            
            ("automated_testing_system.yaml", {
                "source_dir": "./src/orchestrator/core",
                "test_dir": "./tests/unit",
                "coverage_target": 85.0,
                "test_types": ["unit", "integration"],
                "test_framework": "pytest",
                "include_edge_cases": True,
                "include_performance": True
            }),
            
            ("creative_writing_assistant.yaml", {
                "genre": "science fiction",
                "length": "short_story",
                "writing_style": "literary",
                "target_audience": "adult readers",
                "initial_premise": "A scientist discovers that memories can be transferred between parallel universes",
                "include_worldbuilding": True,
                "chapter_count": 3
            }),
            
            ("interactive_chat_bot.yaml", {
                "message": "Can you help me understand how neural networks learn?",
                "conversation_id": "conv_12345",
                "persona": "knowledgeable-teacher",
                "enable_streaming": False,
                "safety_level": "moderate",
                "available_tools": ["web_search", "calculator"],
                "max_response_length": 500
            }),
            
            ("scalable_customer_service_agent.yaml", {
                "interaction_id": "INT-2024-789",
                "customer_id": "CUST-PREMIUM-456",
                "channel": "chat",
                "content": "I've been charged twice for my subscription this month. This is the second time this has happened.",
                "metadata": {
                    "priority": "high",
                    "account_type": "premium",
                    "sentiment": "frustrated",
                    "previous_issues": 2
                }
            }),
            
            ("document_intelligence.yaml", {
                "input_dir": "./sample_documents",
                "output_dir": "./extracted_data",
                "enable_ocr": True,
                "languages": ["en"],
                "custom_entities": ["product_names", "pricing", "contract_terms"],
                "output_format": "json",
                "extract_tables": True,
                "build_knowledge_graph": True
            }),
            
            ("financial_analysis_bot.yaml", {
                "symbols": ["AAPL", "MSFT", "GOOGL"],
                "timeframe": "6m",
                "analysis_type": "comprehensive",
                "risk_tolerance": "moderate",
                "asset_type": "equity",
                "run_backtest": True,
                "include_predictions": True
            })
        ]
        
        # Summary for final report
        summary = {
            "total": len(test_cases),
            "successful": 0,
            "failed": 0,
            "examples": []
        }
        
        # Run tests
        for example_name, inputs in test_cases:
            await self.test_example(example_name, inputs, output_dir)
            
            # Check result
            output_file = output_dir / f"{example_name.replace('.yaml', '')}_output.json"
            if output_file.exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    if data["status"] == "success":
                        summary["successful"] += 1
                    else:
                        summary["failed"] += 1
                    summary["examples"].append({
                        "name": example_name,
                        "status": data["status"],
                        "time": data.get("execution_time", "N/A")
                    })
            
            # Add a delay between tests to avoid rate limiting
            await asyncio.sleep(3)
        
        # Generate summary report
        self.generate_summary_report(output_dir, summary)
    
    def generate_summary_report(self, output_dir: Path, summary: Dict[str, Any]):
        """Generate a summary report of all tests."""
        report_path = output_dir / "summary_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# YAML Examples Execution Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Output Directory:** `{output_dir}`\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Examples:** {summary['total']}\n")
            f.write(f"- **Successful:** {summary['successful']}\n")
            f.write(f"- **Failed:** {summary['failed']}\n")
            f.write(f"- **Success Rate:** {(summary['successful']/summary['total']*100):.1f}%\n\n")
            
            f.write("## Individual Results\n\n")
            f.write("| Example | Status | Execution Time |\n")
            f.write("|---------|--------|----------------|\n")
            
            for example in summary["examples"]:
                status_emoji = "✅" if example["status"] == "success" else "❌"
                time_str = f"{example['time']:.2f}s" if isinstance(example['time'], (int, float)) else example['time']
                f.write(f"| {example['name']} | {status_emoji} {example['status']} | {time_str} |\n")
            
            f.write("\n## Analysis Instructions\n\n")
            f.write("1. Review each JSON output file for detailed step-by-step results\n")
            f.write("2. Look for:\n")
            f.write("   - Steps that produced unexpected or low-quality outputs\n")
            f.write("   - Error messages or failed steps\n")
            f.write("   - Outputs that don't match the intended task\n")
            f.write("   - Performance bottlenecks (long execution times)\n")
            f.write("3. Common issues to check:\n")
            f.write("   - Prompts that are too vague or ambiguous\n")
            f.write("   - Missing context between steps\n")
            f.write("   - Incorrect step dependencies\n")
            f.write("   - Model selection problems\n")
        
        print(f"\n\n{'=' * 80}")
        print(f"Summary report saved to: {report_path}")
        print(f"All outputs saved in: {output_dir}")


async def main():
    """Main test runner."""
    print("Starting Detailed YAML Examples Test")
    print("=" * 80)
    
    tester = DetailedExampleTester()
    
    # Setup models
    if not await tester.setup_models():
        print("\nExiting: No models configured")
        return
    
    # Run all tests
    await tester.run_all_tests()
    
    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("Please review the output files for detailed analysis.")


if __name__ == "__main__":
    asyncio.run(main())