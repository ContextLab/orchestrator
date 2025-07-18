#!/usr/bin/env python3
"""Update run_all_documented_examples.py to use correct input variable names."""

import json

# Read the current script
with open('scripts/run_all_documented_examples.py', 'r') as f:
    content = f.read()

# Define the correct input mappings based on YAML files
input_mappings = {
    "research_assistant.yaml": {
        "topic": "query",  # YAML expects 'query' not 'topic'
        "depth": "max_sources",  # YAML has max_sources, not depth
        "sources": None  # Remove this, not in YAML
    },
    "code_analysis_suite.yaml": {
        "repository_path": "repo_path",  # YAML expects 'repo_path'
        "analysis_type": "analysis_depth",  # YAML expects 'analysis_depth'
        "output_format": None  # Remove this, not in YAML
    },
    "content_creation_pipeline.yaml": {
        "content_type": None,  # Not in YAML, remove
        "target_audience": "audience",  # YAML expects 'audience'
        "tone": "brand_voice",  # YAML expects 'brand_voice'
        "length": "target_length"  # YAML expects 'target_length'
    },
    "data_processing_workflow.yaml": {
        "data_source": "source",  # YAML expects 'source'
        "processing_type": None,  # Not in YAML
        "output_format": None  # Not in YAML
    },
    "multi_agent_collaboration.yaml": {
        "collaboration_mode": None  # Not in YAML
    },
    "automated_testing_system.yaml": {
        "codebase_path": "source_dir",  # YAML expects 'source_dir'
        "test_framework": "test_framework",  # Keep same
        "coverage_target": "coverage_target"  # Keep same
    },
    "document_intelligence.yaml": {
        "document_path": "input_dir",  # YAML expects 'input_dir'
        "analysis_type": None,  # Not in YAML
        "extract_insights": None  # Not in YAML
    },
    "creative_writing_assistant.yaml": {
        "theme": "initial_premise",  # YAML expects 'initial_premise'
        "tone": "writing_style",  # YAML expects 'writing_style'
        "length": "length"  # Keep same
    },
    "financial_analysis_bot.yaml": {
        "ticker": "symbols",  # YAML expects 'symbols' (list)
        "period": "time_period",  # YAML expects 'time_period'
        "analysis_type": "analysis_type"  # Keep same
    },
    "interactive_chat_bot.yaml": {
        "persona": "bot_personality",  # YAML expects 'bot_personality'
        "conversation_style": None,  # Not in YAML
        "topic": "initial_context"  # YAML expects 'initial_context'
    }
}

# Update the examples configuration
new_examples = [
    {
        "file": "research_assistant.yaml",
        "inputs": {
            "query": "The Impact of Large Language Models on Software Development",
            "context": "Focus on practical applications and real-world impact",
            "max_sources": 10,
            "quality_threshold": 0.8
        }
    },
    {
        "file": "code_analysis_suite.yaml",
        "inputs": {
            "repo_path": "src/orchestrator",
            "languages": ["python"],
            "analysis_depth": "comprehensive",
            "security_scan": True,
            "performance_check": True,
            "doc_check": True
        }
    },
    {
        "file": "content_creation_pipeline.yaml", 
        "inputs": {
            "topic": "Best Practices for Building AI-Powered Applications",
            "formats": ["blog", "social"],
            "audience": "software developers",
            "brand_voice": "professional yet engaging",
            "target_length": 1500,
            "auto_publish": False
        }
    },
    {
        "file": "data_processing_workflow.yaml",
        "inputs": {
            "source": "examples/test_data/*.csv",
            "output_dir": "examples/output/processed_data",
            "transformations": ["clean", "normalize", "aggregate"],
            "quality_checks": True,
            "generate_report": True
        }
    },
    {
        "file": "multi_agent_collaboration.yaml",
        "inputs": {
            "problem": "Design a scalable microservices architecture for an e-commerce platform",
            "num_agents": 4,
            "max_rounds": 5,
            "decision_method": "consensus",
            "include_critic": True
        }
    },
    {
        "file": "automated_testing_system.yaml",
        "inputs": {
            "source_dir": "src/orchestrator/core",
            "test_dir": "tests",
            "coverage_target": 85.0,
            "test_types": ["unit", "integration"],
            "test_framework": "pytest",
            "include_edge_cases": True,
            "include_performance": False
        }
    },
    {
        "file": "document_intelligence.yaml",
        "inputs": {
            "input_dir": "docs",
            "output_dir": "examples/output/document_analysis",
            "analysis_types": ["classification", "extraction", "summary"],
            "extract_entities": True,
            "detect_pii": True,
            "build_knowledge_graph": True,
            "output_format": "json"
        }
    },
    {
        "file": "creative_writing_assistant.yaml",
        "inputs": {
            "genre": "science fiction",
            "length": "short_story",
            "writing_style": "contemporary",
            "target_audience": "young_adult",
            "initial_premise": "Humanity's first contact with alien intelligence through dreams",
            "include_worldbuilding": True,
            "chapter_count": 5,
            "write_detailed_chapters": True
        }
    },
    {
        "file": "financial_analysis_bot.yaml",
        "inputs": {
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "time_period": "6M",
            "analysis_type": "comprehensive",
            "include_fundamentals": True,
            "include_predictions": True,
            "backtest_period": "1Y"
        }
    },
    {
        "file": "interactive_chat_bot.yaml",
        "inputs": {
            "bot_personality": "helpful AI assistant",
            "initial_context": "You are helping users learn about the Orchestrator framework",
            "conversation_mode": "educational",
            "max_turns": 10,
            "temperature": 0.7
        }
    },
    {
        "file": "scalable_customer_service_agent.yaml",
        "inputs": {
            "knowledge_base": "docs/faq.md",
            "escalation_threshold": 0.3,
            "response_style": "professional and empathetic",
            "categories": ["technical", "billing", "general"],
            "language": "en"
        }
    },
    {
        "file": "customer_support_automation.yaml",
        "inputs": {
            "ticket_source": "email",
            "priority_rules": "auto",
            "response_templates": "default",
            "auto_assign": True,
            "sentiment_analysis": True
        }
    }
]

# Write the updated script
updated_content = '''#!/usr/bin/env python3
"""Run all documented examples with real models and realistic inputs."""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Updated examples with correct input variable names
EXAMPLES = ''' + json.dumps(new_examples, indent=4) + '''

def check_api_keys():
    """Check if required API keys are set."""
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    missing_keys = []
    
    print("📋 API Key Status:")
    for key in required_keys:
        if os.getenv(key):
            print(f"  {key}: ✅ Set")
        else:
            print(f"  {key}: ❌ Missing")
            missing_keys.append(key)
    
    if missing_keys:
        print(f"\\n⚠️  Warning: Missing API keys: {', '.join(missing_keys)}")
        print("Some examples may fail without these keys.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    print()

def run_example(example):
    """Run a single example pipeline."""
    file_path = f"examples/{example['file']}"
    inputs = example.get("inputs", {})
    
    print("=" * 80)
    print(f"🚀 Running: {example['file']}")
    print("=" * 80)
    
    # Build command
    cmd = ["python", "scripts/run_pipeline.py", file_path]
    
    # Add inputs as command line arguments
    for key, value in inputs.items():
        if isinstance(value, list):
            cmd.extend([f"--{key}", json.dumps(value)])
        elif isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"Command: {' '.join(cmd[:3])} ... (with {len(inputs)} inputs)")
    print(f"Inputs: {json.dumps(inputs, indent=2)}")
    
    try:
        # Run the pipeline
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print("Output:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print(f"❌ Failed with return code: {result.returncode}")
            if result.stderr:
                print("Error:", result.stderr)
            if result.stdout:
                print("Output:", result.stdout)
                
    except subprocess.TimeoutExpired:
        print("❌ Timeout exceeded (5 minutes)")
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print()

def main():
    """Run all documented examples."""
    print("🎯 Running All Documented Examples with Real Models")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check API keys
    check_api_keys()
    
    # Run each example
    success_count = 0
    failed_examples = []
    
    for example in EXAMPLES:
        try:
            run_example(example)
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to run {example['file']}: {e}")
            failed_examples.append(example['file'])
    
    # Summary
    print("=" * 80)
    print("📊 Summary")
    print("=" * 80)
    print(f"Total examples: {len(EXAMPLES)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_examples)}")
    
    if failed_examples:
        print(f"\\nFailed examples: {', '.join(failed_examples)}")
    
    print(f"\\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
'''

with open('scripts/run_all_documented_examples.py', 'w') as f:
    f.write(updated_content)

print("Updated run_all_documented_examples.py with correct input variable names")