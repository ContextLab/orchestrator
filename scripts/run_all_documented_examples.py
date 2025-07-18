#!/usr/bin/env python3
"""Run all documented examples with real models and appropriate inputs."""

import subprocess
import json
from pathlib import Path
from datetime import datetime

# Define appropriate inputs for each example
EXAMPLE_INPUTS = {
    "research_assistant.yaml": {
        "topic": "The Impact of Large Language Models on Software Development",
        "depth": "comprehensive",
        "sources": 10
    },
    "code_analysis_suite.yaml": {
        "repository_path": "src/orchestrator",
        "analysis_type": "comprehensive",
        "output_format": "detailed"
    },
    "content_creation_pipeline.yaml": {
        "topic": "Best Practices for Building AI-Powered Applications",
        "content_type": "blog_post",
        "target_audience": "software developers",
        "tone": "professional yet engaging",
        "length": 1500
    },
    "data_processing_workflow.yaml": {
        "data_source": "examples/test_data/sample_data.csv",
        "processing_type": "analysis",
        "output_format": "report"
    },
    "multi_agent_collaboration.yaml": {
        "problem": "Design a scalable microservices architecture for an e-commerce platform",
        "num_agents": 4,
        "collaboration_mode": "consensus"
    },
    "automated_testing_system.yaml": {
        "codebase_path": "src/orchestrator/core",
        "test_framework": "pytest",
        "coverage_target": 85
    },
    "document_intelligence.yaml": {
        "document_path": "README.md",
        "analysis_type": "comprehensive",
        "extract_insights": True
    },
    "creative_writing_assistant.yaml": {
        "genre": "science fiction",
        "theme": "humanity's first contact with alien intelligence",
        "tone": "thought-provoking",
        "length": "short story"
    },
    "financial_analysis_bot.yaml": {
        "company": "Tesla",
        "analysis_period": "2024",
        "metrics": ["revenue", "profit margins", "market position"],
        "report_type": "comprehensive"
    },
    "interactive_chat_bot.yaml": {
        "bot_name": "TechAssist",
        "bot_personality": "helpful and knowledgeable",
        "topic": "cloud computing",
        "user_queries": [
            "What are the main benefits of cloud computing?",
            "How do I choose between AWS, Azure, and Google Cloud?",
            "What are some best practices for cloud security?",
            "Can you explain serverless architecture?",
            "Thank you for your help!"
        ]
    },
    "scalable_customer_service_agent.yaml": {
        "customer_name": "Sarah Johnson",
        "issue_type": "technical",
        "issue_description": "Unable to access account after password reset",
        "priority": "high",
        "customer_history": "premium customer for 3 years"
    },
    "customer_support_automation.yaml": {
        "ticket_id": "SUPPORT-2024-001",
        "customer_email": "john.doe@example.com",
        "issue_category": "billing",
        "issue_details": "Duplicate charge on credit card for last month's subscription",
        "urgency": "medium"
    }
}

def run_example(yaml_file, inputs):
    """Run a single example using scripts/run_pipeline.py."""
    print(f"\n{'='*80}")
    print(f"🚀 Running: {yaml_file}")
    print(f"{'='*80}")
    
    # Prepare command - use the version with real models
    cmd = ["python", "scripts/run_pipeline_with_models.py", f"examples/{yaml_file}"]
    
    # Add output directory
    cmd.extend(["-o", "examples/output"])
    
    # Add inputs
    for key, value in inputs.items():
        if isinstance(value, (list, dict)):
            # For complex values, save to temp JSON file
            temp_file = Path(f"temp_inputs_{yaml_file}.json")
            with open(temp_file, 'w') as f:
                json.dump({key: value}, f)
            cmd.extend(["-f", str(temp_file)])
        else:
            cmd.extend(["-i", f"{key}={json.dumps(value)}"])
    
    print(f"Command: {' '.join(cmd[:3])} ... (with {len(inputs)} inputs)")
    print(f"Inputs: {json.dumps(inputs, indent=2)}")
    
    # Run the pipeline
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Success!")
            # Show last few lines of output
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) > 5:
                print("\nLast 5 lines of output:")
                for line in output_lines[-5:]:
                    print(f"  {line}")
        else:
            print(f"❌ Failed with return code: {result.returncode}")
            print(f"Error: {result.stderr}")
            
        # Clean up temp file if created
        temp_file = Path(f"temp_inputs_{yaml_file}.json")
        if temp_file.exists():
            temp_file.unlink()
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def main():
    """Run all documented examples."""
    print("🎯 Running All Documented Examples with Real Models")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for API keys
    import os
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY")
    }
    
    print("\n📋 API Key Status:")
    for key, value in api_keys.items():
        status = "✅ Set" if value else "❌ Not set"
        print(f"  {key}: {status}")
    
    if not any(api_keys.values()):
        print("\n⚠️  Warning: No API keys found. Examples will use mock models.")
    
    # Run each example
    results = {}
    for yaml_file, inputs in EXAMPLE_INPUTS.items():
        success = run_example(yaml_file, inputs)
        results[yaml_file] = success
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 Summary")
    print(f"{'='*80}")
    
    success_count = sum(1 for s in results.values() if s)
    total_count = len(results)
    
    print(f"\nTotal examples: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    if success_count < total_count:
        print("\n❌ Failed examples:")
        for example, success in results.items():
            if not success:
                print(f"  - {example}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List generated files
    output_dir = Path("examples/output")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        if files:
            print(f"\n📁 Generated {len(files)} output files:")
            for f in sorted(files)[:10]:  # Show first 10
                size = f.stat().st_size
                print(f"  {f.name:<50} ({size:>8,} bytes)")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")

if __name__ == "__main__":
    main()