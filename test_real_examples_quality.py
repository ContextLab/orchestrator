#!/usr/bin/env python3
"""
Test real examples with actual models to assess output quality.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, Any

from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel


async def test_research_assistant():
    """Test the research assistant with a real topic."""
    print("=" * 60)
    print("TESTING: Research Assistant")
    print("=" * 60)
    
    # Setup
    model_registry = ModelRegistry()
    model = OpenAIModel(model_name="gpt-4o-mini")
    model_registry.register_model(model)
    
    control_system = ModelBasedControlSystem(model_registry)
    compiler = YAMLCompiler()
    
    # Load YAML
    yaml_path = Path("examples/research_assistant.yaml")
    with open(yaml_path, 'r') as f:
        yaml_content = f.read()
    
    # Test with a concrete topic
    inputs = {
        "topic": "The impact of AI on software development productivity",
        "depth": "comprehensive",
        "output_format": "markdown"
    }
    
    print(f"Input Topic: {inputs['topic']}")
    print(f"Depth: {inputs['depth']}")
    print("-" * 40)
    
    try:
        pipeline = await compiler.compile(yaml_content, inputs)
        results = await control_system.execute_pipeline(pipeline)
        
        # Get outputs
        outputs = pipeline.get_outputs(results)
        
        print("RESEARCH PLAN:")
        print(results.get("create_research_plan", "No plan generated"))
        print("\n" + "=" * 40 + "\n")
        
        print("INFORMATION GATHERED:")
        print(results.get("gather_information", "No information gathered"))
        print("\n" + "=" * 40 + "\n")
        
        print("ANALYSIS:")
        print(results.get("analyze_findings", "No analysis performed"))
        print("\n" + "=" * 40 + "\n")
        
        print("FINAL RESEARCH REPORT:")
        print(outputs.get("research_report", "No report generated"))
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False


async def test_content_creation():
    """Test content creation pipeline."""
    print("\n" + "=" * 60)
    print("TESTING: Content Creation Pipeline")
    print("=" * 60)
    
    # Setup
    model_registry = ModelRegistry()
    model = OpenAIModel(model_name="gpt-4o-mini")
    model_registry.register_model(model)
    
    control_system = ModelBasedControlSystem(model_registry)
    compiler = YAMLCompiler()
    
    # Load YAML
    yaml_path = Path("examples/content_creation_pipeline.yaml")
    with open(yaml_path, 'r') as f:
        yaml_content = f.read()
    
    # Test with content creation task
    inputs = {
        "topic": "Best practices for code reviews in agile teams",
        "content_type": "blog_post",
        "target_audience": "software developers",
        "tone": "professional but approachable",
        "length": "medium"
    }
    
    print(f"Topic: {inputs['topic']}")
    print(f"Content Type: {inputs['content_type']}")
    print(f"Target Audience: {inputs['target_audience']}")
    print(f"Tone: {inputs['tone']}")
    print("-" * 40)
    
    try:
        pipeline = await compiler.compile(yaml_content, inputs)
        results = await control_system.execute_pipeline(pipeline)
        
        # Get outputs
        outputs = pipeline.get_outputs(results)
        
        print("CONTENT STRATEGY:")
        print(results.get("develop_content_strategy", "No strategy developed"))
        print("\n" + "=" * 40 + "\n")
        
        print("OUTLINE:")
        print(results.get("create_outline", "No outline created"))
        print("\n" + "=" * 40 + "\n")
        
        print("FINAL CONTENT:")
        print(outputs.get("final_content", "No content generated"))
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False


async def test_data_processing():
    """Test data processing workflow."""
    print("\n" + "=" * 60)
    print("TESTING: Data Processing Workflow")
    print("=" * 60)
    
    # Setup
    model_registry = ModelRegistry()
    model = OpenAIModel(model_name="gpt-4o-mini")
    model_registry.register_model(model)
    
    control_system = ModelBasedControlSystem(model_registry)
    compiler = YAMLCompiler()
    
    # Load YAML
    yaml_path = Path("examples/data_processing_workflow.yaml")
    with open(yaml_path, 'r') as f:
        yaml_content = f.read()
    
    # Test with data processing task
    inputs = {
        "input_data": {
            "source": "customer_feedback_2024.csv",
            "format": "csv",
            "columns": ["date", "customer_id", "feedback_text", "rating", "category"]
        },
        "processing_config": {
            "remove_duplicates": True,
            "normalize_text": True,
            "validation_threshold": 0.95,
            "sentiment_analysis": True
        },
        "output_format": "json"
    }
    
    print(f"Input Data Source: {inputs['input_data']['source']}")
    print(f"Processing Config: {inputs['processing_config']}")
    print(f"Output Format: {inputs['output_format']}")
    print("-" * 40)
    
    try:
        pipeline = await compiler.compile(yaml_content, inputs)
        results = await control_system.execute_pipeline(pipeline)
        
        # Get outputs
        outputs = pipeline.get_outputs(results)
        
        print("DATA LOADING:")
        print(results.get("load_data", "No data loaded"))
        print("\n" + "=" * 40 + "\n")
        
        print("DATA VALIDATION:")
        print(results.get("validate_data", "No validation performed"))
        print("\n" + "=" * 40 + "\n")
        
        print("DATA CLEANING:")
        print(results.get("clean_data", "No cleaning performed"))
        print("\n" + "=" * 40 + "\n")
        
        print("FINAL PROCESSED DATA:")
        print(outputs.get("processed_data", "No processed data"))
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False


async def test_creative_writing():
    """Test creative writing assistant."""
    print("\n" + "=" * 60)
    print("TESTING: Creative Writing Assistant")
    print("=" * 60)
    
    # Setup
    model_registry = ModelRegistry()
    model = OpenAIModel(model_name="gpt-4o-mini")
    model_registry.register_model(model)
    
    control_system = ModelBasedControlSystem(model_registry)
    compiler = YAMLCompiler()
    
    # Load YAML
    yaml_path = Path("examples/creative_writing_assistant.yaml")
    with open(yaml_path, 'r') as f:
        yaml_content = f.read()
    
    # Test with creative writing task
    inputs = {
        "genre": "science fiction",
        "theme": "artificial intelligence gaining consciousness",
        "length": "short story",
        "style": "descriptive and thought-provoking"
    }
    
    print(f"Genre: {inputs['genre']}")
    print(f"Theme: {inputs['theme']}")
    print(f"Length: {inputs['length']}")
    print(f"Style: {inputs['style']}")
    print("-" * 40)
    
    try:
        pipeline = await compiler.compile(yaml_content, inputs)
        results = await control_system.execute_pipeline(pipeline)
        
        # Get outputs
        outputs = pipeline.get_outputs(results)
        
        print("CONCEPT DEVELOPMENT:")
        print(results.get("develop_concept", "No concept developed"))
        print("\n" + "=" * 40 + "\n")
        
        print("CHARACTER CREATION:")
        print(results.get("create_characters", "No characters created"))
        print("\n" + "=" * 40 + "\n")
        
        print("PLOT OUTLINE:")
        print(results.get("create_plot_outline", "No plot outline"))
        print("\n" + "=" * 40 + "\n")
        
        print("FINAL STORY:")
        print(outputs.get("final_story", "No story generated"))
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False


async def main():
    """Run all tests and assess quality."""
    print("RUNNING REAL EXAMPLES WITH ACTUAL MODELS")
    print("Testing declarative framework output quality...")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    results = []
    
    # Test each example
    tests = [
        ("Research Assistant", test_research_assistant),
        ("Content Creation", test_content_creation),
        ("Data Processing", test_data_processing),
        ("Creative Writing", test_creative_writing)
    ]
    
    for name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"STARTING: {name}")
        print(f"{'='*80}")
        
        try:
            success = await test_func()
            results.append((name, success))
            
            if success:
                print(f"\n✅ {name} - COMPLETED SUCCESSFULLY")
            else:
                print(f"\n❌ {name} - FAILED")
                
        except Exception as e:
            print(f"\n💥 {name} - CRASHED: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{'='*80}")
    print("QUALITY ASSESSMENT SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"Tests completed: {successful}/{total}")
    print("\nResults:")
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
    
    if successful == total:
        print(f"\n🎉 All examples executed successfully!")
        print("The declarative framework is working well with real models.")
    else:
        print(f"\n⚠️  {total - successful} examples failed.")
        print("Some issues may need attention.")


if __name__ == "__main__":
    asyncio.run(main())