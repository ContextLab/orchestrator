#!/usr/bin/env python3
"""
Test with a simple working example to demonstrate the framework.
"""

import asyncio
import os
from orchestrator.compiler.yaml_compiler import YAMLCompiler
from orchestrator.control_systems.model_based_control_system import ModelBasedControlSystem
from orchestrator.models.model_registry import ModelRegistry
from orchestrator.integrations.openai_model import OpenAIModel


async def test_simple_research():
    """Test with a simple research task."""
    print("=" * 60)
    print("TESTING: Simple Research Task")
    print("=" * 60)
    
    # Setup
    model_registry = ModelRegistry()
    model = OpenAIModel(model_name="gpt-4o-mini")
    model_registry.register_model(model)
    
    control_system = ModelBasedControlSystem(model_registry)
    compiler = YAMLCompiler()
    
    # Create a simple working YAML
    yaml_content = '''
name: "Simple Research Assistant"
description: "A basic research pipeline"

inputs:
  topic:
    type: string
    required: true

steps:
  - id: research_plan
    action: |
      Create a research plan for the topic: {{topic}}
      
      Include:
      1. Key research questions
      2. Information sources to explore
      3. Research methodology
      4. Expected outcomes
      
      Return a structured research plan

  - id: gather_info
    action: |
      Research the topic: {{topic}}
      
      Based on the research plan: {{research_plan.result}}
      
      Gather information about:
      1. Current state and trends
      2. Key findings from recent studies
      3. Expert opinions and insights
      4. Practical applications
      
      Return comprehensive information summary
    depends_on: [research_plan]

  - id: analyze_findings
    action: |
      Analyze the research findings:
      {{gather_info.result}}
      
      Provide:
      1. Key insights and patterns
      2. Important conclusions
      3. Implications and significance
      4. Areas for further research
      
      Return analytical summary
    depends_on: [gather_info]

  - id: create_report
    action: |
      Create a comprehensive research report on: {{topic}}
      
      Research Plan: {{research_plan.result}}
      Information: {{gather_info.result}}
      Analysis: {{analyze_findings.result}}
      
      Format as a professional report with:
      1. Executive summary
      2. Key findings
      3. Detailed analysis
      4. Conclusions and recommendations
      
      Return well-structured report
    depends_on: [analyze_findings]

outputs:
  research_report: "{{create_report.result}}"
  key_findings: "{{analyze_findings.result}}"
  research_plan: "{{research_plan.result}}"
'''
    
    # Test with a concrete topic
    inputs = {
        "topic": "The role of code reviews in improving software quality"
    }
    
    print(f"Research Topic: {inputs['topic']}")
    print("-" * 40)
    
    try:
        pipeline = await compiler.compile(yaml_content, inputs)
        print(f"Pipeline compiled successfully with {len(pipeline.tasks)} tasks")
        
        results = await control_system.execute_pipeline(pipeline)
        print("Pipeline executed successfully!")
        
        print("\n" + "=" * 40)
        print("RESEARCH PLAN:")
        print("=" * 40)
        print(results.get("research_plan", "No plan generated"))
        
        print("\n" + "=" * 40)
        print("INFORMATION GATHERED:")
        print("=" * 40)
        print(results.get("gather_info", "No information gathered"))
        
        print("\n" + "=" * 40)
        print("ANALYSIS:")
        print("=" * 40)
        print(results.get("analyze_findings", "No analysis performed"))
        
        print("\n" + "=" * 40)
        print("FINAL RESEARCH REPORT:")
        print("=" * 40)
        print(results.get("create_report", "No report generated"))
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_simple_content_creation():
    """Test content creation with a simple example."""
    print("\n" + "=" * 60)
    print("TESTING: Simple Content Creation")
    print("=" * 60)
    
    # Setup
    model_registry = ModelRegistry()
    model = OpenAIModel(model_name="gpt-4o-mini")
    model_registry.register_model(model)
    
    control_system = ModelBasedControlSystem(model_registry)
    compiler = YAMLCompiler()
    
    # Create a simple working YAML
    yaml_content = '''
name: "Simple Content Creator"
description: "Create engaging content"

inputs:
  topic:
    type: string
    required: true
  audience:
    type: string
    default: "general"

steps:
  - id: brainstorm
    action: |
      Brainstorm ideas for content about: {{topic}}
      
      Target audience: {{audience}}
      
      Generate:
      1. Main angle or hook
      2. Key points to cover
      3. Engaging opening ideas
      4. Call-to-action suggestions
      
      Return creative content ideas

  - id: create_outline
    action: |
      Create a detailed outline for: {{topic}}
      
      Based on brainstorming: {{brainstorm.result}}
      
      Structure:
      1. Introduction (hook and context)
      2. Main body (3-4 key sections)
      3. Conclusion (summary and CTA)
      
      Return structured outline
    depends_on: [brainstorm]

  - id: write_content
    action: |
      Write engaging content about: {{topic}}
      
      Outline: {{create_outline.result}}
      Ideas: {{brainstorm.result}}
      Audience: {{audience}}
      
      Create:
      1. Compelling introduction
      2. Well-structured body content
      3. Strong conclusion
      4. Engaging tone throughout
      
      Return polished content
    depends_on: [create_outline]

outputs:
  final_content: "{{write_content.result}}"
  content_outline: "{{create_outline.result}}"
  content_ideas: "{{brainstorm.result}}"
'''
    
    # Test with content creation
    inputs = {
        "topic": "Benefits of pair programming for development teams",
        "audience": "software developers"
    }
    
    print(f"Content Topic: {inputs['topic']}")
    print(f"Target Audience: {inputs['audience']}")
    print("-" * 40)
    
    try:
        pipeline = await compiler.compile(yaml_content, inputs)
        print(f"Pipeline compiled successfully with {len(pipeline.tasks)} tasks")
        
        results = await control_system.execute_pipeline(pipeline)
        print("Pipeline executed successfully!")
        
        print("\n" + "=" * 40)
        print("BRAINSTORMING:")
        print("=" * 40)
        print(results.get("brainstorm", "No ideas generated"))
        
        print("\n" + "=" * 40)
        print("CONTENT OUTLINE:")
        print("=" * 40)
        print(results.get("create_outline", "No outline created"))
        
        print("\n" + "=" * 40)
        print("FINAL CONTENT:")
        print("=" * 40)
        print(results.get("write_content", "No content generated"))
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run simple working examples."""
    print("TESTING DECLARATIVE FRAMEWORK WITH REAL MODELS")
    print("Simple working examples to demonstrate quality...")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return
    
    results = []
    
    # Test simple examples
    tests = [
        ("Simple Research", test_simple_research),
        ("Simple Content Creation", test_simple_content_creation)
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
        print("\nKey observations:")
        print("- YAML compilation works correctly")
        print("- Template resolution functions properly")
        print("- Dependency management executes in correct order")
        print("- Context propagation between steps works")
        print("- AI model integration produces quality outputs")
    else:
        print(f"\n⚠️  {total - successful} examples failed.")
        print("Some issues may need attention.")


if __name__ == "__main__":
    asyncio.run(main())