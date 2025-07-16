#!/usr/bin/env python3
"""Run the pipeline from README.md with proper implementation."""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import orchestrator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import orchestrator as orc
from research_control_system import ResearchReportControlSystem

async def main():
    """Run the research report pipeline as shown in README."""
    
    print("🎯 RUNNING PIPELINE FROM README.md")
    print("=" * 60)
    
    # Initialize models as shown in README
    orc.init_models()
    
    # Set up the control system
    output_dir = "./output/readme_report"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    control_system = ResearchReportControlSystem(output_dir=output_dir)
    
    # The orchestrator will be created during compile() - we'll update it then
    
    # Compile the pipeline
    print("\n📋 Compiling pipeline...")
    try:
        report_writer = orc.compile('pipelines/research-report-simplified.yaml')
    except FileNotFoundError:
        # Try from examples directory
        report_writer = orc.compile('examples/pipelines/research-report-simplified.yaml')
    
    # Now update the orchestrator to use our control system
    orc._orchestrator.control_system = control_system
    
    print("\n🚀 Running pipeline...")
    print("   Topic: agents") 
    print("   Instructions: Teach me everything about how AI agents work...")
    
    # Generate the report by running the pipeline
    try:
        report = await report_writer._run_async(
            topic='agents',
            instructions='Teach me everything about how AI agents work, how to create them, and how to use them. Be sure to include example use cases and cite specific studies and resources-- especially Python toolboxes and open source tools.'
        )
        
        print("\n✅ Pipeline completed successfully!")
        
        # Display results
        if isinstance(report, dict):
            if 'final_report' in report and 'file' in report['final_report']:
                print(f"\n📄 Report saved to: {report['final_report']['file']}")
                
                # Read and display excerpt
                report_path = Path(report['final_report']['file'])
                if report_path.exists():
                    with open(report_path, 'r') as f:
                        content = f.read()
                    
                    print("\n📊 Report Statistics:")
                    print(f"   Word count: {len(content.split())}")
                    print(f"   Characters: {len(content)}")
                    print(f"   Lines: {len(content.splitlines())}")
                    
                    print("\n📜 Report Excerpt (first 1000 chars):")
                    print("-" * 60)
                    print(content[:1000] + "...")
                    print("-" * 60)
                    
                    # Quality assessment
                    quality_indicators = {
                        "Has introduction": "Introduction" in content,
                        "Has code examples": "```python" in content or "```" in content,
                        "Mentions frameworks": all(f in content for f in ["LangChain", "AutoGen", "CrewAI"]),
                        "Has references": "References" in content,
                        "Technical depth": "Technical Details" in content
                    }
                    
                    print("\n🎯 Quality Assessment:")
                    score = sum(quality_indicators.values()) / len(quality_indicators)
                    for indicator, present in quality_indicators.items():
                        print(f"   {'✅' if present else '❌'} {indicator}")
                    
                    print(f"\n   Overall Quality Score: {score:.0%}")
                    
                    if score >= 0.8:
                        print("   Grade: A - Excellent comprehensive report")
                    elif score >= 0.6:
                        print("   Grade: B - Good report with room for improvement")
                    else:
                        print("   Grade: C - Basic report, needs enhancement")
            else:
                print("\n📊 Pipeline Results:")
                for key, value in report.items():
                    if isinstance(value, dict) and 'file' in value:
                        print(f"   {key}: saved to {value['file']}")
                    else:
                        print(f"   {key}: {type(value).__name__}")
        else:
            print(f"\n📊 Result: {report}")
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("🏁 DEMONSTRATION COMPLETE")
    print("\n💡 Note: This simplified version demonstrates the core concepts from the README:")
    print("   - YAML pipeline definition with <AUTO> tags")
    print("   - Model-based ambiguity resolution")
    print("   - Multi-step research workflow")
    print("   - Quality validation")
    print("\n📝 The full README pipeline would additionally include:")
    print("   - Real web search with headless browser")
    print("   - Parallel source verification")
    print("   - PDF generation with pandoc")
    print("   - Larger models (20B-40B) for better quality")
    
    return True


if __name__ == "__main__":
    # Run with asyncio
    success = asyncio.run(main())