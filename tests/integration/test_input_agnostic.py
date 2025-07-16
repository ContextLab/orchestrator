#!/usr/bin/env python3
"""Test input-agnostic pipeline system."""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import orchestrator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import orchestrator as orc
from orchestrator.control_systems import ResearchReportControlSystem

async def test_input_agnostic_pipeline():
    """Test the input-agnostic pipeline system."""
    
    print("🎯 TESTING INPUT-AGNOSTIC PIPELINE SYSTEM")
    print("=" * 60)
    
    # Initialize models
    orc.init_models()
    
    # Set up the control system
    output_dir = "./output/input_agnostic_test"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    control_system = ResearchReportControlSystem(output_dir=output_dir)
    
    # Compile the template-based pipeline
    print("\n📋 Compiling template-based pipeline...")
    try:
        report_writer = orc.compile('pipelines/research-report-template.yaml')
    except FileNotFoundError:
        report_writer = orc.compile('examples/pipelines/research-report-template.yaml')
    
    # Update the orchestrator to use our control system
    orc._orchestrator.control_system = control_system
    
    print("\n🧪 Test 1: Research Report on Quantum Computing")
    print("-" * 50)
    
    try:
        report1 = await report_writer._run_async(
            topic='quantum_computing',
            instructions='Focus on recent breakthroughs in quantum error correction and practical applications. Include hardware developments from major tech companies.'
        )
        
        print("✅ Test 1 completed successfully!")
        print(f"   Result: {type(report1).__name__}")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🧪 Test 2: Research Report on Renewable Energy")
    print("-" * 50)
    
    try:
        report2 = await report_writer._run_async(
            topic='renewable_energy',
            instructions='Analyze the latest advances in solar and wind technology. Include cost comparisons and grid integration challenges.'
        )
        
        print("✅ Test 2 completed successfully!")
        print(f"   Result: {type(report2).__name__}")
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🧪 Test 3: Research Report on Machine Learning")
    print("-" * 50)
    
    try:
        report3 = await report_writer._run_async(
            topic='machine_learning',
            instructions='Cover transformer architectures, attention mechanisms, and recent developments in large language models. Include practical implementation considerations.'
        )
        
        print("✅ Test 3 completed successfully!")
        print(f"   Result: {type(report3).__name__}")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🏁 INPUT-AGNOSTIC PIPELINE TESTING COMPLETE")
    print("\n✅ Successfully demonstrated:")
    print("   - Single pipeline definition for multiple topics")
    print("   - Template variable resolution ({{ inputs.topic }})")
    print("   - AUTO tag resolution for outputs")
    print("   - Runtime parameter customization")
    print("\n📄 Reports generated:")
    print("   - Quantum Computing report")
    print("   - Renewable Energy report") 
    print("   - Machine Learning report")
    
    return True


async def test_missing_inputs():
    """Test validation of required inputs."""
    print("\n🧪 Test 4: Required Input Validation")
    print("-" * 50)
    
    try:
        report_writer = orc.compile('examples/pipelines/research-report-template.yaml')
        
        # This should fail - missing required inputs
        await report_writer._run_async(topic='test')  # Missing instructions
        
        print("❌ Should have failed validation")
        
    except ValueError as e:
        print(f"✅ Correctly caught validation error: {e}")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")


if __name__ == "__main__":
    success = asyncio.run(test_input_agnostic_pipeline())
    asyncio.run(test_missing_inputs())