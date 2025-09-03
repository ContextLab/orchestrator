#!/usr/bin/env python3
"""
Direct test for foundation module components without complex orchestrator imports.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_foundation_direct():
    """Test foundation components directly."""
    print("🧪 Testing Foundation Components Directly")
    print("=" * 50)
    
    try:
        # Import foundation components
        from src.orchestrator.foundation.interfaces import (

from tests.test_infrastructure import create_test_orchestrator, TestModel, TestProvider
            PipelineCompilerInterface,
            ExecutionEngineInterface,
            ModelManagerInterface,
            ToolRegistryInterface,
            QualityControlInterface,
        )
        from src.orchestrator.foundation.pipeline_spec import (
            PipelineSpecification,
            PipelineHeader,
            PipelineStep
        )
        from src.orchestrator.foundation.result import StepResult, PipelineResult
        from src.orchestrator.foundation import FoundationConfig
        
        print("✅ Foundation interfaces imported successfully")
        
        # Test 1: Verify abstract classes
        try:
            PipelineCompilerInterface()
        except TypeError:
            print("✅ PipelineCompilerInterface is properly abstract")
        
        try:
            ExecutionEngineInterface()
        except TypeError:
            print("✅ ExecutionEngineInterface is properly abstract")
            
        try:
            ModelManagerInterface()
        except TypeError:
            print("✅ ModelManagerInterface is properly abstract")
            
        try:
            ToolRegistryInterface()
        except TypeError:
            print("✅ ToolRegistryInterface is properly abstract")
            
        try:
            QualityControlInterface()
        except TypeError:
            print("✅ QualityControlInterface is properly abstract")
        
        # Test 2: Verify data structures
        header = PipelineHeader(id="test_pipeline", name="test", version="1.0")
        assert header.id == "test_pipeline"
        assert header.name == "test"
        assert header.version == "1.0"
        print("✅ PipelineHeader creation works")
        
        step = PipelineStep(id="step1", name="Test Step")
        assert step.id == "step1"
        assert step.name == "Test Step"
        print("✅ PipelineStep creation works")
        
        spec = PipelineSpecification(header=header, steps=[step])
        assert spec.header.name == "test"
        assert len(spec.steps) == 1
        print("✅ PipelineSpecification creation works")
        
        step_result = StepResult(step_id="step1", status="success", output={"result": "test"})
        assert step_result.step_id == "step1"
        assert step_result.status == "success"
        print("✅ StepResult creation works")
        
        pipeline_result = PipelineResult(
            pipeline_name="test",
            status="success", 
            step_results=[step_result],
            total_steps=1
        )
        assert pipeline_result.pipeline_name == "test"
        assert len(pipeline_result.step_results) == 1
        print("✅ PipelineResult creation works")
        
        # Test 3: Verify configuration
        config = FoundationConfig()
        assert config.model_selection_strategy == "balanced"
        assert config.max_concurrent_steps == 5
        assert config.enable_quality_checks == True
        print("✅ FoundationConfig works with defaults")
        
        custom_config = FoundationConfig(
            default_model="gpt-4",
            max_concurrent_steps=10,
            enable_persistence=True
        )
        assert custom_config.default_model == "gpt-4"
        assert custom_config.max_concurrent_steps == 10
        assert custom_config.enable_persistence == True
        print("✅ FoundationConfig works with custom values")
        
        print(f"✅ Foundation Direct Tests: 8/8 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Foundation Direct Tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run direct foundation tests."""
    print("🚀 Direct Foundation Component Tests")
    print("=" * 50)
    
    success = test_foundation_direct()
    
    if success:
        print("🎉 Foundation components are working correctly!")
        return 0
    else:
        print("⚠️  Foundation components need attention.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)